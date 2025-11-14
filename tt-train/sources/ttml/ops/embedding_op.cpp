// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr embedding_op(const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight) {
    // prepare for embedding
    auto weight_tensor = weight->get_value();
    weight_tensor = ttnn::untilize(weight_tensor);

    auto input_tensor = tensor->get_value();
    auto input_shape = input_tensor.logical_shape();
    auto batch_size = input_shape[0];

    // WORKAROUND: ttnn::embedding has a batch processing bug for batch_size > 1
    // where it returns incorrect embeddings for batches after the first one.
    // Process each batch separately and concatenate the results.
    // TODO: Remove this workaround once ttnn::embedding is fixed

    ttnn::Tensor embeddings;

    if (batch_size > 1) {
        // Process each batch separately
        std::vector<ttnn::Tensor> batch_embeddings;
        batch_embeddings.reserve(batch_size);

        for (uint32_t i = 0; i < batch_size; ++i) {
            // Slice single batch: [batch, 1, 1, seq] -> [1, 1, 1, seq]
            ttnn::SmallVector<uint32_t> start_indices = {i, 0, 0, 0};
            ttnn::SmallVector<uint32_t> end_indices = {i + 1, input_shape[1], input_shape[2], input_shape[3]};
            ttnn::SmallVector<uint32_t> stride = {1, 1, 1, 1};

            auto batch_input = ttnn::slice(input_tensor, start_indices, end_indices, stride);

            // Process this batch
            auto batch_emb =
                ttnn::embedding(batch_input, weight_tensor, /* pad_token */ std::nullopt, ttnn::Layout::TILE);
            batch_embeddings.push_back(batch_emb);
        }

        // Concatenate along batch dimension
        embeddings = ttnn::concat(batch_embeddings, /* dim */ 0);
    } else {
        // Single batch - use direct path
        embeddings = ttnn::embedding(input_tensor, weight_tensor, /* pad_token */ std::nullopt, ttnn::Layout::TILE);
    }

    auto embeddings_shape = embeddings.logical_shape();
    batch_size = embeddings_shape[0];
    auto sentence_size = embeddings_shape[1];
    auto embedding_dim = embeddings_shape[2];
    embeddings = ttnn::reshape(embeddings, ttnn::Shape({batch_size, 1, sentence_size, embedding_dim}));
    auto out = autograd::create_tensor(embeddings);

    autograd::GradFunction grad = [tensor, weight, out]() {
        auto out_grad = out->get_grad();
        auto tensor_shape = tensor->get_value().logical_shape();
        out_grad = ttnn::reshape(
            out_grad, ttnn::Shape({1, 1, tensor_shape[0] * tensor_shape[-1], out_grad.logical_shape()[-1]}));

        // embedding_bw requires index tensor in ROW_MAJOR layout
        auto tensor_value = tensor->get_value();
        if (tensor_value.layout() != ttnn::Layout::ROW_MAJOR) {
            tensor_value = ttnn::to_layout(tensor_value, ttnn::Layout::ROW_MAJOR);
        }

        auto weight_grad = ttnn::embedding_bw(tensor_value, weight->get_value(), out_grad);
        weight->add_grad(weight_grad);
    };

    auto links = autograd::get_links(weight);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
