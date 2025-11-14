// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_dim(const tt::tt_metal::Tensor& t, uint32_t dim);
tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t);
tt::tt_metal::Tensor log_softmax(const tt::tt_metal::Tensor& t, int dim);

// ⚠️ WARNING: use_fp32_accumulation_workaround=true causes PERFORMANCE DEGRADATION
// Defaults to true as TEMPORARY WORKAROUND for TTNN bfloat16 softmax precision bug
// TODO: Change default to false once TTNN fixes bfloat16 softmax kernel
tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim, bool use_fp32_accumulation_workaround = true);

tt::tt_metal::Tensor divide(const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b);

tt::tt_metal::Tensor mean_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);
tt::tt_metal::Tensor mean_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);

tt::tt_metal::Tensor sum_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);
tt::tt_metal::Tensor sum_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);

tt::tt_metal::Tensor sample(
    const tt::tt_metal::Tensor& t,
    float temperature,
    uint32_t seed,
    std::optional<tt::tt_metal::Tensor> logits_padding_mask = std::nullopt);

tt::tt_metal::Tensor to_l1_interleaved(const tt::tt_metal::Tensor& t);
tt::tt_metal::Tensor to_dram_interleaved(const tt::tt_metal::Tensor& t);

}  // namespace ttml::ttnn_fixed
