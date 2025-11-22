// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads.hpp"

#include <utility>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

namespace ttnn::operations::experimental::transformer {

namespace {

// Fallback/workaround implementation for head_dim < TILE_WIDTH using high-level ops.
// This is needed because the kernel-level implementation assumes tile-aligned head dimensions
// and uses integer division (head_dim / TILE_WIDTH) which produces 0 when head_dim < 32.
//
// WARNING: This workaround may have performance degradation compared to a proper kernel-level
// implementation since it uses multiple high-level operations (untilize, slice, squeeze,
// reshape, permute, tilize) instead of optimized kernels. A real fix would require modifying
// the reader/writer kernels to handle sub-tile head dimensions with proper tile packing/unpacking.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads_small_head_dim(
    const Tensor& input_tensor,
    const std::optional<Tensor>& input_tensor_kv,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config) {
    using namespace tt::constants;

    const auto& input_shape = input_tensor.logical_shape();
    const uint32_t batch = input_shape[0];
    const uint32_t seq_len = input_shape[2];

    auto mem_config = memory_config.value_or(input_tensor.memory_config());

    // Step 1: Untilize to ROW_MAJOR for easier manipulation
    auto input_rm = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt);

    ttnn::Tensor q_rm, k_rm, v_rm;

    if (input_tensor_kv.has_value()) {
        // Separate Q and KV tensors
        // Q is input_tensor, KV is input_tensor_kv
        auto kv_rm = ttnn::to_layout(input_tensor_kv.value(), Layout::ROW_MAJOR, std::nullopt, std::nullopt);

        // Q: [B, 1, S, num_q * head_dim]
        q_rm = input_rm;

        // KV: [B, 1, S, 2 * num_kv * head_dim]
        // Slice K and V
        uint32_t k_start = 0;
        uint32_t k_end = num_kv_heads * head_dim;
        uint32_t v_start = k_end;
        uint32_t v_end = 2 * num_kv_heads * head_dim;

        k_rm = ttnn::slice(
            kv_rm,
            std::array<uint32_t, 4>{0, 0, 0, k_start},
            std::array<uint32_t, 4>{batch, 1, seq_len, k_end},
            std::array<uint32_t, 4>{1, 1, 1, 1});
        v_rm = ttnn::slice(
            kv_rm,
            std::array<uint32_t, 4>{0, 0, 0, v_start},
            std::array<uint32_t, 4>{batch, 1, seq_len, v_end},
            std::array<uint32_t, 4>{1, 1, 1, 1});
    } else {
        // Fused QKV tensor
        // Input: [B, 1, S, (num_q + 2*num_kv) * head_dim]
        uint32_t q_start = 0;
        uint32_t q_end = num_q_heads * head_dim;
        uint32_t k_start = q_end;
        uint32_t k_end = k_start + num_kv_heads * head_dim;
        uint32_t v_start = k_end;
        uint32_t v_end = v_start + num_kv_heads * head_dim;

        q_rm = ttnn::slice(
            input_rm,
            std::array<uint32_t, 4>{0, 0, 0, q_start},
            std::array<uint32_t, 4>{batch, 1, seq_len, q_end},
            std::array<uint32_t, 4>{1, 1, 1, 1});
        k_rm = ttnn::slice(
            input_rm,
            std::array<uint32_t, 4>{0, 0, 0, k_start},
            std::array<uint32_t, 4>{batch, 1, seq_len, k_end},
            std::array<uint32_t, 4>{1, 1, 1, 1});
        v_rm = ttnn::slice(
            input_rm,
            std::array<uint32_t, 4>{0, 0, 0, v_start},
            std::array<uint32_t, 4>{batch, 1, seq_len, v_end},
            std::array<uint32_t, 4>{1, 1, 1, 1});
    }

    // Step 2: Squeeze dim 1, reshape to [B, S, num_heads, head_dim], then permute to [B, num_heads, S, head_dim]

    // Q: [B, 1, S, num_q * head_dim] -> [B, S, num_q * head_dim] -> [B, S, num_q, head_dim] -> [B, num_q, S, head_dim]
    auto q_squeezed = ttnn::squeeze(q_rm, 1);
    auto q_reshaped = ttnn::reshape(q_squeezed, ttnn::Shape({batch, seq_len, num_q_heads, head_dim}));
    auto q_permuted = ttnn::permute(q_reshaped, SmallVector<int64_t>{0, 2, 1, 3});

    // K: [B, 1, S, num_kv * head_dim] -> [B, S, num_kv * head_dim] -> [B, S, num_kv, head_dim] -> [B, num_kv, S,
    // head_dim] or if transpose_k_heads: -> [B, num_kv, head_dim, S]
    auto k_squeezed = ttnn::squeeze(k_rm, 1);
    auto k_reshaped = ttnn::reshape(k_squeezed, ttnn::Shape({batch, seq_len, num_kv_heads, head_dim}));
    ttnn::Tensor k_permuted;
    if (transpose_k_heads) {
        k_permuted = ttnn::permute(k_reshaped, SmallVector<int64_t>{0, 2, 3, 1});
    } else {
        k_permuted = ttnn::permute(k_reshaped, SmallVector<int64_t>{0, 2, 1, 3});
    }

    // V: [B, 1, S, num_kv * head_dim] -> [B, S, num_kv * head_dim] -> [B, S, num_kv, head_dim] -> [B, num_kv, S,
    // head_dim]
    auto v_squeezed = ttnn::squeeze(v_rm, 1);
    auto v_reshaped = ttnn::reshape(v_squeezed, ttnn::Shape({batch, seq_len, num_kv_heads, head_dim}));
    auto v_permuted = ttnn::permute(v_reshaped, SmallVector<int64_t>{0, 2, 1, 3});

    // Step 3: Tilize back to TILE layout with requested memory config
    auto q_out = ttnn::to_layout(q_permuted, Layout::TILE, std::nullopt, mem_config);
    auto k_out = ttnn::to_layout(k_permuted, Layout::TILE, std::nullopt, mem_config);
    auto v_out = ttnn::to_layout(v_permuted, Layout::TILE, std::nullopt, mem_config);

    return {q_out, k_out, v_out};
}

}  // anonymous namespace

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> NlpCreateHeadsOperation::invoke(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_q_heads,
    std::optional<uint32_t> num_kv_heads,
    bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors) {
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_q_heads);
    uint32_t head_dim;
    if (input_tensor_kv.has_value()) {
        TT_FATAL(input_tensor_q.padded_shape()[3] % num_q_heads == 0, "Unsupported input shape");
        TT_FATAL(input_tensor_kv.value().padded_shape()[3] % (2 * num_kv_heads_val) == 0, "Unsupported input shape");
        head_dim = input_tensor_q.padded_shape()[3] / num_q_heads;
        TT_FATAL(
            input_tensor_kv.value().padded_shape()[3] / (2 * num_kv_heads_val) == head_dim,
            "Head dims must be the same for Q and K, V");
    } else {
        TT_FATAL(
            input_tensor_q.padded_shape()[3] % (num_q_heads + 2 * num_kv_heads_val) == 0, "Unsupported input shape");
        head_dim = input_tensor_q.padded_shape()[3] / (num_q_heads + 2 * num_kv_heads_val);
    }

    // Use fallback implementation for head_dim < TILE_WIDTH (32)
    // The kernel-level implementation assumes tile-aligned head dimensions
    if (head_dim < tt::constants::TILE_WIDTH) {
        return nlp_create_qkv_heads_small_head_dim(
            input_tensor_q, input_tensor_kv, num_q_heads, num_kv_heads_val, head_dim, transpose_k_heads, memory_config);
    }

    return ttnn::prim::nlp_create_qkv_heads(
        input_tensor_q,
        input_tensor_kv,
        num_q_heads,
        num_kv_heads,
        head_dim,
        transpose_k_heads,
        memory_config,
        optional_output_tensors);
}

}  // namespace ttnn::operations::experimental::transformer
