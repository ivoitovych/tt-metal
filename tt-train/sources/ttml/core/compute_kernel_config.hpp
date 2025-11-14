// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

namespace ttml::core {

class ComputeKernelConfig {
public:
    static ttnn::WormholeComputeKernelConfig precise();

    // ⚠️ WARNING: use_fp32_accumulation_workaround=true causes PERFORMANCE DEGRADATION
    // This parameter defaults to true as a TEMPORARY WORKAROUND for a TTNN bfloat16 softmax bug.
    // Using FP32 instead of bfloat16 is slower and should NOT be considered a permanent solution.
    // TODO: Set default to false once TTNN bfloat16 softmax kernel is fixed
    static ttnn::WormholeComputeKernelConfig softmax(bool use_fp32_accumulation_workaround = true);

    static ttnn::WormholeComputeKernelConfig matmul();
    static ttnn::WormholeComputeKernelConfig fast();
};

}  // namespace ttml::core
