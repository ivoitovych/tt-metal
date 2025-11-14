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
    // Default is false (native bfloat16) to preserve TTNN framework behavior.
    // Only workaround code should explicitly pass true when needed.
    // Using FP32 instead of bfloat16 is slower and should NOT be considered a permanent solution.
    static ttnn::WormholeComputeKernelConfig softmax(bool use_fp32_accumulation_workaround = false);

    static ttnn::WormholeComputeKernelConfig matmul();
    static ttnn::WormholeComputeKernelConfig fast();
};

}  // namespace ttml::core
