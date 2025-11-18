// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_config.hpp"

namespace ttml::core {

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::precise() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.packer_l1_acc = true;
    return config;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::softmax(bool use_fp32_accumulation_workaround) {
    ttnn::WormholeComputeKernelConfig config;
    // ⚠️ WARNING: TEMPORARY WORKAROUND WITH PERFORMANCE DEGRADATION ⚠️
    //
    // This is NOT a fix! Using FP32 instead of bfloat16 causes performance penalty.
    //
    // ISSUE: Softmax with bfloat16 accumulation loses precision on BERT attention
    //        score patterns (PCC drops to 0.81 instead of expected >0.999)
    //
    // WORKAROUND: Force FP32 accumulation (fp32_dest_acc_en=true) to restore precision
    //             but this degrades performance compared to native bfloat16 accumulation
    //
    // PROPER FIX NEEDED: TTNN/hardware team must fix bfloat16 softmax kernel to handle
    //                    attention score distributions correctly without precision loss
    //
    // TODO: Remove this workaround once TTNN bfloat16 softmax is fixed
    //       Performance-critical applications should NOT rely on this workaround
    config.fp32_dest_acc_en = use_fp32_accumulation_workaround;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.packer_l1_acc = true;
    return config;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::matmul() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.packer_l1_acc = true;
    return config;
}

ttnn::WormholeComputeKernelConfig ComputeKernelConfig::fast() {
    ttnn::WormholeComputeKernelConfig config;
    config.fp32_dest_acc_en = false;
    config.math_approx_mode = true;
    config.math_fidelity = MathFidelity::LoFi;
    config.packer_l1_acc = false;
    return config;
}

}  // namespace ttml::core
