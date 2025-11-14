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
    // WORKAROUND (NOT A FIX): FP32 accumulation to avoid bfloat16 precision bug
    // Bug: Softmax with bfloat16 accumulation loses precision on attention patterns (PCC 0.81)
    // Workaround: Use FP32 accumulation (performance penalty but PCC >0.999)
    // Real fix needed: TTNN/hardware team must fix bfloat16 softmax kernel
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
