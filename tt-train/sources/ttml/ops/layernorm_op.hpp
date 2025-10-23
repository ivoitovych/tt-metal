// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {

// LayerNorm operation with configurable epsilon and hardware clamping control
// Default eps=1e-5F is safe for most cases; BERT uses 1e-12F (see bert.hpp)
// enable_hardware_clamp: When true, applies max(eps, 1e-4F) for BFLOAT16 safety
autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& gamma,
    const autograd::TensorPtr& beta,
    float eps = 1e-5F,
    bool enable_hardware_clamp = true);

// Composite LayerNorm (manual implementation) with configurable epsilon and hardware clamping
autograd::TensorPtr composite_layernorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& gamma,
    const autograd::TensorPtr& beta,
    float eps = 1e-5F,
    bool enable_hardware_clamp = true);

}  // namespace ttml::ops
