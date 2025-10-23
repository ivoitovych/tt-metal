// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {

// LayerNorm operation with configurable epsilon
// Default eps=1e-5F is safe for most cases; BERT uses 1e-12F (see bert.hpp)
autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& gamma,
    const autograd::TensorPtr& beta,
    float eps = 1e-5F);

// Composite LayerNorm (manual implementation) with configurable epsilon
autograd::TensorPtr composite_layernorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& gamma,
    const autograd::TensorPtr& beta,
    float eps = 1e-5F);

}  // namespace ttml::ops
