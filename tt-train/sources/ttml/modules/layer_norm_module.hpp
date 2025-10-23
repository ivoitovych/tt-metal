// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/tensor.hpp"
#include "modules/module_base.hpp"
#include "ops/layernorm_op.hpp"

namespace ttml::modules {

class LayerNormLayer : public ModuleBase {
private:
    bool m_use_composite_op = false;
    // Epsilon for numerical stability in LayerNorm computation.
    // Selection guide based on tensor dtype:
    // - BFLOAT16: Recommend 1e-5F to 1e-4F (values below 1e-4F are clamped internally
    //   to prevent underflow; bfloat16 machine epsilon ~0.0078125)
    // - FLOAT32: Can use smaller values like 1e-12F (BERT standard) for precision
    // - Default: 1e-5F provides good balance for most use cases
    float m_eps = 1e-5F;
    // Hardware precision clamping control.
    // When true (default): Applies max(eps, 1e-4F) for BFLOAT16 to prevent underflow.
    // When false: Uses epsilon as-is regardless of dtype (expert mode for pure FP32).
    // Recommendation: Keep true unless you have specific numerical requirements.
    bool m_enable_hardware_clamp = true;
    autograd::TensorPtr m_gamma;
    autograd::TensorPtr m_beta;

public:
    void initialize_tensors(uint32_t features);
    explicit LayerNormLayer(
        uint32_t features, float eps = 1e-5F, bool use_composite_op = false, bool enable_hardware_clamp = true);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;

    // Accessors for testing and introspection
    [[nodiscard]] float get_epsilon() const {
        return m_eps;
    }

    [[nodiscard]] bool get_enable_hardware_clamp() const {
        return m_enable_hardware_clamp;
    }
};

}  // namespace ttml::modules
