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
    float m_eps = 1e-5F;  // Epsilon for numerical stability
    autograd::TensorPtr m_gamma;
    autograd::TensorPtr m_beta;

public:
    void initialize_tensors(uint32_t features);
    explicit LayerNormLayer(uint32_t features, float eps = 1e-5F, bool use_composite_op = false);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;

    // Accessor for testing and introspection
    [[nodiscard]] float get_epsilon() const {
        return m_eps;
    }
};

}  // namespace ttml::modules
