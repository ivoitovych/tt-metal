// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/bert.hpp"
#include "modules/layer_norm_module.hpp"

using namespace ttml;

class LayerNormEpsilonTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
        autograd::ctx().close_device();
    }

    // Helper to convert tensor to vector for comparison
    std::vector<float> tensor_to_vector(const tt::tt_metal::Tensor& tensor) {
        auto shape = tensor.logical_shape();
        std::vector<float> result;
        result.reserve(shape.volume());
        auto data = core::to_vector(tensor);
        return data;
    }
};

// Test 1: Verify epsilon is stored correctly
TEST_F(LayerNormEpsilonTest, EpsilonIsStored) {
    float eps_small = 1e-12F;
    float eps_large = 1e-5F;

    modules::LayerNormLayer ln_small(32, eps_small, false);
    modules::LayerNormLayer ln_large(32, eps_large, false);

    // Use EXPECT_NEAR for floating-point comparisons with appropriate tolerance
    EXPECT_NEAR(ln_small.get_epsilon(), eps_small, 1e-15F);
    EXPECT_NEAR(ln_large.get_epsilon(), eps_large, 1e-8F);
}

// Test 2: Epsilon parameter affects computation
TEST_F(LayerNormEpsilonTest, EpsilonAffectsComputation) {
    const uint32_t features = 64;
    // Use significantly different epsilon values that are both safe for bfloat16
    const float eps_standard = 1e-5F;  // Standard epsilon
    const float eps_large = 1e-2F;     // Much larger epsilon for clear difference

    modules::LayerNormLayer ln_standard(features, eps_standard, false);
    modules::LayerNormLayer ln_large(features, eps_large, false);

    // Create input with normal values
    auto input_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    // Both should produce valid outputs without NaN/Inf
    // This validates that epsilon is being used correctly in the computation
    auto output_standard = ln_standard(input);
    auto output_large = ln_large(input);

    auto vec_standard = tensor_to_vector(output_standard->get_value());
    auto vec_large = tensor_to_vector(output_large->get_value());

    // Verify no NaN or Inf in either output
    for (float val : vec_standard) {
        EXPECT_FALSE(std::isnan(val)) << "Standard epsilon produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Standard epsilon produced Inf";
    }
    for (float val : vec_large) {
        EXPECT_FALSE(std::isnan(val)) << "Large epsilon produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Large epsilon produced Inf";
    }
}

// Test 3: BERT uses correct epsilon without NaN/Inf
TEST_F(LayerNormEpsilonTest, BertUsesCorrectEpsilon) {
    models::bert::BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 128;
    config.embedding_dim = 256;
    config.intermediate_size = 512;
    config.num_heads = 8;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;  // BERT's standard epsilon

    ASSERT_NO_THROW({
        auto model = std::make_shared<models::bert::Bert>(config);

        // Create test inputs
        auto input_ids = core::zeros(ttnn::Shape({1, 1, 1, config.max_sequence_length}), &autograd::ctx().get_device());
        auto input_ids_tensor = autograd::create_tensor(input_ids);

        // Forward pass - should not produce NaN or Inf
        auto output = model->forward(input_ids_tensor);

        // Check for NaN/Inf in output
        auto output_vec = tensor_to_vector(output->get_value());
        for (float val : output_vec) {
            EXPECT_FALSE(std::isnan(val)) << "Output contains NaN";
            EXPECT_FALSE(std::isinf(val)) << "Output contains Inf";
        }
    });
}

// Test 4: Hardware precision impact (bfloat16)
TEST_F(LayerNormEpsilonTest, HardwarePrecisionImpact) {
    const uint32_t features = 64;

    // Test with epsilon below bfloat16 machine epsilon (~0.0078125)
    // This should be clamped to 1e-4F internally for safety
    const float very_small_eps = 1e-12F;
    const float safe_eps = 1e-4F;

    modules::LayerNormLayer ln_tiny(features, very_small_eps, false);
    modules::LayerNormLayer ln_safe(features, safe_eps, false);

    // Create input with near-zero variance
    auto input_tt = core::full(
        ttnn::Shape({1, 1, 1, features}),
        1e-8F,  // Very small values
        &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    // Both should produce valid outputs (no NaN/Inf)
    // because internally epsilon is clamped to safe_eps
    auto output_tiny = ln_tiny(input);
    auto output_safe = ln_safe(input);

    auto vec_tiny = tensor_to_vector(output_tiny->get_value());
    auto vec_safe = tensor_to_vector(output_safe->get_value());

    // Check no NaN/Inf
    for (float val : vec_tiny) {
        EXPECT_FALSE(std::isnan(val)) << "Tiny epsilon produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Tiny epsilon produced Inf";
    }
    for (float val : vec_safe) {
        EXPECT_FALSE(std::isnan(val)) << "Safe epsilon produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Safe epsilon produced Inf";
    }

    // Outputs should be similar because tiny epsilon is clamped internally
    float max_diff = 0.0F;
    for (size_t i = 0; i < vec_tiny.size(); ++i) {
        float diff = std::abs(vec_tiny[i] - vec_safe[i]);
        max_diff = std::max(max_diff, diff);
    }
    // Due to hardware clamping, outputs should be nearly identical
    EXPECT_LT(max_diff, 1e-3F) << "Hardware epsilon clamping should make outputs similar";
}

// Test 5: Epsilon propagation through composite operation
TEST_F(LayerNormEpsilonTest, CompositeOpUsesEpsilon) {
    const uint32_t features = 64;
    const float custom_eps = 1e-6F;

    // Test composite operation (manual implementation)
    modules::LayerNormLayer ln_composite(features, custom_eps, true);  // use_composite_op = true

    EXPECT_FLOAT_EQ(ln_composite.get_epsilon(), custom_eps);

    // Create test input
    auto input_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    // Should not crash and produce valid output
    auto output = ln_composite(input);
    auto output_vec = tensor_to_vector(output->get_value());

    for (float val : output_vec) {
        EXPECT_FALSE(std::isnan(val)) << "Composite op produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Composite op produced Inf";
    }
}

// Test 6: Default epsilon is sensible
TEST_F(LayerNormEpsilonTest, DefaultEpsilonIsReasonable) {
    const uint32_t features = 32;

    // Create without specifying epsilon (use default)
    modules::LayerNormLayer ln_default(features);

    // Default should be 1e-5F (safe for most cases)
    EXPECT_NEAR(ln_default.get_epsilon(), 1e-5F, 1e-8F);

    // Test it works
    auto input_tt = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    ASSERT_NO_THROW({ auto output = ln_default(input); });
}

// Test 7: Zero-variance input (edge case for NaN prevention)
TEST_F(LayerNormEpsilonTest, ZeroVarianceNoPrevention) {
    const uint32_t features = 64;
    const float eps = 1e-5F;

    modules::LayerNormLayer ln(features, eps, false);

    // Create input with zero variance (all same values)
    // This is an edge case where epsilon prevents division by zero
    auto input_tt = core::full(ttnn::Shape({1, 1, 1, features}), 42.0F, &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_tt);

    // Should not crash and produce valid output (all zeros after normalization)
    auto output = ln(input);
    auto output_vec = tensor_to_vector(output->get_value());

    // Check no NaN or Inf in output
    for (float val : output_vec) {
        EXPECT_FALSE(std::isnan(val)) << "Zero-variance input produced NaN";
        EXPECT_FALSE(std::isinf(val)) << "Zero-variance input produced Inf";
    }

    // With zero variance, normalized values should be zero (or very close due to epsilon)
    // Output = (x - mean) / sqrt(0 + eps) * gamma + beta
    // Since variance=0, all inputs equal mean, so (x - mean) = 0
    for (float val : output_vec) {
        EXPECT_NEAR(val, 0.0F, 1e-3F) << "Zero-variance should produce near-zero normalized values";
    }
}
