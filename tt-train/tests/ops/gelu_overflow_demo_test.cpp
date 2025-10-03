// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ==============================================================================
// GELU OVERFLOW DEMONSTRATION TEST
// ==============================================================================
// This test demonstrates a critical numerical stability issue with exact GELU
// implementation when using BFLOAT16 precision.
//
// PROBLEM: The exact GELU uses erf() which causes catastrophic overflow in
//          BFLOAT16, producing values near 2^63 even for moderate inputs.
//
// SOLUTION: Use approximate GELU (tanh-based) which is numerically stable.
//
// This test can be run standalone to demonstrate the issue to colleagues:
//   ./build/tests/ttml_tests --gtest_filter=GeluOverflowDemo.*
// ==============================================================================

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops::tests {

class GeluOverflowDemo : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

// ==============================================================================
// TEST 1: Demonstrates the overflow with current implementation
// ==============================================================================
TEST_F(GeluOverflowDemo, CurrentImplementationOverflows) {
    auto* device = &autograd::ctx().get_device();

    // Create input with typical BERT activation values (after linear projection)
    // These are moderate values that should NOT cause any overflow
    std::vector<float> input_data = {
        -1.5f, -1.0f, -0.5f, 0.0f,
         0.5f,  1.0f,  1.5f, 2.0f
    };

    auto shape = ttnn::Shape({1, 1, 2, 4});
    auto tensor = core::from_vector(input_data, shape, device);
    auto tensor_ptr = autograd::create_tensor(tensor);

    // Apply current GELU implementation (uses exact GELU by default)
    auto result = gelu(tensor_ptr);
    auto result_data = core::to_vector(result->get_value());

    // Check for overflow
    int overflow_count = 0;
    std::cout << "\n=== CURRENT GELU IMPLEMENTATION RESULTS ===\n";
    for (size_t i = 0; i < result_data.size(); ++i) {
        std::cout << "Input: " << input_data[i] << " -> Output: " << result_data[i];
        
        if (std::isnan(result_data[i]) || std::isinf(result_data[i]) || 
            std::abs(result_data[i]) > 1e10f) {
            overflow_count++;
            std::cout << " [OVERFLOW!]";
        }
        std::cout << "\n";
    }

    std::cout << "\n*** RESULT: " << overflow_count << " out of " << result_data.size() 
              << " values overflowed ***\n\n";

    // This test documents the current broken behavior
    // In production, you would use EXPECT_EQ(overflow_count, 0) after the fix
    if (overflow_count > 0) {
        std::cout << "⚠️  WARNING: GELU implementation has numerical overflow issues!\n";
        std::cout << "   This is the BUG we need to fix.\n\n";
    }
}

// ==============================================================================
// TEST 2: Shows that approximate GELU works correctly
// ==============================================================================
TEST_F(GeluOverflowDemo, ApproximateGeluIsStable) {
    auto* device = &autograd::ctx().get_device();

    // Same input data as above
    std::vector<float> input_data = {
        -1.5f, -1.0f, -0.5f, 0.0f,
         0.5f,  1.0f,  1.5f, 2.0f
    };

    auto shape = ttnn::Shape({1, 1, 2, 4});
    auto tensor = core::from_vector(input_data, shape, device);

    // Apply approximate GELU directly (bypass the wrapper function)
    auto result = ttnn::gelu(tensor, /* fast_and_approximate_mode */ true);
    auto result_data = core::to_vector(result);

    // Expected approximate GELU values (tanh-based formula)
    std::vector<float> expected_approx = {
        -0.1305f,  // GELU(-1.5) ≈ -0.13
        -0.1587f,  // GELU(-1.0) ≈ -0.16
        -0.1543f,  // GELU(-0.5) ≈ -0.15
         0.0f,     // GELU(0.0) = 0
         0.3457f,  // GELU(0.5) ≈ 0.35
         0.8413f,  // GELU(1.0) ≈ 0.84
         1.3695f,  // GELU(1.5) ≈ 1.37
         1.9546f   // GELU(2.0) ≈ 1.95
    };

    std::cout << "=== APPROXIMATE GELU RESULTS ===\n";
    int overflow_count = 0;
    for (size_t i = 0; i < result_data.size(); ++i) {
        std::cout << "Input: " << input_data[i] 
                  << " -> Output: " << result_data[i]
                  << " (expected: " << expected_approx[i] << ")";
        
        if (std::isnan(result_data[i]) || std::isinf(result_data[i]) || 
            std::abs(result_data[i]) > 1e10f) {
            overflow_count++;
            std::cout << " [OVERFLOW!]";
        } else {
            EXPECT_NEAR(result_data[i], expected_approx[i], 0.1f);
        }
        std::cout << "\n";
    }

    std::cout << "\n*** RESULT: " << overflow_count << " overflows (should be 0) ***\n";
    EXPECT_EQ(overflow_count, 0) << "Approximate GELU should be numerically stable";
    
    std::cout << "✓ Approximate GELU works correctly!\n\n";
}

// ==============================================================================
// TEST 3: Direct comparison showing the difference
// ==============================================================================
TEST_F(GeluOverflowDemo, DirectComparison) {
    auto* device = &autograd::ctx().get_device();

    std::vector<float> input_data = {-1.0f, 0.0f, 1.0f};
    auto shape = ttnn::Shape({1, 1, 1, 3});
    auto tensor = core::from_vector(input_data, shape, device);

    std::cout << "=== DIRECT COMPARISON ===\n";
    std::cout << "Input values: -1.0, 0.0, 1.0\n\n";

    // Test exact GELU
    auto exact_result = ttnn::gelu(tensor, /* approximate */ false);
    auto exact_data = core::to_vector(exact_result);
    
    std::cout << "Exact GELU output:\n";
    bool exact_stable = true;
    for (size_t i = 0; i < exact_data.size(); ++i) {
        std::cout << "  [" << i << "] = " << exact_data[i];
        if (std::abs(exact_data[i]) > 1e10f) {
            std::cout << " ❌ OVERFLOW";
            exact_stable = false;
        }
        std::cout << "\n";
    }

    // Test approximate GELU
    auto approx_result = ttnn::gelu(tensor, /* approximate */ true);
    auto approx_data = core::to_vector(approx_result);
    
    std::cout << "\nApproximate GELU output:\n";
    bool approx_stable = true;
    for (size_t i = 0; i < approx_data.size(); ++i) {
        std::cout << "  [" << i << "] = " << approx_data[i];
        if (std::abs(approx_data[i]) > 1e10f) {
            std::cout << " ❌ OVERFLOW";
            approx_stable = false;
        } else {
            std::cout << " ✓";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
    std::cout << "Exact GELU:       " << (exact_stable ? "✓ STABLE" : "❌ UNSTABLE") << "\n";
    std::cout << "Approximate GELU: " << (approx_stable ? "✓ STABLE" : "❌ UNSTABLE") << "\n";
    std::cout << "\n";

    EXPECT_TRUE(approx_stable) << "Approximate GELU must be stable";
    
    // Document the current broken state
    if (!exact_stable) {
        std::cout << "⚠️  CONFIRMED: Exact GELU has overflow issues in BFLOAT16\n";
        std::cout << "   SOLUTION: Switch ops::gelu() to use approximate mode\n\n";
    }
}

// ==============================================================================
// TEST 4: BERT-realistic workload showing production impact
// ==============================================================================
TEST_F(GeluOverflowDemo, BERTRealisticWorkload) {
    auto* device = &autograd::ctx().get_device();

    // Simulate BERT intermediate layer: batch=2, seq_len=32, hidden_dim=64
    uint32_t batch = 2;
    uint32_t seq_len = 32;
    uint32_t intermediate_size = 64;
    uint32_t total_elements = batch * seq_len * intermediate_size;

    // Create realistic BERT activation values
    std::vector<float> input_data(total_elements);
    for (size_t i = 0; i < input_data.size(); ++i) {
        // Typical BERT intermediate values range from -2 to +2
        input_data[i] = -2.0f + (i % 128) * (4.0f / 127.0f);
    }

    auto shape = ttnn::Shape({batch, 1, seq_len, intermediate_size});
    auto tensor = core::from_vector(input_data, shape, device);
    auto tensor_ptr = autograd::create_tensor(tensor);

    // Apply current GELU (which uses exact mode)
    auto result = gelu(tensor_ptr);
    auto result_data = core::to_vector(result->get_value());

    // Count overflow occurrences
    int overflow_count = 0;
    for (const auto& val : result_data) {
        if (std::isnan(val) || std::isinf(val) || std::abs(val) > 1e10f) {
            overflow_count++;
        }
    }

    float overflow_percentage = 100.0f * overflow_count / total_elements;

    std::cout << "=== BERT-REALISTIC WORKLOAD ===\n";
    std::cout << "Input shape: [" << batch << ", " << seq_len << ", " << intermediate_size << "]\n";
    std::cout << "Total elements: " << total_elements << "\n";
    std::cout << "Overflowed elements: " << overflow_count 
              << " (" << overflow_percentage << "%)\n";

    if (overflow_count > 0) {
        std::cout << "\n❌ PRODUCTION IMPACT: " << overflow_percentage 
                  << "% of BERT activations are corrupted!\n";
        std::cout << "   This will cause:\n";
        std::cout << "   - Training instability\n";
        std::cout << "   - Poor model convergence\n";
        std::cout << "   - Incorrect predictions\n\n";
    } else {
        std::cout << "\n✓ All values are valid\n\n";
    }

    // In production, after fix: EXPECT_EQ(overflow_count, 0);
}

}  // namespace ttml::ops::tests

