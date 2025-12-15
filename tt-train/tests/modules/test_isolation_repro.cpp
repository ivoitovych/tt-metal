// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Test Isolation Bug Reproduction
 *
 * This file demonstrates a test isolation issue where tests that manipulate
 * the autograd context can affect subsequent tests.
 *
 * Bug: When certain tests run before others, the autograd::ctx() global state
 * may not be properly reset, causing subsequent tests to fail.
 *
 * To reproduce:
 * 1. Run tests in batch: ./ttml_tests --gtest_filter="TestIsolationRepro.*"
 * 2. Run the failing test alone: ./ttml_tests --gtest_filter="TestIsolationRepro.SimpleModuleCreation"
 *
 * If the bug exists, the batch run may fail while individual test passes.
 */

#include <gtest/gtest.h>

#include <memory>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/layer_norm_module.hpp"

using namespace ttml;

class TestIsolationRepro : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
        autograd::ctx().close_device();
    }
};

// Test 1: Heavy tensor operations that may leave residual state
TEST_F(TestIsolationRepro, HeavyTensorOperations) {
    const uint32_t features = 256;
    const uint32_t batch = 4;
    const uint32_t seq = 32;

    // Create multiple tensors to potentially pollute state
    for (int i = 0; i < 5; ++i) {
        auto tensor = core::ones(ttnn::Shape({batch, 1, seq, features}), &autograd::ctx().get_device());

        auto gamma = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
        auto beta = core::zeros(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());

        // Perform operations
        auto result = ttnn::add(tensor, gamma);
        result = ttnn::multiply(result, beta);
    }

    // Reset graph but device state may persist
    autograd::ctx().reset_graph();

    EXPECT_TRUE(true);  // Test passes but may leave state
}

// Test 2: Multiple device open/close cycles
TEST_F(TestIsolationRepro, MultipleDeviceCycles) {
    // Close and reopen device multiple times
    for (int i = 0; i < 3; ++i) {
        autograd::ctx().close_device();
        autograd::ctx().open_device();

        // Create a tensor to verify device works
        auto tensor = core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device());
        EXPECT_EQ(tensor.logical_shape().volume(), 32);
    }
}

// Test 3: Simple module creation - this is the test that may fail due to isolation
// If previous tests left the context in a bad state, this simple test may fail
TEST_F(TestIsolationRepro, SimpleModuleCreation) {
    const uint32_t features = 32;

    // This simple operation should always succeed
    // But may fail if autograd::ctx() is in a bad state from previous tests
    modules::LayerNormLayer ln(features, false);

    // Verify the module was created correctly
    auto params = ln.parameters();
    EXPECT_EQ(params.size(), 2);  // gamma and beta

    // Verify tensors have correct shapes
    for (const auto& [name, tensor] : params) {
        auto shape = tensor->get_value().logical_shape();
        EXPECT_EQ(shape[-1], features);
    }
}

// Test 4: Graph operations that may affect state
TEST_F(TestIsolationRepro, GraphOperations) {
    const uint32_t features = 64;

    auto input_data = core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device());
    auto input = autograd::create_tensor(input_data);

    modules::LayerNormLayer ln(features, false);
    auto output = ln(input);

    // Build up computation graph
    auto target = autograd::create_tensor(core::zeros_like(output->get_value()));

    // Don't call backward - just verify forward works
    auto output_vec = core::to_vector(output->get_value());
    EXPECT_EQ(output_vec.size(), features);
}

// Test 5: Verify context state after previous tests
TEST_F(TestIsolationRepro, ContextStateVerification) {
    // This test verifies that the context is in a clean state
    // It should pass if test isolation is working correctly

    // Check device is accessible
    auto& device = autograd::ctx().get_device();
    EXPECT_TRUE(device.is_initialized());

    // Create a simple tensor
    auto tensor = core::ones(ttnn::Shape({1, 1, 1, 32}), &device);
    EXPECT_EQ(tensor.logical_shape().volume(), 32);

    // Verify we can create autograd tensors
    auto autograd_tensor = autograd::create_tensor(tensor);
    EXPECT_NE(autograd_tensor, nullptr);
}
