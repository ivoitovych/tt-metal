// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <tuple>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/layernorm_bw/layernorm_bw.hpp"

// Reference implementation using xtensor
struct LayerNormCache {
    xt::xarray<float> x;
    xt::xarray<float> x_hat;
    xt::xarray<float> mu;
    xt::xarray<float> s;
    xt::xarray<float> gamma;
    xt::xarray<float> beta;
    uint32_t batch_size;
    uint32_t features;
    float eps;
};

std::tuple<xt::xarray<float>, LayerNormCache> layernorm_forward_reference(
    const xt::xarray<float>& x,
    const xt::xarray<float>& gamma,
    const xt::xarray<float>& beta,
    uint32_t batch_size,
    uint32_t features,
    float eps = 1e-6f) {
    // Reshape input to (batch_size, features) for easier manipulation
    auto x_reshaped = xt::reshape_view(x, {batch_size, features});

    // Compute mean along features axis
    xt::xarray<float> mu = xt::mean(x_reshaped, {1});

    // Compute variance along features axis
    xt::xarray<float> x_centered = x_reshaped - xt::view(mu, xt::all(), xt::newaxis());
    xt::xarray<float> var = xt::mean(xt::square(x_centered), {1});

    // Compute standard deviation with epsilon
    xt::xarray<float> s = xt::sqrt(var + eps);

    // Normalize
    xt::xarray<float> x_hat = x_centered / xt::view(s, xt::all(), xt::newaxis());

    // Scale and shift
    xt::xarray<float> y = x_hat * xt::view(gamma, xt::newaxis(), xt::all()) + xt::view(beta, xt::newaxis(), xt::all());

    // Flatten outputs back to 1D
    y = xt::flatten(y);
    x_hat = xt::flatten(x_hat);

    LayerNormCache cache{x, x_hat, mu, s, gamma, beta, batch_size, features, eps};
    return std::make_tuple(y, cache);
}

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> layernorm_backward_reference(
    const xt::xarray<float>& dy, const LayerNormCache& cache) {
    // Reshape to (batch_size, features)
    auto dy_reshaped = xt::reshape_view(dy, {cache.batch_size, cache.features});
    auto x_hat_reshaped = xt::reshape_view(cache.x_hat, {cache.batch_size, cache.features});

    // Compute dgamma and dbeta - sum over batch dimension
    xt::xarray<float> dgamma = xt::sum(dy_reshaped * x_hat_reshaped, {0});
    xt::xarray<float> dbeta = xt::sum(dy_reshaped, {0});

    // Compute dxhat = dy * gamma
    xt::xarray<float> dxhat = dy_reshaped * xt::view(cache.gamma, xt::newaxis(), xt::all());

    // Compute mean_dxhat - mean along features axis
    xt::xarray<float> mean_dxhat = xt::mean(dxhat, {1});

    // Compute mean_dxhat_xhat - mean along features axis
    xt::xarray<float> mean_dxhat_xhat = xt::mean(dxhat * x_hat_reshaped, {1});

    // Compute final dx
    xt::xarray<float> dx = (dxhat - xt::view(mean_dxhat, xt::all(), xt::newaxis()) -
                            x_hat_reshaped * xt::view(mean_dxhat_xhat, xt::all(), xt::newaxis())) /
                           xt::view(cache.s, xt::all(), xt::newaxis());

    // Flatten dx back to 1D
    dx = xt::flatten(dx);

    return std::make_tuple(dx, dgamma, dbeta);
}

class LayerNormBackwardOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Helper function to compare metal kernel results against xtensor reference
static void CompareKernelVsXArray(
    const uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const int num_iterations = 3) {
    using namespace ttml;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Generate test data using flattened 1D arrays
        uint32_t total_elements = batch_size * seq_len * heads * features;
        uint32_t combined_batch = batch_size * seq_len * heads;

        xt::xarray<float> x_data = xt::empty<float>({total_elements});
        auto rng = autograd::ctx().get_generator();
        uint32_t seed1 = rng();
        core::parallel_generate<float>(
            x_data, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed1);

        xt::xarray<float> gamma_data = xt::empty<float>({features});
        uint32_t seed2 = rng();
        core::parallel_generate<float>(
            gamma_data, []() { return std::uniform_real_distribution<float>(0.0F, 1.0F); }, seed2);

        xt::xarray<float> beta_data = xt::empty<float>({features});
        uint32_t seed3 = rng();
        core::parallel_generate<float>(
            beta_data, []() { return std::uniform_real_distribution<float>(0.0F, 1.0F); }, seed3);

        xt::xarray<float> dy_data = xt::empty<float>({total_elements});
        uint32_t seed4 = rng();
        core::parallel_generate<float>(
            dy_data, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed4);

        // Compute reference results
        auto [y_ref, cache] =
            layernorm_forward_reference(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        // Copy and reshape data to 4D for device tensors (copy to avoid corrupting reference data)
        xt::xarray<float> x_4d = x_data;
        x_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> gamma_4d = gamma_data;
        gamma_4d.reshape({1, 1, 1, features});
        xt::xarray<float> dy_4d = dy_data;
        dy_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> mu_4d = cache.mu;
        mu_4d.reshape({batch_size, heads, seq_len, 1});

        // Compute rstd from s and reshape
        xt::xarray<float> rstd_data = 1.0f / cache.s;
        rstd_data.reshape({batch_size, heads, seq_len, 1});

        // Create tensors on device using from_xtensor
        auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
        auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
        auto mean_tensor = core::from_xtensor(mu_4d, &autograd::ctx().get_device());
        auto rstd_tensor = core::from_xtensor(rstd_data, &autograd::ctx().get_device());
        auto dy_tensor = core::from_xtensor(dy_4d, &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx_xtensor = core::to_xtensor(output_tensors[0].value());
        auto metal_dgamma_xtensor = core::to_xtensor(output_tensors[1].value());
        auto metal_dbeta_xtensor = core::to_xtensor(output_tensors[2].value());

        // Flatten metal results for comparison
        xt::xarray<float> metal_dx_flat = xt::flatten(metal_dx_xtensor);
        xt::xarray<float> metal_dgamma_flat = xt::flatten(metal_dgamma_xtensor);
        xt::xarray<float> metal_dbeta_flat = xt::flatten(metal_dbeta_xtensor);

        // Compare shapes
        ASSERT_EQ(dx_ref.shape(), metal_dx_flat.shape());
        ASSERT_EQ(dgamma_ref.shape(), metal_dgamma_flat.shape());
        ASSERT_EQ(dbeta_ref.shape(), metal_dbeta_flat.shape());

        // Compare values
        EXPECT_TRUE(xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, 5e-1F));
        EXPECT_TRUE(xt::allclose(metal_dgamma_flat, dgamma_ref, 1.0e-3F, 5e-1F));
        EXPECT_TRUE(xt::allclose(metal_dbeta_flat, dbeta_ref, 1.0e-3F, 5e-1F));
    }
}

// Helper function with configurable tolerance for tighter precision tests
static void CompareKernelVsXArrayWithTolerance(
    const uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const float rtol,
    const float atol,
    const int num_iterations = 3) {
    using namespace ttml;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Generate test data using flattened 1D arrays
        uint32_t total_elements = batch_size * seq_len * heads * features;
        uint32_t combined_batch = batch_size * seq_len * heads;

        xt::xarray<float> x_data = xt::empty<float>({total_elements});
        auto rng = autograd::ctx().get_generator();
        uint32_t seed1 = rng();
        core::parallel_generate<float>(
            x_data, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed1);

        xt::xarray<float> gamma_data = xt::empty<float>({features});
        uint32_t seed2 = rng();
        core::parallel_generate<float>(
            gamma_data, []() { return std::uniform_real_distribution<float>(0.0F, 1.0F); }, seed2);

        xt::xarray<float> beta_data = xt::empty<float>({features});
        uint32_t seed3 = rng();
        core::parallel_generate<float>(
            beta_data, []() { return std::uniform_real_distribution<float>(0.0F, 1.0F); }, seed3);

        xt::xarray<float> dy_data = xt::empty<float>({total_elements});
        uint32_t seed4 = rng();
        core::parallel_generate<float>(
            dy_data, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed4);

        // Compute reference results
        auto [y_ref, cache] =
            layernorm_forward_reference(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        // Copy and reshape data to 4D for device tensors (copy to avoid corrupting reference data)
        xt::xarray<float> x_4d = x_data;
        x_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> gamma_4d = gamma_data;
        gamma_4d.reshape({1, 1, 1, features});
        xt::xarray<float> dy_4d = dy_data;
        dy_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> mu_4d = cache.mu;
        mu_4d.reshape({batch_size, heads, seq_len, 1});

        // Compute rstd from s and reshape
        xt::xarray<float> rstd_data = 1.0f / cache.s;
        rstd_data.reshape({batch_size, heads, seq_len, 1});

        // Create tensors on device using from_xtensor
        auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
        auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
        auto mean_tensor = core::from_xtensor(mu_4d, &autograd::ctx().get_device());
        auto rstd_tensor = core::from_xtensor(rstd_data, &autograd::ctx().get_device());
        auto dy_tensor = core::from_xtensor(dy_4d, &autograd::ctx().get_device());

        auto output_tensors = metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke(
            input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx_xtensor = core::to_xtensor(output_tensors[0].value());
        auto metal_dgamma_xtensor = core::to_xtensor(output_tensors[1].value());
        auto metal_dbeta_xtensor = core::to_xtensor(output_tensors[2].value());

        // Flatten metal results for comparison
        xt::xarray<float> metal_dx_flat = xt::flatten(metal_dx_xtensor);
        xt::xarray<float> metal_dgamma_flat = xt::flatten(metal_dgamma_xtensor);
        xt::xarray<float> metal_dbeta_flat = xt::flatten(metal_dbeta_xtensor);

        // Compare shapes
        ASSERT_EQ(dx_ref.shape(), metal_dx_flat.shape());
        ASSERT_EQ(dgamma_ref.shape(), metal_dgamma_flat.shape());
        ASSERT_EQ(dbeta_ref.shape(), metal_dbeta_flat.shape());

        // Compare values with configurable tolerance
        EXPECT_TRUE(xt::allclose(metal_dx_flat, dx_ref, rtol, atol))
            << "dx failed with tighter tolerance (iter=" << iter << ")";
        EXPECT_TRUE(xt::allclose(metal_dgamma_flat, dgamma_ref, rtol, atol))
            << "dgamma failed with tighter tolerance (iter=" << iter << ")";
        EXPECT_TRUE(xt::allclose(metal_dbeta_flat, dbeta_ref, rtol, atol))
            << "dbeta failed with tighter tolerance (iter=" << iter << ")";
    }
}

// ============================================================================
// Test Cases - LayerNorm Backward Metal Kernel vs XArray Reference
// ============================================================================

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_OneTile) {
    CompareKernelVsXArray(1, 13, 1, 20);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_TwoIncompleteTiles) {
    CompareKernelVsXArray(1, 32, 1, 33);
}

TEST_F(LayerNormBackwardOpTest, NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit) {
    CompareKernelVsXArray(3, 273, 1, 8462);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4) {
    CompareKernelVsXArray(3, 100, 1, 8191, 10);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_OneTilePerRow) {
    CompareKernelVsXArray(1, 19, 1, 213, 10);
}

// ============================================================================
// Test Isolation Investigation Tests
// ============================================================================
//
// These tests use tighter tolerance (atol=0.2) to detect device state corruption
// that occurs when certain test suites run before LayerNorm backward tests.
//
// REPRODUCTION:
// 1. Test passes in isolation:
//    ./tt-train/build/tests/ttml_tests
//    --gtest_filter="LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features"
//
// 2. Test FAILS after running SoftmaxTest + SiLUOpTest + SDPAForwardTest.NIGHTLY:
//    rm -rf ~/.cache/tt-metal-cache/*
//    ./tt-train/build/tests/ttml_tests
//    --gtest_filter="SoftmaxTest.*:SiLUOpTest.*:SDPAForwardTest.NIGHTLY*:LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features"
//
// KEY FINDINGS:
// - Failure requires ALL THREE: SoftmaxTest + SiLUOpTest + BOTH SDPA NIGHTLY tests
// - Any two of the three test suites pass - only the triple combination fails
// - Only 4096 features (128 tiles) fails; 8192 features (256 tiles) passes
// - Standard tolerance (atol=0.5) masks the issue; tighter tolerance exposes it
//

// This test exposes device state corruption after SoftmaxTest + SiLUOpTest + SDPA NIGHTLY.
// Uses tighter tolerance (atol=0.2) which is still reasonable for bfloat16.
TEST_F(LayerNormBackwardOpTest, TestIsolation_TighterTolerance_4096Features) {
    // batch=1, seq_len=100, heads=1, features=4096 (128 tiles)
    // Tighter tolerance: rtol=0.01, atol=0.2 (standard is 0.5)
    CompareKernelVsXArrayWithTolerance(1, 100, 1, 4096, 1.0e-2F, 2.0e-1F, 3);
}

// Control test with same tolerance but different tile count - should PASS in same sequence.
TEST_F(LayerNormBackwardOpTest, TestIsolation_TighterTolerance_8192Features) {
    // batch=1, seq_len=100, heads=1, features=8192 (256 tiles)
    // Same tighter tolerance - this one passes, suggesting tile-count-specific issue
    CompareKernelVsXArrayWithTolerance(1, 100, 1, 8192, 1.0e-2F, 2.0e-1F, 3);
}
