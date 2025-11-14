// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for embeddings batch processing
//
// This test verifies that ops::embedding_op() works correctly with different batch sizes.
// It ensures that batch processing maintains high accuracy (PCC > 0.999) across
// different batch sizes (1, 2, 4, etc.).
//
// These tests serve as regression tests to prevent batch processing issues in the future.
// This test is BERT-independent and tests the core embedding operation directly.

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/embedding_op.hpp"

class EmbeddingBatchRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }

    // Helper function to compute Pearson Correlation Coefficient
    float compute_pcc(const std::vector<float>& tensor1, const std::vector<float>& tensor2) {
        if (tensor1.size() != tensor2.size()) {
            return 0.0F;
        }

        float mean1 = 0.0F, mean2 = 0.0F;
        for (size_t i = 0; i < tensor1.size(); i++) {
            mean1 += tensor1[i];
            mean2 += tensor2[i];
        }
        mean1 /= static_cast<float>(tensor1.size());
        mean2 /= static_cast<float>(tensor2.size());

        float numerator = 0.0F;
        float denom1 = 0.0F, denom2 = 0.0F;
        for (size_t i = 0; i < tensor1.size(); i++) {
            float diff1 = tensor1[i] - mean1;
            float diff2 = tensor2[i] - mean2;
            numerator += diff1 * diff2;
            denom1 += diff1 * diff1;
            denom2 += diff2 * diff2;
        }

        float denominator = std::sqrt(denom1 * denom2);
        if (denominator == 0.0F) {
            return 0.0F;
        }

        return numerator / denominator;
    }
};

// Test embedding operation with batch_size=1 (baseline - this works)
TEST_F(EmbeddingBatchRegressionTest, EmbeddingBatchSize1_Baseline) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t vocab_size = 128;    // Use 32-divisible size for better compatibility
    uint32_t embedding_dim = 64;  // Use 32-divisible size
    uint32_t batch_size = 1;
    uint32_t seq_len = 32;

    // Create simple sequential embedding weights for easy verification
    // Weight[i] = [i, i, i, ...] for easier debugging
    std::vector<float> weight_data((size_t)vocab_size * embedding_dim);
    for (uint32_t i = 0; i < vocab_size; i++) {
        for (uint32_t j = 0; j < embedding_dim; j++) {
            weight_data[i * embedding_dim + j] = static_cast<float>(i) * 0.1F;  // Simple pattern
        }
    }
    auto weight_tensor = core::from_vector(weight_data, ttnn::Shape({1, 1, vocab_size, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    // Create input indices [0, 1, 2, ..., 31]
    std::vector<uint32_t> input_data((size_t)batch_size * seq_len);
    for (uint32_t i = 0; i < batch_size * seq_len; i++) {
        input_data[i] = i % vocab_size;
    }
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    // Run embedding operation
    autograd::TensorPtr embeddings = ops::embedding_op(input, weight);

    // Get output
    auto embeddings_data = core::to_vector(embeddings->get_value());

    // Expected output: for each index i, output should be weight[i,:]
    std::vector<float> expected_output;
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {
            uint32_t idx = input_data[b * seq_len + s];
            for (uint32_t e = 0; e < embedding_dim; e++) {
                expected_output.push_back(weight_data[idx * embedding_dim + e]);
            }
        }
    }

    // Compute PCC
    float pcc = compute_pcc(embeddings_data, expected_output);

    // Also check mean absolute difference
    float mean_abs_diff = 0.0F;
    float max_abs_diff = 0.0F;
    for (size_t i = 0; i < embeddings_data.size(); i++) {
        float diff = std::abs(embeddings_data[i] - expected_output[i]);
        mean_abs_diff += diff;
        max_abs_diff = std::max(max_abs_diff, diff);
    }
    mean_abs_diff /= static_cast<float>(embeddings_data.size());

    std::cout << "Batch size 1 results:" << std::endl;
    std::cout << "  PCC: " << pcc << std::endl;
    std::cout << "  Mean abs diff: " << mean_abs_diff << std::endl;
    std::cout << "  Max abs diff: " << max_abs_diff << std::endl;

    // With batch_size=1, this should achieve very high accuracy (PCC > 0.999)
    EXPECT_GT(pcc, 0.999F) << "batch_size=1 PCC should be > 0.999, got: " << pcc;
}

// Test embedding operation with batch_size=2 (regression test for batch processing)
TEST_F(EmbeddingBatchRegressionTest, EmbeddingBatchSize2) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t vocab_size = 128;    // Use 32-divisible size for better compatibility
    uint32_t embedding_dim = 64;  // Use 32-divisible size
    uint32_t batch_size = 2;      // THIS IS THE KEY DIFFERENCE
    uint32_t seq_len = 32;

    // Create simple sequential embedding weights for easy verification
    std::vector<float> weight_data((size_t)vocab_size * embedding_dim);
    for (uint32_t i = 0; i < vocab_size; i++) {
        for (uint32_t j = 0; j < embedding_dim; j++) {
            weight_data[i * embedding_dim + j] = static_cast<float>(i) * 0.1F;  // Simple pattern
        }
    }
    auto weight_tensor = core::from_vector(weight_data, ttnn::Shape({1, 1, vocab_size, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    // Create input indices [0, 1, 2, ..., 63] for 2 batches
    std::vector<uint32_t> input_data((size_t)batch_size * seq_len);
    for (uint32_t i = 0; i < batch_size * seq_len; i++) {
        input_data[i] = i % vocab_size;
    }
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    // Run embedding operation
    autograd::TensorPtr embeddings = ops::embedding_op(input, weight);

    // Get output
    auto embeddings_data = core::to_vector(embeddings->get_value());

    // Expected output: for each index i, output should be weight[i,:]
    std::vector<float> expected_output;
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {
            uint32_t idx = input_data[b * seq_len + s];
            for (uint32_t e = 0; e < embedding_dim; e++) {
                expected_output.push_back(weight_data[idx * embedding_dim + e]);
            }
        }
    }

    // Compute PCC
    float pcc = compute_pcc(embeddings_data, expected_output);

    // Also check mean absolute difference
    float mean_abs_diff = 0.0F;
    float max_abs_diff = 0.0F;
    for (size_t i = 0; i < embeddings_data.size(); i++) {
        float diff = std::abs(embeddings_data[i] - expected_output[i]);
        mean_abs_diff += diff;
        max_abs_diff = std::max(max_abs_diff, diff);
    }
    mean_abs_diff /= static_cast<float>(embeddings_data.size());

    std::cout << "Batch size 2 results:" << std::endl;
    std::cout << "  PCC: " << pcc << std::endl;
    std::cout << "  Mean abs diff: " << mean_abs_diff << std::endl;
    std::cout << "  Max abs diff: " << max_abs_diff << std::endl;

    // Batch processing should maintain high accuracy (PCC > 0.999)
    EXPECT_GT(pcc, 0.999F) << "batch_size=2 PCC should be > 0.999 (same as batch_size=1), "
                           << "but got: " << pcc;

    // Additional check: mean abs diff should be small (< 0.01)
    EXPECT_LT(mean_abs_diff, 0.01F) << "Mean abs diff should be < 0.01, "
                                    << "but got: " << mean_abs_diff;

    // Additional check: max abs diff should be < 0.1
    EXPECT_LT(max_abs_diff, 0.1F) << "Max abs diff should be < 0.1, "
                                  << "but got: " << max_abs_diff;
}

// Test that demonstrates the bug is NOT in the expected output calculation
TEST_F(EmbeddingBatchRegressionTest, VerifyExpectedOutputIsCorrect) {
    // This test verifies that our expected output calculation is correct
    // by manually checking a few values

    uint32_t vocab_size = 100;
    uint32_t embedding_dim = 32;
    uint32_t batch_size = 2;
    uint32_t seq_len = 4;

    std::srand(42);
    std::vector<float> weight_data((size_t)vocab_size * embedding_dim);
    for (size_t i = 0; i < weight_data.size(); i++) {
        weight_data[i] = static_cast<float>(i);  // Simple pattern for verification
    }

    std::vector<uint32_t> input_data = {0, 1, 2, 3, 4, 5, 6, 7};  // 2 batches, 4 tokens each

    // Expected: batch 0, token 0, index 0 should give weight[0:32]
    // Expected: batch 0, token 1, index 1 should give weight[32:64]
    // Expected: batch 1, token 0, index 4 should give weight[128:160]
    // etc.

    std::vector<float> expected_first_embedding(embedding_dim);
    for (uint32_t e = 0; e < embedding_dim; e++) {
        expected_first_embedding[e] = weight_data[0 * embedding_dim + e];  // index 0
    }

    // First embedding should be [0, 1, 2, ..., 31]
    for (uint32_t e = 0; e < embedding_dim; e++) {
        EXPECT_FLOAT_EQ(expected_first_embedding[e], static_cast<float>(e));
    }

    std::vector<float> expected_fifth_embedding(embedding_dim);
    for (uint32_t e = 0; e < embedding_dim; e++) {
        expected_fifth_embedding[e] = weight_data[4 * embedding_dim + e];  // index 4 (batch 1, token 0)
    }

    // Fifth embedding should be [128, 129, 130, ..., 159]
    for (uint32_t e = 0; e < embedding_dim; e++) {
        EXPECT_FLOAT_EQ(expected_fifth_embedding[e], static_cast<float>(128 + e));
    }

    std::cout << "Expected output calculation verified correct" << std::endl;
}
