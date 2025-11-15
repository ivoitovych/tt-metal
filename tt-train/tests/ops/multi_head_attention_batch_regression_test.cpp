// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for multi-head attention batch processing
//
// This test verifies that multi-head attention operations work correctly with different batch sizes.
// It ensures that:
// - modules::MultiHeadAttention() works with batch_size > 1
// - ops::heads_creation() correctly handles batch dimensions for Q, K, V tensors
// - All batch dimensions are preserved correctly through the attention mechanism
//
// These tests serve as regression tests to prevent batch processing issues in the future.
// This test is BERT-independent and tests the core multi-head attention ops directly.

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/multi_head_attention.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

class MultiHeadAttentionBatchRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Test multi-head attention with batch_size=1 (baseline - this works)
TEST_F(MultiHeadAttentionBatchRegressionTest, MultiHeadAttentionBatchSize1_Baseline) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t batch_size = 1;
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 128;
    uint32_t num_heads = 2;

    // Create random input
    std::vector<float> input_data((size_t)batch_size * seq_len * embedding_dim);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0F - 1.0F;
    }
    auto input_tensor = core::from_vector(input_data, ttnn::Shape({batch_size, 1, seq_len, embedding_dim}), device);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    // Create MultiHeadAttention module
    auto mha = modules::MultiHeadAttention(embedding_dim, num_heads, 0.0F);

    // This should work without issues
    EXPECT_NO_THROW({
        auto output = mha(input, nullptr);
        std::cout << "Batch size 1: Success - Output shape: " << output->get_value().logical_shape() << std::endl;
    });
}

// Test multi-head attention with batch_size=2 (regression test for batch processing)
TEST_F(MultiHeadAttentionBatchRegressionTest, MultiHeadAttentionBatchSize2) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t batch_size = 2;  // Testing with larger batch size
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 128;
    uint32_t num_heads = 2;

    // Create random input
    std::srand(42);
    std::vector<float> input_data((size_t)batch_size * seq_len * embedding_dim);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0F - 1.0F;
    }
    auto input_tensor = core::from_vector(input_data, ttnn::Shape({batch_size, 1, seq_len, embedding_dim}), device);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    // Create MultiHeadAttention module
    auto mha = modules::MultiHeadAttention(embedding_dim, num_heads, 0.0F);

    // Multi-head attention should work correctly with batch_size > 1
    EXPECT_NO_THROW({
        auto output = mha(input, nullptr);
        std::cout << "Batch size 2: Success - Output shape: " << output->get_value().logical_shape() << std::endl;
    });
}

// Test ops::heads_creation directly to isolate the bug
TEST_F(MultiHeadAttentionBatchRegressionTest, HeadsCreationBatchSize2_DirectTest) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t batch_size = 2;
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 128;
    uint32_t num_heads = 2;

    // Create QKV tensor (output of linear layer before splitting)
    // Shape should be [batch_size, 1, seq_len, embedding_dim * 3]
    std::vector<float> qkv_data((size_t)batch_size * seq_len * embedding_dim * 3);
    for (size_t i = 0; i < qkv_data.size(); i++) {
        qkv_data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0F - 1.0F;
    }
    auto qkv_tensor = core::from_vector(qkv_data, ttnn::Shape({batch_size, 1, seq_len, embedding_dim * 3}), device);
    autograd::TensorPtr qkv = autograd::create_tensor(qkv_tensor);

    std::cout << "Input QKV shape: " << qkv->get_value().logical_shape() << std::endl;

    // Call heads_creation
    auto [query, key, value] = ops::heads_creation(qkv, num_heads);

    std::cout << "After heads_creation:" << std::endl;
    std::cout << "  Query shape: " << query->get_value().logical_shape() << std::endl;
    std::cout << "  Key shape:   " << key->get_value().logical_shape() << std::endl;
    std::cout << "  Value shape: " << value->get_value().logical_shape() << std::endl;

    // Expected shapes (all should have batch_size=2):
    // Query: [batch_size=2, num_heads=2, seq_len=32, head_dim=64]
    // Key:   [batch_size=2, num_heads=2, seq_len=32, head_dim=64]
    // Value: [batch_size=2, num_heads=2, seq_len=32, head_dim=64]

    auto query_shape = query->get_value().logical_shape();
    auto key_shape = key->get_value().logical_shape();
    auto value_shape = value->get_value().logical_shape();

    // Check batch dimension
    uint32_t query_batch = query_shape[0];
    uint32_t key_batch = key_shape[0];
    uint32_t value_batch = value_shape[0];

    std::cout << "Batch dimensions: Q=" << query_batch << ", K=" << key_batch << ", V=" << value_batch << std::endl;

    // All tensors should have the correct batch dimension
    EXPECT_EQ(query_batch, batch_size) << "Query batch dimension should be " << batch_size;
    EXPECT_EQ(key_batch, batch_size) << "Key batch dimension should be " << batch_size << " but got " << key_batch;
    EXPECT_EQ(value_batch, batch_size) << "Value batch dimension should be " << batch_size << " but got "
                                       << value_batch;

    // All three should have the same batch size
    EXPECT_EQ(query_batch, key_batch) << "Query and Key should have same batch size";
    EXPECT_EQ(query_batch, value_batch) << "Query and Value should have same batch size";
}

// Test that demonstrates batch_size=1 works correctly in heads_creation
TEST_F(MultiHeadAttentionBatchRegressionTest, HeadsCreationBatchSize1_Baseline) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t batch_size = 1;
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 128;
    uint32_t num_heads = 2;

    // Create QKV tensor
    std::vector<float> qkv_data((size_t)batch_size * seq_len * embedding_dim * 3);
    for (size_t i = 0; i < qkv_data.size(); i++) {
        qkv_data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0F - 1.0F;
    }
    auto qkv_tensor = core::from_vector(qkv_data, ttnn::Shape({batch_size, 1, seq_len, embedding_dim * 3}), device);
    autograd::TensorPtr qkv = autograd::create_tensor(qkv_tensor);

    // Call heads_creation
    auto [query, key, value] = ops::heads_creation(qkv, num_heads);

    auto query_shape = query->get_value().logical_shape();
    auto key_shape = key->get_value().logical_shape();
    auto value_shape = value->get_value().logical_shape();

    // With batch_size=1, all should work correctly
    EXPECT_EQ(query_shape[0], batch_size);
    EXPECT_EQ(key_shape[0], batch_size);
    EXPECT_EQ(value_shape[0], batch_size);

    std::cout << "Batch size 1 (baseline):" << std::endl;
    std::cout << "  Query shape: " << query_shape << std::endl;
    std::cout << "  Key shape:   " << key_shape << std::endl;
    std::cout << "  Value shape: " << value_shape << std::endl;
    std::cout << "  All batch dimensions correct: "
              << (query_shape[0] == key_shape[0] && key_shape[0] == value_shape[0] ? "PASS" : "FAIL") << std::endl;
}

// Test with larger batch size to verify the issue scales
TEST_F(MultiHeadAttentionBatchRegressionTest, HeadsCreationBatchSize4_ScalingTest) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t batch_size = 4;  // Even larger batch
    uint32_t seq_len = 32;
    uint32_t embedding_dim = 128;
    uint32_t num_heads = 2;

    std::vector<float> qkv_data((size_t)batch_size * seq_len * embedding_dim * 3);
    for (size_t i = 0; i < qkv_data.size(); i++) {
        qkv_data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0F - 1.0F;
    }
    auto qkv_tensor = core::from_vector(qkv_data, ttnn::Shape({batch_size, 1, seq_len, embedding_dim * 3}), device);
    autograd::TensorPtr qkv = autograd::create_tensor(qkv_tensor);

    auto [query, key, value] = ops::heads_creation(qkv, num_heads);

    auto query_batch = query->get_value().logical_shape()[0];
    auto key_batch = key->get_value().logical_shape()[0];
    auto value_batch = value->get_value().logical_shape()[0];

    std::cout << "Batch size 4 test:" << std::endl;
    std::cout << "  Expected batch: " << batch_size << std::endl;
    std::cout << "  Query batch: " << query_batch << std::endl;
    std::cout << "  Key batch: " << key_batch << std::endl;
    std::cout << "  Value batch: " << value_batch << std::endl;

    // All tensors should have correct batch dimensions
    EXPECT_EQ(query_batch, batch_size) << "Query batch dimension should be " << batch_size;
    EXPECT_EQ(key_batch, batch_size) << "Key batch dimension should be " << batch_size;
    EXPECT_EQ(value_batch, batch_size) << "Value batch dimension should be " << batch_size;
}
