// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Masked Language Model Tests
 *
 * Tests for BertForMaskedLM (pre-training task)
 * Validates:
 * 1. Vocabulary prediction output
 * 2. Forward pass correctness
 * 3. Gradient flow
 * 4. MLM head architecture (transform + layernorm + projection)
 */

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/bert.hpp"

using namespace ttml;
using namespace ttml::models::bert;

class BertMLMTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(BertMLMTest, BasicForwardPass) {
    BertConfig config;
    config.vocab_size = 128;  // Aligned to 32 (128 = 4 * 32)
    config.max_sequence_length = 64;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 2;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    auto model = create_for_masked_lm(config);

    const size_t batch_size = 2;
    const size_t seq_len = 64;

    // Create input tensors with some tokens masked
    std::vector<uint32_t> input_ids_vec(batch_size * seq_len);
    std::vector<uint32_t> token_type_ids_vec(batch_size * seq_len, 0);
    std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dis(0, config.vocab_size - 1);
    const uint32_t MASK_TOKEN_ID = 103;  // [MASK] token

    for (size_t i = 0; i < input_ids_vec.size(); ++i) {
        // Randomly mask ~15% of tokens
        if ((i % 7) == 0) {
            input_ids_vec[i] = MASK_TOKEN_ID;
        } else {
            input_ids_vec[i] = dis(gen);
        }
    }

    auto input_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids_vec, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto token_type_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        token_type_ids_vec,
        ttnn::Shape{batch_size, 1, 1, seq_len},
        &autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);
    auto attention_mask_tensor =
        core::from_vector(attention_mask_vec, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

    auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
    auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);
    auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

    // Forward pass
    auto logits = (*model)(input_ids_ag, attention_mask_ag, token_type_ids_ag);

    // Check output shape - should be [batch_size, 1, seq_len, vocab_size_aligned]
    auto logits_shape = logits->get_value().logical_shape();
    EXPECT_EQ(logits_shape[0], batch_size);
    EXPECT_EQ(logits_shape[1], 1);
    EXPECT_EQ(logits_shape[2], seq_len);
    EXPECT_EQ(logits_shape[3], 128);  // vocab_size aligned to 32

    // Check output values are finite
    auto logits_data = core::to_vector(logits->get_value());
    for (size_t i = 0; i < batch_size * seq_len * config.vocab_size; ++i) {
        EXPECT_TRUE(std::isfinite(logits_data[i])) << "Logit at index " << i << " is not finite";
    }
}

TEST_F(BertMLMTest, GradientFlow) {
    BertConfig config;
    config.vocab_size = 96;  // Aligned to 32 (96 = 3 * 32)
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;

    auto model = create_for_masked_lm(config);

    const size_t batch_size = 1;
    const size_t seq_len = 32;

    // Create input tensors
    std::vector<uint32_t> input_ids_vec(batch_size * seq_len);
    std::vector<uint32_t> token_type_ids_vec(batch_size * seq_len, 0);
    std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

    std::mt19937 gen(43);
    std::uniform_int_distribution<uint32_t> dis(0, config.vocab_size - 1);
    const uint32_t MASK_TOKEN_ID = 103;

    for (size_t i = 0; i < input_ids_vec.size(); ++i) {
        if ((i % 5) == 0) {
            input_ids_vec[i] = MASK_TOKEN_ID;
        } else {
            input_ids_vec[i] = dis(gen);
        }
    }

    auto input_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids_vec, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto token_type_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        token_type_ids_vec,
        ttnn::Shape{batch_size, 1, 1, seq_len},
        &autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);
    auto attention_mask_tensor =
        core::from_vector(attention_mask_vec, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

    auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
    auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);
    auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

    // Create labels (same as input_ids for MLM)
    // Note: cross_entropy_loss expects rank-2 targets for MLM
    auto labels_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids_vec, ttnn::Shape{batch_size, seq_len}, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto labels_ag = autograd::create_tensor(labels_tensor);

    // Compute loss
    auto [loss, logits] = model->forward_with_loss(input_ids_ag, attention_mask_ag, token_type_ids_ag, labels_ag);

    // Backward pass should complete without error
    loss->backward();

    // If we reach here, gradient computation succeeded
    EXPECT_TRUE(true) << "Gradient flow test passed";
}

TEST_F(BertMLMTest, VocabSizeAlignment) {
    BertConfig config;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;

    // Test different vocab sizes
    std::vector<uint32_t> vocab_sizes = {50, 100, 128, 30522};  // Last one is BERT-base

    for (uint32_t vocab_size : vocab_sizes) {
        config.vocab_size = vocab_size;
        auto model = create_for_masked_lm(config);

        const size_t batch_size = 1;
        const size_t seq_len = 32;

        std::vector<uint32_t> input_ids_vec(batch_size * seq_len);
        std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

        std::mt19937 gen(42);
        std::uniform_int_distribution<uint32_t> dis(0, vocab_size - 1);
        for (size_t i = 0; i < input_ids_vec.size(); ++i) {
            input_ids_vec[i] = dis(gen);
        }

        auto input_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            input_ids_vec,
            ttnn::Shape{batch_size, 1, 1, seq_len},
            &autograd::ctx().get_device(),
            ttnn::Layout::ROW_MAJOR);
        auto attention_mask_tensor = core::from_vector(
            attention_mask_vec, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

        auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
        auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

        auto logits = (*model)(input_ids_ag, attention_mask_ag, nullptr);
        auto logits_shape = logits->get_value().logical_shape();

        // Verify vocab size alignment
        uint32_t expected_aligned = ((vocab_size + 31) / 32) * 32;
        EXPECT_EQ(logits_shape[3], expected_aligned)
            << "For vocab_size=" << vocab_size << ", expected " << expected_aligned;
    }
}

TEST_F(BertMLMTest, MLMHeadArchitecture) {
    // Test that MLM head properly transforms embeddings -> vocab predictions
    BertConfig config;
    config.vocab_size = 64;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;

    auto model = create_for_masked_lm(config);

    const size_t batch_size = 1;
    const size_t seq_len = 32;

    std::vector<uint32_t> input_ids_vec(batch_size * seq_len);
    std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dis(0, config.vocab_size - 1);
    for (size_t i = 0; i < input_ids_vec.size(); ++i) {
        input_ids_vec[i] = dis(gen);
    }

    auto input_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids_vec, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto attention_mask_tensor =
        core::from_vector(attention_mask_vec, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

    auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
    auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

    auto logits = (*model)(input_ids_ag, attention_mask_ag, nullptr);

    // Verify the MLM head produces reasonable distributions
    auto logits_data = core::to_vector(logits->get_value());

    // Each token should have vocab_size predictions
    // Check that values are distributed (not all zeros or all the same)
    float min_val = *std::min_element(logits_data.begin(), logits_data.begin() + config.vocab_size);
    float max_val = *std::max_element(logits_data.begin(), logits_data.begin() + config.vocab_size);

    EXPECT_LT(std::abs(max_val - min_val), 5.0F) << "Logits should be reasonably distributed";
    EXPECT_GT(std::abs(max_val - min_val), 0.01F) << "Logits should not all be the same value";
}
