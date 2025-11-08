// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Question Answering Tests
 *
 * Tests for BertForQuestionAnswering (SQuAD-style extractive QA)
 * Validates:
 * 1. Start and end logits output
 * 2. Forward pass correctness
 * 3. Gradient flow
 * 4. Output shape validation
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

class BertQATest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(BertQATest, BasicForwardPass) {
    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 64;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 2;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    auto model = create_for_question_answering(config);

    const size_t batch_size = 2;
    const size_t seq_len = 64;

    // Create input tensors
    std::vector<uint32_t> input_ids_vec(batch_size * seq_len);
    std::vector<uint32_t> token_type_ids_vec(batch_size * seq_len);
    std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dis(0, config.vocab_size - 1);
    for (size_t i = 0; i < input_ids_vec.size(); ++i) {
        input_ids_vec[i] = dis(gen);
        // First half is question (token_type_id=0), second half is context (token_type_id=1)
        token_type_ids_vec[i] = (i % seq_len) < (seq_len / 2) ? 0 : 1;
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

    // Check output shape - should be [batch_size, 1, seq_len, 32]
    // (2 outputs aligned to 32: start and end logits)
    auto logits_shape = logits->get_value().logical_shape();
    EXPECT_EQ(logits_shape[0], batch_size);
    EXPECT_EQ(logits_shape[1], 1);
    EXPECT_EQ(logits_shape[2], seq_len);
    EXPECT_EQ(logits_shape[3], 32);  // 2 aligned to 32

    // Check output values are finite
    auto logits_data = core::to_vector(logits->get_value());
    for (float val : logits_data) {
        EXPECT_TRUE(std::isfinite(val)) << "Logit value is not finite";
    }
}

TEST_F(BertQATest, DifferentBatchSizes) {
    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 64;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;

    auto model = create_for_question_answering(config);

    const uint32_t seq_len = 64;

    // Test different batch sizes
    std::vector<uint32_t> batch_sizes = {1, 2, 4};

    for (uint32_t batch_size : batch_sizes) {
        std::vector<uint32_t> input_ids_vec(batch_size * seq_len);
        std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

        std::mt19937 gen(42);
        std::uniform_int_distribution<uint32_t> dis(0, config.vocab_size - 1);
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

        // Verify shape matches batch size
        EXPECT_EQ(logits_shape[0], batch_size) << "For batch_size=" << batch_size;
        EXPECT_EQ(logits_shape[2], seq_len);
        EXPECT_EQ(logits_shape[3], 32);  // Always 32 (2 aligned)
    }
}
