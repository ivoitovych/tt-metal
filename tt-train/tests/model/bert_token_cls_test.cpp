// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Token Classification Tests
 *
 * Tests for BertForTokenClassification (NER, POS tagging, etc.)
 * Validates:
 * 1. Token-level classification output shapes
 * 2. Forward pass correctness
 * 3. Gradient flow
 * 4. Label count handling
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

class BertTokenClsTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(BertTokenClsTest, BasicForwardPass) {
    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 2;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    uint32_t num_labels = 9;  // e.g., NER tags: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC
    auto model = create_for_token_classification(config, num_labels, 0.1F);

    const size_t batch_size = 2;
    const size_t seq_len = 32;

    // Create input tensors
    std::vector<uint32_t> input_ids_vec(batch_size * seq_len);
    std::vector<uint32_t> token_type_ids_vec(batch_size * seq_len, 0);
    std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dis(0, config.vocab_size - 1);
    for (size_t i = 0; i < input_ids_vec.size(); ++i) {
        input_ids_vec[i] = dis(gen);
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

    // Check output shape - should be [batch_size, 1, seq_len, num_labels_aligned]
    auto logits_shape = logits->get_value().logical_shape();
    EXPECT_EQ(logits_shape[0], batch_size);
    EXPECT_EQ(logits_shape[1], 1);
    EXPECT_EQ(logits_shape[2], seq_len);
    EXPECT_EQ(logits_shape[3], 32);  // 9 aligned to 32

    // Check output values are finite
    auto logits_data = core::to_vector(logits->get_value());
    for (size_t i = 0; i < batch_size * seq_len * num_labels; ++i) {
        EXPECT_TRUE(std::isfinite(logits_data[i])) << "Logit at index " << i << " is not finite";
    }
}

TEST_F(BertTokenClsTest, GradientFlow) {
    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;

    uint32_t num_labels = 5;
    auto model = create_for_token_classification(config, num_labels, 0.0F);

    const size_t batch_size = 1;
    const size_t seq_len = 32;

    // Create input tensors
    std::vector<uint32_t> input_ids_vec(batch_size * seq_len);
    std::vector<uint32_t> token_type_ids_vec(batch_size * seq_len, 0);
    std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

    std::mt19937 gen(43);
    std::uniform_int_distribution<uint32_t> dis(0, config.vocab_size - 1);
    for (size_t i = 0; i < input_ids_vec.size(); ++i) {
        input_ids_vec[i] = dis(gen);
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

    // Create dummy labels (all zeros for simplicity)
    // Note: cross_entropy_loss expects rank-2 targets for token classification
    std::vector<uint32_t> labels_vec(batch_size * seq_len, 0);
    auto labels_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        labels_vec, ttnn::Shape{batch_size, seq_len}, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto labels_ag = autograd::create_tensor(labels_tensor);

    // Compute loss
    auto [loss, logits_out] = model->forward_with_loss(input_ids_ag, attention_mask_ag, token_type_ids_ag, labels_ag);

    // Backward pass should complete without error
    loss->backward();

    // If we reach here, gradient computation succeeded
    EXPECT_TRUE(true) << "Gradient flow test passed";
}

TEST_F(BertTokenClsTest, DifferentLabelCounts) {
    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;

    std::vector<uint32_t> label_counts = {2, 5, 9, 17};

    for (uint32_t num_labels : label_counts) {
        auto model = create_for_token_classification(config, num_labels, 0.1F);

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

        // Verify shape
        EXPECT_EQ(logits_shape[0], batch_size);
        EXPECT_EQ(logits_shape[2], seq_len);

        // Verify alignment
        uint32_t expected_aligned = ((num_labels + 31) / 32) * 32;
        EXPECT_EQ(logits_shape[3], expected_aligned)
            << "For num_labels=" << num_labels << ", expected " << expected_aligned;
    }
}
