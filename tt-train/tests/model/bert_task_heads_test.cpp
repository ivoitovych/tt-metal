// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Task Heads Unit Tests
 *
 * Tests BERT task head modules and task models in isolation.
 * This validates:
 * 1. Head module creation and forward passes
 * 2. Task model creation and composition
 * 3. Output shapes for all task types
 * 4. Loss computation integration
 * 5. SafeTensors weight loading
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/bert.hpp"
#include "models/bert_tasks.hpp"
#include "modules/bert_heads.hpp"
#include "ops/bert_losses.hpp"

using namespace ttml;

namespace {

/**
 * Helper function to compute Pearson Correlation Coefficient
 */
[[maybe_unused]] static float compute_pcc(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return 0.0F;
    }

    float mean_a = 0.0F, mean_b = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<float>(a.size());
    mean_b /= static_cast<float>(a.size());

    float numerator = 0.0F;
    float denom_a = 0.0F;
    float denom_b = 0.0F;

    for (size_t i = 0; i < a.size(); ++i) {
        float diff_a = a[i] - mean_a;
        float diff_b = b[i] - mean_b;
        numerator += diff_a * diff_b;
        denom_a += diff_a * diff_a;
        denom_b += diff_b * diff_b;
    }

    float denominator = std::sqrt(denom_a * denom_b);
    return denominator > 0.0F ? numerator / denominator : 0.0F;
}

/**
 * Helper to create random tensor data
 */
std::vector<float> create_random_data(size_t size, float mean = 0.0F, float stddev = 1.0F, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

/**
 * Small BERT config for fast testing
 */
models::bert::BertConfig create_small_config() {
    models::bert::BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 32;
    config.embedding_dim = 128;
    config.intermediate_size = 512;
    config.num_heads = 4;
    config.num_blocks = 2;
    config.dropout_prob = 0.0F;  // Deterministic
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;
    config.type_vocab_size = 2;
    return config;
}

}  // namespace

// ============================================================================
// Head Module Tests
// ============================================================================

class BertHeadsTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
        autograd::ctx().close_device();
    }
};

TEST_F(BertHeadsTest, SequenceClassificationHeadCreation) {
    const uint32_t hidden_size = 128;
    const uint32_t num_labels = 3;
    const float dropout_prob = 0.1F;

    auto head = std::make_shared<modules::BertSequenceClassificationHead>(hidden_size, num_labels, dropout_prob);

    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->get_num_labels(), num_labels);
}

TEST_F(BertHeadsTest, SequenceClassificationHeadForward) {
    const uint32_t batch_size = 2;
    const uint32_t hidden_size = 128;
    const uint32_t num_labels = 3;

    auto head = std::make_shared<modules::BertSequenceClassificationHead>(hidden_size, num_labels, 0.0F);

    // Create pooled input [B, 1, 1, E]
    auto pooled_data = create_random_data(batch_size * 1 * 1 * hidden_size);
    auto pooled_input =
        core::from_vector(pooled_data, ttnn::Shape({batch_size, 1, 1, hidden_size}), &autograd::ctx().get_device());
    auto pooled_tensor = autograd::create_tensor(pooled_input);

    // Forward pass
    auto logits = (*head)(pooled_tensor);

    // Check output shape [B, 1, 1, num_labels]
    auto output_shape = logits->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], 1);
    EXPECT_EQ(output_shape[3], num_labels);
}

TEST_F(BertHeadsTest, TokenClassificationHeadForward) {
    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned
    const uint32_t hidden_size = 128;
    const uint32_t num_labels = 9;  // BIO-NER tags

    auto head = std::make_shared<modules::BertTokenClassificationHead>(hidden_size, num_labels, 0.0F);

    // Create sequence input [B, 1, S, E]
    auto seq_data = create_random_data(batch_size * 1 * seq_len * hidden_size);
    auto seq_input =
        core::from_vector(seq_data, ttnn::Shape({batch_size, 1, seq_len, hidden_size}), &autograd::ctx().get_device());
    auto seq_tensor = autograd::create_tensor(seq_input);

    // Forward pass
    auto logits = (*head)(seq_tensor);

    // Check output shape [B, 1, S, num_labels]
    auto output_shape = logits->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], num_labels);
}

TEST_F(BertHeadsTest, QuestionAnsweringHeadForward) {
    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned
    const uint32_t hidden_size = 128;

    auto head = std::make_shared<modules::BertQuestionAnsweringHead>(hidden_size);

    // Create sequence input [B, 1, S, E]
    auto seq_data = create_random_data(batch_size * 1 * seq_len * hidden_size);
    auto seq_input =
        core::from_vector(seq_data, ttnn::Shape({batch_size, 1, seq_len, hidden_size}), &autograd::ctx().get_device());
    auto seq_tensor = autograd::create_tensor(seq_input);

    // Forward pass
    auto combined_logits = (*head)(seq_tensor);

    // Check output shape [B, 1, S, 2]
    auto output_shape = combined_logits->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], 2);

    // Test split utility
    auto qa_logits = modules::BertQuestionAnsweringHead::split_logits(combined_logits);
    ASSERT_NE(qa_logits.start_logits, nullptr);
    ASSERT_NE(qa_logits.end_logits, nullptr);

    auto start_shape = qa_logits.start_logits->get_value().logical_shape();
    EXPECT_EQ(start_shape[0], batch_size);
    EXPECT_EQ(start_shape[2], seq_len);
    EXPECT_EQ(start_shape[3], 1);
}

TEST_F(BertHeadsTest, MaskedLMHeadForward) {
    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned
    const uint32_t hidden_size = 128;
    const uint32_t vocab_size = 1000;

    auto head = std::make_shared<modules::BertMaskedLMHead>(hidden_size, vocab_size);

    // Create sequence input [B, 1, S, E]
    auto seq_data = create_random_data(batch_size * 1 * seq_len * hidden_size);
    auto seq_input =
        core::from_vector(seq_data, ttnn::Shape({batch_size, 1, seq_len, hidden_size}), &autograd::ctx().get_device());
    auto seq_tensor = autograd::create_tensor(seq_input);

    // Forward pass
    auto logits = (*head)(seq_tensor);

    // Check output shape [B, 1, S, vocab_size]
    auto output_shape = logits->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], vocab_size);

    // Test weight tying
    EXPECT_FALSE(head->has_tied_weights());
    // TODO: Test actual weight tying when embeddings are available
}

TEST_F(BertHeadsTest, NSPHeadForward) {
    const uint32_t batch_size = 2;
    const uint32_t hidden_size = 128;

    auto head = std::make_shared<modules::BertNSPHead>(hidden_size);

    // Create pooled input [B, 1, 1, E]
    auto pooled_data = create_random_data(batch_size * 1 * 1 * hidden_size);
    auto pooled_input =
        core::from_vector(pooled_data, ttnn::Shape({batch_size, 1, 1, hidden_size}), &autograd::ctx().get_device());
    auto pooled_tensor = autograd::create_tensor(pooled_input);

    // Forward pass
    auto logits = (*head)(pooled_tensor);

    // Check output shape [B, 1, 1, 2]
    auto output_shape = logits->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], 1);
    EXPECT_EQ(output_shape[3], 2);
}

// ============================================================================
// Task Model Tests
// ============================================================================

class BertTaskModelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
        m_config = create_small_config();
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
        autograd::ctx().close_device();
    }

    models::bert::BertConfig m_config;
};

TEST_F(BertTaskModelsTest, SequenceClassificationCreation) {
    models::bert::SequenceClassificationConfig task_config;
    task_config.bert_config = m_config;
    task_config.num_labels = 3;
    task_config.classifier_dropout = 0.1F;

    auto model = models::bert::create_for_sequence_classification(task_config);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->get_num_labels(), 3);
}

TEST_F(BertTaskModelsTest, SequenceClassificationForward) {
    models::bert::SequenceClassificationConfig task_config;
    task_config.bert_config = m_config;
    task_config.num_labels = 2;

    auto model = models::bert::create_for_sequence_classification(task_config);

    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned

    // Create input tensors
    std::vector<uint32_t> input_ids(batch_size * seq_len, 100);
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());
    auto input_ptr = autograd::create_tensor(input_tensor);

    // Forward pass
    auto logits = (*model)(input_ptr, nullptr, nullptr);

    // Check output shape [B, 1, 1, 2]
    auto output_shape = logits->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[3], 2);
}

TEST_F(BertTaskModelsTest, TokenClassificationForward) {
    models::bert::TokenClassificationConfig task_config;
    task_config.bert_config = m_config;
    task_config.num_labels = 9;

    auto model = models::bert::create_for_token_classification(task_config);

    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned

    std::vector<uint32_t> input_ids(batch_size * seq_len, 100);
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());
    auto input_ptr = autograd::create_tensor(input_tensor);

    auto logits = (*model)(input_ptr, nullptr, nullptr);

    // Check output shape [B, 1, S, num_labels]
    auto output_shape = logits->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], 9);
}

TEST_F(BertTaskModelsTest, QuestionAnsweringForward) {
    models::bert::QuestionAnsweringConfig task_config;
    task_config.bert_config = m_config;

    auto model = models::bert::create_for_question_answering(task_config);

    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned

    std::vector<uint32_t> input_ids(batch_size * seq_len, 100);
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());
    auto input_ptr = autograd::create_tensor(input_tensor);

    auto logits = (*model)(input_ptr, nullptr, nullptr);

    // Check output shape [B, 1, S, 2]
    auto output_shape = logits->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], 2);
}

TEST_F(BertTaskModelsTest, MaskedLMForward) {
    models::bert::MaskedLMConfig task_config;
    task_config.bert_config = m_config;
    task_config.tie_word_embeddings = false;  // Don't tie for unit tests (weights not initialized)

    auto model = models::bert::create_for_masked_lm(task_config);

    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned

    std::vector<uint32_t> input_ids(batch_size * seq_len, 100);
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());
    auto input_ptr = autograd::create_tensor(input_tensor);

    auto logits = (*model)(input_ptr, nullptr, nullptr);

    // Check output shape [B, 1, S, vocab_size]
    auto output_shape = logits->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[2], seq_len);
    EXPECT_EQ(output_shape[3], m_config.vocab_size);
}

TEST_F(BertTaskModelsTest, PreTrainingForward) {
    models::bert::PreTrainingConfig task_config;
    task_config.bert_config = m_config;
    task_config.tie_word_embeddings = false;  // Don't tie for unit tests (weights not initialized)

    auto model = models::bert::create_for_pretraining(task_config);

    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned

    std::vector<uint32_t> input_ids(batch_size * seq_len, 100);
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());
    auto input_ptr = autograd::create_tensor(input_tensor);

    // Test forward_pretraining - returns both MLM and NSP
    auto output = model->forward_pretraining(input_ptr, nullptr, nullptr);

    ASSERT_NE(output.mlm_logits, nullptr);
    ASSERT_NE(output.nsp_logits, nullptr);

    // Check MLM output shape [B, 1, S, vocab_size]
    auto mlm_shape = output.mlm_logits->get_value().logical_shape();
    EXPECT_EQ(mlm_shape[0], batch_size);
    EXPECT_EQ(mlm_shape[2], seq_len);
    EXPECT_EQ(mlm_shape[3], m_config.vocab_size);

    // Check NSP output shape [B, 1, 1, 2]
    auto nsp_shape = output.nsp_logits->get_value().logical_shape();
    EXPECT_EQ(nsp_shape[0], batch_size);
    EXPECT_EQ(nsp_shape[1], 1);
    EXPECT_EQ(nsp_shape[2], 1);
    EXPECT_EQ(nsp_shape[3], 2);
}

// ============================================================================
// Loss Function Tests
// ============================================================================

class BertLossesTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
        autograd::ctx().close_device();
    }
};

TEST_F(BertLossesTest, SequenceClassificationLoss) {
    const uint32_t batch_size = 4;
    const uint32_t num_labels = 3;

    // Create dummy logits [B, 1, 1, num_labels]
    auto logits_data = create_random_data(batch_size * num_labels);
    auto logits_tensor =
        core::from_vector(logits_data, ttnn::Shape({batch_size, 1, 1, num_labels}), &autograd::ctx().get_device());
    auto logits = autograd::create_tensor(logits_tensor);

    // Create labels [B, 1] with ROW_MAJOR layout
    std::vector<uint32_t> labels_data = {0, 1, 2, 0};
    auto labels_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        labels_data, ttnn::Shape({batch_size, 1}), &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto labels = autograd::create_tensor(labels_tensor);

    // Compute loss
    auto loss = ops::bert_losses::compute_sequence_classification_loss(logits, labels);

    ASSERT_NE(loss, nullptr);
    // Loss should be scalar [1, 1, 1, 1]
    auto loss_shape = loss->get_value().logical_shape();
    EXPECT_EQ(loss_shape[0], 1);
    EXPECT_EQ(loss_shape[1], 1);
    EXPECT_EQ(loss_shape[2], 1);
    EXPECT_EQ(loss_shape[3], 1);
}

TEST_F(BertLossesTest, PreTrainingCombinedLoss) {
    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned
    const uint32_t vocab_size = 1000;

    // Create dummy MLM logits [B, 1, S, vocab_size]
    auto mlm_data = create_random_data(batch_size * seq_len * vocab_size);
    auto mlm_tensor =
        core::from_vector(mlm_data, ttnn::Shape({batch_size, 1, seq_len, vocab_size}), &autograd::ctx().get_device());
    auto mlm_logits = autograd::create_tensor(mlm_tensor);

    // Create dummy NSP logits [B, 1, 1, 2]
    auto nsp_data = create_random_data(batch_size * 2);
    auto nsp_tensor = core::from_vector(nsp_data, ttnn::Shape({batch_size, 1, 1, 2}), &autograd::ctx().get_device());
    auto nsp_logits = autograd::create_tensor(nsp_tensor);

    // Create labels with ROW_MAJOR layout (use UINT32 as required by cross_entropy)
    std::vector<uint32_t> mlm_labels_data(batch_size * seq_len, 0);  // Use 0 for masked tokens
    mlm_labels_data[0] = 100;                                        // One unmasked token
    auto mlm_labels_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        mlm_labels_data, ttnn::Shape({batch_size, seq_len}), &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto mlm_labels = autograd::create_tensor(mlm_labels_tensor);

    std::vector<uint32_t> nsp_labels_data = {0, 1};
    auto nsp_labels_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        nsp_labels_data, ttnn::Shape({batch_size, 1}), &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto nsp_labels = autograd::create_tensor(nsp_labels_tensor);

    // Compute combined loss
    auto loss = ops::bert_losses::compute_pretraining_loss(mlm_logits, nsp_logits, mlm_labels, nsp_labels);

    ASSERT_NE(loss, nullptr);
    // Loss should be scalar [1, 1, 1, 1]
    auto loss_shape = loss->get_value().logical_shape();
    EXPECT_EQ(loss_shape[0], 1);
    EXPECT_EQ(loss_shape[1], 1);
    EXPECT_EQ(loss_shape[2], 1);
    EXPECT_EQ(loss_shape[3], 1);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(BertTaskModelsTest, EndToEndSequenceClassification) {
    // Create model
    models::bert::SequenceClassificationConfig task_config;
    task_config.bert_config = m_config;
    task_config.num_labels = 2;

    auto model = models::bert::create_for_sequence_classification(task_config);

    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;  // Must be tile-aligned

    // Create inputs
    std::vector<uint32_t> input_ids(batch_size * seq_len, 100);
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids, ttnn::Shape({batch_size, 1, 1, seq_len}), &autograd::ctx().get_device());
    auto input_ptr = autograd::create_tensor(input_tensor);

    // Forward pass
    auto logits = (*model)(input_ptr, nullptr, nullptr);

    // Create labels [B, 1] with ROW_MAJOR layout
    std::vector<uint32_t> labels_data = {0, 1};
    auto labels_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        labels_data, ttnn::Shape({batch_size, 1}), &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto labels = autograd::create_tensor(labels_tensor);

    // Compute loss
    auto loss = ops::bert_losses::compute_sequence_classification_loss(logits, labels);

    ASSERT_NE(loss, nullptr);

    // Backward pass
    loss->backward();

    // Check gradients exist
    EXPECT_TRUE(core::is_tensor_initialized(logits->get_grad()));
}

// TODO: Add SafeTensors loading tests when HuggingFace models are available
// TODO: Add numerical accuracy tests comparing with reference implementations
// TODO: Add weight tying verification tests
// TODO: Add gradient flow tests for all task models
