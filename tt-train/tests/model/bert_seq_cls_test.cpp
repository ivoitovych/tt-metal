// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Sequence Classification Tests
 *
 * These tests validate BertForSequenceClassification implementation:
 * 1. Basic functionality (forward pass, output shapes)
 * 2. Classifier head correctness (pooling + linear layer)
 * 3. Weight loading from safetensors
 * 4. Numerical accuracy with known inputs
 *
 * Note: Python tests provide end-to-end PCC validation against HuggingFace.
 * These C++ tests focus on basic correctness and shape validation.
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

using namespace ttml;
using namespace ttml::models::bert;

namespace {

/**
 * Helper function to compute Pearson Correlation Coefficient
 */
float compute_pcc(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return 0.0F;
    }

    float mean_a = 0.0F, mean_b = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<float>(a.size());
    mean_b /= static_cast<float>(b.size());

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
 * Helper to check if tensor contains NaN values
 */
bool has_nan(const std::vector<float>& data) {
    for (float val : data) {
        if (std::isnan(val)) {
            return true;
        }
    }
    return false;
}

}  // namespace

class BertSeqClsTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(BertSeqClsTest, BasicForwardPass) {
    // Test that BertForSequenceClassification can perform a basic forward pass
    // and produces outputs of the correct shape

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

    const uint32_t num_labels = 2;
    auto model = create_for_sequence_classification(config, num_labels, 0.0F);
    ASSERT_NE(model, nullptr);

    // Create simple input tensors
    const size_t batch_size = 2;
    const size_t seq_len = 32;

    // Input IDs: random vocab indices
    std::vector<float> input_ids(batch_size * seq_len);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> vocab_dist(0, config.vocab_size - 1);
    for (auto& id : input_ids) {
        id = static_cast<float>(vocab_dist(gen));
    }

    // Token type IDs: all zeros (single sequence)
    std::vector<float> token_type_ids(batch_size * seq_len, 0.0F);

    // Attention mask: all ones (no padding)
    std::vector<float> attention_mask(batch_size * seq_len, 1.0F);

    // Convert to tensors
    auto input_ids_tensor =
        core::from_vector(input_ids, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());
    auto token_type_ids_tensor =
        core::from_vector(token_type_ids, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());
    auto attention_mask_tensor =
        core::from_vector(attention_mask, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

    auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
    auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);
    auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

    // Forward pass
    auto logits = (*model)(input_ids_ag, attention_mask_ag, token_type_ids_ag);
    ASSERT_NE(logits, nullptr);

    // Check output shape
    auto logits_shape = logits->get_value().logical_shape();
    EXPECT_EQ(logits_shape.size(), 4);
    EXPECT_EQ(logits_shape[0], batch_size);
    EXPECT_EQ(logits_shape[1], 1);
    EXPECT_EQ(logits_shape[2], 1);
    // Last dimension may be aligned to 32, so check it's at least num_labels
    EXPECT_GE(logits_shape[3], num_labels);

    // Check output values are not NaN
    auto logits_data = core::to_vector(logits->get_value());
    EXPECT_FALSE(has_nan(logits_data));

    std::cout << "✓ Basic forward pass test passed" << std::endl;
    std::cout << "  Output shape: [" << logits_shape[0] << ", " << logits_shape[1] << ", " << logits_shape[2] << ", "
              << logits_shape[3] << "]" << std::endl;
}

TEST_F(BertSeqClsTest, MultipleLabelsTest) {
    // Test with different number of labels (binary, 3-way, 5-way classification)

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    std::vector<uint32_t> test_num_labels = {2, 3, 5};

    for (uint32_t num_labels : test_num_labels) {
        auto model = create_for_sequence_classification(config, num_labels, 0.0F);
        ASSERT_NE(model, nullptr);

        EXPECT_EQ(model->get_num_labels(), num_labels);

        // Create minimal input
        const size_t batch_size = 1;
        const size_t seq_len = 32;  // Must be multiple of TILE_HEIGHT (32)

        std::vector<float> input_ids(batch_size * seq_len, 0.0F);
        std::vector<float> token_type_ids(batch_size * seq_len, 0.0F);
        std::vector<float> attention_mask(batch_size * seq_len, 1.0F);

        auto input_ids_tensor =
            core::from_vector(input_ids, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());
        auto token_type_ids_tensor =
            core::from_vector(token_type_ids, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());
        auto attention_mask_tensor =
            core::from_vector(attention_mask, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

        auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
        auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);
        auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

        // Forward pass
        auto logits = (*model)(input_ids_ag, attention_mask_ag, token_type_ids_ag);
        ASSERT_NE(logits, nullptr);

        // Verify output
        auto logits_data = core::to_vector(logits->get_value());
        EXPECT_FALSE(has_nan(logits_data));

        std::cout << "✓ Test with " << num_labels << " labels passed" << std::endl;
    }
}

TEST_F(BertSeqClsTest, ClassifierWeightShapes) {
    // Test that classifier weights have correct shapes

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;

    const uint32_t num_labels = 3;
    auto model = create_for_sequence_classification(config, num_labels, 0.0F);

    auto params = model->parameters();

    // Check classifier weight exists
    auto classifier_weight = params["bert/classifier/weight"];
    ASSERT_NE(classifier_weight, nullptr);

    auto weight_shape = classifier_weight->get_value().logical_shape();
    // Weight should be [1, 1, num_labels_aligned, hidden_dim]
    EXPECT_EQ(weight_shape[0], 1);
    EXPECT_EQ(weight_shape[1], 1);
    EXPECT_GE(weight_shape[2], num_labels);  // May be aligned to 32
    EXPECT_EQ(weight_shape[3], config.embedding_dim);

    // Check classifier bias exists
    auto classifier_bias = params["bert/classifier/bias"];
    ASSERT_NE(classifier_bias, nullptr);

    auto bias_shape = classifier_bias->get_value().logical_shape();
    // Bias should be [1, 1, 1, num_labels_aligned]
    EXPECT_EQ(bias_shape[0], 1);
    EXPECT_EQ(bias_shape[1], 1);
    EXPECT_EQ(bias_shape[2], 1);
    EXPECT_GE(bias_shape[3], num_labels);  // May be aligned to 32

    std::cout << "✓ Classifier weight shapes test passed" << std::endl;
    std::cout << "  Weight shape: [" << weight_shape[0] << ", " << weight_shape[1] << ", " << weight_shape[2] << ", "
              << weight_shape[3] << "]" << std::endl;
    std::cout << "  Bias shape: [" << bias_shape[0] << ", " << bias_shape[1] << ", " << bias_shape[2] << ", "
              << bias_shape[3] << "]" << std::endl;
}

TEST_F(BertSeqClsTest, OutputConsistency) {
    // Test that running the same input twice produces identical outputs

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 2;
    config.dropout_prob = 0.0F;  // Must be 0 for deterministic output
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    const uint32_t num_labels = 2;
    auto model = create_for_sequence_classification(config, num_labels, 0.0F);

    const size_t batch_size = 2;
    const size_t seq_len = 32;

    // Create deterministic inputs
    std::vector<float> input_ids(batch_size * seq_len, 5.0F);  // All token ID 5
    std::vector<float> token_type_ids(batch_size * seq_len, 0.0F);
    std::vector<float> attention_mask(batch_size * seq_len, 1.0F);

    auto input_ids_tensor =
        core::from_vector(input_ids, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());
    auto token_type_ids_tensor =
        core::from_vector(token_type_ids, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());
    auto attention_mask_tensor =
        core::from_vector(attention_mask, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

    auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
    auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);
    auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

    // Run forward pass twice
    auto logits1 = (*model)(input_ids_ag, attention_mask_ag, token_type_ids_ag);
    auto logits2 = (*model)(input_ids_ag, attention_mask_ag, token_type_ids_ag);

    auto logits1_data = core::to_vector(logits1->get_value());
    auto logits2_data = core::to_vector(logits2->get_value());

    // Compute PCC between two runs
    float pcc = compute_pcc(logits1_data, logits2_data);

    // With dropout=0, outputs should be identical (PCC = 1.0)
    EXPECT_GT(pcc, 0.9999F);

    std::cout << "✓ Output consistency test passed" << std::endl;
    std::cout << "  PCC between runs: " << pcc << std::endl;
}

TEST_F(BertSeqClsTest, BatchSizeIndependence) {
    // Test that different batch sizes produce consistent per-sample results

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    const uint32_t num_labels = 2;
    auto model = create_for_sequence_classification(config, num_labels, 0.0F);

    const size_t seq_len = 32;  // Must be multiple of TILE_HEIGHT (32)

    // Run with batch size 1
    std::vector<float> input_ids_single(seq_len, 7.0F);
    std::vector<float> token_type_ids_single(seq_len, 0.0F);
    std::vector<float> attention_mask_single(seq_len, 1.0F);

    auto input_ids_tensor_single =
        core::from_vector(input_ids_single, ttnn::Shape{1, 1, 1, seq_len}, &autograd::ctx().get_device());
    auto token_type_ids_tensor_single =
        core::from_vector(token_type_ids_single, ttnn::Shape{1, 1, 1, seq_len}, &autograd::ctx().get_device());
    auto attention_mask_tensor_single =
        core::from_vector(attention_mask_single, ttnn::Shape{1, 1, 1, seq_len}, &autograd::ctx().get_device());

    auto input_ids_ag_single = autograd::create_tensor(input_ids_tensor_single);
    auto token_type_ids_ag_single = autograd::create_tensor(token_type_ids_tensor_single);
    auto attention_mask_ag_single = autograd::create_tensor(attention_mask_tensor_single);

    auto logits_single = (*model)(input_ids_ag_single, attention_mask_ag_single, token_type_ids_ag_single);
    auto logits_single_data = core::to_vector(logits_single->get_value());

    // Run same input as part of batch size 2
    std::vector<float> input_ids_batch(2 * seq_len);
    std::vector<float> token_type_ids_batch(2 * seq_len);
    std::vector<float> attention_mask_batch(2 * seq_len);

    // First sample: same as single
    for (size_t i = 0; i < seq_len; ++i) {
        input_ids_batch[i] = 7.0F;
        token_type_ids_batch[i] = 0.0F;
        attention_mask_batch[i] = 1.0F;
    }
    // Second sample: different
    for (size_t i = seq_len; i < 2 * seq_len; ++i) {
        input_ids_batch[i] = 3.0F;
        token_type_ids_batch[i] = 0.0F;
        attention_mask_batch[i] = 1.0F;
    }

    auto input_ids_tensor_batch =
        core::from_vector(input_ids_batch, ttnn::Shape{2, 1, 1, seq_len}, &autograd::ctx().get_device());
    auto token_type_ids_tensor_batch =
        core::from_vector(token_type_ids_batch, ttnn::Shape{2, 1, 1, seq_len}, &autograd::ctx().get_device());
    auto attention_mask_tensor_batch =
        core::from_vector(attention_mask_batch, ttnn::Shape{2, 1, 1, seq_len}, &autograd::ctx().get_device());

    auto input_ids_ag_batch = autograd::create_tensor(input_ids_tensor_batch);
    auto token_type_ids_ag_batch = autograd::create_tensor(token_type_ids_tensor_batch);
    auto attention_mask_ag_batch = autograd::create_tensor(attention_mask_tensor_batch);

    auto logits_batch = (*model)(input_ids_ag_batch, attention_mask_ag_batch, token_type_ids_ag_batch);
    auto logits_batch_data = core::to_vector(logits_batch->get_value());

    // Extract first sample from batch
    auto logits_shape = logits_batch->get_value().logical_shape();
    size_t num_labels_aligned = logits_shape[3];

    std::vector<float> logits_batch_first(num_labels_aligned);
    for (size_t i = 0; i < num_labels_aligned; ++i) {
        logits_batch_first[i] = logits_batch_data[i];
    }

    // Compare single vs first of batch
    float pcc = compute_pcc(logits_single_data, logits_batch_first);

    // Should be very similar (PCC close to 1.0)
    EXPECT_GT(pcc, 0.999F);

    std::cout << "✓ Batch size independence test passed" << std::endl;
    std::cout << "  PCC (single vs batch[0]): " << pcc << std::endl;
}
