// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Isolated test to pinpoint where batch processing fails.
 *
 * Tests each stage of BERT forward pass:
 * 1. Embedding layer output
 * 2. After first transformer block
 * 3. After pooler (CLS token extraction)
 */

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/bert.hpp"

using namespace ttml;
using namespace ttml::models::bert;

class BertBatchIsolationTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }

    void print_first_values(const std::vector<float>& data, size_t count, const std::string& label) {
        std::cout << label << ": [";
        for (size_t i = 0; i < std::min(count, data.size()); ++i) {
            std::cout << data[i];
            if (i < std::min(count, data.size()) - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    bool are_different(const std::vector<float>& vec1, const std::vector<float>& vec2, float threshold = 0.001F) {
        if (vec1.size() != vec2.size())
            return true;

        float max_diff = 0.0F;
        for (size_t i = 0; i < vec1.size(); ++i) {
            float diff = std::abs(vec1[i] - vec2[i]);
            max_diff = std::max(max_diff, diff);
        }

        return max_diff > threshold;
    }
};

TEST_F(BertBatchIsolationTest, EmbeddingLayerBatchHandling) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "TEST: Embedding Layer Batch Handling" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 0;  // NO transformer blocks - just embeddings!
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    auto model = std::make_shared<Bert>(config);

    const size_t batch_size = 2;
    const size_t seq_len = 32;

    // Create batch with VERY different inputs
    // IMPORTANT: Token IDs must be uint32, not float!
    std::vector<uint32_t> input_ids_batch(batch_size * seq_len);
    std::vector<uint32_t> token_type_ids_batch(batch_size * seq_len);
    std::vector<float> attention_mask_batch(batch_size * seq_len);

    // Sample 0: all token ID = 7
    for (size_t i = 0; i < seq_len; ++i) {
        input_ids_batch[i] = 7;
        token_type_ids_batch[i] = 0;
        attention_mask_batch[i] = 1.0F;
    }

    // Sample 1: all token ID = 99 (very different!)
    for (size_t i = seq_len; i < 2 * seq_len; ++i) {
        input_ids_batch[i] = 99;
        token_type_ids_batch[i] = 0;
        attention_mask_batch[i] = 1.0F;
    }

    std::cout << "\nInput:" << std::endl;
    std::cout << "  Sample 0: all tokens = 7" << std::endl;
    std::cout << "  Sample 1: all tokens = 99" << std::endl;

    // Create tensors with correct dtypes
    auto input_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids_batch,
        ttnn::Shape{batch_size, 1, 1, seq_len},
        &autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);
    auto token_type_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        token_type_ids_batch,
        ttnn::Shape{batch_size, 1, 1, seq_len},
        &autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);
    auto attention_mask_tensor =
        core::from_vector(attention_mask_batch, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

    auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
    auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);
    auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

    // Get embeddings directly (bypass transformer blocks)
    auto embeddings = model->get_embeddings(input_ids_ag, token_type_ids_ag);
    auto embeddings_data = core::to_vector(embeddings->get_value());

    std::cout << "\nEmbedding output shape: " << embeddings->get_shape() << std::endl;

    // Extract first 10 values from each sample's first token
    const size_t embedding_dim = 64;
    std::vector<float> sample0_first_token(std::min(size_t(10), embedding_dim));
    std::vector<float> sample1_first_token(std::min(size_t(10), embedding_dim));

    // Assuming layout is [batch, 1, seq, embedding_dim]
    // First token of sample 0: indices 0 to embedding_dim-1
    // First token of sample 1: indices (seq_len * embedding_dim) to (seq_len * embedding_dim + embedding_dim - 1)

    for (size_t i = 0; i < std::min(size_t(10), embedding_dim); ++i) {
        sample0_first_token[i] = embeddings_data[i];
        sample1_first_token[i] = embeddings_data[seq_len * embedding_dim + i];
    }

    print_first_values(sample0_first_token, 10, "Sample 0 first token (first 10 dims)");
    print_first_values(sample1_first_token, 10, "Sample 1 first token (first 10 dims)");

    if (are_different(sample0_first_token, sample1_first_token)) {
        std::cout << "\n✓ EMBEDDING LAYER WORKS: Different inputs produce different embeddings" << std::endl;
    } else {
        std::cout << "\n❌ EMBEDDING LAYER BUG: Different inputs produce IDENTICAL embeddings!" << std::endl;
        FAIL() << "Embedding layer produces identical outputs for different inputs";
    }

    std::cout << std::string(80, '=') << std::endl;
}

TEST_F(BertBatchIsolationTest, PoolerBatchHandling) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "TEST: Pooler (CLS Token Extraction) Batch Handling" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 0;  // NO transformer blocks
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    auto model = std::make_shared<Bert>(config);

    const size_t batch_size = 2;
    const size_t seq_len = 32;

    // Create batch with different inputs
    // IMPORTANT: Token IDs must be uint32, not float!
    std::vector<uint32_t> input_ids_batch(batch_size * seq_len);
    std::vector<uint32_t> token_type_ids_batch(batch_size * seq_len);

    for (size_t i = 0; i < seq_len; ++i) {
        input_ids_batch[i] = 7;
        token_type_ids_batch[i] = 0;
    }
    for (size_t i = seq_len; i < 2 * seq_len; ++i) {
        input_ids_batch[i] = 99;
        token_type_ids_batch[i] = 0;
    }

    auto input_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids_batch,
        ttnn::Shape{batch_size, 1, 1, seq_len},
        &autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);
    auto token_type_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        token_type_ids_batch,
        ttnn::Shape{batch_size, 1, 1, seq_len},
        &autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);

    auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
    auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);

    // Get embeddings
    auto embeddings = model->get_embeddings(input_ids_ag, token_type_ids_ag);

    std::cout << "\nEmbedding shape: " << embeddings->get_shape() << std::endl;

    // Manually extract [CLS] token (first token) using same logic as pooler
    auto hidden_shape = embeddings->get_shape();
    auto batch_size_actual = hidden_shape[0];
    auto embedding_dim = hidden_shape[3];

    std::cout << "Batch size: " << batch_size_actual << std::endl;
    std::cout << "Embedding dim: " << embedding_dim << std::endl;

    ttnn::SmallVector<uint32_t> start_indices = {0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> end_indices = {batch_size_actual, 1, 1, embedding_dim};
    ttnn::SmallVector<uint32_t> stride = {1, 1, 1, 1};

    std::cout << "\nSlice parameters:" << std::endl;
    std::cout << "  start: [0, 0, 0, 0]" << std::endl;
    std::cout << "  end: [" << batch_size_actual << ", 1, 1, " << embedding_dim << "]" << std::endl;
    std::cout << "  stride: [1, 1, 1, 1]" << std::endl;

    auto cls_token = ttnn::slice(embeddings->get_value(), start_indices, end_indices, stride);
    auto cls_data = core::to_vector(cls_token);

    std::cout << "\nCLS token output shape: " << cls_token.logical_shape() << std::endl;
    std::cout << "CLS token output size: " << cls_data.size() << std::endl;

    // Extract first 10 values from each sample
    std::vector<float> sample0_cls(std::min(size_t(10), size_t(embedding_dim)));
    std::vector<float> sample1_cls(std::min(size_t(10), size_t(embedding_dim)));

    for (size_t i = 0; i < std::min(size_t(10), size_t(embedding_dim)); ++i) {
        sample0_cls[i] = cls_data[i];
        sample1_cls[i] = cls_data[size_t(embedding_dim) + i];
    }

    print_first_values(sample0_cls, 10, "Sample 0 CLS (first 10 dims)");
    print_first_values(sample1_cls, 10, "Sample 1 CLS (first 10 dims)");

    if (are_different(sample0_cls, sample1_cls)) {
        std::cout << "\n✓ POOLER WORKS: Different samples have different CLS tokens" << std::endl;
    } else {
        std::cout << "\n❌ POOLER BUG: Different samples have IDENTICAL CLS tokens!" << std::endl;
        FAIL() << "Pooler extracts identical CLS tokens for different samples";
    }

    std::cout << std::string(80, '=') << std::endl;
}
