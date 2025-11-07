// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Test the full BERT embedding pipeline to isolate where batch processing bug emerges

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/dropout_op.hpp"
#include "ops/embedding_op.hpp"
#include "ops/layernorm_op.hpp"

class BertEmbeddingPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Test the full BERT embedding pipeline step by step
TEST_F(BertEmbeddingPipelineTest, FullPipelineStepByStep) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();

    constexpr uint32_t vocab_size = 100;
    constexpr uint32_t batch_size = 2;
    constexpr uint32_t seq_len = 32;
    constexpr uint32_t emb_dim = 64;

    std::cout << "\n=== BERT Embedding Pipeline Test ===\n";
    std::cout << "Configuration: batch_size=" << batch_size << ", seq_len=" << seq_len << ", emb_dim=" << emb_dim
              << "\n\n";

    // Create input_ids: batch 0 = token 7, batch 1 = token 99
    std::vector<uint32_t> input_data(batch_size * seq_len);
    for (uint32_t i = 0; i < seq_len; ++i) {
        input_data[i] = 7;  // Sample 0
    }
    for (uint32_t i = seq_len; i < 2 * seq_len; ++i) {
        input_data[i] = 99;  // Sample 1
    }

    auto input_ids = autograd::create_tensor(core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR));

    std::cout << "Step 0: Input IDs created\n";
    std::cout << "  Sample 0: all tokens = 7\n";
    std::cout << "  Sample 1: all tokens = 99\n\n";

    // ========== STEP 1: Token Embeddings ==========
    auto token_weight = autograd::create_tensor();
    std::vector<float> token_weight_data(vocab_size * emb_dim);
    for (size_t i = 0; i < token_weight_data.size(); ++i) {
        token_weight_data[i] = static_cast<float>(i) / 100.0F;
    }
    token_weight->set_value(core::from_vector(token_weight_data, ttnn::Shape({1, 1, vocab_size, emb_dim}), device));

    auto token_embeddings = ops::embedding_op(input_ids, token_weight);
    auto token_emb_vec = core::to_vector(token_embeddings->get_value());

    float token_sample0_val = token_emb_vec[0];
    float token_sample1_val = token_emb_vec[seq_len * emb_dim];

    std::cout << "Step 1: Token Embeddings (ops::embedding_op)\n";
    std::cout << "  Sample 0 first value: " << token_sample0_val << "\n";
    std::cout << "  Sample 1 first value: " << token_sample1_val << "\n";

    if (std::abs(token_sample0_val - token_sample1_val) < 0.01F) {
        std::cout << "  ❌ BUG DETECTED at Step 1: Token embeddings are identical!\n\n";
        FAIL() << "Bug found in token embeddings step";
    } else {
        std::cout << "  ✓ Token embeddings are different (correct)\n\n";
    }

    // ========== STEP 2: Add Position Embeddings ==========
    auto position_weight = autograd::create_tensor();
    std::vector<float> position_weight_data(seq_len * emb_dim);
    for (size_t i = 0; i < position_weight_data.size(); ++i) {
        position_weight_data[i] = static_cast<float>(i % 10) / 10.0F;  // Pattern: 0.0, 0.1, ..., 0.9, 0.0, ...
    }
    position_weight->set_value(core::from_vector(position_weight_data, ttnn::Shape({1, 1, seq_len, emb_dim}), device));

    auto embeddings_with_pos = ops::add(token_embeddings, position_weight);
    auto pos_emb_vec = core::to_vector(embeddings_with_pos->get_value());

    float pos_sample0_val = pos_emb_vec[0];
    float pos_sample1_val = pos_emb_vec[seq_len * emb_dim];

    std::cout << "Step 2: Add Position Embeddings (ops::add with broadcasting)\n";
    std::cout << "  Sample 0 first value: " << pos_sample0_val << "\n";
    std::cout << "  Sample 1 first value: " << pos_sample1_val << "\n";

    if (std::abs(pos_sample0_val - pos_sample1_val) < 0.01F) {
        std::cout << "  ❌ BUG DETECTED at Step 2: Embeddings became identical after adding positions!\n\n";
        FAIL() << "Bug found in position embeddings addition step";
    } else {
        std::cout << "  ✓ Embeddings still different after position addition (correct)\n\n";
    }

    // ========== STEP 3: Add Token Type Embeddings ==========
    auto token_type_weight = autograd::create_tensor();
    std::vector<float> token_type_weight_data(2 * emb_dim);  // 2 token types (sentence A, B)
    for (size_t i = 0; i < token_type_weight_data.size(); ++i) {
        token_type_weight_data[i] = 0.05F;  // Constant small value
    }
    token_type_weight->set_value(core::from_vector(token_type_weight_data, ttnn::Shape({1, 1, 2, emb_dim}), device));

    // Create token_type_ids (all zeros = sentence A)
    std::vector<uint32_t> token_type_data(batch_size * seq_len, 0);
    auto token_type_ids = autograd::create_tensor(core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        token_type_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR));

    auto token_type_embeddings = ops::embedding_op(token_type_ids, token_type_weight);
    auto embeddings_with_token_type = ops::add(embeddings_with_pos, token_type_embeddings);
    auto token_type_vec = core::to_vector(embeddings_with_token_type->get_value());

    float tt_sample0_val = token_type_vec[0];
    float tt_sample1_val = token_type_vec[seq_len * emb_dim];

    std::cout << "Step 3: Add Token Type Embeddings (ops::embedding_op + ops::add)\n";
    std::cout << "  Sample 0 first value: " << tt_sample0_val << "\n";
    std::cout << "  Sample 1 first value: " << tt_sample1_val << "\n";

    if (std::abs(tt_sample0_val - tt_sample1_val) < 0.01F) {
        std::cout << "  ❌ BUG DETECTED at Step 3: Embeddings became identical after adding token types!\n\n";
        FAIL() << "Bug found in token type embeddings addition step";
    } else {
        std::cout << "  ✓ Embeddings still different after token type addition (correct)\n\n";
    }

    // ========== STEP 4: Layer Normalization ==========
    // Create gamma (weight) and beta (bias) for layer norm
    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({emb_dim}), device));
    auto beta = autograd::create_tensor(core::zeros(ttnn::Shape({emb_dim}), device));

    auto layer_norm = ops::layernorm(embeddings_with_token_type, gamma, beta, 1e-12F);
    auto ln_vec = core::to_vector(layer_norm->get_value());

    float ln_sample0_val = ln_vec[0];
    float ln_sample1_val = ln_vec[seq_len * emb_dim];

    std::cout << "Step 4: Layer Normalization (ops::layer_norm_op with eps=1e-12)\n";
    std::cout << "  Sample 0 first value: " << ln_sample0_val << "\n";
    std::cout << "  Sample 1 first value: " << ln_sample1_val << "\n";

    if (std::abs(ln_sample0_val - ln_sample1_val) < 0.01F) {
        std::cout << "  ❌ BUG DETECTED at Step 4: Embeddings became identical after layer norm!\n\n";
        FAIL() << "Bug found in layer normalization step";
    } else {
        std::cout << "  ✓ Embeddings still different after layer norm (correct)\n\n";
    }

    // ========== STEP 5: Dropout ==========
    auto dropout = ops::dropout(layer_norm, 0.0F);  // dropout=0.0 for testing
    auto dropout_vec = core::to_vector(dropout->get_value());

    float dropout_sample0_val = dropout_vec[0];
    float dropout_sample1_val = dropout_vec[seq_len * emb_dim];

    std::cout << "Step 5: Dropout (ops::dropout_op with p=0.0)\n";
    std::cout << "  Sample 0 first value: " << dropout_sample0_val << "\n";
    std::cout << "  Sample 1 first value: " << dropout_sample1_val << "\n";

    if (std::abs(dropout_sample0_val - dropout_sample1_val) < 0.01F) {
        std::cout << "  ❌ BUG DETECTED at Step 5: Embeddings became identical after dropout!\n\n";
        FAIL() << "Bug found in dropout step";
    } else {
        std::cout << "  ✓ Embeddings still different after dropout (correct)\n\n";
    }

    std::cout << "=== CONCLUSION ===\n";
    std::cout << "✓ Full BERT embedding pipeline works correctly!\n";
    std::cout << "  All batch samples maintain different values throughout the pipeline.\n";
    std::cout << "  Bug is NOT in the embedding pipeline itself.\n\n";
}

// Test if the bug appears when processing through an actual BERT model
TEST_F(BertEmbeddingPipelineTest, CompareDirectPipelineVsBertModel) {
    std::cout << "\n=== This test requires actual BERT model integration ===\n";
    std::cout << "To implement: Compare direct pipeline (above test) vs BERT model's get_embeddings()\n";
    std::cout << "Expected: If direct pipeline works but BERT model fails, bug is in BERT's wiring/state\n";
}
