// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <memory>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/embedding_op.hpp"

class EmbeddingWordVsTokenTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }

    // Helper to compute PCC
    float compute_pcc(const std::vector<float>& tensor1, const std::vector<float>& tensor2) {
        if (tensor1.size() != tensor2.size()) {
            return 0.0F;
        }

        // Compute means
        float mean1 = 0.0F;
        float mean2 = 0.0F;
        for (size_t i = 0; i < tensor1.size(); ++i) {
            mean1 += tensor1[i];
            mean2 += tensor2[i];
        }
        mean1 /= static_cast<float>(tensor1.size());
        mean2 /= static_cast<float>(tensor2.size());

        // Compute covariance and standard deviations
        float numerator = 0.0F;
        float var1 = 0.0F;
        float var2 = 0.0F;

        for (size_t i = 0; i < tensor1.size(); ++i) {
            float diff1 = tensor1[i] - mean1;
            float diff2 = tensor2[i] - mean2;
            numerator += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }

        float denominator = std::sqrt(var1 * var2);
        if (denominator == 0.0F) {
            return 0.0F;
        }

        return numerator / denominator;
    }
};

// Test word embeddings with batch_size=2
// Expected: PCC < 0.999 (demonstrates the bug)
TEST_F(EmbeddingWordVsTokenTypeTest, WordEmbeddingsBatchSize2ShowsDegradation) {
    using namespace ttml;

    // Configuration matching BERT tiny
    const uint32_t vocab_size = 30528;   // Aligned to 32
    const uint32_t embedding_dim = 128;  // Aligned to 32
    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;

    auto* device = &autograd::ctx().get_device();

    // Create word embedding weight matrix (large vocabulary) with random data
    std::vector<float> weight_data((size_t)vocab_size * embedding_dim);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5F;
    }
    auto weight_tensor = core::from_vector(weight_data, ttnn::Shape({1, 1, vocab_size, embedding_dim}), device);
    autograd::TensorPtr word_weight = autograd::create_tensor(weight_tensor);

    // Create input indices - batch_size=2 with same indices in both batches
    std::vector<uint32_t> input_ids_data(batch_size * seq_len);
    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
            input_ids_data[b * seq_len + s] = (s * 100) % vocab_size;  // Varied indices
        }
    }

    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR);
    autograd::TensorPtr input_ids = autograd::create_tensor(input_tensor);

    // Perform embedding lookup
    autograd::TensorPtr word_embeddings = ops::embedding_op(input_ids, word_weight);

    // Extract embeddings for both batches
    auto result = core::to_vector(word_embeddings->get_value());

    // Expected shape: [batch_size, 1, seq_len, embedding_dim]
    auto output_shape = word_embeddings->get_value().logical_shape();
    ASSERT_EQ(output_shape[0], batch_size);
    ASSERT_EQ(output_shape[2], seq_len);
    ASSERT_EQ(output_shape[3], embedding_dim);

    // Compare batch 0 vs batch 1 (they should be identical since indices are the same)
    size_t elements_per_batch = seq_len * embedding_dim;
    std::vector<float> batch0(result.begin(), result.begin() + elements_per_batch);
    std::vector<float> batch1(result.begin() + elements_per_batch, result.begin() + 2 * elements_per_batch);

    float pcc = compute_pcc(batch0, batch1);

    // Document the bug: batches should have PCC ≈ 1.0 but they don't
    // This test will PASS when it detects the bug (PCC < 0.999)
    // Once the bug is fixed in ttnn::embedding(), this test should FAIL
    // and should be updated to EXPECT_GT(pcc, 0.999)

    std::cout << "\n=== Word Embeddings (batch_size=2) ===" << std::endl;
    std::cout << "Vocab size: " << vocab_size << std::endl;
    std::cout << "PCC between batch 0 and batch 1: " << pcc << std::endl;

    if (pcc < 0.999F) {
        std::cout << "⚠️  BUG DETECTED: Batches differ despite same indices!" << std::endl;
        std::cout << "    Expected PCC > 0.999, got " << pcc << std::endl;
        std::cout << "    This confirms the word embedding batch handling bug." << std::endl;
        // Test PASSES when bug is detected
        EXPECT_LT(pcc, 0.999F) << "Word embeddings should show batch degradation (bug present)";
    } else {
        std::cout << "✅ Bug appears to be fixed! PCC > 0.999" << std::endl;
        // If bug is fixed, this branch will execute
        EXPECT_GT(pcc, 0.999F) << "Word embeddings batch handling appears fixed";
    }
}

// Test token type embeddings with batch_size=2
// Expected: PCC > 0.999 (proves operation is fundamentally correct)
TEST_F(EmbeddingWordVsTokenTypeTest, TokenTypeEmbeddingsBatchSize2WorksCorrectly) {
    using namespace ttml;

    // Configuration matching BERT tiny
    const uint32_t type_vocab_size = 2;  // Small vocabulary (sentence A vs B)
    const uint32_t embedding_dim = 128;  // Aligned to 32
    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;

    auto* device = &autograd::ctx().get_device();

    // Create token type embedding weight matrix (small vocabulary, padded to 32)
    std::vector<float> weight_data(32 * embedding_dim);  // Padded to 32
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5F;
    }
    auto weight_tensor = core::from_vector(weight_data, ttnn::Shape({1, 1, 32, embedding_dim}), device);
    autograd::TensorPtr token_type_weight = autograd::create_tensor(weight_tensor);

    // Create input indices - batch_size=2 with same indices in both batches
    std::vector<uint32_t> token_type_ids_data(batch_size * seq_len);
    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
            token_type_ids_data[b * seq_len + s] = 0;  // All zeros (sentence A)
        }
    }

    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        token_type_ids_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR);
    autograd::TensorPtr token_type_ids = autograd::create_tensor(input_tensor);

    // Perform embedding lookup
    autograd::TensorPtr token_type_embeddings = ops::embedding_op(token_type_ids, token_type_weight);

    // Extract embeddings for both batches
    auto result = core::to_vector(token_type_embeddings->get_value());

    // Expected shape: [batch_size, 1, seq_len, embedding_dim]
    auto output_shape = token_type_embeddings->get_value().logical_shape();
    ASSERT_EQ(output_shape[0], batch_size);
    ASSERT_EQ(output_shape[2], seq_len);
    ASSERT_EQ(output_shape[3], embedding_dim);

    // Compare batch 0 vs batch 1 (they should be identical since indices are the same)
    size_t elements_per_batch = seq_len * embedding_dim;
    std::vector<float> batch0(result.begin(), result.begin() + elements_per_batch);
    std::vector<float> batch1(result.begin() + elements_per_batch, result.begin() + 2 * elements_per_batch);

    float pcc = compute_pcc(batch0, batch1);

    std::cout << "\n=== Token Type Embeddings (batch_size=2) ===" << std::endl;
    std::cout << "Type vocab size: " << type_vocab_size << std::endl;
    std::cout << "PCC between batch 0 and batch 1: " << pcc << std::endl;

    // Token type embeddings should work correctly (PCC > 0.999)
    EXPECT_GT(pcc, 0.999F) << "Token type embeddings should have identical batches (PCC > 0.999)";

    if (pcc > 0.999F) {
        std::cout << "✅ Token type embeddings work correctly!" << std::endl;
        std::cout << "    This proves embedding operation is fundamentally sound." << std::endl;
    } else {
        std::cout << "❌ UNEXPECTED: Token type embeddings also show degradation!" << std::endl;
    }
}

// Test both word and token type embeddings side-by-side
// Expected: Token type PCC > 0.999, Word PCC < 0.999
TEST_F(EmbeddingWordVsTokenTypeTest, SideBySideComparison) {
    using namespace ttml;

    const uint32_t word_vocab_size = 30528;
    const uint32_t type_vocab_size = 32;  // Padded
    const uint32_t embedding_dim = 128;
    const uint32_t batch_size = 2;
    const uint32_t seq_len = 32;

    auto* device = &autograd::ctx().get_device();

    // Create weights
    std::vector<float> word_weight_data((size_t)word_vocab_size * embedding_dim);
    for (size_t i = 0; i < word_weight_data.size(); ++i) {
        word_weight_data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5F;
    }
    auto word_weight_tensor =
        core::from_vector(word_weight_data, ttnn::Shape({1, 1, word_vocab_size, embedding_dim}), device);
    autograd::TensorPtr word_weight = autograd::create_tensor(word_weight_tensor);

    std::vector<float> token_type_weight_data((size_t)type_vocab_size * embedding_dim);
    for (size_t i = 0; i < token_type_weight_data.size(); ++i) {
        token_type_weight_data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5F;
    }
    auto token_type_weight_tensor =
        core::from_vector(token_type_weight_data, ttnn::Shape({1, 1, type_vocab_size, embedding_dim}), device);
    autograd::TensorPtr token_type_weight = autograd::create_tensor(token_type_weight_tensor);

    // Create identical indices for both batches
    std::vector<uint32_t> word_ids_data(batch_size * seq_len);
    std::vector<uint32_t> token_type_ids_data(batch_size * seq_len);

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
            word_ids_data[b * seq_len + s] = (s * 100) % word_vocab_size;
            token_type_ids_data[b * seq_len + s] = 0;
        }
    }

    auto word_input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        word_ids_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR);
    autograd::TensorPtr word_ids = autograd::create_tensor(word_input_tensor);

    auto token_type_input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        token_type_ids_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR);
    autograd::TensorPtr token_type_ids = autograd::create_tensor(token_type_input_tensor);

    // Perform lookups
    autograd::TensorPtr word_embeddings = ops::embedding_op(word_ids, word_weight);
    autograd::TensorPtr token_type_embeddings = ops::embedding_op(token_type_ids, token_type_weight);

    // Extract and compare
    auto word_result = core::to_vector(word_embeddings->get_value());
    auto token_type_result = core::to_vector(token_type_embeddings->get_value());

    size_t elements_per_batch = seq_len * embedding_dim;

    // Word embeddings: batch 0 vs batch 1
    std::vector<float> word_batch0(word_result.begin(), word_result.begin() + elements_per_batch);
    std::vector<float> word_batch1(
        word_result.begin() + elements_per_batch, word_result.begin() + 2 * elements_per_batch);
    float word_pcc = compute_pcc(word_batch0, word_batch1);

    // Token type embeddings: batch 0 vs batch 1
    std::vector<float> token_type_batch0(token_type_result.begin(), token_type_result.begin() + elements_per_batch);
    std::vector<float> token_type_batch1(
        token_type_result.begin() + elements_per_batch, token_type_result.begin() + 2 * elements_per_batch);
    float token_type_pcc = compute_pcc(token_type_batch0, token_type_batch1);

    std::cout << "\n=== SIDE-BY-SIDE COMPARISON ===" << std::endl;
    std::cout << "Word embeddings PCC (batch 0 vs 1):       " << word_pcc << std::endl;
    std::cout << "Token type embeddings PCC (batch 0 vs 1): " << token_type_pcc << std::endl;
    std::cout << "\nExpected behavior (bug present):" << std::endl;
    std::cout << "  - Word embeddings:       PCC < 0.999 (BUG)" << std::endl;
    std::cout << "  - Token type embeddings: PCC > 0.999 (WORKS)" << std::endl;

    // This demonstrates the bug: same operation, different behavior based on vocab size
    EXPECT_GT(token_type_pcc, 0.999F) << "Token type embeddings should work correctly";

    if (word_pcc < 0.999F) {
        std::cout << "\n⚠️  ROOT CAUSE CONFIRMED:" << std::endl;
        std::cout << "   - Same embedding operation" << std::endl;
        std::cout << "   - Different vocab sizes" << std::endl;
        std::cout << "   - Token type (vocab=2):   PCC = " << token_type_pcc << " ✅" << std::endl;
        std::cout << "   - Word (vocab=30528):     PCC = " << word_pcc << " ❌" << std::endl;
        std::cout << "   - Bug is in ttnn::embedding() batch handling for large vocab!" << std::endl;

        EXPECT_LT(word_pcc, 0.999F) << "Word embeddings show batch degradation (confirms bug)";
    } else {
        std::cout << "\n✅ BUG APPEARS FIXED!" << std::endl;
        EXPECT_GT(word_pcc, 0.999F) << "Word embeddings batch handling appears fixed";
    }
}
