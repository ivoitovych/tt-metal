// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// BUG REPRODUCTION TEST for ttnn::embedding batch processing
//
// This test demonstrates a critical bug in ttnn::embedding() where batched inputs
// (batch_size > 1) produce identical embeddings for all samples, regardless of
// different input token IDs.
//
// EVIDENCE for bug report: This test directly calls ttnn::embedding() library
// function to prove the bug exists at the TTNN level, not in TTML code.

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

class TtnnEmbeddingBatchBugTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// TEST 1: Single sample (batch_size=1) - BASELINE (should work)
TEST_F(TtnnEmbeddingBatchBugTest, SingleSample_Baseline) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();

    // Configuration
    constexpr uint32_t vocab_size = 100;
    constexpr uint32_t embedding_dim = 64;
    constexpr uint32_t seq_len = 32;
    constexpr uint32_t batch_size = 1;

    // Create weight matrix: [vocab_size, embedding_dim]
    auto weight_shape = ttnn::Shape({vocab_size, embedding_dim});
    std::vector<float> weight_data(vocab_size * embedding_dim);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>(i) / 100.0F;  // Predictable values
    }

    auto weight_cpu = core::from_vector(weight_data, weight_shape, device);
    auto weight_tensor = ttnn::to_layout(weight_cpu, ttnn::Layout::TILE);
    weight_tensor = ttnn::untilize(weight_tensor);

    // Create input: Single sample with token ID 7
    auto input_shape = ttnn::Shape({batch_size, seq_len});
    std::vector<uint32_t> input_data(batch_size * seq_len, 7);  // All token ID 7

    auto input_cpu =
        core::from_vector<uint32_t, ttnn::DataType::UINT32>(input_data, input_shape, device, ttnn::Layout::ROW_MAJOR);

    std::cout << "\n=================================================================\n";
    std::cout << "TEST 1: Single sample baseline (batch_size=1)\n";
    std::cout << "=================================================================\n";
    std::cout << "Input: 1 sample with all tokens = 7\n";

    // Call ttnn::embedding directly
    auto embeddings = ttnn::embedding(input_cpu, weight_tensor, /* pad_token */ std::nullopt, ttnn::Layout::TILE);

    auto embeddings_shape = embeddings.logical_shape();
    ASSERT_EQ(embeddings_shape[0], batch_size);
    ASSERT_EQ(embeddings_shape[1], seq_len);
    ASSERT_EQ(embeddings_shape[2], embedding_dim);

    auto embeddings_vector = core::to_vector(embeddings);

    // Extract first token embedding
    std::vector<float> first_token(embedding_dim);
    std::copy(embeddings_vector.begin(), embeddings_vector.begin() + embedding_dim, first_token.begin());

    std::cout << "✓ Single sample test passed\n";
    std::cout << "  First 5 dims: [";
    for (int i = 0; i < 5; ++i) {
        std::cout << first_token[i];
        if (i < 4)
            std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "=================================================================\n\n";
}

// TEST 2: Batch with DIFFERENT tokens - BUG REPRODUCTION
TEST_F(TtnnEmbeddingBatchBugTest, BatchWithDifferentTokens_BUG_EVIDENCE) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();

    // Configuration
    constexpr uint32_t vocab_size = 100;
    constexpr uint32_t embedding_dim = 64;
    constexpr uint32_t seq_len = 32;
    constexpr uint32_t batch_size = 2;

    // Create weight matrix
    auto weight_shape = ttnn::Shape({vocab_size, embedding_dim});
    std::vector<float> weight_data(vocab_size * embedding_dim);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>(i) / 100.0F;
    }

    auto weight_cpu = core::from_vector(weight_data, weight_shape, device);
    auto weight_tensor = ttnn::to_layout(weight_cpu, ttnn::Layout::TILE);
    weight_tensor = ttnn::untilize(weight_tensor);

    // Create input: Sample 0 = token 7, Sample 1 = token 99
    auto input_shape = ttnn::Shape({batch_size, seq_len});
    std::vector<uint32_t> input_data(batch_size * seq_len);

    // Fill sample 0 with token 7
    for (uint32_t i = 0; i < seq_len; ++i) {
        input_data[i] = 7;
    }
    // Fill sample 1 with token 99
    for (uint32_t i = seq_len; i < 2 * seq_len; ++i) {
        input_data[i] = 99;
    }

    auto input_cpu =
        core::from_vector<uint32_t, ttnn::DataType::UINT32>(input_data, input_shape, device, ttnn::Layout::ROW_MAJOR);

    std::cout << "\n=================================================================\n";
    std::cout << "TEST 2: BUG REPRODUCTION - Different tokens in batch\n";
    std::cout << "=================================================================\n";
    std::cout << "Input configuration:\n";
    std::cout << "  Sample 0: All tokens = 7\n";
    std::cout << "  Sample 1: All tokens = 99\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Sequence length: " << seq_len << "\n";
    std::cout << "  Embedding dim: " << embedding_dim << "\n\n";

    std::cout << "EXPECTED: Different token IDs should produce different embeddings\n";
    std::cout << "Calling ttnn::embedding() directly...\n\n";

    // Call ttnn::embedding - THIS IS WHERE THE BUG OCCURS
    auto embeddings = ttnn::embedding(input_cpu, weight_tensor, /* pad_token */ std::nullopt, ttnn::Layout::TILE);

    auto embeddings_shape = embeddings.logical_shape();
    ASSERT_EQ(embeddings_shape[0], batch_size);
    ASSERT_EQ(embeddings_shape[1], seq_len);
    ASSERT_EQ(embeddings_shape[2], embedding_dim);

    auto embeddings_vector = core::to_vector(embeddings);

    // Extract first token from each sample
    std::vector<float> sample0_token0(embedding_dim);
    std::vector<float> sample1_token0(embedding_dim);

    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        sample0_token0[dim] = embeddings_vector[0 * seq_len * embedding_dim + 0 * embedding_dim + dim];
        sample1_token0[dim] = embeddings_vector[1 * seq_len * embedding_dim + 0 * embedding_dim + dim];
    }

    // Calculate difference
    float max_diff = 0.0F;
    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        float diff = std::abs(sample0_token0[dim] - sample1_token0[dim]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Results:\n";
    std::cout << "  Sample 0 (token 7) first 5 dims: [";
    for (int i = 0; i < 5; ++i) {
        std::cout << sample0_token0[i];
        if (i < 4)
            std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "  Sample 1 (token 99) first 5 dims: [";
    for (int i = 0; i < 5; ++i) {
        std::cout << sample1_token0[i];
        if (i < 4)
            std::cout << ", ";
    }
    std::cout << "]\n\n";

    std::cout << "  Max difference between embeddings: " << max_diff << "\n\n";

    if (max_diff < 0.001F) {
        std::cout << "❌ BUG CONFIRMED!\n";
        std::cout << "   Different token IDs (7 vs 99) produce IDENTICAL embeddings\n";
        std::cout << "   This proves ttnn::embedding() ignores the batch dimension\n";
        std::cout << "   and returns the same embedding for all samples.\n";
        std::cout << "=================================================================\n\n";

        // Document the bug
        std::cout << "\nBUG REPORT EVIDENCE:\n";
        std::cout << "====================\n";
        std::cout << "Function: ttnn::embedding()\n";
        std::cout << "Issue: Batch processing bug - returns identical embeddings for all samples\n";
        std::cout << "Input shape: [" << batch_size << ", " << seq_len << "]\n";
        std::cout << "Weight shape: [" << vocab_size << ", " << embedding_dim << "]\n";
        std::cout << "Sample 0 token IDs: all 7\n";
        std::cout << "Sample 1 token IDs: all 99\n";
        std::cout << "ACTUAL result: Identical embeddings (max diff=" << max_diff << ")\n";
        std::cout << "EXPECTED result: Different embeddings (tokens 7 and 99 are different)\n";
        std::cout << "\nThis test FAILS to demonstrate the bug exists.\n";
        std::cout << "Location: tt-train/tests/core/ttnn_embedding_batch_bug_test.cpp\n";
        std::cout << "====================\n\n";

        FAIL() << "ttnn::embedding batch processing bug confirmed";
    } else {
        std::cout << "✓ CORRECT: Different token IDs produce different embeddings\n";
        std::cout << "   Bug appears to be fixed!\n";
        std::cout << "=================================================================\n\n";
    }
}

// TEST 3: Larger batch (4 samples) - Extended evidence
TEST_F(TtnnEmbeddingBatchBugTest, LargerBatch_ExtendedEvidence) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();

    // Configuration
    constexpr uint32_t vocab_size = 100;
    constexpr uint32_t embedding_dim = 64;
    constexpr uint32_t seq_len = 16;  // Shorter for efficiency
    constexpr uint32_t batch_size = 4;

    // Create weight matrix
    auto weight_shape = ttnn::Shape({vocab_size, embedding_dim});
    std::vector<float> weight_data(vocab_size * embedding_dim);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>(i) / 100.0F;
    }

    auto weight_cpu = core::from_vector(weight_data, weight_shape, device);
    auto weight_tensor = ttnn::to_layout(weight_cpu, ttnn::Layout::TILE);
    weight_tensor = ttnn::untilize(weight_tensor);

    // Create input: Each sample uses a different token ID
    auto input_shape = ttnn::Shape({batch_size, seq_len});
    std::vector<uint32_t> input_data(batch_size * seq_len);

    const std::vector<uint32_t> token_ids = {5, 15, 25, 35};
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (uint32_t pos = 0; pos < seq_len; ++pos) {
            input_data[batch_idx * seq_len + pos] = token_ids[batch_idx];
        }
    }

    auto input_cpu =
        core::from_vector<uint32_t, ttnn::DataType::UINT32>(input_data, input_shape, device, ttnn::Layout::ROW_MAJOR);

    std::cout << "\n=================================================================\n";
    std::cout << "TEST 3: Larger batch (4 samples) - Extended evidence\n";
    std::cout << "=================================================================\n";
    std::cout << "Input configuration:\n";
    for (uint32_t i = 0; i < batch_size; ++i) {
        std::cout << "  Sample " << i << ": All tokens = " << token_ids[i] << "\n";
    }
    std::cout << "\n";

    // Call ttnn::embedding
    auto embeddings = ttnn::embedding(input_cpu, weight_tensor, /* pad_token */ std::nullopt, ttnn::Layout::TILE);

    auto embeddings_shape = embeddings.logical_shape();
    auto embeddings_vector = core::to_vector(embeddings);

    // Extract first token embedding from each sample
    std::vector<std::vector<float>> sample_embeddings(batch_size);
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        sample_embeddings[batch_idx].resize(embedding_dim);
        for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
            sample_embeddings[batch_idx][dim] =
                embeddings_vector[batch_idx * seq_len * embedding_dim + 0 * embedding_dim + dim];
        }
    }

    // Print first 5 dims of each sample
    std::cout << "Results (first 5 dimensions):\n";
    for (uint32_t i = 0; i < batch_size; ++i) {
        std::cout << "  Sample " << i << " (token " << token_ids[i] << "): [";
        for (int d = 0; d < 5; ++d) {
            std::cout << sample_embeddings[i][d];
            if (d < 4)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    // Check if all samples have identical embeddings (BUG)
    bool all_identical = true;
    for (uint32_t batch_idx = 1; batch_idx < batch_size; ++batch_idx) {
        for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
            if (std::abs(sample_embeddings[0][dim] - sample_embeddings[batch_idx][dim]) > 1e-5) {
                all_identical = false;
                break;
            }
        }
        if (!all_identical)
            break;
    }

    if (all_identical) {
        std::cout << "❌ BUG CONFIRMED in larger batch!\n";
        std::cout << "   All 4 samples have IDENTICAL embeddings despite different token IDs\n";
        std::cout << "   (5, 15, 25, 35 all produce same embedding)\n";
        std::cout << "=================================================================\n\n";
        FAIL() << "ttnn::embedding bug confirmed in 4-sample batch";
    } else {
        std::cout << "✓ CORRECT: Different samples produce different embeddings\n";
        std::cout << "=================================================================\n\n";
    }
}
