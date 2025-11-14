// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal test to reproduce ttnn::embedding batch processing bug
//
// BUG DESCRIPTION:
// When using ttnn::embedding with batch_size > 1, batches after the first
// (batch index > 0) return incorrect embeddings. They often return the
// embedding for token ID=0 instead of the actual token IDs.
//
// EXPECTED: Same token ID should return same embedding in all batches
// ACTUAL: Batch 0 correct, batch 1+ returns wrong embeddings
//
// REPRODUCTION:
//   cd build
//   ./tests/ttml_tests --gtest_filter="*EmbeddingBatchBug*"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

using namespace ttml;

class EmbeddingBatchBugTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(EmbeddingBatchBugTest, MinimalReproduction) {
    // Test configuration
    constexpr uint32_t vocab_size = 10;      // Small vocabulary
    constexpr uint32_t embedding_dim = 8;    // Small embedding dimension
    constexpr uint32_t batch_size = 2;       // 2 batches to show the bug
    constexpr uint32_t sequence_length = 4;  // 4 tokens per batch

    // Align vocab_size to 32 (TTNN requirement)
    constexpr uint32_t vocab_size_aligned = 32;

    fmt::print("\n=== TTNN Embedding Batch Bug Reproduction ===\n\n");
    fmt::print("Configuration:\n");
    fmt::print("  Vocab size: {} (aligned: {})\n", vocab_size, vocab_size_aligned);
    fmt::print("  Embedding dim: {}\n", embedding_dim);
    fmt::print("  Batch size: {}\n", batch_size);
    fmt::print("  Sequence length: {}\n\n", sequence_length);

    // Create weight matrix: [1, 1, vocab_size_aligned, embedding_dim]
    // Fill with distinctive values so we can identify which token was retrieved
    std::vector<float> weight_data(vocab_size_aligned * embedding_dim, 0.0f);
    for (uint32_t token_id = 0; token_id < vocab_size; ++token_id) {
        for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
            // Each token has a distinctive pattern: token_id * 0.1 + dim * 0.01
            weight_data[token_id * embedding_dim + dim] = static_cast<float>(token_id) * 0.1f + dim * 0.01f;
        }
    }

    auto weight_tensor = core::from_vector(
        weight_data, ttnn::Shape({1, 1, vocab_size_aligned, embedding_dim}), &autograd::ctx().get_device());

    fmt::print("Weight matrix created with distinctive patterns:\n");
    fmt::print("  Token 0: [0.00, 0.01, 0.02, 0.03, ...]\n");
    fmt::print("  Token 1: [0.10, 0.11, 0.12, 0.13, ...]\n");
    fmt::print("  Token 2: [0.20, 0.21, 0.22, 0.23, ...]\n");
    fmt::print("  Token 3: [0.30, 0.31, 0.32, 0.33, ...]\n\n");

    // Create input tensor: [batch_size, 1, 1, sequence_length]
    // Batch 0: [1, 2, 3, 4]
    // Batch 1: [1, 5, 6, 7]  <- Note: both batches start with token ID=1
    std::vector<uint32_t> input_data = {
        1,
        2,
        3,
        4,  // Batch 0
        1,
        5,
        6,
        7  // Batch 1 - starts with same token (1) as batch 0
    };

    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data,
        ttnn::Shape({batch_size, 1, 1, sequence_length}),
        &autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);

    fmt::print("Input token IDs:\n");
    fmt::print("  Batch 0: [1, 2, 3, 4]\n");
    fmt::print("  Batch 1: [1, 5, 6, 7]\n");
    fmt::print("  NOTE: Both batches start with token ID=1\n\n");

    // Untilize weight (required by ttnn::embedding)
    weight_tensor = ttnn::untilize(weight_tensor);

    // Call ttnn::embedding
    fmt::print("Calling ttnn::embedding...\n");
    auto output_tensor = ttnn::embedding(input_tensor, weight_tensor, std::nullopt, ttnn::Layout::ROW_MAJOR);

    // Convert output back to host for verification
    auto output_data = core::to_vector(output_tensor);

    fmt::print("\n=== Results ===\n\n");

    // Expected: For token ID=1, embedding should be [0.10, 0.11, 0.12, 0.13, ...]
    std::vector<float> expected_token1_embedding(embedding_dim);
    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        expected_token1_embedding[dim] = 1.0f * 0.1f + dim * 0.01f;
    }

    fmt::print("Expected embedding for token ID=1: [");
    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        fmt::print("{:.2f}", expected_token1_embedding[dim]);
        if (dim < embedding_dim - 1)
            fmt::print(", ");
    }
    fmt::print("]\n\n");

    // Extract embeddings for first token in each batch
    auto output_shape = output_tensor.logical_shape();
    uint32_t output_seq_len = output_shape[1];
    uint32_t output_emb_dim = output_shape[2];

    std::vector<float> batch0_token0(embedding_dim);
    std::vector<float> batch1_token0(embedding_dim);

    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        // Batch 0, token 0
        uint32_t batch0_idx = 0 * output_seq_len * output_emb_dim + 0 * output_emb_dim + dim;
        batch0_token0[dim] = output_data[batch0_idx];

        // Batch 1, token 0
        uint32_t batch1_idx = 1 * output_seq_len * output_emb_dim + 0 * output_emb_dim + dim;
        batch1_token0[dim] = output_data[batch1_idx];
    }

    fmt::print("Batch 0, Token 0 (ID=1): [");
    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        fmt::print("{:.2f}", batch0_token0[dim]);
        if (dim < embedding_dim - 1)
            fmt::print(", ");
    }
    fmt::print("]\n");

    fmt::print("Batch 1, Token 0 (ID=1): [");
    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        fmt::print("{:.2f}", batch1_token0[dim]);
        if (dim < embedding_dim - 1)
            fmt::print(", ");
    }
    fmt::print("]\n\n");

    // Check if batch 0 is correct
    bool batch0_correct = true;
    float batch0_max_diff = 0.0f;
    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        float diff = std::abs(batch0_token0[dim] - expected_token1_embedding[dim]);
        batch0_max_diff = std::max(batch0_max_diff, diff);
        if (diff > 0.01f) {  // Allow small numerical errors
            batch0_correct = false;
        }
    }

    // Check if batch 1 is correct
    bool batch1_correct = true;
    float batch1_max_diff = 0.0f;
    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        float diff = std::abs(batch1_token0[dim] - expected_token1_embedding[dim]);
        batch1_max_diff = std::max(batch1_max_diff, diff);
        if (diff > 0.01f) {
            batch1_correct = false;
        }
    }

    // Check if batch 1 returned token 0 instead
    bool batch1_got_token0 = true;
    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        float expected_token0 = 0.0f * 0.1f + dim * 0.01f;
        float diff = std::abs(batch1_token0[dim] - expected_token0);
        if (diff > 0.01f) {
            batch1_got_token0 = false;
        }
    }

    fmt::print("=== Analysis ===\n\n");
    fmt::print("Batch 0 (max diff: {:.4f}): ", batch0_max_diff);
    if (batch0_correct) {
        fmt::print("✅ CORRECT\n");
    } else {
        fmt::print("❌ WRONG\n");
    }

    fmt::print("Batch 1 (max diff: {:.4f}): ", batch1_max_diff);
    if (batch1_correct) {
        fmt::print("✅ CORRECT\n");
    } else {
        fmt::print("❌ WRONG");
        if (batch1_got_token0) {
            fmt::print(" (retrieved token 0 instead of token 1!)");
        }
        fmt::print("\n");
    }

    fmt::print("\n");

    if (!batch0_correct) {
        fmt::print("⚠️  UNEXPECTED: Batch 0 should be correct\n");
    }

    if (!batch1_correct) {
        fmt::print("🐛 BUG CONFIRMED: ttnn::embedding batch processing bug\n");
        fmt::print("   For batch index > 0, wrong embeddings are returned\n");
        if (batch1_got_token0) {
            fmt::print("   Batch 1 retrieved embedding for token 0 instead of token 1\n");
        }
    }

    fmt::print("\n=== Expected Behavior ===\n");
    fmt::print("Both batches should return the same embedding for token ID=1\n");
    fmt::print("since they both have token ID=1 at position 0.\n\n");

    // Test assertions
    EXPECT_TRUE(batch0_correct) << "Batch 0 should retrieve correct embedding (max diff: " << batch0_max_diff << ")";
    EXPECT_TRUE(batch1_correct) << "Batch 1 should retrieve correct embedding (max diff: " << batch1_max_diff
                                << "). BUG: ttnn::embedding fails for batch > 0";

    fmt::print("=== End of Test ===\n\n");
}

// Test with different input types and configurations
TEST_F(EmbeddingBatchBugTest, DifferentBatchSizes) {
    constexpr uint32_t vocab_size_aligned = 32;
    constexpr uint32_t embedding_dim = 8;
    constexpr uint32_t sequence_length = 2;

    // Create weight matrix
    std::vector<float> weight_data(vocab_size_aligned * embedding_dim, 0.0f);
    for (uint32_t token_id = 0; token_id < 10; ++token_id) {
        for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
            weight_data[token_id * embedding_dim + dim] = static_cast<float>(token_id) * 0.1f + dim * 0.01f;
        }
    }
    auto weight_tensor = core::from_vector(
        weight_data, ttnn::Shape({1, 1, vocab_size_aligned, embedding_dim}), &autograd::ctx().get_device());
    weight_tensor = ttnn::untilize(weight_tensor);

    // Test with batch_size = 3
    fmt::print("\n=== Testing with batch_size = 3 ===\n");
    std::vector<uint32_t> input_data_3 = {
        1,
        2,  // Batch 0
        1,
        3,  // Batch 1
        1,
        4  // Batch 2
    };
    auto input_tensor_3 = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data_3, ttnn::Shape({3, 1, 1, sequence_length}), &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto output_tensor_3 = ttnn::embedding(input_tensor_3, weight_tensor, std::nullopt, ttnn::Layout::ROW_MAJOR);
    auto output_data_3 = core::to_vector(output_tensor_3);

    // Check all batches
    std::vector<float> expected(embedding_dim);
    for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
        expected[dim] = 1.0f * 0.1f + dim * 0.01f;
    }

    auto output_shape = output_tensor_3.logical_shape();
    uint32_t output_seq_len = output_shape[1];
    uint32_t output_emb_dim = output_shape[2];

    for (uint32_t batch = 0; batch < 3; ++batch) {
        std::vector<float> actual(embedding_dim);
        for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
            uint32_t idx = batch * output_seq_len * output_emb_dim + 0 * output_emb_dim + dim;
            actual[dim] = output_data_3[idx];
        }

        float max_diff = 0.0f;
        for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
            max_diff = std::max(max_diff, std::abs(actual[dim] - expected[dim]));
        }

        fmt::print("Batch {} (max diff: {:.4f}): ", batch, max_diff);
        if (max_diff < 0.01f) {
            fmt::print("✅ CORRECT\n");
        } else {
            fmt::print("❌ WRONG\n");
        }

        EXPECT_LT(max_diff, 0.01f) << "Batch " << batch << " should retrieve correct embedding";
    }

    fmt::print("\n");
}
