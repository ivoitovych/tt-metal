// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Test hypothesis: ttnn::add broadcasting bug when adding {1,1,seq,emb} to {batch,1,seq,emb}

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/embedding_op.hpp"

class BroadcastingHypothesisTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// Test if ttnn::embedding works with 4D input tensors (like BERT uses)
TEST_F(BroadcastingHypothesisTest, Embedding4DInputBatch) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();

    constexpr uint32_t vocab_size = 100;
    constexpr uint32_t batch_size = 2;
    constexpr uint32_t seq_len = 32;
    constexpr uint32_t emb_dim = 64;

    // Create weight matrix
    std::vector<float> weight_data(vocab_size * emb_dim);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>(i) / 100.0F;
    }
    auto weights = core::from_vector(weight_data, ttnn::Shape({vocab_size, emb_dim}), device);
    weights = ttnn::to_layout(weights, ttnn::Layout::TILE);
    weights = ttnn::untilize(weights);

    // Create 4D input like BERT does: [batch, 1, 1, seq]
    std::vector<uint32_t> input_data(batch_size * seq_len);
    for (uint32_t i = 0; i < seq_len; ++i) {
        input_data[i] = 7;  // Sample 0: all token 7
    }
    for (uint32_t i = seq_len; i < 2 * seq_len; ++i) {
        input_data[i] = 99;  // Sample 1: all token 99
    }

    auto input_4d = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR);

    std::cout << "\n4D Embedding Test (BERT-style input):\n";
    std::cout << "  Input shape: [" << batch_size << ", 1, 1, " << seq_len << "]\n";
    std::cout << "  Sample 0: all tokens = 7\n";
    std::cout << "  Sample 1: all tokens = 99\n";

    // Call ttnn::embedding with 4D input (like original BERT code does)
    auto embeddings = ttnn::embedding(input_4d, weights, std::nullopt, ttnn::Layout::TILE);

    auto emb_shape = embeddings.logical_shape();
    std::cout << "  Output shape: [" << emb_shape[0] << ", " << emb_shape[1] << ", " << emb_shape[2];
    if (emb_shape.rank() > 3) {
        std::cout << ", " << emb_shape[3];
    }
    std::cout << "]\n";

    auto emb_vec = core::to_vector(embeddings);

    // Extract first embedding value from each sample
    float sample0_val = emb_vec[0];
    float sample1_val = emb_vec[seq_len * emb_dim];  // First value of second sample

    std::cout << "  Sample 0 first embedding value: " << sample0_val << "\n";
    std::cout << "  Sample 1 first embedding value: " << sample1_val << "\n";

    if (std::abs(sample0_val - sample1_val) < 0.01F) {
        std::cout << "\n❌ BUG FOUND: 4D input produces identical embeddings!\n";
        std::cout << "   This reproduces the BERT batch bug with 4D inputs.\n";
        FAIL() << "ttnn::embedding bug with 4D batch inputs confirmed";
    } else {
        std::cout << "\n✓ Different tokens produce different embeddings (4D input works).\n";
    }
}

// Test through actual ops::embedding_op path (with autograd)
TEST_F(BroadcastingHypothesisTest, EmbeddingOpWithAutograd) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();

    constexpr uint32_t vocab_size = 100;
    constexpr uint32_t batch_size = 2;
    constexpr uint32_t seq_len = 32;
    constexpr uint32_t emb_dim = 64;

    // Create weight tensor with autograd
    auto weight_ag = autograd::create_tensor();
    std::vector<float> weight_data(vocab_size * emb_dim);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>(i) / 100.0F;
    }
    weight_ag->set_value(core::from_vector(weight_data, ttnn::Shape({1, 1, vocab_size, emb_dim}), device));

    // Create input tensor with autograd
    std::vector<uint32_t> input_data(batch_size * seq_len);
    for (uint32_t i = 0; i < seq_len; ++i) {
        input_data[i] = 7;  // Sample 0: all token 7
    }
    for (uint32_t i = seq_len; i < 2 * seq_len; ++i) {
        input_data[i] = 99;  // Sample 1: all token 99
    }

    auto input_ag = autograd::create_tensor(core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, seq_len}), device, ttnn::Layout::ROW_MAJOR));

    std::cout << "\nops::embedding_op Test (WITH autograd):\n";
    std::cout << "  Input shape: [" << batch_size << ", 1, 1, " << seq_len << "]\n";
    std::cout << "  Sample 0: all tokens = 7\n";
    std::cout << "  Sample 1: all tokens = 99\n";

    // Call through ops::embedding_op (the actual BERT path)
    auto embeddings_ag = ops::embedding_op(input_ag, weight_ag);
    auto emb_tensor = embeddings_ag->get_value();
    auto emb_shape = emb_tensor.logical_shape();

    std::cout << "  Output shape: [" << emb_shape[0] << ", " << emb_shape[1] << ", " << emb_shape[2] << ", "
              << emb_shape[3] << "]\n";

    auto emb_vec = core::to_vector(emb_tensor);

    // Extract first embedding value from each sample
    float sample0_val = emb_vec[0];
    float sample1_val = emb_vec[seq_len * emb_dim];  // First value of second sample

    std::cout << "  Sample 0 first embedding value: " << sample0_val << "\n";
    std::cout << "  Sample 1 first embedding value: " << sample1_val << "\n";

    if (std::abs(sample0_val - sample1_val) < 0.01F) {
        std::cout << "\n❌ BUG FOUND: ops::embedding_op produces identical embeddings!\n";
        std::cout << "   This reproduces the BERT batch bug through autograd path.\n";
        FAIL() << "ops::embedding_op bug with batched inputs confirmed";
    } else {
        std::cout << "\n✓ Different tokens produce different embeddings (autograd path works).\n";
    }
}

// Test if ttnn::add correctly broadcasts {1,1,32,64} to {2,1,32,64}
TEST_F(BroadcastingHypothesisTest, AddBroadcastingBatchDimension) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();

    constexpr uint32_t batch_size = 2;
    constexpr uint32_t seq_len = 32;
    constexpr uint32_t emb_dim = 64;

    // Create tensor A with batch_size=2, different values per batch
    std::vector<float> data_a(batch_size * seq_len * emb_dim);
    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t i = 0; i < seq_len * emb_dim; ++i) {
            data_a[b * seq_len * emb_dim + i] = static_cast<float>(b) + 1.0F;  // batch 0 = 1.0, batch 1 = 2.0
        }
    }
    auto tensor_a = core::from_vector(data_a, ttnn::Shape({batch_size, 1, seq_len, emb_dim}), device);

    // Create tensor B with batch_size=1 to be broadcasted
    std::vector<float> data_b(seq_len * emb_dim, 10.0F);  // All values = 10.0
    auto tensor_b = core::from_vector(data_b, ttnn::Shape({1, 1, seq_len, emb_dim}), device);

    // Add with broadcasting
    auto result = ttnn::add(tensor_a, tensor_b);
    auto result_vec = core::to_vector(result);

    // Verify:
    // Sample 0 should be: 1.0 + 10.0 = 11.0
    // Sample 1 should be: 2.0 + 10.0 = 12.0

    float sample0_val = result_vec[0];
    float sample1_val = result_vec[seq_len * emb_dim];  // First value of second batch

    std::cout << "\nBroadcasting Test Results:\n";
    std::cout << "  Sample 0 first value: " << sample0_val << " (expected: 11.0)\n";
    std::cout << "  Sample 1 first value: " << sample1_val << " (expected: 12.0)\n";

    // Check if broadcasting worked correctly
    EXPECT_NEAR(sample0_val, 11.0F, 0.01F) << "Sample 0 should be 1.0 + 10.0 = 11.0";
    EXPECT_NEAR(sample1_val, 12.0F, 0.01F) << "Sample 1 should be 2.0 + 10.0 = 12.0";

    if (std::abs(sample0_val - sample1_val) < 0.01F) {
        std::cout << "\n❌ BUG CONFIRMED: Broadcasting produced identical outputs!\n";
        std::cout << "   Both samples got value: " << sample0_val << "\n";
        std::cout << "   This confirms ttnn::add broadcasting bug hypothesis.\n";
    } else {
        std::cout << "\n✓ Broadcasting works correctly.\n";
        std::cout << "   Hypothesis REJECTED - bug is elsewhere.\n";
    }
}
