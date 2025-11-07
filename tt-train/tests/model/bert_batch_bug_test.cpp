// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Batch Processing Bug Test
 *
 * This test verifies that batch processing correctly produces DIFFERENT outputs
 * for DIFFERENT inputs. This is a critical correctness test for any batched model.
 *
 * CURRENT STATUS: FAILING - This test exposes a critical batch processing bug.
 *
 * BUG DESCRIPTION:
 * When batch_size > 1, the BERT model produces IDENTICAL outputs for all samples
 * in the batch, regardless of input differences. This makes batch processing
 * completely non-functional.
 *
 * EVIDENCE:
 * - Python tests: All samples in batch produce identical outputs
 * - C++ tests: This test confirms the same behavior
 * - Individual processing (batch_size=1) works correctly
 * - The bug affects both base BERT and BertForSequenceClassification
 *
 * ROOT CAUSE: Under investigation. Possible issues:
 * - Embedding layer not handling batch dimension correctly
 * - Attention mechanism collapsing batch information
 * - Slice operation in pooler extracting wrong batch elements
 *
 * IMPACT: BLOCKING for production use. Batch inference and training are broken.
 *
 * NOTE: The existing BatchSizeIndependence test in bert_seq_cls_test.cpp is a
 * FALSE POSITIVE. It only verifies that sample 0 from a batch matches an individual
 * run, but never checks if samples within the batch differ from each other.
 *
 * See also:
 * - tt-train/debug_batch_processing.py: Detailed Python reproduction
 * - tt-train/debug_tensor_shapes.py: Shape inspection across batch sizes
 */

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/bert.hpp"

using namespace ttml;
using namespace ttml::models::bert;

class BertBatchBugTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(BertBatchBugTest, DifferentInputsProduceDifferentOutputs) {
    // Test that DIFFERENT inputs in a batch produce DIFFERENT outputs

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

    const size_t seq_len = 32;
    const size_t batch_size = 2;

    // Create batch with VERY DIFFERENT inputs
    // IMPORTANT: Token IDs must be uint32, not float!
    std::vector<uint32_t> input_ids_batch(batch_size * seq_len);
    std::vector<uint32_t> token_type_ids_batch(batch_size * seq_len);
    std::vector<float> attention_mask_batch(batch_size * seq_len);

    // First sample: all token ID = 7
    for (size_t i = 0; i < seq_len; ++i) {
        input_ids_batch[i] = 7;
        token_type_ids_batch[i] = 0;
        attention_mask_batch[i] = 1.0F;
    }

    // Second sample: all token ID = 99 (very different from 7)
    for (size_t i = seq_len; i < 2 * seq_len; ++i) {
        input_ids_batch[i] = 99;
        token_type_ids_batch[i] = 0;
        attention_mask_batch[i] = 1.0F;
    }

    std::cout << "Creating batched tensor with shape [2, 1, 1, 32]" << std::endl;
    std::cout << "Sample 0: all tokens = 7" << std::endl;
    std::cout << "Sample 1: all tokens = 99" << std::endl;

    // Create batched tensors with correct dtypes
    auto input_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_ids_batch, ttnn::Shape{2, 1, 1, seq_len}, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto token_type_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        token_type_ids_batch, ttnn::Shape{2, 1, 1, seq_len}, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    auto attention_mask_tensor =
        core::from_vector(attention_mask_batch, ttnn::Shape{2, 1, 1, seq_len}, &autograd::ctx().get_device());

    auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
    auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);
    auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

    // Forward pass
    auto logits = (*model)(input_ids_ag, attention_mask_ag, token_type_ids_ag);
    auto logits_data = core::to_vector(logits->get_value());

    auto logits_shape = logits->get_value().logical_shape();
    size_t num_labels_aligned = logits_shape[3];

    std::cout << "Output shape: [" << logits_shape[0] << ", " << logits_shape[1] << ", " << logits_shape[2] << ", "
              << logits_shape[3] << "]" << std::endl;

    // Extract outputs for each sample
    std::vector<float> logits_sample0(num_labels_aligned);
    std::vector<float> logits_sample1(num_labels_aligned);

    // Assuming row-major layout: [batch, 1, 1, num_labels_aligned]
    for (size_t i = 0; i < num_labels_aligned; ++i) {
        logits_sample0[i] = logits_data[i];                       // First num_labels_aligned elements
        logits_sample1[i] = logits_data[num_labels_aligned + i];  // Next num_labels_aligned elements
    }

    std::cout << "Sample 0 logits: [";
    for (size_t i = 0; i < num_labels; ++i) {
        std::cout << logits_sample0[i];
        if (i < num_labels - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "Sample 1 logits: [";
    for (size_t i = 0; i < num_labels; ++i) {
        std::cout << logits_sample1[i];
        if (i < num_labels - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // Check if outputs are different
    float max_diff = 0.0F;
    for (size_t i = 0; i < num_labels; ++i) {
        float diff = std::abs(logits_sample0[i] - logits_sample1[i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Max difference: " << max_diff << std::endl;

    // CRITICAL TEST: Different inputs should produce different outputs
    // If max_diff is very small (< 0.001), the outputs are essentially identical
    if (max_diff < 0.001F) {
        std::cout << "❌ BUG CONFIRMED: Different inputs produce IDENTICAL outputs!" << std::endl;
        std::cout << "   This is a critical batch processing bug." << std::endl;
        FAIL() << "Batch processing bug: Different inputs produce identical outputs";
    } else {
        std::cout << "✓ Different inputs produce different outputs" << std::endl;
        EXPECT_GT(max_diff, 0.001F);
    }
}
