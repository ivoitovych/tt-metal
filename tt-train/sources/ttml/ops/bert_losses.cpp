// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "bert_losses.hpp"

#include "modules/bert_heads.hpp"
#include "ops/binary_ops.hpp"
#include "ops/losses.hpp"

namespace ttml::ops::bert_losses {

// ============================================================================
// Sequence Classification Loss
// ============================================================================

autograd::TensorPtr compute_sequence_classification_loss(
    const autograd::TensorPtr& logits, const autograd::TensorPtr& labels) {
    // logits: [B, 1, 1, num_labels]
    // labels: [B] or [B, 1]
    return ops::cross_entropy_loss(logits, labels, ops::ReduceType::MEAN);
}

// ============================================================================
// Token Classification Loss
// ============================================================================

autograd::TensorPtr compute_token_classification_loss(
    const autograd::TensorPtr& logits, const autograd::TensorPtr& labels, const autograd::TensorPtr& attention_mask) {
    // logits: [B, 1, S, num_labels]
    // labels: [B, S] with -100 for padding
    // Cross-entropy automatically ignores -100 labels
    return ops::cross_entropy_loss(logits, labels, ops::ReduceType::MEAN);
}

// ============================================================================
// Question Answering Loss
// ============================================================================

autograd::TensorPtr compute_question_answering_loss(
    const autograd::TensorPtr& combined_logits,
    const autograd::TensorPtr& start_positions,
    const autograd::TensorPtr& end_positions) {
    // Split combined logits [B, 1, S, 2] into start and end
    auto split = modules::BertQuestionAnsweringHead::split_logits(combined_logits);

    return compute_question_answering_loss_split(split.start_logits, split.end_logits, start_positions, end_positions);
}

autograd::TensorPtr compute_question_answering_loss_split(
    const autograd::TensorPtr& start_logits,
    const autograd::TensorPtr& end_logits,
    const autograd::TensorPtr& start_positions,
    const autograd::TensorPtr& end_positions) {
    // Compute start loss
    auto start_loss = ops::cross_entropy_loss(start_logits, start_positions, ops::ReduceType::MEAN);

    // Compute end loss
    auto end_loss = ops::cross_entropy_loss(end_logits, end_positions, ops::ReduceType::MEAN);

    // Average the two losses
    auto total_loss = ops::add(start_loss, end_loss);
    return ops::mul(total_loss, 0.5F);
}

// ============================================================================
// Masked Language Modeling Loss
// ============================================================================

autograd::TensorPtr compute_masked_lm_loss(
    const autograd::TensorPtr& logits, const autograd::TensorPtr& labels, const autograd::TensorPtr& attention_mask) {
    // logits: [B, 1, S, vocab_size]
    // labels: [B, S] with -100 for non-masked tokens
    // Cross-entropy automatically ignores -100 labels
    return ops::cross_entropy_loss(logits, labels, ops::ReduceType::MEAN);
}

// ============================================================================
// Next Sentence Prediction Loss
// ============================================================================

autograd::TensorPtr compute_nsp_loss(const autograd::TensorPtr& logits, const autograd::TensorPtr& labels) {
    // logits: [B, 1, 1, 2]
    // labels: [B] (0 or 1)
    return ops::cross_entropy_loss(logits, labels, ops::ReduceType::MEAN);
}

// ============================================================================
// Pre-Training Combined Loss
// ============================================================================

autograd::TensorPtr compute_pretraining_loss(
    const autograd::TensorPtr& mlm_logits,
    const autograd::TensorPtr& nsp_logits,
    const autograd::TensorPtr& mlm_labels,
    const autograd::TensorPtr& nsp_labels,
    float mlm_weight,
    float nsp_weight) {
    // Compute MLM loss
    auto mlm_loss = compute_masked_lm_loss(mlm_logits, mlm_labels, nullptr);

    // Compute NSP loss
    auto nsp_loss = compute_nsp_loss(nsp_logits, nsp_labels);

    // Weight and combine losses
    auto weighted_mlm = ops::mul(mlm_loss, mlm_weight);
    auto weighted_nsp = ops::mul(nsp_loss, nsp_weight);

    return ops::add(weighted_mlm, weighted_nsp);
}

}  // namespace ttml::ops::bert_losses
