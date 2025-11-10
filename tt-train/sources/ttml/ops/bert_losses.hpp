// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops::bert_losses {

/**
 * @brief Free function loss helpers for BERT tasks
 *
 * Following TTML tradition (GPT-2/Llama): trainers own loss computation.
 * These are convenience functions - trainers can also use
 * standard ops::cross_entropy_loss directly.
 */

// ============================================================================
// Sequence Classification Loss
// ============================================================================

/**
 * @brief Compute cross-entropy loss for sequence classification
 * @param logits [B, 1, 1, num_labels]
 * @param labels [B] or [B, 1]
 * @return Scalar loss
 */
[[nodiscard]] autograd::TensorPtr compute_sequence_classification_loss(
    const autograd::TensorPtr& logits, const autograd::TensorPtr& labels);

// ============================================================================
// Token Classification Loss
// ============================================================================

/**
 * @brief Compute token classification loss with padding mask
 * @param logits [B, 1, S, num_labels]
 * @param labels [B, S] with -100 for padding
 * @param attention_mask [B, 1, 1, S] (optional)
 * @return Scalar loss
 *
 * Note: Cross-entropy automatically ignores -100 labels
 */
[[nodiscard]] autograd::TensorPtr compute_token_classification_loss(
    const autograd::TensorPtr& logits,
    const autograd::TensorPtr& labels,
    const autograd::TensorPtr& attention_mask = nullptr);

// ============================================================================
// Question Answering Loss
// ============================================================================

/**
 * @brief Compute QA loss from combined logits
 * @param combined_logits [B, 1, S, 2] (start and end concatenated)
 * @param start_positions [B] or [B, 1]
 * @param end_positions [B] or [B, 1]
 * @return Scalar loss (average of start and end losses)
 */
[[nodiscard]] autograd::TensorPtr compute_question_answering_loss(
    const autograd::TensorPtr& combined_logits,
    const autograd::TensorPtr& start_positions,
    const autograd::TensorPtr& end_positions);

/**
 * @brief Compute QA loss from separate start/end logits
 * @param start_logits [B, 1, S, 1]
 * @param end_logits [B, 1, S, 1]
 * @param start_positions [B] or [B, 1]
 * @param end_positions [B] or [B, 1]
 * @return Scalar loss (average of start and end losses)
 */
[[nodiscard]] autograd::TensorPtr compute_question_answering_loss_split(
    const autograd::TensorPtr& start_logits,
    const autograd::TensorPtr& end_logits,
    const autograd::TensorPtr& start_positions,
    const autograd::TensorPtr& end_positions);

// ============================================================================
// Masked Language Modeling Loss
// ============================================================================

/**
 * @brief Compute MLM loss with masking
 * @param logits [B, 1, S, vocab_size]
 * @param labels [B, S] with -100 for non-masked tokens
 * @param attention_mask [B, 1, 1, S] (optional)
 * @return Scalar loss
 *
 * Note: Only masked positions (labels != -100) contribute to loss
 */
[[nodiscard]] autograd::TensorPtr compute_masked_lm_loss(
    const autograd::TensorPtr& logits,
    const autograd::TensorPtr& labels,
    const autograd::TensorPtr& attention_mask = nullptr);

// ============================================================================
// Next Sentence Prediction Loss
// ============================================================================

/**
 * @brief Compute NSP binary classification loss
 * @param logits [B, 1, 1, 2]
 * @param labels [B] (0 or 1)
 * @return Scalar loss
 */
[[nodiscard]] autograd::TensorPtr compute_nsp_loss(
    const autograd::TensorPtr& logits, const autograd::TensorPtr& labels);

// ============================================================================
// Pre-Training Combined Loss
// ============================================================================

/**
 * @brief Compute combined MLM + NSP loss for pre-training
 * @param mlm_logits [B, 1, S, vocab_size]
 * @param nsp_logits [B, 1, 1, 2]
 * @param mlm_labels [B, S] with -100 for non-masked tokens
 * @param nsp_labels [B] (0 or 1)
 * @param mlm_weight Weight for MLM loss (default 1.0)
 * @param nsp_weight Weight for NSP loss (default 1.0)
 * @return Scalar loss (weighted sum of MLM and NSP losses)
 */
[[nodiscard]] autograd::TensorPtr compute_pretraining_loss(
    const autograd::TensorPtr& mlm_logits,
    const autograd::TensorPtr& nsp_logits,
    const autograd::TensorPtr& mlm_labels,
    const autograd::TensorPtr& nsp_labels,
    float mlm_weight = 1.0F,
    float nsp_weight = 1.0F);

}  // namespace ttml::ops::bert_losses
