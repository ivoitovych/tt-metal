// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "models/base_transformer.hpp"
#include "models/bert.hpp"
#include "modules/bert_heads.hpp"

/**
 * @file bert_tasks.hpp
 * @brief Complete BERT task models for fine-tuning and inference
 *
 * This file provides 5 complete BERT task models that combine the base BERT
 * encoder with task-specific heads:
 *
 * 1. BertForSequenceClassification - Sentence classification (sentiment, etc.)
 * 2. BertForTokenClassification - Token tagging (NER, POS tagging)
 * 3. BertForQuestionAnswering - Span extraction (SQuAD, extractive QA)
 * 4. BertForMaskedLM - Masked language modeling (BERT pre-training task)
 * 5. BertForPreTraining - Combined MLM + NSP (full BERT pre-training)
 *
 * Design Principles:
 * - Composition pattern: shared_ptr<Bert> + task head (not inheritance)
 * - All inherit BaseTransformer only (no task base class)
 * - Per-task configs use composition (not inheritance)
 * - External loss computation (trainers own loss semantics)
 * - HuggingFace-compatible weight loading
 *
 * Usage Example:
 * @code
 *   // Create model from config
 *   auto config = bert::SequenceClassificationConfig();
 *   config.bert_config.embedding_dim = 768;
 *   config.num_labels = 2;
 *   auto model = bert::create_for_sequence_classification(config);
 *
 *   // Load pretrained weights
 *   model->load_from_safetensors("bert-base-uncased");
 *
 *   // Forward pass
 *   auto logits = (*model)(input_ids, attention_mask, token_type_ids);
 *
 *   // External loss computation (TTML pattern)
 *   auto loss = ops::bert_losses::compute_sequence_classification_loss(logits, labels);
 *   loss->backward();
 * @endcode
 *
 * @see modules::BertSequenceClassificationHead
 * @see modules::BertTokenClassificationHead
 * @see modules::BertQuestionAnsweringHead
 * @see modules::BertMaskedLMHead
 * @see modules::BertNSPHead
 * @see ops::bert_losses
 */

namespace ttml::models::bert {

// ============================================================================
// Per-Task Configuration Structs
// Using COMPOSITION (not inheritance) - more flexible
// ============================================================================

struct SequenceClassificationConfig {
    BertConfig bert_config;  // Composition (not inheritance)
    uint32_t num_labels = 2;
    float classifier_dropout = 0.1F;
};

struct TokenClassificationConfig {
    BertConfig bert_config;
    uint32_t num_labels;  // Required (e.g., 9 for BIO-NER)
    float classifier_dropout = 0.1F;
};

struct QuestionAnsweringConfig {
    BertConfig bert_config;
    // No additional task-specific fields
};

struct MaskedLMConfig {
    BertConfig bert_config;
    bool tie_word_embeddings = true;
};

/**
 * @brief Configuration for BERT pre-training (MLM + NSP)
 *
 * Combines Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
 * for full BERT-style pre-training on unlabeled text.
 *
 * @note This requires the BertOutput helper for proper dual-output support.
 *       See BertForPreTraining::forward_pretraining() for details.
 */
struct PreTrainingConfig {
    BertConfig bert_config;
    bool tie_word_embeddings = true;  ///< Tie MLM decoder with input embeddings
    float mlm_loss_weight = 1.0F;     ///< Weight for MLM loss in combined loss
    float nsp_loss_weight = 1.0F;     ///< Weight for NSP loss in combined loss
};

// ============================================================================
// Task Models - Clean Composition Pattern (GPT-2/Llama Style)
// All inherit BaseTransformer ONLY (no task base class)
// ============================================================================

// ============================================================================
// 1. Sequence Classification
// ============================================================================

class BertForSequenceClassification : public BaseTransformer {
private:
    std::shared_ptr<Bert> m_bert;
    std::shared_ptr<modules::BertSequenceClassificationHead> m_head;
    SequenceClassificationConfig m_config;

public:
    explicit BertForSequenceClassification(const SequenceClassificationConfig& config);
    virtual ~BertForSequenceClassification() = default;

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr) override;

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    // Accessors
    [[nodiscard]] const SequenceClassificationConfig& get_config() const {
        return m_config;
    }
    [[nodiscard]] uint32_t get_num_labels() const {
        return m_config.num_labels;
    }
    [[nodiscard]] std::shared_ptr<Bert> get_bert() const {
        return m_bert;
    }
};

// ============================================================================
// 2. Token Classification
// ============================================================================

class BertForTokenClassification : public BaseTransformer {
private:
    std::shared_ptr<Bert> m_bert;
    std::shared_ptr<modules::BertTokenClassificationHead> m_head;
    TokenClassificationConfig m_config;

public:
    explicit BertForTokenClassification(const TokenClassificationConfig& config);
    virtual ~BertForTokenClassification() = default;

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr) override;

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    [[nodiscard]] const TokenClassificationConfig& get_config() const {
        return m_config;
    }
    [[nodiscard]] uint32_t get_num_labels() const {
        return m_config.num_labels;
    }
};

// ============================================================================
// 3. Question Answering
// ============================================================================

class BertForQuestionAnswering : public BaseTransformer {
private:
    std::shared_ptr<Bert> m_bert;
    std::shared_ptr<modules::BertQuestionAnsweringHead> m_head;
    QuestionAnsweringConfig m_config;

public:
    explicit BertForQuestionAnswering(const QuestionAnsweringConfig& config);
    virtual ~BertForQuestionAnswering() = default;

    // Returns combined [start; end] logits [B, 1, S, 2]
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr) override;

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    [[nodiscard]] const QuestionAnsweringConfig& get_config() const {
        return m_config;
    }
};

// ============================================================================
// 4. Masked Language Modeling
// ============================================================================

class BertForMaskedLM : public BaseTransformer {
private:
    std::shared_ptr<Bert> m_bert;
    std::shared_ptr<modules::BertMaskedLMHead> m_head;
    MaskedLMConfig m_config;

public:
    explicit BertForMaskedLM(const MaskedLMConfig& config);
    virtual ~BertForMaskedLM() = default;

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr) override;

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    [[nodiscard]] const MaskedLMConfig& get_config() const {
        return m_config;
    }
    [[nodiscard]] bool has_tied_embeddings() const {
        return m_head->has_tied_weights();
    }
};

// ============================================================================
// 5. Pre-Training (MLM + NSP Combined)
// CRITICAL FIX: Uses BertOutput helper to properly support both heads
// ============================================================================

/**
 * @brief BERT model for pre-training with Masked LM and Next Sentence Prediction
 *
 * This model combines two pre-training objectives:
 * 1. Masked Language Modeling (MLM) - predicts masked tokens
 * 2. Next Sentence Prediction (NSP) - predicts if sentence B follows sentence A
 *
 * CRITICAL BUG FIX:
 * This implementation uses the BertOutput helper struct to properly get both
 * sequence output (for MLM) and pooled output (for NSP) in a single forward pass.
 * Previous implementations had placeholder code or semantic errors when trying
 * to support dual outputs.
 *
 * Usage:
 * @code
 *   auto config = bert::PreTrainingConfig();
 *   config.bert_config.embedding_dim = 768;
 *   config.tie_word_embeddings = true;  // Standard BERT practice
 *   auto model = bert::create_for_pretraining(config);
 *
 *   // Forward pass returns both outputs
 *   auto output = model->forward_pretraining(input_ids, attention_mask, token_type_ids);
 *   auto mlm_logits = output.mlm_logits;  // [B, 1, S, vocab_size]
 *   auto nsp_logits = output.nsp_logits;  // [B, 1, 1, 2]
 *
 *   // Compute combined loss
 *   auto loss = ops::bert_losses::compute_pretraining_loss(
 *       mlm_logits, nsp_logits, mlm_labels, nsp_labels,
 *       config.mlm_loss_weight, config.nsp_loss_weight);
 * @endcode
 *
 * @see BertOutput
 * @see Bert::forward_structured()
 * @see ops::bert_losses::compute_pretraining_loss()
 */
class BertForPreTraining : public BaseTransformer {
private:
    std::shared_ptr<Bert> m_bert;
    std::shared_ptr<modules::BertMaskedLMHead> m_mlm_head;
    std::shared_ptr<modules::BertNSPHead> m_nsp_head;
    PreTrainingConfig m_config;

public:
    explicit BertForPreTraining(const PreTrainingConfig& config);
    virtual ~BertForPreTraining() = default;

    // Returns MLM logits (for BaseTransformer compatibility)
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr) override;

    // Pre-training specific: returns both MLM and NSP logits
    struct PreTrainingOutput {
        autograd::TensorPtr mlm_logits;  // [B, 1, S, vocab_size]
        autograd::TensorPtr nsp_logits;  // [B, 1, 1, 2]
    };

    [[nodiscard]] PreTrainingOutput forward_pretraining(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr);

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    [[nodiscard]] const PreTrainingConfig& get_config() const {
        return m_config;
    }
};

// ============================================================================
// Factory Functions
// ============================================================================

[[nodiscard]] std::shared_ptr<BertForSequenceClassification> create_for_sequence_classification(
    const SequenceClassificationConfig& config);

[[nodiscard]] std::shared_ptr<BertForTokenClassification> create_for_token_classification(
    const TokenClassificationConfig& config);

[[nodiscard]] std::shared_ptr<BertForQuestionAnswering> create_for_question_answering(
    const QuestionAnsweringConfig& config);

[[nodiscard]] std::shared_ptr<BertForMaskedLM> create_for_masked_lm(const MaskedLMConfig& config);

[[nodiscard]] std::shared_ptr<BertForPreTraining> create_for_pretraining(const PreTrainingConfig& config);

// ============================================================================
// YAML Config Readers
// ============================================================================

[[nodiscard]] SequenceClassificationConfig read_sequence_classification_config(const YAML::Node& config);

[[nodiscard]] TokenClassificationConfig read_token_classification_config(const YAML::Node& config);

[[nodiscard]] QuestionAnsweringConfig read_question_answering_config(const YAML::Node& config);

[[nodiscard]] MaskedLMConfig read_masked_lm_config(const YAML::Node& config);

[[nodiscard]] PreTrainingConfig read_pretraining_config(const YAML::Node& config);

}  // namespace ttml::models::bert
