// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "models/base_transformer.hpp"
#include "models/bert.hpp"
#include "modules/bert_heads.hpp"

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

struct PreTrainingConfig {
    BertConfig bert_config;
    bool tie_word_embeddings = true;
    float mlm_loss_weight = 1.0F;  // For combined loss
    float nsp_loss_weight = 1.0F;
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
