// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

/**
 * @file bert_heads.hpp
 * @brief Task-specific head modules for BERT models
 *
 * This file provides 5 task-specific head modules that can be composed with
 * the base BERT encoder to create complete task models:
 *
 * 1. BertSequenceClassificationHead - For sentence-level classification
 * 2. BertTokenClassificationHead - For token-level tagging (NER, POS)
 * 3. BertQuestionAnsweringHead - For span extraction (SQuAD)
 * 4. BertMaskedLMHead - For masked language modeling
 * 5. BertNSPHead - For next sentence prediction
 *
 * Design Principles:
 * - NO abstract base class (simplest possible, GPT-2 pattern)
 * - NO loss methods (external only - trainers own loss computation)
 * - HF-exact architectures (validated against HuggingFace)
 * - Pure tensor transformers (ModuleBase subclasses)
 *
 * @see models::bert::BertForSequenceClassification
 * @see models::bert::BertForTokenClassification
 * @see models::bert::BertForQuestionAnswering
 * @see models::bert::BertForMaskedLM
 * @see models::bert::BertForPreTraining
 */

// ============================================================================
// 1. Sequence Classification Head
// ============================================================================
// HF Architecture (validated): dropout → linear
// NO tanh, NO dense layer (explicit from reviews)
// Input: [B, 1, 1, E] (pooled)
// Output: [B, 1, 1, num_labels]

class BertSequenceClassificationHead : public ModuleBase {
private:
    std::shared_ptr<DropoutLayer> m_dropout;  // Optional if dropout_prob > 0
    std::shared_ptr<LinearLayer> m_classifier;
    uint32_t m_num_labels;

public:
    /**
     * @brief Sequence classification head
     * HF-exact: dropout → linear (NO tanh, NO dense)
     * @param hidden_size BERT embedding dimension
     * @param num_labels Number of classification labels
     * @param dropout_prob Dropout probability (0.0 = no dropout)
     */
    BertSequenceClassificationHead(uint32_t hidden_size, uint32_t num_labels, float dropout_prob = 0.1F);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& pooled_output) override;

    [[nodiscard]] uint32_t get_num_labels() const {
        return m_num_labels;
    }
};

// ============================================================================
// 2. Token Classification Head (NER, POS tagging)
// ============================================================================
// HF Architecture: dropout → linear
// Input: [B, 1, S, E] (sequence)
// Output: [B, 1, S, num_labels]

class BertTokenClassificationHead : public ModuleBase {
private:
    std::shared_ptr<DropoutLayer> m_dropout;
    std::shared_ptr<LinearLayer> m_classifier;
    uint32_t m_num_labels;

public:
    BertTokenClassificationHead(uint32_t hidden_size, uint32_t num_labels, float dropout_prob = 0.1F);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& sequence_output) override;

    [[nodiscard]] uint32_t get_num_labels() const {
        return m_num_labels;
    }
};

// ============================================================================
// 3. Question Answering Head
// ============================================================================
// HF Architecture: linear → 2 (start/end positions)
// Input: [B, 1, S, E]
// Output: [B, 1, S, 2]

class BertQuestionAnsweringHead : public ModuleBase {
private:
    std::shared_ptr<LinearLayer> m_qa_outputs;

public:
    explicit BertQuestionAnsweringHead(uint32_t hidden_size);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& sequence_output) override;

    // Utility to split combined output
    struct QALogits {
        autograd::TensorPtr start_logits;  // [B, 1, S, 1]
        autograd::TensorPtr end_logits;    // [B, 1, S, 1]
    };

    [[nodiscard]] static QALogits split_logits(const autograd::TensorPtr& combined_logits);
};

// ============================================================================
// 4. Masked Language Modeling Head
// ============================================================================
// HF Architecture: dense → GELU → LayerNorm → decoder (tied with embeddings)
// Input: [B, 1, S, E]
// Output: [B, 1, S, vocab_size]

class BertMaskedLMHead : public ModuleBase {
private:
    std::shared_ptr<LinearLayer> m_dense;
    std::shared_ptr<LayerNormLayer> m_layer_norm;
    std::shared_ptr<LinearLayer> m_decoder;
    bool m_weights_tied;
    uint32_t m_vocab_size;

public:
    BertMaskedLMHead(uint32_t hidden_size, uint32_t vocab_size, float layer_norm_eps = 1e-12F);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& sequence_output) override;

    /**
     * @brief Tie decoder weights with input embeddings (standard BERT practice)
     * @param embeddings_weight Token embeddings [vocab_size, hidden_size]
     */
    void tie_decoder_weights(const autograd::TensorPtr& embeddings_weight);

    [[nodiscard]] bool has_tied_weights() const {
        return m_weights_tied;
    }
    [[nodiscard]] uint32_t get_vocab_size() const {
        return m_vocab_size;
    }
};

// ============================================================================
// 5. Next Sentence Prediction Head
// ============================================================================
// HF Architecture: linear → 2
// Input: [B, 1, 1, E] (pooled)
// Output: [B, 1, 1, 2]

class BertNSPHead : public ModuleBase {
private:
    std::shared_ptr<LinearLayer> m_seq_relationship;

public:
    explicit BertNSPHead(uint32_t hidden_size);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& pooled_output) override;
};

}  // namespace ttml::modules
