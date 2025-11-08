// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "autograd/tensor.hpp"
#include "models/base_transformer.hpp"
#include "models/common/transformer_common.hpp"
#include "modules/bert_block.hpp"
#include "modules/dropout_module.hpp"
#include "modules/embedding_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/positional_embeddings.hpp"

namespace ttml::models::bert {

struct BertConfig {
    uint32_t vocab_size = 30522U;
    uint32_t max_sequence_length = 512U;
    uint32_t embedding_dim = 768U;
    uint32_t intermediate_size = 3072U;
    uint32_t num_heads = 12U;
    uint32_t num_blocks = 12U;
    float dropout_prob = 0.1F;
    float layer_norm_eps = 1e-12F;
    bool use_token_type_embeddings = true;
    uint32_t type_vocab_size = 2U;  // For sentence A/B distinction
    common::transformer::RunnerType runner_type = common::transformer::RunnerType::Default;
    bool use_pooler = false;  // For classification tasks
};

class Bert : public BaseTransformer {
private:
    std::shared_ptr<modules::Embedding> m_token_embeddings;
    std::shared_ptr<modules::TrainablePositionalEmbedding> m_position_embeddings;
    std::shared_ptr<modules::Embedding> m_token_type_embeddings;
    std::shared_ptr<modules::LayerNormLayer> m_embedding_norm;
    std::shared_ptr<modules::DropoutLayer> m_embedding_dropout;
    std::vector<std::shared_ptr<modules::BertBlock>> m_blocks;
    std::shared_ptr<modules::LinearLayer> m_pooler;  // Optional pooler for classification

    BertConfig m_config;
    common::transformer::RunnerType m_runner_type;

public:
    explicit Bert(const BertConfig& config);
    virtual ~Bert() = default;

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    // BaseTransformer interface - required override
    // This version assumes input_ids in x and optional attention_mask in mask
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;

    // BERT-specific interface with all three inputs
    // This is the primary implementation with full BERT functionality
    [[nodiscard]] autograd::TensorPtr forward(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr);

    // Convenience method for backward compatibility
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask,
        const autograd::TensorPtr& token_type_ids);

    // Public accessors for testing and introspection
    [[nodiscard]] const BertConfig& get_config() const {
        return m_config;
    }
    [[nodiscard]] bool is_pooler_enabled() const {
        return m_pooler != nullptr;
    }

    // Intermediate outputs structure for layer-by-layer debugging
    struct IntermediateOutputs {
        autograd::TensorPtr embeddings;                            // After embedding layer
        std::vector<autograd::TensorPtr> block_attention_outputs;  // After each block's attention
        std::vector<autograd::TensorPtr> block_outputs;            // After each complete block
        autograd::TensorPtr final_output;                          // Final model output
    };

    // Forward pass with intermediate outputs for debugging/validation
    [[nodiscard]] IntermediateOutputs forward_with_intermediates(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr);

    // Public accessors for isolated layer testing
    [[nodiscard]] autograd::TensorPtr get_embeddings(
        const autograd::TensorPtr& input_ids, const autograd::TensorPtr& token_type_ids = nullptr);

    [[nodiscard]] const std::vector<std::shared_ptr<modules::BertBlock>>& get_blocks() const {
        return m_blocks;
    }

    [[nodiscard]] std::shared_ptr<modules::BertBlock> get_block(size_t index) const {
        if (index >= m_blocks.size()) {
            throw std::out_of_range("Block index out of range");
        }
        return m_blocks[index];
    }

private:
    [[nodiscard]] autograd::TensorPtr process_attention_mask(const autograd::TensorPtr& attention_mask) const;
};

BertConfig read_config(const YAML::Node& config);
YAML::Node write_config(const BertConfig& bert_config);
std::shared_ptr<Bert> create(const BertConfig& config);
std::shared_ptr<Bert> create(const YAML::Node& config);

void load_model_from_safetensors(const std::filesystem::path& path, serialization::NamedParameters& parameters);

// Task-specific head for sequence classification
class BertForSequenceClassification : public Bert {
private:
    std::shared_ptr<modules::DropoutLayer> m_classifier_dropout;
    std::shared_ptr<modules::LinearLayer> m_classifier;
    uint32_t m_num_labels;

public:
    BertForSequenceClassification(const BertConfig& config, uint32_t num_labels, float classifier_dropout = 0.1F);

    // Forward pass returning classification logits (3-parameter version)
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr);

    // BaseTransformer interface - required override (2-parameter version)
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;

    // Forward pass with loss computation for training
    [[nodiscard]] std::tuple<autograd::TensorPtr, autograd::TensorPtr> forward_with_loss(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask,
        const autograd::TensorPtr& token_type_ids,
        const autograd::TensorPtr& labels);

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    [[nodiscard]] uint32_t get_num_labels() const {
        return m_num_labels;
    }
};

[[nodiscard]] std::shared_ptr<BertForSequenceClassification> create_for_sequence_classification(
    const BertConfig& config, uint32_t num_labels, float classifier_dropout = 0.1F);

// Task-specific head for token classification (NER, POS tagging, etc.)
class BertForTokenClassification : public Bert {
private:
    std::shared_ptr<modules::DropoutLayer> m_classifier_dropout;
    std::shared_ptr<modules::LinearLayer> m_classifier;
    uint32_t m_num_labels;

public:
    BertForTokenClassification(const BertConfig& config, uint32_t num_labels, float classifier_dropout = 0.1F);

    // Forward pass returning token-level classification logits (3-parameter version)
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr);

    // BaseTransformer interface - required override (2-parameter version)
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;

    // Forward pass with loss computation for training
    [[nodiscard]] std::tuple<autograd::TensorPtr, autograd::TensorPtr> forward_with_loss(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask,
        const autograd::TensorPtr& token_type_ids,
        const autograd::TensorPtr& labels);

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    [[nodiscard]] uint32_t get_num_labels() const {
        return m_num_labels;
    }
};

[[nodiscard]] std::shared_ptr<BertForTokenClassification> create_for_token_classification(
    const BertConfig& config, uint32_t num_labels, float classifier_dropout = 0.1F);

// Task-specific head for question answering (SQuAD, etc.)
class BertForQuestionAnswering : public Bert {
private:
    std::shared_ptr<modules::LinearLayer> m_qa_outputs;

public:
    explicit BertForQuestionAnswering(const BertConfig& config);

    // Forward pass returning start and end logits (3-parameter version)
    // Returns concatenated [start_logits, end_logits] with shape [batch, 1, 1, seq_len * 2]
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr);

    // BaseTransformer interface - required override (2-parameter version)
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;

    // Forward pass with loss computation for training
    [[nodiscard]] std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> forward_with_loss(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask,
        const autograd::TensorPtr& token_type_ids,
        const autograd::TensorPtr& start_positions,
        const autograd::TensorPtr& end_positions);

    void load_from_safetensors(const std::filesystem::path& model_path) override;
};

[[nodiscard]] std::shared_ptr<BertForQuestionAnswering> create_for_question_answering(const BertConfig& config);

// Task-specific head for masked language modeling (pre-training)
class BertForMaskedLM : public Bert {
private:
    std::shared_ptr<modules::LinearLayer> m_transform_dense;
    std::shared_ptr<modules::LayerNormLayer> m_transform_norm;
    std::shared_ptr<modules::LinearLayer> m_lm_head;

public:
    explicit BertForMaskedLM(const BertConfig& config);

    // Forward pass returning vocabulary predictions (3-parameter version)
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr);

    // BaseTransformer interface - required override (2-parameter version)
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& x, const autograd::TensorPtr& mask) override;

    // Forward pass with loss computation for training
    [[nodiscard]] std::tuple<autograd::TensorPtr, autograd::TensorPtr> forward_with_loss(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask,
        const autograd::TensorPtr& token_type_ids,
        const autograd::TensorPtr& labels);

    void load_from_safetensors(const std::filesystem::path& model_path) override;
};

[[nodiscard]] std::shared_ptr<BertForMaskedLM> create_for_masked_lm(const BertConfig& config);

}  // namespace ttml::models::bert
