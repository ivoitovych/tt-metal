// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "bert_tasks.hpp"

#include "autograd/auto_context.hpp"
#include "models/bert_tasks_serialization.hpp"
#include "models/common/transformer_common.hpp"

namespace ttml::models::bert {

// ============================================================================
// BertForSequenceClassification
// ============================================================================

BertForSequenceClassification::BertForSequenceClassification(const SequenceClassificationConfig& config) :
    m_config(config) {
    // Force pooler for classification
    auto bert_config = config.bert_config;
    bert_config.use_pooler = true;  // CRITICAL
    m_bert = std::make_shared<Bert>(bert_config);

    m_head = std::make_shared<modules::BertSequenceClassificationHead>(
        config.bert_config.embedding_dim, config.num_labels, config.classifier_dropout);

    // Register modules for parameter tracking
    create_name("bert_for_sequence_classification");
    register_module(m_bert, "bert");
    register_module(m_head, "classifier");

    fmt::print("BertForSequenceClassification created: {} labels\n", config.num_labels);
}

autograd::TensorPtr BertForSequenceClassification::operator()(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids) {
    // Get pooled output from BERT
    auto pooled = (*m_bert)(input_ids, attention_mask, token_type_ids);

    // Apply head with RunnerType support
    if (m_config.bert_config.runner_type == common::transformer::RunnerType::MemoryEfficient) {
        auto forward_fn = [this](const autograd::TensorPtr& in, const autograd::TensorPtr& /* mask */) {
            return (*m_head)(in);
        };
        return common::transformer::memory_efficient_runner(forward_fn, pooled, nullptr);
    }

    return (*m_head)(pooled);
}

void BertForSequenceClassification::load_from_safetensors(const std::filesystem::path& model_path) {
    fmt::print("Loading BertForSequenceClassification from: {}\n", model_path.string());

    // Load base BERT
    m_bert->load_from_safetensors(model_path);

    // Load head weights if available
    // HF format: "classifier.weight", "classifier.bias"
    // TTML format: "bert_for_sequence_classification/classifier/classifier/weight", etc.
    auto parameters = this->parameters();

    std::map<std::string, std::string> weight_mapping = {
        {"classifier.weight", "bert_for_sequence_classification/classifier/classifier/weight"},
        {"classifier.bias", "bert_for_sequence_classification/classifier/classifier/bias"}};

    load_task_head_weights(model_path, parameters, weight_mapping, "SequenceClassification");

    fmt::print("Model loaded successfully\n");
}

// ============================================================================
// BertForTokenClassification
// ============================================================================

BertForTokenClassification::BertForTokenClassification(const TokenClassificationConfig& config) : m_config(config) {
    // Token classification doesn't need pooler
    auto bert_config = config.bert_config;
    bert_config.use_pooler = false;
    m_bert = std::make_shared<Bert>(bert_config);

    m_head = std::make_shared<modules::BertTokenClassificationHead>(
        config.bert_config.embedding_dim, config.num_labels, config.classifier_dropout);

    create_name("bert_for_token_classification");
    register_module(m_bert, "bert");
    register_module(m_head, "classifier");

    fmt::print("BertForTokenClassification created: {} labels\n", config.num_labels);
}

autograd::TensorPtr BertForTokenClassification::operator()(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids) {
    // Get sequence output from BERT (all token representations)
    auto sequence_output = (*m_bert)(input_ids, attention_mask, token_type_ids);

    // Apply head with RunnerType support
    if (m_config.bert_config.runner_type == common::transformer::RunnerType::MemoryEfficient) {
        auto forward_fn = [this](const autograd::TensorPtr& in, const autograd::TensorPtr& /* mask */) {
            return (*m_head)(in);
        };
        return common::transformer::memory_efficient_runner(forward_fn, sequence_output, nullptr);
    }

    return (*m_head)(sequence_output);
}

void BertForTokenClassification::load_from_safetensors(const std::filesystem::path& model_path) {
    fmt::print("Loading BertForTokenClassification from: {}\n", model_path.string());

    m_bert->load_from_safetensors(model_path);

    // Load head weights if available
    auto parameters = this->parameters();

    std::map<std::string, std::string> weight_mapping = {
        {"classifier.weight", "bert_for_token_classification/classifier/classifier/weight"},
        {"classifier.bias", "bert_for_token_classification/classifier/classifier/bias"}};

    load_task_head_weights(model_path, parameters, weight_mapping, "TokenClassification");

    fmt::print("Model loaded successfully\n");
}

// ============================================================================
// BertForQuestionAnswering
// ============================================================================

BertForQuestionAnswering::BertForQuestionAnswering(const QuestionAnsweringConfig& config) : m_config(config) {
    // QA doesn't need pooler
    auto bert_config = config.bert_config;
    bert_config.use_pooler = false;
    m_bert = std::make_shared<Bert>(bert_config);

    m_head = std::make_shared<modules::BertQuestionAnsweringHead>(config.bert_config.embedding_dim);

    create_name("bert_for_question_answering");
    register_module(m_bert, "bert");
    register_module(m_head, "qa_outputs");

    fmt::print("BertForQuestionAnswering created\n");
}

autograd::TensorPtr BertForQuestionAnswering::operator()(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids) {
    // Get sequence output from BERT
    auto sequence_output = (*m_bert)(input_ids, attention_mask, token_type_ids);

    // Apply head with RunnerType support
    if (m_config.bert_config.runner_type == common::transformer::RunnerType::MemoryEfficient) {
        auto forward_fn = [this](const autograd::TensorPtr& in, const autograd::TensorPtr& /* mask */) {
            return (*m_head)(in);
        };
        return common::transformer::memory_efficient_runner(forward_fn, sequence_output, nullptr);
    }

    return (*m_head)(sequence_output);
}

void BertForQuestionAnswering::load_from_safetensors(const std::filesystem::path& model_path) {
    fmt::print("Loading BertForQuestionAnswering from: {}\n", model_path.string());

    m_bert->load_from_safetensors(model_path);

    // Load head weights if available
    auto parameters = this->parameters();

    std::map<std::string, std::string> weight_mapping = {
        {"qa_outputs.weight", "bert_for_question_answering/qa_outputs/qa_outputs/weight"},
        {"qa_outputs.bias", "bert_for_question_answering/qa_outputs/qa_outputs/bias"}};

    load_task_head_weights(model_path, parameters, weight_mapping, "QuestionAnswering");

    fmt::print("Model loaded successfully\n");
}

// ============================================================================
// BertForMaskedLM
// ============================================================================

BertForMaskedLM::BertForMaskedLM(const MaskedLMConfig& config) : m_config(config) {
    // MLM doesn't need pooler
    auto bert_config = config.bert_config;
    bert_config.use_pooler = false;
    m_bert = std::make_shared<Bert>(bert_config);

    m_head = std::make_shared<modules::BertMaskedLMHead>(
        config.bert_config.embedding_dim, config.bert_config.vocab_size, config.bert_config.layer_norm_eps);

    create_name("bert_for_masked_lm");
    register_module(m_bert, "bert");
    register_module(m_head, "cls.predictions");  // HF naming

    // Tie weights if requested
    if (config.tie_word_embeddings) {
        auto params = m_bert->parameters();
        auto embeddings_weight = params["bert/token_embeddings/weight"];
        m_head->tie_decoder_weights(embeddings_weight);
    }

    fmt::print("BertForMaskedLM created (vocab_size={})\n", config.bert_config.vocab_size);
}

autograd::TensorPtr BertForMaskedLM::operator()(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids) {
    // Get sequence output from BERT
    auto sequence_output = (*m_bert)(input_ids, attention_mask, token_type_ids);

    // Apply head with RunnerType support
    if (m_config.bert_config.runner_type == common::transformer::RunnerType::MemoryEfficient) {
        auto forward_fn = [this](const autograd::TensorPtr& in, const autograd::TensorPtr& /* mask */) {
            return (*m_head)(in);
        };
        return common::transformer::memory_efficient_runner(forward_fn, sequence_output, nullptr);
    }

    return (*m_head)(sequence_output);
}

void BertForMaskedLM::load_from_safetensors(const std::filesystem::path& model_path) {
    fmt::print("Loading BertForMaskedLM from: {}\n", model_path.string());

    m_bert->load_from_safetensors(model_path);

    // Load MLM head weights if available
    // HF uses "cls.predictions" prefix for the MLM head
    auto parameters = this->parameters();

    std::map<std::string, std::string> weight_mapping = {
        {"cls.predictions.transform.dense.weight", "bert_for_masked_lm/cls.predictions/transform.dense/weight"},
        {"cls.predictions.transform.dense.bias", "bert_for_masked_lm/cls.predictions/transform.dense/bias"},
        {"cls.predictions.transform.LayerNorm.weight", "bert_for_masked_lm/cls.predictions/transform.LayerNorm/weight"},
        {"cls.predictions.transform.LayerNorm.bias", "bert_for_masked_lm/cls.predictions/transform.LayerNorm/bias"},
        {"cls.predictions.decoder.weight", "bert_for_masked_lm/cls.predictions/decoder/weight"},
        {"cls.predictions.bias",  // HF stores decoder bias separately
         "bert_for_masked_lm/cls.predictions/decoder/bias"}};

    load_task_head_weights(model_path, parameters, weight_mapping, "MaskedLM");

    fmt::print("Model loaded successfully\n");
}

// ============================================================================
// BertForPreTraining - CRITICAL FIX
// Uses BertOutput helper to properly support MLM + NSP
// ============================================================================

BertForPreTraining::BertForPreTraining(const PreTrainingConfig& config) : m_config(config) {
    // Need pooler for NSP
    auto bert_config = config.bert_config;
    bert_config.use_pooler = true;  // CRITICAL for NSP
    m_bert = std::make_shared<Bert>(bert_config);

    m_mlm_head = std::make_shared<modules::BertMaskedLMHead>(
        config.bert_config.embedding_dim, config.bert_config.vocab_size, config.bert_config.layer_norm_eps);

    m_nsp_head = std::make_shared<modules::BertNSPHead>(config.bert_config.embedding_dim);

    create_name("bert_for_pretraining");
    register_module(m_bert, "bert");
    register_module(m_mlm_head, "cls.predictions");  // HF naming
    register_module(m_nsp_head, "cls.seq_relationship");

    // Tie weights if requested
    if (config.tie_word_embeddings) {
        auto params = m_bert->parameters();
        auto embeddings_weight = params["bert/token_embeddings/weight"];
        m_mlm_head->tie_decoder_weights(embeddings_weight);
    }

    fmt::print("BertForPreTraining created (MLM + NSP)\n");
}

autograd::TensorPtr BertForPreTraining::operator()(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids) {
    // Return MLM logits for BaseTransformer compatibility
    auto output = forward_pretraining(input_ids, attention_mask, token_type_ids);
    return output.mlm_logits;
}

BertForPreTraining::PreTrainingOutput BertForPreTraining::forward_pretraining(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids) {
    // ========================================================================
    // CRITICAL FIX: Use BertOutput helper (Review 4)
    // This properly gets both sequence output AND pooled output
    // NO placeholder code, NO semantic errors
    // ========================================================================
    auto bert_output = m_bert->forward_structured(input_ids, attention_mask, token_type_ids);

    // MLM logits from full sequence
    auto mlm_logits = (*m_mlm_head)(bert_output.last_hidden_state);  // [B, 1, S, vocab]

    // NSP logits from pooled CLS token
    auto nsp_logits = (*m_nsp_head)(bert_output.pooler_output);  // [B, 1, 1, 2]

    return PreTrainingOutput{.mlm_logits = mlm_logits, .nsp_logits = nsp_logits};
}

void BertForPreTraining::load_from_safetensors(const std::filesystem::path& model_path) {
    fmt::print("Loading BertForPreTraining from: {}\n", model_path.string());

    m_bert->load_from_safetensors(model_path);

    // Load both MLM and NSP head weights if available
    auto parameters = this->parameters();

    std::map<std::string, std::string> weight_mapping = {
        // MLM head (same as BertForMaskedLM)
        {"cls.predictions.transform.dense.weight", "bert_for_pretraining/cls.predictions/transform.dense/weight"},
        {"cls.predictions.transform.dense.bias", "bert_for_pretraining/cls.predictions/transform.dense/bias"},
        {"cls.predictions.transform.LayerNorm.weight",
         "bert_for_pretraining/cls.predictions/transform.LayerNorm/weight"},
        {"cls.predictions.transform.LayerNorm.bias", "bert_for_pretraining/cls.predictions/transform.LayerNorm/bias"},
        {"cls.predictions.decoder.weight", "bert_for_pretraining/cls.predictions/decoder/weight"},
        {"cls.predictions.bias", "bert_for_pretraining/cls.predictions/decoder/bias"},
        // NSP head
        {"cls.seq_relationship.weight", "bert_for_pretraining/cls.seq_relationship/seq_relationship/weight"},
        {"cls.seq_relationship.bias", "bert_for_pretraining/cls.seq_relationship/seq_relationship/bias"}};

    load_task_head_weights(model_path, parameters, weight_mapping, "PreTraining");

    fmt::print("PreTraining model loaded successfully\n");
}

// ============================================================================
// Factory Functions
// ============================================================================

std::shared_ptr<BertForSequenceClassification> create_for_sequence_classification(
    const SequenceClassificationConfig& config) {
    return std::make_shared<BertForSequenceClassification>(config);
}

std::shared_ptr<BertForTokenClassification> create_for_token_classification(const TokenClassificationConfig& config) {
    return std::make_shared<BertForTokenClassification>(config);
}

std::shared_ptr<BertForQuestionAnswering> create_for_question_answering(const QuestionAnsweringConfig& config) {
    return std::make_shared<BertForQuestionAnswering>(config);
}

std::shared_ptr<BertForMaskedLM> create_for_masked_lm(const MaskedLMConfig& config) {
    return std::make_shared<BertForMaskedLM>(config);
}

std::shared_ptr<BertForPreTraining> create_for_pretraining(const PreTrainingConfig& config) {
    return std::make_shared<BertForPreTraining>(config);
}

// ============================================================================
// YAML Config Readers
// ============================================================================

SequenceClassificationConfig read_sequence_classification_config(const YAML::Node& config) {
    SequenceClassificationConfig sc_config;
    sc_config.bert_config = read_config(config["bert_config"]);
    sc_config.num_labels = config["num_labels"].as<uint32_t>(2);
    sc_config.classifier_dropout = config["classifier_dropout"].as<float>(0.1F);
    return sc_config;
}

TokenClassificationConfig read_token_classification_config(const YAML::Node& config) {
    TokenClassificationConfig tc_config;
    tc_config.bert_config = read_config(config["bert_config"]);
    tc_config.num_labels = config["num_labels"].as<uint32_t>();  // Required
    tc_config.classifier_dropout = config["classifier_dropout"].as<float>(0.1F);
    return tc_config;
}

QuestionAnsweringConfig read_question_answering_config(const YAML::Node& config) {
    QuestionAnsweringConfig qa_config;
    qa_config.bert_config = read_config(config["bert_config"]);
    return qa_config;
}

MaskedLMConfig read_masked_lm_config(const YAML::Node& config) {
    MaskedLMConfig mlm_config;
    mlm_config.bert_config = read_config(config["bert_config"]);
    mlm_config.tie_word_embeddings = config["tie_word_embeddings"].as<bool>(true);
    return mlm_config;
}

PreTrainingConfig read_pretraining_config(const YAML::Node& config) {
    PreTrainingConfig pt_config;
    pt_config.bert_config = read_config(config["bert_config"]);
    pt_config.tie_word_embeddings = config["tie_word_embeddings"].as<bool>(true);
    pt_config.mlm_loss_weight = config["mlm_loss_weight"].as<float>(1.0F);
    pt_config.nsp_loss_weight = config["nsp_loss_weight"].as<float>(1.0F);
    return pt_config;
}

}  // namespace ttml::models::bert
