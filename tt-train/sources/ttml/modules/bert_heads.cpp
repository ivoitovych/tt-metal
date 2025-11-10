// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "bert_heads.hpp"

#include "init/tensor_initializers.hpp"
#include "models/common/transformer_common.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

// ============================================================================
// BertSequenceClassificationHead
// HF-exact: dropout → linear (NO tanh, NO dense)
// ============================================================================

BertSequenceClassificationHead::BertSequenceClassificationHead(
    uint32_t hidden_size, uint32_t num_labels, float dropout_prob) :
    m_num_labels(num_labels) {
    // Optional dropout (only if dropout_prob > 0)
    if (dropout_prob > 0.0F) {
        m_dropout = std::make_shared<DropoutLayer>(dropout_prob);
        register_module(m_dropout, "dropout");
    }

    m_classifier = std::make_shared<LinearLayer>(hidden_size, num_labels);
    register_module(m_classifier, "classifier");

    create_name("sequence_classification_head");

    // Initialize with GPT-2 style (proven effective in TTML)
    models::common::transformer::initialize_weights_gpt2(*this);
}

autograd::TensorPtr BertSequenceClassificationHead::operator()(const autograd::TensorPtr& pooled_output) {
    // HF-exact: dropout → linear
    auto x = pooled_output;
    if (m_dropout) {
        x = (*m_dropout)(x);
    }
    return (*m_classifier)(x);  // [B, 1, 1, num_labels]
}

// ============================================================================
// BertTokenClassificationHead
// ============================================================================

BertTokenClassificationHead::BertTokenClassificationHead(
    uint32_t hidden_size, uint32_t num_labels, float dropout_prob) :
    m_num_labels(num_labels) {
    m_dropout = std::make_shared<DropoutLayer>(dropout_prob);
    m_classifier = std::make_shared<LinearLayer>(hidden_size, num_labels);

    create_name("token_classification_head");
    register_module(m_dropout, "dropout");
    register_module(m_classifier, "classifier");

    models::common::transformer::initialize_weights_gpt2(*this);
}

autograd::TensorPtr BertTokenClassificationHead::operator()(const autograd::TensorPtr& sequence_output) {
    auto x = (*m_dropout)(sequence_output);
    return (*m_classifier)(x);  // [B, 1, S, num_labels]
}

// ============================================================================
// BertQuestionAnsweringHead
// ============================================================================

BertQuestionAnsweringHead::BertQuestionAnsweringHead(uint32_t hidden_size) {
    m_qa_outputs = std::make_shared<LinearLayer>(hidden_size, 2);

    create_name("question_answering_head");
    register_module(m_qa_outputs, "qa_outputs");

    models::common::transformer::initialize_weights_gpt2(*this);
}

autograd::TensorPtr BertQuestionAnsweringHead::operator()(const autograd::TensorPtr& sequence_output) {
    return (*m_qa_outputs)(sequence_output);  // [B, 1, S, 2]
}

BertQuestionAnsweringHead::QALogits BertQuestionAnsweringHead::split_logits(
    const autograd::TensorPtr& combined_logits) {
    auto shape = combined_logits->get_shape();
    auto batch_size = shape[0];
    auto seq_len = shape[2];

    // Split last dimension: [:, :, :, 0] = start, [:, :, :, 1] = end
    ttnn::SmallVector<uint32_t> stride = {1, 1, 1, 1};

    auto start_logits = ttnn::slice(
        combined_logits->get_value(),
        ttnn::SmallVector<uint32_t>{0, 0, 0, 0},
        ttnn::SmallVector<uint32_t>{batch_size, 1, seq_len, 1},
        stride);

    auto end_logits = ttnn::slice(
        combined_logits->get_value(),
        ttnn::SmallVector<uint32_t>{0, 0, 0, 1},
        ttnn::SmallVector<uint32_t>{batch_size, 1, seq_len, 2},
        stride);

    return QALogits{
        .start_logits = autograd::create_tensor(start_logits), .end_logits = autograd::create_tensor(end_logits)};
}

// ============================================================================
// BertMaskedLMHead
// HF-exact: dense → GELU → LayerNorm → decoder (tied)
// ============================================================================

BertMaskedLMHead::BertMaskedLMHead(uint32_t hidden_size, uint32_t vocab_size, float layer_norm_eps) :
    m_weights_tied(false), m_vocab_size(vocab_size) {
    m_dense = std::make_shared<LinearLayer>(hidden_size, hidden_size);
    m_layer_norm = std::make_shared<LayerNormLayer>(hidden_size, layer_norm_eps, false, false);
    m_decoder = std::make_shared<LinearLayer>(hidden_size, vocab_size, false);  // No bias

    create_name("masked_lm_head");
    register_module(m_dense, "transform.dense");
    register_module(m_layer_norm, "transform.LayerNorm");
    register_module(m_decoder, "decoder");

    models::common::transformer::initialize_weights_gpt2(*this);
}

autograd::TensorPtr BertMaskedLMHead::operator()(const autograd::TensorPtr& sequence_output) {
    // HF-exact: dense → GELU → LayerNorm → decoder
    auto hidden = (*m_dense)(sequence_output);
    hidden = ops::gelu(hidden);
    hidden = (*m_layer_norm)(hidden);
    return (*m_decoder)(hidden);  // [B, 1, S, vocab_size]
}

void BertMaskedLMHead::tie_decoder_weights(const autograd::TensorPtr& embeddings_weight) {
    // Share weights between decoder and input embeddings
    override_tensor(embeddings_weight, "decoder/weight");
    m_weights_tied = true;

    fmt::print("MLM head: Tied decoder weights with input embeddings\n");
}

// ============================================================================
// BertNSPHead
// ============================================================================

BertNSPHead::BertNSPHead(uint32_t hidden_size) {
    m_seq_relationship = std::make_shared<LinearLayer>(hidden_size, 2);

    create_name("nsp_head");
    register_module(m_seq_relationship, "seq_relationship");

    models::common::transformer::initialize_weights_gpt2(*this);
}

autograd::TensorPtr BertNSPHead::operator()(const autograd::TensorPtr& pooled_output) {
    return (*m_seq_relationship)(pooled_output);  // [B, 1, 1, 2]
}

}  // namespace ttml::modules
