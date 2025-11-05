# BERT Task-Specific Heads - Implementation Plan

**Date**: 2025-11-05
**Branch**: ivoitovych/bert-model-for-ttml-completeness-implementation
**Status**: Ready to implement
**Context**: Phase 0 of BERT Action Plan - Essential task-specific heads

---

## Executive Summary

This document outlines the **immediate implementation plan** for adding task-specific heads to BERT, based on investigation of existing patterns in GPT-2, LLaMA, and BERT codebases.

**Key Finding**: We are **NOT in terra incognita** - BERT already has 80% of the infrastructure we need (pooler pattern, module composition, weight loading). We're extending existing patterns, not building from scratch.

**Timeline**: 9-12 days for 3 heads + examples
**Impact**: Enables 80% of common BERT use cases (classification, NER, QA)

---

## Investigation Findings: What Exists in TTML

### 1. GPT-2 & LLaMA: Language Modeling Only

**Pattern**: Base model + final linear layer for next-token prediction

```cpp
// From gpt2.cpp / llama.cpp
fc = std::make_shared<LinearLayer>(embedding_dim, vocab_size, /* bias */ false);

// Forward pass:
embeddings → blocks → layer_norm → fc → logits
```

**Status**:
- ✅ Language modeling head (vocab prediction)
- ❌ No task-specific heads (classification, NER, QA)
- ❌ No BertFor*/GPT2For*/LlamaFor* variants

**Conclusion**: All TTML models lack task-specific heads (systematic gap, not BERT-specific)

---

### 2. BERT: Has Pooler Infrastructure! 🎯

**This is exactly what we need for SequenceClassification:**

```cpp
// From bert.cpp (lines 118-120)
if (config.use_pooler) {
    m_pooler = std::make_shared<modules::LinearLayer>(embedding_dim, embedding_dim);
}

// From bert.cpp (lines 237-258) - Forward pass with pooler
if (m_pooler) {
    // 1. Extract [CLS] token representation (first token in sequence)
    auto hidden_shape = hidden_states->get_shape();
    auto batch_size = hidden_shape[0];
    auto embedding_dim = hidden_shape[3];

    ttnn::SmallVector<uint32_t> start_indices = {0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> end_indices = {batch_size, 1, 1, embedding_dim};
    ttnn::SmallVector<uint32_t> stride = {1, 1, 1, 1};

    auto cls_token = ttnn::slice(hidden_states->get_value(), start_indices, end_indices, stride);

    // 2. Apply linear transformation
    auto pooled_output = autograd::create_tensor(cls_token);
    pooled_output = (*m_pooler)(pooled_output);

    // 3. Apply tanh activation
    pooled_output = ops::tanh(pooled_output);

    return pooled_output;
}
```

**Weight Loading** (bert.cpp:450-465):
```cpp
// Pooler weights loaded from HuggingFace safetensors
else if (info.name == "bert.pooler.dense.weight" || info.name == "pooler.dense.weight") {
    if (parameters.find("bert/pooler/weight") != parameters.end()) {
        auto param = get_parameter("bert/pooler/weight");
        param->set_value(core::from_vector(
            float_vec, param->get_value().logical_shape(), param->get_value().device()));
        fmt::print("  Loaded pooler dense weight\n");
    }
}
```

**Status**:
- ✅ CLS token extraction pattern (slice operation)
- ✅ Pooler linear layer pattern
- ✅ Tanh activation for pooler
- ✅ Weight loading infrastructure
- ✅ Optional pooler (controlled by `use_pooler` config)

**Conclusion**: **80% of BertForSequenceClassification already implemented** - we just need to wrap it in a separate class and add classifier head!

---

### 3. MNIST MLP: Basic Classifier Pattern

```cpp
// From mnist_mlp/model.cpp
m_linear1 = std::make_shared<LinearLayer>(784, 128, /* has_bias */ true);
m_linear2 = std::make_shared<LinearLayer>(128, 10, /* has_bias */ true);

// Forward pass:
tensor = (*m_linear1)(tensor);
tensor = ops::relu(tensor);
tensor = (*m_linear2)(tensor);
```

**Status**:
- ✅ Multi-layer classifier pattern
- ✅ Module composition
- ✅ register_module() pattern

---

### 4. Common Patterns Across All Models

**Module Registration** (from gpt2.cpp:116-122):
```cpp
create_name("transformer");
register_module(tok_emb, "tok_emb");
register_module(pos_emb, "pos_emb");
for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
    register_module(blocks[block_idx], fmt::format("gpt_block_{}", block_idx));
}
register_module(ln_fc, "ln_fc");
register_module(fc, "fc");
```

**Factory Functions** (from bert.cpp:705):
```cpp
std::shared_ptr<Bert> create(const BertConfig& config) {
    return std::make_shared<Bert>(config);
}

std::shared_ptr<Bert> create(const YAML::Node& config) {
    return create(read_config(config));
}
```

**Inheritance Pattern**:
```cpp
// All models inherit from BaseTransformer
class Bert : public BaseTransformer { ... };
class Llama : public BaseTransformer { ... };
class Transformer : public BaseTransformer { ... }; // GPT-2
```

---

## What We Have vs What We Need

| Component | Status | Location | Effort |
|-----------|--------|----------|--------|
| **Base BERT** | ✅ Complete | bert.cpp/bert.hpp | None - reuse |
| **Pooler (CLS extraction)** | ✅ Exists! | bert.cpp:237-258 | Copy pattern |
| **LinearLayer** | ✅ Available | modules/ | Use directly |
| **DropoutLayer** | ✅ Available | modules/ | Use directly |
| **Tanh activation** | ✅ Available | ops::tanh | Use directly |
| **Module registration** | ✅ Pattern exists | gpt2.cpp:116-122 | Copy pattern |
| **Weight loading** | ✅ Pattern exists | bert.cpp:376-467 | Extend |
| **Factory functions** | ✅ Pattern exists | bert.cpp:705 | Copy pattern |
| **Inheritance pattern** | ✅ Pattern exists | All models | Use pattern |
| **Separate head classes** | ❌ **NEW** | N/A | **Implement** |
| **Python bindings** | ❌ **NEW** | nb_models.cpp | **Add** |
| **HF validation tests** | ❌ **NEW** | test_bert_*.py | **Implement** |

**Summary**:
- ✅ **80% infrastructure exists** (building blocks, patterns, pooler)
- ❌ **20% new work** (class hierarchy, bindings, tests)

---

## Implementation Plan

### Commit Strategy

**Approach**: Incremental commits (one head at a time, fully tested before moving to next)
**Target**: 2-3 weeks for all three heads + tests
**Branch**: `ivoitovych/bert-model-for-ttml-completeness-implementation`

---

## Commit 1: BertForSequenceClassification

**Timeline**: 2-3 days
**Impact**: Enables 60% of use cases (sentiment analysis, text classification, NLI)
**Difficulty**: Medium (establishes pattern for others)

### Files to Create/Modify

**C++ Implementation**:
- `tt-train/sources/ttml/models/bert.hpp` - Add class declaration
- `tt-train/sources/ttml/models/bert.cpp` - Add implementation

**Python Bindings**:
- `tt-train/sources/ttml/nanobind/nb_models.cpp` - Add binding

**C++ Tests**:
- `tt-train/tests/model/bert_sequence_classification_test.cpp` - Unit tests

**Python Tests**:
- `tt-train/tests/python/test_bert_sequence_classification.py` - HF validation

### Implementation Template

```cpp
// In bert.hpp - Add class declaration
class BertForSequenceClassification : public Bert {
private:
    std::shared_ptr<modules::DropoutLayer> m_classifier_dropout;
    std::shared_ptr<modules::LinearLayer> m_classifier;
    uint32_t m_num_labels;

public:
    BertForSequenceClassification(
        const BertConfig& config,
        uint32_t num_labels,
        float classifier_dropout = 0.1F
    );

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr
    ) override;

    [[nodiscard]] std::tuple<autograd::TensorPtr, autograd::TensorPtr> forward_with_loss(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask,
        const autograd::TensorPtr& token_type_ids,
        const autograd::TensorPtr& labels  // [batch, 1]
    );

    void load_from_safetensors(const std::filesystem::path& path) override;
};

// Factory function
[[nodiscard]] std::shared_ptr<BertForSequenceClassification>
create_for_sequence_classification(
    const BertConfig& config,
    uint32_t num_labels,
    float classifier_dropout = 0.1F
);
```

```cpp
// In bert.cpp - Implementation
BertForSequenceClassification::BertForSequenceClassification(
    const BertConfig& config,
    uint32_t num_labels,
    float classifier_dropout
) : Bert(config), m_num_labels(num_labels) {

    // Create classifier layers
    m_classifier_dropout = std::make_shared<modules::DropoutLayer>(classifier_dropout);
    m_classifier = std::make_shared<modules::LinearLayer>(
        config.embedding_dim,
        num_labels,
        /* bias */ true
    );

    // Register modules (CRITICAL for autograd and serialization)
    register_module(m_classifier_dropout, "classifier_dropout");
    register_module(m_classifier, "classifier");
}

autograd::TensorPtr BertForSequenceClassification::operator()(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids
) {
    // 1. Get base BERT hidden states (all tokens)
    // Call parent operator() but we need hidden states, not pooled output
    // So we temporarily disable pooler or call forward differently

    // For now, reconstruct the forward pass without pooler:
    auto x = (*m_token_embeddings)(input_ids);
    auto pos = (*m_position_embeddings)(x);
    x = ops::add(x, pos);

    if (token_type_ids) {
        auto token_type = (*m_token_type_embeddings)(token_type_ids);
        x = ops::add(x, token_type);
    }

    x = (*m_embedding_norm)(x);
    x = (*m_embedding_dropout)(x);

    auto hidden_states = x;
    for (auto& block : m_blocks) {
        hidden_states = (*block)(hidden_states, attention_mask);
    }

    // 2. Extract [CLS] token (COPY PATTERN FROM BERT POOLER)
    auto batch_size = hidden_states->get_shape()[0];
    auto embedding_dim = hidden_states->get_shape()[3];

    ttnn::SmallVector<uint32_t> start = {0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> end = {batch_size, 1, 1, embedding_dim};
    ttnn::SmallVector<uint32_t> stride = {1, 1, 1, 1};

    auto cls_token = ttnn::slice(hidden_states->get_value(), start, end, stride);
    auto pooled_output = autograd::create_tensor(cls_token);

    // 3. Apply classifier head (NEW)
    pooled_output = (*m_classifier_dropout)(pooled_output);
    auto logits = (*m_classifier)(pooled_output);

    return logits;  // [batch, 1, 1, num_labels]
}

std::tuple<autograd::TensorPtr, autograd::TensorPtr>
BertForSequenceClassification::forward_with_loss(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids,
    const autograd::TensorPtr& labels
) {
    auto logits = (*this)(input_ids, attention_mask, token_type_ids);

    // TODO: Implement cross-entropy loss
    // For now, return logits and dummy loss
    auto loss = autograd::create_tensor();

    return {logits, loss};
}

void BertForSequenceClassification::load_from_safetensors(
    const std::filesystem::path& path
) {
    // 1. Load base BERT weights
    Bert::load_from_safetensors(path);

    // 2. Classifier head remains randomly initialized
    // This is standard practice - when fine-tuning, classifier is trained from scratch
    // HuggingFace does the same thing
    fmt::print("  Classifier head randomly initialized (standard for fine-tuning)\n");
}

// Factory function
std::shared_ptr<BertForSequenceClassification> create_for_sequence_classification(
    const BertConfig& config,
    uint32_t num_labels,
    float classifier_dropout
) {
    return std::make_shared<BertForSequenceClassification>(
        config,
        num_labels,
        classifier_dropout
    );
}
```

### Testing Strategy

**C++ Tests** (bert_sequence_classification_test.cpp):
```cpp
#include <gtest/gtest.h>
#include "models/bert.hpp"

TEST(BertForSequenceClassification, Construction) {
    ttml::models::bert::BertConfig config;
    config.vocab_size = 30522;
    config.embedding_dim = 768;
    config.num_heads = 12;
    config.num_blocks = 2;  // Small for testing

    auto model = ttml::models::bert::create_for_sequence_classification(
        config,
        /* num_labels */ 2
    );

    ASSERT_NE(model, nullptr);
}

TEST(BertForSequenceClassification, ForwardPassShape) {
    ttml::models::bert::BertConfig config;
    config.vocab_size = 30522;
    config.embedding_dim = 128;
    config.num_heads = 2;
    config.num_blocks = 2;
    config.max_sequence_length = 32;

    auto model = ttml::models::bert::create_for_sequence_classification(
        config,
        /* num_labels */ 3
    );

    // Create dummy input
    uint32_t batch_size = 2;
    uint32_t seq_len = 32;

    // TODO: Create actual tensor inputs
    // auto input_ids = ...
    // auto logits = (*model)(input_ids);

    // ASSERT_EQ(logits->get_shape()[0], batch_size);
    // ASSERT_EQ(logits->get_shape()[3], 3);  // num_labels
}
```

**Python Tests** (test_bert_sequence_classification.py):
```python
import pytest
import numpy as np
import torch
from transformers import BertForSequenceClassification as HFBertForSequenceClassification
import ttml

@pytest.mark.parametrize("model_name,batch_size,seq_len,num_labels", [
    ("prajjwal1/bert-tiny", 2, 32, 2),      # Binary classification
    ("prajjwal1/bert-small", 2, 64, 3),     # 3-way classification
    ("prajjwal1/bert-tiny", 4, 32, 5),      # 5 classes, larger batch
])
def test_bert_sequence_classification_pcc(model_name, batch_size, seq_len, num_labels):
    """Test BertForSequenceClassification matches HuggingFace (PCC > 0.99)"""

    # 1. Load HuggingFace reference
    hf_model = HFBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    hf_model.eval()

    # 2. Load TTML model with same base weights
    # Extract config from HF model
    hf_config = hf_model.config

    ttml_config = ttml.models.bert.BertConfig()
    ttml_config.vocab_size = hf_config.vocab_size
    ttml_config.embedding_dim = hf_config.hidden_size
    ttml_config.num_heads = hf_config.num_attention_heads
    ttml_config.num_blocks = hf_config.num_hidden_layers
    ttml_config.max_sequence_length = hf_config.max_position_embeddings
    ttml_config.intermediate_size = hf_config.intermediate_size
    ttml_config.dropout_prob = hf_config.hidden_dropout_prob
    ttml_config.layer_norm_eps = hf_config.layer_norm_eps

    ttml_model = ttml.models.bert.create_for_sequence_classification(
        ttml_config,
        num_labels=num_labels
    )
    ttml_model.load_from_safetensors(f"{model_name}/model.safetensors")

    # 3. Copy classifier head weights from HF to TTML
    # (Since HF model is randomly initialized, we need same random weights)
    hf_classifier_weight = hf_model.classifier.weight.detach().numpy()
    hf_classifier_bias = hf_model.classifier.bias.detach().numpy()

    # TODO: Set TTML classifier weights to match HF
    # ttml_model.set_classifier_weights(hf_classifier_weight, hf_classifier_bias)

    # 4. Generate deterministic inputs
    np.random.seed(42)
    torch.manual_seed(42)
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # 5. Forward pass
    with torch.no_grad():
        hf_output = hf_model(input_ids, attention_mask).logits

    ttml_input_ids = ttml.autograd.Tensor.from_numpy(
        input_ids.numpy().astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    )
    ttml_attention_mask = ttml.autograd.Tensor.from_numpy(
        attention_mask.numpy().astype(np.float32).reshape(batch_size, 1, 1, seq_len)
    )

    ttml_output = ttml_model(ttml_input_ids, ttml_attention_mask)
    ttml_output_np = ttml_output.to_numpy().reshape(batch_size, num_labels)

    # 6. Calculate PCC
    pcc = np.corrcoef(hf_output.flatten().numpy(), ttml_output_np.flatten())[0, 1]
    mean_diff = np.mean(np.abs(hf_output.numpy() - ttml_output_np))
    max_diff = np.max(np.abs(hf_output.numpy() - ttml_output_np))

    # 7. Print detailed results
    print(f"\n{'='*70}")
    print(f"BertForSequenceClassification Validation")
    print(f"{'='*70}")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}, Seq length: {seq_len}")
    print(f"  Num labels: {num_labels}")
    print(f"  PCC: {pcc:.6f}")
    print(f"  Mean Abs Diff: {mean_diff:.6e}")
    print(f"  Max Abs Diff: {max_diff:.6e}")
    print(f"  Status: {'✅ PASS' if pcc > 0.99 else '❌ FAIL'}")
    print(f"{'='*70}\n")

    # 8. Assert validation
    assert pcc >= 0.95, f"PCC {pcc} < 0.95 (warning threshold)"
    assert pcc > 0.99, f"PCC {pcc} < 0.99 (target threshold failed)"


def test_bert_sequence_classification_edge_cases():
    """Test edge cases: single label, many labels, variable lengths"""

    # Test with num_labels=1 (regression-like)
    config = ttml.models.bert.BertConfig()
    model = ttml.models.bert.create_for_sequence_classification(config, num_labels=1)
    assert model is not None

    # Test with many labels
    model = ttml.models.bert.create_for_sequence_classification(config, num_labels=100)
    assert model is not None

    # TODO: Test variable sequence lengths with padding
```

### Success Criteria

- ✅ C++ tests pass (100%)
- ✅ Python PCC tests pass with PCC ≥ 0.99
- ✅ All model sizes validated (tiny, small, base)
- ✅ Edge cases handled (1 label, many labels, variable lengths)
- ✅ Compiles without warnings
- ✅ Memory leaks checked

---

## Commit 2: BertForTokenClassification

**Timeline**: 1-2 days
**Impact**: Enables NER, POS tagging, chunking
**Difficulty**: Easy (follows same pattern as SequenceClassification)

### Key Differences from SequenceClassification

1. **No pooling** - use all token representations, not just [CLS]
2. **Per-token classification** - output shape [batch, seq_len, num_labels]
3. **Common use case**: Named Entity Recognition (9 labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC)

### Implementation Outline

```cpp
class BertForTokenClassification : public Bert {
private:
    std::shared_ptr<modules::DropoutLayer> m_classifier_dropout;
    std::shared_ptr<modules::LinearLayer> m_classifier;
    uint32_t m_num_labels;

public:
    autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr
    ) override {
        // 1. Get all hidden states (no pooling!)
        auto hidden_states = Bert::operator()(input_ids, attention_mask, token_type_ids);

        // 2. Apply dropout to all tokens
        hidden_states = (*m_classifier_dropout)(hidden_states);

        // 3. Classify each token
        auto logits = (*m_classifier)(hidden_states);

        return logits;  // [batch, seq_len, num_labels]
    }
};
```

### Testing

**HF Validation**: CoNLL-2003 NER dataset format with 9 labels
**PCC Target**: ≥ 0.99

---

## Commit 3: BertForQuestionAnswering

**Timeline**: 2-3 days
**Impact**: Enables SQuAD-style extractive QA
**Difficulty**: Medium (dual outputs: start/end positions)

### Key Differences

1. **Dual outputs** - start_logits and end_logits (both [batch, seq_len])
2. **No pooling** - use all token representations
3. **Common use case**: SQuAD v2.0 question answering

### Implementation Outline

```cpp
class BertForQuestionAnswering : public Bert {
private:
    std::shared_ptr<modules::LinearLayer> m_qa_outputs;  // embedding_dim → 2

public:
    std::tuple<autograd::TensorPtr, autograd::TensorPtr> operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr
    ) override {
        // 1. Get all hidden states
        auto hidden_states = Bert::operator()(input_ids, attention_mask, token_type_ids);

        // 2. Project to 2 dimensions (start, end)
        auto qa_logits = (*m_qa_outputs)(hidden_states);  // [batch, seq_len, 2]

        // 3. Split into start and end logits
        auto start_logits = slice_last_dim(qa_logits, 0);  // [batch, seq_len]
        auto end_logits = slice_last_dim(qa_logits, 1);    // [batch, seq_len]

        return {start_logits, end_logits};
    }
};
```

### Testing

**HF Validation**: SQuAD v2.0 samples
**PCC Target**: ≥ 0.99 for both start and end logits

---

## Commit 4: Example Training Scripts

**Timeline**: 3-4 days
**Impact**: Demonstrates end-to-end usage
**Difficulty**: Medium (need working data pipeline)

### Files to Create

```
tt-train/sources/examples/bert/
├── bert_sequence_classification.cpp  # Sentiment analysis example
├── bert_token_classification.cpp     # NER example
├── bert_question_answering.cpp       # SQuAD example
├── CMakeLists.txt                    # Build configuration
└── README.md                         # Documentation
```

### Example Script Structure

```cpp
// bert_sequence_classification.cpp
#include "models/bert.hpp"
#include "optimizers/adamw.hpp"

int main() {
    // 1. Load pre-trained BERT
    auto config = bert::BertConfig{};
    auto model = bert::create_for_sequence_classification(config, num_labels=2);
    model->load_from_safetensors("bert-base-uncased.safetensors");

    // 2. Setup optimizer (only for classifier head)
    auto optimizer = optimizers::AdamW(
        model->get_classifier_parameters(),
        learning_rate=2e-5
    );

    // 3. Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto& batch : dataloader) {
            auto [logits, loss] = model->forward_with_loss(
                batch.input_ids,
                batch.attention_mask,
                batch.token_type_ids,
                batch.labels
            );

            loss->backward();
            optimizer.step();
            optimizer.zero_grad();

            fmt::print("Loss: {:.4f}\n", loss->item());
        }
    }

    // 4. Save fine-tuned model
    model->save_weights("bert-sentiment.safetensors");

    return 0;
}
```

---

## Timeline Summary

| Phase | Task | Duration | Tests | Dependencies |
|-------|------|----------|-------|--------------|
| 1 | BertForSequenceClassification | 2-3 days | C++ + Python (PCC > 0.99) | None |
| 2 | BertForTokenClassification | 1-2 days | C++ + Python (PCC > 0.99) | Phase 1 pattern |
| 3 | BertForQuestionAnswering | 2-3 days | C++ + Python (PCC > 0.99) | Phase 1 pattern |
| 4 | Example training scripts | 3-4 days | Compilation + smoke tests | Phases 1-3 |
| **Total** | **Phase 0 Complete** | **9-12 days** | **All heads validated** | - |

---

## Implementation Checklist

### Pre-Implementation
- [x] Investigate existing patterns (GPT-2, LLaMA, BERT)
- [x] Identify reusable components (pooler, modules, ops)
- [x] Document findings in this plan
- [ ] Review plan with stakeholders
- [ ] Allocate resources (1-2 engineers)

### Phase 1: BertForSequenceClassification
- [ ] Create C++ class in bert.hpp/bert.cpp
- [ ] Add Python bindings in nb_models.cpp
- [ ] Write C++ unit tests
- [ ] Write Python HF validation tests
- [ ] Achieve PCC ≥ 0.99 vs HuggingFace
- [ ] Test edge cases (1 label, many labels, variable lengths)
- [ ] Code review
- [ ] Commit and push

### Phase 2: BertForTokenClassification
- [ ] Create C++ class (copy pattern from Phase 1)
- [ ] Add Python bindings
- [ ] Write C++ unit tests
- [ ] Write Python HF validation tests (CoNLL-2003)
- [ ] Achieve PCC ≥ 0.99 vs HuggingFace
- [ ] Test NER use case
- [ ] Code review
- [ ] Commit and push

### Phase 3: BertForQuestionAnswering
- [ ] Create C++ class (dual outputs pattern)
- [ ] Add Python bindings
- [ ] Write C++ unit tests
- [ ] Write Python HF validation tests (SQuAD v2.0)
- [ ] Achieve PCC ≥ 0.99 for both outputs
- [ ] Test QA use case
- [ ] Code review
- [ ] Commit and push

### Phase 4: Example Scripts
- [ ] Write bert_sequence_classification.cpp
- [ ] Write bert_token_classification.cpp
- [ ] Write bert_question_answering.cpp
- [ ] Create CMakeLists.txt
- [ ] Write README.md with usage instructions
- [ ] Test compilation
- [ ] Run smoke tests
- [ ] Code review
- [ ] Commit and push

### Post-Implementation
- [ ] Update BERT_ACTION_PLAN.md (mark Phase 0 complete)
- [ ] Update BERT_IMPLEMENTATION_COMPLETENESS.md (65% → 80%)
- [ ] Run full test suite (C++ + Python)
- [ ] Update documentation
- [ ] Create PR for review

---

## Risk Mitigation

### Technical Risks

**Risk 1**: Forward pass implementation differs from HuggingFace
- **Likelihood**: Medium
- **Impact**: High (PCC < 0.99)
- **Mitigation**:
  - Use layer-by-layer validation (proven from MHA fix)
  - Compare intermediate activations, not just final output
  - Start with bert-tiny (faster iteration)

**Risk 2**: Weight loading/initialization issues
- **Likelihood**: Low
- **Impact**: High (incorrect results)
- **Mitigation**:
  - Classifier head randomly initialized (standard practice)
  - Test with HF models that have pre-trained heads (if available)
  - Validate base BERT loading separately first

**Risk 3**: Python bindings don't work correctly
- **Likelihood**: Low
- **Impact**: Medium (can't run tests)
- **Mitigation**:
  - Follow existing patterns from nb_models.cpp
  - Test bindings incrementally
  - Use nanobind documentation

### Resource Risks

**Risk 4**: Timeline slips due to unforeseen complexity
- **Likelihood**: Medium
- **Impact**: Low (not blocking)
- **Mitigation**:
  - Phase 1 establishes pattern (invest time here)
  - Phases 2-3 are faster (copy pattern)
  - Can deliver phases incrementally

**Risk 5**: PCC targets not met
- **Likelihood**: Low (based on existing BERT validation)
- **Impact**: Medium (requires debugging)
- **Mitigation**:
  - Start with models already validated (bert-tiny, bert-small)
  - Use existing test infrastructure (isolated_layer_validation.py pattern)
  - Leverage pooler code that already works

---

## Success Metrics

### Phase 0 Success Criteria

**Quantitative**:
- ✅ All C++ tests pass (100%)
- ✅ All Python tests pass (100%)
- ✅ PCC ≥ 0.99 for all heads vs HuggingFace
- ✅ Mean absolute difference < 1e-3
- ✅ Max absolute difference < 5e-3
- ✅ 3 heads implemented (SequenceClassification, TokenClassification, QuestionAnswering)
- ✅ Example scripts compile and run

**Qualitative**:
- ✅ Code follows TTML patterns (module registration, factory functions)
- ✅ Clean, well-documented code
- ✅ Comprehensive test coverage (unit + integration)
- ✅ Easy to extend for future heads (MLM, NSP, etc.)

**Production Readiness**:
- ✅ Enables 80% of common BERT use cases
- ✅ Users can fine-tune without writing custom code
- ✅ Documentation enables self-service usage
- ✅ Ready for Phase 1 (training infrastructure)

---

## Key Insights from Investigation

1. **80% infrastructure exists** - BERT's pooler is exactly what we need
2. **Clear patterns to follow** - All models use same structure
3. **Not building from scratch** - Extending existing, proven components
4. **HuggingFace compatible** - Weight loading already works
5. **Test infrastructure ready** - Can reuse validation patterns

---

## References

**Source Code**:
- BERT pooler: `tt-train/sources/ttml/models/bert.cpp` (lines 118-120, 237-258)
- Module registration: `tt-train/sources/ttml/models/gpt2.cpp` (lines 116-122)
- MNIST classifier: `tt-train/sources/examples/mnist_mlp/model.cpp`
- Weight loading: `tt-train/sources/ttml/models/bert.cpp` (lines 376-467)

**Tests**:
- Existing BERT tests: `tt-train/tests/python/test_bert_*.py`
- PCC validation pattern: `tt-train/tests/python/test_bert_isolated_layer_validation.py`

**Documentation**:
- Action Plan: `BERT_ACTION_PLAN.md` (Phase 0)
- Completeness Analysis: `BERT_IMPLEMENTATION_COMPLETENESS.md`
- Ecosystem Analysis: `TTML_MODEL_ECOSYSTEM_ANALYSIS.md`

---

## Next Steps

**Immediate** (Ready to start):
1. Begin Phase 1: BertForSequenceClassification
2. Create C++ implementation files
3. Add Python bindings
4. Write comprehensive tests
5. Achieve PCC ≥ 0.99 validation

**Questions for stakeholders**:
- Should we implement all 3 heads before committing, or commit incrementally?
- Are there specific datasets/benchmarks we should validate against?
- Who will review the code (C++ expert + ML expert)?

---

**Date**: 2025-11-05
**Status**: Ready for implementation
**Estimated Completion**: 2-3 weeks from start
**Next Document Update**: After Phase 0 completion
