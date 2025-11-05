# BERT Implementation Completeness - Action Plan

**Date**: 2025-11-05
**Branch**: ivoitovych/bert-model-for-ttml-completeness-analysis
**Context**: Based on BERT_IMPLEMENTATION_COMPLETENESS.md and TTML_MODEL_ECOSYSTEM_ANALYSIS.md

---

## Executive Summary

This action plan addresses the **35% completeness gap** in BERT implementation, focusing on two critical areas:
1. **Distributed Training Infrastructure** (align with GPT-2/LLaMA patterns)
2. **Task-Specific Heads** (enable production use cases)

**Current Status**: BERT at 65% completeness
- Core: 95% ✅
- Task Heads: 0% ❌
- Distributed Training: 0% ❌
- Training Utilities: 80% ⚠️

**Target**: Achieve 90% completeness with distributed training + essential task heads

---

## Strategic Context

### Framework Philosophy (from Ecosystem Analysis)

TTML is **training-focused**, not deployment-focused:
- Primary use case: Pre-training and base model training
- Secondary use case: Feature extraction for downstream tasks
- Out of scope (currently): Production fine-tuning, serving infrastructure

### Key Insight

**GPT-2 and LLaMA lack task-specific heads too** - this is systematic across TTML. Therefore, task-specific heads should be considered **optional extensions** rather than core requirements.

**However**: BERT **lacks distributed training support** that GPT-2 and LLaMA have. This is a critical gap for the training-focused framework.

---

## Priority Framework

### P0: Critical for Training Parity 🚨
Features that bring BERT to parity with GPT-2/LLaMA for training workloads

### P1: High-Impact Production Enablers 🔥
Features that unlock 80% of common use cases with minimal effort

### P2: Training Infrastructure 🛠️
Features that improve training experience and usability

### P3: Advanced Features 📋
Nice-to-have features for specialized use cases

---

## Action Plan

## P0: Distributed Training Infrastructure 🚨

**Priority**: HIGHEST - Critical gap vs GPT-2/LLaMA
**Timeline**: 3-4 weeks
**Effort**: High
**Impact**: Enables large-scale BERT training

### Task 0.1: Add Tensor Parallelism (TP) Support

**Rationale**: Both GPT-2 and LLaMA have TP support. BERT needs this for multi-device training.

**Implementation Pattern** (follow LLaMA):
```cpp
// Location: tt-train/sources/ttml/models/distributed/bert.hpp
namespace ttml::models::bert::distributed {
    struct BertConfig {
        // Inherit from base BertConfig
        uint32_t tensor_parallel_size = 1U;
        // Other distributed training params
    };

    class Bert : public ttml::models::bert::Bert {
    private:
        uint32_t m_tensor_parallel_size;
        uint32_t m_device_id;

    public:
        explicit Bert(const BertConfig& config);
        void setup_tensor_parallel(uint32_t rank, uint32_t world_size);
        // Override forward pass for TP
    };
}
```

**Files to Create**:
- `tt-train/sources/ttml/models/distributed/bert.hpp`
- `tt-train/sources/ttml/models/distributed/bert.cpp`

**Reference Implementations**:
- LLaMA: `tt-train/sources/ttml/models/distributed/llama.{cpp,hpp}` (preferred - more modern)
- GPT-2: `tt-train/sources/ttml/models/distributed/gpt2.{cpp,hpp}` (3-tier architecture)

**Key Components**:
1. Split attention heads across devices
2. Split MLP intermediate dimension across devices
3. All-reduce after attention output projection
4. All-reduce after MLP output projection
5. Collective communication primitives

**Testing**:
- [ ] Single-device matches non-distributed (sanity check)
- [ ] 2-device TP produces correct outputs
- [ ] 4-device TP produces correct outputs
- [ ] 8-device TP produces correct outputs
- [ ] Gradient correctness across devices

**Estimated Effort**: 2-3 weeks

---

### Task 0.2: Add Pipeline Parallelism (PP) Support

**Rationale**: LLaMA has PP support for very large models (split layers across devices)

**Implementation Pattern**:
```cpp
struct BertConfig {
    uint32_t pipeline_parallel_size = 1U;
    uint32_t num_pipeline_stages;
};
```

**Key Components**:
1. Partition transformer blocks across devices
2. Implement pipeline schedule (GPipe, 1F1B, interleaved)
3. Handle activation communication between stages
4. Gradient accumulation across micro-batches

**Testing**:
- [ ] 2-stage pipeline produces correct outputs
- [ ] 4-stage pipeline produces correct outputs
- [ ] Gradient correctness across pipeline stages
- [ ] Memory savings vs single-device

**Estimated Effort**: 1-2 weeks

---

### Task 0.3: Add Distributed Training Config Files

**Rationale**: Both GPT-2 and LLaMA have extensive config files. BERT has **zero**.

**Files to Create** (follow GPT-2/LLaMA patterns):
```
tt-train/configs/bert/
├── bert-tiny-tp2.yaml          # 2-device tensor parallel
├── bert-tiny-tp4.yaml          # 4-device tensor parallel
├── bert-small-tp2.yaml
├── bert-small-tp4.yaml
├── bert-base-tp2.yaml
├── bert-base-tp4.yaml
├── bert-base-tp8.yaml
├── bert-base-pp2.yaml          # 2-stage pipeline
├── bert-base-tp4-pp2.yaml      # Hybrid TP+PP
├── bert-large-tp8.yaml
└── bert-large-tp8-pp4.yaml     # Large model with hybrid
```

**Config Content Pattern** (follow `tt-train/configs/gpt2/gpt2_117m.yaml`):
```yaml
model:
  model_type: bert
  vocab_size: 30522
  max_sequence_length: 512
  embedding_dim: 768
  num_heads: 12
  num_blocks: 12
  dropout_prob: 0.1
  layer_norm_eps: 1.0e-12
  use_token_type_embeddings: true
  type_vocab_size: 2

distributed:
  tensor_parallel_size: 4
  pipeline_parallel_size: 1

training:
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  warmup_steps: 10000
  max_steps: 1000000
  weight_decay: 0.01
```

**Estimated Effort**: 2-3 days

---

### Task 0.4: Add BERT to Model Factory

**Rationale**: GPT-2 and LLaMA are in `TransformerModelFactory`. BERT is missing.

**File to Modify**:
- `tt-train/sources/ttml/ttml/common/model_factory.py`

**Implementation**:
```python
class TransformerModelFactory:
    def create_model(self):
        if self.model_type == "gpt2":
            return self._create_gpt2()
        elif self.model_type == "llama":
            return self._create_llama()
        elif self.model_type == "bert":
            return self._create_bert()
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

    def _create_bert(self):
        # Create distributed or non-distributed BERT based on config
        import ttml.models.bert as bert_module
        if self.config.get('distributed', {}).get('tensor_parallel_size', 1) > 1:
            import ttml.models.distributed.bert as dist_bert
            return dist_bert.create(self.config)
        else:
            return bert_module.create(self.config)
```

**Estimated Effort**: 1-2 days

---

## P1: High-Impact Task-Specific Heads 🔥

**Priority**: HIGH - Enables production use cases
**Timeline**: 2-3 weeks
**Effort**: Medium
**Impact**: Unlocks 80% of common NLP tasks

### Task 1.1: Implement BertForSequenceClassification

**Impact**: Enables sentiment analysis, text classification, NLI
**Use Cases**: 60% of BERT fine-tuning tasks
**Estimated Effort**: 2-3 days

**Implementation**:
```cpp
// Location: tt-train/sources/ttml/models/bert.hpp (add to existing file)

namespace ttml::models::bert {
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

        // Forward pass
        [[nodiscard]] autograd::TensorPtr operator()(
            const autograd::TensorPtr& input_ids,
            const autograd::TensorPtr& attention_mask = nullptr,
            const autograd::TensorPtr& token_type_ids = nullptr
        ) override;

        // Forward with loss (for training)
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
        uint32_t num_labels
    );
}
```

**Architecture**:
1. Base BERT forward pass → final hidden state [B, S, H]
2. Extract [CLS] token (first token): [B, 1, H]
3. Apply dropout
4. Linear classifier: H → num_labels
5. Return logits [B, num_labels]

**Training Support**:
- Cross-entropy loss for classification
- Label smoothing (optional)
- Class weights (for imbalanced datasets)

**Testing**:
- [ ] Load pre-trained BERT weights
- [ ] Forward pass produces correct shape
- [ ] Compare with HuggingFace BertForSequenceClassification
- [ ] PCC validation > 0.99
- [ ] Fine-tuning convergence test (small dataset)

**Example Use**:
```cpp
// Sentiment analysis (2 classes: positive/negative)
auto config = bert::BertConfig{};
auto model = bert::create_for_sequence_classification(config, num_labels=2);
model->load_from_safetensors("bert-base-uncased.safetensors");

auto logits = model->forward(input_ids, attention_mask);  // [batch, 2]
// Apply softmax for probabilities, argmax for predictions
```

---

### Task 1.2: Implement BertForTokenClassification

**Impact**: Enables NER, POS tagging, chunking
**Use Cases**: 20% of BERT fine-tuning tasks
**Estimated Effort**: 1-2 days

**Implementation**:
```cpp
namespace ttml::models::bert {
    class BertForTokenClassification : public Bert {
    private:
        std::shared_ptr<modules::DropoutLayer> m_classifier_dropout;
        std::shared_ptr<modules::LinearLayer> m_classifier;
        uint32_t m_num_labels;

    public:
        BertForTokenClassification(
            const BertConfig& config,
            uint32_t num_labels,
            float classifier_dropout = 0.1F
        );

        [[nodiscard]] autograd::TensorPtr operator()(
            const autograd::TensorPtr& input_ids,
            const autograd::TensorPtr& attention_mask = nullptr,
            const autograd::TensorPtr& token_type_ids = nullptr
        ) override;

        // Forward with loss (for training)
        [[nodiscard]] std::tuple<autograd::TensorPtr, autograd::TensorPtr> forward_with_loss(
            const autograd::TensorPtr& input_ids,
            const autograd::TensorPtr& attention_mask,
            const autograd::TensorPtr& token_type_ids,
            const autograd::TensorPtr& labels  // [batch, seq_len]
        );
    };

    [[nodiscard]] std::shared_ptr<BertForTokenClassification>
    create_for_token_classification(
        const BertConfig& config,
        uint32_t num_labels
    );
}
```

**Architecture**:
1. Base BERT forward pass → final hidden state [B, S, H]
2. Apply dropout to all tokens
3. Linear classifier for each token: H → num_labels
4. Return logits [B, S, num_labels]

**Testing**:
- [ ] Forward pass produces correct shape [B, S, num_labels]
- [ ] Compare with HuggingFace BertForTokenClassification
- [ ] PCC validation > 0.99
- [ ] NER fine-tuning test (CoNLL-2003 sample)

**Example Use**:
```cpp
// Named Entity Recognition (9 classes: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC)
auto model = bert::create_for_token_classification(config, num_labels=9);
model->load_from_safetensors("bert-base-uncased.safetensors");

auto logits = model->forward(input_ids, attention_mask);  // [batch, seq_len, 9]
// Apply softmax per token, argmax for predictions
```

---

### Task 1.3: Implement BertForQuestionAnswering

**Impact**: Enables extractive QA (SQuAD, reading comprehension)
**Use Cases**: 10% of BERT fine-tuning tasks
**Estimated Effort**: 2-3 days

**Implementation**:
```cpp
namespace ttml::models::bert {
    class BertForQuestionAnswering : public Bert {
    private:
        std::shared_ptr<modules::LinearLayer> m_qa_outputs;  // H → 2 (start/end logits)

    public:
        explicit BertForQuestionAnswering(const BertConfig& config);

        // Forward pass returns start and end logits separately
        [[nodiscard]] std::tuple<autograd::TensorPtr, autograd::TensorPtr> forward(
            const autograd::TensorPtr& input_ids,
            const autograd::TensorPtr& attention_mask = nullptr,
            const autograd::TensorPtr& token_type_ids = nullptr
        );

        // Forward with loss (for training)
        [[nodiscard]] std::tuple<
            autograd::TensorPtr,  // start_logits
            autograd::TensorPtr,  // end_logits
            autograd::TensorPtr   // loss
        > forward_with_loss(
            const autograd::TensorPtr& input_ids,
            const autograd::TensorPtr& attention_mask,
            const autograd::TensorPtr& token_type_ids,
            const autograd::TensorPtr& start_positions,  // [batch, 1]
            const autograd::TensorPtr& end_positions     // [batch, 1]
        );
    };

    [[nodiscard]] std::shared_ptr<BertForQuestionAnswering>
    create_for_question_answering(const BertConfig& config);
}
```

**Architecture**:
1. Base BERT forward pass → final hidden state [B, S, H]
2. Linear layer: H → 2 (start logits, end logits)
3. Split into start_logits [B, S] and end_logits [B, S]
4. Return both tensors

**Training Support**:
- Cross-entropy loss for start position
- Cross-entropy loss for end position
- Combined loss = start_loss + end_loss

**Testing**:
- [ ] Forward pass produces two [B, S] tensors
- [ ] Compare with HuggingFace BertForQuestionAnswering
- [ ] PCC validation > 0.99
- [ ] SQuAD v2.0 sample fine-tuning test

**Example Use**:
```cpp
// Question: "Where is Paris?" Context: "Paris is the capital of France."
// Input format: [CLS] question [SEP] context [SEP]
auto model = bert::create_for_question_answering(config);
model->load_from_safetensors("bert-base-uncased.safetensors");

auto [start_logits, end_logits] = model->forward(input_ids, attention_mask, token_type_ids);
// Apply softmax, find best span (start_idx, end_idx)
```

---

### Task 1.4: Create Example Training Scripts

**Impact**: Demonstrates how to use task-specific heads
**Estimated Effort**: 3-4 days

**Files to Create**:
```
tt-train/sources/examples/bert/
├── bert_sequence_classification.cpp    # Fine-tune for sentiment analysis
├── bert_token_classification.cpp       # Fine-tune for NER
├── bert_question_answering.cpp         # Fine-tune for SQuAD
└── README.md                            # Documentation
```

**Example Content** (bert_sequence_classification.cpp):
```cpp
#include "models/bert.hpp"
#include "optimizers/sgd.hpp"

int main() {
    // 1. Load model
    auto config = bert::BertConfig{};
    auto model = bert::create_for_sequence_classification(config, num_labels=2);
    model->load_from_safetensors("bert-base-uncased.safetensors");

    // 2. Setup optimizer
    auto optimizer = optimizers::SGD(model->parameters(), learning_rate=2e-5);

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
        }
    }

    // 4. Save fine-tuned model
    model->save_weights("bert-sentiment.safetensors");
}
```

**Testing**:
- [ ] All examples compile successfully
- [ ] Sequence classification example trains and converges
- [ ] Token classification example trains and converges
- [ ] Question answering example trains and converges

---

## P2: Training Infrastructure Improvements 🛠️

**Priority**: MEDIUM - Improves training experience
**Timeline**: 2-3 weeks
**Effort**: Medium
**Impact**: Better usability, faster convergence

### Task 2.1: Add Learning Rate Schedulers

**Rationale**: BERT fine-tuning requires warmup + linear decay for good convergence

**Implementation**:
```cpp
// Location: tt-train/sources/ttml/optimizers/scheduler.hpp

namespace ttml::optimizers {
    class LRScheduler {
    public:
        virtual ~LRScheduler() = default;
        virtual float get_lr(uint32_t step) const = 0;
    };

    class LinearWarmupLinearDecay : public LRScheduler {
    private:
        float m_base_lr;
        uint32_t m_warmup_steps;
        uint32_t m_total_steps;

    public:
        LinearWarmupLinearDecay(
            float base_lr,
            uint32_t warmup_steps,
            uint32_t total_steps
        );

        float get_lr(uint32_t step) const override;
    };

    class CosineAnnealingWarmup : public LRScheduler {
        // Cosine annealing with warmup (used in BERT pre-training)
    };
}
```

**Common Patterns**:
1. **Linear warmup + Linear decay**: Standard BERT fine-tuning
2. **Linear warmup + Cosine annealing**: BERT pre-training
3. **Constant LR with warmup**: Simple baseline

**Testing**:
- [ ] Warmup produces correct LR ramp
- [ ] Decay produces correct LR decay
- [ ] Integration with optimizer works correctly

**Estimated Effort**: 2-3 days

---

### Task 2.2: Add Gradient Clipping

**Rationale**: Prevents exploding gradients during training

**Implementation**:
```cpp
// Location: tt-train/sources/ttml/optimizers/gradient_utils.hpp

namespace ttml::optimizers {
    // Clip gradients by global norm
    void clip_grad_norm_(
        const std::vector<autograd::TensorPtr>& parameters,
        float max_norm
    );

    // Clip gradients by value
    void clip_grad_value_(
        const std::vector<autograd::TensorPtr>& parameters,
        float clip_value
    );
}
```

**Testing**:
- [ ] Gradient norm clipping works correctly
- [ ] Gradient value clipping works correctly
- [ ] No gradient corruption

**Estimated Effort**: 1-2 days

---

### Task 2.3: Add Standard Loss Functions

**Rationale**: Task-specific heads need standard losses

**Implementation**:
```cpp
// Location: tt-train/sources/ttml/ops/loss_functions.hpp

namespace ttml::ops {
    // Cross-entropy loss for classification
    autograd::TensorPtr cross_entropy_loss(
        const autograd::TensorPtr& logits,      // [batch, num_classes]
        const autograd::TensorPtr& labels,       // [batch]
        const autograd::TensorPtr& weights = nullptr  // Optional class weights
    );

    // Binary cross-entropy loss
    autograd::TensorPtr binary_cross_entropy_loss(
        const autograd::TensorPtr& logits,
        const autograd::TensorPtr& labels
    );

    // Label smoothing cross-entropy
    autograd::TensorPtr label_smoothing_cross_entropy_loss(
        const autograd::TensorPtr& logits,
        const autograd::TensorPtr& labels,
        float smoothing = 0.1F
    );
}
```

**Testing**:
- [ ] Losses match PyTorch reference
- [ ] Backward pass produces correct gradients
- [ ] Label smoothing works correctly

**Estimated Effort**: 2-3 days

---

### Task 2.4: Add Checkpointing Utilities

**Rationale**: Save/restore training state (model + optimizer)

**Implementation**:
```cpp
// Location: tt-train/sources/ttml/utils/checkpoint.hpp

namespace ttml::utils {
    struct CheckpointState {
        std::unordered_map<std::string, autograd::TensorPtr> model_state;
        std::unordered_map<std::string, autograd::TensorPtr> optimizer_state;
        uint32_t step;
        uint32_t epoch;
        float best_metric;
    };

    void save_checkpoint(
        const std::filesystem::path& path,
        const CheckpointState& state
    );

    CheckpointState load_checkpoint(
        const std::filesystem::path& path
    );
}
```

**Estimated Effort**: 2-3 days

---

## P3: Advanced Features 📋

**Priority**: LOW - Nice to have for specialized use cases
**Timeline**: 1-2 months
**Effort**: High
**Impact**: Specialized use cases

### Task 3.1: Implement BertForMaskedLM

**Impact**: Enables domain adaptation via continued pre-training
**Estimated Effort**: 3-5 days

**Components**:
```cpp
class BertForMaskedLM : public Bert {
private:
    std::shared_ptr<modules::LinearLayer> m_transform;      // H → H
    std::shared_ptr<modules::LayerNormLayer> m_transform_norm;
    std::shared_ptr<modules::LinearLayer> m_decoder;        // H → vocab_size
    bool m_tie_weights;  // Tie decoder weights with input embeddings
};
```

**Use Cases**:
- Continue pre-training BERT on domain-specific data (medical, legal, code)
- Fill-mask tasks
- Reproduce BERT pre-training

---

### Task 3.2: Add Gradient Checkpointing (Memory-Efficient Runner)

**Impact**: Train larger models with limited memory
**Estimated Effort**: 1 week

**Note**: GPT-2 and LLaMA already have `RunnerType::MemoryEfficient` support via `memory_efficient_runner()` pattern. BERT should follow the same pattern.

**Implementation**:
```cpp
// Use existing memory_efficient_runner from transformer_common.hpp
autograd::TensorPtr Bert::forward_memory_efficient(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask
) {
    auto x = compute_embeddings(input_ids, attention_mask);

    // Apply memory-efficient runner to each block
    for (auto& block : m_blocks) {
        x = memory_efficient_runner(
            [&](const autograd::TensorPtr& h) { return (*block)(h, attention_mask); },
            x,
            attention_mask
        );
    }

    return x;
}
```

**Testing**:
- [ ] Memory usage reduction vs standard runner
- [ ] Forward pass matches standard runner (numerical equivalence)
- [ ] Backward pass produces correct gradients
- [ ] Training convergence matches standard runner

---

### Task 3.3: Add Mixed Precision Training Support

**Impact**: Faster training, lower memory usage
**Estimated Effort**: 2 weeks

**Components**:
- FP16/BF16 forward pass
- FP32 gradient accumulation
- Loss scaling to prevent underflow
- Dynamic loss scaling

**Note**: This is a framework-wide feature, not BERT-specific

---

### Task 3.4: Implement BertForNextSentencePrediction

**Impact**: Limited (NSP task is mostly deprecated in modern BERT training)
**Estimated Effort**: 2-3 days

**Components**:
```cpp
class BertForNextSentencePrediction : public Bert {
private:
    std::shared_ptr<modules::LinearLayer> m_nsp_classifier;  // H → 2
};
```

**Use Cases**:
- Reproduce original BERT pre-training
- Sentence pair coherence tasks

---

### Task 3.5: Implement BertForPreTraining

**Impact**: Full BERT pre-training (MLM + NSP)
**Estimated Effort**: 5-7 days

**Components**:
```cpp
class BertForPreTraining : public Bert {
private:
    std::shared_ptr<BertMLMHead> m_mlm_head;
    std::shared_ptr<modules::LinearLayer> m_nsp_classifier;

public:
    std::tuple<
        autograd::TensorPtr,  // mlm_logits [B, S, V]
        autograd::TensorPtr   // nsp_logits [B, 2]
    > forward(...);
};
```

**Use Cases**:
- Pre-train BERT from scratch
- Full BERT training pipeline reproduction

---

### Task 3.6: Add Model Export (ONNX)

**Impact**: Enables deployment to production serving
**Estimated Effort**: 2-3 weeks

**Note**: This is framework-wide feature, not BERT-specific

---

## Implementation Phases

### Phase 0: Distributed Training Parity (3-4 weeks) 🚨

**Goal**: Bring BERT to parity with GPT-2/LLaMA for training workloads

**Tasks**:
- [x] Review ecosystem patterns
- [ ] Implement TP support (Task 0.1) - 2-3 weeks
- [ ] Implement PP support (Task 0.2) - 1-2 weeks
- [ ] Create distributed config files (Task 0.3) - 2-3 days
- [ ] Add BERT to model factory (Task 0.4) - 1-2 days

**Success Metrics**:
- [ ] BERT can train on 2/4/8 devices with TP
- [ ] BERT can train with PP (2/4 stages)
- [ ] Performance parity with GPT-2/LLaMA (throughput, memory)
- [ ] All distributed configs work correctly

**Outcome**: BERT ready for large-scale training workloads

---

### Phase 1: Essential Task Heads (2-3 weeks) 🔥

**Goal**: Enable 80% of common production use cases

**Tasks**:
- [ ] BertForSequenceClassification (Task 1.1) - 2-3 days
- [ ] BertForTokenClassification (Task 1.2) - 1-2 days
- [ ] BertForQuestionAnswering (Task 1.3) - 2-3 days
- [ ] Example training scripts (Task 1.4) - 3-4 days

**Success Metrics**:
- [ ] All task heads match HuggingFace outputs (PCC > 0.99)
- [ ] Example scripts train and converge
- [ ] Documentation complete

**Outcome**: Users can fine-tune BERT for classification, NER, QA without custom code

---

### Phase 2: Training Infrastructure (2-3 weeks) 🛠️

**Goal**: Improve training experience and usability

**Tasks**:
- [ ] Learning rate schedulers (Task 2.1) - 2-3 days
- [ ] Gradient clipping (Task 2.2) - 1-2 days
- [ ] Standard loss functions (Task 2.3) - 2-3 days
- [ ] Checkpointing utilities (Task 2.4) - 2-3 days

**Success Metrics**:
- [ ] Schedulers produce correct LR schedules
- [ ] Gradient clipping prevents exploding gradients
- [ ] Losses match PyTorch reference
- [ ] Checkpointing works correctly

**Outcome**: Better training experience, faster convergence

---

### Phase 3: Advanced Features (1-2 months) 📋

**Goal**: Support specialized use cases

**Tasks**:
- [ ] BertForMaskedLM (Task 3.1) - 3-5 days
- [ ] Gradient checkpointing (Task 3.2) - 1 week
- [ ] Mixed precision training (Task 3.3) - 2 weeks
- [ ] BertForNextSentencePrediction (Task 3.4) - 2-3 days
- [ ] BertForPreTraining (Task 3.5) - 5-7 days
- [ ] Model export (Task 3.6) - 2-3 weeks

**Success Metrics**:
- [ ] MLM head enables domain adaptation
- [ ] Gradient checkpointing reduces memory usage
- [ ] Mixed precision speeds up training
- [ ] Pre-training pipeline works end-to-end

**Outcome**: Full BERT lifecycle support

---

## Resource Allocation

### Team Structure (Recommended)

**Core Team** (1-2 engineers):
- Focus on P0 (distributed training) + P1 (task heads)
- Timeline: 5-7 weeks

**Infrastructure Team** (1 engineer):
- Focus on P2 (training utilities)
- Timeline: 2-3 weeks (parallel with P1)

**Research Team** (1 engineer, part-time):
- Focus on P3 (advanced features)
- Timeline: 1-2 months (after P0/P1/P2)

### Timeline Summary

**Short-term** (6-8 weeks):
- Phase 0: Distributed training (3-4 weeks)
- Phase 1: Task heads (2-3 weeks, parallel with Phase 2)
- Phase 2: Training utilities (2-3 weeks, parallel with Phase 1)

**Medium-term** (2-3 months):
- Phase 3: Advanced features (1-2 months)

**Total**: ~3-4 months for full 90% completeness

---

## Success Metrics

### Phase 0 Success (Distributed Training)
- [ ] BERT achieves >90% throughput vs GPT-2/LLaMA on same hardware
- [ ] 8-device TP training produces bit-exact outputs vs reference
- [ ] Memory usage matches theoretical expectations
- [ ] All distributed configs work without issues

### Phase 1 Success (Task Heads)
- [ ] All task heads match HuggingFace (PCC > 0.99)
- [ ] Fine-tuning converges on standard benchmarks (GLUE, CoNLL, SQuAD)
- [ ] Example scripts work out-of-the-box
- [ ] Documentation enables users to fine-tune without help

### Phase 2 Success (Training Utilities)
- [ ] LR schedulers improve convergence speed by 10-20%
- [ ] Gradient clipping prevents training instability
- [ ] Loss functions match PyTorch exactly
- [ ] Checkpointing enables training resumption

### Overall Success (90% Completeness Target)
- [ ] BERT at training parity with GPT-2/LLaMA
- [ ] 80% of common NLP tasks supported out-of-the-box
- [ ] Training experience comparable to HuggingFace Transformers
- [ ] Production-ready for both training and fine-tuning

---

## Risk Mitigation

### Technical Risks

**Risk 1**: Distributed training implementation is complex
- **Mitigation**: Follow proven LLaMA/GPT-2 patterns closely
- **Mitigation**: Start with 2-device TP, validate thoroughly before scaling
- **Mitigation**: Test gradient correctness at each step

**Risk 2**: Task head outputs don't match HuggingFace
- **Mitigation**: Layer-by-layer validation (proven approach from MHA fix)
- **Mitigation**: Use test_bert_isolated_layer_validation.py pattern
- **Mitigation**: Compare intermediate activations, not just final output

**Risk 3**: Memory usage exceeds expectations
- **Mitigation**: Profile memory usage early and often
- **Mitigation**: Implement gradient checkpointing (Task 3.2) if needed
- **Mitigation**: Use smaller batch sizes for testing

### Resource Risks

**Risk 4**: Timeline slips due to unforeseen complexity
- **Mitigation**: Start with P0 (highest priority) and deliver incrementally
- **Mitigation**: P1/P2 can be parallelized
- **Mitigation**: P3 is optional and can be deferred

**Risk 5**: Lack of specialized knowledge (distributed training)
- **Mitigation**: Study existing LLaMA/GPT-2 implementations first
- **Mitigation**: Start with simpler TP before PP
- **Mitigation**: Consult with team members who worked on GPT-2/LLaMA

---

## Decision Log

### Decision 1: Prioritize Distributed Training Over Task Heads
**Rationale**: BERT is the **only** TTML model without distributed training. This is a critical gap for the training-focused framework. Task heads, while useful, are optional extensions (GPT-2/LLaMA don't have them either).

### Decision 2: Follow LLaMA Patterns, Not GPT-2
**Rationale**: LLaMA is more mature (95% completeness) and has cleaner architecture than GPT-2's 3-tier system. BERT's attention mechanism is more similar to LLaMA's than GPT-2's.

### Decision 3: Implement 3 Task Heads, Not All 7
**Rationale**: BertForSequenceClassification, BertForTokenClassification, and BertForQuestionAnswering cover 80% of use cases. The other 4 (MLM, NSP, PreTraining, MultipleChoice) are specialized and can be deferred to P3.

### Decision 4: Training Utilities (P2) Can Run Parallel with Task Heads (P1)
**Rationale**: These are independent work streams. P2 improves training experience for ALL models (not just BERT), so can be done by a separate team member.

### Decision 5: Model Export (ONNX) is Low Priority
**Rationale**: TTML is training-focused, not deployment-focused. Export is useful but not critical for the 90% completeness target. Can be deferred to P3 or later.

---

## References

- **BERT Completeness Analysis**: `/workspace/tt-metal/BERT_IMPLEMENTATION_COMPLETENESS.md`
- **Ecosystem Analysis**: `/workspace/tt-metal/TTML_MODEL_ECOSYSTEM_ANALYSIS.md`
- **Cross-Model Impact Analysis**: `/workspace/tt-metal/CROSS_MODEL_IMPACT_ANALYSIS.md`
- **GPT-2 Implementation**: `tt-train/sources/ttml/models/gpt2.{cpp,hpp}`
- **LLaMA Implementation**: `tt-train/sources/ttml/models/llama.{cpp,hpp}`
- **LLaMA Distributed**: `tt-train/sources/ttml/models/distributed/llama.{cpp,hpp}`
- **GPT-2 Distributed**: `tt-train/sources/ttml/models/distributed/gpt2.{cpp,hpp}`
- **Transformer Common**: `tt-train/sources/ttml/models/common/transformer_common.hpp`
- **Recent BERT Fix**: Commit 4448e84e9b (MHA reshape bug fix)

---

**Next Steps**: Review this action plan with stakeholders, prioritize phases, and allocate resources for Phase 0 (distributed training).
