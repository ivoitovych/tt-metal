# Complete File-by-File Review and Classification

**Total files**: 74

---

## 1. `BERT_IMPLEMENTATION_COMPREHENSIVE_REVIEW__INTERNAL.md`

**File size**: 428 lines

**Git stats**:
```
...MPLEMENTATION_COMPREHENSIVE_REVIEW__INTERNAL.md | 428 +++++++++++++++++++++
 1 file changed, 428 insertions(+)
```

**First 30 lines preview**:
```
# Comprehensive Review of BERT Implementation for TTML

**Overview**: Complete BERT model implementation for Tenstorrent's TTML framework with full support for HuggingFace weight loading from safetensors format. Implementation spans 14 commits from September 2 to October 24, 2025, progressing from basic architecture to production-ready weight loading.

---

## Commit Timeline

### 1. **17420e3d11** - BERT for TTML: Initial commit (Sep 2, 2025)

**Status**: Foundational implementation. Compiles. Not tested.

**Key Components Introduced**:

1. **BertConfig Structure**
   - Comprehensive configuration with 11 parameters
   - Default values matching BERT-base architecture
   - Configurable dropout, layer norm epsilon, and feature flags

2. **Core BERT Architecture**
   - Token embeddings with alignment handling (vocab_size_aligned)
   - Trainable positional embeddings with configurable sequence length
   - Optional token type embeddings for sentence A/B distinction
   - Embedding layer norm and dropout
   - N transformer blocks via BertBlock modules
   - Optional pooler for classification tasks

3. **BertBlock Module**
   - BertMLP: Dense → GELU → Output linear → Dropout
   - BertAttention: Multi-head self-attention with separate output projection

```

**Classification**: [TBD - See analysis below]

---

## 2. `tt-train/BERT_BUG_ROOT_CAUSE_IDENTIFIED.md`

**File size**: 233 lines

**Git stats**:
```
tt-train/BERT_BUG_ROOT_CAUSE_IDENTIFIED.md | 233 +++++++++++++++++++++++++++++
 1 file changed, 233 insertions(+)
```

**First 30 lines preview**:
```
# BERT Forward Pass Bug - Root Cause Identified

**Date**: 2025-10-25
**Status**: 🔴 **ROOT CAUSE IDENTIFIED** with **95% confidence**

---

## 🎯 ROOT CAUSE: Heads Creation Reshape Bug

**File**: `sources/ttml/ops/multi_head_utils.cpp`
**Lines**: 89-98
**Severity**: **CRITICAL**

### The Bug

The manual head splitting reshape **scrambles token and head assignments**:

```cpp
// Current WRONG implementation:
// Reshape: (B, 1, S, E) -> (B, 1, S*H, E/H) -> (B, H, S, E/H)
auto q_reshaped = ttnn::reshape(q_flat, ttnn::Shape{batch_size, 1, seq_len * num_heads, head_dim});
auto q = ttnn::reshape(q_reshaped, ttnn::Shape{batch_size, num_heads, seq_len, head_dim});
```

### Why This Is Wrong

For `batch=1, seq_len=32, embedding_dim=128, num_heads=2, head_dim=64`:

**Input**: `[1, 1, 32, 128]`
- Memory layout: `tok0[d0...d127], tok1[d0...d127], ..., tok31[d0...d127]`

```

**Classification**: [TBD - See analysis below]

---

## 3. `tt-train/BERT_DEBUGGING_PROGRESS.md`

**File size**: 203 lines

**Git stats**:
```
tt-train/BERT_DEBUGGING_PROGRESS.md | 203 ++++++++++++++++++++++++++++++++++++
 1 file changed, 203 insertions(+)
```

**First 30 lines preview**:
```
# BERT Forward Pass Debugging Progress

**Date**: 2025-10-25
**Status**: 🟡 **PARTIAL PROGRESS** - 2 bugs fixed, issue persists

---

## ✅ BUGS FIXED

### Bug #1: Attention Masking (FIXED)
- **File**: `scaled_dot_product_attention.cpp:166`
- **Change**: `1e9F` → `-1e9F`
- **Status**: ✅ Fixed
- **Impact**: Not visible in current tests (no attention masks used)
- **Will matter**: When attention masks are added to tests

### Bug #2: Heads Creation Reshape (FIXED)
- **File**: `multi_head_utils.cpp:89-104`
- **Change**: Replaced incorrect `S*H` reshape with proper transpose-based head splitting
- **Status**: ✅ Fixed
- **Impact**:
  - ✅ Signs now correct (HF: -1.303, TTML: -0.385 vs previous +0.270)
  - ✅ PCC improved slightly: 0.569 → 0.581
  - ❌ Still not good enough (need >0.95)

---

## 📊 CURRENT RESULTS

### Before Any Fixes

```

**Classification**: [TBD - See analysis below]

---

## 4. `tt-train/BERT_FIX_SUMMARY.md`

**File size**: 246 lines

**Git stats**:
```
tt-train/BERT_FIX_SUMMARY.md | 246 +++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 246 insertions(+)
```

**First 30 lines preview**:
```
# BERT Forward Pass - Debugging Summary

**Date**: 2025-10-25
**Current Status**: 🟡 **PARTIAL PROGRESS** - 3 bugs fixed, PCC improved to 0.58

---

## ✅ BUGS FIXED

### Bug #1: Attention Masking Sign Error
- **File**: `scaled_dot_product_attention.cpp:166`
- **Change**: `+1e9F` → `-1e9F`
- **Status**: ✅ Fixed
- **Impact**: Not visible in current tests (no attention masks used)
- **Will matter**: When attention masks are added to tests

### Bug #2: Heads Creation Reshape
- **File**: `multi_head_utils.cpp:89-104`
- **Original Bug**: Incorrect reshape sequence that scrambled tokens
  ```cpp
  // WRONG: Reshaped sequence dimension instead of embedding dimension
  auto q_reshaped = ttnn::reshape(q_flat, ttnn::Shape{batch_size, 1, seq_len * num_heads, head_dim});
  ```
- **Fix**: Proper transpose-based head splitting
  ```cpp
  // CORRECT: Split embedding into heads, then transpose
  auto q_no_channel = ttnn::reshape(q_flat, ttnn::Shape{batch_size, seq_len, embedding_dim});
  auto q_with_heads = ttnn::reshape(q_no_channel, ttnn::Shape{batch_size, seq_len, num_heads, head_dim});
  auto q = ttnn::transpose(q_with_heads, 1, 2);  // [B, H, S, E/H]
  ```

```

**Classification**: [TBD - See analysis below]

---

## 5. `tt-train/BERT_FORWARD_PASS_BUG_REPORT.md`

**File size**: 278 lines

**Git stats**:
```
tt-train/BERT_FORWARD_PASS_BUG_REPORT.md | 278 +++++++++++++++++++++++++++++++
 1 file changed, 278 insertions(+)
```

**First 30 lines preview**:
```
# BERT Forward Pass Bug Report

## Executive Summary

**Issue**: TTML BERT forward pass produces incorrect outputs (PCC 0.17-0.74) despite weights loading correctly (PCC >0.999).

**Root Cause**: LayerNorm epsilon mismatch due to hardware clamping enabled by default.

**Impact**: All BERT models (tiny, small, base) produce wrong inference results.

**Status**: **ROOT CAUSE IDENTIFIED** - Fix required in C++ code.

---

## Problem Description

### Symptoms

When running BERT inference:
- ✅ Weight loading: **PERFECT** (PCC >0.9999 for all layers)
- ❌ Forward pass: **BROKEN** (PCC 0.17-0.74)
- First 5 output values show completely different results:
  ```
  HuggingFace: [-1.303, -0.770, -2.909, -2.297,  0.983]
  TTML:        [ 0.270, -0.828, -4.906, -0.144, -1.188]
  ```

### Evidence

From comprehensive validation tests:

```

**Classification**: [TBD - See analysis below]

---

## 6. `tt-train/BERT_IMPLEMENTATION_CODE_REVIEW.md`

**File size**: 458 lines

**Git stats**:
```
tt-train/BERT_IMPLEMENTATION_CODE_REVIEW.md | 458 ++++++++++++++++++++++++++++
 1 file changed, 458 insertions(+)
```

**First 30 lines preview**:
```
# BERT Implementation Code Review

**Review Date**: 2025-10-25
**Reviewed Commits**: `17420e3d11^..HEAD` (BERT implementation series)
**Status**: ❌ **CRITICAL BUGS FOUND** - Forward pass broken (PCC 0.17-0.74)

---

## 🔴 CRITICAL BUGS FOUND

### **BUG #1: ATTENTION MASK PROCESSING ERROR** (HIGHEST PRIORITY)

**File**: `sources/ttml/models/bert.cpp`
**Lines**: 180-204
**Severity**: **CRITICAL** - This is likely THE primary cause of forward pass failure

```cpp
autograd::TensorPtr Bert::process_attention_mask(const autograd::TensorPtr& attention_mask) const {
    if (!attention_mask) {
        return nullptr;
    }

    // Convert attention mask from (1, 0) to (0, -10000) for additive attention
    // mask = (1 - attention_mask) * -10000
    // This makes padding tokens have very negative scores before softmax
    auto inverted_mask = ops::sub(
        autograd::create_tensor(core::ones(attention_mask->get_shape(), &autograd::ctx().get_device())),
        attention_mask);
    auto processed_mask = ops::mul(inverted_mask, -10000.0F);  // ❌ WRONG VALUE!


```

**Classification**: [TBD - See analysis below]

---

## 7. `tt-train/BERT_INVESTIGATION_SUMMARY.md`

**File size**: 163 lines

**Git stats**:
```
tt-train/BERT_INVESTIGATION_SUMMARY.md | 163 +++++++++++++++++++++++++++++++++
 1 file changed, 163 insertions(+)
```

**First 30 lines preview**:
```
# BERT Forward Pass Investigation Summary

## Current Status (2025-10-25)

### What We Know

1. **Weight Loading**: ✅ **PERFECT** (PCC >0.9999)
   - All weights load correctly from HuggingFace to TTML
   - Token embeddings, QKV weights, FFN weights, LayerNorm parameters all match
   - Verified by `test_bert_comprehensive_validation.py`

2. **Forward Pass**: ❌ **BROKEN** (PCC 0.17-0.74)
   - All BERT model variants produce incorrect outputs
   - bert-tiny: PCC 0.57-0.74
   - bert-small: PCC 0.38
   - bert-base: PCC 0.17
   - First values show opposite signs and large differences

### Investigation Progress

#### Issue #1: LayerNorm Epsilon Clamping (IDENTIFIED)

**Finding**: BERT LayerNorm layers use hardware-clamped epsilon instead of BERT's intended epsilon.

**Details**:
- BERT requires epsilon = 1e-12
- TTML LayerNorm defaults to `enable_hardware_clamp=true` and `min_safe_eps=1e-4`
- For BFLOAT16 tensors, epsilon gets clamped: `max(1e-12, 1e-4) = 1e-4`
- This is **8 orders of magnitude** larger than intended!


```

**Classification**: [TBD - See analysis below]

---

## 8. `tt-train/BERT_QKV_WEIGHT_LOADING_BUG_REPORT__INTERNAL.md`

**File size**: 252 lines

**Git stats**:
```
...BERT_QKV_WEIGHT_LOADING_BUG_REPORT__INTERNAL.md | 252 +++++++++++++++++++++
 1 file changed, 252 insertions(+)
```

**First 30 lines preview**:
```
# BERT QKV Weight Loading Bug Report

**Date:** 2025-10-24
**Branch:** ivoitovych/bert-base-uncased-validation
**Status:** 🚨 CRITICAL BUG - QKV weight loading completely broken

---

## Executive Summary

The QKV weight loading implementation in commit `04d1cf4f3e` is **fundamentally broken**. Despite claims of being "production-ready" with PCC 0.84, actual validation reveals:

- **bert-base-uncased**: PCC = **-0.013** (negative correlation!)
- **bert-tiny**: PCC = **-0.011** (also fails!)
- **Root cause**: Loaded weights are completely wrong, not matching any expected concatenation pattern

The implementation has **never worked correctly** for loading HuggingFace BERT models.

---

## Test Results

### bert-base-uncased
```
Model: bert-base-uncased (12 layers, 768 hidden, 12 heads)
Test: test_bert_qkv_loading_golden_reference[1-32-bert-base-uncased]

Results:
  PCC: -0.012934 (Expected: >0.99)
  Mean absolute diff: 0.943 (Expected: <0.01)

```

**Classification**: [TBD - See analysis below]

---

## 9. `tt-train/DEVELOPMENT_GUIDELINES__INTERNAL.md`

**File size**: 157 lines

**Git stats**:
```
tt-train/DEVELOPMENT_GUIDELINES__INTERNAL.md | 157 +++++++++++++++++++++++++++
 1 file changed, 157 insertions(+)
```

**First 30 lines preview**:
```
# Development Guidelines

## Git Commit Workflow

### Branch Management and Push Strategy

**CRITICAL: Preserve work by creating consequent branches**

**Never force-push feature branches:**
- Feature branches serve as backup checkpoints
- Force-pushing destroys history and can lose significant work
- Only force-push the pull request branch after features are stabilized and merged

**Proper branching workflow:**

1. **Initial work on feature branch:**
   ```bash
   git checkout -b ivoitovych/feature-name
   # Make commits
   git push -u myfork ivoitovych/feature-name
   ```

2. **When making changes (amendments, new commits):**
   ```bash
   # Create new consequent branch to preserve backup
   git checkout -b ivoitovych/feature-name-2
   # Make changes/amendments
   git push -u myfork ivoitovych/feature-name-2
   ```


```

**Classification**: [TBD - See analysis below]

---

## 10. `tt-train/OPERATOR_TESTING_LIMITATIONS.md`

**File size**: 201 lines

**Git stats**:
```
tt-train/OPERATOR_TESTING_LIMITATIONS.md | 201 +++++++++++++++++++++++++++++++
 1 file changed, 201 insertions(+)
```

**First 30 lines preview**:
```
# Operator Testing Limitations - Bottom-Up Approach

**Date**: 2025-10-25
**Status**: 🔴 **BLOCKED** - Most operations not exposed to Python

---

## Goal

Test each BERT operation in isolation with controlled random data to identify which specific operation is broken.

## Approach

1. Generate identical random inputs for PyTorch and TTML
2. Run operation through both implementations
3. Compare outputs with strict PCC threshold (>0.999)
4. Isolate broken operations

## Problem Discovered

**Most BERT internal operations are NOT exposed to Python**. They are only accessible through the full C++ BERT model.

### Operations NOT Accessible from Python:

❌ `heads_creation` - C++ only, returns tuple (Python binding issue)
❌ `heads_fusion` - C++ only
❌ `scaled_dot_product_attention` - C++ only (has Python binding but tuple return fails)
❌ LayerNorm - Module exists but named differently
❌ Direct access to MHA internals


```

**Classification**: [TBD - See analysis below]

---

## 11. `tt-train/OPERATOR_VALIDATION_RESULTS.md`

**File size**: 255 lines

**Git stats**:
```
tt-train/OPERATOR_VALIDATION_RESULTS.md | 255 ++++++++++++++++++++++++++++++++
 1 file changed, 255 insertions(+)
```

**First 30 lines preview**:
```
# BERT Operator Validation Results

**Date**: 2025-10-25
**Approach**: Bottom-up operator validation with Python bindings
**Status**: 🟡 **PARTIALLY SUCCESSFUL** - 6/9 tests passing, 1 critical bug found

---

## Summary

Successfully implemented comprehensive Python bindings for BERT operations and created extensive test suite. Testing revealed **excellent performance for most operations** but identified a **critical masking bug** in scaled_dot_product_attention.

---

## Implementation Completed

### 1. Python Bindings Added (`sources/ttml/nanobind/nb_ops.cpp`)

**Added includes:**
- `<nanobind/stl/tuple.h>` - Enables automatic std::tuple to Python tuple conversion
- `"ops/scaled_dot_product_attention.hpp"` - Attention operation header

**New bindings added:**
```cpp
// Multi-head attention utilities
py_multi_head_utils.def("scaled_dot_product_attention", ...);  // NEW
py_multi_head_utils.def("scaled_sigmoid_dot_product_attention", ...);  // NEW

// Unary operations
py_unary.def("tanh", &ttml::ops::tanh, ...);  // NEW

```

**Classification**: [TBD - See analysis below]

---

## 12. `tt-train/WEIGHT_LOADING_INVESTIGATION_RESULTS__INTERNAL.md`

**File size**: 216 lines

**Git stats**:
```
...IGHT_LOADING_INVESTIGATION_RESULTS__INTERNAL.md | 216 +++++++++++++++++++++
 1 file changed, 216 insertions(+)
```

**First 30 lines preview**:
```
# BERT Weight Loading Investigation - RESOLVED

**Date:** 2025-10-24
**Branch:** ivoitovych/bert-model-for-ttml-qkv-weight-loading-3
**Status:** ✅ RESOLVED - No bug found, issue was test environment

---

## Executive Summary

**Initial Concern:** Bug report showed BERT QKV weight loading completely broken with negative PCC values.

**Investigation Result:** ✅ **NO BUG EXISTS** - Weight loading works correctly. The issue was that Python tests were using an outdated compiled module that hadn't been properly rebuilt.

**Resolution:** Clean rebuild of the Python module (`_ttml.so`) resolves all issues.

---

## Test Results After Rebuild

### bert-tiny (prajjwal1/bert-tiny)
```
✅ Token Embeddings: PCC 0.999999 (was 0.000669 before rebuild)
✅ QKV Weights:      PCC 0.999999 (was -0.001452 before rebuild)
❌ Final Output:     PCC 0.836250 (acceptable per reviewer feedback)
```

### bert-base-uncased
```
✅ Token Embeddings: PCC 0.999998

```

**Classification**: [TBD - See analysis below]

---

## 13. `tt-train/sources/ttml/models/bert.cpp`

**File size**: 714 lines

**Git stats**:
```
tt-train/sources/ttml/models/bert.cpp | 67 +++++++++++++++++++++++++++++++++--
 1 file changed, 65 insertions(+), 2 deletions(-)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bert.hpp"

#include <yaml-cpp/yaml.h>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "modules/bert_block.hpp"
#include "modules/dropout_module.hpp"
#include "modules/embedding_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/positional_embeddings.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "serialization/safetensors.hpp"
#include "serialization/serializable.hpp"

namespace ttml::models::bert {

Bert::Bert(const BertConfig& config) : m_config(config), m_runner_type(config.runner_type) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    uint32_t intermediate_size = config.intermediate_size;

```

**Classification**: [TBD - See analysis below]

---

## 14. `tt-train/sources/ttml/models/bert.hpp`

**File size**: 122 lines

**Git stats**:
```
tt-train/sources/ttml/models/bert.hpp | 28 +++++++++++++++++++++++++++-
 1 file changed, 27 insertions(+), 1 deletion(-)
```

**First 30 lines preview**:
```
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

```

**Classification**: [TBD - See analysis below]

---

## 15. `tt-train/sources/ttml/modules/bert_block.cpp`

**File size**: 112 lines

**Git stats**:
```
tt-train/sources/ttml/modules/bert_block.cpp | 29 ++++++++++++++++++++++++++--
 1 file changed, 27 insertions(+), 2 deletions(-)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bert_block.hpp"

#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_head_attention.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

BertMLP::BertMLP(uint32_t embedding_dim, uint32_t intermediate_size, float dropout_prob) {
    m_dense = std::make_shared<LinearLayer>(embedding_dim, intermediate_size);
    m_output = std::make_shared<LinearLayer>(intermediate_size, embedding_dim);
    m_dropout = std::make_shared<DropoutLayer>(dropout_prob);

    create_name("bert_mlp");
    register_module(m_dense, "dense");
    register_module(m_output, "output");
    register_module(m_dropout, "dropout");
}

autograd::TensorPtr BertMLP::operator()(const autograd::TensorPtr& input) {
    auto x = (*m_dense)(input);
    x = ops::gelu(x);  // BERT uses GELU activation
    x = (*m_output)(x);

```

**Classification**: [TBD - See analysis below]

---

## 16. `tt-train/sources/ttml/modules/bert_block.hpp`

**File size**: 71 lines

**Git stats**:
```
tt-train/sources/ttml/modules/bert_block.hpp | 10 ++++++++++
 1 file changed, 10 insertions(+)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/multi_head_attention.hpp"

namespace ttml::modules {

struct BertBlockConfig {
    uint32_t embedding_dim{};
    uint32_t intermediate_size{};
    uint32_t num_heads{};
    float dropout_prob{};
    float layer_norm_eps{1e-12F};
};

class BertMLP : public ModuleBase {
private:
    std::shared_ptr<LinearLayer> m_dense;
    std::shared_ptr<LinearLayer> m_output;
    std::shared_ptr<DropoutLayer> m_dropout;

public:

```

**Classification**: [TBD - See analysis below]

---

## 17. `tt-train/sources/ttml/modules/multi_head_attention.cpp`

**File size**: 55 lines

**Git stats**:
```
.../sources/ttml/modules/multi_head_attention.cpp  | 22 +++++++++++-----------
 1 file changed, 11 insertions(+), 11 deletions(-)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_attention.hpp"

#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

MultiHeadAttention::MultiHeadAttention(uint32_t embedding_dim_, uint32_t num_heads_, float dropout_prob_) :
    m_embedding_dim(embedding_dim_), m_num_heads(num_heads_) {
    // create layers
    m_qkv_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim * 3);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(dropout_prob_);
    m_out_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim);

    // register modules
    create_name("multi_head_attention");
    register_module(m_qkv_linear, "qkv_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
}

ttml::autograd::TensorPtr MultiHeadAttention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    // fmt::print("[MHA] Input shape: {}\n", x->get_value().logical_shape());
    // fmt::print("[MHA] embedding_dim: {}, num_heads: {}\n", m_embedding_dim, m_num_heads);


```

**Classification**: [TBD - See analysis below]

---

## 18. `tt-train/sources/ttml/nanobind/nb_models.cpp`

**File size**: 358 lines

**Git stats**:
```
tt-train/sources/ttml/nanobind/nb_models.cpp | 64 ++++++++++++++++++++++++++++
 1 file changed, 64 insertions(+)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "models/base_transformer.hpp"
#include "models/bert.hpp"
#include "models/distributed/gpt2.hpp"
#include "models/distributed/llama.hpp"
#include "models/distributed/pipeline_parallel_llama.hpp"
#include "models/gpt2.hpp"
#include "models/linear_regression.hpp"
#include "models/llama.hpp"
#include "models/mlp.hpp"
#include "modules/module_base.hpp"
#include "modules/multi_layer_perceptron.hpp"
#include "nb_export_enum.hpp"
#include "nb_fwd.hpp"
#include "nb_modules.hpp"

namespace ttml::nanobind::models {
using namespace ttml::models;


```

**Classification**: [TBD - See analysis below]

---

## 19. `tt-train/sources/ttml/nanobind/nb_ops.cpp`

**File size**: 312 lines

**Git stats**:
```
tt-train/sources/ttml/nanobind/nb_ops.cpp | 23 +++++++++++++++++++++++
 1 file changed, 23 insertions(+)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>

#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
#include "nb_export_enum.hpp"
#include "nb_fwd.hpp"
#include "ops/binary_ops.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/dropout_op.hpp"
#include "ops/embedding_op.hpp"
#include "ops/layernorm_op.hpp"
#include "ops/linear_op.hpp"
#include "ops/losses.hpp"
#include "ops/matmul_op.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/rmsnorm_op.hpp"
#include "ops/rope_op.hpp"
#include "ops/sampling_op.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::nanobind::ops {
using namespace ttml::ops;


```

**Classification**: [TBD - See analysis below]

---

## 20. `tt-train/sources/ttml/nanobind/nb_util.cpp`

**File size**: 552 lines

**Git stats**:
```
tt-train/sources/ttml/nanobind/nb_util.cpp | 140 +++++++++++++++++++++++++++++
 1 file changed, 140 insertions(+)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_util.hpp"

#include <nanobind/nanobind.h>

#include "autograd/auto_context.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttml::nanobind::util {

namespace UnsupportedMessages {

constexpr auto BFLOAT8_B = "Unsupported type: BFLOAT8_B";
constexpr auto BFLOAT4_B = "Unsupported type: BFLOAT4_B";
constexpr auto UINT8 = "Unsupported type: UINT8";
constexpr auto UINT16 = "Unsupported type: UINT16";
constexpr auto INVALID = "Unsupported type: INVALID";
constexpr auto UNKNOWN = "Unsupported type: unknown";
constexpr auto COMPLEX = "Unsupported type: Complex";
constexpr auto BOOL = "Unsupported type: Bool";
constexpr auto BFLOAT = "Unsupported type: Bfloat";

}  // namespace UnsupportedMessages

```

**Classification**: [TBD - See analysis below]

---

## 21. `tt-train/sources/ttml/ops/multi_head_utils.cpp`

**File size**: 227 lines

**Git stats**:
```
tt-train/sources/ttml/ops/multi_head_utils.cpp | 94 ++++++++++++++++----------
 1 file changed, 57 insertions(+), 37 deletions(-)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_utils.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

#ifdef nlp_create_qkv_heads_program_factory_bug_fixed

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> heads_creation(
    const autograd::TensorPtr& qkv, uint32_t num_heads) {
    // qkv shape is (B, 1, S, E * 3)
    // q, k, v shapes are (B, num_heads, S, E / num_heads)
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        qkv->get_value(),
        std::nullopt,
        num_heads,
        num_heads,
        /* transpose_k */ false,
        /* memory_config */ std::nullopt,
        /* optional_output_tensors */ std::nullopt);


```

**Classification**: [TBD - See analysis below]

---

## 22. `tt-train/sources/ttml/ops/scaled_dot_product_attention.cpp`

**File size**: 313 lines

**Git stats**:
```
tt-train/sources/ttml/ops/scaled_dot_product_attention.cpp | 7 +++++--
 1 file changed, 5 insertions(+), 2 deletions(-)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scaled_dot_product_attention.hpp"

#include <cmath>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/operations.hpp"
#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {
namespace {

// Wrapper around matmul to handle sharing of KV heads across groups of query
// heads.
// For e.g. Q @ V, there are two cases:
// - G == H: (B, H, S, S) x (B, H, S, V) -> (B, H, S, V)
// - G != H:
//    - In this case value has shape (B,G,S,V):
//      1. Reshape attention_weights to (B*G, H/G, S, S).
//      2. Reshape value to (B*G, 1, S, V).
//      3. Manually broadcast values over groupsize.
//      4. Matmul.
//      5. Reshape the result to (B, H, S, V).

```

**Classification**: [TBD - See analysis below]

---

## 23. `tt-train/tests/CMakeLists.txt`

**File size**: 114 lines

**Git stats**:
```
tt-train/tests/CMakeLists.txt | 2 ++
 1 file changed, 2 insertions(+)
```

**First 30 lines preview**:
```
include(CTest)
enable_testing()

set(SOURCES
    core/tensor_utils_test.cpp
    core/n300_utils_test.cpp
    core/scoped_test.cpp
    core/clip_grad_norm_test.cpp
    core/tile_layout_round_trip_test.cpp
    model/linear_regression_ddp_test.cpp
    model/nano_gpt_test.cpp
    model/model_names_test.cpp
    model/gpt2s_test.cpp
    model/linear_regression_full_test.cpp
    model/weight_tying_test.cpp
    model/bert_polymorphism_test.cpp
    model/bert_weight_loading_test.cpp
    model/bert_operator_test.cpp
    ttnn_fixed/concat_op_test.cpp
    ttnn_fixed/distributed/distributed_ttnn_ops_test.cpp
    ttnn_fixed/trivial_ttnn_ops_test.cpp
    ttnn_fixed/matmuls_test.cpp
    ttnn_fixed/reduce_ops_test.cpp
    ttnn_fixed/dropout_op_test.cpp
    ttnn_fixed/slice_op_test.cpp
    autograd/autograd_tensor.cpp
    autograd/autograd_test.cpp
    autograd/module_base_parameters_test.cpp
    schedulers/schedulers_test.cpp
    tokenizers/bpe_tokenizer_test.cpp

```

**Classification**: [TBD - See analysis below]

---

## 24. `tt-train/tests/core/tile_layout_round_trip_test.cpp`

**File size**: 307 lines

**Git stats**:
```
.../tests/core/tile_layout_round_trip_test.cpp     | 307 +++++++++++++++++++++
 1 file changed, 307 insertions(+)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * TILE Layout Round-Trip Test
 *
 * This test demonstrates a critical bug in TILE layout conversion operations.
 *
 * BUG: Structured data is scrambled during ROW_MAJOR -> TILE -> ROW_MAJOR conversion,
 * while random data passes through correctly.
 *
 * Test procedure:
 * 1. Create structured data (sequential pattern)
 * 2. Create random data (same shape)
 * 3. Convert both to device tensors (ROW_MAJOR -> TILE layout)
 * 4. Convert back to host (TILE -> ROW_MAJOR layout)
 * 5. Compare input vs output
 *
 * Expected: Both should have PCC > 0.99
 * Actual: Random data works (PCC ~1.0), structured data fails (PCC ~0.1-0.3)
 *
 * This bug affects all real learned weights in neural networks.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

```

**Classification**: [TBD - See analysis below]

---

## 25. `tt-train/tests/model/bert_operator_test.cpp`

**File size**: 889 lines

**Git stats**:
```
tt-train/tests/model/bert_operator_test.cpp | 889 ++++++++++++++++++++++++++++
 1 file changed, 889 insertions(+)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Comprehensive BERT Operator Unit Tests
 *
 * Tests each BERT operation in isolation using C++ directly.
 * This implements Option 2 from OPERATOR_TESTING_LIMITATIONS.md
 *
 * These tests verify:
 * 1. Heads creation (QKV splitting)
 * 2. Heads fusion (merging heads back)
 * 3. Scaled dot-product attention
 * 4. Complete MHA pipeline
 * 5. LayerNorm
 * 6. GELU activation
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/layernorm_op.hpp"

```

**Classification**: [TBD - See analysis below]

---

## 26. `tt-train/tests/model/bert_real_data_test.cpp`

**File size**: 97 lines

**Git stats**:
```
tt-train/tests/model/bert_real_data_test.cpp | 97 ++++++++++++++++++++++++++++
 1 file changed, 97 insertions(+)
```

**First 30 lines preview**:
```
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Real Data Test - NO PYTHON BINDINGS
 *
 * This test uses REAL data extracted from prajjwal1/bert-tiny and embedded
 * directly in C++ to bypass all Python bindings and test the C++ implementation
 * directly with real learned BERT weights.
 *
 * This will tell us definitively if the issue is in:
 * - C++ computation (if this test fails)
 * - Python bindings (if this test passes but Python tests fail)
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

using namespace ttml;

namespace {


```

**Classification**: [TBD - See analysis below]

---

## 27. `tt-train/tests/python/BERT_DTYPE_FIX_RESULTS.md`

**File size**: 180 lines

**Git stats**:
```
tt-train/tests/python/BERT_DTYPE_FIX_RESULTS.md | 180 ++++++++++++++++++++++++
 1 file changed, 180 insertions(+)
```

**First 30 lines preview**:
```
# BERT Dtype Fix Results - Dramatic Improvement

**Date**: 2025-10-28
**Fix**: Changed input_ids and token_type_ids from `float32` to `uint32`
**Root Cause**: TTNN embedding lookup requires UINT32 integer indices, not float32

## Executive Summary

**COMPLETE SUCCESS**: Changing the dtype from float32 to uint32 fixed all embedding layer failures across all models and hidden dimensions.

All models now achieve **PCC > 0.9999 (near perfect)** for both embeddings and transformer blocks.

## Before vs After Comparison

### bert-tiny (128-dim, 2 layers)

| Component | Before PCC | After PCC | Improvement |
|-----------|-----------|----------|-------------|
| Embeddings | 0.956 (borderline) | **0.999977** | 0.044 |
| Block 0 | 0.999979 | 0.999979 | - |
| Block 1 | 0.999968 | 0.999968 | - |

**Embeddings mean diff**: 0.157 → 0.003 (**50x improvement**)

---

### bert-small (512-dim, 4 layers)

| Component | Before PCC | After PCC | Improvement |
|-----------|-----------|----------|-------------|

```

**Classification**: [TBD - See analysis below]

---

## 28. `tt-train/tests/python/BERT_ISOLATED_LAYER_VALIDATION_RESULTS.md`

**File size**: 235 lines

**Git stats**:
```
.../BERT_ISOLATED_LAYER_VALIDATION_RESULTS.md      | 235 +++++++++++++++++++++
 1 file changed, 235 insertions(+)
```

**First 30 lines preview**:
```
# BERT Isolated Layer Validation Results

**Date**: 2025-10-27
**Test**: `test_bert_isolated_layer_validation.py`
**Method**: True isolation - each layer tested with reference inputs from HuggingFace

## Executive Summary

**BREAKTHROUGH FINDING**: When transformer blocks receive reference inputs from HuggingFace, they achieve **PCC > 0.9999 (near perfect)** across ALL models and ALL hidden dimensions (128, 512, 768).

**The root cause is NOT the transformer blocks. The root cause is the embedding layer for 512-dim models.**

## Test Methodology

Unlike cumulative validation where errors propagate:
1. Run HuggingFace BERT to capture intermediate tensors at every layer
2. Feed those **reference tensors** to corresponding TTML layers
3. Compare outputs to isolate each layer's intrinsic accuracy

This reveals whether layers are individually broken or if errors accumulate.

## Results Summary

| Model | Hidden Dim | Embeddings PCC | Blocks PCC (avg) | Result |
|-------|-----------|---------------|------------------|--------|
| bert-tiny | 128 | **0.956** (borderline) | **0.9999+** (perfect) | Embeddings weak |
| bert-small | 512 | **0.653** (FAIL) | **0.9999+** (perfect) | Embeddings broken |
| google/bert | 512 | **0.653** (FAIL) | **0.9999+** (perfect) | Embeddings broken |
| bert-base | 768 | **0.977** (good) | **0.9999+** (perfect) | All layers good |


```

**Classification**: [TBD - See analysis below]

---

## 29. `tt-train/tests/python/BERT_LAYER_PCC_REPORT.txt`

**File size**: 68 lines

**Git stats**:
```
tt-train/tests/python/BERT_LAYER_PCC_REPORT.txt | 68 +++++++++++++++++++++++++
 1 file changed, 68 insertions(+)
```

**First 30 lines preview**:
```
BERT LAYER-BY-LAYER PCC REPORT
====================================================================================================

Comparing HuggingFace reference vs TTML implementation
Note: Each layer receives output from previous TTML layer (cumulative errors)
====================================================================================================

Processing prajjwal1/bert-tiny...

Processing prajjwal1/bert-small...

Processing google/bert_uncased_L-4_H-512_A-8...

Processing bert-base-uncased...

====================================================================================================
Layer                     bert-tiny     bert-smallbert_uncased_L-bert-base-uncas
----------------------------------------------------------------------------------------------------
Embeddings            ✅ 0.9557      ❌ 0.6531      ❌ 0.6531      ✅ 0.9765
Block 0 Attn          ✅ 0.9644      ❌ 0.8495      ❌ 0.8495      ❌ 0.9302
Block 0 Out           ✅ 0.9691      ❌ 0.7940      ❌ 0.7940      ❌ 0.8924
Block 1 Attn          ❌ 0.9404      ❌ 0.8687      ❌ 0.8687      ❌ 0.6707
Block 1 Out           ❌ 0.9493      ❌ 0.7578      ❌ 0.7578      ❌ 0.6106
Final                 ❌ 0.9493      ❌ 0.8461      ❌ 0.8461      ❌ 0.5646
Block 2 Out                          ❌ 0.7791      ❌ 0.7791      ❌ 0.4058
Block 3 Attn                         ❌ 0.6756      ❌ 0.6756      ❌ 0.4396
Block 3 Out                          ❌ 0.5969      ❌ 0.5969      ❌ 0.2906
Final                                ❌ 0.5969      ❌ 0.5969      ❌ 0.3781
Block 4 Out                                                        ❌ 0.2227
Block 5 Attn                                                       ❌ 0.2281

```

**Classification**: [TBD - See analysis below]

---

## 30. `tt-train/tests/python/BERT_MULTI_MODEL_VALIDATION_RESULTS.md`

**File size**: 305 lines

**Git stats**:
```
.../python/BERT_MULTI_MODEL_VALIDATION_RESULTS.md  | 305 +++++++++++++++++++++
 1 file changed, 305 insertions(+)
```

**First 30 lines preview**:
```
# BERT Multi-Model Layer-by-Layer Validation Results

**Date**: 2025-10-26
**Purpose**: Comprehensive validation of TTML BERT implementation across 4 model variants
**Test**: `test_bert_layer_by_layer_multi_model.py`

## Executive Summary

Layer-by-layer validation across 4 BERT model variants reveals **dimension-dependent divergence**:
- **bert-tiny (128 hidden)**: Works well (avg PCC=0.952)
- **bert-small (512 hidden)**: Fails at embeddings (avg PCC=0.759)
- **bert-base (768 hidden)**: Catastrophic accumulation (avg PCC=0.458)

## Test Configuration

**Models Tested**:
1. `prajjwal1/bert-tiny`: 2 layers, 128 hidden, 2 heads
2. `prajjwal1/bert-small`: 4 layers, 512 hidden, 8 heads
3. `google/bert_uncased_L-4_H-512_A-8`: 4 layers, 512 hidden, 8 heads (official)
4. `bert-base-uncased`: 12 layers, 768 hidden, 12 heads

**Input Configuration**:
- Batch size: 1
- Sequence length: 32
- Input IDs: `[542, 67, 876, 414, 26, 335, 620, 924, 950, 113, ...]` (deterministic)
- Pass threshold: PCC ≥ 0.95

**Layers Validated Per Model**:
- Embedding layer output
- Each block's attention output (after attention + residual + norm)

```

**Classification**: [TBD - See analysis below]

---

## 31. `tt-train/tests/python/BERT_TILE_LAYOUT_BUG_INVESTIGATION.md`

**File size**: 139 lines

**Git stats**:
```
.../python/BERT_TILE_LAYOUT_BUG_INVESTIGATION.md   | 139 +++++++++++++++++++++
 1 file changed, 139 insertions(+)
```

**First 30 lines preview**:
```
# BERT Forward Pass Investigation - ROOT CAUSE FOUND

## Summary

The BERT forward pass fails with real learned weights (PCC=0.38) due to a **critical bug in TILE layout conversion** (tilize/untilize operations). Real BERT Q,K,V data is completely scrambled during the round-trip through TTML tensors, **before any computation happens**.

## Timeline of Investigation

### 1. Initial Issue
- ✅ Synthetic random data: PCC > 0.99 (works perfectly)
- ❌ Real BERT data: PCC = 0.38-0.65 (fails completely)

### 2. Fixed Attention Masking Bug
- **Bug**: `(mask - 1) * (-1e9F)` gave +1e9 for padding (double negative)
- **Fix**: Changed to `(mask - 1) * (+1e9F)` = -1e9 for padding
- **Result**: C++ tests pass with PCC > 0.999
- **However**: Real BERT data still fails

### 3. Ruled Out Precision Issues
- Attempted to use `PreferredPrecision::FULL` instead of `HALF`
- **Found**: Device operations (softmax, etc.) require BFLOAT16
- **Conclusion**: Cannot use FLOAT32 for computations, stuck with BFLOAT16

###  4. **ROOT CAUSE FOUND: TILE Layout Bug**

#### Test Results: Data Round-Trip (No Computation)

**Synthetic Random Data:**
```
Mean abs diff: 2.677e-04

```

**Classification**: [TBD - See analysis below]

---

## 32. `tt-train/tests/python/TEST_RESULTS_AFTER_NON_CONTIGUOUS_FIX.md`

**File size**: 191 lines

**Git stats**:
```
.../TEST_RESULTS_AFTER_NON_CONTIGUOUS_FIX.md       | 191 +++++++++++++++++++++
 1 file changed, 191 insertions(+)
```

**First 30 lines preview**:
```
# Test Results After Non-Contiguous Array Fix (Commit 38f05bb43c)

## Executive Summary
✅ **ROOT CAUSE FIX SUCCESSFUL**: Non-contiguous numpy array handling fixed
✅ **OPERATOR LEVEL**: All operators work perfectly (PCC ≥ 0.9999)
⚠️ **FULL MODEL**: Forward pass still shows issues (PCC ~0.49)

## Detailed Test Results

### 1. test_numpy_contiguity.py (THE SMOKING GUN TEST)
**Before Fix:**
- Non-contiguous array: PCC = 0.127 ❌ (data scrambled)
- Contiguous array: PCC = 1.0 ✅

**After Fix:**
- Non-contiguous array: PCC = 1.000000 ✅
- Contiguous array: PCC = 1.000000 ✅
- Mean precision loss: ~2.2e-4 (expected for BFLOAT16)

**Status:** ✅ FIXED - Both contiguous and non-contiguous arrays work perfectly

---

### 2. test_bert_data_roundtrip.py (REAL vs SYNTHETIC DATA)
**Before Fix:**
- Real BERT Q: PCC = 0.127 (2784x worse than synthetic!)
- Real BERT K: PCC = 0.023
- Real BERT V: PCC = 0.015
- Synthetic Q: PCC = 1.0


```

**Classification**: [TBD - See analysis below]

---

## 33. `tt-train/tests/python/TEST_SUITE_INVENTORY__INTERNAL.md`

**File size**: 337 lines

**Git stats**:
```
.../tests/python/TEST_SUITE_INVENTORY__INTERNAL.md | 337 +++++++++++++++++++++
 1 file changed, 337 insertions(+)
```

**First 30 lines preview**:
```
# BERT Test Suite Inventory

This document catalogs all test files created during the BERT weight loading investigation. These files can serve as the foundation for a comprehensive BERT test suite.

---

## Diagnostic Scripts (Standalone)

### 1. `debug_weight_loading_pipeline.py`
**Purpose:** Traces the entire weight loading pipeline step-by-step

**What it tests:**
- Roundtrip test (store → retrieve)
- Token embedding loading before/after
- QKV weight loading before/after
- Compares with HuggingFace at each stage

**Usage:**
```bash
python3 tests/python/debug_weight_loading_pipeline.py
```

**Key features:**
- Detailed statistics at each stage (mean, std, min, max)
- Point-by-point value comparison
- Safetensors validation

---

### 2. `quick_check_weights.py`

```

**Classification**: [TBD - See analysis below]

---

## 34. `tt-train/tests/python/check_tensor_dtypes.py`

**File size**: 106 lines

**Git stats**:
```
tt-train/tests/python/check_tensor_dtypes.py | 106 +++++++++++++++++++++++++++
 1 file changed, 106 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Check what data types TTML tensors are using internally.
"""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def main():
    print("\n" + "=" * 80)
    print("CHECKING TTML TENSOR DATA TYPES")
    print("=" * 80)

    # Create a simple float32 array
    data_np = np.array([[1.0, 2.5, 3.7, 4.123456789]], dtype=np.float32)
    print(f"\nNumPy input:")
    print(f"  dtype: {data_np.dtype}")
    print(f"  values: {data_np}")

    # Convert to TTML tensor
    tensor_ttml = ttml.autograd.Tensor.from_numpy(data_np)
    print(f"\nTTML tensor created")

    # Try to inspect the tensor
    print(f"  Tensor type: {type(tensor_ttml)}")

```

**Classification**: [TBD - See analysis below]

---

## 35. `tt-train/tests/python/compare_bert_qkv_extraction.py`

**File size**: 185 lines

**Git stats**:
```
.../tests/python/compare_bert_qkv_extraction.py    | 185 +++++++++++++++++++++
 1 file changed, 185 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Compare Q, K, V extraction methods to ensure we're getting the right values.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def main():
    print("=" * 80)
    print("COMPARING BERT Q, K, V EXTRACTION METHODS")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    text = "The quick brown fox jumps."
    encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]


```

**Classification**: [TBD - See analysis below]

---

## 36. `tt-train/tests/python/compare_loaded_weights.py`

**File size**: 69 lines

**Git stats**:
```
tt-train/tests/python/compare_loaded_weights.py | 69 +++++++++++++++++++++++++
 1 file changed, 69 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""Compare what TTML loaded vs what it should have loaded."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

# Load expected QKV
qkv_expected = np.load("/tmp/qkv_expected.npy")
print(f"Expected QKV shape: {qkv_expected.shape}")
print(f"Expected QKV[0,0]: {qkv_expected[0,0]}")
print(f"Expected QKV[768,0]: {qkv_expected[768,0]}")
print(f"Expected QKV[1536,0]: {qkv_expected[1536,0]}")

# Load TTML model (already created with correct weights)
import transformers
from pathlib import Path

model_name = "bert-base-uncased"
hf_model = transformers.BertModel.from_pretrained(model_name)
hf_config = hf_model.config

ttml_config = ttml.models.bert.BertConfig()
ttml_config.vocab_size = hf_config.vocab_size
ttml_config.max_sequence_length = 32
ttml_config.embedding_dim = hf_config.hidden_size
ttml_config.intermediate_size = hf_config.intermediate_size

```

**Classification**: [TBD - See analysis below]

---

## 37. `tt-train/tests/python/debug_attention_intermediate_steps.py`

**File size**: 198 lines

**Git stats**:
```
.../python/debug_attention_intermediate_steps.py   | 198 +++++++++++++++++++++
 1 file changed, 198 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Debug attention computation step-by-step with real BERT data.

This test compares each intermediate step of the attention computation
to identify exactly where the divergence occurs.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x, y):
    """Compute Pearson correlation coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


def main():
    print("\n" + "=" * 80)
    print("DEBUGGING ATTENTION: Step-by-Step Intermediate Results")
    print("=" * 80)

```

**Classification**: [TBD - See analysis below]

---

## 38. `tt-train/tests/python/debug_attention_masking.py`

**File size**: 213 lines

**Git stats**:
```
tt-train/tests/python/debug_attention_masking.py | 213 +++++++++++++++++++++++
 1 file changed, 213 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Debug Attention Masking - Inspect intermediate values

This script extracts intermediate values from the attention computation
to see exactly where the masking is going wrong.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def extract_real_qkv_and_mask():
    """Extract REAL Q, K, V tensors and attention mask from BERT."""
    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    text = "The quick brown fox jumps."
    encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():

```

**Classification**: [TBD - See analysis below]

---

## 39. `tt-train/tests/python/debug_bert_forward_pass.py`

**File size**: 266 lines

**Git stats**:
```
tt-train/tests/python/debug_bert_forward_pass.py | 266 +++++++++++++++++++++++
 1 file changed, 266 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Debug BERT Forward Pass - Layer by Layer Comparison

Systematically compares HuggingFace vs TTML outputs at each stage:
1. Embeddings (token + position + token_type)
2. Embedding LayerNorm
3. Each transformer block (attention, FFN, layer norms)
4. Final output

Goal: Identify exactly where the divergence starts.
"""

import numpy as np
import os
import sys
from pathlib import Path
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    mean_x = np.mean(x_flat)
    mean_y = np.mean(y_flat)

```

**Classification**: [TBD - See analysis below]

---

## 40. `tt-train/tests/python/debug_qkv_loading.py`

**File size**: 132 lines

**Git stats**:
```
tt-train/tests/python/debug_qkv_loading.py | 132 +++++++++++++++++++++++++++++
 1 file changed, 132 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""Debug script to inspect QKV weight loading."""

import numpy as np
import torch
import transformers
from pathlib import Path
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

model_name = "bert-base-uncased"

# Load HF model
print("Loading HuggingFace BERT...")
hf_model = transformers.BertModel.from_pretrained(model_name)
hf_config = hf_model.config

# Save to safetensors
safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
if not safetensors_path.exists():
    from safetensors.torch import save_file

    save_file(hf_model.state_dict(), str(safetensors_path))

# Create TTML model
print("Creating TTML BERT...")
ttml_config = ttml.models.bert.BertConfig()

```

**Classification**: [TBD - See analysis below]

---

## 41. `tt-train/tests/python/debug_weight_loading_pipeline.py`

**File size**: 252 lines

**Git stats**:
```
.../tests/python/debug_weight_loading_pipeline.py  | 252 +++++++++++++++++++++
 1 file changed, 252 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Debug script to trace weight loading corruption through the entire pipeline.

This script tests each stage of weight loading to identify where corruption occurs:
1. Safetensors file reading
2. Python float vector
3. core::from_vector storage
4. to_numpy() retrieval
"""

import numpy as np
import torch
import transformers
from pathlib import Path
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

from safetensors import safe_open


def analyze_weights(weights: np.ndarray, name: str):
    """Print detailed statistics about weights."""
    print(f"\n{name}:")
    print(f"  Shape: {weights.shape}")
    print(f"  Dtype: {weights.dtype}")
    print(f"  Mean: {weights.mean():.6f}")

```

**Classification**: [TBD - See analysis below]

---

## 42. `tt-train/tests/python/extract_bert_data_to_cpp.py`

**File size**: 120 lines

**Git stats**:
```
tt-train/tests/python/extract_bert_data_to_cpp.py | 120 ++++++++++++++++++++++
 1 file changed, 120 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Extract real BERT Q, K, V data and output as C++ arrays.
This allows us to test with real data directly in C++ without any Python bindings.
"""

import numpy as np
import torch
import transformers


def array_to_cpp(arr: np.ndarray, name: str) -> str:
    """Convert numpy array to C++ initializer list."""
    flat = arr.flatten()

    # Format as C++ array with proper line breaks
    values = []
    for i, val in enumerate(flat):
        if i > 0 and i % 8 == 0:
            values.append(f"\n        {val:.8f}F")
        else:
            values.append(f"{val:.8f}F")

    cpp_code = f"    // Shape: {list(arr.shape)}\n"
    cpp_code += f"    std::vector<float> {name} = {{\n        "
    cpp_code += ", ".join(values)
    cpp_code += "\n    };\n"

    return cpp_code


```

**Classification**: [TBD - See analysis below]

---

## 43. `tt-train/tests/python/inspect_safetensors.py`

**File size**: 51 lines

**Git stats**:
```
tt-train/tests/python/inspect_safetensors.py | 51 ++++++++++++++++++++++++++++
 1 file changed, 51 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""Inspect what's actually in the safetensors file."""

import numpy as np
from safetensors import safe_open

safetensors_path = "/tmp/bert-base-uncased.safetensors"

print("Opening safetensors file...")
with safe_open(safetensors_path, framework="numpy") as f:
    # List all keys
    keys = f.keys()
    print(f"Total keys: {len(list(keys))}")
    print("\nLayer 0 attention keys:")
    for key in f.keys():
        if "layer.0.attention" in key:
            print(f"  {key}")

    # Check Q, K, V for layer 0
    q_key = "encoder.layer.0.attention.self.query.weight"
    k_key = "encoder.layer.0.attention.self.key.weight"
    v_key = "encoder.layer.0.attention.self.value.weight"

    print(f"\nKey: {q_key}")
    q_weight = f.get_tensor(q_key)
    print(f"Shape: {q_weight.shape}")
    print(f"Dtype: {q_weight.dtype}")
    print(f"Q[0,0]: {q_weight[0, 0]}")
    print(f"Q[0,:5]: {q_weight[0, :5]}")
    print(f"Q[:5,0]: {q_weight[:5, 0]}")

```

**Classification**: [TBD - See analysis below]

---

## 44. `tt-train/tests/python/quick_check_weights.py`

**File size**: 82 lines

**Git stats**:
```
tt-train/tests/python/quick_check_weights.py | 82 ++++++++++++++++++++++++++++
 1 file changed, 82 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""Quick check that bert-base-uncased weights load correctly."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml
import transformers
from pathlib import Path

model_name = "bert-base-uncased"

# Load HF model
hf_model = transformers.BertModel.from_pretrained(model_name)
hf_config = hf_model.config

# Save to safetensors
safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
if not safetensors_path.exists():
    from safetensors.torch import save_file

    save_file(hf_model.state_dict(), str(safetensors_path))

# Create TTML model
ttml_config = ttml.models.bert.BertConfig()
ttml_config.vocab_size = hf_config.vocab_size
ttml_config.max_sequence_length = 32
ttml_config.embedding_dim = hf_config.hidden_size

```

**Classification**: [TBD - See analysis below]

---

## 45. `tt-train/tests/python/test_bert_base_uncased_diagnostic.py`

**File size**: 322 lines

**Git stats**:
```
.../python/test_bert_base_uncased_diagnostic.py    | 322 +++++++++++++++++++++
 1 file changed, 322 insertions(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layer-by-layer diagnostic test for bert-base-uncased.
Compares TTML BERT against HuggingFace BERT at each intermediate step
to identify exactly where divergences occur.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml  # noqa: E402

# Skip if transformers not available
transformers = pytest.importorskip("transformers", reason="transformers not installed")


def compute_metrics(golden, actual, name=""):
    """Compute comprehensive comparison metrics between two tensors."""
    golden_flat = golden.flatten()
    actual_flat = actual.flatten()

    if len(golden_flat) != len(actual_flat):

```

**Classification**: [TBD - See analysis below]

---

## 46. `tt-train/tests/python/test_bert_comprehensive_validation.py`

**File size**: 568 lines

**Git stats**:
```
.../python/test_bert_comprehensive_validation.py   | 568 +++++++++++++++++++++
 1 file changed, 568 insertions(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive stepwise validation test for multiple BERT model variants.

This test validates TTML BERT implementation against HuggingFace reference
models by comparing intermediate results at every stage of inference.

Test coverage:
1. Weight loading validation (embeddings, QKV, FFN, layer norms)
2. Embedding layer outputs (token, position, token_type, combined, normalized)
3. Each transformer block intermediate outputs (attention, FFN)
4. Final model outputs

Tested model variants:
- prajjwal1/bert-tiny (2 layers, 128 hidden, 2 heads) - Smallest for fast testing
- prajjwal1/bert-small (4 layers, 512 hidden, 8 heads) - Small variant
- google/bert_uncased_L-4_H-512_A-8 (4 layers, 512 hidden, 8 heads) - Official small
- bert-base-uncased (12 layers, 768 hidden, 12 heads) - Standard BERT
- distilbert-base-uncased (6 layers, 768 hidden, 12 heads) - Distilled variant
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

```

**Classification**: [TBD - See analysis below]

---

## 47. `tt-train/tests/python/test_bert_data_roundtrip.py`

**File size**: 131 lines

**Git stats**:
```
tt-train/tests/python/test_bert_data_roundtrip.py | 131 ++++++++++++++++++++++
 1 file changed, 131 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Test if real BERT Q,K,V data survives round-trip through TTML tensors.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x, y):
    """Compute Pearson correlation coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


def main():
    print("\n" + "=" * 80)
    print("TEST: Real BERT Data Round-Trip Through TTML Tensors")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)

```

**Classification**: [TBD - See analysis below]

---

## 48. `tt-train/tests/python/test_bert_embedding_decomposition.py`

**File size**: 268 lines

**Git stats**:
```
.../python/test_bert_embedding_decomposition.py    | 268 +++++++++++++++++++++
 1 file changed, 268 insertions(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Embedding Decomposition Test

Tests each embedding sub-component independently:
1. Word embeddings (token lookup)
2. Position embeddings
3. Token type embeddings
4. Pre-LayerNorm combined embeddings
5. Post-LayerNorm final embeddings

This isolates which specific embedding component might have issues.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")
from transformers import BertModel  # noqa: E402



```

**Classification**: [TBD - See analysis below]

---

## 49. `tt-train/tests/python/test_bert_embeddings_only.py`

**File size**: 160 lines

**Git stats**:
```
tt-train/tests/python/test_bert_embeddings_only.py | 160 +++++++++++++++++++++
 1 file changed, 160 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""Test BERT embeddings layer-by-layer to identify where divergence starts."""

import numpy as np
import os
import sys
from pathlib import Path
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    mean_x = np.mean(x_flat)
    mean_y = np.mean(y_flat)
    numerator = np.sum((x_flat - mean_x) * (y_flat - mean_y))
    denominator = np.sqrt(np.sum((x_flat - mean_x) ** 2) * np.sum((y_flat - mean_y) ** 2))
    return numerator / denominator if denominator > 0 else 0.0


def test_embeddings():
    model_name = "prajjwal1/bert-tiny"
    test_text = "The quick brown fox."

    print(f"Testing BERT embeddings for: {model_name}")

```

**Classification**: [TBD - See analysis below]

---

## 50. `tt-train/tests/python/test_bert_end_to_end_validation.py`

**File size**: 283 lines

**Git stats**:
```
.../python/test_bert_end_to_end_validation.py      | 283 +++++++++++++++++++++
 1 file changed, 283 insertions(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT End-to-End Full Model Validation

Tests complete BERT models end-to-end with various configurations:
- Multiple batch sizes
- Multiple sequence lengths
- All model variants (tiny, small, base)

This validates that the complete model works correctly after the dtype fix.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")
from transformers import BertModel  # noqa: E402


class BERTEndToEndValidator:
    """Validates complete BERT model end-to-end."""

```

**Classification**: [TBD - See analysis below]

---

## 51. `tt-train/tests/python/test_bert_golden_reference.py`

**File size**: 282 lines

**Git stats**:
```
tt-train/tests/python/test_bert_golden_reference.py | 1 +
 1 file changed, 1 insertion(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Golden reference test comparing TTML BERT against HuggingFace BERT.
This test validates that QKV weight loading correctly transforms HuggingFace's
separate Q, K, V weights into TTML's combined QKV format.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml  # noqa: E402

# Skip if transformers not available
transformers = pytest.importorskip("transformers", reason="transformers not installed")


def compute_pcc(golden, actual):
    """Compute Pearson Correlation Coefficient between two tensors."""
    golden_flat = golden.flatten()
    actual_flat = actual.flatten()

    if len(golden_flat) != len(actual_flat):
        return 0.0

```

**Classification**: [TBD - See analysis below]

---

## 52. `tt-train/tests/python/test_bert_inference_showcase.py`

**File size**: 418 lines

**Git stats**:
```
.../tests/python/test_bert_inference_showcase.py   | 418 +++++++++++++++++++++
 1 file changed, 418 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
BERT Inference Showcase Test

Demonstrates valid BERT inference operation on meaningful text data with
comprehensive reports showing:
- Complete input text
- HuggingFace reference model output
- TTML model output
- Side-by-side comparison of both outputs
"""

import numpy as np
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


@dataclass
class InferenceComparison:
    """Comparison results from HuggingFace vs TTML inference."""

    model_name: str

```

**Classification**: [TBD - See analysis below]

---

## 53. `tt-train/tests/python/test_bert_isolated_layer_validation.py`

**File size**: 299 lines

**Git stats**:
```
.../python/test_bert_isolated_layer_validation.py  | 299 +++++++++++++++++++++
 1 file changed, 299 insertions(+)
```

**First 30 lines preview**:
```
"""
BERT Isolated Layer Validation Test

Tests each BERT layer independently using reference inputs from HuggingFace.
This isolates whether layers are individually broken or if errors accumulate.

For each layer, we:
1. Get the reference input from HuggingFace
2. Feed that reference input to the corresponding TTML layer
3. Compare outputs

This shows the intrinsic accuracy of each layer in isolation.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")
from transformers import BertModel, BertConfig  # noqa: E402


class BERTIsolatedLayerValidator:
    """Validates BERT layers independently with reference inputs."""

```

**Classification**: [TBD - See analysis below]

---

## 54. `tt-train/tests/python/test_bert_layer_by_layer_multi_model.py`

**File size**: 314 lines

**Git stats**:
```
.../python/test_bert_layer_by_layer_multi_model.py | 314 +++++++++++++++++++++
 1 file changed, 314 insertions(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive layer-by-layer BERT validation across multiple model variants.

Tests 4 BERT models from tiny to base, comparing HuggingFace vs TTML
at every layer to identify exactly where and how divergence occurs.

Models tested:
- prajjwal1/bert-tiny: 2 layers, 128 hidden, 2 heads
- prajjwal1/bert-small: 4 layers, 512 hidden, 8 heads
- google/bert_uncased_L-4_H-512_A-8: 4 layers, 512 hidden, 8 heads
- bert-base-uncased: 12 layers, 768 hidden, 12 heads

For each model, validates:
1. Embedding layer output
2. Each transformer block's attention output
3. Each transformer block's final output (after FFN)
4. Final model output

Identifies first layer where PCC drops below threshold and tracks
divergence accumulation across layers.
"""

import numpy as np
import pytest
import os
import sys
import torch

```

**Classification**: [TBD - See analysis below]

---

## 55. `tt-train/tests/python/test_bert_layer_pcc_report.py`

**File size**: 275 lines

**Git stats**:
```
.../tests/python/test_bert_layer_pcc_report.py     | 275 +++++++++++++++++++++
 1 file changed, 275 insertions(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Layer PCC Report - Clean reporting format

Generates a clean PCC report for each layer across multiple BERT models.
Shows layer-by-layer PCC in a table format for easy comparison.

Note: This runs through the model cumulatively (each layer receives the
previous layer's output from TTML). For true isolation (feeding reference
inputs to each layer), we would need to expose individual blocks in Python bindings.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")


def compute_pcc(golden: np.ndarray, actual: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient."""

```

**Classification**: [TBD - See analysis below]

---

## 56. `tt-train/tests/python/test_bert_operator_validation.py`

**File size**: 459 lines

**Git stats**:
```
.../tests/python/test_bert_operator_validation.py  | 459 +++++++++++++++++++++
 1 file changed, 459 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
BERT Operator Validation Tests - Bottom-Up Approach

Tests each BERT operation in isolation with controlled random data to identify
which specific operations are broken.

Strategy:
1. Generate identical random inputs for both PyTorch and TTML
2. Run operation through both implementations
3. Compare outputs with strict PCC threshold (>0.999)
4. Isolate and identify broken operations

Test Levels:
- Level 1: Primitive operations (matmul, softmax, layer_norm, GELU, etc.)
- Level 2: Composite operations (QKV projection, attention mechanism)
- Level 3: Layer-by-layer (embeddings, attention blocks, MLP blocks)
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml



```

**Classification**: [TBD - See analysis below]

---

## 57. `tt-train/tests/python/test_bert_operators_comprehensive.py`

**File size**: 415 lines

**Git stats**:
```
.../python/test_bert_operators_comprehensive.py    | 415 +++++++++++++++++++++
 1 file changed, 415 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Comprehensive BERT Operator Validation Tests

Tests each BERT operation in isolation with controlled random data.
Compares PyTorch reference implementation vs TTML implementation.

This is a HEAVY test suite implementing Option 1 from OPERATOR_TESTING_LIMITATIONS.md
"""

import numpy as np
import os
import sys
import torch
import pytest

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    if len(x_flat) != len(y_flat):
        return 0.0
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0



```

**Classification**: [TBD - See analysis below]

---

## 58. `tt-train/tests/python/test_bert_operators_with_real_data.py`

**File size**: 321 lines

**Git stats**:
```
.../python/test_bert_operators_with_real_data.py   | 321 +++++++++++++++++++++
 1 file changed, 321 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
BERT Operator Validation with REAL BERT Data and REAL Attention Masks

This test addresses the CRITICAL FLAW: Previous tests used random data, not real BERT outputs.

This test:
1. Loads a real pre-trained BERT model
2. Runs forward pass to get REAL intermediate values (Q, K, V from actual embeddings)
3. Extracts REAL attention masks from actual inputs
4. Tests TTML operators with these REAL values
5. Compares against HuggingFace reference with PROPER MASKING

Test Scenarios:
- Minimal padding (90% real tokens, 10% padding)
- Medium padding (50% real tokens, 50% padding)
- Heavy padding (20% real tokens, 80% padding)
- No padding (100% real tokens)
"""

import numpy as np
import os
import sys
import torch
import pytest
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


```

**Classification**: [TBD - See analysis below]

---

## 59. `tt-train/tests/python/test_bert_padding_mask_validation.py`

**File size**: 284 lines

**Git stats**:
```
.../python/test_bert_padding_mask_validation.py    | 284 +++++++++++++++++++++
 1 file changed, 284 insertions(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Padding Mask Validation

Tests BERT behavior with variable-length sequences and attention masks:
- Variable-length sequences with padding
- Different attention mask patterns
- Verify masked tokens don't affect output
- Compare HuggingFace and TTML masking behavior

This ensures that padding and attention masking work correctly.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")
from transformers import BertModel  # noqa: E402


class BERTPaddingMaskValidator:

```

**Classification**: [TBD - See analysis below]

---

## 60. `tt-train/tests/python/test_bert_real_qkv_attention.py`

**File size**: 257 lines

**Git stats**:
```
.../tests/python/test_bert_real_qkv_attention.py   | 257 +++++++++++++++++++++
 1 file changed, 257 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Test TTML attention with real BERT Q, K, V data.

This test extracts real learned Q, K, V values from prajjwal1/bert-tiny
and tests them directly with TTML's scaled_dot_product_attention to isolate
whether the issue is in the attention computation itself.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x, y):
    """Compute Pearson correlation coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


def test_real_bert_attention_no_mask():
    """Test with real BERT data WITHOUT masking (all real tokens)."""
    print("\n" + "=" * 80)

```

**Classification**: [TBD - See analysis below]

---

## 61. `tt-train/tests/python/test_bert_stepwise_validation.py`

**File size**: 381 lines

**Git stats**:
```
.../tests/python/test_bert_stepwise_validation.py  | 381 +++++++++++++++++++++
 1 file changed, 381 insertions(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Stepwise validation test for BERT models.

This test executes HuggingFace BERT and TTML BERT step-by-step,
comparing intermediate outputs at each stage to identify exactly
where any divergence occurs.

Validation stages:
1. Token embeddings
2. Position embeddings
3. Token type embeddings
4. Embedding layer norm + dropout
5. Each transformer block (attention + FFN)
6. Final output

Strategy:
- Use HuggingFace model as golden reference
- Execute TTML model through Python bindings
- Compare tensors at each step with comprehensive metrics
- Report first divergence point with detailed analysis
"""

import numpy as np
import pytest
import os
import sys
import torch

```

**Classification**: [TBD - See analysis below]

---

## 62. `tt-train/tests/python/test_bert_stepwise_validation_manual.py`

**File size**: 256 lines

**Git stats**:
```
.../python/test_bert_stepwise_validation_manual.py | 256 +++++++++++++++++++++
 1 file changed, 256 insertions(+)
```

**First 30 lines preview**:
```
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Manual stepwise validation test for BERT models.

This test manually steps through BERT computation using TTML parameters
and operations, comparing each step against HuggingFace BERT.

This approach works with existing TTML Python bindings without requiring
C++ modifications.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Tuple

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")


def compute_pcc(golden: np.ndarray, actual: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient."""
    golden_flat = golden.flatten()

```

**Classification**: [TBD - See analysis below]

---

## 63. `tt-train/tests/python/test_binding_data_flow.py`

**File size**: 166 lines

**Git stats**:
```
tt-train/tests/python/test_binding_data_flow.py | 166 ++++++++++++++++++++++++
 1 file changed, 166 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Test data flow through Python bindings.

This test verifies that data is correctly converted from numpy -> TTML -> computation -> numpy.
"""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def test_simple_attention_roundtrip():
    """Test that simple attention computation matches expected result."""
    print("=" * 80)
    print("TEST: Simple Attention Data Flow")
    print("=" * 80)

    # Create simple test data - same as C++ test
    batch_size = 1
    num_heads = 2
    seq_len = 8
    head_dim = 4

    # Use specific values for debugging
    np.random.seed(42)
    q_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

```

**Classification**: [TBD - See analysis below]

---

## 64. `tt-train/tests/python/test_full_precision_attention.py`

**File size**: 119 lines

**Git stats**:
```
.../tests/python/test_full_precision_attention.py  | 119 +++++++++++++++++++++
 1 file changed, 119 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Test TTML attention with FULL precision (not HALF precision).

The issue discovered: TTML uses HALF precision (bfloat16/float16) by default,
which causes precision loss with real BERT weights.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x, y):
    """Compute Pearson correlation coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


def test_precision_settings():
    """Test if we can control precision settings."""
    print("\n" + "=" * 80)
    print("TESTING PRECISION SETTINGS")

```

**Classification**: [TBD - See analysis below]

---

## 65. `tt-train/tests/python/test_heads_creation_debug.py`

**File size**: 109 lines

**Git stats**:
```
tt-train/tests/python/test_heads_creation_debug.py | 109 +++++++++++++++++++++
 1 file changed, 109 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""Debug the heads creation reshape to see if it's scrambling data."""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

# Create a simple test case with known pattern
# For 128 embedding dim, 2 heads, head_dim=64:
# We want:
#   Head 0 to get dims 0-63
#   Head 1 to get dims 64-127

batch_size = 1
seq_len = 4  # Small for easy verification
embedding_dim = 128
num_heads = 2
head_dim = embedding_dim // num_heads

print(f"Testing heads creation with:")
print(f"  Sequence length: {seq_len}")
print(f"  Embedding dim: {embedding_dim}")
print(f"  Num heads: {num_heads}")
print(f"  Head dim: {head_dim}")

# Create QKV tensor with a known pattern
# Each token has a unique pattern so we can track where it goes

```

**Classification**: [TBD - See analysis below]

---

## 66. `tt-train/tests/python/test_layernorm_epsilon_verification.py`

**File size**: 109 lines

**Git stats**:
```
.../python/test_layernorm_epsilon_verification.py  | 109 +++++++++++++++++++++
 1 file changed, 109 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""Verify LayerNorm epsilon configuration in BERT model."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml
import transformers
from pathlib import Path

model_name = "prajjwal1/bert-tiny"

# Load HF model
hf_model = transformers.BertModel.from_pretrained(model_name)
hf_config = hf_model.config

print(f"HuggingFace BERT epsilon: {hf_config.layer_norm_eps}")

# Create TTML model
ttml_config = ttml.models.bert.BertConfig()
ttml_config.vocab_size = hf_config.vocab_size
ttml_config.max_sequence_length = 32
ttml_config.embedding_dim = hf_config.hidden_size
ttml_config.intermediate_size = hf_config.intermediate_size
ttml_config.num_heads = hf_config.num_attention_heads
ttml_config.num_blocks = hf_config.num_hidden_layers
ttml_config.dropout_prob = 0.0
ttml_config.layer_norm_eps = hf_config.layer_norm_eps

```

**Classification**: [TBD - See analysis below]

---

## 67. `tt-train/tests/python/test_mha_isolated.py`

**File size**: 177 lines

**Git stats**:
```
tt-train/tests/python/test_mha_isolated.py | 177 +++++++++++++++++++++++++++++
 1 file changed, 177 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Test Multi-Head Attention in isolation with controlled random data.
This will help identify if MHA is the problem or something else.
"""

import numpy as np
import os
import sys
import torch

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x, y):
    """Compute Pearson Correlation Coefficient."""
    x_flat, y_flat = x.flatten(), y.flatten()
    if len(x_flat) != len(y_flat):
        print(f"Shape mismatch: {len(x_flat)} vs {len(y_flat)}")
        return 0.0
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


print("=" * 80)
print("ISOLATED MULTI-HEAD ATTENTION TEST")
print("Testing MHA with controlled random weights and data")
print("=" * 80)


```

**Classification**: [TBD - See analysis below]

---

## 68. `tt-train/tests/python/test_numpy_contiguity.py`

**File size**: 127 lines

**Git stats**:
```
tt-train/tests/python/test_numpy_contiguity.py | 127 +++++++++++++++++++++++++
 1 file changed, 127 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Test if BERT Q,K,V numpy arrays are contiguous.

Non-contiguous arrays could cause data corruption during conversion to C++.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def main():
    print("\n" + "=" * 80)
    print("TEST: NumPy Array Contiguity")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    text = "Hello world"
    max_length = 8


```

**Classification**: [TBD - See analysis below]

---

## 69. `tt-train/tests/python/test_parameters_update.py`

**File size**: 110 lines

**Git stats**:
```
tt-train/tests/python/test_parameters_update.py | 110 ++++++++++++++++++++++++
 1 file changed, 110 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""Test if parameters() returns references to the same tensor objects."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml
import transformers
from pathlib import Path

# Create a simple BERT model
print("Creating BERT model...")
config = ttml.models.bert.BertConfig()
config.vocab_size = 100
config.max_sequence_length = 32
config.embedding_dim = 64
config.intermediate_size = 256
config.num_heads = 2
config.num_blocks = 1
config.dropout_prob = 0.0

bert = ttml.models.bert.create(config)

# Get parameters twice
print("\nGetting parameters twice...")
params1 = bert.parameters()
params2 = bert.parameters()


```

**Classification**: [TBD - See analysis below]

---

## 70. `tt-train/tests/python/test_row_major_layout.py`

**File size**: 116 lines

**Git stats**:
```
tt-train/tests/python/test_row_major_layout.py | 116 +++++++++++++++++++++++++
 1 file changed, 116 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Test if using ROW_MAJOR layout instead of TILE layout fixes the real BERT data issue.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x, y):
    """Compute Pearson correlation coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


def main():
    print("\n" + "=" * 80)
    print("TEST: Real BERT Data with ROW_MAJOR vs TILE Layout")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)

```

**Classification**: [TBD - See analysis below]

---

## 71. `tt-train/tests/python/test_set_value_basic.py`

**File size**: 36 lines

**Git stats**:
```
tt-train/tests/python/test_set_value_basic.py | 36 +++++++++++++++++++++++++++
 1 file changed, 36 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""Test if set_value() actually updates tensor values."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

# Create a simple tensor
print("Creating tensor with ones...")
data = np.ones((4, 4), dtype=np.float32)
tensor = ttml.autograd.Tensor.from_numpy(data.reshape(1, 1, 4, 4))

print("Initial values:")
retrieved = tensor.to_numpy().reshape(4, 4)
print(f"  Mean: {retrieved.mean():.6f}")
print(f"  [0,0]: {retrieved[0,0]}")

# Try to set new value
print("\nCalling set_value with zeros...")
new_data = np.zeros((4, 4), dtype=np.float32)
tensor.set_value(
    ttml.core.from_vector(new_data.flatten().tolist(), tensor.get_value().logical_shape(), tensor.get_value().device())
)

print("After set_value:")
retrieved_after = tensor.to_numpy().reshape(4, 4)
print(f"  Mean: {retrieved_after.mean():.6f}")

```

**Classification**: [TBD - See analysis below]

---

## 72. `tt-train/tests/python/test_simple_operator_validation.py`

**File size**: 205 lines

**Git stats**:
```
.../python/test_simple_operator_validation.py      | 205 +++++++++++++++++++++
 1 file changed, 205 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Simple Operator Validation - Bottom-Up Testing
Test each BERT operation in isolation to find which one is broken.
"""

import numpy as np
import os
import sys
import torch

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x, y):
    """Compute Pearson Correlation Coefficient."""
    x_flat, y_flat = x.flatten(), y.flatten()
    if len(x_flat) != len(y_flat):
        return 0.0
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


def print_result(name, ref, ttml_output, threshold=0.999):
    """Print comparison result."""
    ref_np = ref.detach().numpy() if torch.is_tensor(ref) else ref
    ttml_np = ttml_output.to_numpy() if hasattr(ttml_output, "to_numpy") else ttml_output

    pcc = compute_pcc(ref_np, ttml_np)

```

**Classification**: [TBD - See analysis below]

---

## 73. `tt-train/tests/python/test_softmax_extreme_values.py`

**File size**: 150 lines

**Git stats**:
```
.../tests/python/test_softmax_extreme_values.py    | 150 +++++++++++++++++++++
 1 file changed, 150 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Test if TTML softmax handles extreme values correctly (like -1e9 from masking).
"""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml
import torch


def test_softmax_with_masking():
    """Test softmax with extreme negative values from masking."""
    print("=" * 80)
    print("TEST: Softmax with Extreme Values (Masking)")
    print("=" * 80)

    # Create scores with masking pattern
    # Realistic BERT attention scores: small values for real positions, -1e9 for masked
    scores = np.array(
        [
            [0.5, -0.3, 0.2, -1e9, -1e9, -1e9],  # 3 real, 3 masked
            [0.1, 0.8, -0.2, 0.4, -1e9, -1e9],  # 4 real, 2 masked
        ],
        dtype=np.float32,
    ).reshape(1, 1, 2, 6)


```

**Classification**: [TBD - See analysis below]

---

## 74. `tt-train/tests/python/verify_masking_fix.py`

**File size**: 85 lines

**Git stats**:
```
tt-train/tests/python/verify_masking_fix.py | 85 +++++++++++++++++++++++++++++
 1 file changed, 85 insertions(+)
```

**First 30 lines preview**:
```
#!/usr/bin/env python3
"""
Simple verification that the masking fix is working correctly.
"""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

# Create simple test data
batch_size = 1
num_heads = 2
seq_len = 8
head_dim = 4

# Simple Q, K, V - all ones for simplicity
q_data = np.ones((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
k_data = np.ones((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
v_data = np.arange(batch_size * num_heads * seq_len * head_dim, dtype=np.float32).reshape(
    batch_size, num_heads, seq_len, head_dim
)

# Mask: first 4 positions = attend (1), last 4 positions = masked (0)
mask_data = np.array([[[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32)

print(f"Input shapes:")
print(f"  Q: {q_data.shape}")

```

**Classification**: [TBD - See analysis below]

---
