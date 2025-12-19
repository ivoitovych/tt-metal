# BERT Task Heads V2 Branch - Comprehensive Commit Analysis

**Branch:** `myfork/ivoitovych/bert-model-for-ttml-task-heads-v2`
**Base:** `myfork/ivoitovych/bert-model-for-ttml-backup-2025-12-18`
**Total Commits:** 49
**Analysis Date:** 2025-12-19

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Commit Categories Overview](#commit-categories-overview)
3. [Detailed Commit Descriptions](#detailed-commit-descriptions)
   - [Phase 1: Core Implementation](#phase-1-core-implementation-commits-1-6)
   - [Phase 2: Python Integration & Fixes](#phase-2-python-integration--fixes-commits-7-12)
   - [Phase 3: Bug Investigation - Batch Processing](#phase-3-bug-investigation---batch-processing-commits-13-23)
   - [Phase 4: Embedding Batch Bug Fix](#phase-4-embedding-batch-bug-fix-commits-24-28)
   - [Phase 5: Bug Investigation - Attention Mechanism](#phase-5-bug-investigation---attention-mechanism-commits-29-36)
   - [Phase 6: Softmax Precision Workaround](#phase-6-softmax-precision-workaround-commits-37-44)
   - [Phase 7: Final Cleanup & Bug Reports](#phase-7-final-cleanup--bug-reports-commits-45-49)
4. [File Change Summary](#file-change-summary)
5. [Recommendations](#recommendations)

---

## Executive Summary

The Task Heads V2 branch contains 49 commits spanning approximately 2+ weeks of development. The work can be characterized as:

| Category | Commits | Lines Changed | Production Code |
|----------|---------|---------------|-----------------|
| Core Implementation | 6 | ~5,000+ | Yes |
| Python Integration & Fixes | 6 | ~1,500+ | Yes |
| Bug Investigation (docs) | ~20 | ~3,000+ | No (docs only) |
| Embedding Batch Workaround | 5 | ~800+ | Yes |
| Softmax FP32 Workaround | 8 | ~600+ | Yes |
| Test Infrastructure | 4 | ~400+ | Tests only |

**Key Deliverables:**
- 5 BERT task models (SequenceClassification, TokenClassification, QuestionAnswering, MaskedLM, PreTraining)
- 5 head modules with HuggingFace-exact architectures
- 7 loss helper functions
- Python bindings and factory functions
- 2 critical bug workarounds (embedding batch, softmax precision)
- Comprehensive test suite (C++ and Python)

**Technical Debt:**
- ~25 documentation/investigation commits that should be cleaned up
- Multiple debug/investigation test files to remove
- Bug reports in markdown that should be filed as GitHub issues

---

## Commit Categories Overview

```
Core Implementation (6 commits):
├── 2ee6c1839d - feat: BERT task heads architecture v2
├── 457d6175bc - feat: Critical integration components
├── 35d709cc85 - feat: Comprehensive skeleton implementations
├── e0daa0febb - feat: Complete validation tests and documentation
├── 97cb2258b0 - docs: Final implementation status report
└── a8284d8e28 - fix: Compilation errors and test isolation

Python Integration (6 commits):
├── 65a3f7e94c - fix: Python bindings for BERT task models
├── 8233fd06a2 - fix: type_vocab_size binding
├── f55c609017 - fix: File/directory path handling
├── 8827592407 - fix: Golden reference tests
├── 311e94094f - docs: Implementation notes
└── 11a71a01f3 - docs: Comprehensive fix report

Embedding Batch Workaround (5 commits):
├── cd1c004988 - fix: Workaround for TTNN embedding batch bug
├── edd61851c3 - test: Minimal C++ test for embedding bug
├── fcbedf610f - test: C++ regression tests
├── 84723951a8 - fix: I64 dtype error in safetensors
└── 6eee1d3d12 - feat: Root cause identification

Softmax Precision Workaround (8 commits):
├── 9a261d86cc - workaround: FP32 accumulation option
├── 668c173f81 - fix: Enable FP32 accumulation
├── 77e57b4335 - test: C++ softmax precision tests
├── d3b81efa58 - fix: Change default to false
├── cf46393fc4 - fix: Device cleanup in test
├── a8b8527d06 - docs: Emphasize NOT a fix
├── 28b4300bea - docs+code: Emphasize workarounds
└── f29e470dea - docs: Test breaks subsequent tests

Investigation Documentation (~20 commits):
├── Multiple investigation reports
├── Status updates
├── Root cause analysis documents
└── Layer-by-layer debugging docs

Bug Reports (4 commits):
├── 7b09b22293 - docs: TTNN bug reports
├── 40051a3d8f - docs: Fix softmax bug report
├── 57ad7e4e66 - docs: Bug reproduction documents
└── 91fe6e0b80 - test: Fix test infrastructure
```

---

## Detailed Commit Descriptions

### Phase 1: Core Implementation (Commits 1-6)

#### Commit 1: `2ee6c1839d`
**Subject:** feat: Implement BERT task heads architecture v2 per design document

**Type:** Feature (Core Implementation)
**Production Code:** Yes
**Complexity:** High

**Description:**
Implements the complete BERT task heads architecture following a comprehensive design document. This is the foundational commit establishing all core components.

**Components Implemented:**
- **Layer 0 (Base BERT):** Added `BertOutput` struct and `forward_structured()` method
- **Layer 1 (Head Modules):** 5 heads in `bert_heads.hpp/cpp`:
  - BertSequenceClassificationHead
  - BertTokenClassificationHead
  - BertQuestionAnsweringHead
  - BertMaskedLMHead
  - BertNSPHead
- **Layer 2 (Task Models):** 5 models in `bert_tasks.hpp/cpp`:
  - BertForSequenceClassification
  - BertForTokenClassification
  - BertForQuestionAnswering
  - BertForMaskedLM
  - BertForPreTraining
- **Layer 3 (Loss Helpers):** 7 functions in `bert_losses.hpp/cpp`

**Files Created:**
- `tt-train/sources/ttml/models/bert_tasks.cpp`, `bert_tasks.hpp`
- `tt-train/sources/ttml/modules/bert_heads.cpp`, `bert_heads.hpp`
- `tt-train/sources/ttml/ops/bert_losses.cpp`, `bert_losses.hpp`

**Design Principles:**
- Pure encoder base (zero BERT changes)
- No forced abstractions
- External loss only
- HuggingFace-exact layers
- Composition pattern (GPT-2/Llama style)

---

#### Commit 2: `457d6175bc`
**Subject:** feat: Implement critical integration components for BERT task heads

**Type:** Feature (Integration)
**Production Code:** Yes
**Complexity:** High

**Description:**
Makes the core implementation usable by adding build system integration, SafeTensors serialization, Python bindings, configuration templates, and training examples.

**Components Implemented:**
- **Layer 4 (Build System):** CMakeLists.txt updates
- **Layer 5 (SafeTensors):** `bert_tasks_serialization.hpp` with `load_task_head_weights()`
- **Layer 6 (Python Bindings):** All 5 task models + 7 loss functions in `nb_models.cpp`, `nb_ops.cpp`
- **Layer 6b (Python Factory):** `bert_task_factory.py`
- **Layer 4b (Configs):** `bert_sequence_classification.yaml`, `bert_token_classification.yaml`
- **Layer 8 (Examples):** `train_bert_classifier.py`
- **Layer 11 (Basic Tests):** `test_bert_task_heads_basic.py`

**Files Created:**
- `tt-train/sources/ttml/models/bert_tasks_serialization.hpp`
- `tt-train/sources/ttml/ttml/common/bert_task_factory.py`
- `tt-train/configs/bert_sequence_classification.yaml`
- `tt-train/configs/bert_token_classification.yaml`
- `tt-train/examples/train_bert_classifier.py`
- `tt-train/tests/python/test_bert_task_heads_basic.py`
- `tt-train/TASK_HEADS_V2_DESIGN_DOCUMENT.md`

---

#### Commit 3: `35d709cc85`
**Subject:** feat: Add comprehensive skeleton implementations for BERT task heads infrastructure

**Type:** Feature (Infrastructure)
**Production Code:** Yes (tests + examples)
**Complexity:** High

**Description:**
Completes the infrastructure with C++ unit tests, Python HuggingFace validation tests, YAML configs, C++ training example, and MsgPack training state serialization.

**Components Implemented:**
- **C++ Unit Tests:** 800+ lines in `bert_task_heads_test.cpp`
- **Python HF Validation:** 500+ lines in `test_bert_task_heads_hf_validation.py`
- **YAML Configs:** `bert_question_answering.yaml`, `bert_masked_lm.yaml`, `bert_pretraining.yaml`
- **C++ Example:** 400+ lines in `train_bert_classifier.cpp`
- **Serialization:** `bert_training_state.hpp/cpp` (300+ lines)

**Files Created:**
- `tt-train/tests/model/bert_task_heads_test.cpp`
- `tt-train/tests/python/test_bert_task_heads_hf_validation.py`
- `tt-train/configs/bert_*.yaml` (3 files)
- `tt-train/examples/train_bert_classifier.cpp`
- `tt-train/sources/ttml/serialization/bert_training_state.cpp`, `.hpp`
- `tt-train/TASK_HEADS_IMPLEMENTATION_STATUS.md`
- `tt-train/TASK_HEADS_SKELETON_IMPLEMENTATIONS.md`

---

#### Commit 4: `e0daa0febb`
**Subject:** feat: Complete BERT Task Heads validation tests and documentation

**Type:** Feature (Tests + Docs)
**Production Code:** Tests
**Complexity:** Medium

**Description:**
Addresses critical gaps: completes HuggingFace PCC validation tests, weight loading integration tests, and adds comprehensive documentation to header files.

**Key Changes:**
- Completed all 5 HF validation test classes
- Added weight loading integration test
- Added Doxygen documentation to `bert_heads.hpp` and `bert_tasks.hpp`
- Created `TASK_HEADS_IMPLEMENTATION_STATUS_V3.md`

---

#### Commit 5: `97cb2258b0`
**Subject:** docs: Add comprehensive final implementation status report

**Type:** Documentation
**Production Code:** No
**Complexity:** Low

**Description:**
Complete analysis of the BERT Task Heads implementation across all commits. Status report showing 98% completion.

**Files Created:**
- `tt-train/TASK_HEADS_FINAL_STATUS.md`

---

#### Commit 6: `a8284d8e28`
**Subject:** fix: Fix compilation errors and test isolation issues in BERT task heads implementation

**Type:** Bug Fix (Critical)
**Production Code:** Yes
**Complexity:** High

**Description:**
Resolves compilation errors and test isolation issues preventing build and test success.

**Compilation Fixes:**
- Fixed ambiguous `MsgPackFile::put()` calls
- Fixed namespace resolution (`common::transformer` → `models::common::transformer`)
- Added stride parameter to `ttnn::slice()` calls
- Removed incorrect `override` keywords

**Test Isolation Fixes:**
- Fixed 6 test fixtures with proper `SetUp()`/`TearDown()`
- Converted `TEST()` to `TEST_F()` where needed

**Result:** 140/140 C++ tests passing

---

### Phase 2: Python Integration & Fixes (Commits 7-12)

#### Commit 7: `11a71a01f3`
**Subject:** docs: Add comprehensive fix report for BERT task heads implementation

**Type:** Documentation
**Files Created:** `tt-train/BERT_TASK_HEADS_FIX_REPORT.md`

---

#### Commit 8: `65a3f7e94c`
**Subject:** fix: Add Python bindings for BERT task models and fix Python test issues

**Type:** Bug Fix
**Production Code:** Yes

**Key Changes:**
- Added `__call__` bindings for all 5 BERT task models
- Fixed tensor creation (4D shapes, correct dtypes)
- Fixed imports and API calls in Python tests
- Result: 17/20 Python tests passing

---

#### Commit 9: `8233fd06a2`
**Subject:** fix: Add type_vocab_size binding and fix weight tying issues in HF validation tests

**Type:** Bug Fix
**Production Code:** Yes

**Key Changes:**
- Added `type_vocab_size` Python binding
- Fixed `save_hf_model()` to use `save_pretrained()`
- Disabled weight tying in MaskedLM/PreTraining tests
- Result: 33 passed, 8 skipped, 3 failed (improved from 10 failed)

---

#### Commit 10: `f55c609017`
**Subject:** fix: Handle both file and directory paths in load_from_safetensors

**Type:** Bug Fix
**Production Code:** Yes

**Description:**
Fixed `load_from_safetensors` to accept both single `.safetensors` files and directories containing multiple files.

**Result:** +2 tests passing (35 passed, 6 skipped, 3 failed)

---

#### Commit 11: `8827592407`
**Subject:** fix: Fix golden reference tests with correct uint32 embeddings and adjusted thresholds

**Type:** Bug Fix
**Production Code:** Tests

**Key Changes:**
- Fixed critical bug: Convert input_ids to uint32 instead of float32
- Adjusted PCC threshold from >0.99 to >0.95
- Adjusted mean error threshold from <1e-2 to <2e-1

---

#### Commit 12: `311e94094f`
**Subject:** docs: Add implementation notes and address review findings

**Type:** Documentation
**Production Code:** No

**Files Created:**
- `tt-train/BERT_TASK_HEADS_IMPLEMENTATION_NOTES.md`
- `tt-train/BERT_TASK_HEADS_IMPLEMENTATION_REVIEW.md`

---

### Phase 3: Bug Investigation - Batch Processing (Commits 13-23)

These commits document the investigation into batch processing issues. Most are documentation-only.

#### Commit 13: `1d3d6faa8d`
**Subject:** feat: Add layer-by-layer BERT validation for debugging

**Type:** Feature (Debug Tools)
**Files Created:**
- `tt-train/tests/python/test_bert_base_uncased_debug.py`
- `tt-train/tests/python/test_bert_layer_by_layer_validation.py`

**Key Finding:** Embeddings PCC=0.98, error accumulates through 12 layers (0.92→0.83)

---

#### Commit 14: `270f3846fb`
**Subject:** docs: Add comprehensive batch size bug investigation report

**Type:** Documentation
**Files Created:** `tt-train/BERT_BATCH_SIZE_BUG_INVESTIGATION.md`

**Key Findings:**
- Bug #1: Embeddings batch processing (PCC degrades with batch>1)
- Bug #2: Attention shape mismatch (crashes with batch>1)

---

#### Commit 15: `efa954c53e`
**Subject:** Add batch processing regression tests for BERT operations

**Type:** Tests
**Files Created:**
- `tt-train/tests/ops/embedding_batch_regression_test.cpp`
- `tt-train/tests/ops/multi_head_attention_batch_regression_test.cpp`
- `tt-train/tests/ops/README_BATCH_REGRESSION_TESTS.md`
- `tt-train/tests/python/test_bert_batch_processing.py`

---

#### Commit 16: `c1e0d463d2`
**Subject:** docs: Add comprehensive BERT batch processing investigation report

**Type:** Documentation
**Files Created:** `tt-train/BERT_BATCH_PROCESSING_INVESTIGATION_REPORT.md`

---

#### Commit 17: `62b6b5733b`
**Subject:** docs: Consolidate and correct BERT documentation - NOT production ready

**Type:** Documentation (Cleanup)

**Key Changes:**
- Removed 7 redundant/superseded reports (~4,300 lines)
- Created `BERT_TASK_HEADS_SUMMARY.md` with honest assessment
- **Critical correction:** Implementation NOT production ready (PCC<0.999)

---

#### Commit 18: `6eee1d3d12`
**Subject:** feat: Identify root cause of BERT accuracy degradation with granular embedding decomposition

**Type:** Feature (Debug Infrastructure)
**Production Code:** Yes

**Key Changes:**
- Added `EmbeddingIntermediates` struct to `bert.hpp/cpp`
- Added `get_embeddings_with_intermediates()` method
- Created `test_granular_embedding_debug.py`

**Root Cause Identified:**
- Word embeddings: PCC=0.975456 (FIRST ERROR)
- Token type embeddings: PCC=0.999999 (proves op is correct)
- Bug is specific to word embedding table with batch>1

---

#### Commits 19-23: Investigation Documentation

| Commit | Subject | Type |
|--------|---------|------|
| `fcbedf610f` | test: Add C++ regression tests for embedding batch processing | Tests |
| `d5bdcadb7c` | docs: Update investigation report | Docs |
| `6f59ac4f2a` | docs: Confirm root cause is in weight loading | Docs |
| `34478fc6d6` | docs: BREAKTHROUGH - Identify weight loading corruption | Docs |
| `84723951a8` | fix: Fix I64 dtype error in safetensors loading | **Fix** |

---

### Phase 4: Embedding Batch Bug Fix (Commits 24-28)

#### Commit 24: `cd1c004988` **[CRITICAL - WORKAROUND]**
**Subject:** fix: Workaround for TTNN embedding batch processing bug (PCC 0.608 -> 0.999999)

**Type:** Bug Workaround
**Production Code:** Yes
**Complexity:** Medium

**Root Cause:**
`ttnn::embedding` kernel has batch processing bug where batch indices > 0 retrieve incorrect embeddings.

**Symptoms:**
- Batch 0: PCC 0.999999 (perfect)
- Batch 1: PCC 0.608615 (completely wrong)

**Workaround:**
Process each batch separately in `embedding_op.cpp` and concatenate results.

**Files Modified:**
- `tt-train/sources/ttml/ops/embedding_op.cpp`

**Files Created:**
- `tt-train/EMBEDDING_BATCH_BUG_FIX.md`
- `tt-train/EMBEDDING_BATCH_BUG_ROOT_CAUSE.md`
- `tt-train/tests/python/test_embedding_execution_trace.py`

**Performance Impact:** ~2-4x overhead for batch_size > 1

---

#### Commit 25: `edd61851c3`
**Subject:** test: Add minimal C++ test for TTNN embedding batch processing bug

**Type:** Tests
**Files Created:** `tt-train/tests/core/ttnn_embedding_batch_bug_test.cpp`

**Purpose:** Minimal reproduction test for TTNN team to investigate.

---

#### Commits 26-28: Documentation Updates

| Commit | Subject |
|--------|---------|
| `fe035e2392` | docs: Update embedding batch bug fix documentation |
| `c122e33030` | docs: Mark error accumulation as requiring investigation |
| `689bc04003` | docs: Complete layer-by-layer investigation |

---

### Phase 5: Bug Investigation - Attention Mechanism (Commits 29-36)

#### Commit 29: `d900e75a7c`
**Subject:** docs: Update batch processing report with attention mechanism bug findings

**Type:** Documentation
**Key Finding:** Error accumulation caused by attention mechanism bug, not weight loading.

---

#### Commit 30: `6990fd2d3f`
**Subject:** docs: Add comprehensive attention mechanism bug investigation

**Type:** Documentation + Tests
**Files Created:**
- `tt-train/ATTENTION_MECHANISM_INVESTIGATION.md`
- `tt-train/tests/python/test_attention_scaling_order.py`

**Key Finding:** Scaling order hypothesis REJECTED (both achieve PCC >0.9999)

---

#### Commit 31: `54f41110f7`
**Subject:** docs: Consolidate branch documentation (10 files → 2 files)

**Type:** Documentation (Cleanup)
**Result:** 10 files deleted, 2 comprehensive files created

---

#### Commit 32: `8abe1f4692`
**Subject:** docs: Verify core attention operations are correct - bug is in linear layers

**Type:** Documentation + Tests
**Files Created:**
- `tt-train/tests/python/test_attention_operations_debug.py`
- `tt-train/tests/python/test_heads_operations.py`
- `tt-train/tests/python/test_attention_step_by_step_debug.py`

**Key Finding:** All attention operations achieve PCC >0.99999. Bug is in LINEAR LAYERS.

---

#### Commit 33: `02c663f07c`
**Subject:** docs: Update implementation status with linear layer hypothesis

**Type:** Documentation

---

#### Commit 34: `d66abcca02`
**Subject:** test: Verify linear layer operations are correct with random weights

**Type:** Tests
**Files Created:** `tt-train/tests/python/test_linear_layer_debug.py`

**Key Finding:** ALL linear layer operations achieve PCC >0.99999 with random weights.
Bug only appears with pre-trained weights.

---

#### Commit 35: `92cb1a23b2`
**Subject:** feat: Root cause found - SDPA fails with real BERT data (PCC 0.81)

**Type:** Feature (Debug Tools)
**Files Created:**
- `tt-train/BUG_ROOT_CAUSE_FOUND.md`
- `tt-train/tests/python/test_attention_step_by_step.py`
- `tt-train/tests/python/test_layernorm_debug.py`
- `tt-train/tests/python/test_linear_layer_loaded_weights.py`
- `tt-train/tests/python/test_multihead_attention_loaded_weights.py`
- `tt-train/tests/python/test_residual_connection_debug.py`

**Critical Breakthrough:**
- SDPA fails with real BERT data (PCC 0.81)
- Works perfectly with random test data (PCC >0.99999)
- Bug is DATA-DEPENDENT

---

#### Commit 36: `def06e1590`
**Subject:** docs: Update investigation with root cause confirmation

**Type:** Documentation + Tests
**Files Created:**
- `tt-train/tests/python/test_sdpa_direct_comparison.py`
- `tt-train/tests/python/test_sdpa_substeps.py`

---

### Phase 6: Softmax Precision Workaround (Commits 37-44)

#### Commit 37: `668c173f81` **[CRITICAL - FIX]**
**Subject:** fix: Enable FP32 accumulation in softmax for BERT precision

**Type:** Bug Fix
**Production Code:** Yes

**The Problem:**
Softmax using bfloat16 accumulation causes catastrophic precision loss with BERT attention patterns.

**The Fix:**
```cpp
config.fp32_dest_acc_en = true;  // Was: false
```

**Results:**
- bert-tiny: Block 0 PCC 0.94 → 0.999979
- bert-base: ALL 12 blocks PCC >0.999

**Files Modified:** `tt-train/sources/ttml/core/compute_kernel_config.cpp`

---

#### Commit 38: `9a261d86cc` **[CRITICAL - WORKAROUND]**
**Subject:** workaround: Add FP32 accumulation option for softmax bfloat16 precision bug

**Type:** Workaround
**Production Code:** Yes

**Key Changes:**
- Added `use_fp32_accumulation_workaround` parameter
- Updated `ComputeKernelConfig::softmax()`
- Added standalone C++ test

**Files Modified:**
- `tt-train/sources/ttml/core/compute_kernel_config.cpp`, `.hpp`
- `tt-train/sources/ttml/ttnn_fixed/trivial_ttnn_ops.cpp`, `.hpp`

**Files Created:**
- `tt-train/tests/core/softmax_precision_test.cpp`

---

#### Commit 39: `77e57b4335`
**Subject:** test: Add C++ softmax precision tests (bug not reproduced)

**Type:** Tests

**Critical Finding:** Bug cannot be reproduced in isolated C++ tests, even with exact BERT Q@K^T scores.

---

#### Commit 40: `a8b8527d06`
**Subject:** docs: Emphasize FP32 softmax workaround is NOT a fix - PERFORMANCE DEGRADATION

**Type:** Documentation + Code Comments

**Key Message:**
- This is a WORKAROUND, not a fix
- Causes PERFORMANCE DEGRADATION
- TODO: Remove once TTNN fixes bfloat16 softmax kernel

---

#### Commit 41: `d3b81efa58`
**Subject:** fix: Change softmax workaround default to false - preserve TTNN framework

**Type:** Bug Fix
**Production Code:** Yes

**Rationale:**
- Default true would break TTNN framework globally
- Forces FP32 accumulation on ALL softmax operations
- Workaround should be opt-in, not opt-out

---

#### Commit 42: `f29e470dea`
**Subject:** docs: Critical - SoftmaxPrecisionBug test suite breaks ALL subsequent tests

**Type:** Documentation
**Files Created:** `tt-train/TEST_FAILURE_BertPolymorphismTest.md`

---

#### Commit 43: `cf46393fc4`
**Subject:** fix: Add proper device cleanup to SoftmaxPrecisionBug test fixture

**Type:** Bug Fix
**Production Code:** Tests

**Problem:** SoftmaxPrecisionBug tests left device open, breaking all subsequent tests.

**Fix:** Added proper `TearDown()` with device cleanup.

---

#### Commit 44: `28b4300bea`
**Subject:** docs+code: Emphasize workarounds are NOT fixes - TTNN bugs remain

**Type:** Documentation + Code

**Key Changes:**
- Added WARNING comments to `embedding_op.cpp` and `unary_ops.cpp`
- Updated all documentation to emphasize workarounds are temporary

---

### Phase 7: Final Cleanup & Bug Reports (Commits 45-49)

#### Commit 45: `058b5c0471`
**Subject:** fix: Silence RAND_MAX implicit conversion warnings in test

**Type:** Bug Fix (Minor)
**Files Modified:** `tt-train/tests/ops/multi_head_attention_batch_regression_test.cpp`

---

#### Commit 46: `91fe6e0b80`
**Subject:** test: Fix test infrastructure issues

**Type:** Bug Fix
**Production Code:** Tests

**Changes:**
- Commented out `softmax_precision_test.cpp` (missing API)
- Fixed test fixtures with proper device lifecycle
- Fixed unused function warning

---

#### Commit 47: `7b09b22293`
**Subject:** docs: Add comprehensive TTNN bug reports for workarounded issues

**Type:** Documentation
**Files Created:**
- `tt-train/TTNN_BUG_REPORT_EMBEDDING_BATCH_PROCESSING.md`
- `tt-train/TTNN_BUG_REPORT_SOFTMAX_BFLOAT16_PRECISION.md`

**Purpose:** Self-contained bug reports ready to submit to TTNN team.

---

#### Commit 48: `40051a3d8f`
**Subject:** docs: Fix softmax bug report - acknowledge failed reproduction

**Type:** Documentation

**Key Correction:** Bug cannot be reproduced in isolation. Only appears in full BERT execution.

---

#### Commit 49: `57ad7e4e66`
**Subject:** docs: Add comprehensive bug reports for workarounded issues

**Type:** Documentation
**Files Created:**
- `tt-train/TTNN_BUG_REPRODUCTION_EMBEDDING.md`
- `tt-train/TTNN_BUG_REPRODUCTION_SOFTMAX.md`

**Purpose:** Complete runnable reproduction scripts for bug reports.

---

## File Change Summary

### Production Code Files (Keep)

| File | Purpose | Commits |
|------|---------|---------|
| `sources/ttml/models/bert_tasks.cpp`, `.hpp` | 5 task models | 2ee6c1839d, 457d6175bc |
| `sources/ttml/modules/bert_heads.cpp`, `.hpp` | 5 head modules | 2ee6c1839d |
| `sources/ttml/ops/bert_losses.cpp`, `.hpp` | 7 loss functions | 2ee6c1839d |
| `sources/ttml/serialization/bert_training_state.cpp`, `.hpp` | Training state | 35d709cc85 |
| `sources/ttml/models/bert_tasks_serialization.hpp` | Weight loading | 457d6175bc |
| `sources/ttml/nanobind/nb_models.cpp` | Python bindings | 457d6175bc, 65a3f7e94c |
| `sources/ttml/nanobind/nb_ops.cpp` | Loss bindings | 457d6175bc |
| `sources/ttml/ops/embedding_op.cpp` | Batch workaround | cd1c004988 |
| `sources/ttml/core/compute_kernel_config.cpp`, `.hpp` | Softmax workaround | 9a261d86cc, 668c173f81 |
| `sources/ttml/ttnn_fixed/trivial_ttnn_ops.cpp`, `.hpp` | Softmax workaround | 9a261d86cc |
| `sources/ttml/ops/unary_ops.cpp` | Workaround usage | d3b81efa58, 28b4300bea |

### Test Files (Review - Keep Production Tests)

| File | Purpose | Keep? |
|------|---------|-------|
| `tests/model/bert_task_heads_test.cpp` | C++ unit tests | Yes |
| `tests/python/test_bert_task_heads_basic.py` | Basic validation | Yes |
| `tests/python/test_bert_task_heads_hf_validation.py` | HF comparison | Yes |
| `tests/python/test_bert_batch_processing.py` | Batch regression | Yes |
| `tests/core/ttnn_embedding_batch_bug_test.cpp` | Bug reproduction | Yes |
| `tests/ops/embedding_word_vs_token_type_test.cpp` | Regression | Yes |
| `tests/core/softmax_precision_test.cpp` | Bug reproduction | Review |
| `tests/python/test_*_debug.py` | Debug tests | Remove |
| `tests/python/test_attention_*.py` | Investigation | Remove |
| `tests/python/test_embedding_*.py` | Investigation | Remove |
| `tests/python/test_linear_layer_*.py` | Investigation | Remove |
| `tests/python/test_sdpa_*.py` | Investigation | Remove |

### Documentation Files (Cleanup Needed)

| File | Action |
|------|--------|
| `TASK_HEADS_V2_DESIGN_DOCUMENT.md` | Keep (design reference) |
| `BERT_BUG_INVESTIGATION_STATUS.md` | Remove or consolidate |
| `BERT_TASK_HEADS_IMPLEMENTATION_STATUS.md` | Remove or consolidate |
| `BUG_ROOT_CAUSE_FOUND.md` | Remove (use GitHub issues) |
| `TTNN_BUG_REPORT_*.md` | Extract to GitHub issues, then remove |
| `TTNN_BUG_REPRODUCTION_*.md` | Extract to GitHub issues, then remove |
| `*_INVESTIGATION*.md` | Remove |
| `TEST_FAILURE_*.md` | Remove |

---

## Recommendations

### 1. Immediate Actions

1. **File GitHub issues** for the two workarounded bugs:
   - Embedding batch processing bug (verify vs #30418)
   - Softmax bfloat16 precision bug

2. **Clean up documentation:**
   - Delete ~20 investigation/status markdown files
   - Consolidate to single `BERT_TASK_HEADS_README.md`

3. **Remove debug tests:**
   - Delete `test_*_debug.py` files
   - Delete `test_attention_*.py` investigation files
   - Delete `test_embedding_*.py` investigation files
   - Delete `test_linear_layer_*.py` investigation files

### 2. PR Split Strategy

| PR | Contents | Commits |
|----|----------|---------|
| PR A | Embedding batch workaround | cd1c004988, edd61851c3, fcbedf610f |
| PR B | Softmax FP32 option | 9a261d86cc, 668c173f81, 77e57b4335, d3b81efa58, cf46393fc4, 28b4300bea |
| PR C | Task heads implementation | 2ee6c1839d, 457d6175bc, 35d709cc85, a8284d8e28, + fixes |
| PR D | Task heads tests | e0daa0febb, 8827592407, + test files |

### 3. Commits to Squash/Remove

- ~25 documentation commits → squash to 1
- ~10 debug test commits → remove entirely
- ~5 investigation test commits → remove or convert to proper tests
