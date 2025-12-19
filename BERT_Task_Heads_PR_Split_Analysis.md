# BERT Task Heads PR Split Analysis

**PR Branch:** `ivoitovych/bert-model-for-ttml-task-heads-v2`
**Base Branch:** `ivoitovych/bert-model-for-ttml-backup-2025-12-18`
**Total Commits:** 49
**Analysis Date:** 2025-12-19

## Executive Summary

The BERT Task Heads branch contains 49 commits with significant technical debt:
- ~25 documentation/investigation commits (development notes, not production docs)
- Multiple bug workarounds that should be separate PRs
- Core task heads implementation mixed with debugging code

**Recommended approach:** Split into 5 focused PRs and clean up documentation.

**Blockers:** Several known issues affect test reliability and require workarounds. See [Known Blockers and Issues](#known-blockers-and-issues) section.

## Commit Categories

### Category 1: Core Task Heads Implementation (~12 commits)

These commits form the core feature and should stay together:

| Commit | Description |
|--------|-------------|
| `2ee6c1839d` | feat: Implement BERT task heads architecture v2 |
| `457d6175bc` | feat: Implement critical integration components |
| `35d709cc85` | feat: Add comprehensive skeleton implementations |
| `e0daa0febb` | feat: Complete BERT Task Heads validation tests |
| `a8284d8e28` | fix: Fix compilation errors and test isolation |
| `65a3f7e94c` | fix: Add Python bindings for BERT task models |
| `8233fd06a2` | fix: Add type_vocab_size binding |
| `f55c609017` | fix: Handle both file and directory paths |
| `8827592407` | fix: Fix golden reference tests |
| `6eee1d3d12` | feat: Identify root cause with granular embedding decomposition |

**Files:**
- `tt-train/sources/ttml/models/bert_tasks.cpp`, `bert_tasks.hpp`
- `tt-train/sources/ttml/modules/bert_heads.cpp`, `bert_heads.hpp`
- `tt-train/sources/ttml/ops/bert_losses.cpp`, `bert_losses.hpp`
- `tt-train/sources/ttml/serialization/bert_training_state.cpp`, `.hpp`
- `tt-train/sources/ttml/nanobind/nb_models.cpp`
- `tt-train/configs/bert_*.yaml`
- `tt-train/tests/model/bert_task_heads_test.cpp`

---

### Category 2: Embedding Batch Bug Workaround (INDEPENDENT PR)

**Status:** Can be split as independent PR

| Commit | Description |
|--------|-------------|
| `cd1c004988` | fix: Workaround for TTNN embedding batch processing bug (PCC 0.608 -> 0.999999) |
| `edd61851c3` | test: Add minimal C++ test for TTNN embedding batch processing bug |
| `fcbedf610f` | test: Add C++ regression tests for embedding batch processing |
| `84723951a8` | fix: Fix I64 dtype error in safetensors loading |

**Files:**
- `tt-train/sources/ttml/ops/embedding_op.cpp` - Workaround implementation
- `tt-train/tests/core/ttnn_embedding_batch_bug_test.cpp` - Bug reproduction test
- `tt-train/tests/ops/embedding_word_vs_token_type_test.cpp` - Regression tests

**Why Independent:**
- Workaround for TTNN bug #30418
- Benefits all embedding users, not just BERT
- Has dedicated tests

**Related Issue:** [#30418](https://github.com/tenstorrent/tt-metal/issues/30418)

---

### Category 3: Softmax Precision Workaround (INDEPENDENT PR)

**Status:** Can be split as independent PR

| Commit | Description |
|--------|-------------|
| `9a261d86cc` | workaround: Add FP32 accumulation option for softmax bfloat16 precision bug |
| `668c173f81` | fix: Enable FP32 accumulation in softmax for BERT precision |
| `77e57b4335` | test: Add C++ softmax precision tests (bug not reproduced) |
| `d3b81efa58` | fix: Change softmax workaround default to false |
| `cf46393fc4` | fix: Add proper device cleanup to SoftmaxPrecisionBug test |
| `28b4300bea` | docs+code: Emphasize workarounds are NOT fixes |

**Files:**
- `tt-train/sources/ttml/core/compute_kernel_config.cpp`, `.hpp`
- `tt-train/sources/ttml/ttnn_fixed/trivial_ttnn_ops.cpp`, `.hpp`
- `tt-train/sources/ttml/ops/unary_ops.cpp`
- `tt-train/tests/core/softmax_precision_test.cpp`

**Why Independent:**
- Optional FP32 accumulation for softmax precision
- Benefits all softmax users
- Default is OFF (preserves performance)

**Note:** Bug reproduction was unsuccessful in isolated tests, but workaround helps BERT accuracy.

---

### Category 4: TTNN Bug Reports (INDEPENDENT - For Issue Filing)

**Status:** Can be extracted for filing GitHub issues

| Commit | Description |
|--------|-------------|
| `57ad7e4e66` | docs: Add comprehensive bug reports for workarounded issues |
| `40051a3d8f` | docs: Fix softmax bug report - acknowledge failed reproduction |
| `7b09b22293` | docs: Add comprehensive TTNN bug reports for workarounded issues |

**Files:**
- `tt-train/TTNN_BUG_REPRODUCTION_EMBEDDING.md`
- `tt-train/TTNN_BUG_REPRODUCTION_SOFTMAX.md`
- `tt-train/TTNN_BUG_REPORT_EMBEDDING_BATCH_PROCESSING.md`
- `tt-train/TTNN_BUG_REPORT_SOFTMAX_BFLOAT16_PRECISION.md`

**Action:** Extract these to file as GitHub issues, then remove from PR.

---

### Category 5: Documentation (CLEANUP NEEDED)

**~25 commits** of investigation notes and status reports that should NOT be in production:

| Type | Files | Action |
|------|-------|--------|
| Investigation reports | `BERT_BUG_INVESTIGATION_STATUS.md`, `BUG_ROOT_CAUSE_FOUND.md` | Remove or move to wiki |
| Implementation notes | `BERT_TASK_HEADS_IMPLEMENTATION_*.md` | Keep 1 consolidated doc |
| Status reports | `TASK_HEADS_*.md` | Remove |
| Layer PCC reports | `BERT_LAYER_PCC_REPORT.txt` | Remove |
| Test failure notes | `TEST_FAILURE_*.md` | Remove |

**Recommendation:** Consolidate to single `BERT_TASK_HEADS_README.md` with:
- Feature overview
- Usage examples
- Known limitations
- Link to related issues

---

### Category 6: Debug/Validation Tests (REVIEW NEEDED)

Many Python tests added for debugging that may not be suitable for CI:

| Test File | Purpose | Keep? |
|-----------|---------|-------|
| `test_bert_batch_processing.py` | Batch regression | Yes |
| `test_bert_task_heads_basic.py` | Basic validation | Yes |
| `test_bert_task_heads_hf_validation.py` | HuggingFace comparison | Yes |
| `test_bert_base_uncased_debug.py` | Debug test | Review |
| `test_bert_layer_by_layer_validation.py` | Debug test | Review |
| `test_attention_*.py` | Investigation tests | Remove |
| `test_embedding_*.py` | Investigation tests | Remove |
| `test_linear_layer_*.py` | Investigation tests | Remove |
| `test_*_debug.py` | Debug tests | Remove |

---

## Recommended PR Split

### PR A: Embedding Batch Workaround

**Commits:** 4
**Complexity:** Low
**Ready:** Yes

- Workaround for TTNN embedding batch bug
- Regression tests
- Reference to issue #30418

---

### PR B: Softmax FP32 Accumulation Option

**Commits:** 6 (squashed to 1-2)
**Complexity:** Low
**Ready:** Yes (after cleanup)

- Optional FP32 accumulation for softmax
- Default OFF for performance
- Tests for precision validation

---

### PR C: BERT Task Heads Implementation

**Commits:** ~8 (after cleanup)
**Complexity:** High
**Dependencies:** PRs A, B (for workarounds)

- Task heads architecture (BertForSequenceClassification, BertForTokenClassification, etc.)
- Training state serialization
- Python bindings
- Config files

**Files:**
- `tt-train/sources/ttml/models/bert_tasks.cpp`, `bert_tasks.hpp`
- `tt-train/sources/ttml/modules/bert_heads.cpp`, `bert_heads.hpp`
- `tt-train/sources/ttml/ops/bert_losses.cpp`, `bert_losses.hpp`
- `tt-train/sources/ttml/serialization/bert_training_state.cpp`, `.hpp`
- `tt-train/sources/ttml/nanobind/nb_models.cpp`
- `tt-train/configs/bert_*.yaml`
- `tt-train/examples/train_bert_classifier.cpp`, `.py`

---

### PR D: BERT Task Heads Tests

**Commits:** ~4 (after cleanup)
**Complexity:** Medium
**Dependencies:** PR C

- C++ validation tests
- Python HuggingFace validation tests
- Batch processing regression tests

**Files:**
- `tt-train/tests/model/bert_task_heads_test.cpp`
- `tt-train/tests/python/test_bert_task_heads_basic.py`
- `tt-train/tests/python/test_bert_task_heads_hf_validation.py`
- `tt-train/tests/python/test_bert_batch_processing.py`

**Why Split Tests:**
- Tests can be reviewed independently from implementation
- Enables faster iteration on test improvements
- Reduces cognitive load for implementation reviewers

---

### Bug Reports (File as Issues - NOT a PR)

**Action:** Don't create PR - file as GitHub issues instead
- Extract bug reproduction steps from markdown files
- File issues against tt-metal/ttnn
- Remove markdown files from branch after issues are filed

---

## Cleanup Actions

1. **Squash documentation commits** - Reduce 25+ docs commits to 1-2
2. **Remove debug tests** - Keep only production-quality tests
3. **Consolidate docs** - Single README instead of 10+ files
4. **Remove investigation notes** - These are development artifacts

---

## Summary Table

| PR | Commits (current) | Commits (cleaned) | Complexity | Dependencies | Ready |
|----|-------------------|-------------------|------------|--------------|-------|
| PR A: Embedding Workaround | 4 | 2 | Low | None | Yes |
| PR B: Softmax FP32 Option | 6 | 2 | Low | None | Yes (after cleanup) |
| PR C: Task Heads Implementation | ~8 | ~6 | High | PRs A, B | After above |
| PR D: Task Heads Tests | ~4 | ~3 | Medium | PR C | After PR C |
| **Documentation cleanup** | ~25 | 1 | - | - | - |
| **Debug tests removal** | ~10 | 0 | - | - | - |

**Current total:** 49 commits
**After cleanup:** ~14 commits across 4 PRs

---

## Known Blockers and Issues

### Workarounded in This Branch (Issues NOT YET Filed)

These bugs are workarounded in the code but GitHub issues have not been filed yet:

| Bug | Description | Workaround | Status |
|-----|-------------|------------|--------|
| **Embedding Batch Bug** | TTNN embedding produces incorrect results for batch>1 | Loop over batch dimension in embedding_op.cpp | **Needs issue filed** |
| **Softmax BFloat16 Precision** | Softmax with bfloat16 accumulation causes accuracy degradation | Optional FP32 accumulation (default OFF) | **Needs issue filed** |

**Action Required:** File these as GitHub issues before submitting PRs A and B.

### Workarounded in Code (Issue Already Filed)

| Issue | Description | Status | Impact |
|-------|-------------|--------|--------|
| [#30418](https://github.com/tenstorrent/tt-metal/issues/30418) | TTNN embedding batch processing issue | Workarounded | PR A contains workaround |

### Flaky Test Issues (Poison Regression Analysis)

These issues cause unpredictable test failures during pre-PR validation:

| Issue | Description | Symptom | Status |
|-------|-------------|---------|--------|
| [#34761](https://github.com/tenstorrent/tt-metal/issues/34761) | Test isolation: LayerNorm backward dgamma tolerance failure | dgamma fails after SoftmaxTest + SiLUOpTest + SDPA NIGHTLY | Open |
| [#34747](https://github.com/tenstorrent/tt-metal/issues/34747) | Test isolation: LayerNorm backward NIGHTLY produces NaN | NaN after OneTile + TwoIncompleteTiles tests | Open |
| [#34625](https://github.com/tenstorrent/tt-metal/issues/34625) | LayerNorm backward kernel accumulation bug | Only last tiles contribute to sum | Fix PR submitted |
| [#34451](https://github.com/tenstorrent/tt-metal/issues/34451) | Test flakiness issue | Unpredictable failures | Open |

**Impact:** These issues can cause false-negative CI results, requiring manual investigation to determine if failures are caused by the PR under review or pre-existing flaky tests.

---

## Recommended Submission Order

```
Phase 1 (File bugs first):
├── File GitHub issue for Embedding Batch Bug
└── File GitHub issue for Softmax BFloat16 Precision Bug

Phase 2 (Parallel - No Dependencies):
├── PR A: Embedding Batch Workaround
└── PR B: Softmax FP32 Option

Phase 3 (After PRs A, B):
└── PR C: Task Heads Implementation

Phase 4 (After PR C):
└── PR D: Task Heads Tests
```

---

## Next Steps

1. [ ] File GitHub issue for Embedding Batch Bug
2. [ ] File GitHub issue for Softmax BFloat16 Precision Bug
3. [ ] Create PR A (Embedding Workaround) - reference new issue
4. [ ] Create PR B (Softmax FP32 Option) - reference new issue
5. [ ] Clean up documentation (squash/remove)
6. [ ] Remove debug/investigation tests
7. [ ] Create PR C (Task Heads Implementation) - after PRs A, B merge
8. [ ] Create PR D (Task Heads Tests) - after PR C merges
9. [ ] Monitor blockers - track resolution of issues #34761, #34747, #34625, #34451
