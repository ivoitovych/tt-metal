# BERT PR Split Analysis

**PR Branch:** `ivoitovych/bert-model-for-ttml`
**Total Commits:** 20
**Analysis Date:** 2025-12-19

## Executive Summary

The BERT PR contains 20 commits that can be split into **8 separate PRs**:
- 6 independent PRs (general-purpose enhancements and tests)
- 2 BERT-specific PRs (implementation + tests)

**Benefits:**
- Enable parallel review of independent components
- Allow faster merging of general-purpose enhancements
- Separate BERT implementation from BERT tests for focused reviews
- Reduce cognitive load per PR

**Blockers:** Several known issues affect test reliability and may cause false CI failures. See [Known Blockers and Issues](#known-blockers-and-issues) section.

## Independent PR Candidates

### PR 1: Tanh Activation Function

**Status:** Needs test improvements before submission

| Commit | Description |
|--------|-------------|
| `48a43d04b6` | feat(ops): Add tanh activation function with autograd support |

**Files Changed:**
- `tt-train/sources/ttml/ops/unary_ops.cpp` - Add tanh forward/backward
- `tt-train/sources/ttml/ops/unary_ops.hpp` - Add tanh declaration
- `tt-train/tests/ops/unary_ops_test.cpp` - Add tanh tests

**Why Independent:**
- Follows established unary ops pattern
- Self-contained with its own tests
- Zero BERT dependencies
- Useful for any model needing tanh activation

**Current Test Coverage (~95 lines):**
- Forward pass with 8 fixed values, single shape `{2,1,1,4}`
- Backward pass gradient verification using MSE loss
- Saturation region behavior (extreme values ±10, ±5)

**Test Quality Issues:**

| Gap | Description |
|-----|-------------|
| Shape variations | Only tests `{2,1,1,4}` - no batch sizes, sequence lengths, tile boundaries |
| Random data | Fixed values only - other tests (e.g., `Silu`) use `load_random_data_from_os()` |
| Precision comparison | No metal vs CPU reference comparison |
| NaN/Inf handling | GELU tests have edge case tests, tanh doesn't |
| Different dtypes | No bfloat16 testing |
| Tile alignment | No non-tile-aligned tensor tests |
| Autograd chain | No testing in larger computation graphs |
| Numerical stability | No gradient vanishing/exploding checks |

**Comparison:** GELU tests have ~640 lines with comprehensive coverage; tanh tests have ~95 lines with minimal coverage.

**Action Required:** Enhance tanh tests to match GELU test quality before submitting as independent PR, or note in PR description that test coverage is minimal and should be expanded in follow-up.

---

### PR 2: Binary Ops Broadcasting Support

**Status:** Ready to submit independently

| Commit | Description |
|--------|-------------|
| `6349cf7cfd` | Add broadcasting support and comprehensive tests for binary operations |

**Files Changed:**
- `tt-train/sources/ttml/ops/binary_ops.cpp` - Add broadcasting to operator- and operator*
- `tt-train/tests/ops/binary_ops_test.cpp` - 22 comprehensive tests (new file)

**Why Independent:**
- Extends existing binary ops (addition already had broadcasting)
- Enables proper gradient computation for broadcast operations
- No BERT-specific code in implementation
- Critical for residual connections in any transformer model

**Test Coverage:**
- All binary operations (add, sub, mul, div)
- Broadcasting edge cases
- Gradient accumulation
- Complex expression graphs

---

### PR 3: Slice/Repeat Operations Tests

**Status:** Ready to submit independently

| Commit | Description |
|--------|-------------|
| `866a69defa` | test: Add comprehensive tests for ttnn::slice and ttnn::repeat operations |

**Files Changed:**
- `tt-train/tests/ops/slice_repeat_ops_test.cpp` - New test file (603 lines)

**Why Independent:**
- Pure test coverage for existing ttnn operations
- No source code changes
- Documents expected behavior of slice/repeat
- Useful for any model using these operations

**Test Coverage:**
- ttnn::repeat: 6 test cases (1D, multi-dim, batch, no-op, large shapes, non-aligned)
- ttnn::slice: 8 test cases (1D, multi-dim, stride, edge cases)

---

### PR 4: Scaled Dot Product Attention Tests

**Status:** Ready to submit independently

| Commit | Description |
|--------|-------------|
| `fb87b170cb` | Add comprehensive scaled_dot_product_attention autograd tests |

**Files Changed:**
- `tt-train/tests/ops/scaled_dot_product_attention_test.cpp` - New test file (506 lines)

**Why Independent:**
- Pure test coverage for existing SDPA autograd interface
- No source code changes
- Complements existing metal-level SDPA tests
- Critical for any transformer model validation

**Test Coverage:**
- High-level autograd SDPA interface
- Backward pass with gradient verification
- Grouped query attention (GQA)
- Attention masks and padding
- Numerical stability with large sequences

---

### PR 5: GELU Activation Test Suite

**Status:** Ready to submit independently

| Commit | Description |
|--------|-------------|
| `5ed6a7eae1` | Add comprehensive GELU activation test suite for BERT implementation |
| `720fc020ea` | Add NaN/Inf edge case tests and improve GELU test suite |

**Files Changed:**
- `tt-train/tests/ops/gelu_op_test.cpp` - New test file (536+ lines)

**Why Independent:**
- Pure test coverage for existing GELU operation
- No source code changes
- Documents GELU behavior and edge cases
- Essential for transformer model validation

**Test Coverage:**
- Forward pass accuracy
- Backward pass gradients
- NaN/Inf edge case handling
- Numerical precision validation

---

### PR 6: LayerNorm Configurable Epsilon

**Status:** Needs minor refactoring before independent submission

| Commit | Description |
|--------|-------------|
| `6ee8d0e5df` | feat(ttml): Add configurable epsilon with fully tunable hardware clamping control |

**Files Changed:**
- `tt-train/sources/ttml/ops/layernorm_op.cpp` - Add epsilon parameter
- `tt-train/sources/ttml/ops/layernorm_op.hpp` - Add epsilon parameter
- `tt-train/sources/ttml/modules/layer_norm_module.cpp` - Store epsilon
- `tt-train/sources/ttml/modules/layer_norm_module.hpp` - Add epsilon config
- `tt-train/sources/ttml/nanobind/nb_ops.cpp` - Python bindings
- `tt-train/tests/modules/layer_norm_epsilon_test.cpp` - C++ tests
- `tt-train/tests/python/test_layernorm_epsilon.py` - Python tests
- `tt-train/sources/ttml/models/bert.cpp` - BERT integration (remove for independent PR)
- `tt-train/configs/bert_config*.yaml` - BERT config (remove for independent PR)

**Why Partially Independent:**
- Core layernorm changes are general-purpose
- BERT-specific changes can be moved to BERT PR
- Enables proper epsilon handling for all models

**Refactoring Required:**
1. Remove BERT model changes from this commit
2. Remove BERT config file changes
3. Keep only layernorm_op, layer_norm_module, and test changes

---

## BERT Core PRs (Remaining Commits)

After splitting independent PRs, the BERT work should be split into **two separate PRs**:

### PR 7: BERT Implementation (Model + Infrastructure)

| # | Commit | Description |
|---|--------|-------------|
| 1 | `80747d0499` | BERT for TTML: Initial commit |
| 2 | `2825c073de` | Integrate Safetensors to the BERT model |
| 3 | `6a5173847a` | Fix BERT example: align input sequence length |
| 4 | `3c8029e4f8` | Enable tanh activation in pooler output |
| 5 | `b1da3974ec` | Fix BERT build: update ModuleBase namespace |
| 6 | `df22512692` | Add BaseTransformer polymorphism and fix embedding layout |
| 7 | `f8b3b78f99` | Implement QKV weight loading from HuggingFace safetensors |
| 8 | `141bffe8a2` | Fix critical bugs in BERT implementation (implementation parts) |
| 9 | `4eb8848881` | Fix build errors after rebase: layernorm -> layernorm_moreh |

**Files:**
- `tt-train/sources/ttml/models/bert.cpp`, `bert.hpp`
- `tt-train/sources/ttml/modules/bert_block.cpp`, `bert_block.hpp`
- `tt-train/sources/ttml/modules/multi_head_attention.cpp`
- `tt-train/sources/ttml/ops/multi_head_utils.cpp`
- `tt-train/sources/ttml/ops/scaled_dot_product_attention.cpp`
- `tt-train/sources/ttml/nanobind/nb_models.cpp`, `nb_ops.cpp`
- `tt-train/sources/examples/bert_example/`
- `tt-train/configs/bert_config*.yaml`

**Plus:** LayerNorm epsilon BERT integration (from PR 6 refactoring)

---

### PR 8: BERT Tests + Test Utilities

| # | Commit | Description |
|---|--------|-------------|
| 1 | `141bffe8a2` | Fix critical bugs in BERT implementation (test parts) |
| 2 | `f6a9a66c92` | Fix test isolation bug in BERTOperatorTest |
| 3 | `478fb5f074` | Fix test isolation in TileLayoutRoundTripTest |
| 4 | `e0d55fab90` | Fix Python test bugs and standardize test utilities |
| 5 | `228e5783f0` | Fix compute_pcc bfloat16 bug |

**Files:**
- `tt-train/tests/model/bert_operator_test.cpp`
- `tt-train/tests/model/bert_polymorphism_test.cpp`
- `tt-train/tests/model/bert_weight_loading_test.cpp`
- `tt-train/tests/core/tile_layout_round_trip_test.cpp`
- `tt-train/tests/python/test_bert_*.py` (all BERT Python tests)
- `tt-train/tests/python/test_utils.py`
- `tt-train/tests/python/test_layernorm_epsilon.py`

**Why Split Tests:**
- Tests can be reviewed independently from implementation
- Test utilities (`test_utils.py`, `compute_pcc` fix) benefit all tests, not just BERT
- Enables faster iteration on test improvements
- Reduces cognitive load for implementation reviewers

---

### Dependencies

```
PR 1 (Tanh) ─────────────────┐
PR 6 (LayerNorm Epsilon) ────┼──► PR 7 (BERT Implementation) ──► PR 8 (BERT Tests)
PRs 2-5 (Independent) ───────┘
```

- PR 1 (Tanh) must merge before PR 7 (BERT uses tanh in pooler)
- PR 6 (LayerNorm epsilon) should merge before PR 7 for cleaner integration
- PR 7 (BERT Implementation) must merge before PR 8 (BERT Tests)

---

## Recommended Submission Order

```
Phase 1 (Parallel - No Dependencies):
├── PR 3: Slice/Repeat Tests
├── PR 4: SDPA Tests
└── PR 5: GELU Tests

Phase 2 (After Phase 1 or Parallel):
├── PR 1: Tanh Activation
└── PR 2: Binary Ops Broadcasting

Phase 3 (After PR 1):
└── PR 6: LayerNorm Epsilon (refactored)

Phase 4 (After PRs 1, 6):
└── PR 7: BERT Implementation

Phase 5 (After PR 7):
└── PR 8: BERT Tests
```

---

## Summary Table

| PR | Commits | Lines Changed | Complexity | Dependencies | Ready |
|----|---------|---------------|------------|--------------|-------|
| PR 1: Tanh Op | 1 | ~110 | Low | None | Needs test improvements |
| PR 2: Binary Ops | 1 | ~740 | Low | None | Yes |
| PR 3: Slice/Repeat Tests | 1 | ~604 | Trivial | None | Yes |
| PR 4: SDPA Tests | 1 | ~507 | Trivial | None | Yes |
| PR 5: GELU Tests | 2 | ~640 | Trivial | None | Yes |
| PR 6: LayerNorm Epsilon | 1 | ~780 | Medium | None | Needs refactor |
| PR 7: BERT Implementation | ~9 | ~2500+ | High | PRs 1, 6 | After above |
| PR 8: BERT Tests | ~5 | ~2000+ | Medium | PR 7 | After PR 7 |

**Total:** 8 PRs from original 1, enabling parallel review and faster merging

---

## Known Blockers and Issues

### Workarounded in Code

| Issue | Description | Status | Impact on BERT |
|-------|-------------|--------|----------------|
| [#30418](https://github.com/tenstorrent/tt-metal/issues/30418) | ttnn operation issue | Workarounded | Code contains workaround |

### Flaky Test Issues (Poison Regression Analysis)

These issues cause unpredictable test failures during pre-PR validation, triggering unnecessary investigation cycles:

| Issue | Description | Symptom | Status |
|-------|-------------|---------|--------|
| [#34761](https://github.com/tenstorrent/tt-metal/issues/34761) | Test isolation: LayerNorm backward dgamma tolerance failure | dgamma fails after SoftmaxTest + SiLUOpTest + SDPA NIGHTLY | Open |
| [#34747](https://github.com/tenstorrent/tt-metal/issues/34747) | Test isolation: LayerNorm backward NIGHTLY produces NaN | NaN after OneTile + TwoIncompleteTiles tests | Open |
| [#34625](https://github.com/tenstorrent/tt-metal/issues/34625) | LayerNorm backward kernel accumulation bug | Only last tiles contribute to sum | Fix PR submitted |
| [#34451](https://github.com/tenstorrent/tt-metal/issues/34451) | Test flakiness issue | Unpredictable failures | Open |

**Impact:** Each of these issues can cause false-negative CI results, requiring manual investigation to determine if failures are caused by the PR under review or pre-existing flaky tests. This significantly increases review cycle time.

**Mitigation:** When reviewing BERT PRs, be aware that test failures may be caused by these known issues rather than the PR changes. Check if failures match known patterns before requesting fixes.

---

## Next Steps

1. [ ] Create PR 3 (Slice/Repeat Tests) - trivial, merge quickly
2. [ ] Create PR 4 (SDPA Tests) - trivial, merge quickly
3. [ ] Create PR 5 (GELU Tests) - trivial, merge quickly
4. [ ] Create PR 1 (Tanh Op) - low complexity
5. [ ] Create PR 2 (Binary Ops) - low complexity
6. [ ] Refactor PR 6 (LayerNorm Epsilon) - separate BERT changes
7. [ ] Create PR 7 (BERT Implementation) - after PRs 1, 6 merge
8. [ ] Create PR 8 (BERT Tests) - after PR 7 merges
9. [ ] Monitor blockers - track resolution of issues #34761, #34747, #34625, #34451
