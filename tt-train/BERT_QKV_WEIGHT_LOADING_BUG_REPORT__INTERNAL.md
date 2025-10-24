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
  Max diff: 4.56
  Status: ❌ CATASTROPHIC FAILURE
```

### bert-tiny
```
Model: prajjwal1/bert-tiny (2 layers, 128 hidden, 2 heads)
Test: test_bert_qkv_loading_golden_reference[1-32-prajjwal1/bert-tiny]

Results:
  PCC: -0.010925 (Expected: >0.99)
  Status: ❌ CATASTROPHIC FAILURE
```

---

## Root Cause Analysis

### Layer 0 QKV Weight Comparison

**Expected values (from safetensors):**
```
Q[0,0] = -0.0164
K[768,0] = 0.0081  (K matrix starts at row 768)
V[1536,0] = 0.0114  (V matrix starts at row 1536)
```

**Actual loaded values (TTML):**
```
TTML[0,0] = 0.0102     ❌ Should be -0.0164
TTML[768,0] = 0.0369   ❌ Should be 0.0081
TTML[1536,0] = -0.0238 ❌ Should be 0.0114
```

**Difference statistics:**
- Mean difference: 0.034
- Max difference: 0.829
- Standard deviation mismatch: Actual std=0.020, Expected std=0.039 (50% of expected!)

### Concatenation Pattern Testing

Tested 4 different concatenation patterns:

1. **cat(Q, K, V, dim=0)** → PCC = -0.000854 ❌
2. **cat(Q.T, K.T, V.T, dim=1).T** → PCC = -0.000854 ❌
3. **cat(Q, K, V, dim=1)** → Shape mismatch
4. **cat(Q.T, K.T, V.T, dim=0)** → PCC = -0.000041 ❌

**All patterns fail identically**, indicating the problem is NOT the concatenation logic but something earlier in the pipeline.

---

## Investigation Findings

### 1. Safetensors File Validation ✅
```python
# Direct inspection confirms safetensors has correct values
with safe_open("/tmp/bert-base-uncased.safetensors") as f:
    q_weight = f.get_tensor("encoder.layer.0.attention.self.query.weight")
    # Q[0,0] = -0.0164  ✓ Correct
```

### 2. C++ Code Review
The concatenation logic in `bert.cpp:558` appears correct:
```cpp
auto Q = xt::adapt(Q_vec, std::vector<size_t>{hidden_size, hidden_size});
auto K = xt::adapt(K_vec, std::vector<size_t>{hidden_size, hidden_size});
auto V = xt::adapt(V_vec, std::vector<size_t>{hidden_size, hidden_size});
auto qkv_combined = core::concat(std::vector<xt::xarray<float>>{Q, K, V}, 0);
```

### 3. Suspected Issues

**Hypothesis 1: `core::from_vector` / device storage corruption**
- Weights might be corrupted during `core::from_vector()` call
- Device storage might reorder/transform data unexpectedly

**Hypothesis 2: `to_numpy()` retrieval issue**
- Weights stored correctly but retrieved incorrectly
- Possible row-major vs column-major confusion

**Hypothesis 3: xtensor `concat` function bug**
- `core::concat` may not work as expected
- Possible axis interpretation issue

**Hypothesis 4: Float conversion**
- `bytes_to_floats_copy()` may have endianness or alignment issues

---

## Test Artifacts Created

### 1. Diagnostic Test
**File:** `tests/python/test_bert_base_uncased_diagnostic.py`

Features:
- Layer-by-layer comparison with HuggingFace
- Captures intermediate outputs with hooks
- Tests all 4 QKV concatenation patterns
- Comprehensive metrics (PCC, mean/max diff, std deviation)

### 2. Debug Scripts
**Files:**
- `tests/python/debug_qkv_loading.py` - Direct weight inspection
- `tests/python/inspect_safetensors.py` - Safetensors validation
- `tests/python/compare_loaded_weights.py` - Expected vs actual comparison

### 3. Saved Artifacts
- `/tmp/qkv_expected.npy` - Expected QKV weights from HF
- `/tmp/qkv_actual.npy` - Actual loaded weights from TTML
- `/tmp/diagnostic_output.log` - Full diagnostic test output

---

## Impact Assessment

### Severity: 🔴 CRITICAL

**Affected Functionality:**
- ❌ Cannot load any HuggingFace BERT models
- ❌ All pretrained BERT weights unusable
- ❌ bert-tiny tests fail
- ❌ bert-base-uncased tests fail
- ❌ Likely affects all BERT variants (bert-large, etc.)

**Production Readiness:**
- Current status: **NOT production-ready**
- Previous claim of PCC 0.84: **Incorrect/unvalidated**
- Commit `04d1cf4f3e` message: **Misleading**

---

## Recommended Next Steps

### Immediate Actions

1. **Add C++ Debug Logging**
   - Log Q, K, V vectors before concatenation
   - Log concatenated result before `core::from_vector`
   - Log retrieved weights after `to_numpy()`
   - Compare at each step

2. **Test `core::concat` in Isolation**
   - Create minimal C++ unit test
   - Verify concatenation works correctly
   - Rule out xtensor issues

3. **Test `core::from_vector` Roundtrip**
   - Store known values with `from_vector`
   - Retrieve with `to_numpy()`
   - Verify no corruption

4. **Check Device Storage**
   - Investigate if device has specific layout requirements
   - Check if bf16 conversion affects weight loading
   - Verify memory alignment

### Long-term Fixes

1. **Rewrite QKV Loading**
   - Use proven working patterns from other models
   - Add comprehensive validation at each step
   - Create golden reference tests BEFORE claiming success

2. **Add Continuous Validation**
   - Run golden reference tests in CI
   - Require PCC >0.99 for any weight loading changes
   - Test multiple model sizes (tiny, base, large)

3. **Update Documentation**
   - Remove "production-ready" claims from commit 04d1cf4f3e
   - Document known limitations
   - Add troubleshooting guide

---

## Files Modified in This Investigation

### Branch: `ivoitovych/bert-base-uncased-validation`

**Modified:**
- `tests/python/test_bert_golden_reference.py` - Added bert-base-uncased to parametrize

**Created:**
- `tests/python/test_bert_base_uncased_diagnostic.py` - Layer-by-layer diagnostic
- `tests/python/debug_qkv_loading.py` - Debug script
- `tests/python/inspect_safetensors.py` - Safetensors inspector
- `tests/python/compare_loaded_weights.py` - Weight comparison
- `BERT_QKV_WEIGHT_LOADING_BUG_REPORT__INTERNAL.md` - This document

---

## Conclusion

The QKV weight loading implementation requires **complete rewrite**. The current implementation:
- Produces completely wrong weights
- Fails for both small and large models
- Cannot load any HuggingFace BERT models correctly
- Was never properly validated

**Do NOT merge** any code depending on this functionality until the root cause is identified and fixed, with comprehensive validation showing PCC >0.99 on multiple model sizes.

---

## Appendix: Test Commands

### Run diagnostic test:
```bash
python3 -m pytest tests/python/test_bert_base_uncased_diagnostic.py::test_bert_base_uncased_layer_by_layer -v -s
```

### Run golden reference tests:
```bash
python3 -m pytest tests/python/test_bert_golden_reference.py -v -s
```

### Run debug scripts:
```bash
python3 tests/python/debug_qkv_loading.py
python3 tests/python/inspect_safetensors.py
python3 tests/python/compare_loaded_weights.py
```
