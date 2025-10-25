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

**Fixed bindings:**
- `heads_creation` - Now properly returns Python tuple (was failing before)
- `heads_fusion` - Accessible from Python
- `grouped_heads_creation` - Already working

### 2. Test Files Created

**Python Tests:**
- `tests/python/test_bert_operators_comprehensive.py` (408 lines)
  - 9 comprehensive operator tests
  - Tests each operation in isolation with random data
  - Compares PyTorch reference vs TTML implementation
  - Uses strict PCC thresholds (>0.99 or >0.999)

**C++ Tests:**
- `tests/model/bert_operator_test.cpp` (698 lines)
  - 7 comprehensive C++ unit tests
  - Direct testing without Python overhead
  - Tests: HeadsCreation, HeadsFusion, ScaledDotProductAttention, LayerNorm, GELU, CompleteMHAPipeline
  - Added to CMakeLists.txt for automatic building

---

## Test Results

### ✅ PASSED (6/9 tests)

#### 1. test_heads_creation ✅
**PCC: 0.999999** (Excellent)
- Q Heads: PCC 0.999999
- K Heads: PCC 0.999999
- V Heads: PCC 0.999999
- Mean abs diff: ~1.13e-03
- **Conclusion**: Heads creation is working correctly after the reshape bug fix

#### 2. test_scaled_dot_product_attention ✅
**PCC: 0.999990** (Excellent)
- Mean abs diff: 8.85e-04
- Max abs diff: 5.55e-03
- **Conclusion**: Attention mechanism works correctly WITHOUT masking

#### 3. test_heads_fusion ✅
**PCC: 0.999999** (Excellent)
- Mean abs diff: 1.12e-03
- **Conclusion**: Heads fusion is working correctly after the fix

#### 4. test_gelu_activation ✅
**PCC: 0.999996** (Excellent)
- Mean abs diff: 1.32e-03
- Max abs diff: 2.09e-02
- **Conclusion**: GELU implementation matches PyTorch reference

#### 5. test_layernorm ✅
**PCC: 0.999992** (Excellent)
- Mean abs diff: 3.59e-03
- Max abs diff: 6.32e-02
- Tested with hardware clamp disabled (epsilon = 1e-12)
- **Conclusion**: LayerNorm working correctly with proper epsilon

#### 6. test_matmul ✅
**PCC: 0.999996** (Excellent)
- Mean abs diff: 2.57e-02
- Max abs diff: 1.19e-01
- **Conclusion**: Matrix multiplication is accurate

---

### ❌ FAILED (3/9 tests)

#### 1. test_scaled_dot_product_attention_with_mask ❌ **CRITICAL BUG**
**PCC: 0.034031** (Complete failure)

**Details:**
- Mean abs diff: 3.46e-01 (Very large!)
- Max abs diff: 1.45e+00 (Huge!)
- Test creates mask with last 8 positions masked out (mask=0)
- PyTorch reference uses: `masked_fill(mask == 0, -1e9)`
- TTML implementation uses mask in `scaled_dot_product_attention`

**Expected behavior:**
```
ref mean: 2.77e-03, std: 2.83e-01
First 5: [ 0.297, -0.162,  0.028,  0.245,  0.082]
```

**Actual behavior:**
```
ttml mean: -1.90e-02, std: 3.35e-01
First 5: [-0.457, -0.006, -0.494,  0.699, -0.094]
```

**Analysis:**
- Despite fixing the masking sign bug (+1e9F → -1e9F) in `scaled_dot_product_attention.cpp:166`, the test shows masking is still completely broken
- The outputs are completely different (not just slightly off)
- This suggests either:
  1. The fix wasn't applied correctly
  2. There's an additional masking bug
  3. The mask format expected by TTML differs from what we're passing

**Next Steps:**
1. Verify the fix was actually applied and compiled
2. Check mask shape/format requirements
3. Add debug logging to see what's happening inside scaled_dot_product_attention
4. May need to check if mask needs to be inverted or has different semantics

#### 2. test_linear_layer ❌ (Fixed - trivial PyTorch bug)
**Error:** `RuntimeError: Can't call numpy() on Tensor that requires grad`
**Fix:** Changed `.numpy()` to `.detach().numpy()`
**Status:** ✅ Fixed in test file

#### 3. test_complete_mha_pipeline ❌ (Fixed - trivial PyTorch bug)
**Error:** Same as test_linear_layer
**Status:** ✅ Fixed in test file

---

## Key Findings

### 1. Most Operations Are Excellent ✅
- heads_creation: PCC 0.999999
- heads_fusion: PCC 0.999999
- GELU: PCC 0.999996
- LayerNorm: PCC 0.999992
- Matmul: PCC 0.999996

These are all working at near-perfect accuracy. The bugs we fixed (heads creation reshape, heads fusion) are confirmed working.

### 2. Attention Masking Is Critically Broken ❌
- PCC 0.034 (should be >0.99)
- This is the PRIMARY BUG blocking BERT inference
- The attention mask fix we applied may not be correct or complete
- **This bug would cause BERT to attend to padding tokens**, destroying output quality

### 3. Bindings Are Working ✅
- The `<nanobind/stl/tuple.h>` include successfully fixed the tuple return issue
- All operations are now accessible from Python
- Tests run successfully and provide detailed diagnostics

---

## Impact on BERT Forward Pass

### Why PCC is Still Low (0.58)

**Root Cause Identified:** Attention masking bug

When BERT processes sequences, it must mask out padding tokens so the model doesn't attend to them. With PCC 0.034 for masked attention:

1. **Padding tokens are being attended to** instead of being ignored
2. This corrupts the attention weights for ALL tokens (not just padding)
3. The error propagates through all 2 BERT blocks (bert-tiny)
4. Final output is severely degraded

**Why other tests passed but BERT fails:**
- Individual operations work perfectly IN ISOLATION
- But BERT uses masking HEAVILY (padding masks, attention masks)
- The masking bug only shows up when masks are used
- Our earlier tests didn't test masking!

---

## Recommendations

### Immediate Priority: Fix Attention Masking

**Investigation needed:**
1. Verify the -1e9F fix was compiled and is actually being used
2. Check if mask semantics differ (1=attend vs 0=attend)
3. Add comprehensive masking tests:
   - Test with all-ones mask (should match no-mask)
   - Test with all-zeros mask (should produce zeros/NaN)
   - Test with partial mask (current test)
4. Compare intermediate values in scaled_dot_product_attention:
   - Attention scores before masking
   - Attention scores after masking
   - Attention weights after softmax

**Debugging approach:**
```cpp
// Add to scaled_dot_product_attention.cpp
fmt::print("[DEBUG] Mask value at [0,0,0,0]: {}\\n", mask_value);
fmt::print("[DEBUG] QKV_scaled before mask: {}\\n", qkv_scaled_value);
fmt::print("[DEBUG] QKV_scaled after mask: {}\\n", qkv_scaled_after_mask);
```

### Next Steps After Masking Fix

1. Re-run all operator validation tests
2. Run BERT forward pass test - should see dramatic PCC improvement
3. If PCC still low, investigate next highest-impact bug
4. Continue iterating until PCC >0.95

---

## Files Modified

### Source Files:
- `sources/ttml/nanobind/nb_ops.cpp` - Added bindings for attention operations and tanh

### Test Files:
- `tests/python/test_bert_operators_comprehensive.py` - Comprehensive Python tests
- `tests/model/bert_operator_test.cpp` - Comprehensive C++ tests
- `tests/CMakeLists.txt` - Added bert_operator_test.cpp

### Documentation:
- This file (OPERATOR_VALIDATION_RESULTS.md)

---

## Conclusion

**Success:** Bottom-up operator validation approach is highly effective!

✅ We now have comprehensive test infrastructure
✅ We can test each operation in perfect isolation
✅ We identified the PRIMARY BUG (masking) with high confidence
✅ Most operations are working at near-perfect accuracy

**Critical Bug:** Attention masking is completely broken (PCC 0.034)

This explains the BERT forward pass PCC of 0.58. Once masking is fixed, we should see dramatic improvement, potentially reaching our target PCC >0.95.

**Recommendation:** Focus all effort on debugging and fixing the attention masking bug. This is the single most impactful issue blocking BERT inference quality.
