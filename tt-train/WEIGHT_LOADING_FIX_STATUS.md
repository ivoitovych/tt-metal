# Weight Loading Fix - Status Report

**Date**: November 14, 2025
**Branch**: `ivoitovych/bert-model-for-ttml-task-heads-v2`

## Executive Summary

**MAJOR PROGRESS**: Fixed I64 dtype error and confirmed weights load correctly. However, embedding output PCC degradation persists (0.975456), indicating an issue in the embedding operation itself, not weight storage.

---

## Issues Fixed ✅

### 1. I64 Dtype Error in Safetensors Loading
**Problem**: Loading safetensors failed with `RuntimeError: Unsupported dtype: I64` for `bert.embeddings.position_ids` tensor.

**Root Cause**: The dtype check happened BEFORE identifying which tensor was being loaded. `position_ids` is a metadata tensor [0,1,2,...,511] (int64), not a learned parameter.

**Fix** (`bert.cpp:503-505`):
```cpp
// Skip position_ids - it's a metadata tensor (not a learned parameter)
// We generate positions dynamically, so we don't need to load this
if (info.name == "bert.embeddings.position_ids" || info.name == "embeddings.position_ids") {
    return true;  // Skip this tensor
}
```

**Status**: ✅ FIXED - Weights now load successfully

**Files Modified**:
- `/workspace/tt-metal/tt-train/sources/ttml/models/bert.cpp` (lines 503-505)

**Rebuilt Components**:
- `ninja ttml_tests` - C++ library
- `ninja _ttml` - Python bindings

---

### 2. Test API Error
**Problem**: Test used `model.named_parameters()` which doesn't exist in TTML API.

**Fix**: Use `model.parameters()` which returns `ttml.NamedParameters` dict-like object.

**Example**:
```python
# Wrong:
params = model.named_parameters()
for name, param in params.items():
    ...

# Correct:
params = model.parameters()
param = params["bert/token_embeddings/weight"]
```

**Status**: ✅ FIXED

**Files Modified**:
- `/workspace/tt-metal/tt-train/tests/python/test_embedding_weight_loading.py`

---

## Verified Working ✅

### Weight Storage Accuracy
**Test**: Direct comparison of HuggingFace weights vs TTML loaded weights

**Results**:
```
Max absolute difference: 1.95e-03
Mean absolute difference: 6.45e-05
```

**Analysis**: Differences are consistent with **bfloat16 quantization** (expected behavior). Example:
- HF (float32): `-0.411438`
- TTML (bfloat16): `-0.41210938`
- Diff: `0.000671` (0.16%)

**Verification**:
- ✅ Weights are NOT transposed
- ✅ Rows are contiguous
- ✅ All vocabulary entries show uniform precision loss
- ✅ First and last dimensions of embedding vectors are correct

**Conclusion**: **Weights load correctly** with expected precision loss.

---

## Remaining Issues ❌

### Embedding Output PCC Degradation
**Problem**: Despite correct weight storage, embedding output shows severe PCC degradation.

**Observed**:
```
Word embeddings PCC: 0.975456
Mean abs diff: 2.880594e-02
Max abs diff: 5.596775e-01
```

**Critical Analysis**:
- Weight error: ~0.002 (max diff)
- Output error: ~0.56 (max diff)
- **Error amplification: 800x** ❌

This CANNOT be explained by bfloat16 quantization alone!

**Hypothesis**: The embedding operation has a bug when using loaded weights that causes error amplification. Possible causes:
1. Incorrect tensor layout assumptions in `ttnn::embedding`
2. Padding issue (vocab 30522 → 30528) affecting index lookups
3. Memory stride/layout mismatch between weight storage and embedding operation
4. Batch processing issue (similar to the uint32 dtype bug from previous branch)

**Evidence from Previous Branch**:
- With **random weights** + uint32 inputs: PCC = 1.0 ✅
- With **loaded weights** + uint32 inputs: PCC = 0.975456 ❌

This suggests the bug is triggered by **specific weight values or patterns** in pre-trained weights, not random initialization.

---

## Test Results

### test_embedding_weight_loading.py
**Status**: ❌ FAILS (test threshold too strict for bfloat16)

**Output**:
```
Overall match (rtol=1e-5, atol=1e-6): False
Max diff: 1.95e-03
```

**Note**: Test should PASS if threshold adjusted for bfloat16. The weights ARE correct.

### test_granular_embedding_debug.py
**Status**: ❌ FAILS with PCC 0.975456

**Output**:
```
❌ 1. Word (token) embeddings: PCC: 0.975456
✅ 3. Token type embeddings: PCC: 0.999999
```

**Key Insight**: Token type embeddings work perfectly, proving:
- Loading mechanism is fundamentally correct
- Bug is specific to large vocabulary embeddings (30522 vs 2 tokens)

---

## Next Investigation Steps

### Priority 1: Understand Error Amplification
**Goal**: Identify why weight error (0.002) becomes output error (0.56).

**Approach**:
1. Create isolated test: `ttnn::embedding(input_ids, loaded_weights)` vs `hf_embedding(input_ids, loaded_weights)`
2. Test with same input IDs and same weights
3. Check if error is in embedding lookup or in subsequent operations

### Priority 2: Inspect Embedding Operation Implementation
**Files to Review**:
- `tt-train/sources/ttml/ops/embedding_op.cpp`
- TTNN embedding kernel implementation

**Questions**:
- How does `ttnn::embedding` handle weight tensor layout?
- Are there assumptions about weight memory stride?
- How is padding (30522 → 30528) handled?

### Priority 3: Compare with Token Type Embeddings
**Goal**: Understand why token type embeddings work (PCC 0.999999) but word embeddings fail.

**Comparison**:
| Feature | Word Embeddings | Token Type |
|---------|----------------|------------|
| Vocab Size | 30522 | 2 |
| Padded Size | 30528 | 32 |
| PCC | 0.975456 ❌ | 0.999999 ✅ |
| Max Weight Diff | 0.002 | ~0.002 |
| Max Output Diff | 0.56 ❌ | ~0.01 ✅ |

**Hypothesis**: Large vocabulary size exposes indexing or memory layout bugs.

---

## Files Modified

### Source Code
1. `/workspace/tt-metal/tt-train/sources/ttml/models/bert.cpp`
   - Lines 503-505: Skip `position_ids` tensor

### Tests
1. `/workspace/tt-metal/tt-train/tests/python/test_embedding_weight_loading.py`
   - Fixed API calls to use `parameters()` instead of `named_parameters()`

2. `/workspace/tt-metal/tt-train/tests/python/test_weight_layout_debug.py`
   - **NEW**: Debug test to verify weight layout and check for transpose issues

### Documentation
1. `/workspace/tt-metal/tt-train/WEIGHT_LOADING_BUG_ANALYSIS.md`
   - Updated with historical context from previous branch

2. `/workspace/tt-metal/tt-train/BERT_BATCH_PROCESSING_INVESTIGATION_REPORT.md`
   - Added critical update about weight loading bug discovery

---

## Historical Context

### Previous Branch: `ivoitovych/bert-model-for-ttml-completeness-implementation`

**Commit `3f7458e6e6`**: "ROOT CAUSE FOUND - Token IDs must be uint32, not float32"
- Discovered ttnn::embedding does NOT handle float32 inputs correctly for batch processing
- Fixed by using `np.uint32` for input_ids
- Result with random weights: PCC = 1.0 ✅

**Current Branch Builds On This**:
- Inputs are correctly uint32 ✅
- But with loaded weights: PCC = 0.975456 ❌
- This reveals a SECOND, separate bug in weight handling

---

## Conclusion

We've made significant progress:
1. ✅ Fixed I64 dtype error blocking weight loading
2. ✅ Confirmed weights load and store correctly (bfloat16 precision)
3. ✅ Verified weight layout is correct (not transposed)

**However**, the core PCC degradation issue remains. The 800x error amplification from weights (0.002) to output (0.56) indicates a bug in the **embedding operation itself**, not in weight storage.

**Next Step**: Create isolated test to directly compare `ttnn::embedding` vs HuggingFace embedding with the same input IDs and weights, to pinpoint where the error amplification occurs.

---

**Status**: 🔍 **INVESTIGATION ONGOING**
**Blocker**: Embedding operation produces 800x error amplification with loaded weights
