# Weight Loading Bug Analysis - BERT Word Embeddings

**Date**: November 14, 2025
**Status**: 🔍 **ROOT CAUSE IDENTIFIED - Weight Loading Corruption**

## Executive Summary

Investigation reveals **TWO SEPARATE BUGS** affecting BERT embedding accuracy:

1. ✅ **Input Dtype Bug** (FIXED): ttnn::embedding does NOT handle float32 inputs correctly for batch processing
2. ❌ **Weight Loading Bug** (ACTIVE): Word embedding weights are corrupted during loading from safetensors

## Historical Context from Previous Branch

### Previous Investigation (`ivoitovych/bert-model-for-ttml-completeness-implementation`)

The first implementation branch encountered similar PCC degradation issues and went through this evolution:

#### Phase 1: Initial Workaround (Commit `10a9d642d2`)
- **Belief**: ttnn::embedding() has a batch processing bug
- **Solution**: Implemented workaround to process each batch sample individually
- **Code**: Slice → Embed → Concat for batch_size > 1

#### Phase 2: Root Cause Discovery (Commit `3f7458e6e6`)
- **Breakthrough**: Bug was NOT in ttnn::embedding, but in **TEST CODE using wrong dtype**
- **Finding**: ttnn::embedding does NOT correctly handle **float32** inputs for batched processing
- **Fix**: Changed input_ids from float32 to **uint32**
- **Result**: All tests passed with PCC = 1.0 (random weights)
- **Action**: Removed workaround, restored clean embedding_op.cpp

#### Key Quote from Commit `3f7458e6e6`:
```
ROOT CAUSE:
Token IDs were being passed as float32 instead of uint32/int32.
ttnn::embedding does NOT correctly handle float32 inputs for batched
processing - it returns identical outputs for all batch samples.

Tests with uint32 (correct):
- BertEmbeddingPipelineTest: PASS (-1.52 vs -0.94)
- BertBatchIsolationTest: PASS (-1.52 vs -0.94)
- All C++ tests: ALL PASS with uint32_t

Tests with float32 (WRONG):
- BertBatchIsolationTest (before fix): FAIL (identical: -1.52 vs -1.52)
```

## Current Investigation - The Missing Piece

### Test Configuration
- **Branch**: `ivoitovych/bert-model-for-ttml-task-heads-v2`
- **Model**: prajjwal1/bert-tiny (pre-trained weights)
- **Input dtype**: `np.uint32` ✅ (correctly applied)
- **Batch size**: 2
- **Sequence length**: 32

### Critical Finding: Weight Loading Corruption

**Test**: `test_granular_embedding_debug.py`

**Results**:
```
❌ Word (token) embeddings:        PCC: 0.975456 (vocab_size = 30522)
✅ Token type embeddings:          PCC: 0.999999 (vocab_size = 2)
```

**Analysis**:
1. Inputs are correctly uint32 (same as previous branch fix)
2. With **random weights** (previous branch): PCC = 1.0 ✅
3. With **loaded weights** (current branch): PCC = 0.975456 ❌

**Conclusion**: The weights are being **corrupted during loading from safetensors**, specifically for large embedding matrices (word embeddings).

## Evidence Comparison

| Scenario | Weights | Input Dtype | Word Emb PCC | Token Type PCC |
|----------|---------|-------------|--------------|----------------|
| Previous branch | Random | uint32 | 1.0 ✅ | 1.0 ✅ |
| Previous branch | Random | float32 | <0.99 ❌ | <0.99 ❌ |
| Current branch | Safetensors | uint32 | 0.975456 ❌ | 0.999999 ✅ |

**Key Insight**: Token type embeddings work perfectly even with loaded weights, proving:
- The loading mechanism is NOT fundamentally broken
- The bug is specific to **large vocabulary embeddings** (30522 vs 2 tokens)

## Root Cause Hypothesis

### Suspected Code: `bert.cpp:509-517`

```cpp
if (info.name == "bert.embeddings.word_embeddings.weight" ||
    info.name == "embeddings.word_embeddings.weight") {
    auto param = get_parameter("bert/token_embeddings/weight");
    auto padded = pad_vocab_embeddings(
        float_vec, info.shape[0], info.shape[1], param->get_value().logical_shape()[-2]);
    param->set_value(core::from_vector(
        padded, param->get_value().logical_shape(), param->get_value().device()));
    fmt::print("  Loaded word embeddings\n");
}
```

### Suspect Function: `pad_vocab_embeddings()` (bert.cpp:486-493)

```cpp
auto pad_vocab_embeddings = [](const std::vector<float>& flat, int64_t rows, int64_t cols, int64_t target_rows) {
    if (rows >= target_rows) {
        return flat;
    }
    std::vector<float> out(static_cast<size_t>(target_rows * cols), 0.0f);
    std::copy(flat.begin(), flat.end(), out.begin());
    return out;
};
```

### Potential Issues

1. **Layout Mismatch**:
   - HuggingFace: Row-major [vocab_size, embedding_dim]
   - TTNN: May expect column-major or different layout
   - Large matrices (30522×128) more sensitive to layout errors

2. **Padding Corruption**:
   - Padding from 30522 → 30528 might corrupt data for large matrices
   - Token type (2 → 32) might work due to smaller scale

3. **Tensor Reshape Issues**:
   - `param->get_value().logical_shape()` might not match actual data layout
   - `core::from_vector()` might assume wrong memory layout

## Why Token Type Embeddings Work

**Token type embeddings succeed** (PCC = 0.999999) because:
- **Vocabulary size**: Only 2 tokens (padded to ~32)
- **Matrix size**: 2×128 = 256 elements vs 30522×128 = 3,906,816 elements
- **Smaller** padding ratio: 2→32 (16x) vs 30522→30528 (1.0002x)
- **Less sensitive** to layout/indexing errors

## Recommended Investigation Steps

### 1. Verify Weight Values Directly
Compare HuggingFace weights vs TTML loaded weights (row-by-row):
```python
hf_embeddings = hf_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
ttml_embeddings = ttml_model.get_parameter("bert/token_embeddings/weight").to_numpy()
# Compare specific rows: 0, 100, 1000, 5000, etc.
```

### 2. Check Layout and Shape
```python
print(f"HF shape: {hf_embeddings.shape}")        # Expected: [30522, 128]
print(f"TTML shape: {ttml_embeddings.shape}")    # Expected: [1, 1, 30528, 128]?
print(f"HF first row: {hf_embeddings[0, :10]}")
print(f"TTML first row: {ttml_embeddings[0, 0, 0, :10]}")  # Check indexing
```

### 3. Test pad_vocab_embeddings Directly
```cpp
// Extract first few rows before and after padding
std::vector<float> test_input = {/* first 256 floats from HF */};
auto padded = pad_vocab_embeddings(test_input, 2, 128, 32);
// Verify first 256 elements match input
```

### 4. Check core::from_vector Usage
- Verify shape parameter matches actual data dimensions
- Check if row-major vs column-major assumption is correct
- Test with small embedding matrix (100 rows) vs large (30522 rows)

## Files to Investigate

1. **`tt-train/sources/ttml/models/bert.cpp:486-517`**
   - `pad_vocab_embeddings()` lambda function
   - `load_model_from_safetensors()` word embeddings section

2. **`tt-train/sources/ttml/core/tt_tensor_utils.hpp/cpp`**
   - `core::from_vector()` implementation
   - Layout conversion logic

3. **`tt-train/sources/ttml/ops/embedding_op.cpp`**
   - Verify no assumptions about weight tensor layout

## Test Files

### Existing Tests
- `tests/python/test_granular_embedding_debug.py` - **Reproduces bug** ✅
- `tests/ops/embedding_word_vs_token_type_test.cpp` - Works with random weights ✅

### Proposed New Test
- `tests/python/test_embedding_weight_loading.py` - **Directly compare HF vs TTML weights**
  - Load weights from safetensors
  - Extract word embedding matrix from both models
  - Compare row-by-row with detailed diagnostics

## Expected Fix

Once the weight loading bug is fixed:
- Word embeddings PCC should be > 0.999 (matching token type embeddings)
- All downstream BERT components should improve automatically
- Final output PCC should reach > 0.999 threshold

## References

- Previous branch: `ivoitovych/bert-model-for-ttml-completeness-implementation`
- Key commits:
  - `3f7458e6e6` - ROOT CAUSE FOUND - Token IDs must be uint32, not float32
  - `10a9d642d2` - fix(embedding): Work around ttnn::embedding batch processing bug
  - `bc83195d74` - docs: Update status - batch processing bug RESOLVED
