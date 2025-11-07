# TTNN Embedding Batch Processing Bug Report

**Branch**: `ivoitovych/ttnn-embedding-batch-bug-reproduction`
**Base Commit**: `eca8b5a8f1`

> **NOTE**: This branch contains bug report documentation and direct `ttnn::embedding()`
> tests for investigating potential batch processing issues.
>
> **Test Status**: All tests on this branch currently PASS with clean 2D tensors.
>
> The bug was observed in a production scenario where tensors were reshaped from
> 4D to 2D before being passed to `ttnn::embedding()`. This may suggest the issue
> is specific to certain tensor layouts or reshape operations.

---

## Executive Summary

Investigation of `ttnn::embedding()` batch processing revealed potential issues where
batch processing with `batch_size > 1` may produce identical outputs for different
inputs in the same batch under certain conditions.

**Status**: Under investigation - tests with clean tensors pass, but issue observed in production usage
**Severity**: HIGH - Could block multi-sample batch processing for models using embeddings
**TTNN Version**: 6.0.0 (as of commit eca8b5a8f1)

---

## Problem Description

### Observed Behavior (in production usage)
When processing a batch of 2 or more samples with different input token IDs:
- **EXPECTED**: Different token IDs produce different embeddings
- **ACTUAL (observed)**: All samples in the batch received identical embeddings, regardless of input differences

### Example of Observed Issue
```
Input batch (2 samples, sequence length 32):
  Sample 0: All tokens = 7
  Sample 1: All tokens = 99

Output (observed in production):
  Sample 0 embedding: [0.45, 1.23, -0.89, ...]
  Sample 1 embedding: [0.45, 1.23, -0.89, ...]  // IDENTICAL!
```

---

## Root Cause Analysis

### Investigation Summary

Investigation traced the issue to `ttnn::embedding()` not correctly processing batched
inputs in certain scenarios:

1. **Initial Discovery**: Batch processing in production showed identical outputs for different inputs

2. **Isolation Testing**: Created targeted tests to isolate the embedding operation
   - File: `tt-train/tests/core/ttnn_embedding_batch_bug_test.cpp`
   - Tests call `ttnn::embedding()` directly with batched inputs

3. **Observation**: Debug logging showed:
   ```cpp
   // Input tensor with different token IDs per sample
   Input shape: [2, 1, 1, 32]
   Sample 0 data: [7, 7, 7, ...]      // All token ID 7
   Sample 1 data: [99, 99, 99, ...]   // All token ID 99

   // After reshape to 2D: [2, 32]
   Data still correct: different values per sample

   // After ttnn::embedding() in production scenario:
   Output: Identical embeddings for both samples (BUG)
   ```

4. **Key Finding**: The issue appears specific to certain tensor operations or layouts,
   as clean 2D tensor tests pass correctly.

### Technical Details

**Function**: `ttnn::embedding(input_tensor, weight_tensor, pad_token, layout)`

**Problem Scenario (observed in production)**:
```cpp
// Input: 4D tensor [batch_size, 1, 1, seq_len]
auto input_4d = ...; // Shape: [2, 1, 1, 32], data: [7,7,...,99,99,...]

// Reshape to 2D for ttnn::embedding
auto input_2d = ttnn::reshape(input_4d, ttnn::Shape({2, 32}));
// Data still correct: [7,7,...,99,99,...]

// Call ttnn::embedding
auto embeddings = ttnn::embedding(input_2d, weights, nullopt, TILE_LAYOUT);

// BUG: Output embeddings are IDENTICAL for both samples
// Sample 0 and Sample 1 have same embedding values
```

**Hypothesis**: The bug may be related to how `ttnn::embedding()` handles tensors that were reshaped from 4D to 2D, or some internal state/layout issue with batched inputs.

---

## Evidence Files

### Test File on This Branch

**tt-train/tests/core/ttnn_embedding_batch_bug_test.cpp**
- Direct testing of `ttnn::embedding()` function
- 3 test scenarios:
  1. `SingleSample_Baseline` - Verifies batch_size=1 works
  2. `BatchWithDifferentTokens_BUG_EVIDENCE` - Tests 2-sample batch
  3. `LargerBatch_ExtendedEvidence` - Tests 4-sample batch
- **Result on this branch**: All tests PASS with clean 2D tensors

This suggests the observed production issue may be specific to:
- Tensors reshaped from 4D to 2D
- Specific tensor layouts or memory arrangements
- Certain sequence of operations before `ttnn::embedding()`

---

## Potential Workaround (if bug confirmed)

If the issue is confirmed with specific tensor layouts, a potential workaround could be:

### Strategy
Process each batch sample individually, then concatenate results

### Pseudocode
```cpp
if (batch_size == 1) {
    // Single sample - direct path
    embeddings = ttnn::embedding(input_2d, weight_tensor, ...);
} else {
    // Process each sample individually
    std::vector<Tensor> batch_embeddings;
    for (uint32_t i = 0; i < batch_size; ++i) {
        // Extract single sample
        auto sample = ttnn::slice(input, start_indices, end_indices, stride);

        // Embed single sample
        auto sample_embedding = ttnn::embedding(sample, weight_tensor, ...);

        batch_embeddings.push_back(sample_embedding);
    }

    // Concatenate all embeddings
    embeddings = ttnn::concat(batch_embeddings, 0);
}
```

### Performance Impact (if workaround needed)
- Additional overhead for slice and concat operations
- Trade-off: Correctness vs performance
- Should be removed once root cause is fixed in TTNN

---

## Test Results on This Branch

### All Tests Pass
```
TtnnEmbeddingBatchBugTest.SingleSample_Baseline:
  ✓ PASSED - Single sample works correctly

TtnnEmbeddingBatchBugTest.BatchWithDifferentTokens_BUG_EVIDENCE:
  Sample 0 (token 7):  [4.47, 4.50, 4.50, ...]
  Sample 1 (token 99): [63.25, 63.25, 63.50, ...]  // DIFFERENT - CORRECT!
  ✓ PASSED

TtnnEmbeddingBatchBugTest.LargerBatch_ExtendedEvidence:
  All 4 samples produce different embeddings
  ✓ PASSED
```

**Conclusion**: Clean 2D tensor tests pass. Production issue may require
specific tensor layout or reshape sequence to reproduce.

---

## Reproduction Steps

### Option 1: Run Direct TTNN Test (on this branch)
```bash
cd /workspace/tt-metal/tt-train
cmake --build build --config Debug
./build/tests/ttml_tests --gtest_filter="TtnnEmbeddingBatchBugTest.*"
```

**Result**: All tests currently PASS (with clean 2D tensors)
**Note**: Tests use clean 2D tensors. Production issue may require specific
tensor layouts or reshape sequences to reproduce.

### Option 2: Minimal Reproduction (Pseudocode)
```cpp
// Create weight matrix [100, 64]
auto weights = create_random_weights(100, 64);

// Create batched input [2, 32] with DIFFERENT token IDs
std::vector<uint32_t> input = {
    7, 7, 7, ...,   // Sample 0: all token ID 7
    99, 99, 99, ... // Sample 1: all token ID 99
};
auto input_tensor = from_vector(input, Shape({2, 32}));

// Call ttnn::embedding
auto embeddings = ttnn::embedding(input_tensor, weights, nullopt, TILE_LAYOUT);

// Extract embeddings
auto emb0 = embeddings[0][0];  // First token of sample 0
auto emb1 = embeddings[1][0];  // First token of sample 1

// Check if different
assert(emb0 != emb1);  // May FAIL if bug exists
```

---

## Impact Assessment

### Affected Components (if bug is confirmed)
1. **Any model using ttnn::embedding with batch_size > 1**: Potentially affected
2. **Training**: Could impact mini-batch training
3. **Inference**: Could affect multi-sample batch processing
4. **Production deployments**: May require workarounds

### Workaround Considerations (if needed)
1. **Performance**: Slice/concat approach slower than native batch processing
2. **Memory**: Higher peak memory usage during operations
3. **Complexity**: Adds code complexity
4. **Temporary**: Should be removed once root cause is fixed

---

## Recommended Actions

### For TTNN Team
1. **Investigate `ttnn::embedding()` batch handling**
   - Focus on how it processes 2D tensors with `batch_size > 1`
   - Check if issue is specific to tensors reshaped from 4D
   - Verify tensor layout/storage handling for batched inputs

2. **Add unit tests** for `ttnn::embedding` with:
   - `batch_size = 1` (baseline)
   - `batch_size = 2` with different token IDs
   - `batch_size = 4` with different token IDs
   - Tensors created via reshape from 4D

3. **Fix and regression test**
   - Add comprehensive batch processing tests
   - Verify fix with various tensor layouts and shapes

### For Users (if workaround needed)
1. **Implement slice-and-concat workaround** if batch processing fails
2. **Monitor performance** impact of workaround
3. **Test thoroughly** to ensure correctness
4. **Remove workaround** once TTNN fix is confirmed

---

## Test Files

**Available on this branch**:
- `tt-train/tests/core/ttnn_embedding_batch_bug_test.cpp` - Direct ttnn::embedding tests
- `tt-train/TTNN_EMBEDDING_BUG_REPORT.md` - This documentation

---

**Last Updated**: 2025-11-07
**Branch**: `ivoitovych/ttnn-embedding-batch-bug-reproduction`
**Priority**: HIGH
**Status**: Under Investigation
