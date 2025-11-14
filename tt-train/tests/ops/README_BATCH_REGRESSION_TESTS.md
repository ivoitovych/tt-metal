# Batch Processing Regression Tests

This directory contains regression tests for BERT-independent batch processing functionality.

## Tests Created

### 1. embedding_batch_regression_test.cpp
**Purpose:** Verify embedding operations work correctly with batch_size > 1
**Location:** `ops/embedding_op.hpp`

**Description:**
Tests that the `ops::embedding_op()` operation maintains high accuracy (PCC > 0.999) across different batch sizes. These tests serve as regression tests to ensure batch processing continues to work correctly.

**Test Cases:**
- `EmbeddingBatchSize1_Baseline`: Baseline test with batch_size=1 (PCC > 0.999)
- `EmbeddingBatchSize2`: Tests batch_size=2 (PCC > 0.999)
- `VerifyExpectedOutputIsCorrect`: Verification test to ensure test logic is correct

**Expected Results:**
```cpp
batch_size=1: PCC > 0.999 ✓
batch_size=2: PCC > 0.999 ✓
```

**How to Run:**
```bash
cd build
./tests/ttml_tests --gtest_filter="EmbeddingBatchRegressionTest.*"
```

---

### 2. multi_head_attention_batch_regression_test.cpp
**Purpose:** Verify multi-head attention operations work correctly with batch_size > 1
**Location:** `ops/multi_head_utils.cpp`, `modules/multi_head_attention.hpp`

**Description:**
Tests that the multi-head attention operations (specifically `ops::heads_creation()` and `modules::MultiHeadAttention()`) correctly handle batch dimensions for all batch sizes.

**Test Cases:**
- `MultiHeadAttentionBatchSize1_Baseline`: Baseline test with batch_size=1
- `MultiHeadAttentionBatchSize2`: Tests batch_size=2
- `HeadsCreationBatchSize2_DirectTest`: Directly tests `ops::heads_creation()` with batch_size=2
- `HeadsCreationBatchSize1_Baseline`: Baseline test for heads_creation with batch_size=1
- `HeadsCreationBatchSize4_ScalingTest`: Tests with larger batch to verify scaling

**Expected Results:**
```cpp
// All tests should PASS
// All Q, K, V tensors should have correct batch dimensions:
batch_size=1: Q=[1,2,32,64], K=[1,2,32,64], V=[1,2,32,64] ✓
batch_size=2: Q=[2,2,32,64], K=[2,2,32,64], V=[2,2,32,64] ✓
batch_size=4: Q=[4,2,32,64], K=[4,2,32,64], V=[4,2,32,64] ✓
```

**How to Run:**
```bash
cd build
./tests/ttml_tests --gtest_filter="MultiHeadAttentionBatchRegressionTest.*"
```

---

## Integration

These tests have been added to `tests/CMakeLists.txt`:
```cmake
set(SOURCES
    ...
    ops/gelu_op_test.cpp
    ops/embedding_batch_regression_test.cpp               # Regression test
    ops/multi_head_attention_batch_regression_test.cpp   # Regression test
)
```

## Purpose

These tests are BERT-independent and test core operations directly:
- No BERT model creation
- No model weight loading
- Direct testing of `ops::embedding_op()` and `modules::MultiHeadAttention()`
- Minimal dependencies - only the specific ops being tested

This makes them ideal for:
1. **Regression testing** - Verify batch processing continues to work correctly
2. **CI/CD** - Fast, focused tests without BERT overhead
3. **Performance validation** - Ensure batch processing maintains high accuracy

## Test Philosophy

Each test includes:
1. **Baseline test** (batch_size=1) - Demonstrates expected behavior
2. **Batch processing tests** (batch_size>1) - Verifies batch processing works correctly
3. **Verification tests** - Validates test logic is correct
4. **Detailed output** - Prints PCC, mean_diff, max_diff, shapes for debugging

## Expected Test Results

**All tests should PASS:**
```
[ RUN      ] EmbeddingBatchRegressionTest.EmbeddingBatchSize1_Baseline
Batch size 1 results:
  PCC: 0.999985
  Mean abs diff: 0.002
  Max abs diff: 0.006
[       OK ] ✓

[ RUN      ] EmbeddingBatchRegressionTest.EmbeddingBatchSize2
Batch size 2 results:
  PCC: 0.999989
  Mean abs diff: 0.004
  Max abs diff: 0.012
[       OK ] ✓

[ RUN      ] MultiHeadAttentionBatchRegressionTest.MultiHeadAttentionBatchSize2
Batch size 2: Success - Output shape: Shape([2, 1, 32, 128])
[       OK ] ✓
```

## Related Files

- `tests/python/test_bert_batch_processing.py` - Python regression tests for batch processing
- `BERT_BATCH_SIZE_BUG_INVESTIGATION.md` - Historical investigation report
- `tests/python/test_bert_isolated_layer_validation.py` - Python validation tests
- `sources/ttml/ops/embedding_op.cpp` - Embedding operation implementation
- `sources/ttml/ops/multi_head_utils.cpp` - Multi-head attention utilities

## Notes

These tests were created after investigating batch processing behavior in the BERT model.
The investigation confirmed that the core operations work correctly with batch_size > 1,
achieving high accuracy (PCC > 0.999) for all tested batch sizes.
