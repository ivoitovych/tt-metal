# BERT Batch Processing Bug Investigation

## Investigation Approach

Following recommendation to:
1. Focus on comprehensive C++ test suite first
2. Validate C++ implementation is correct
3. Then investigate Python binding issues separately

## Findings

### 1. C++ Implementation Status

**All C++ tests PASS**: 18/18 BERT tests passing (100%)

```
[==========] Running 18 tests from 4 test suites.
[  PASSED  ] 18 tests.
```

**Tests include:**
- `BertPolymorphismTest.*` (8 tests) - All passed
- `BertWeightLoadingTest.*` (4 tests) - All passed
- `BertSeqClsTest.*` (5 tests) - All passed
  - Including `BatchSizeIndependence` - **FALSE POSITIVE** (see below)
  - Including `MultipleLabelsTest` (2, 3, 5 labels) - All passed!
- `LayerNormEpsilonTest.BertUsesCorrectEpsilon` - Passed

**Key Discovery**: Multi-label classification (3 and 5 labels) **WORKS PERFECTLY in C++**!
This contradicts Python test results where 3+ labels fail.

### 2. False Positive Test Analysis

**Test**: `BertSeqClsTest.BatchSizeIndependence` (bert_seq_cls_test.cpp:329)

**What it claims to test:**
"Test that different batch sizes produce consistent per-sample results"

**What it actually tests:**
- Runs sample with input=7.0 individually (batch_size=1)
- Runs batch with sample[0]=7.0, sample[1]=3.0 (batch_size=2)
- **ONLY** compares individual run vs batch[0]
- **NEVER** checks if batch[0] != batch[1]

**Why this is a false positive:**
The test would pass even if ALL batch samples produce identical outputs,
as long as batch[0] matches the individual run of the same input.

**Fix applied:**
Added critical check to verify batch[0] != batch[1]:
```cpp
// CRITICAL ADDITION: Check if batch[0] and batch[1] are actually DIFFERENT
std::vector<float> logits_batch_second(num_labels_aligned);
for (size_t i = 0; i < num_labels_aligned; ++i) {
    logits_batch_second[i] = logits_batch_data[num_labels_aligned + i];
}

float max_diff = 0.0F;
for (size_t i = 0; i < num_labels_aligned; ++i) {
    float diff = std::abs(logits_batch_first[i] - logits_batch_second[i]);
    max_diff = std::max(max_diff, diff);
}

if (max_diff < 0.001F) {
    std::cout << "❌ BATCH PROCESSING BUG: Different inputs produce IDENTICAL outputs!" << std::endl;
} else {
    std::cout << "✓ CORRECT: Different inputs produce different outputs (diff=" << max_diff << ")" << std::endl;
}
```

**Status**: Test modified but not yet re-run due to build system issues.

### 3. Python Binding Investigation

**Previous Binding Bug**: Commit 38f05bb43c fixed critical bug with non-contiguous numpy arrays.
The fix handles transpose operations that create non-contiguous arrays.

**Hypothesis tested**: Maybe batch processing creates non-contiguous arrays?

**Test**: Created `test_batch_contiguity.py` to check array layout.

**Result**: Batch arrays **ARE C-contiguous** ✓

```
Original array: (2, 32) - C-contiguous: True
After reshape to (2, 1, 1, 32) - C-contiguous: True
```

**Conclusion**: The batch processing bug is NOT related to non-contiguous arrays.
The existing binding fix from commit 38f05bb43c should handle batch arrays correctly.

### 4. Python Test Results

**Python batch processing**: FAILS
- batch_size=1: Works perfectly (PCC=1.0)
- batch_size=2,4: All samples produce IDENTICAL outputs

**Python multi-label**: FAILS
- 2 labels: Works (PCC ≥ 0.98)
- 3+ labels: Fails (PCC ~0.93)

**Python attention masks**: FAILS
- Partial masks: Works (PCC ≥ 0.98)
- All-ones masks: Fails (PCC ~0.80-0.93)

## Current Status

### Evidence Summary

|  Feature | C++ Status | Python Status | Likely Root Cause |
|---|---|---|---|
| **Batch processing** | Unknown (test is false positive, need proper validation) | BROKEN | Either C++ implementation OR Python binding |
| **Multi-label (3+)** | **WORKS** ✅ | BROKEN | Python binding issue |
| **Attention masks** | Unknown | BROKEN | Unknown |

### Critical Questions

1. **Does C++ batch processing actually work?**
   - Current test is a false positive
   - Need to rebuild and run modified test
   - Build system currently broken (xtl patch issues)

2. **Why does multi-label work in C++ but fail in Python?**
   - This strongly suggests a Python binding issue
   - Not a contiguity issue (arrays are contiguous)
   - Might be related to how multi-dimensional outputs are handled

3. **Is attention mask issue also binding-related?**
   - Need to test in pure C++ first
   - Could be implementation bug or binding bug

## Next Steps

### Immediate (Blocked by build issues):
1. Fix build system (xtl patch problem)
2. Rebuild tests with modified BatchSizeIndependence check
3. Run test to determine if C++ batch processing works

### If C++ batch processing works:
Then ALL bugs are Python binding issues:
- Investigate how batched tensors are passed through bindings
- Check if there's special handling needed for batch dimension
- Look at how `from_numpy` processes multi-dimensional arrays

### If C++ batch processing also fails:
Then it's a C++ implementation bug:
- Investigate embedding layer batch handling
- Check attention mechanism batch dimension handling
- Examine pooler slice operation for batch extraction

## Files Modified

1. **tt-train/tests/model/bert_seq_cls_test.cpp**
   - Added critical validation to BatchSizeIndependence test
   - Now checks if batch[0] != batch[1] (what it should have been checking all along)

2. **tt-train/test_batch_contiguity.py** (new)
   - Tests if batch arrays are contiguous
   - Result: Arrays ARE contiguous

3. **tt-train/BATCH_PROCESSING_INVESTIGATION.md** (this file)
   - Documents investigation findings
   - Tracks next steps

## References

- Previous binding bug fix: commit 38f05bb43c
- BERT implementation status: tt-train/BERT_IMPLEMENTATION_STATUS.md
- Batch bug reproduction: tt-train/debug_batch_processing.py
- Shape inspection: tt-train/debug_tensor_shapes.py
