# Test Failure: BertPolymorphismTest.BaseTransformerOperatorCall

**Date**: 2025-11-14
**Status**: ✅ **FIXED** (2025-11-15)
**Severity**: LOW - Test ordering issue, unrelated to softmax workaround changes

---

## Summary

The C++ test `BertPolymorphismTest.BaseTransformerOperatorCall` fails with a device initialization error **ONLY when run in the full test suite**. When run in isolation, the test **PASSES**. This indicates a **test ordering/cleanup issue** where a previous test doesn't properly clean up device state. This failure is **not related** to the softmax FP32 workaround changes.

---

## Failure Details

### Error Message
```
unknown file: Failure
C++ exception with description "open_device was called after the device was created." thrown in SetUp().
[  FAILED  ] BertPolymorphismTest.BaseTransformerOperatorCall (2 ms)
```

### Root Cause
**Test Cleanup Issue**: `SoftmaxPrecisionBug.RealBertAttentionScores` leaves the device in an open state after completion. When `BertPolymorphismTest.BaseTransformerOperatorCall` runs next, it attempts to call `open_device()` while the device is already open, causing the failure.

**Proven Minimal Sequence**:
```bash
# This 2-test sequence reproduces the failure 100% of the time
./build/tests/ttml_tests --gtest_filter="SoftmaxPrecisionBug.RealBertAttentionScores:BertPolymorphismTest.BaseTransformerOperatorCall"
```

**Evidence**:
- ✅ **PASSES** when run alone
- ❌ **FAILS** when run after `SoftmaxPrecisionBug.RealBertAttentionScores`
- ❌ **FAILS** when run in full test suite

This is a **test cleanup issue** in `SoftmaxPrecisionBug.RealBertAttentionScores`, not a functional bug in either test's actual logic.

### Test Location
- **File**: `tests/model/bert_polymorphism_test.cpp`
- **Test Suite**: `BertPolymorphismTest`
- **Test Case**: `BaseTransformerOperatorCall`

---

## Reproduction Steps

### Minimal Reproduction (2 Tests - REPRODUCES FAILURE)

```bash
# Run these TWO tests in sequence - BaseTransformerOperatorCall FAILS
./build/tests/ttml_tests --gtest_filter="SoftmaxPrecisionBug.RealBertAttentionScores:BertPolymorphismTest.BaseTransformerOperatorCall"
```

**Result**: ❌ **FAILS** (SoftmaxPrecisionBug passes, BaseTransformerOperatorCall fails with device error)

```
[  PASSED  ] SoftmaxPrecisionBug.RealBertAttentionScores (3925 ms)
[  FAILED  ] BertPolymorphismTest.BaseTransformerOperatorCall (2 ms)
            "open_device was called after the device was created."
```

### Test Passes in Isolation

```bash
# Run the test alone - IT PASSES
./build/tests/ttml_tests --gtest_filter="BertPolymorphismTest.BaseTransformerOperatorCall"
```

**Result**: ✅ **PASSES** (test runs for ~15 seconds and completes successfully)

### Full Test Suite (Shows Same Failure)

```bash
# Run all BERT tests - BaseTransformerOperatorCall FAILS
./build/tests/ttml_tests --gtest_filter="*Bert*:*BERT*"
```

**Result**: ❌ **FAILS** (52/53 tests pass, BaseTransformerOperatorCall fails with device error)

### Clean Build (Optional - If Starting Fresh)

```bash
cd /workspace/tt-metal/tt-train/
rm -rf build
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -B build \
      -GNinja
cmake --build build --config Debug --clean-first
```

---

## Test Results Context

### All BERT C++ Tests (53 total)
```
[==========] Running 53 tests from 12 test suites.
[  PASSED  ] 52 tests.
[  FAILED  ] 1 test

[  FAILED  ] BertPolymorphismTest.BaseTransformerOperatorCall
```

### Other Tests in Same Suite (All Pass)
- ✅ `BertPolymorphismTest.BertSpecificForward` (14247 ms)
- ✅ `BertPolymorphismTest.PolymorphicContainer` (295 ms)
- ✅ `BertPolymorphismTest.BackwardCompatibleOperator` (8192 ms)
- ✅ `BertPolymorphismTest.PolymorphicWithPooler` (1464 ms)
- ✅ `BertPolymorphismTest.ErrorHandlingMismatchedShapes` (1405 ms)
- ✅ `BertPolymorphismTest.GradientFlowPolymorphic` (15508 ms)
- ✅ `BertPolymorphismTest.ConfigAccess` (3381 ms)

---

## Analysis

### Why This is NOT Related to Softmax Workaround

1. **Error Type**: Device initialization error, not numerical precision issue
2. **Failure Location**: Test setup (`SetUp()`), not test execution
3. **Timing**: Fails immediately (2 ms), not during computation
4. **Other Tests**: All other 52 BERT tests pass, including:
   - Tests using softmax operations
   - Tests with attention mechanisms
   - Tests with BERT forward/backward passes

### What This Actually Is

This is a **test cleanup/ordering issue**:
- A previous test in the suite doesn't properly close the device
- When `BaseTransformerOperatorCall` runs, it tries to open an already-open device
- The test itself is correct (proven by passing when run alone)
- The issue is with test isolation and cleanup between tests

---

## Impact Assessment

### Production Impact: NONE
- Does not affect BERT model functionality
- Does not affect softmax operations
- Does not affect numerical correctness
- All functional BERT tests pass

### Development Impact: LOW
- Single test in polymorphism suite fails
- Test failure is deterministic and isolated
- Does not block BERT development or deployment

---

## Investigation and Fix

### Culprit Identified

**File**: `tests/core/softmax_precision_test.cpp`
**Test Suite**: `SoftmaxPrecisionBug` (ALL TESTS in this suite)
**Affected Tests**:
- `SoftmaxPrecisionBug.AttentionScorePattern`
- `SoftmaxPrecisionBug.RealBertAttentionScores`

**Issue**: The entire `SoftmaxPrecisionBug` test fixture does not properly close device in TearDown()

**Impact Scope**: **BREAKS ALL TESTS THAT RUN AFTER IT**
- Tested: ALL BertPolymorphismTest tests fail after SoftmaxPrecisionBug
- Tested: ALL BertWeightLoadingTest tests fail after SoftmaxPrecisionBug
- Tested: ALL BERTOperatorTest tests fail after SoftmaxPrecisionBug
- Tested: ALL BertHeadsTest tests fail after SoftmaxPrecisionBug
- **CRITICAL**: This test suite breaks the ENTIRE test run for any tests after it!

### Fix Required

Add proper TearDown() to `SoftmaxPrecisionBug` test fixture:

```cpp
class SoftmaxPrecisionBug : public ::testing::Test {
protected:
    void SetUp() override {
        // Existing setup code
    }

    void TearDown() override {
        // ADD THIS: Close device properly
        ttml::autograd::ctx().get_device().close();
        // Or equivalent cleanup
    }
};
```

### Verification After Fix

```bash
# Run the minimal 2-test sequence - should both PASS
./build/tests/ttml_tests --gtest_filter="SoftmaxPrecisionBug.RealBertAttentionScores:BertPolymorphismTest.BaseTransformerOperatorCall"

# Run full BERT suite - should get 53/53 PASS
./build/tests/ttml_tests --gtest_filter="*Bert*:*BERT*"
```

---

## Recommended Actions

### Immediate (Priority: LOW)
- ✅ Document issue (this file)
- ⏸️ Does not block current work on softmax workaround
- ⏸️ Does not block BERT model deployment

### Short-term (When Time Permits)
1. Investigate test fixture setup in `bert_polymorphism_test.cpp`
2. Fix device initialization order
3. Verify test runs independently and in suite

### Long-term
- Review all test fixtures for proper device lifecycle management
- Add device state validation in test infrastructure
- Ensure all tests properly clean up device state

---

## Verification

After fixing, verify with:

```bash
# Run failing test 10 times to ensure stability
for i in {1..10}; do
    echo "Run $i:"
    ./build/tests/ttml_tests --gtest_filter="BertPolymorphismTest.BaseTransformerOperatorCall"
done

# Run full BERT test suite to ensure no regressions
./build/tests/ttml_tests --gtest_filter="*Bert*:*BERT*"
```

Expected result: All 53 tests pass consistently.

---

## Related Files

- `tests/model/bert_polymorphism_test.cpp` - Test file
- `sources/ttml/autograd/auto_context.hpp` - Device context management
- Test fixture base classes - Device lifecycle management

---

## Conclusion

This test failure was a **test infrastructure issue** unrelated to the softmax FP32 workaround changes. It did not impact:
- BERT model functionality
- Softmax operation correctness
- Numerical precision
- Production deployments

---

## Fix Applied (November 15, 2025)

### Solution Implemented

Added proper test fixture to `tests/core/softmax_precision_test.cpp`:

```cpp
/**
 * Test fixture for SoftmaxPrecisionBug tests.
 * Properly manages device lifecycle to prevent breaking subsequent tests.
 */
class SoftmaxPrecisionBug : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};
```

Converted all tests from `TEST()` to `TEST_F()`:
- `TEST(SoftmaxPrecisionBug, AttentionScorePattern)` → `TEST_F(SoftmaxPrecisionBug, AttentionScorePattern)`
- `TEST(SoftmaxPrecisionBug, OtherBfloat16OpsWorkCorrectly)` → `TEST_F(SoftmaxPrecisionBug, OtherBfloat16OpsWorkCorrectly)`
- `TEST(SoftmaxPrecisionBug, RealBertAttentionScores)` → `TEST_F(SoftmaxPrecisionBug, RealBertAttentionScores)`

### Verification Results

✅ **All BERT C++ tests now pass**: 53/53

```bash
# Minimal reproduction - NOW PASSES
./build/tests/ttml_tests --gtest_filter="SoftmaxPrecisionBug.RealBertAttentionScores:BertPolymorphismTest.BaseTransformerOperatorCall"
# Result: Both tests PASS ✅

# Full BERT test suite - ALL PASS
./build/tests/ttml_tests --gtest_filter="*Bert*:*BERT*"
# Result: 53/53 tests PASS ✅
```

**Status**: ✅ **RESOLVED** - Test suite cleanup issue fixed, all BERT tests passing.
