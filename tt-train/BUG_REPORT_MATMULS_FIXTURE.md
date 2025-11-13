# Bug Report: tt-train MatmulsTest Device Lifecycle Bug

## Title
tt-train: MatmulsTest uses TEST() instead of TEST_F() causing device lifecycle leak and test-interaction failures

## Describe the bug

`ReduceOpTest.TestMeanDim0` **passes when run alone** but **fails when run after `MatmulsTest`** with the error:
```
C++ exception with description "open_device was called after the device was created." thrown in SetUp().
```

**Root Cause:** `MatmulsTest` defines a GoogleTest fixture with `SetUp()` and `TearDown()` methods for proper device lifecycle management, but all 8 test functions incorrectly use the `TEST()` macro instead of `TEST_F()`. This means:

1. The fixture class is defined but never used
2. `SetUp()` and `TearDown()` are **never called**
3. Device opens implicitly on first `ctx().get_device()` call
4. Device **never closes** because TearDown is not executed
5. Subsequent tests expecting a closed device fail

This is a **classic GoogleTest fixture usage error** - defining a fixture but forgetting to use `TEST_F` to activate it.

## Steps to reproduce the issue

### Demonstrating the Bug

```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal

# Step 1: Test PASSES when run alone
./tt-train/build/tests/ttml_tests --gtest_filter="ReduceOpTest.TestMeanDim0"
```

**Output:**
```
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from ReduceOpTest
[ RUN      ] ReduceOpTest.TestMeanDim0
[       OK ] ReduceOpTest.TestMeanDim0 (1792 ms)
[----------] 1 test from ReduceOpTest (1792 ms total)

[  PASSED  ] 1 test.
```

**Result:** ✅ PASSES

```bash
# Step 2: Test FAILS when run after MatmulsTest
./tt-train/build/tests/ttml_tests --gtest_filter="MatmulsTest.MatMulNoTranspose:ReduceOpTest.TestMeanDim0"
```

**Output:**
```
[==========] Running 2 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 1 test from MatmulsTest
[ RUN      ] MatmulsTest.MatMulNoTranspose
[       OK ] MatmulsTest.MatMulNoTranspose (446 ms)
[----------] 1 test from MatmulsTest (446 ms total)

[----------] 1 test from ReduceOpTest
[ RUN      ] ReduceOpTest.TestMeanDim0
unknown file: Failure
C++ exception with description "open_device was called after the device was created." thrown in SetUp().
[  FAILED  ] ReduceOpTest.TestMeanDim0 (2 ms)
[----------] 1 test from ReduceOpTest (2 ms total)

[  FAILED  ] 1 test, listed below:
[  FAILED  ] ReduceOpTest.TestMeanDim0
```

**Result:** ❌ FAILS

### Full Test Suite Impact

```bash
# Full test suite (excluding SSE crashes) hits this bug
./tt-train/build/tests/ttml_tests --gtest_filter=-RandomGenerationTests.ParallelUniformInitDeterminismSSE:RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE
```

**Result:** `ReduceOpTest.TestMeanDim0` fails at test 29/234

## Expected behavior

- All MatmulsTest tests should properly initialize and cleanup the device using the fixture
- Device should close after each test completes
- ReduceOpTest.TestMeanDim0 should pass whether run alone or after MatmulsTest
- Full test suite should run without device lifecycle conflicts

## Root Cause Analysis

### Current Code (INCORRECT)

**File:** `tt-train/tests/ttnn_fixed/matmuls_test.cpp`

**Lines 18-27** - Fixture is correctly defined:
```cpp
class MatmulsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};
```

**Lines 79, 100, 116, 132, 150, 174, 198, 222** - Tests use wrong macro:
```cpp
TEST(MatmulsTest, MatMulNoTranspose) {           // ❌ WRONG - doesn't use fixture
    // ...
}

TEST(MatmulsTest, MatMulTransposeA) {            // ❌ WRONG
    // ...
}

// ... 6 more tests, all using TEST() ...
```

**Problem:** Using `TEST()` creates standalone tests that **don't inherit from the fixture**, so SetUp/TearDown are never called.

### Execution Flow (Current - BROKEN)

1. `MatmulsTest.MatMulNoTranspose` runs
2. No SetUp() called (wrong macro used)
3. Test calls `ctx().get_device()` → device opens implicitly
4. Test completes
5. No TearDown() called (wrong macro used) → **device remains open**
6. `ReduceOpTest.TestMeanDim0` starts
7. SetUp() calls `open_device()` → **exception thrown** (device already open)

## Proposed Fix

**Change `TEST` to `TEST_F` in 8 locations:**

### Required Changes

**File:** `tt-train/tests/ttnn_fixed/matmuls_test.cpp`

| Line | Current (WRONG) | Fixed (CORRECT) |
|------|----------------|-----------------|
| 79   | `TEST(MatmulsTest, MatMulNoTranspose)` | `TEST_F(MatmulsTest, MatMulNoTranspose)` |
| 97   | `TEST(MatmulsTest, MatMulTransposeA)` | `TEST_F(MatmulsTest, MatMulTransposeA)` |
| 113  | `TEST(MatmulsTest, MatMulTransposeB)` | `TEST_F(MatmulsTest, MatMulTransposeB)` |
| 129  | `TEST(MatmulsTest, MatMulTransposeBoth)` | `TEST_F(MatmulsTest, MatMulTransposeBoth)` |
| 145  | `TEST(MatmulsTest, MatMulBackwardNoTranspose)` | `TEST_F(MatmulsTest, MatMulBackwardNoTranspose)` |
| 170  | `TEST(MatmulsTest, MatMulBackwardTransposeA)` | `TEST_F(MatmulsTest, MatMulBackwardTransposeA)` |
| 194  | `TEST(MatmulsTest, MatMulBackwardTransposeB)` | `TEST_F(MatmulsTest, MatMulBackwardTransposeB)` |
| 218  | `TEST(MatmulsTest, MatMulBackwardTransposeBoth)` | `TEST_F(MatmulsTest, MatMulBackwardTransposeBoth)` |

**Total changes:** 8 lines (add `_F` to each `TEST` macro)

**Effort estimate:** ~5 minutes

### How to Apply the Fix

```bash
# Edit the file and change all 8 TEST macros to TEST_F
# Or use sed:
cd /workspace/tt-metal/tt-train
sed -i 's/^TEST(MatmulsTest,/TEST_F(MatmulsTest,/g' tests/ttnn_fixed/matmuls_test.cpp

# Rebuild tests
cmake --build build --target ttml_tests

# Verify fix
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal
./tt-train/build/tests/ttml_tests --gtest_filter="MatmulsTest.MatMulNoTranspose:ReduceOpTest.TestMeanDim0"
```

**Expected result after fix:** Both tests PASS ✅

## Fix Verification Results

The fix has been tested and verified:

### Test 1: Previously Failing Combination
```bash
./tt-train/build/tests/ttml_tests --gtest_filter="MatmulsTest.MatMulNoTranspose:ReduceOpTest.TestMeanDim0"
```
**Result:** ✅ PASSED - Both tests now pass together

### Test 2: All MatmulsTest + ReduceOpTest
```bash
./tt-train/build/tests/ttml_tests --gtest_filter="MatmulsTest.*:ReduceOpTest.TestMeanDim0"
```
**Result:** ✅ PASSED - All 9 tests pass (8 MatmulsTest + 1 ReduceOpTest)

### Test 3: Three Consecutive Test Groups
```bash
./tt-train/build/tests/ttml_tests --gtest_filter="TrivialTnnFixedTest.*:MatmulsTest.*:ReduceOpTest.*"
```
**Result:** ✅ PASSED - All 24 tests pass (11 + 8 + 5)
- Verified no side effects on preceding tests (TrivialTnnFixedTest)
- Verified no side effects on subsequent tests (ReduceOpTest)
- Device lifecycle management working correctly

### Key Observation After Fix

**Before Fix:** Device opened once and never closed (leak)
**After Fix:** Device properly opens in SetUp() and closes in TearDown() for each test

Evidence: Test logs show "TopologyMapper mapping start" and "Profiler started" for each individual test, confirming proper device lifecycle.

## GoogleTest Background

For context, this is a well-documented GoogleTest pattern:

**`TEST()` macro:** Creates a standalone test
- Does **not** use fixtures
- SetUp/TearDown not called

**`TEST_F()` macro:** Creates a test that uses a fixture
- Test class inherits from the fixture
- SetUp() called before test runs
- TearDown() called after test completes

**Common mistake:** Defining a fixture but forgetting to use `TEST_F` to activate it (exactly what happened here).

## Environment Information

**Test Environment:**
- **OS:** Linux 6.8.0-79-generic (Ubuntu)
- **Branch:** `main`
- **Build Type:** Debug with ccache
- **Compiler:** Clang-17, C++20
- **Build System:** CMake + Ninja
- **Device:** Wormhole (chip 0)
- **Firmware Version:** 18.5.0

**Test Framework:**
- GoogleTest (version included with tt-train dependencies)
- Test executable: `./tt-train/build/tests/ttml_tests`
- Total tests in suite: 236

## Impact Assessment

**Severity:** Medium
- Easy fix (5 minutes, 8 character additions)
- Affects test reliability and developer productivity
- Causes confusion when tests pass individually but fail in suite
- Blocks proper device lifecycle management for subsequent tests

**Current Impact:**
- 1 test fails in full suite that should pass (`ReduceOpTest.TestMeanDim0`)
- Prevents clean test suite runs
- Device resource leak (device not closed between tests)

**After Fix:**
- All MatmulsTest tests will properly manage device lifecycle
- ReduceOpTest.TestMeanDim0 will pass in all scenarios
- Test suite will run cleanly (excluding SSE crash issues - separate bug)

## Summary

This bug report documents a test-interaction issue where `ReduceOpTest.TestMeanDim0` passes when run individually but fails after `MatmulsTest` due to improper GoogleTest fixture usage. The fix is simple (8 character additions) and has been fully tested and verified.

**Key Points:**
- ✅ **Easy fix:** Change `TEST` to `TEST_F` in 8 locations (~5 minutes)
- ✅ **Verified:** Fix tested with 3 consecutive test groups (24 tests, 100% pass rate)
- ✅ **No side effects:** Preceding and subsequent tests unaffected
- ✅ **Root cause:** Classic GoogleTest mistake - fixture defined but not used
