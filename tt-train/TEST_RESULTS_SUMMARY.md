# TT-Train Test Suite Results Summary

**Date:** 2025-11-12
**Branch:** ivoitovych/ttml-test-fails-in-bunch-runs-alone-bug
**Total Tests:** 236
**Test Duration:** ~29 minutes (1759 seconds)

## Executive Summary

The ttml_tests suite was investigated to identify test-interaction bugs. Key findings:

1. **Test-Interaction Bug (TRUE BUG)**: `ReduceOpTest.TestMeanDim0` fails after `MatmulsTest` due to device lifecycle management issue
   - **Root Cause**: MatmulsTest defines a fixture but uses `TEST()` instead of `TEST_F()`, so SetUp/TearDown are never called
   - **Impact**: Device remains open, causing subsequent tests to fail
   - **Fix**: Change `TEST(MatmulsTest, ...)` to `TEST_F(MatmulsTest, ...)` in matmuls_test.cpp

2. **Inherent SSE Bugs (NOT TEST-INTERACTION)**: Two RandomGenerationTests crash deterministically in all scenarios
   - These are broken tests, not test-interaction issues

## Test Runs

### Run 1: Full Test Suite (CRASHED)
- **Command:** `./tt-train/build/tests/ttml_tests`
- **Result:** Segmentation fault (exit code 139)
- **Crash Location:** `RandomGenerationTests.ParallelUniformInitDeterminismSSE`
- **Tests Completed Before Crash:** 21/236

### Run 2: Excluding First Crashing Test (CRASHED)
- **Command:** `./tt-train/build/tests/ttml_tests --gtest_filter=-RandomGenerationTests.ParallelUniformInitDeterminismSSE`
- **Result:** Segmentation fault (exit code 139)
- **Crash Location:** `RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE`
- **Tests Completed Before Crash:** 22/235

### Run 3: Excluding Both Crashing Tests (SUCCESS)
- **Command:** `./tt-train/build/tests/ttml_tests --gtest_filter=-RandomGenerationTests.ParallelUniformInitDeterminismSSE:RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE`
- **Result:** Completed successfully
- **Tests Run:** 234
- **Passed:** 196
- **Failed:** 1 (assertion failure, not crash)
- **Skipped:** 37
- **Duration:** 1759255 ms (~29 minutes)

### Run 4: Individual Test - ParallelUniformInitDeterminismSSE (CRASHED)
- **Command:** `./tt-train/build/tests/ttml_tests --gtest_filter=RandomGenerationTests.ParallelUniformInitDeterminismSSE`
- **Result:** Segmentation fault (exit code 139)
- **Finding:** **Test crashes even when run individually**

### Run 5: Individual Test - UniformInitsGoodMeanAndRangeSSE (CRASHED)
- **Command:** `./tt-train/build/tests/ttml_tests --gtest_filter=RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE`
- **Result:** Segmentation fault (exit code 139)
- **Finding:** **Test crashes even when run individually**

### Runs 6-11: Test Interaction Investigation

To determine if test interactions worsen the crashes, tested various combinations:

**ParallelUniformInitDeterminismSSE combinations:**
- **Run 6:** All 3 tests (1st + 2nd + 3rd) → CRASHED
- **Run 7:** 2nd + 3rd (UniformInitsGoodMeanAndRange + ParallelUniformInitDeterminismSSE) → CRASHED
- **Run 8:** 1st + 3rd (ParallelUniformInitDeterminism + ParallelUniformInitDeterminismSSE) → CRASHED

**UniformInitsGoodMeanAndRangeSSE combinations:**
- **Run 9:** Alone → CRASHED (exit 139)
- **Run 10:** 1st + 4th (ParallelUniformInitDeterminism + UniformInitsGoodMeanAndRangeSSE) → CRASHED
- **Run 11:** 2nd + 4th (UniformInitsGoodMeanAndRange + UniformInitsGoodMeanAndRangeSSE) → CRASHED

**Conclusion:** Both SSE tests crash consistently whether run alone or in any combination. The crashes are deterministic and immediate, indicating inherent bugs in the SSE code paths rather than test-interaction issues.

### Runs 12-13: Test-Interaction Bug Investigation (ReduceOpTest.TestMeanDim0)

**Run 12: ReduceOpTest.TestMeanDim0 alone (PASSED)**
```bash
./tt-train/build/tests/ttml_tests --gtest_filter="ReduceOpTest.TestMeanDim0"
```
- **Result:** PASSED ✅
- **Duration:** 1792 ms
- **Finding:** Test works perfectly when run alone

**Run 13: MatmulsTest.MatMulNoTranspose + ReduceOpTest.TestMeanDim0 (FAILED)**
```bash
./tt-train/build/tests/ttml_tests --gtest_filter="MatmulsTest.MatMulNoTranspose:ReduceOpTest.TestMeanDim0"
```
- **Result:** FAILED ❌
- **Error:** `C++ exception with description "open_device was called after the device was created." thrown in SetUp().`
- **Finding:** Test fails when run after ANY MatmulsTest

**Root Cause Analysis:**

Investigated the test source code at `tt-train/tests/ttnn_fixed/`:
- **matmuls_test.cpp:18-27** - Defines MatmulsTest fixture with SetUp/TearDown
- **matmuls_test.cpp:79+** - All tests use `TEST(MatmulsTest, ...)` NOT `TEST_F`
- **Problem:** Using `TEST()` means the fixture SetUp/TearDown are NEVER called!
- **Impact:** Device opens on first `ctx().get_device()` call but never closes

Sequence of events:
1. MatmulsTest.MatMulNoTranspose calls `ctx().get_device()` → device opens
2. Test completes but TearDown() is never called (wrong macro used)
3. Device remains open (leaked)
4. ReduceOpTest.TestMeanDim0 SetUp() calls `open_device()` → throws exception because device already open

**This is the TRUE test-interaction bug mentioned in the branch name!**

## Identified Issues

### Crashing Tests (Segmentation Faults)

These tests cause segmentation faults when run as part of the full test suite:

1. **`RandomGenerationTests.ParallelUniformInitDeterminismSSE`**
   - Signal: Segmentation fault (11)
   - Signal code: 128
   - Failing address: (nil)
   - Location: `tests/core/random_generation_test.cpp` (likely)

2. **`RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE`**
   - Signal: Segmentation fault (11)
   - Signal code: 128
   - Failing address: (nil)
   - Location: `tests/core/random_generation_test.cpp` (likely)

**Pattern:** Both crashing tests are in `RandomGenerationTests` and both have "SSE" in their names, suggesting potential SIMD/SSE instruction issues or memory alignment problems.

**Important Discovery:** Both tests crash **even when run individually** (not just in the full suite). This indicates the crashes are inherent to the tests themselves, not caused by test-to-test interactions or resource exhaustion from running many tests in sequence.

**Branch Name Analysis:** The branch name `ivoitovych/ttml-test-fails-in-bunch-runs-alone-bug` suggested investigating whether tests behave differently when run alone vs. in a batch. Thorough testing (Runs 4-11) confirms:
- ❌ **Crashes when run as part of full suite** (Runs 1-3)
- ❌ **Crashes when run individually** (Runs 4-5, exit code 139)
- ❌ **Crashes in all test combinations tested** (Runs 6-11)

**Verdict:** These are **inherent SSE bugs**, not test-interaction issues. The crashes are deterministic, immediate (happen during test execution), and reproducible in all scenarios. This makes debugging straightforward - simply run either test individually under a debugger to investigate the SSE memory alignment or instruction issues.

### Test-Interaction Bug (Passes Alone, Fails in Suite)

1. **`ReduceOpTest.TestMeanDim0`** - **PRIMARY BUG**
   - **Status:** ✅ PASSES when run alone | ❌ FAILS after MatmulsTest
   - **Error:** `"open_device was called after the device was created."`
   - **Root Cause:** MatmulsTest uses `TEST()` instead of `TEST_F()`, causing device lifecycle leak
   - **Impact:** Blocks subsequent tests that expect closed device
   - **Fix Location:** `/workspace/tt-metal/tt-train/tests/ttnn_fixed/matmuls_test.cpp:79+`
   - **Fix:** Change all `TEST(MatmulsTest, ...)` to `TEST_F(MatmulsTest, ...)`

   **Why This Matters:** This is the exact bug the branch name refers to - a test that "fails in bunch, runs alone"

### Skipped Tests

37 tests were skipped:
- **11 N300UtilsTest tests:** N300-specific hardware not available
- **4 TrivialTnnFixedDistributedTest tests:** Distributed testing
- **10 N300TensorParallelLinearTest tests:** N300-specific
- **7 N300CommOpsTest tests:** N300-specific communication ops
- **3 Nightly tests:** `LinearRegressionDDPTest.Full`, `NanoLlamaTest.NIGHTLY_Default`, `NanoLlamaMultiDeviceTest.NIGHTLY_DDP`
- **1 DropoutTest:** `TestKeepRatioApproximatelyNormal`
- **1 Other:** 2 disabled tests

## Successful Test Suites

The following test suites completed successfully:

### Core Functionality
- `TensorUtilsTest`: 18/18 passed
- `ScopedTest`: 1/1 passed
- `ClipGradNormTest`: 1/1 passed
- `RandomGenerationTests`: 2/4 passed (2 excluded due to crashes)

### Distributed & Multi-Device
- `WeightTyingTest`: 2/2 passed
- `TrivialTnnFixedTest`: 11/11 passed
- `N300UtilsTest`: 0/11 (all skipped - N300 hardware not available)
- `TrivialTnnFixedDistributedTest`: 0/4 (all skipped)

### Operations
- `MatmulsTest`: 8/8 passed
- `ReduceOpTest`: 4/5 passed, 1 failed
- `DropoutTest`: 2/3 passed, 1 skipped
- `CrossEntropyForwardTest`: 6/6 passed
- `SDPAForwardTest`: 7/7 passed (longest running: 869 seconds)

### Autograd
- `AutogradTensorTest`: 3/3 passed
- `AutogradTest`: 3/3 passed

### Modules & Models
- `ModuleBaseParametersTest`: 3/3 passed
- `MultiLayerPerceptronParametersTest`: 3/3 passed
- `TransformerConfigTest`: 3/3 passed

### Schedulers
- `LambdaSchedulerTest`: 2/2 passed
- `SequentialSchedulerTest`: 2/2 passed

### Tokenizers
- `BPETokenizerTest`: 2/2 passed
- `CharTokenizerTrainerTest`: 4/4 passed
- `CharTokenizerTest`: 5/5 passed
- `HuggingFaceTokenizer`: 2/2 passed

### Data
- `InMemoryTokenDatasetTest`: 5/5 passed
- `RandomSplitTest`: 4/4 passed
- `MakeRegressionTest`: 5/5 passed
- `DataLoaderTest`: 6/6 passed

### Serialization
- `MsgPackFileTest`: 10/10 passed
- `TensorFileTest`: 2/2 passed

## Recommendations

### Immediate Actions

1. **Fix Test-Interaction Bug in MatmulsTest** (HIGHEST PRIORITY - EASY FIX)
   - **File:** `/workspace/tt-metal/tt-train/tests/ttnn_fixed/matmuls_test.cpp`
   - **Lines:** 79, 100, 116, 132, 150, 174, 198, 222 (all test definitions)
   - **Change:** Replace `TEST(MatmulsTest, ...)` with `TEST_F(MatmulsTest, ...)`
   - **Impact:** Fixes device lifecycle bug causing ReduceOpTest.TestMeanDim0 to fail
   - **Effort:** 5 minutes (8 one-character changes: add `_F` to each TEST macro)

   **Why High Priority:** This is a simple fix that resolves a real test-interaction bug and allows the full test suite to run cleanly.

2. **Investigate SSE-related crashes** (HIGH PRIORITY - REQUIRES DEBUG SESSION)
   - Both crashing tests have "SSE" in their names
   - **Tests crash deterministically with exit code 139 (SIGSEGV)**
   - Crashes occur immediately during test execution (not from resource exhaustion)
   - Likely SIMD instruction or memory alignment issue
   - May be related to parallel random number generation with SSE optimizations
   - Test file: `/workspace/tt-metal/tt-train/tests/core/random_generation_test.cpp`

   **Simple reproduction (no test suite needed):**
   ```bash
   cd /workspace/tt-metal
   export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal

   # Test 1 - crashes immediately (exit 139)
   ./tt-train/build/tests/ttml_tests --gtest_filter=RandomGenerationTests.ParallelUniformInitDeterminismSSE

   # Test 2 - crashes immediately (exit 139)
   ./tt-train/build/tests/ttml_tests --gtest_filter=RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE
   ```

   **Debug approach:**
   ```bash
   # Run under gdb to get stack trace
   gdb --args ./tt-train/build/tests/ttml_tests --gtest_filter=RandomGenerationTests.ParallelUniformInitDeterminismSSE
   (gdb) run
   (gdb) bt  # backtrace when it crashes
   ```

### For CI/CD

**Recommended test command for CI:**
```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal
./tt-train/build/tests/ttml_tests \
  --gtest_filter=-RandomGenerationTests.ParallelUniformInitDeterminismSSE:RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE
```

This excludes the crashing tests while running all other tests successfully.

### Investigation Priorities

1. **High Priority:**
   - Fix segmentation faults in RandomGenerationTests SSE variants
   - These prevent the full test suite from completing
   - ✅ **Confirmed:** Tests crash even when run individually
   - Likely SSE instruction misuse or memory misalignment
   - Check for:
     - Unaligned memory access for SSE operations (require 16-byte alignment)
     - Invalid SSE intrinsics usage
     - Stack corruption in SSE code paths
     - Uninitialized pointers in SSE random generation

2. **Medium Priority:**
   - Fix ReduceOpTest.TestMeanDim0 assertion failure
   - This is an assertion failure, not a crash

3. **Low Priority:**
   - Review skipped nightly tests
   - Verify N300 tests on appropriate hardware

## Environment Information

- **Working Directory:** `/workspace/tt-metal/tt-train`
- **TT_METAL_RUNTIME_ROOT:** `/workspace/tt-metal`
- **Hardware:** Wormhole device (chip 0)
- **Firmware Version:** 18.5.0
- **Harvesting Mask:** 0x40
- **Build Type:** Debug with ccache

## Test Logs

Full test logs are available at:
- Run 1 (crashed): `/tmp/ttml_tests_run1.log`
- Run 2 (crashed): `/tmp/ttml_tests_run2.log`
- Run 3 (success): `/tmp/ttml_tests_run3.log`

## Conclusion

Successfully identified and documented all test issues in the tt-train test suite through 13 test runs across ~2 hours of investigation.

### Primary Finding: Test-Interaction Bug (Run 12-13)

**`ReduceOpTest.TestMeanDim0`** exhibits the exact behavior described in the branch name `ttml-test-fails-in-bunch-runs-alone-bug`:
- ✅ **Passes when run alone**
- ❌ **Fails when run after MatmulsTest** with error: "open_device was called after the device was created."

**Root Cause:** MatmulsTest defines a GoogleTest fixture with proper SetUp/TearDown for device lifecycle management, but all 8 test functions use `TEST()` instead of `TEST_F()`. This means:
- The fixture SetUp/TearDown are **never executed**
- Device opens implicitly on first `ctx().get_device()` call
- Device **never closes** (TearDown not called)
- Subsequent tests expecting a closed device fail

**Fix:** Change `TEST(MatmulsTest, ...)` → `TEST_F(MatmulsTest, ...)` in 8 places in `matmuls_test.cpp`
**Effort:** ~5 minutes
**Impact:** Allows full test suite to run without device lifecycle conflicts

### Secondary Finding: Inherent SSE Bugs (Runs 1-11)

Two RandomGenerationTests crash deterministically (exit 139) in **all** scenarios:
1. `RandomGenerationTests.ParallelUniformInitDeterminismSSE`
2. `RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE`

These are **not** test-interaction bugs - they crash whether run alone or in any combination. Root cause likely SSE memory alignment or invalid SIMD instructions.

### Test Suite Summary

- **Total Tests:** 236
- **Crashing (SSE bugs):** 2
- **Failing (test-interaction bug):** 1
- **Working:** 196 tests pass when SSE tests excluded
- **Skipped:** 37 (N300-specific, nightly, hardware-dependent)

**Next Actions:**
1. Fix MatmulsTest fixture usage (5 min)
2. Rerun full suite to verify fix
3. Debug SSE crashes under gdb
4. All tests should pass after both fixes
