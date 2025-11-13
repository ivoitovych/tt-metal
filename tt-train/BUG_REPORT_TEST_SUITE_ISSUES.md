# Bug Report: tt-train Test Suite Issues - Device Lifecycle Bug and SSE Crashes

## Title
tt-train: MatmulsTest device lifecycle bug causes test-interaction failures + SSE RandomGenerationTests crash

## Describe the bug

The tt-train test suite (`ttml_tests`) has two distinct issues preventing clean test runs:

### Issue 1: Test-Interaction Bug (HIGH PRIORITY - EASY FIX)
**`ReduceOpTest.TestMeanDim0` fails when run after `MatmulsTest` but passes when run alone.**

**Root Cause:** `MatmulsTest` defines a GoogleTest fixture with `SetUp()` and `TearDown()` methods for device lifecycle management, but all 8 test functions incorrectly use `TEST()` macro instead of `TEST_F()`. This causes:
- Fixture `SetUp()`/`TearDown()` are never executed
- Device opens implicitly on first `ctx().get_device()` call
- Device never closes (TearDown not called)
- Subsequent tests expecting a closed device fail with: `"open_device was called after the device was created."`

**Impact:** Blocks any test that runs after MatmulsTest and expects proper device lifecycle management.

### Issue 2: SSE Crashes (REQUIRES DEBUG SESSION)
**Two RandomGenerationTests crash with SIGSEGV (exit code 139) in all scenarios:**
- `RandomGenerationTests.ParallelUniformInitDeterminismSSE`
- `RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE`

**Root Cause:** Unknown - likely SSE memory alignment issue or invalid SIMD instructions. These tests crash whether run alone or in any combination, indicating inherent bugs (not test-interaction).

## Steps to reproduce the issue

### Reproducing Issue 1 (Test-Interaction Bug)

```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal

# Test passes when run alone
./tt-train/build/tests/ttml_tests --gtest_filter="ReduceOpTest.TestMeanDim0"
# Result: ✅ PASSED (1792 ms)

# Test fails when run after MatmulsTest
./tt-train/build/tests/ttml_tests --gtest_filter="MatmulsTest.MatMulNoTranspose:ReduceOpTest.TestMeanDim0"
# Result: ❌ FAILED
# Error: "C++ exception with description "open_device was called after the device was created." thrown in SetUp()."

# Full test suite also hits this issue
./tt-train/build/tests/ttml_tests --gtest_filter=-RandomGenerationTests.ParallelUniformInitDeterminismSSE:RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE
# Result: ReduceOpTest.TestMeanDim0 fails after MatmulsTest completes
```

### Reproducing Issue 2 (SSE Crashes)

```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal

# Both tests crash even when run individually
./tt-train/build/tests/ttml_tests --gtest_filter="RandomGenerationTests.ParallelUniformInitDeterminismSSE"
# Result: Segmentation fault (exit code 139)

./tt-train/build/tests/ttml_tests --gtest_filter="RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE"
# Result: Segmentation fault (exit code 139)

# Also crashes in full suite
./tt-train/build/tests/ttml_tests
# Result: Crashes at RandomGenerationTests.ParallelUniformInitDeterminismSSE (test 21/236)
```

## Expected behavior

### For Issue 1
- All MatmulsTest tests should properly initialize and cleanup the device using the fixture
- ReduceOpTest.TestMeanDim0 should pass whether run alone or after MatmulsTest
- Full test suite should run without device lifecycle conflicts

### For Issue 2
- RandomGenerationTests with SSE optimizations should execute without crashing
- Tests should pass or fail with assertion errors, not segmentation faults

## Proposed Fix

### Fix for Issue 1 (5 minutes, 8 character changes)

**File:** `/workspace/tt-metal/tt-train/tests/ttnn_fixed/matmuls_test.cpp`

**Changes Required:** Replace `TEST(` with `TEST_F(` on lines:
- Line 79: `TEST(MatmulsTest, MatMulNoTranspose)` → `TEST_F(MatmulsTest, MatMulNoTranspose)`
- Line 100: `TEST(MatmulsTest, MatMulTransposeA)` → `TEST_F(MatmulsTest, MatMulTransposeA)`
- Line 116: `TEST(MatmulsTest, MatMulTransposeB)` → `TEST_F(MatmulsTest, MatMulTransposeB)`
- Line 132: `TEST(MatmulsTest, MatMulTransposeBoth)` → `TEST_F(MatmulsTest, MatMulTransposeBoth)`
- Line 150: `TEST(MatmulsTest, MatMulBackwardNoTranspose)` → `TEST_F(MatmulsTest, MatMulBackwardNoTranspose)`
- Line 174: `TEST(MatmulsTest, MatMulBackwardTransposeA)` → `TEST_F(MatmulsTest, MatMulBackwardTransposeA)`
- Line 198: `TEST(MatmulsTest, MatMulBackwardTransposeB)` → `TEST_F(MatmulsTest, MatMulBackwardTransposeB)`
- Line 222: `TEST(MatmulsTest, MatMulBackwardTransposeBoth)` → `TEST_F(MatmulsTest, MatMulBackwardTransposeBoth)`

**Explanation:** The fixture is correctly defined (lines 18-27) but never used because tests don't inherit from it. Using `TEST_F` makes tests inherit from the fixture, ensuring `SetUp()` and `TearDown()` are called.

### Fix for Issue 2 (Requires Investigation)

Debug under gdb to identify SSE issue:
```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal
gdb --args ./tt-train/build/tests/ttml_tests --gtest_filter=RandomGenerationTests.ParallelUniformInitDeterminismSSE
(gdb) run
(gdb) bt
```

Likely issues to investigate:
- SSE memory alignment (SSE requires 16-byte alignment)
- Invalid SSE intrinsics usage
- Uninitialized pointers in SSE code paths
- Stack corruption in parallel random generation

## Environment Information

**Discovered on:**
- **OS:** Linux 6.8.0-79-generic
- **Branch:** `ivoitovych/ttml-test-fails-in-bunch-runs-alone-bug`
- **Build:** Debug with ccache, Ninja, C++20, Clang-17
- **Device:** Wormhole (chip 0)
- **Firmware:** 18.5.0
- **Harvesting Mask:** 0x40
- **Test Suite:** tt-train/build/tests/ttml_tests
- **Total Tests:** 236
- **GoogleTest:** Version included with tt-train dependencies

**Test Results:**
- With both issues: 21/236 tests complete before crash
- Excluding SSE tests but with Issue 1: ReduceOpTest.TestMeanDim0 fails (22/235)
- Excluding SSE tests + fixing Issue 1: Expected 234/234 tests pass (196 pass, 37 skip, 1 assertion failure unrelated to these bugs)

## Additional Context

**Investigation Summary:**
- Conducted 13 test runs to isolate and identify root causes
- Full investigation documented in: `/workspace/tt-metal/tt-train/TEST_RESULTS_SUMMARY.md`
- Test logs available: `/tmp/ttml_tests_run{1,2,3}.log`, `/tmp/reduce_test_alone.log`, `/tmp/matmuls_reduce_test.log`

**Priority Recommendation:**
1. **Fix Issue 1 immediately** - trivial fix with major impact (allows test suite to run)
2. **Investigate Issue 2** - requires debug session but doesn't block other tests if filtered out

**Workaround for CI:**
Exclude crashing SSE tests until Issue 2 is resolved:
```bash
./tt-train/build/tests/ttml_tests --gtest_filter=-RandomGenerationTests.ParallelUniformInitDeterminismSSE:RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE
```

---

**Related Files:**
- Test source: `tt-train/tests/ttnn_fixed/matmuls_test.cpp` (Issue 1)
- Test source: `tt-train/tests/core/random_generation_test.cpp` (Issue 2)
- Full investigation: `tt-train/TEST_RESULTS_SUMMARY.md`
