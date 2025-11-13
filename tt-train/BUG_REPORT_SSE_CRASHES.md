# Bug Report: tt-train RandomGenerationTests SSE Crashes

## Title
tt-train: RandomGenerationTests with SSE crash with SIGSEGV (exit 139)

## Describe the bug

Two tests in the tt-train test suite (`ttml_tests`) crash with segmentation fault (SIGSEGV, exit code 139) in all scenarios:

1. `RandomGenerationTests.ParallelUniformInitDeterminismSSE`
2. `RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE`

Both tests crash **deterministically** whether run:
- Individually (alone)
- In pairs with other tests
- As part of the full test suite

The crashes are **immediate** (occur during test execution, not from resource exhaustion) and **reproducible** on every run. Both failing tests have "SSE" in their names, suggesting SIMD/SSE instruction issues or memory alignment problems.

**Pattern:** The non-SSE variants of these tests pass successfully:
- ✅ `RandomGenerationTests.ParallelUniformInitDeterminism` - PASSES
- ✅ `RandomGenerationTests.UniformInitsGoodMeanAndRange` - PASSES
- ❌ `RandomGenerationTests.ParallelUniformInitDeterminismSSE` - CRASHES
- ❌ `RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE` - CRASHES

## Steps to reproduce the issue

### Reproducing First Crash

```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal

# Test crashes when run alone
./tt-train/build/tests/ttml_tests --gtest_filter="RandomGenerationTests.ParallelUniformInitDeterminismSSE"
```

**Output:**
```
Running main() from gmock_main.cc
Note: Google Test filter = RandomGenerationTests.ParallelUniformInitDeterminismSSE
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from RandomGenerationTests
[ RUN      ] RandomGenerationTests.ParallelUniformInitDeterminismSSE
Segmentation fault (core dumped)
```

**Exit code:** 139 (SIGSEGV)

### Reproducing Second Crash

```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal

# Test crashes when run alone
./tt-train/build/tests/ttml_tests --gtest_filter="RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE"
```

**Output:**
```
Running main() from gmock_main.cc
Note: Google Test filter = RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from RandomGenerationTests
[ RUN      ] RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE
Segmentation fault (core dumped)
```

**Exit code:** 139 (SIGSEGV)

### Full Test Suite Impact

```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal

# Full test suite crashes at first SSE test
./tt-train/build/tests/ttml_tests
```

**Result:** Test suite crashes at test 21/236 (ParallelUniformInitDeterminismSSE), preventing remaining 215 tests from running.

### Verification: Non-SSE Tests Pass

```bash
# Non-SSE variants work correctly
./tt-train/build/tests/ttml_tests --gtest_filter="RandomGenerationTests.ParallelUniformInitDeterminism:RandomGenerationTests.UniformInitsGoodMeanAndRange"
```

**Result:** ✅ Both tests PASS

## Expected behavior

- SSE-optimized random generation tests should execute without crashing
- Tests should either pass or fail with assertion errors, not segmentation faults
- Test suite should complete all 236 tests without crashes

## Log Output

### From Full Test Suite Run

```
[ RUN      ] RandomGenerationTests.ParallelUniformInitDeterminism
[       OK ] RandomGenerationTests.ParallelUniformInitDeterminism (2254 ms)
[ RUN      ] RandomGenerationTests.UniformInitsGoodMeanAndRange
[       OK ] RandomGenerationTests.UniformInitsGoodMeanAndRange (10418 ms)
[ RUN      ] RandomGenerationTests.ParallelUniformInitDeterminismSSE
Signal: Segmentation fault (11)
Signal code: 128
Failing address: (nil)
```

### Test Combinations Attempted

All combinations crash:
- **Alone:** Both SSE tests crash individually ❌
- **After 1st test:** `ParallelUniformInitDeterminism` + `ParallelUniformInitDeterminismSSE` → CRASH ❌
- **After 2nd test:** `UniformInitsGoodMeanAndRange` + `ParallelUniformInitDeterminismSSE` → CRASH ❌
- **All 3 together:** `ParallelUniformInitDeterminism` + `UniformInitsGoodMeanAndRange` + `ParallelUniformInitDeterminismSSE` → CRASH ❌

**Conclusion:** These are inherent bugs in the SSE code paths, not test-interaction issues.

## Debug Recommendations

### Immediate Debug Steps

```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal

# Run under gdb to get stack trace
gdb --args ./tt-train/build/tests/ttml_tests --gtest_filter=RandomGenerationTests.ParallelUniformInitDeterminismSSE
(gdb) run
(gdb) bt
(gdb) info registers
```

### Likely Root Causes to Investigate

1. **SSE Memory Alignment:**
   - SSE instructions require 16-byte aligned memory
   - Check all SSE data structures for proper `alignas(16)` or `__attribute__((aligned(16)))`
   - Verify malloc/new allocations return aligned memory for SSE operations

2. **Invalid SSE Intrinsics:**
   - Review SSE intrinsic usage in parallel random generation
   - Verify proper SSE header includes (`<emmintrin.h>`, `<xmmintrin.h>`, etc.)
   - Check for incorrect SSE operation sequences

3. **Uninitialized Pointers:**
   - Verify all pointers used in SSE code paths are properly initialized
   - Check for null pointer dereferences before SSE operations

4. **Stack/Heap Corruption:**
   - Check for buffer overruns in SSE loop unrolling
   - Verify SIMD vector sizes match allocated buffer sizes

### Code Locations to Review

**Test File (likely):** `/workspace/tt-metal/tt-train/tests/core/random_generation_test.cpp`

Look for:
- Test definitions: `ParallelUniformInitDeterminismSSE` and `UniformInitsGoodMeanAndRangeSSE`
- Differences between SSE and non-SSE variants
- SSE-specific code paths in parallel random number generation
- Memory allocation for SSE operations

## Environment Information

**Test Environment:**
- **OS:** Linux 6.8.0-79-generic (Ubuntu)
- **Branch:** `main` (also reproducible on `ivoitovych/ttml-test-fails-in-bunch-runs-alone-bug`)
- **Build Type:** Debug with ccache
- **Compiler:** Clang-17, C++20
- **Build System:** CMake + Ninja
- **Device:** Wormhole (chip 0)
- **Firmware Version:** 18.5.0
- **Harvesting Mask:** 0x40
- **TT_METAL_RUNTIME_ROOT:** `/workspace/tt-metal`

**Test Framework:**
- GoogleTest (version included with tt-train dependencies)
- Test executable: `./tt-train/build/tests/ttml_tests`
- Total tests in suite: 236

**Crash Statistics:**
- **Crashing tests:** 2/236 (0.85%)
- **Crash type:** SIGSEGV (signal 11)
- **Exit code:** 139
- **Reproducibility:** 100% (crashes on every run)
- **Crash timing:** Immediate (during test execution)

## Impact Assessment

**Severity:** Medium-High
- Prevents full test suite from completing (blocks 215 tests after first crash)
- Reproducible on all environments tested
- Affects test reliability and CI/CD pipelines

**Current Workaround:**
Exclude crashing tests from test runs:
```bash
./tt-train/build/tests/ttml_tests --gtest_filter=-RandomGenerationTests.ParallelUniformInitDeterminismSSE:RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE
```

With this filter:
- **Result:** 234/236 tests complete successfully
- **Passed:** 196 tests
- **Failed:** 1 test (unrelated assertion failure)
- **Skipped:** 37 tests (N300-specific, nightly, hardware-dependent)

## Additional Context

**Investigation Summary:**
- Tested all combinations: alone, pairs, sequences
- Crashes are consistent across all scenarios (100% reproducible)
- Non-SSE variants work correctly, isolating issue to SSE code paths

**Related Tests:**
- `RandomGenerationTests.ParallelUniformInitDeterminism` ✅ PASSES
- `RandomGenerationTests.UniformInitsGoodMeanAndRange` ✅ PASSES
- `RandomGenerationTests.ParallelUniformInitDeterminismSSE` ❌ CRASHES
- `RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE` ❌ CRASHES
