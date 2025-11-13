# Fix MatmulsTest device lifecycle leak by using TEST_F macro

### Ticket
Fixes #32252

### Problem description

**Test-interaction bug:** `ReduceOpTest.TestMeanDim0` passes when run individually but fails after `MatmulsTest` with error:
```
"open_device was called after the device was created."
```

**Root Cause:**
`MatmulsTest` defined a GoogleTest fixture with `SetUp()` and `TearDown()` methods for proper device lifecycle management, but all 8 test functions incorrectly used the `TEST()` macro instead of `TEST_F()`.

This is a classic GoogleTest mistake - the fixture class exists but is never used because the tests don't inherit from it. As a result:
- Fixture `SetUp()` and `TearDown()` methods are never executed
- Device opens implicitly on first `ctx().get_device()` call
- Device never closes (TearDown not called)
- Subsequent tests expecting a closed device fail

This bug only manifests when tests run sequentially (in the full test suite), not when run individually, making it a true test-interaction issue as described by the branch name `ttml-test-fails-in-bunch-runs-alone-bug`.

### What's changed

**Fix:** Changed `TEST()` to `TEST_F()` for all 8 MatmulsTest functions to properly activate the fixture.

**Files Modified:**
- `tt-train/tests/ttnn_fixed/matmuls_test.cpp` (lines 79, 97, 113, 129, 145, 170, 194, 218)
- 8 insertions, 8 deletions (one character addition per line: `_F`)

**Impact:**
- ✅ Device lifecycle now managed correctly (opens in SetUp(), closes in TearDown())
- ✅ Resolves test-interaction failure of `ReduceOpTest.TestMeanDim0`
- ✅ All MatmulsTest tests properly initialize and cleanup device resources
- ✅ Test suite can run without device lifecycle conflicts

**Testing:**
- Verified all 8 MatmulsTest tests pass with proper device lifecycle
- Verified ReduceOpTest.TestMeanDim0 now passes after MatmulsTest
- Tested 3 consecutive test groups (TrivialTnnFixedTest + MatmulsTest + ReduceOpTest = 24 tests)
- Result: 100% pass rate, no side effects on preceding or subsequent tests
