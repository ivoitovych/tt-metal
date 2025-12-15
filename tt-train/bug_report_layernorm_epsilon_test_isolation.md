# Bug Report: LayerNormEpsilonTest.EpsilonIsStored Test Isolation Issue

## Issue Metadata

**Component / Area:** tests, tt-train, ttml

**Issue Type:** Test Isolation / Flaky Test

---

## Issue Description

### Observed

The test `LayerNormEpsilonTest.EpsilonIsStored` fails when run as part of a batch of tests but passes when run in isolation. This is a test isolation issue where state from previously executed tests affects the outcome.

**Batch run failure:**
```
[  FAILED  ] LayerNormEpsilonTest.EpsilonIsStored
```

**Individual run success:**
```
[ RUN      ] LayerNormEpsilonTest.EpsilonIsStored
[       OK ] LayerNormEpsilonTest.EpsilonIsStored (348 ms)
```

### Expected

The test should pass consistently regardless of whether it's run individually or as part of a batch. Tests should be isolated from each other.

---

## Steps to Reproduce the Issue

### 1. Steps (exact commands)

```bash
# Build tt-train
cd $TT_METAL_HOME/tt-train
rm -rf build/
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build --config Debug --clean-first

# Run test in batch (FAILS)
./build/tests/ttml_tests --gtest_filter="BERTOperatorTest.*:BertPolymorphismTest.*:BertWeightLoadingTest.*:BinaryOpsTest.*:GELUOpTest.*:LayerNormEpsilonTest.*:ScaledDotProductAttentionTest.*:SliceRepeatOpsTest.*:UnaryOpsTest.*:TileLayoutTest.*"

# Run test in isolation (PASSES)
./build/tests/ttml_tests --gtest_filter="LayerNormEpsilonTest.EpsilonIsStored"
```

### 2. Input data / link or description

No external data required. The test is self-contained.

**Test source location:** `tt-train/tests/modules/layer_norm_epsilon_test.cpp`

**Test code:**
```cpp
TEST_F(LayerNormEpsilonTest, EpsilonIsStored) {
    float eps_small = 1e-12F;
    float eps_large = 1e-5F;

    modules::LayerNormLayer ln_small(32, eps_small, false);
    modules::LayerNormLayer ln_large(32, eps_large, false);

    EXPECT_NEAR(ln_small.get_epsilon(), eps_small, 1e-15F);
    EXPECT_NEAR(ln_large.get_epsilon(), eps_large, 1e-8F);
}
```

### 3. Frequency

Intermittent when run in batch - fails when run after certain test suites, passes in isolation.

- **In batch with BERT/GELU/SDPA tests:** ~100% failure rate
- **In isolation:** 0% failure rate (always passes)

---

## System Details

### 1. Software Versions

- **OS version:** Ubuntu 22.04.5 LTS
- **Kernel:** Linux 6.8.0-87-generic
- **tt-metal commit:** 403df4beb0 (main-403df4beb0 branch)
- **BERT branch:** ivoitovych/bert-model-for-ttml-rebase-attempr-over-main-403df4beb0-2025-12-15
- **Compiler:** Clang 17.0.6

### 2. Hardware Details

- **Product:** Wormhole
- **Card/System:** N150 L (single card)
- **Board ID:** 0100018611902024
- **Driver:** TT-KMD 2.2.0
- **FW Bundle:** 18.5.0

---

## Regression Info (Optional)

**Is this a regression?** Unknown - test is new (added in BERT branch)

**Regression Details:**
- **First bad version:** N/A (new test)
- **Last known good version:** N/A (new test)
- **Git bisect status:** Not applicable

---

## Logs & Diagnostics (Optional)

### Batch Run Output (115 tests, 1 failed)

```
Running main() from gmock_main.cc
Note: Google Test filter = BERTOperatorTest.*:BertPolymorphismTest.*:BertWeightLoadingTest.*:BinaryOpsTest.*:GELUOpTest.*:LayerNormEpsilonTest.*:ScaledDotProductAttentionTest.*:SliceRepeatOpsTest.*:UnaryOpsTest.*:TileLayoutTest.*
[==========] Running 115 tests from 9 test suites.
[----------] Global test environment set-up.
...
[  FAILED  ] LayerNormEpsilonTest.EpsilonIsStored
...
[==========] 115 tests from 9 test suites ran. (99810 ms total)
[  PASSED  ] 114 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] LayerNormEpsilonTest.EpsilonIsStored

 1 FAILED TEST
```

### Isolation Run Output (passes)

```
Running main() from gmock_main.cc
Note: Google Test filter = LayerNormEpsilonTest.EpsilonIsStored
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from LayerNormEpsilonTest
[ RUN      ] LayerNormEpsilonTest.EpsilonIsStored
...
[       OK ] LayerNormEpsilonTest.EpsilonIsStored (348 ms)
[----------] 1 test from LayerNormEpsilonTest (348 ms total)
[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (348 ms total)
[  PASSED  ] 1 test.
```

---

## Impact & Priority (Optional)

**Priority:** P3 (Low)

**Impact:**
- **Affected workflows:** CI pipeline may show inconsistent results
- **Release or date risk:** Low - test is in feature branch, not merged to main

---

## Root Cause Analysis (Preliminary)

The test fixture uses `autograd::ctx()` for device management:

```cpp
void SetUp() override {
    autograd::ctx().open_device();
}

void TearDown() override {
    autograd::ctx().reset_graph();
    autograd::ctx().close_device();
}
```

Potential causes:
1. **Shared global state:** `autograd::ctx()` may retain state from previous tests
2. **Device state pollution:** Previous tests may leave the device in an unexpected state
3. **Memory not fully cleared:** Tensor allocations from prior tests may affect new allocations
4. **Order-dependent failure:** Specific test ordering triggers the issue

**Recommended investigation:**
1. Identify which specific test(s) running before cause the failure
2. Check if `autograd::ctx().reset_graph()` fully clears all state
3. Verify device is properly reinitialized between test fixtures

---

## Test Location

**Branch:** `ivoitovych/bert-model-for-ttml-rebase-attempr-over-main-403df4beb0-2025-12-15` (pushed to myfork)

**File:** `tt-train/tests/modules/layer_norm_epsilon_test.cpp`

---

## Date

2025-12-15
