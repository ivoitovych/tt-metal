TT-Train: Test Isolation Bug - LayerNorm backward dgamma tolerance failure after SoftmaxTest + SiLUOpTest + SDPAForwardTest NIGHTLY

### Component / Area

tt-train / LayerNorm backward kernel

### Issue Type (optional)

Bad Outputs

### Observed

`LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features` test FAILS when run after SoftmaxTest + SiLUOpTest + SDPAForwardTest NIGHTLY tests with dgamma tolerance failure:

```
Value of: xt::allclose(metal_dgamma_flat, dgamma_ref, 1.0e-2F, 2.0e-1F)
dgamma failed with tighter tolerance (iter=0)
```

The same test PASSES when run in isolation.

### Expected

Test should PASS regardless of which tests run before it. Device state from previous tests should not affect LayerNorm backward dgamma calculations.

### 1. Steps (exact commands)

**Test in isolation - PASSES:**
```bash
cd ~/tt/tt-metal
rm -rf ~/.cache/tt-metal-cache/*
./tt-train/build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features"
# Result: PASS
```

**Minimal reproduction - FAILS:**
```bash
cd ~/tt/tt-metal
rm -rf ~/.cache/tt-metal-cache/*
./tt-train/build/tests/ttml_tests --gtest_filter="SoftmaxTest.*:SiLUOpTest.*:SDPAForwardTest.NIGHTLY*:LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features"
# Result: FAIL - dgamma failed with tighter tolerance (iter=0)
```

### 2. Input data / link or description

**Investigation results - cumulative effect requires ALL THREE:**

| SoftmaxTest | SiLUOpTest | SDPA NIGHTLY (both) | Result |
|-------------|------------|---------------------|--------|
| - | - | - | PASS |
| Yes | - | - | PASS |
| - | Yes | - | PASS |
| - | - | Yes | PASS |
| Yes | Yes | - | PASS |
| Yes | - | Yes | PASS |
| - | Yes | Yes | PASS |
| **Yes** | **Yes** | **Yes** | **FAIL** |

Key findings:
1. Failure requires ALL THREE: SoftmaxTest (4 tests) + SiLUOpTest (11 tests) + BOTH SDPAForwardTest.NIGHTLY tests
2. Any two combinations pass - only the triple combination fails
3. Running just one SDPA NIGHTLY test instead of both also passes
4. This is a cumulative device state issue, not a single culprit test
5. **Only 4096Features fails** - the 8192Features test PASSES in the same sequence
6. Test parameters: batch=1, seq_len=100, heads=1, features=4096 (128 tiles) vs 8192 (256 tiles)

**Related issues:**

| Issue | Symptom | Trigger Sequence | Status |
|-------|---------|------------------|--------|
| **#34747** | NIGHTLY produces NaN | OneTile + TwoIncompleteTiles | Open |
| **This bug** | dgamma tolerance failure | SoftmaxTest + SiLUOpTest + SDPA NIGHTLY | New |

These appear to be two distinct test isolation bugs with different symptoms and triggers, but both affect LayerNorm backward operations.

**Reproduction test branch:** [`ivoitovych/bug-report-layernorm-test-isolation-dgamma`](https://github.com/ivoitovych/tt-metal/tree/ivoitovych/bug-report-layernorm-test-isolation-dgamma)

### Reproduction from Cherry-Pick

To reproduce this bug on any branch based on main:

1. Cherry-pick the reproduction test commit (HEAD of reproduction branch):
   ```bash
   git remote add ivoitovych https://github.com/ivoitovych/tt-metal.git
   git fetch ivoitovych ivoitovych/bug-report-layernorm-test-isolation-dgamma
   git cherry-pick ivoitovych/ivoitovych/bug-report-layernorm-test-isolation-dgamma
   ```

2. Build tt-train:
   ```bash
   cd tt-train && rm -rf build/
   cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
         -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
   cmake --build build --config Debug
   ```

3. Run the reproduction:
   ```bash
   rm -rf ~/.cache/tt-metal-cache/*
   ./tt-train/build/tests/ttml_tests --gtest_filter="SoftmaxTest.*:SiLUOpTest.*:SDPAForwardTest.NIGHTLY*:LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features"
   ```

### 3. Frequency

100% reproducible with the minimal reproduction command above.

### 1. Software Versions

- Branch: `ivoitovych/bug-report-layernorm-test-isolation-dgamma` (based on main at 8321610e95)
- OS: Ubuntu 22.04.5 LTS
- Kernel: 6.8.0-87-generic
- Python: 3.10.12

### 2. Hardware Details

- Machine: movsianikov-tt
- Card: Wormhole n150 L (single card)
- Board ID: 0100018611902024
- Driver: TT-KMD 2.2.0
- FW Bundle: 18.5.0

### Is this a regression?

Unknown

### Regression Details

This is a **test isolation bug** discovered during test suite investigation. The underlying device state accumulation issue exists in origin/main.

### Logs & Diagnostics

Full failure output:
```
[ RUN      ] LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features
Value of: xt::allclose(metal_dgamma_flat, dgamma_ref, 1.0e-2F, 2.0e-1F)
  Actual: false
Expected: true
dgamma failed with tighter tolerance (iter=0)
[  FAILED  ] LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features (667 ms)
```

**Root cause hypothesis:** The combination of SoftmaxTest + SiLUOpTest + SDPAForwardTest NIGHTLY tests accumulates device state (possibly L1 memory corruption, tile buffer residue, or kernel state) that causes subsequent LayerNorm backward dgamma calculations to produce slightly incorrect results.

**Why 4096 fails but 8192 passes:** The 4096-feature test uses 128 tiles while the 8192-feature test uses 256 tiles. The corrupted state may affect a specific memory region or tile buffer configuration that only manifests with the 128-tile case. This could be related to:
- L1 buffer sizing/alignment differences
- Kernel grid configuration
- Tile accumulator state

#### Container Verification

**Docker container:** `ivoitovych-tt-metal-env-built-debug-84f83fcf55` (origin/main, Dec 17 2025)

**Test 1: Existing tests with standard tolerance (PASS - masks issue)**

```bash
docker run --rm --privileged \
    --device=/dev/tenstorrent:/dev/tenstorrent \
    -v /dev:/dev -v /dev/hugepages:/dev/hugepages \
    ivoitovych-tt-metal-env-built-debug-84f83fcf55:latest \
    bash -c "cd /workspace/tt-metal && rm -rf ~/.cache/tt-metal-cache/* && \
             ./tt-train/build/tests/ttml_tests --gtest_filter='SoftmaxTest.*:SiLUOpTest.*:SDPAForwardTest.NIGHTLY*:LayerNormBackwardOpTest.*'"
```
- **Result:** All 22 tests PASSED (4 SoftmaxTest + 11 SiLUOpTest + 2 SDPA NIGHTLY + 5 LayerNorm)
- **Interpretation:** Standard tolerance (atol=0.5) masks the underlying device state corruption.

**Test 2: Cherry-pick reproduction test and run (FAIL - confirms bug)**

Full container reproduction with cherry-pick:
```bash
docker run --rm --privileged \
    --device=/dev/tenstorrent:/dev/tenstorrent \
    -v /dev:/dev -v /dev/hugepages:/dev/hugepages \
    ivoitovych-tt-metal-env-built-debug-84f83fcf55:latest \
    bash -c "cd /workspace/tt-metal && \
             git remote add ivoitovych https://github.com/ivoitovych/tt-metal.git && \
             git fetch ivoitovych ivoitovych/bug-report-layernorm-test-isolation-dgamma && \
             git cherry-pick --no-commit ivoitovych/ivoitovych/bug-report-layernorm-test-isolation-dgamma && \
             cd tt-train && rm -rf build/ && \
             cmake -DCMAKE_BUILD_TYPE=Debug -B build -GNinja && \
             cmake --build build --config Debug && \
             rm -rf ~/.cache/tt-metal-cache/* && \
             ./build/tests/ttml_tests --gtest_filter='SoftmaxTest.*:SiLUOpTest.*:SDPAForwardTest.NIGHTLY*:LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features'"
```

**Result:**
```
[  PASSED  ] 17 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] LayerNormBackwardOpTest.TestIsolation_TighterTolerance_4096Features

dgamma failed with tighter tolerance (iter=0)
dgamma failed with tighter tolerance (iter=1)
dgamma failed with tighter tolerance (iter=2)
```

| Suite | Tests | Result |
|-------|-------|--------|
| SoftmaxTest | 4 | PASS |
| SiLUOpTest | 11 | PASS |
| SDPAForwardTest.NIGHTLY | 2 | PASS |
| LayerNormBackwardOpTest.TestIsolation | 1 | **FAIL** |

**Conclusion:** Bug confirmed in origin/main via container reproduction. The tighter tolerance test (atol=0.2) exposes the device state corruption that standard tolerance (atol=0.5) masks.

### Priority

P2

### Impact

- **Test Reliability:** Tests may produce false positives with standard tolerance
- **CI Impact:** Adding tighter-tolerance tests will fail in full suite runs
- **Accuracy Concerns:** Device state corruption may affect production accuracy in edge cases
- **Workarounds available:**
  1. Run LayerNorm tests in isolation
  2. Use standard tolerance (atol=0.5)
  3. Fix root cause: Investigate device state accumulation across test suites

**Classification:**
- Type: Test Isolation Bug
- Severity: Medium (affects test reliability, workaround available)

**Test file:** `tt-train/tests/ops/layernorm_bw_fused_op_test.cpp`

**Reproduction test branch:** `ivoitovych/bug-report-layernorm-test-isolation-dgamma`

**Reproduction test code:**
```cpp
// Test passes in isolation, FAILS after SoftmaxTest + SiLUOpTest + SDPA NIGHTLY
TEST_F(LayerNormBackwardOpTest, TestIsolation_TighterTolerance_4096Features) {
    // batch=1, seq_len=100, heads=1, features=4096 (128 tiles)
    // Tighter tolerance: rtol=0.01, atol=0.2 (standard is 0.5)
    CompareKernelVsXArrayWithTolerance(1, 100, 1, 4096, 1.0e-2F, 2.0e-1F, 3);
}
```

**Verification after fix:**
```bash
# Build tt-train
cd ~/tt/tt-metal/tt-train && rm -rf build/ && \
  cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja && \
  cmake --build build --config Debug

# Clear kernel cache
rm -rf ~/.cache/tt-metal-cache/*

# Run reproduction sequence - should PASS after fix
./tt-train/build/tests/ttml_tests --gtest_filter="SoftmaxTest.*:SiLUOpTest.*:SDPAForwardTest.NIGHTLY*:LayerNormBackwardOpTest.TestIsolation*"
```
