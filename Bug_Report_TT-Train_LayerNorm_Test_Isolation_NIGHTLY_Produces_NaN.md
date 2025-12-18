[TT-Train] LayerNorm Backward: NIGHTLY Test Produces NaN When Run After Other Tests

### Component / Area

TT-Train / LayerNorm Backward Operation / Test Infrastructure

### Issue Type (optional)

Bad Outputs

### Observed

The `LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit` test produces NaN values and fails when run in sequence after `MetalLayerNormBw_OneTile` and `MetalLayerNormBw_TwoIncompleteTiles` tests.

**Critical observation:** The issue ONLY occurs when BOTH smaller tests run before the NIGHTLY test:
| Test Sequence | Result |
|---------------|--------|
| NIGHTLY alone | PASS |
| OneTile → NIGHTLY | PASS |
| TwoIncompleteTiles → NIGHTLY | PASS |
| OneTile → TwoIncompleteTiles → NIGHTLY | **FAIL (NaN)** |

### Expected

All tests should pass regardless of execution order. The NIGHTLY test should produce valid numerical results whether run alone or after other tests in the same suite.

### 1. Steps (exact commands)

**Fastest Reproduction (existing build):**
```bash
cd $TT_METAL_HOME

# This PASSES:
./tt-train/build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit"

# This FAILS with NaN:
./tt-train/build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.MetalLayerNormBw_OneTile:LayerNormBackwardOpTest.MetalLayerNormBw_TwoIncompleteTiles:LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit"
```

**Full Reproduction (from clean state):**
```bash
# Clone and build
git clone --recurse-submodules git@github.com:tenstorrent/tt-metal.git
cd tt-metal
./create_venv.sh
source python_env/bin/activate
./build_metal.sh --debug --build-all --enable-ccache

# Or standalone tt-train build (recommended):
cd tt-train
rm -rf build/
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build --config Debug --clean-first

# Run failing sequence
./tt-train/build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.MetalLayerNormBw_OneTile:LayerNormBackwardOpTest.MetalLayerNormBw_TwoIncompleteTiles:LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit"
```

**Container Reproduction:**
```bash
docker run --rm \
    --privileged \
    --device=/dev/tenstorrent:/dev/tenstorrent \
    -v /dev:/dev \
    -v /sys:/sys \
    -v /lib/modules:/lib/modules:ro \
    -v /dev/hugepages:/dev/hugepages \
    --cap-add=ALL \
    --security-opt apparmor=unconfined \
    <container-image>:latest \
    bash -c "cd /workspace/tt-metal && ./tt-train/build/tests/ttml_tests --gtest_filter='LayerNormBackwardOpTest.MetalLayerNormBw_OneTile:LayerNormBackwardOpTest.MetalLayerNormBw_TwoIncompleteTiles:LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit'"
```

### 2. Input data / link or description

**Test Parameters:**
| Test | batch | seq | heads | features | tiles |
|------|-------|-----|-------|----------|-------|
| `MetalLayerNormBw_OneTile` | 1 | 13 | 1 | 20 | <1 |
| `MetalLayerNormBw_TwoIncompleteTiles` | 1 | 32 | 1 | 33 | ~1 |
| `NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit` | 3 | 273 | 1 | 8462 | 265 |

**Relevant Source Files:**
- Test file: [`tt-train/tests/ops/layernorm_bw_fused_op_test.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp)
- Kernel: [`tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp)
- Program Factory: [`tt-train/sources/ttml/metal/ops/layernorm_bw/device/layernorm_bw_program_factory.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/layernorm_bw_program_factory.cpp)

### 3. Frequency

100% reproducible when running the specific 3-test sequence.

### 1. Software Versions

**Verified on multiple independent builds:**

| Environment | Commit Hash | Verification Date | Result |
|-------------|-------------|-------------------|--------|
| Host (origin/main) | `8321610e95c1a27f7b911243e56b5ae8aaa6ae55` | Dec 17-18, 2025 | Bug confirmed |
| Host (standalone rebuild) | `8321610e95c1a27f7b911243e56b5ae8aaa6ae55` | Dec 18, 2025 | Bug confirmed |
| Host (after board reset) | `8321610e95c1a27f7b911243e56b5ae8aaa6ae55` | Dec 18, 2025 | Bug confirmed |
| Host (clean rebuild + kernel cache clear) | `8321610e95c1a27f7b911243e56b5ae8aaa6ae55` | Dec 18, 2025 | Bug confirmed |
| Container | `84f83fcf55b9fbfb774393edcdc264b4d08f84dc` | Dec 17-18, 2025 | Bug confirmed |

All environments exhibit identical behavior: NIGHTLY passes alone, fails after the two smaller tests.

### 2. Hardware Details

- **Device:** Wormhole n150 L (single card)
- **Board ID:** 0100018611902024
- **Host OS:** Ubuntu 22.04.5 LTS
- **Kernel:** 6.8.0-87-generic
- **Driver:** TT-KMD 2.2.0
- **FW Bundle:** 18.5.0

### Is this a regression?

Unknown

### Regression Details

The bug exists in:
1. Host build from commit `8321610e95c1a27f7b911243e56b5ae8aaa6ae55` (Dec 17, 2025)
2. Container build from commit `84f83fcf55b9fbfb774393edcdc264b4d08f84dc` (Dec 17, 2025)

Earlier commits not tested for this specific issue. Bisection would be needed to determine if this is a regression.

### Logs & Diagnostics

**Failing run output:**
```
[ RUN      ] LayerNormBackwardOpTest.MetalLayerNormBw_OneTile
[       OK ] LayerNormBackwardOpTest.MetalLayerNormBw_OneTile (XXX ms)
[ RUN      ] LayerNormBackwardOpTest.MetalLayerNormBw_TwoIncompleteTiles
[       OK ] LayerNormBackwardOpTest.MetalLayerNormBw_TwoIncompleteTiles (XXX ms)
[ RUN      ] LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit
Value of: xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, 5e-1F)
  Actual: false
Expected: true
[  FAILED  ] LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit (14658 ms)
[  PASSED  ] 2 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit
```

**Diagnostic investigation (with added max_diff logging):**
- dx max_diff values are NaN or extremely large (1e+16 to 1e+36)
- This indicates garbage/uninitialized data from the metal kernel
- The issue is NOT tolerance-related - these are NaN values, not precision errors

**Attempted mitigations that did NOT resolve the issue:**
| Attempt | Result |
|---------|--------|
| Adding `reset_graph()` to TearDown | No effect |
| Clearing kernel cache (`rm -rf ~/.cache/tt-metal-cache/`) | No effect |
| Clean rebuild from scratch | No effect |
| Clean rebuild + kernel cache clear combined | No effect |
| Different test orderings | Only fails with specific 3-test sequence |
| Board reset (`tt-smi -r 0`) between runs | No effect - bug reproduces immediately after reset |

**Note:** The bug reproducing after a full board reset indicates this is NOT residual device state from previous runs. The corruption happens within a single test run.

### Priority

P2

### Impact

- **Test Suite Reliability:** The full `LayerNormBackwardOpTest.*` suite cannot run reliably
- **CI Impact:** Running all LayerNorm backward tests together produces false failures
- **Development Impact:** Developers may incorrectly assume their changes broke the NIGHTLY test
- **Workaround Available:** Run NIGHTLY test in isolation or exclude from full suite runs

### Acceptance Criteria

This bug will be considered fixed when:
1. [ ] The 3-test sequence (OneTile → TwoIncompleteTiles → NIGHTLY) passes consistently
2. [ ] The full `LayerNormBackwardOpTest.*` suite passes regardless of test ordering
3. [ ] Root cause is identified and documented
4. [ ] Fix does not regress other LayerNorm tests
5. [ ] (Optional) Regression test added to prevent recurrence

### Additional Context

**Discovery:** This issue was discovered while investigating test failures during work on GitHub issue #34625 (LayerNorm backward accumulation fix). The test isolation bug exists in the original codebase (origin/main).

**Potential fix identified:** The accumulation fix branch (`ivoitovych/issue-34625-layernorm-accumulation-fix-2`) appears to also fix or mask this test isolation bug. When running the 3-test sequence with a binary built from that branch, all tests pass. This suggests the root cause may be related to the same kernel code that was modified for the accumulation fix (specifically the `compute_dy_gamma_sum()` function in `layernorm_bw_kernel.cpp`).

**Hypothesis:** The fact that running two small tests (20 and 33 features) before a large test (8462 features) causes device memory corruption suggests a potential issue with:
1. Device memory management between tests
2. Circular buffer allocation/deallocation not being fully cleaned up
3. Kernel state not being properly reset between test fixture SetUp/TearDown
4. Hardware state accumulation that only manifests with specific size transitions

**Note:** The small→small→large pattern specifically triggering the issue (while small→large works fine) suggests the cumulative effect of two tests is required to corrupt the state.

---

## Executive Summary

**Problem:** The LayerNorm backward NIGHTLY test produces NaN/garbage values and fails when run after two specific smaller tests, but passes when run in isolation.

**Root Cause:** Unknown - appears to be device state corruption or memory management issue between tests.

**Impact:** Full LayerNorm backward test suite cannot run reliably; false failures in CI.

**Workaround:** Run NIGHTLY test in isolation.
