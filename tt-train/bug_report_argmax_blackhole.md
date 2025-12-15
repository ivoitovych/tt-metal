# [tt-train/ops]: `ttnn::argmax` returns garbage values on Blackhole with masked sampling and unaligned tensor dimensions

## Issue Metadata

**Component / Area:** ops, tt-train, kernels

**Issue Type:** Incorrect output / Data corruption

## Issue Description

### Observed

The `TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask` test fails on Blackhole hardware, returning garbage values instead of valid argmax indices.

The test creates a tensor with shape `{1, 1, 32, 65}` (note: last dimension 65 is not aligned to tile boundary of 32), applies a mask to the last column, and calls `ttml::ttnn_fixed::sample()` which internally uses `ttnn::argmax` after `ttnn::untilize`.

**Actual output:**
```
Expected: (v) < (64), actual: 3201515335 vs 64
Expected: (v) < (64), actual: 3217342221 vs 64
Expected: (v) < (64), actual: 1066712917 vs 64
Expected: (v) < (64), actual: 3200859828 vs 64
...
```

The returned values (3201515335, 3217342221, etc.) appear to be garbage/uninitialized memory or incorrectly interpreted float bit patterns cast to `uint32_t`. Valid argmax indices should be in range `[0, 63]` since the last column (index 64) is masked.

### Expected

The test should pass, returning valid argmax indices in range `[0, 63]`, as it does on Wormhole hardware.

A similar test `TestSamplingPositiveTemperatureNoMask` with shape `{1, 1, 32, 64}` (aligned last dimension, no mask) passes on both architectures.

## Steps to Reproduce the Issue

### 1. Steps (exact commands)

```bash
# Build tt-train (from tt-metal root)
./build_metal.sh -b Release --build-tt-train

# Or standalone tt-train build
cd $TT_METAL_HOME/tt-train
cmake -DCMAKE_BUILD_TYPE=Debug -B build -GNinja
cmake --build build --config Debug

# Run the failing test
cd $TT_METAL_HOME
./tt-train/build/tests/ttml_tests --gtest_filter="TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask"

# Run the full test group to see passing vs failing
./tt-train/build/tests/ttml_tests --gtest_filter="TrivialTnnFixedTest.*"
```

### 2. Input data / link or description

The test is self-contained in:
`tt-train/tests/ttnn_fixed/trivial_ttnn_ops_test.cpp:277-298`

Test creates:
- Input tensor: shape `{1, 1, 32, 65}`, random float values
- Mask tensor: shape `{1, 1, 32, 65}`, zeros except last column set to `1e4`
- Calls `ttml::ttnn_fixed::sample(tensor_a, 1.0F, 42, tensor_mask)`

The `sample` function in `tt-train/sources/ttml/ttnn_fixed/trivial_ttnn_ops.cpp:79-112`:
1. Applies Gumbel sampling trick with random values
2. Subtracts the mask from logits
3. Returns `ttnn::argmax(ttnn::untilize(out), 3, true, std::nullopt, true)`

### 3. Frequency

**Always** - 100% reproducible on Blackhole hardware.

The test passes consistently on Wormhole hardware.

## System Details

### 1. Software Versions

- **OS version:** Ubuntu 22.04.5 LTS
- **Kernel:** 5.15.0-164-generic
- **tt-metal commit:** `403df4beb0f31a2b771349e58a02a98d859b9039` (main branch)
- **Firmware bundle version:** 19.3.0
- **ETH FW version:** 1.7.1
- **KMD version:** 2.6.1
- **tt-smi version:** 3.0.39

### 2. Hardware Details

- **Product:** Blackhole
- **Card/System:** P150 (PCI device 1e52:b140)
- **Single device configuration**

## Regression Info (Optional)

### Is this a regression?

Unknown - this may be a case of the operation never being validated on Blackhole rather than a regression.

### Regression Details

- **First bad version:** Unknown
- **Last known good version:** Works on Wormhole (N150/N300)
- **Git bisect status:** Not performed

## Logs & Diagnostics (Optional)

### Full test output (single test)

```
Running main() from gmock_main.cc
Note: Google Test filter = TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from TrivialTnnFixedTest
[ RUN      ] TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask
2025-12-15 16:33:42.809 | info     |             UMD | Starting topology discovery.
2025-12-15 16:33:42.895 | warning  |             UMD | Firmware bundle version 19.3.0 on the system is newer than the maximum supported version 19.1.0 for blackhole architecture. New features may not be supported.
...
/home/ivoitovych/tt/tt-metal/tt-train/tests/ttnn_fixed/trivial_ttnn_ops_test.cpp:296: Failure
Expected: (v) < (64), actual: 3201515335 vs 64
Expected: (v) < (64), actual: 3217342221 vs 64
Expected: (v) < (64), actual: 1066712917 vs 64
Expected: (v) < (64), actual: 3200859828 vs 64
Expected: (v) < (64), actual: 3202563797 vs 64
Expected: (v) < (64), actual: 3207970364 vs 64
Expected: (v) < (64), actual: 3217375136 vs 64
Expected: (v) < (64), actual: 3204431830 vs 64
Expected: (v) < (64), actual: 1072873412 vs 64
Expected: (v) < (64), actual: 3207184383 vs 64
Expected: (v) < (64), actual: 1058192921 vs 64
[  FAILED  ] TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask (1548 ms)
```

### Test group summary

```
[==========] 11 tests from TrivialTnnFixedTest ran.
[  PASSED  ] 10 tests.
[  FAILED  ] 1 test: TestSamplingPositiveTemperatureWithMask
```

### Full ttml_tests summary

```
[==========] 323 tests from 74 test suites ran. (344005 ms total)
[  PASSED  ] 274 tests.
[  SKIPPED ] 48 tests (N300 multi-device tests, NIGHTLY tests)
[  FAILED  ] 1 test: TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask
```

## Impact & Priority (Optional)

### Priority

**P2** - Medium

### Impact

- **Affected workflows:** tt-train sampling operations with masks on Blackhole, inference with masked token sampling
- **Affected users:** Developers using tt-train on Blackhole hardware
- **Release or date risk:** Blocks tt-train validation on Blackhole

## Analysis Notes

The key differences between the passing and failing tests:

| Test | Shape | Mask | Result |
|------|-------|------|--------|
| `TestSamplingPositiveTemperatureNoMask` | `{1,1,32,64}` | No | PASS |
| `TestSamplingPositiveTemperatureWithMask` | `{1,1,32,65}` | Yes | FAIL |

Potential root causes to investigate:
1. **Unaligned tensor dimension:** The last dimension (65) is not a multiple of 32, which may cause alignment issues in NOC transactions or circular buffer operations on Blackhole
2. **Untilize + argmax pipeline:** The `ttnn::untilize` followed by `ttnn::argmax` may have Blackhole-specific issues with row-major data
3. **Memory alignment differences:** Blackhole has different DRAM/L1 alignment requirements (`hal::get_dram_alignment()`) that may affect the multi-core argmax kernel
4. **Mask subtraction operation:** The `ttnn::subtract` with the mask tensor may produce incorrect values on Blackhole that then cause argmax to fail
