# GELU ULP Fix Implementation

**GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/35290

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix-draft-pr-prep`

**Base Commit:** `50b633663b48e5dabc2f9ddc32ceb28c0a11c873`

**Status:** IMPLEMENTED AND VERIFIED (v3 + MPFR reference tests)

---

## Summary

The GELU implementation achieves **Max ULP = 7** across the entire BF16 range using high-precision
raw x polynomial segments. All polynomial segments now achieve **Max ULP = 1**; the only remaining
ULP > 1 is in the asymptotic region (exp() approximation error).

### Key Results (DAZ+FTZ Model)

| Metric | Value |
|--------|-------|
| **Max ULP** | 7 (at x = -5.969, asymptotic region) |
| **Mean ULP** | 0.01 |
| **ULP ≤ 1** | 99.80% of values |
| **ULP > 100** | 0% of values |
| **Polynomial regions** | All have Max ULP = 1 |

### Per-Segment ULP Analysis

| Segment | Range | Count | Mean ULP | Max ULP | Worst x |
|---------|-------|------:|----------:|--------:|--------:|
| Deep neg (FTZ) | x < -13.2 | 15,916 | 0.00 | 0 | N/A |
| Deep neg (asymp) | [-13.2, -5.5] | 163 | 2.78 | **7** | **-5.969** |
| Range A | [-5.5, -5.095] | 13 | 0.46 | 1 | -5.125 |
| Range B | [-5.095, -4.136] | 31 | 0.26 | 1 | -4.156 |
| Range C | [-4.136, -3.177] | 57 | 0.16 | 1 | -3.188 |
| Seg 10 | [-3.177, -2.218] | 62 | 0.05 | 1 | -2.984 |
| Seg 11 | [-2.218, -1.258] | 108 | 0.03 | 1 | -1.852 |
| Seg 12 | [-1.258, -0.299] | 264 | 0.02 | 1 | -0.508 |
| Neg overlap | [-0.299, -0.125] | 153 | 0.03 | 1 | -0.136 |
| Near-zero (Taylor) | [-0.125, 0.125] | 31,489 | 0.00 | 1 | 0.074 |
| Pos overlap | [0.125, 0.299] | 154 | 0.09 | 1 | 0.136 |
| Seg 13 | [0.299, 0.660] | 143 | 0.07 | 1 | 0.354 |
| Seg 14 | [0.660, 1.644] | 170 | 0.01 | 1 | 0.852 |
| Seg 15 | [1.644, 3.0] | 109 | 0.04 | 1 | 1.680 |
| Positive sat | x >= 3.0 | 16,192 | 0.01 | 1 | 3.000 |
| **OVERALL** | | **65,024** | **0.01** | **7** | |

### Cumulative Distribution

| ULP ≤ | Count | Percent |
|------:|------:|--------:|
| 0 | 64,621 | 99.38% |
| 1 | 64,897 | 99.80% |
| 2 | 64,940 | 99.87% |
| 3 | 64,974 | 99.92% |
| 5 | 65,016 | 99.99% |
| 7 | 65,024 | 100.00% |

### Version History

| Version | Branch | Max ULP | Key Change |
|---------|--------|---------|------------|
| v1 | -gelu-ulp-fix | 46 | Initial C6 implementation |
| v2 | -gelu-ulp-fix-02 | 11 | Extended asymptotic to -4.136 |
| v3 | -gelu-ulp-fix-03 | 7 | Raw x polynomials for [-5.5, -3.177] |
| v4 | -gelu-ulp-fix-04 | 7 | Added FTZ threshold research doc |
| **PR** | **-draft-pr-prep** | **7** | MPFR/mpmath 256-bit reference tests |

### Hardware Model Correction

**IMPORTANT**: The original bug report used an incorrect ULP calculator that did not account for
Tenstorrent hardware's DAZ+FTZ behavior. Per `tech_reports/Handling_Special_Value/special_values.md`:
"denormals | all | 0x0"

The SFPU treats all denormal values as zero. This affects ULP calculations:
- Denormal inputs are read as zero (DAZ)
- Denormal outputs are flushed to zero (FTZ)
- For ULP purposes, all denormals map to the same value as zero

With the corrected DAZ+FTZ model, the C6 implementation shows excellent accuracy.

---

## Documentation References

### 1. Independent Research Repository

**URL:** https://github.com/ivoitovych/bf16_gelu_research

| File | Description |
|------|-------------|
| `README.md` | Main documentation: 39 methods evaluated, Max ULP <= 1 achievable |
| `gelu_implementations.cpp` | All 39 GELU approximation implementations |
| `adaptive_poly.cpp` | C6 adaptive polynomial (best result: Max ULP = 1, Mean ULP = 0.001) |
| `FinalLists.md` | Strategy taxonomy: 40 methods across 8 categories |
| `SATURATION.md` | BF16 saturation threshold analysis |

**Key Finding:** C6 Adaptive Polynomial achieves Max ULP = 1 using:
- 16 error-optimized segments with degree-4 polynomials
- Taylor series near zero: `GELU(x) ≈ 0.5 * x` for tiny x
- Asymptotic tail expansion for large |x|

### 2. Kernel Implementation Reference

**Branch:** `myfork/ivoitovych/bert-model-for-ttml-pr-gelu-test-suite-amendment-ulp-diagnostic-04`

| File | Description |
|------|-------------|
| `tt-train/tests/ops/gelu_implementation_reference.md` | Complete call stack from ttnn::gelu() to SFPU kernels (~1150 lines) |
| `tt-train/tests/ops/gelu_precision_analysis.md` | Hardware precision analysis report |
| `tt-train/tests/ops/gelu_op_test.cpp` | Diagnostic tests (SpikeAnalysis, FloorAnalysis) |
| `tt-train/tests/ops/gelu_ulp_plot_wh_n150_bf16_ulp_module.png` | ULP error visualization |

**Call Stack (with fix applied):**
```
ttnn::gelu(tensor, fast_and_approximate_mode=false)
  → tt::tt_metal::gelu(queue, input, output, fast_and_approx_mode)
    → run_with_autoformat(queue, op, {input})
      → EltwiseUnary with UnaryOpType::GELU
        → llk_math_eltwise_unary_sfpu<SFPU_OP_GELU>
          → calculate_gelu() → calculate_gelu_c6()
```

### 3. Tenstorrent Kernel Programming Guide

**Location:** `~/tt/TENSTORRENT_KERNEL_PROGRAMMING.md` (local reference)

- DST registers and accumulation patterns
- Hardware mode transitions (UNPACKER involvement)
- SFPU API reference for compute kernels

### 4. Bug Report with Reproduction Tests

**GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/35290

**Branch:** `myfork/ivoitovych/bug-report-gelu-floor-value-ulp-03`

| File | Description |
|------|-------------|
| `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md` | Bug report with DAZ+FTZ correction |
| `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py` | Python test suite (27 tests) |
| `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp` | C++ test suite (18 tests: 10 ULP calculator + 8 device) |

**Cherry-pick tests:**
```bash
git fetch https://github.com/ivoitovych/tt-metal.git ivoitovych/bug-report-gelu-floor-value-ulp-03
git cherry-pick FETCH_HEAD
cmake --build build_Debug --target unit_tests_ttnn
```

---

## Root Cause Analysis

### Source File

**Path:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`

### Original Buggy Implementation (Before Fix)

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu_chebyshev() {
    // Chebyshev coefficients
    constexpr float c0 = 2.98325768482e-05f;  // <-- PROBLEM: floor value for tiny inputs
    constexpr float c1 = 0.500015258789f;
    // ... more coefficients ...

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[d];
        vFloat result = val;

        v_if(val >= -5.5f) {  // <-- PROBLEM: returns 0.0 for x < -5.5
            // Chebyshev polynomial evaluation
            result = c0;  // <-- PROBLEM: c0 dominates for tiny inputs
            result = result * val + c1;
            // ... polynomial evaluation ...
            result = result * val;
        }
        v_endif;

        dst_reg[d] = result;
    }
}
```

### Problem Details

**Region 1 (Deep Negative Tail):**
- `v_if(val >= -5.5f)` causes all x < -5.5 to return 0.0
- Exact GELU(-13.5) ≈ -1.06e-40 (tiny but representable in BF16 as subnormal-adjacent)
- BF16 can represent values down to ~1.18e-38, so these should NOT be zero

**Region 2 (Near-Zero):**
- For tiny x, polynomial evaluates to approximately c0 = 2.98e-05
- Expected: GELU(x) ≈ 0.5 * x for small x (Taylor series)
- Example: GELU(1e-38) should be ~5e-39, not 2.98e-05

**Region 3 (Transition):**
- Polynomial coefficients not optimized for region near -5.5 boundary
- Discontinuity between polynomial region and zero region

---

## Proposed Fixes

### Fix 1: Extend Deep Negative Threshold

Lower threshold from -5.5 to approximately -13.5 (BF16 saturation point):

```cpp
v_if(val >= -13.5f) {  // Extended threshold
    // Chebyshev polynomial
}
v_endif;
```

**Trade-off:** May need asymptotic expansion for -13.5 < x < -5.5 region.

### Fix 2: Taylor Series for Near-Zero

Add special case for tiny inputs where polynomial is inaccurate:

```cpp
v_if(val >= -13.5f) {
    vFloat abs_val = sfpi::abs(val);
    v_if(abs_val < 1e-4f) {
        result = val * 0.5f;  // Taylor series: GELU(x) ≈ 0.5*x for tiny x
    }
    v_else {
        // Chebyshev polynomial for normal range
    }
    v_endif;
}
v_endif;
```

### Fix 3: Adaptive Polynomial (Best Accuracy) ✅ IMPLEMENTED (v3)

Replace single Chebyshev with segmented adaptive polynomial using raw x evaluation:

```cpp
// 8 new high-precision segments using raw x (not normalized u):
// Range A: 4 segments for [-5.5, -5.095], Max ULP = 0-1
// Range B: 2 segments for [-5.095, -4.136], Max ULP = 1
// Range C: 2 segments for [-4.136, -3.177], Max ULP = 1
// Original C6 segments 10-15 for [-3.177, 3.0], Max ULP = 1
// Asymptotic expansion covers [-13.2, -5.5], Max ULP = 7
// See: https://github.com/ivoitovych/bf16_gelu_research/blob/main/adaptive_poly.cpp
```

**Result:** Max ULP = 7 (all polynomial segments have Max ULP = 1)

### Fix 4: Asymptotic Tail Expansion

For large negative x, use asymptotic formula instead of returning 0:

```cpp
// For x < -5.5, GELU(x) ≈ x * exp(-x²/2) / sqrt(2π) * (1/x - 1/x³ + ...)
// In BF16, this is effectively 0 for x < -13.5, but non-zero for -13.5 < x < -5.5
```

---

## Implementation Plan

### Phase 1: Cherry-pick Tests
- [x] Cherry-pick reproduction tests from bug report branch
- [x] Verify all tests fail (confirming bug exists)

### Phase 2: Minimal Fix (Region 1 + Region 2)
- [x] Extend threshold from -5.5 to -13.2 (asymptotic expansion)
- [x] Add Taylor series branch for |x| < 0.125
- [x] Verify Region 1 and Region 2 tests pass

### Phase 3: Polynomial Refit (Region 3)
- [x] Evaluate if transition region errors are acceptable
- [x] Implemented C6 adaptive polynomial with 14 segments (8 raw x + 6 normalized u)

### Phase 4: Validation
- [x] Run full BF16 sweep (65,024 non-denormal values)
- [x] Verify Max ULP is acceptable: **Max ULP = 7** (at x = -5.969 in asymptotic region)
- [x] Run existing GELU tests to ensure no regression

### Phase 5: PR Preparation
- [x] Add tests to CI suite (C++ and Python)
- [x] Update documentation with DAZ+FTZ model
- [x] Create PR with before/after ULP statistics

---

## Files Modified

| File | Status | Description |
|------|--------|-------------|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | ✅ Modified | C6 adaptive polynomial implementation |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | ✅ Modified | Same implementation (synced) |
| `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp` | ✅ Added | C++ test suite (21 tests: 10 ULP calculator + 11 device) |
| `tests/ttnn/unit_tests/gtests/CMakeLists.txt` | ✅ Modified | Added test_gelu_ulp_bug.cpp |
| `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py` | ✅ Added | Python test suite (27 tests) |
| `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md` | ✅ Added | Bug report documentation |
| `GELU_ULP_FIX_IMPLEMENTATION.md` | ✅ Added | This implementation guide |

---

## Progress Log

### 2026-01-06: Branch Created

- Created `ivoitovych/issue-35290-gelu-ulp-fix` from merge-base `50b633663b`
- Created this implementation document

### 2026-01-06: C6 Adaptive Polynomial + Asymptotic Implementation Complete

**Fix Applied:** Complete C6 adaptive polynomial implementation with asymptotic expansion for deep negative, based on research at https://github.com/ivoitovych/bf16_gelu_research

**Implementation Details:**
- Taylor series for near-zero: `GELU(x) ≈ x * (0.5 + 0.3989*x)` for |x| < 0.125
- Positive saturation: `GELU(x) = x` for x >= 3.0
- Asymptotic expansion for deep negative: `GELU(x) ≈ -exp(-x²/2) / √(2π)` for -13.2 < x < -4.136
- Deep negative cutoff: `GELU(x) = 0` for x < -13.2 (values below BF16 denormal minimum)
- 7 adaptive polynomial segments (segments 9-15 from C6) covering [-4.136, 3.0]
- Binary-search-like conditional structure for efficient segment selection

**Final Results (DAZ+FTZ Model):**

| Region | Before Fix (incorrect model) | After Fix (DAZ+FTZ) | Notes |
|--------|------------------------------|---------------------|-------|
| FTZ Region (x < -13.2) | 32,767 | 0 | Both expected and actual are 0 |
| Asymptotic (-13.2 to -4.136) | 32,767 | **11** | Extended asymptotic works well |
| Polynomial Segments | 1,475 | **3** | Segments 9-15 have low error |
| Near-Zero (Taylor) | 14,276 | **1** | Taylor series accurate |
| Positive Saturation | 0 | **1** | Identity function |

**Hardware FTZ Behavior:**
- For x < -13.2: exp(-x²/2) produces denormals that hardware flushes to 0
- Both expected and actual values are 0, so ULP = 0 (correct behavior)
- This is not a bug - it's the correct result under DAZ+FTZ model

**Sample Results by Region (DAZ+FTZ model):**

Deep Negative (Asymptotic, extended to -4.136):
```
x=-5.969:  ULP = 7 (within asymptotic region)
x=-5.125:  ULP = 8 (extended asymptotic region)
x=-4.188:  Max ULP = 11 (worst case, asymptotic/polynomial boundary)
x=-10.0:   ULP ≤ 3
x=-13.0:   ULP ≤ 2 (boundary of FTZ region)
```

Polynomial Segments (9-15):
```
x=-4.125:  Max ULP = 3 (segment 9)
x=-2.984:  Max ULP = 1 (segment 10)
x=-1.852:  Max ULP = 1 (segment 11)
```

Near-Zero (Taylor series):
```
x=0.074:   Max ULP = 1 (within Taylor region)
x=1e-10:   ULP = 0 (tiny values handled correctly)
```

Positive Saturation:
```
x=3.0:     Max ULP = 1 (identity function region)
```

**Files Modified:**
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` (Wormhole)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` (Blackhole - synced)

### 2026-01-08: Documentation Update and Test Verification

**DAZ+FTZ Model Correction:**
- Updated all ULP calculations to use correct DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero) model
- Per `tech_reports/Handling_Special_Value/special_values.md`: "denormals | all | 0x0"
- Original bug report overstated errors by not accounting for hardware denormal handling

**Test Results:**
- C++ tests: 21 passed (10 BFloat16UlpTest + 11 GeluUlpBugTest)
- Python tests: 27 passed (test_gelu_floor_value_bug.py)
- All tests verify fix works (low ULP) rather than asserting old buggy behavior

**Final Verification:**
- Comprehensive BF16 sweep: 65,024 non-denormal values tested
- Max ULP: 11 (at asymptotic/polynomial boundary x = -4.188)
- Mean ULP: 0.02
- 99.67% of values have ULP ≤ 1

### 2026-01-08: Extended Asymptotic Expansion (Seg 8 Fix)

**Problem:** Polynomial segments 7-8 had poor accuracy at boundary (Max ULP = 46 at x = -5.094)

**Solution:** Extended asymptotic expansion from x < -5.5 to x < -4.136, eliminating segments 7-8

**Changes:**
- Asymptotic expansion now covers [-13.2, -4.136] (was [-13.2, -5.5])
- Removed polynomial segments 7 and 8 (problematic boundary region)
- Polynomial segments 9-15 now cover [-4.136, 3.0]

**Results:**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Max ULP | 46 | **11** | 76% reduction |
| Mean ULP | 0.01 | 0.02 | Slight increase |
| Worst x | -5.094 | -4.188 | New boundary |

**New Tests Added:**
- `SubnormalOutputsFlushedToZero` - Verifies FTZ behavior (262 inputs)
- `MonotonicityVerification` - Verifies GELU monotonicity (65,026 values)
- `DenormalInputsProduceSameOutputAsZero` - Verifies DAZ behavior (254 inputs)

**Test Suite:** 21 C++ tests + 27 Python tests = 48 total tests

### 2026-01-10: Raw x Polynomial Segments (v3)

**Problem:** v2 still had Max ULP = 11 at the asymptotic/polynomial boundary (x = -4.188). The polynomial segments 7-8 were eliminated in v2, but the asymptotic expansion had limited accuracy near -4.136.

**Solution:** Added 8 new high-precision polynomial segments using **raw x evaluation** (not normalized `u = (x - mid) / scale`):
- Range A: 4 segments for [-5.5, -5.095], Max ULP = 0-1
- Range B: 2 segments for [-5.095, -4.136], Max ULP = 1
- Range C: 2 segments for [-4.136, -3.177], Max ULP = 1

**Key Insight:** The polynomial coefficients were fitted for raw x values, not normalized u. Using `result = POLY4(c0, c1, c2, c3, c4, val)` directly instead of normalizing first eliminates subtraction/multiplication overhead and achieves better numerical stability.

**Changes:**
- Reduced asymptotic region from x < -4.136 back to x < -5.5
- Added 8 new polynomial segments with raw x evaluation for [-5.5, -3.177]
- Updated test thresholds from 15 to 10 ULP

**Results:**

| Metric | v2 (-02) | v3 (-03) | Improvement |
|--------|----------|----------|-------------|
| Max ULP | 11 | **7** | 36% reduction |
| Worst x | -4.188 (poly boundary) | -5.969 (asymptotic only) | Poly now ≤1 |
| Mean ULP | 0.02 | 0.01 | 50% reduction |
| ULP ≤ 1 | 99.73% | 99.80% | +0.07% |

**All polynomial segments now have Max ULP = 1.** The only remaining ULP > 1 is in the asymptotic region where the exp() approximation introduces error.

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix-03`

### 2026-01-10: FTZ Threshold Research (v4)

**Added:** `GELU_BF16_Zero_Saturation_Threshold_Research.md` documenting MPFR 256-bit precision analysis.

**Key Finding:** True zero saturation threshold is x = -13.1875 (bf16: 0xC153), not -8.375 as fp64 suggests. The fp64 erf() function saturates to -1.0 prematurely, giving incorrect threshold.

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix-04`

### 2026-01-12: Reference Function Improvements (PR Prep)

**Problem:** The reference GELU function used fp64 `erf()` which saturates at x ≈ -8.375, giving incorrect expected values for the deep negative region.

**Solution (C++):** Use fp64 `erfc()` instead of `erf()` for negative x. Mathematical identity:
```
For x < 0: 1 + erf(x/√2) = erfc(|x|/√2)
```
The `erfc()` function returns small positive values for large arguments without saturation.

**Verification:** fp64 `erfc()` matches MPFR-256 with **0 ULP difference** across all 65,026 valid BF16 values.

**Solution (Python):** Uses `mpmath` 256-bit precision (standard Python arbitrary precision library).

**Changes:**
- C++ tests: `gelu_exact()` uses `erfc()` for negative x - no external dependencies
- Python tests: Uses `mpmath` for 256-bit precision reference
- CMakeLists.txt: Removed MPFR/GMP library linking (not needed)

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix-draft-pr-prep`

**Note:** This branch removed experimental v4 commits (11-segment polynomial approach) that were inefficient (5 coefficients per ~4 BF16 points). The core v3 implementation with Max ULP = 7 is retained.

---

## References

- **GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/35290
- **Research Repository:** https://github.com/ivoitovych/bf16_gelu_research
- **Bug Report Branch:** https://github.com/ivoitovych/tt-metal/tree/ivoitovych/bug-report-gelu-floor-value-ulp-03
- **PR Prep Branch:** https://github.com/ivoitovych/tt-metal/tree/ivoitovych/issue-35290-gelu-ulp-fix-draft-pr-prep
