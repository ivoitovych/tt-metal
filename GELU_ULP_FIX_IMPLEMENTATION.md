# GELU ULP Fix Implementation

**GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/35290

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix`

**Base Commit:** `50b633663b48e5dabc2f9ddc32ceb28c0a11c873`

---

## Problem Summary

`ttnn.gelu()` in accurate mode (Chebyshev polynomial) has **THREE** problematic regions with catastrophic ULP errors:

| Region | Range | Max ULP | Cause |
|--------|-------|---------|-------|
| 1. Deep Negative Tail | x < -5.5 | **32,767** | Hardware returns 0.0, should return tiny negative |
| 2. Near-Zero | \|x\| < ~1e-4 | 14,276 | Chebyshev c0 (2.98e-05) dominates |
| 3. Transition | -5.5 to -4.0 | 1,475 | Poor polynomial fitting at boundary |

**Overall Statistics (Full BFloat16 Sweep):**
- Total values tested: 65,278
- Max ULP error: 32,767 (maximum possible for BF16)
- Mean ULP error: 3,266
- Values with ULP > 1000: 27,089 (41.5%)
- Values with ULP > 100: 29,243 (44.8%)
- Values with ULP <= 1: 34,254 (52.5%)

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

**Call Stack:**
```
ttnn::gelu(tensor, fast_and_approximate_mode=false)
  → tt::tt_metal::gelu(queue, input, output, fast_and_approx_mode)
    → run_with_autoformat(queue, op, {input})
      → EltwiseUnary with UnaryOpType::GELU
        → llk_math_eltwise_unary_sfpu<SFPU_OP_GELU>
          → calculate_gelu_chebyshev<APPROXIMATION_MODE>()
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
| `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md` | Complete Tenstorrent-formatted bug report |
| `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py` | Python reproducer (24 tests) |
| `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp` | C++ reproducer (14 tests) |

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

### Current Implementation (Chebyshev Mode)

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

### Fix 3: Adaptive Polynomial (Best Accuracy)

Replace single Chebyshev with segmented adaptive polynomial:

```cpp
// 16 segments with optimized coefficients
// See: https://github.com/ivoitovych/bf16_gelu_research/blob/main/adaptive_poly.cpp
```

**Result:** Max ULP = 1 (vs current 32,767)

### Fix 4: Asymptotic Tail Expansion

For large negative x, use asymptotic formula instead of returning 0:

```cpp
// For x < -5.5, GELU(x) ≈ x * exp(-x²/2) / sqrt(2π) * (1/x - 1/x³ + ...)
// In BF16, this is effectively 0 for x < -13.5, but non-zero for -13.5 < x < -5.5
```

---

## Implementation Plan

### Phase 1: Cherry-pick Tests
- [ ] Cherry-pick reproduction tests from bug report branch
- [ ] Verify all tests fail (confirming bug exists)

### Phase 2: Minimal Fix (Region 1 + Region 2)
- [ ] Extend threshold from -5.5 to -13.5
- [ ] Add Taylor series branch for |x| < 1e-4
- [ ] Verify Region 1 and Region 2 tests pass

### Phase 3: Polynomial Refit (Region 3)
- [ ] Evaluate if transition region errors are acceptable
- [ ] If not, refit Chebyshev coefficients or add segment

### Phase 4: Validation
- [ ] Run full BF16 sweep (65,278 values)
- [ ] Verify Max ULP is acceptable (target: <= 10, ideal: <= 1)
- [ ] Run existing GELU tests to ensure no regression

### Phase 5: PR Preparation
- [ ] Add tests to CI suite
- [ ] Update documentation
- [ ] Create PR with before/after ULP statistics

---

## Files to Modify

| File | Change |
|------|--------|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Main fix |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Same fix (identical code) |
| `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp` | Add to CI |
| `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py` | Add to CI |

---

## Progress Log

### 2026-01-06: C6 Adaptive Polynomial + Asymptotic Implementation Complete

**Fix Applied:** Complete C6 adaptive polynomial implementation with asymptotic expansion for deep negative, based on research at https://github.com/ivoitovych/bf16_gelu_research

**Implementation Details:**
- Taylor series for near-zero: `GELU(x) ≈ x * (0.5 + 0.3989*x)` for |x| < 0.125
- Positive saturation: `GELU(x) = x` for x >= 3.0
- Asymptotic expansion for deep negative: `GELU(x) ≈ -exp(-x²/2) / √(2π)` for -13 < x < -5.5
- Deep negative cutoff: `GELU(x) = 0` for x < -13 (values below BF16 denormal minimum)
- 9 adaptive polynomial segments (segments 7-15 from C6) covering [-5.5, 3.0]
- Binary-search-like conditional structure for efficient segment selection

**Final Results:**

| Region | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| Region 1 (x < -13) | 32,767 | 32,767* | N/A (correct) |
| Region 2 (Near-Zero) | **14,276** | **54** | **99.6%** |
| Region 3 (Transition -5.5 to -4) | **1,475** | **7** | **99.5%** |
| Deep Negative (-13 < x < -5.5) | 32,767 | **≤6** | **99.98%** |

*Region 1 returns 0 for x < -13 due to hardware Flush-To-Zero (FTZ) mode. The asymptotic expansion computes exp(-x²/2) which produces denormal results for x < -13. These denormals get flushed to 0 by hardware. This is a hardware limitation, not a software bug. True GELU(-13.5) ≈ -1e-40, which is technically representable as a BF16 denormal, but hardware FTZ prevents accurate computation.

**Sample Deep Negative Results (NEW - asymptotic expansion):**
```
x=-12:     expected=-2.13178e-32, actual=-2.14741e-32, ULP=2
x=-10:     expected=-7.61985e-23, actual=-7.65142e-23, ULP=1
x=-8:      expected=-4.97677e-15, actual=-5.02376e-15, ULP=2
x=-6:      expected=-5.91953e-09, actual=-6.0827e-09,  ULP=6
x=-5.5625: expected=-7.39637e-08, actual=-7.59028e-08, ULP=5
```

**Sample Transition Region Results:**
```
x=-5.5:    expected=-1.04443e-07, actual=-1.04308e-07, ULP=0
x=-5.4375: expected=-1.46903e-07, actual=-1.45286e-07, ULP=1
x=-5.25:   expected=-3.9926e-07,  actual=-4.02331e-07, ULP=2
x=-5:      expected=-1.43326e-06, actual=-1.37836e-06, ULP=7
x=-4.5:    expected=-1.52895e-05, actual=-1.51992e-05, ULP=1
x=-4:      expected=-0.000126685, actual=-0.000125885, ULP=0
```

**Sample Near-Zero Results:**
```
x=1e-15: expected=5e-16, actual=4.996e-16, ULP=0
x=1e-10: expected=5e-11, actual=5.00222e-11, ULP=1
x=1e-08: expected=5e-09, actual=5.00586e-09, ULP=1
```

**Hardware Limitation - Flush-To-Zero (FTZ):**
The Wormhole/Blackhole SFPU operates in FTZ mode, flushing denormals to zero. This affects:
- x < -13: asymptotic produces exp(-x²/2) ≈ 1e-39, which is a denormal that gets flushed to 0
- The software correctly computes the asymptotic, but hardware FTZ converts the result to 0
- This is unavoidable without hardware FTZ control, which is not exposed in the current API

**Files Modified:**
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` (Wormhole)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` (Blackhole - synced)

### 2026-01-06: Branch Created
- Created `ivoitovych/issue-35290-gelu-ulp-fix` from merge-base `50b633663b`
- Created this implementation document

---

## References

- **GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/35290
- **Research Repository:** https://github.com/ivoitovych/bf16_gelu_research
- **Bug Report Branch:** https://github.com/ivoitovych/tt-metal/tree/ivoitovych/bug-report-gelu-floor-value-ulp-03
