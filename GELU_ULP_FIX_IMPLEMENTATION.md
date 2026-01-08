# GELU ULP Fix Implementation

**GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/35290

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix`

**Base Commit:** `50b633663b48e5dabc2f9ddc32ceb28c0a11c873`

**Status:** IMPLEMENTED AND VERIFIED

---

## Summary

The C6 Adaptive Polynomial GELU implementation achieves **Max ULP = 46** across the entire BF16 range
when measured with the correct DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero) hardware model.

### Key Results (DAZ+FTZ Model)

| Metric | Value |
|--------|-------|
| **Max ULP** | 46 (at x = -5.094, segment 8 boundary) |
| **Mean ULP** | 0.01 |
| **ULP ≤ 1** | 99.78% of values |
| **ULP > 100** | 0% of values |

### Per-Segment ULP Analysis

| Segment | Range | Count | Mean ULP | Max ULP | Worst x |
|---------|-------|------:|----------:|--------:|--------:|
| Deep neg (FTZ) | x < -13.2 | 15,916 | 0.00 | 0 | N/A |
| Deep neg (asymp) | [-13.2, -5.5] | 163 | 2.78 | 7 | -5.969 |
| Seg 7 | [-5.5, -5.095] | 13 | 0.92 | 3 | -5.219 |
| **Seg 8** | **[-5.095, -4.136]** | 31 | **4.26** | **46** | **-5.094** |
| Seg 9 | [-4.136, -3.177] | 57 | 0.39 | 3 | -4.125 |
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
| **OVERALL** | | **65,024** | **0.01** | **46** | |

### Cumulative Distribution

| ULP ≤ | Count | Percent |
|------:|------:|--------:|
| 0 | 64,593 | 99.34% |
| 1 | 64,879 | 99.78% |
| 3 | 64,969 | 99.92% |
| 7 | 65,011 | 99.98% |
| 10 | 65,022 | 100.00% |
| 46 | 65,024 | 100.00% |

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

### Fix 3: Adaptive Polynomial (Best Accuracy) ✅ IMPLEMENTED

Replace single Chebyshev with segmented adaptive polynomial:

```cpp
// 9 segments with optimized coefficients (C6 segments 7-15)
// See: https://github.com/ivoitovych/bf16_gelu_research/blob/main/adaptive_poly.cpp
```

**Result:** Max ULP = 46 (research shows Max ULP = 1 is theoretically achievable with all 16 segments)

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
- [x] Implemented C6 adaptive polynomial with 9 segments

### Phase 4: Validation
- [x] Run full BF16 sweep (65,024 non-denormal values)
- [x] Verify Max ULP is acceptable: **Max ULP = 46** (at segment boundary)
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
| `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp` | ✅ Added | C++ test suite (18 tests: 10 ULP calculator + 8 device) |
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
- Asymptotic expansion for deep negative: `GELU(x) ≈ -exp(-x²/2) / √(2π)` for -13 < x < -5.5
- Deep negative cutoff: `GELU(x) = 0` for x < -13 (values below BF16 denormal minimum)
- 9 adaptive polynomial segments (segments 7-15 from C6) covering [-5.5, 3.0]
- Binary-search-like conditional structure for efficient segment selection

**Final Results (DAZ+FTZ Model):**

| Region | Before Fix (incorrect model) | After Fix (DAZ+FTZ) | Notes |
|--------|------------------------------|---------------------|-------|
| FTZ Region (x < -13.2) | 32,767 | 0 | Both expected and actual are 0 |
| Asymptotic (-13.2 to -5.5) | 32,767 | **7** | Asymptotic expansion works |
| Polynomial Segments | 1,475 | **46** | Seg 8 boundary worst case |
| Near-Zero (Taylor) | 14,276 | **1** | Taylor series accurate |
| Positive Saturation | 0 | **1** | Identity function |

**Hardware FTZ Behavior:**
- For x < -13.2: exp(-x²/2) produces denormals that hardware flushes to 0
- Both expected and actual values are 0, so ULP = 0 (correct behavior)
- This is not a bug - it's the correct result under DAZ+FTZ model

**Sample Results by Region (DAZ+FTZ model):**

Deep Negative (Asymptotic):
```
x=-5.969:  Max ULP in asymptotic region = 7
x=-6.0:    ULP ≤ 7 (asymptotic expansion)
x=-10.0:   ULP ≤ 3
x=-13.0:   ULP ≤ 2 (boundary of FTZ region)
```

Polynomial Segments:
```
x=-5.094:  Max ULP = 46 (segment 8 boundary - worst case)
x=-4.125:  Max ULP = 3 (segment 9)
x=-2.984:  Max ULP = 1 (segment 10)
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
- C++ tests: 18 passed (10 BFloat16UlpTest + 8 GeluUlpBugTest)
- Python tests: 27 passed (test_gelu_floor_value_bug.py)
- All tests verify fix works (low ULP) rather than asserting old buggy behavior

**Final Verification:**
- Comprehensive BF16 sweep: 65,024 non-denormal values tested
- Max ULP: 46 (at segment 8 boundary x = -5.094)
- Mean ULP: 0.01
- 99.78% of values have ULP ≤ 1

---

## References

- **GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/35290
- **Research Repository:** https://github.com/ivoitovych/bf16_gelu_research
- **Bug Report Branch:** https://github.com/ivoitovych/tt-metal/tree/ivoitovych/bug-report-gelu-floor-value-ulp-03
