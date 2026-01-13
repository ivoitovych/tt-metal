# Branch Review: GELU ULP Fix Implementation

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix-draft-pr-prep`  
**Base:** `origin/main`  
**GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/35290  
**Review Date:** 2026-01-13

---

## Executive Summary

This branch implements a significant fix for the GELU (Gaussian Error Linear Unit) activation function in tt-metal's SFPU kernels. The fix replaces the original Chebyshev polynomial approximation with a new **C6 Adaptive Polynomial** implementation that achieves **Max ULP = 7** (down from thousands of ULP errors in specific regions). The implementation is well-researched, thoroughly tested, and includes comprehensive documentation.

### Overall Assessment: ✅ **RECOMMENDED FOR MERGE** (with minor suggestions)

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Code Quality** | ⭐⭐⭐⭐⭐ | Clean, well-documented kernel code |
| **Testing** | ⭐⭐⭐⭐⭐ | Comprehensive C++ and Python test suites |
| **Documentation** | ⭐⭐⭐⭐⭐ | Exceptional - includes research papers |
| **Risk** | ⭐⭐⭐⭐☆ | Low-medium; changes core SFPU kernel |
| **Performance** | ⭐⭐⭐⭐☆ | Needs benchmarking (more branches) |

---

## 1. Files Changed Summary

| File | Type | Lines | Description |
|------|------|------:|-------------|
| `GELU_BF16_Zero_Saturation_Threshold_Research.md` | Added | 497 | MPFR 256-bit research on BF16 zero threshold |
| `GELU_ULP_FIX_IMPLEMENTATION.md` | Added | 526 | Implementation guide and progress log |
| `tests/ttnn/unit_tests/gtests/CMakeLists.txt` | Modified | +5 | Added test file and comment |
| `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp` | Added | 1621 | Comprehensive C++ test suite |
| `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md` | Added | 318 | Bug report with reproduction steps |
| `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py` | Added | 387 | Python test suite |
| `tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_gelu.h` | Modified | +200 | Blackhole kernel implementation |
| `tt_metal/hw/ckernels/wormhole_b0/.../ckernel_sfpu_gelu.h` | Modified | +200 | Wormhole kernel implementation |

**Total:** 8 files changed, ~3,500 lines added

---

## 2. Technical Analysis

### 2.1 Problem Being Solved

The original GELU implementation had three problematic regions:

| Region | Problem | Original Max ULP | Fixed Max ULP |
|--------|---------|----------------:|-------------:|
| **Deep Negative (x < -5.5)** | Returned 0.0 for all values | 32,767* | 0-7 |
| **Near-Zero (|x| < 1e-4)** | c0 coefficient dominated | 14,276* | ≤1 |
| **Transition (-5.5 to -4.0)** | Poor polynomial fit | 1,475 | ≤1 |

*Note: Original ULP values were later corrected to account for DAZ+FTZ hardware model.

### 2.2 Solution Architecture

The new **C6 Adaptive Polynomial** implementation uses a segmented approach:

```
Input Range          | Strategy                    | Max ULP
---------------------|-----------------------------|---------
x < -13.2            | FTZ → 0 (hardware limit)    | 0
-13.2 < x < -5.5     | Asymptotic exp(-x²/2)       | 7
-5.5 < x < -3.177    | Raw x polynomials (8 segs)  | 1
-3.177 < x < 3.0     | Normalized u polynomials    | 1
|x| < 0.125          | Taylor series               | 1
x ≥ 3.0              | Identity (GELU(x) = x)      | 1
```

### 2.3 Key Implementation Details

#### 2.3.1 New Polynomial Macro

```cpp
// Degree-4 polynomial using Horner's method
#define POLY4(c0, c1, c2, c3, c4, u) \
    ((((c4) * (u) + (c3)) * (u) + (c2)) * (u) + (c1)) * (u) + (c0)
```

**Positive:** Efficient evaluation, clear implementation.

#### 2.3.2 Taylor Series for Near-Zero

```cpp
v_if(abs_val < 0.125f) { 
    result = val * (0.5f + INV_SQRT_2PI * val); 
}
```

**Positive:** Mathematically correct approximation for small x.

#### 2.3.3 Asymptotic Expansion

```cpp
v_elseif(val < -5.5f) {
    v_if(val < -13.2f) {
        result = 0.0f;  // FTZ hardware limitation
    }
    v_else {
        sfpi::vFloat x2 = val * val;
        sfpi::vFloat neg_half_x2 = x2 * sfpi::vFloat(-0.5f);
        sfpi::vFloat exp_val = _sfpu_exp_21f_<false>(neg_half_x2);
        result = exp_val * sfpi::vFloat(-0.3989422804f);
    }
    v_endif;
}
```

**Positive:** Correctly handles the deep negative tail using the Gaussian PDF approximation.

#### 2.3.4 Segment Selection

The code uses a binary-search-like conditional structure for efficient segment selection:

```cpp
v_if(val < -3.177f) {
    v_if(val < -5.095f) {
        // Range A: 4 segments
        v_if(val < -5.2975f) { ... }
        v_else { ... }
    }
    v_elseif(val < -4.136f) {
        // Range B: 2 segments
    }
    v_else {
        // Range C: 2 segments
    }
}
```

**Positive:** Well-organized branching, O(log n) segment lookup.

---

## 3. Test Coverage Analysis

### 3.1 C++ Test Suite (`test_gelu_ulp_bug.cpp`)

| Test Category | # Tests | Coverage |
|---------------|--------:|----------|
| BFloat16 ULP Calculator | 10 | Core ULP calculation verification |
| GELU Device Tests | 11 | Hardware execution validation |
| **Total** | 21 | Comprehensive |

**Key Tests:**
- `DeepNegativeTailLowULP` - Validates asymptotic region
- `NearZeroLowULP` - Validates Taylor series
- `TransitionRegionLowULP` - Validates polynomial segments
- `ComprehensiveULPBySegment` - Full BF16 sweep (65,024 values)
- `MonotonicityVerification` - Verifies GELU monotonicity
- `SubnormalOutputsFlushedToZero` - Verifies FTZ behavior
- `DenormalInputsProduceSameOutputAsZero` - Verifies DAZ behavior

**Positive:** Excellent coverage including edge cases and hardware behavior verification.

### 3.2 Python Test Suite (`test_gelu_floor_value_bug.py`)

| Test Class | # Tests | Coverage |
|------------|--------:|----------|
| `TestGeluDeepNegativeTail` | 7 | Deep negative region |
| `TestGeluNearZero` | 10 | Near-zero region |
| `TestGeluTransitionRegion` | 9 | Polynomial segments |
| `test_gelu_ulp_summary` | 1 | Comprehensive summary |
| **Total** | 27 | Good coverage |

**Positive:** Uses `mpmath` for 256-bit precision reference values.

### 3.3 Reference Implementation Quality

The test code includes a sophisticated reference GELU implementation:

```cpp
inline double gelu_exact(double x) {
    constexpr double SQRT2 = 1.4142135623730950488;
    if (x < 0.0) {
        // Uses erfc() to avoid erf() saturation at x ≈ -8.375
        double abs_x_div_sqrt2 = -x / SQRT2;
        return 0.5 * x * std::erfc(abs_x_div_sqrt2);
    } else {
        return 0.5 * x * (1.0 + std::erf(x / SQRT2));
    }
}
```

**Excellent:** The `erfc()` trick avoids fp64 `erf()` saturation and matches MPFR-256 with 0 ULP difference.

---

## 4. Documentation Quality

### 4.1 Research Documentation

`GELU_BF16_Zero_Saturation_Threshold_Research.md` is exceptional:

- ✅ Clear executive summary with key findings
- ✅ Mathematical background on GELU and BF16
- ✅ Detailed analysis of fp64 erf() limitation
- ✅ MPFR 256-bit verification code
- ✅ Complete results tables
- ✅ Implementation recommendations

**Key Finding:** The true BF16 zero threshold is **x = -13.1875**, not -8.375 as fp64 suggests.

### 4.2 Implementation Guide

`GELU_ULP_FIX_IMPLEMENTATION.md` provides:

- ✅ Root cause analysis of original bugs
- ✅ Per-segment ULP analysis tables
- ✅ Version history and progress log
- ✅ Hardware model explanation (DAZ+FTZ)
- ✅ Links to external research repository

### 4.3 Bug Report

`GELU_FLOOR_VALUE_BUG_REPORT.md` includes:

- ✅ Minimal reproduction code
- ✅ Original (incorrect) and corrected statistics
- ✅ Root cause code snippets
- ✅ Proposed fixes
- ✅ Hardware verification (Wormhole + Blackhole)

---

## 5. Code Quality Review

### 5.1 Kernel Code (`ckernel_sfpu_gelu.h`)

#### Strengths

1. **Clear documentation**: Each region is documented with its strategy and expected ULP.
2. **Consistent style**: Both Wormhole and Blackhole headers are identical.
3. **Dead code removed**: `POLYVAL15` macro and `calculate_gelu_chebyshev()` wrapper removed.
4. **Threshold explanation**: Comment explains why -13.2f was chosen over theoretical -13.1875.

#### Areas for Improvement

1. **Magic numbers**: Consider extracting segment boundaries to named constants:
   ```cpp
   // Current:
   v_if(val < -5.095f) { ... }
   
   // Suggested:
   constexpr float SEG_A_END = -5.095f;
   v_if(val < SEG_A_END) { ... }
   ```

2. **Polynomial coefficients**: Consider documenting how coefficients were derived:
   ```cpp
   // Segment A0: [-5.5, -5.39875)
   // Coefficients fitted via minimax optimization on GELU curve
   result = POLY4(0.000313728989568f, ...);
   ```

### 5.2 Test Code

#### Strengths

1. **DAZ+FTZ model**: Correctly implements Tenstorrent hardware behavior.
2. **Comprehensive verification**: Tests cover ULP calculator, device execution, and edge cases.
3. **Clear output**: Tests print detailed diagnostic information.

#### Areas for Improvement

1. **Test runtime**: `ComprehensiveULPBySegment` and `CumulativeULPDistribution` iterate over all 65,024 BF16 values. Consider:
   - Adding `DISABLED_` prefix for CI and enabling manually
   - Splitting into smaller parameterized tests

2. **Missing include**: The test uses `std::memcpy` but doesn't include `<cstring>`:
   ```cpp
   // Add at top of file:
   #include <cstring>
   ```

### 5.3 CMakeLists.txt Change

```cmake
# GELU reference uses fp64 erfc() - no external dependencies needed
# (Previously used MPFR but erfc() matches MPFR-256 with 0 ULP error)
```

**Positive:** Good documentation of why MPFR was removed.

---

## 6. Performance Considerations

### 6.1 Branch Complexity

The new implementation has more branches than the original:

| Implementation | Branches | Max Depth |
|----------------|----------|-----------|
| Original Chebyshev | 2 | 2 |
| New C6 Adaptive | ~15 | 5 |

**Impact:** On SFPU, branches may introduce divergence overhead. However:
- Binary-search structure minimizes average branch count
- Most inputs fall in polynomial segments (predictable)

### 6.2 Instruction Count

| Operation | Original | New |
|-----------|----------|-----|
| Polynomial evaluation | 15 terms | 4 terms × N segments |
| Special functions | None | exp() for deep negative |

**Recommendation:** Benchmark GELU throughput before/after on representative workloads.

### 6.3 Memory Footprint

- **Original:** 16 coefficients × 4 bytes = 64 bytes
- **New:** ~80 coefficients × 4 bytes = 320 bytes

**Impact:** Minimal - kernel code is small relative to available memory.

---

## 7. Risk Assessment

### 7.1 Regression Risk

| Risk | Severity | Mitigation |
|------|----------|------------|
| Accuracy regression in non-tested regions | Medium | Full BF16 sweep tests |
| Performance regression | Low | Benchmark before merge |
| Model accuracy impact | Medium | Run model validation suite |

### 7.2 Compatibility Risk

| Risk | Severity | Mitigation |
|------|----------|------------|
| Wormhole/Blackhole divergence | Low | Headers are identical |
| API changes | None | No API changes |

### 7.3 Hardware Model Assumptions

The code assumes DAZ+FTZ behavior per `tech_reports/Handling_Special_Value/special_values.md`. If hardware behavior differs:
- Tests would fail with ULP mismatches
- The -13.2f threshold might need adjustment

---

## 8. Suggestions for Improvement

### 8.1 High Priority (Before Merge)

1. **Add performance benchmark**: Measure GELU throughput impact.
   ```bash
   # Suggested benchmark script
   python -m pytest tests/ttnn/perf/test_gelu_perf.py -v
   ```

2. **Run model validation**: Verify no model accuracy regression.
   ```bash
   # Run BERT/GPT tests that use GELU
   pytest models/demos/bert/tests/ -v
   ```

### 8.2 Medium Priority (Can Be Done Later)

3. **Extract magic numbers**: Define segment boundaries as constants.

4. **Add `<cstring>` include**: For `std::memcpy` in test file.

5. **Mark slow tests**: Add timeout or disable comprehensive sweep tests in CI.

### 8.3 Low Priority (Nice to Have)

6. **Coefficient documentation**: Document how polynomial coefficients were derived.

7. **Performance mode**: Consider adding `FAST_GELU` macro for performance-critical paths.

---

## 9. Questions for Author

1. **Performance impact**: Has the GELU throughput been benchmarked? What's the expected overhead from additional branches?

2. **Model validation**: Have any models been tested end-to-end with the new implementation?

3. **Coefficient source**: Are the polynomial coefficients from the external research repo (`bf16_gelu_research`) or regenerated locally?

4. **Threshold margin**: Why was -13.2f chosen instead of -13.1875f exactly? Is the 0.0125 margin necessary?

---

## 10. Verdict

### ✅ **APPROVED FOR MERGE**

This is an exemplary bug fix implementation:

1. **Thorough research**: The root cause was carefully analyzed using MPFR 256-bit precision.
2. **Correct solution**: The C6 adaptive polynomial achieves excellent accuracy (Max ULP = 7).
3. **Comprehensive testing**: Both C++ and Python tests cover all edge cases.
4. **Excellent documentation**: Research papers, implementation guides, and bug reports are all included.
5. **Clean code**: Dead code removed, comments added, consistent style.

The only reservations are:
- Performance impact should be benchmarked
- Model validation should be run to ensure no downstream regressions

### Recommended Merge Checklist

- [ ] Benchmark GELU performance (throughput, latency)
- [ ] Run model validation suite (BERT, GPT, etc.)
- [ ] Add `<cstring>` include to test file
- [ ] Consider marking slow tests for manual execution
- [ ] Squash documentation commits for cleaner history

---

## Appendix: Commit History Analysis

The branch contains 19 unique commits for this feature:

| Commit | Description |
|--------|-------------|
| `781c39c37c` | Remove unused imports from Python GELU test |
| `0cf4248b68` | Pre-PR cleanup: remove dead code, add comments |
| `84e82fa04d` | Polish documentation: document erfc() solution |
| `6a99ab19fb` | Eliminate MPFR dependency: use fp64 erfc() |
| `0642f3721c` | Add FP64 vs MPFR-256 reference comparison test |
| `3494c2ef8c` | Use MPFR/mpmath for high-precision reference |
| `546c874729` | Add FTZ threshold research documentation |
| `a4330377b8` | Add raw x polynomial segments for Max ULP = 7 |
| `ca64b0220e` | Extend asymptotic expansion to eliminate Seg 8 error |
| `af717fe7f3` | Add hardware behavior verification tests |
| `828aa6455a` | Update documentation for DAZ+FTZ hardware model |

The commit history shows a well-organized development process with clear progression from research to implementation to testing.

---

*Review conducted by: Independent Reviewer*  
*Review methodology: Full code inspection, test analysis, documentation review*
