# PR Review: GELU ULP Fix for Issue #35290

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix-draft-pr-prep`  
**Merge Base:** `50b633663b48e5dabc2f9ddc32ceb28c0a11c873`  
**Review Date:** 2026-01-12

---

## Executive Summary

This PR implements a fix for the GELU precision bug reported in issue #35290. The fix replaces the original Chebyshev polynomial implementation with a C6 adaptive polynomial approach that achieves **Max ULP = 7** across the entire BF16 range.

### Key Results

| Metric | Value |
|--------|-------|
| **Max ULP** | 7 (at x = -5.969, asymptotic region) |
| **Mean ULP** | 0.01 |
| **ULP ≤ 1** | 99.80% of values |
| **ULP > 100** | 0% of values |

---

## Changes Summary

| File | Lines Changed | Status |
|------|---------------|--------|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | +246/-28 | Modified |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | +246/-28 | Modified |
| `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp` | +1406 | New |
| `tests/ttnn/unit_tests/gtests/CMakeLists.txt` | +7 | Modified |
| `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py` | +385 | New |
| `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md` | +317 | New |
| `GELU_ULP_FIX_IMPLEMENTATION.md` | +477 | New |
| `GELU_BF16_Zero_Saturation_Threshold_Research.md` | +494 | New |

**Total: 8 files changed, 3,522 insertions(+), 56 deletions(-)**

---

## Detailed Review

### ✅ GOOD - Should Keep

#### 1. Core Kernel Implementation

**Files:**
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`

**Assessment:** ✅ Excellent

**Strengths:**
- Well-documented with clear comments explaining the segment architecture
- Achieves Max ULP = 7 across entire BF16 range (excellent accuracy)
- Proper handling of DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero) hardware behavior
- Clean segmented polynomial approach with:
  - Taylor series for near-zero (|x| < 0.125): `GELU(x) ≈ x * (0.5 + 0.3989*x)`
  - Asymptotic expansion for deep negative (-13.2 < x < -5.5): `GELU(x) ≈ -exp(-x²/2) / √(2π)`
  - Raw x polynomials for transition region [-5.5, -3.177] with 8 segments
  - Normalized u polynomials for segments 10-15 [-3.177, 3.0]
  - Positive saturation for x >= 3.0: `GELU(x) = x`
- Efficient binary-search-like conditional structure for segment selection
- Both Wormhole and Blackhole implementations are synchronized

**Implementation Architecture:**
```
Region                    | Range           | Method                  | Max ULP
--------------------------|-----------------|-------------------------|--------
FTZ (hardware limit)      | x < -13.2       | Returns 0               | 0
Asymptotic                | [-13.2, -5.5]   | exp(-x²/2) / √(2π)      | 7
Range A (4 segments)      | [-5.5, -5.095]  | Raw x polynomials       | 1
Range B (2 segments)      | [-5.095, -4.136]| Raw x polynomials       | 1
Range C (2 segments)      | [-4.136, -3.177]| Raw x polynomials       | 1
Segments 10-15            | [-3.177, 3.0]   | Normalized u polynomials| 1
Positive saturation       | x >= 3.0        | Identity (returns x)    | 1
Near-zero (Taylor)        | |x| < 0.125     | Taylor series           | 1
```

---

#### 2. C++ Test Suite

**File:** `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp`

**Assessment:** ✅ Very Good

**Strengths:**
- Comprehensive BFloat16 ULP calculator with verification tests (10 tests)
- MPFR 256-bit precision reference implementation (correct approach to avoid fp64 erf() saturation)
- Device tests covering all regions (11 tests)
- Proper DAZ+FTZ model matching hardware behavior
- Good test naming and organization

**Test Categories:**
1. **ULP Calculator Verification (No Device Required):**
   - `ZeroesHaveSameIndex`
   - `ZeroesHaveUlpDistanceZero`
   - `AdjacentPositiveValuesHaveUlpOne`
   - `AdjacentNegativeValuesHaveUlpOne`
   - `SmallestNormalToZeroIsOne`
   - `DenormalsMapToZero`
   - `CrossZeroDistanceWithNormals`
   - `MaxUlpDistanceWithDAZ`
   - `VerifyIndexMonotonicity`
   - `AdjacentNormalValuesHaveUlpOneOrZeroExceptAtZero`

2. **GELU Device Tests:**
   - `DeepNegativeTailLowULP`
   - `NearZeroLowULP`
   - `TransitionRegionLowULP`
   - `FTZBoundaryVerification`
   - `DebugWorstCases`
   - `ComprehensiveULPBySegment`
   - `CumulativeULPDistribution`
   - `SummaryStatistics`
   - `SubnormalOutputsFlushedToZero`
   - `MonotonicityVerification`
   - `DenormalInputsProduceSameOutputAsZero`

---

#### 3. Python Test Suite

**File:** `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py`

**Assessment:** ✅ Good

**Strengths:**
- Uses mpmath for 256-bit precision reference (correct approach)
- Clean parametrized test structure
- Good coverage of all three regions
- Comprehensive summary test with detailed logging

**Test Classes:**
- `TestGeluDeepNegativeTail` - 7 parametrized tests
- `TestGeluNearZero` - 10 parametrized tests
- `TestGeluTransitionRegion` - 9 parametrized tests
- `test_gelu_ulp_summary` - Comprehensive summary function

---

#### 4. CMakeLists.txt Changes

**File:** `tests/ttnn/unit_tests/gtests/CMakeLists.txt`

**Assessment:** ✅ Good

**Changes:**
- Added `test_gelu_ulp_bug.cpp` to `unit_tests_ttnn_basic` target
- Added MPFR and GMP library linking for high-precision reference

**Concern:** Need to verify MPFR/GMP availability in CI environment or make dependency optional with graceful fallback.

---

### ⚠️ QUESTIONABLE - Needs Cleanup Before PR

#### 1. Root-Level Documentation Files

**Files:**
- `GELU_ULP_FIX_IMPLEMENTATION.md` (477 lines)
- `GELU_BF16_Zero_Saturation_Threshold_Research.md` (494 lines)

**Assessment:** ❌ Should NOT be included in PR

**Problems:**
1. **Location:** Documentation files should NOT be in the repository root for a production PR
2. **Internal details:** Contains progress log entries with specific dates
3. **Branch references:** Contains internal branch names (`ivoitovych/issue-35290-gelu-ulp-fix-02`, etc.)
4. **Too verbose:** 971 combined lines of documentation is excessive for the codebase
5. **Personal references:** References to personal fork branches and local file paths

**Recommendation:** 
- **REMOVE from PR**
- Move key technical content to:
  - PR description
  - GitHub issue comments
  - Code comments (already done well)
  - External research repository (already exists at https://github.com/ivoitovych/bf16_gelu_research)

---

#### 2. Bug Report in Tests Directory

**File:** `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md`

**Assessment:** ❌ Should NOT be included in PR

**Problems:**
1. **Incorrect data:** File admits at the top that ULP statistics were "calculated using an incorrect model"
2. **Confusing:** Having a "bug report" with incorrect data checked into the codebase is misleading
3. **Redundant:** Bug report belongs in GitHub issue #35290, not in the codebase
4. **Outdated:** Contains original analysis that has been superseded by DAZ+FTZ corrected analysis

**Recommendation:** **REMOVE from PR** - Bug reports belong in issue tracker, not codebase.

---

### ❌ SHOULD FIX - Code Issues

#### 1. Dead Code: Unused POLYVAL15 Macro

**Location:** `ckernel_sfpu_gelu.h` lines 227-245

```cpp
// Legacy Chebyshev implementation (kept for reference/comparison)
#define POLYVAL15(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0, x)  \
    ...
```

**Problem:** This macro is declared but never used anywhere in the codebase.

**Recommendation:** **REMOVE** - Dead code should not be committed. If needed for reference, it's available in git history.

---

#### 2. Unnecessary Wrapper Function

**Location:** `ckernel_sfpu_gelu.h` lines 247-250

```cpp
inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    // Use C6 adaptive polynomial implementation
    return calculate_gelu_c6(val);
}
```

**Problem:** This function is a pointless wrapper that just calls `calculate_gelu_c6()`. It adds:
- Unnecessary indirection
- Confusing naming (it's not Chebyshev anymore)
- Maintenance burden

**Recommendation:** 
- If `calculate_gelu_chebyshev` is called externally, rename all call sites to use `calculate_gelu_c6` directly
- If not called externally, **REMOVE** the wrapper

---

#### 3. Long-Running Comprehensive Tests

**Concern:** Several tests iterate over all 65,024+ BF16 values:
- `ComprehensiveULPBySegment`
- `CumulativeULPDistribution`
- `MonotonicityVerification`
- `DenormalInputsProduceSameOutputAsZero`
- `SubnormalOutputsFlushedToZero`

**Problem:** These could take significant time in CI and may not be appropriate for every PR run.

**Recommendation:**
1. Mark these tests with appropriate tags/categories (e.g., `SLOW`, `NIGHTLY`)
2. Consider sampling instead of exhaustive sweep for regular CI
3. Keep exhaustive tests for nightly/weekly runs
4. Add timeout handling

---

#### 4. MPFR Dependency

**File:** `tests/ttnn/unit_tests/gtests/CMakeLists.txt`

```cmake
find_library(MPFR_LIBRARY mpfr)
find_library(GMP_LIBRARY gmp)
target_link_libraries(unit_tests_ttnn_basic PRIVATE ... ${MPFR_LIBRARY} ${GMP_LIBRARY})
```

**Concern:** MPFR/GMP may not be available in all CI environments.

**Recommendations:**
1. Add check for library availability:
   ```cmake
   if(MPFR_LIBRARY AND GMP_LIBRARY)
       target_compile_definitions(unit_tests_ttnn_basic PRIVATE HAVE_MPFR)
       target_link_libraries(unit_tests_ttnn_basic PRIVATE ${MPFR_LIBRARY} ${GMP_LIBRARY})
   endif()
   ```
2. Add fallback in test code with `#ifdef HAVE_MPFR`
3. Or ensure MPFR is installed in CI environment

---

## Recommended Actions Before PR

### Files to KEEP (with modifications):

| File | Action |
|------|--------|
| `ckernel_sfpu_gelu.h` (wormhole) | Remove dead code (POLYVAL15 macro, wrapper function) |
| `ckernel_sfpu_gelu.h` (blackhole) | Remove dead code (POLYVAL15 macro, wrapper function) |
| `test_gelu_ulp_bug.cpp` | Consider marking slow tests; keep all functionality |
| `CMakeLists.txt` | Add MPFR availability check |
| `test_gelu_floor_value_bug.py` | Keep as-is |

### Files to REMOVE:

| File | Reason |
|------|--------|
| `GELU_ULP_FIX_IMPLEMENTATION.md` | Internal documentation; move to PR description |
| `GELU_BF16_Zero_Saturation_Threshold_Research.md` | Research doc; keep in external repo |
| `GELU_FLOOR_VALUE_BUG_REPORT.md` | Contains incorrect data; belongs in issue tracker |

---

## Code Cleanup Checklist

- [ ] Remove `POLYVAL15` macro from both kernel files
- [ ] Remove `calculate_gelu_chebyshev` wrapper function from both kernel files
- [ ] Delete `GELU_ULP_FIX_IMPLEMENTATION.md` from repository root
- [ ] Delete `GELU_BF16_Zero_Saturation_Threshold_Research.md` from repository root
- [ ] Delete `GELU_FLOOR_VALUE_BUG_REPORT.md` from tests directory
- [ ] Add MPFR availability check to CMakeLists.txt (optional)
- [ ] Mark slow tests appropriately (optional)
- [ ] Update PR description with key results and technical summary

---

## Suggested PR Description

```markdown
## Summary

Fixes #35290 - GELU precision bug in accurate mode (non-approximate).

## Problem

The original Chebyshev polynomial GELU implementation had accuracy issues:
1. **Deep negative tail (x < -5.5):** Returned 0.0 instead of tiny negative values
2. **Near-zero (|x| < 0.125):** c0 coefficient dominated, giving floor value ~2.98e-05
3. **Transition region:** Poor polynomial fit at -5.5 boundary

## Solution

Replaced Chebyshev with C6 adaptive polynomial implementation:
- Taylor series for near-zero: `GELU(x) ≈ x * (0.5 + 0.3989*x)`
- Asymptotic expansion for deep negative: `GELU(x) ≈ -exp(-x²/2) / √(2π)`
- 14 optimized polynomial segments covering [-5.5, 3.0]
- Proper FTZ handling for x < -13.2

## Results

| Metric | Before | After |
|--------|--------|-------|
| Max ULP | 32,767 | **7** |
| Mean ULP | 3,266 | **0.01** |
| ULP ≤ 1 | 52.5% | **99.80%** |

## Testing

- 21 C++ tests (10 ULP calculator + 11 device tests)
- 27 Python tests covering all regions
- Comprehensive BF16 sweep (65,024 values)

## References

- Research: https://github.com/ivoitovych/bf16_gelu_research
- Hardware model: DAZ+FTZ per `tech_reports/Handling_Special_Value/special_values.md`
```

---

## Final Assessment

| Category | Score | Notes |
|----------|-------|-------|
| **Kernel Fix Quality** | ⭐⭐⭐⭐⭐ | Excellent implementation |
| **Test Coverage** | ⭐⭐⭐⭐⭐ | Very thorough |
| **Code Cleanliness** | ⭐⭐⭐☆☆ | Dead code needs removal |
| **Documentation** | ⭐⭐☆☆☆ | Too much in wrong places |
| **PR Readiness** | **70%** | Needs cleanup before submission |

**Bottom Line:** The core fix is excellent and well-tested. The PR needs cleanup to remove internal documentation and dead code before public submission.

---

*Review prepared for branch `ivoitovych/issue-35290-gelu-ulp-fix-draft-pr-prep`*
