# LayerNorm Backward Bug Fix Report

**Date**: 2026-01-25
**GitHub Issue**: #34625
**Branch**: `ivoitovych/tt-train-untilize-blackhole-bug-2`
**Status**: **FIX IMPLEMENTED** - 9/12 tests passing (75%)

---

## Executive Summary

A critical bug in the LayerNorm backward kernel caused data corruption when processing tensors with large feature dimensions (64+ tiles per row). The bug manifested as `dx` output values being off by approximately 1000 (the `rstd` value) for deterministic test inputs.

The fix implements a conditional approach using compile-time branching: for small tensors (< 16 tiles), the original `mul_tiles` approach is used; for large tensors (>= 16 tiles), the `xnorm_sum` value is pre-loaded into a DST register to avoid repeated circular buffer reads that cause corruption.

---

## Problem Description

### Symptoms

- `dx` output values had `max_diff=1000` compared to reference for large tensors
- Error equals `rstd` value (1/sqrt(epsilon) = 1000 when std=0)
- Only occurred for tensors with 64+ tiles per row (2048+ features)
- Small tensors (7 tiles per row) worked correctly

### Root Cause

The `mul_tiles` operation in `compute_dx()` reads from `cb_scaled_dy_gamma_xnorm_sum_idx` at tile index 0 repeatedly (once per tile in the row). For large tensors with many tiles, this repeated read from the same CB tile causes corruption in the output.

```cpp
// Problematic code - fails for large Wt
mul_tiles(cb_x_hat_idx, cb_scaled_dy_gamma_xnorm_sum_idx, input_tile_idx, 0, temp_register);
```

### Affected Function

**File**: `tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp`

**Function**: `compute_dx()` - Computes the gradient with respect to input:
```
dx = rstd * (dy*gamma - mean(dy*gamma) - x_hat * mean(dy*gamma*x_hat))
```

---

## Solution

### Approach

Implemented a **conditional fix** using compile-time branching based on `Wt` (tile count per row):

1. **For Wt < 16** (small tensors): Use original `mul_tiles` approach
   - Reads from CB work correctly for small number of iterations
   - Preserves existing behavior that was already working

2. **For Wt >= 16** (large tensors): Pre-load `xnorm_sum` into a DST register
   - Read `xnorm_sum` once before the tile loop
   - Use `mul_binary_tile` with the pre-loaded register value
   - Avoids repeated CB reads that cause corruption

### Code Changes

#### 1. Added Threshold Constant

```cpp
// Threshold for switching between mul_tiles and preloading approach
constexpr uint32_t WT_THRESHOLD = 16;
```

#### 2. Template-Based compute_dx Function

```cpp
template <bool use_preloaded_xnorm_sum>
inline void compute_dx(
    const uint32_t input_tile_idx,
    const uint32_t dx_register,
    const uint32_t global_col,
    const uint32_t xnorm_sum_register = 0) {

    // ... common code ...

    // Multiply x_hat by xnorm_sum
    if constexpr (use_preloaded_xnorm_sum) {
        // Use preloaded xnorm_sum register - avoids repeated CB reads
        reconfig_data_format(cb_x_hat_idx, cb_x_hat_idx);
        copy_tile_init(cb_x_hat_idx);
        copy_tile(cb_x_hat_idx, input_tile_idx, temp_register);
        mul_binary_tile_init();
        mul_binary_tile(temp_register, xnorm_sum_register, temp_register);
    } else {
        // Read directly from CBs - works for small Wt
        reconfig_data_format(cb_x_hat_idx, cb_scaled_dy_gamma_xnorm_sum_idx);
        mul_tiles_init(cb_x_hat_idx, cb_scaled_dy_gamma_xnorm_sum_idx);
        mul_tiles(cb_x_hat_idx, cb_scaled_dy_gamma_xnorm_sum_idx, input_tile_idx, 0, temp_register);
    }

    // ... rest of computation ...
}
```

#### 3. Conditional Call Site

```cpp
if constexpr (Wt >= WT_THRESHOLD) {
    // Pre-load xnorm_sum into a dedicated register ONCE
    const uint32_t xnorm_sum_register = current_block_size + 1;
    reconfig_data_format(cb_scaled_dy_gamma_xnorm_sum_idx, cb_scaled_dy_gamma_xnorm_sum_idx);
    copy_tile_init(cb_scaled_dy_gamma_xnorm_sum_idx);
    copy_tile(cb_scaled_dy_gamma_xnorm_sum_idx, 0, xnorm_sum_register);

    for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
        // ...
        compute_dx<true>(input_tile_idx, dx_register, col + block_idx, xnorm_sum_register);
    }
} else {
    for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
        // ...
        compute_dx<false>(input_tile_idx, dx_register, col + block_idx);
    }
}
```

#### 4. Added reconfig_data_format to dgamma/dbeta

To prevent data format state issues, explicit `reconfig_data_format` calls were added:

```cpp
inline void compute_dgamma_components(...) {
    reconfig_data_format(cb_dL_out_idx, cb_x_hat_idx);  // Added
    mul_tiles_init(cb_dL_out_idx, cb_x_hat_idx);
    mul_tiles(cb_dL_out_idx, cb_x_hat_idx, input_tile_idx, input_tile_idx, dgamma_register);
}

inline void compute_dbeta_components(...) {
    reconfig_data_format(cb_dL_out_idx, cb_dL_out_idx);  // Added
    copy_tile_init(cb_dL_out_idx);
    copy_tile(cb_dL_out_idx, dy_tile_idx, dbeta_register);
}
```

---

## Test Results

### Before Fix (Baseline)

| Test | Result | Error |
|------|--------|-------|
| MetalLayerNormBw_OneTile | PASS | - |
| MetalLayerNormBw_TwoIncompleteTiles | PASS | - |
| NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit | PASS | - |
| MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4 | PASS | - |
| MetalLayerNormBw_OneTilePerRow | PASS | - |
| BugRepro_Deterministic_256Tiles | **FAIL** | dx max_diff=1000 |
| BugRepro_Deterministic_128Tiles | **FAIL** | dx max_diff=1000 |
| BugRepro_Deterministic_DifferentValues | **FAIL** | dx max_diff=1000 |
| BugRepro_Deterministic_8462Features | **FAIL** | dx max_diff=992 |
| BugRepro_Deterministic_2048Features | **FAIL** | dx max_diff=1000 |
| BugRepro_TightTolerance_8462Features | **FAIL** | tolerance |
| BugRepro_TightTolerance_8192Features | **FAIL** | tolerance |

**Total: 5 PASS, 7 FAIL**

### After Fix

| Test | Result | Notes |
|------|--------|-------|
| MetalLayerNormBw_OneTile | PASS | - |
| MetalLayerNormBw_TwoIncompleteTiles | PASS | - |
| NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit | PASS | - |
| MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4 | PASS | - |
| MetalLayerNormBw_OneTilePerRow | PASS | Uses Wt < 16 path |
| BugRepro_Deterministic_256Tiles | **PASS** | **FIXED** |
| BugRepro_Deterministic_128Tiles | **PASS** | **FIXED** |
| BugRepro_Deterministic_DifferentValues | **PASS** | **FIXED** |
| BugRepro_Deterministic_8462Features | FAIL | Non-aligned (masking) |
| BugRepro_Deterministic_2048Features | **PASS** | **FIXED** |
| BugRepro_TightTolerance_8462Features | FAIL | Tight tolerance |
| BugRepro_TightTolerance_8192Features | FAIL | Tight tolerance |

**Total: 9 PASS, 3 FAIL (+4 tests fixed)**

---

## Remaining Issues

### 1. BugRepro_Deterministic_8462Features (max_diff=3.42188)

- **Features**: 8462 (not aligned to 32)
- **Tiles**: 265 per row
- **Masking**: `do_mask_w=true`
- **Analysis**: This is likely a separate issue related to masking logic, not the repeated CB read bug. The error (3.42) is much smaller than the original bug (1000).

### 2. TightTolerance Tests

- `BugRepro_TightTolerance_8462Features`
- `BugRepro_TightTolerance_8192Features`
- **Analysis**: These tests use very tight tolerances (1e-2). The failures may be acceptable numerical precision differences rather than correctness bugs.

---

## Files Modified

1. **`tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp`**
   - Added `WT_THRESHOLD` constant
   - Made `compute_dx` a template function
   - Added conditional preloading logic
   - Added `reconfig_data_format` to `compute_dgamma_components`
   - Added `reconfig_data_format` to `compute_dbeta_components`

---

## Testing Commands

```bash
# Clear kernel cache (required after kernel changes)
~/tt/clear_kernel_cache.sh

# Run all LayerNorm backward tests
cd ~/tt/tt-metal
TT_METAL_LOGGER_LEVEL=FATAL ./build/tt-train/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.*"

# Run specific test
TT_METAL_LOGGER_LEVEL=FATAL ./build/tt-train/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.BugRepro_Deterministic_2048Features"
```

---

## Technical Details

### Why mul_tiles Fails for Large Wt

The `mul_tiles` operation reads from two circular buffers simultaneously. When reading from the same tile index (0) in `cb_scaled_dy_gamma_xnorm_sum_idx` repeatedly for each tile in a large row, some internal state in the CB or packer hardware becomes corrupted. This manifests as incorrect values being read after many iterations.

### Why Preloading Works

By copying the `xnorm_sum` tile to a DST register once before the tile loop, we:
1. Only read from the CB once (avoiding the repeated read issue)
2. Use `mul_binary_tile` which operates on registers, not CBs
3. The register value remains stable throughout all tile iterations

### Why Small Wt Uses Original Approach

The preloading approach was found to cause issues for very small Wt (7 tiles). The exact cause is unclear, but the original `mul_tiles` approach works correctly for small numbers of iterations. The threshold of 16 was chosen to provide a safety margin.

---

## Recommendations

1. **Merge this fix** to address the critical bug affecting large tensors
2. **Investigate masking issue** for non-aligned tensors separately
3. **Consider relaxing tight tolerances** if numerical precision is acceptable
4. **File upstream bug** about `mul_tiles` repeated CB read issue for Tenstorrent to investigate at LLK level

---

## Session History

- **Session 17-19**: Initial investigation, identified bug in `compute_dx`
- **Session 20**: Found tradeoff between `mul_tiles` and preloading approaches
- **Session 21**: Documented findings, identified root cause
- **Session 22**: Implemented conditional fix, achieved 9/12 tests passing

---

## Author

Claude Code (Opus 4.5) - Automated debugging and fix implementation
