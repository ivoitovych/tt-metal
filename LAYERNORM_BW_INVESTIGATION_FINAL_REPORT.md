# LayerNorm Backward Bug Investigation - Final Report

**GitHub Issue:** #34625
**Investigation Period:** Multiple sessions spanning several weeks
**Status:** DEFERRED - Requires Tenstorrent engineering expertise
**Hardware:** Blackhole P150

---

## Executive Summary

The LayerNorm backward kernel for tt-train has a bug affecting small tensors (width < ~2048 features on Blackhole) that use the L1 code path. After extensive investigation spanning 30+ debugging sessions, the root cause has been identified as a **DST register accumulation issue** in the `matmul_tiles` operation, but all attempted software fixes have failed. The bug requires Tenstorrent hardware/LLK engineering expertise to resolve.

**Current State:**
- 9 of 19 tests passing (large tensors that use the non-L1 path)
- 10 tests failing (small tensors using L1 path, plus tight tolerance tests)
- A workaround exists (using `Wt >= 16` threshold for preloading approach) that makes large tensor tests pass

---

## Problem Description

### Symptoms

When running LayerNorm backward on small tensors, the computed `dx` gradient has an error of exactly 1000.0 when using deterministic test inputs with `rstd=1000`. This indicates that `mean(dy*gamma)` is being computed as **2.0 instead of 1.0**.

### Test Results Summary

| Test Category | Result | Notes |
|---------------|--------|-------|
| Small tensors (20-100 features) | FAIL | Error = 1000 |
| Medium tensors (256-1024 features) | FAIL | Error = 1000 |
| Large tensors (2048+ features) | PASS | Uses different code path |
| Tight tolerance tests | FAIL | Same root cause |

### Root Cause Analysis

The bug is in the `compute_dy_gamma_sum()` function in the `EVERYTHING_FITS_IN_L1` code path:

```cpp
// First block: accumulate dy*gamma
tile_regs_acquire();
// ... accumulation loop puts sum (1.0) into register 0 ...
tile_regs_commit();
pack_and_push(sum_register, cb_scaled_dy_gamma_sum_idx);

// Second block: reduce with matmul
tile_regs_acquire();  // BUG: Register 0 still contains 1.0!
// ...
matmul_tiles(..., reduced_sum_register);  // ACCUMULATES: 1.0 + 1.0 = 2.0
```

**The fundamental issue:** `matmul_tiles` always accumulates to DST (DST += result), and after `tile_regs_acquire()`, register 0 still contains the previous value (1.0). The matmul then produces 1.0 (correct result) but adds it to the stale 1.0, giving 2.0.

---

## Fix Attempts Summary

### Attempts That Failed

| # | Approach | Result |
|---|----------|--------|
| 1 | `zero_dst_reg(0)` before matmul | No effect |
| 2 | Use different register (6) | No effect |
| 3 | Use different register with zeroing | No effect |
| 4 | `mm_init_short` instead of `mm_init` | Made things worse (varying errors 90-752) |
| 5 | Explicit `add_binary_tile` accumulation pattern | Much worse (errors 30000-62000) |
| 6 | `copy_tile` + matmul to fresh register | Broke previously passing tests |
| 7 | Proper tile_regs synchronization (`tile_regs_wait`/`tile_regs_release`) | **Device hung** |

### Key Technical Findings

1. **`matmul_tiles` ALWAYS accumulates** - There is no overwrite variant
2. **`tile_regs_acquire()`/`tile_regs_release()` don't clear registers** - Values persist
3. **`zero_dst_reg()` doesn't help** - The register gets overwritten but matmul still accumulates
4. **Forward layernorm uses `reduce_tile` API instead of matmul** - Different approach entirely
5. **Adding proper synchronization causes device hang** - Suggests deeper hardware/timing issue

### Forward Kernel Comparison

The forward layernorm kernel (ttnn) uses a completely different pattern:
- Uses `reduce_tile<REDUCE_ROW>` instead of `matmul_tiles`
- Uses `tile_regs_wait()` and `tile_regs_release()` after every pack
- Has proper synchronization between MATH and PACK threads

Attempting to port this synchronization pattern to the backward kernel caused device hangs.

---

## Technical Details

### Files Involved

- **Kernel:** `tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp`
- **Test:** `tt-train/tests/ops/layernorm_bw_fused_op_test.cpp`
- **API:** `tt_metal/include/compute_kernel_api/matmul.h`
- **Forward reference:** `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp`

### L1 Path Threshold

On Blackhole P150, the L1 path is used when:
- Total tensor data fits in L1 (~1.5MB available)
- Threshold is approximately 64 tiles (~2048 features)

### LayerNorm Backward Formula

```
dx = rstd * (dy*gamma - mean(dy*gamma) - x_hat * mean(dy*gamma*x_hat))
```

For deterministic tests with variance=0, rstd=1000, x_hat=0:
```
dx = rstd * (1.0 - mean(dy*gamma))
dx = 1000 * (1.0 - 2.0)  // BUG: mean is 2.0 instead of 1.0
dx = -1000               // Error = 1000
```

---

## Recommendations for Tenstorrent Engineering

### Option 1: Fix DST Register Accumulation Behavior

Investigate why:
- `zero_dst_reg()` doesn't prevent matmul accumulation
- `tile_regs_release()` followed by `tile_regs_acquire()` doesn't reset register state
- Adding proper synchronization causes device hang

### Option 2: Use `reduce_tile` API Instead of `matmul_tiles`

The forward kernel successfully uses `reduce_tile<REDUCE_ROW>` for mean computation. This API:
- Uses dedicated pooling instructions (TTI_GAPOOL)
- Has different accumulation semantics
- Requires broadcasting result back to full tile width

### Option 3: Restructure Kernel to Avoid Register Reuse

Redesign the kernel so that:
- First accumulation block and matmul reduction use completely separate DST regions
- Or process in a single `tile_regs_acquire/release` block
- Or use SFPU for the scalar multiplication

### Questions for Hardware Team

1. Why does `zero_dst_reg()` (which uses `fill_tile`) not prevent matmul accumulation?
2. What is the correct sequence to reset DST register state between operations?
3. Why does adding `tile_regs_wait()`/`tile_regs_release()` cause device hang?
4. Is there a matmul variant that overwrites instead of accumulates?

---

## Workaround (Currently Implemented)

For large tensors (Wt >= 16), a preloading approach is used that avoids the bug:

```cpp
constexpr uint32_t WT_THRESHOLD = 16;

if constexpr (Wt >= WT_THRESHOLD) {
    // Pre-load xnorm_sum into a dedicated register ONCE
    // This avoids repeated CB reads that trigger the bug
    const uint32_t xnorm_sum_register = current_block_size + 1;
    copy_tile(cb_scaled_dy_gamma_xnorm_sum_idx, 0, xnorm_sum_register);
    // ... use preloaded value ...
} else {
    // Direct CB read approach (STILL BUGGY for small tensors)
    mul_tiles(cb_x_hat_idx, cb_scaled_dy_gamma_xnorm_sum_idx, ...);
}
```

This makes large tensor tests pass but small tensors still fail.

---

## Appendix: Session History

- **Sessions 1-10:** Initial investigation, identified matmul accumulation as root cause
- **Sessions 11-20:** Attempted various register manipulation fixes
- **Sessions 21-25:** Investigated forward kernel patterns, tried reduce_tile approach
- **Sessions 26-30:** Attempted synchronization fixes, all caused device hangs
- **Session 31:** Final attempt with proper `tile_regs_wait/release` - device hung, investigation concluded

---

## Conclusion

This bug requires hardware/LLK expertise beyond what can be resolved through kernel-level software changes. The matmul_tiles accumulation behavior and DST register state management have undocumented interactions that cause all attempted fixes to either have no effect or hang the device.

**Recommended action:** File internal Tenstorrent ticket with this report for hardware engineering review.

---

*Report generated: January 2026*
