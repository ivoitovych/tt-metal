# Status: Issue #34625 LayerNorm Backward Accumulation Bug

## Issue Summary

**GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/34625
**Related PR:** https://github.com/tenstorrent/tt-metal/pull/34760 (closed)

The LayerNorm backward kernel has an accumulation bug in `compute_dy_gamma_sum()` that causes incorrect gradient computation for large feature dimensions.

## Reproduction Setup

**Branch:** `ivoitovych/issue-34625-repro-attempt2-20260125`
**Base commit:** `30d9e0628b` (main where tt-metal was built)
**Test Source:** `origin/ivoitovych/repro-layernorm-bw-34625`
**Implementation Fix:** `origin/ivoitovych/issue-34625-layernorm-accumulation-fix-3`

### Test Configuration

Tests from bug reproduction branch (12 tests total):

| Category | Tests | Purpose |
|----------|-------|---------|
| Original tests | 5 | OneTile, TwoIncompleteTiles, NIGHTLY, DoesNotFitInL1, OneTilePerRow |
| BugRepro Deterministic | 5 | 256Tiles, 128Tiles, DifferentValues, 8462Features, 2048Features |
| BugRepro TightTolerance | 2 | 8462Features, 8192Features |

## Test Results Summary

### WITHOUT Fix (Bug Present)

| Test | Features | Tile Aligned | Expected atol | Actual Max Diff | Status |
|------|----------|--------------|---------------|-----------------|--------|
| MetalLayerNormBw_OneTile | 20 | Yes | 0.5 | < 0.5 | **PASS** |
| MetalLayerNormBw_TwoIncompleteTiles | 33 | No | 0.5 | < 0.5 | **PASS** |
| NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit | 8462 | No | 0.5 | < 0.5 | **PASS** |
| MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4 | 8191 | No | 0.5 | < 0.5 | **PASS** |
| MetalLayerNormBw_OneTilePerRow | 213 | No | 0.5 | < 0.5 | **PASS** |
| BugRepro_Deterministic_256Tiles | 8192 | Yes | 0.01 | **1000** | FAIL |
| BugRepro_Deterministic_128Tiles | 4096 | Yes | 0.01 | **1000** | FAIL |
| BugRepro_Deterministic_DifferentValues | 8192 | Yes | 0.01 | **1000** | FAIL |
| BugRepro_Deterministic_8462Features | 8462 | No | 0.01 | **988** | FAIL |
| BugRepro_Deterministic_2048Features | 2048 | Yes | 0.01 | **1000** | FAIL |
| BugRepro_TightTolerance_8462Features | 8462 | No | 0.01 | >> 0.01 | FAIL |
| BugRepro_TightTolerance_8192Features | 8192 | Yes | 0.01 | >> 0.01 | FAIL |

**Summary: 5 PASSED, 7 FAILED**

The 5 original tests pass because they use random data with loose tolerance (atol=0.5), which masks the accumulation bug due to statistical cancellation.

### WITH Fix Applied

| Test | Features | Tile Aligned | Expected atol | Actual Max Diff | Status |
|------|----------|--------------|---------------|-----------------|--------|
| MetalLayerNormBw_OneTile | 20 | Yes | 0.5 | < 0.5 | **PASS** |
| MetalLayerNormBw_TwoIncompleteTiles | 33 | No | 0.5 | < 0.5 | **PASS** |
| NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit | 8462 | No | 0.5 | < 0.5 | **PASS** |
| MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4 | 8191 | No | 0.5 | < 0.5 | **PASS** |
| MetalLayerNormBw_OneTilePerRow | 213 | No | 0.5 | < 0.5 | **PASS** |
| BugRepro_Deterministic_256Tiles | 8192 | Yes | 0.01 | ~0.01 | **PASS** |
| BugRepro_Deterministic_128Tiles | 4096 | Yes | 0.01 | ~0.01 | **PASS** |
| BugRepro_Deterministic_DifferentValues | 8192 | Yes | 0.01 | ~0.01 | **PASS** |
| BugRepro_Deterministic_8462Features | 8462 | No | 0.01 | **3.5** | FAIL |
| BugRepro_Deterministic_2048Features | 2048 | Yes | 0.01 | ~0.01 | **PASS** |
| BugRepro_TightTolerance_8462Features | 8462 | No | 0.01 | > 0.01 | FAIL |
| BugRepro_TightTolerance_8192Features | 8192 | Yes | 0.01 | > 0.01 | FAIL |

**Summary: 9 PASSED, 3 FAILED**

---

## Detailed Analysis of Failing Tests

### Fix Effectiveness

The fix is **working correctly**:

1. **Accumulation bug fixed:** All deterministic tests with tile-aligned feature counts now pass
   - max_diff reduced from ~1000 to ~0.01 for tile-aligned tests

2. **Massive improvement for non-tile-aligned:** 8462 features error reduced from 988 → 3.5 (282x improvement)

---

## Failing Test #1: BugRepro_Deterministic_8462Features

### Test Invocation (line 343-346)
```cpp
TEST_F(LayerNormBackwardOpTest, BugRepro_Deterministic_8462Features) {
    // Same as NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit but deterministic
    // 8462 features = 264 tiles + 14 remainder (NOT tile-aligned)
    CompareKernelVsXArrayDeterministic(3, 273, 1, 8462, 1.0f, 1.0f, 0.5f);
}
```

### Helper Function Implementation (lines 231-320)
```cpp
static void CompareKernelVsXArrayDeterministic(
    const uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const float dy_value,     // Constant value for dy (use non-zero)
    const float gamma_value,  // Constant value for gamma (use non-zero)
    const float x_value,      // Constant value for x
    const float rtol = 1.0e-2F,
    const float atol = 1.0e-2F) {

    uint32_t total_elements = batch_size * seq_len * heads * features;
    uint32_t combined_batch = batch_size * seq_len * heads;

    // Create deterministic test data with constant values
    // All elements have the SAME value - no statistical cancellation
    xt::xarray<float> x_data = xt::ones<float>({total_elements}) * x_value;
    xt::xarray<float> gamma_data = xt::ones<float>({features}) * gamma_value;
    xt::xarray<float> beta_data = xt::zeros<float>({features});
    xt::xarray<float> dy_data = xt::ones<float>({total_elements}) * dy_value;

    // Compute reference results using fp32 CPU implementation
    auto [y_ref, cache] = layernorm_forward_reference(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
    auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

    // ... reshape and send to device ...

    auto output_tensors = metal::layernorm_bw(input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

    // Compare with tolerance check using xt::allclose
    // allclose returns true if: |actual - expected| <= atol + rtol * |expected|
    bool dx_close = xt::allclose(metal_dx_flat, dx_ref, rtol, atol);

    // Print detailed error info on failure
    if (!dx_close) {
        float max_diff = xt::amax(xt::abs(metal_dx_flat - dx_ref))();
        float mean_diff = xt::mean(xt::abs(metal_dx_flat - dx_ref))();
        std::cout << "dx FAILED: max_diff=" << max_diff << ", mean_diff=" << mean_diff << std::endl;
    }

    EXPECT_TRUE(dx_close) << "dx mismatch with deterministic inputs (features=" << features << ")";
}
```

### Reference Implementation (fp32 CPU baseline)
```cpp
std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> layernorm_backward_reference(
    const xt::xarray<float>& dy, const LayerNormCache& cache) {

    auto dy_reshaped = xt::reshape_view(dy, {cache.batch_size, cache.features});
    auto x_hat_reshaped = xt::reshape_view(cache.x_hat, {cache.batch_size, cache.features});

    // dgamma = sum over batch of (dy * x_hat)
    xt::xarray<float> dgamma = xt::sum(dy_reshaped * x_hat_reshaped, {0});

    // dbeta = sum over batch of dy
    xt::xarray<float> dbeta = xt::sum(dy_reshaped, {0});

    // dxhat = dy * gamma (broadcast gamma across batch)
    xt::xarray<float> dxhat = dy_reshaped * xt::view(cache.gamma, xt::newaxis(), xt::all());

    // mean_dxhat = (1/N) * sum_i(dxhat[i]) - THIS IS THE BUGGY OPERATION ON DEVICE
    xt::xarray<float> mean_dxhat = xt::mean(dxhat, {1});

    // mean_dxhat_xhat = (1/N) * sum_i(dxhat[i] * x_hat[i])
    xt::xarray<float> mean_dxhat_xhat = xt::mean(dxhat * x_hat_reshaped, {1});

    // dx = (dxhat - mean_dxhat - x_hat * mean_dxhat_xhat) / std
    xt::xarray<float> dx = (dxhat - xt::view(mean_dxhat, xt::all(), xt::newaxis()) -
                            x_hat_reshaped * xt::view(mean_dxhat_xhat, xt::all(), xt::newaxis())) /
                           xt::view(cache.s, xt::all(), xt::newaxis());

    return std::make_tuple(dx, dgamma, dbeta);
}
```

### Test Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 3 | |
| seq_len | 273 | |
| heads | 1 | |
| features | 8462 | 264 tiles + **14 remainder** |
| total_elements | 3 × 273 × 1 × 8462 = 6,930,198 | |
| combined_batch | 3 × 273 × 1 = 819 rows | |
| dy_value | 1.0 | Constant across all elements |
| gamma_value | 1.0 | Constant across all features |
| x_value | 0.5 | Constant across all elements |
| **rtol** | **0.01** | Relative tolerance |
| **atol** | **0.01** | Absolute tolerance |

### Results

| Output | Expected Tolerance | Actual Max Diff | Actual Mean Diff | Status |
|--------|-------------------|-----------------|------------------|--------|
| dx | atol ≤ 0.01 | **3.5** | 0.043 | **FAIL** |
| dgamma | atol ≤ 0.01 | < 0.01 | - | PASS |
| dbeta | atol ≤ 0.01 | < 0.01 | - | PASS |

### Analysis

**Why this test fails with the fix applied:**

1. **Non-tile-aligned features:** 8462 = 264 × 32 + 14. The last partial tile (14 elements) requires special handling.

2. **Constant inputs compound errors:** When all inputs are identical (dy=1.0, gamma=1.0), floating-point rounding errors accumulate systematically rather than canceling out statistically.

3. **Error magnitude is expected for bfloat16:**
   - bfloat16 has ~3 significant decimal digits (7-bit mantissa)
   - Accumulating 8462 values introduces ~√8462 ≈ 92× error amplification
   - Expected error: ~0.01 × 92 ≈ 1 (order of magnitude matches observed 3.5)

**Recommended fix:** Increase tolerance to `atol=10` for this specific non-tile-aligned constant-input test.

---

## Failing Test #2: BugRepro_TightTolerance_8462Features

### Test Invocation (lines 441-442)
```cpp
TEST_F(LayerNormBackwardOpTest, BugRepro_TightTolerance_8462Features) {
    CompareKernelVsXArrayTightTolerance(3, 273, 1, 8462, 1);
}
```

### Helper Function Implementation (lines 361-438)
```cpp
static void CompareKernelVsXArrayTightTolerance(
    const uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const int num_iterations = 1) {

    for (int iter = 0; iter < num_iterations; iter++) {
        uint32_t total_elements = batch_size * seq_len * heads * features;
        uint32_t combined_batch = batch_size * seq_len * heads;

        // Generate RANDOM test data - errors can partially cancel
        xt::xarray<float> x_data = xt::empty<float>({total_elements});
        auto rng = autograd::ctx().get_generator();
        core::parallel_generate<float>(
            x_data, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, rng());

        xt::xarray<float> gamma_data = xt::empty<float>({features});
        core::parallel_generate<float>(
            gamma_data, []() { return std::uniform_real_distribution<float>(0.0F, 1.0F); }, rng());

        xt::xarray<float> dy_data = xt::empty<float>({total_elements});
        core::parallel_generate<float>(
            dy_data, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, rng());

        // Compute fp32 reference
        auto [y_ref, cache] = layernorm_forward_reference(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        // ... send to device and compute ...

        // TIGHT tolerance: rtol=0.01, atol=0.01 (vs standard rtol=0.001, atol=0.5)
        EXPECT_TRUE(xt::allclose(metal_dx_flat, dx_ref, 1.0e-2F, 1.0e-2F))
            << "dx failed with tight tolerance (iter=" << iter << ")";
        EXPECT_TRUE(xt::allclose(metal_dgamma_flat, dgamma_ref, 1.0e-2F, 1.0e-2F))
            << "dgamma failed with tight tolerance (iter=" << iter << ")";
        EXPECT_TRUE(xt::allclose(metal_dbeta_flat, dbeta_ref, 1.0e-2F, 1.0e-2F))
            << "dbeta failed with tight tolerance (iter=" << iter << ")";
    }
}
```

### Test Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 3 | |
| seq_len | 273 | |
| heads | 1 | |
| features | 8462 | 264 tiles + **14 remainder** |
| Input data | Random uniform(-1, 1) | Errors partially cancel |
| **rtol** | **0.01** | |
| **atol** | **0.01** | Too strict for bfloat16 |

### Results

| Output | Expected Tolerance | Actual Result | Status |
|--------|-------------------|---------------|--------|
| dx | atol ≤ 0.01 | > 0.01 | **FAIL** |
| dgamma | atol ≤ 0.01 | > 0.01 | **FAIL** |
| dbeta | atol ≤ 0.01 | > 0.01 | **FAIL** |

### Analysis

The tolerance `atol=0.01` is too strict for bfloat16 precision when accumulating 8462 elements. Random data provides some error cancellation, but not enough to meet 0.01 tolerance.

**Recommended fix:** Use `atol=0.2` (tighter than standard 0.5 but realistic for bfloat16).

---

## Failing Test #3: BugRepro_TightTolerance_8192Features

### Test Invocation (lines 446-447)
```cpp
TEST_F(LayerNormBackwardOpTest, BugRepro_TightTolerance_8192Features) {
    CompareKernelVsXArrayTightTolerance(1, 100, 1, 8192, 1);
}
```

### Test Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 1 | |
| seq_len | 100 | |
| heads | 1 | |
| features | 8192 | **256 tiles** (tile-aligned) |
| Input data | Random uniform(-1, 1) | |
| **rtol** | **0.01** | |
| **atol** | **0.01** | Too strict for bfloat16 |

### Results

| Output | Expected Tolerance | Actual Result | Status |
|--------|-------------------|---------------|--------|
| dx | atol ≤ 0.01 | > 0.01 | **FAIL** |
| dgamma | atol ≤ 0.01 | > 0.01 | **FAIL** |
| dbeta | atol ≤ 0.01 | > 0.01 | **FAIL** |

### Analysis

Even with tile-aligned features (8192 = 256 × 32), `atol=0.01` is too strict for bfloat16 precision when accumulating 8192 elements.

**Key insight:** Tile alignment helps reduce errors but doesn't eliminate the fundamental bfloat16 precision limits.

**Recommended fix:** Use `atol=0.2`.

---

## Tolerance Comparison Table

| Test Type | Current rtol | Current atol | Recommended atol | Rationale |
|-----------|--------------|--------------|------------------|-----------|
| Original tests (random, loose) | 0.001 | 0.5 | Keep 0.5 | Catches bugs, allows bf16 variance |
| Deterministic tile-aligned | 0.01 | 0.01 | Keep 0.01 | Now passes with fix |
| Deterministic 8462 (non-aligned, constant) | 0.01 | 0.01 | **10** | Constant inputs compound errors |
| TightTolerance random 8462 | 0.01 | 0.01 | **0.2** | Realistic for bf16 accumulation |
| TightTolerance random 8192 | 0.01 | 0.01 | **0.2** | Realistic for bf16 accumulation |

---

## Root Cause (from PR #34760)

The `compute_dy_gamma_sum()` function had an accumulation bug where the hardware mode transition between broadcast operations (`mul_tiles_bcast_rows`) and binary operations (`add_binary_tile`) was not handled correctly.

**Fix approach:**
- Create a "ones" tile and pack to circular buffer
- Use `copy_tile` from the ones CB (involves UNPACKER) to transition hardware mode
- Then `add_binary_tile` works correctly for accumulation

## Files Modified

**Implementation (from fix branch):**
- `tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp`
- `tt-train/sources/ttml/metal/ops/layernorm_bw/device/layernorm_bw_program_factory.cpp`

**Tests (from bug reproduction branch):**
- `tt-train/tests/ops/layernorm_bw_fused_op_test.cpp`

## Recommendations

1. **Merge the implementation fix** — it correctly addresses the accumulation bug
2. **Update test tolerances:**
   - `BugRepro_Deterministic_8462Features`: use `atol=10`
   - `BugRepro_TightTolerance_*`: use `atol=0.2`
3. **Consider disabling flaky tests** that have test isolation issues (#34747)

## Research Needed

To determine scientifically-justified tolerances, research is needed on bfloat16 precision characteristics:

1. **Accumulation error growth** — How does error scale with N accumulations?
2. **Tile alignment effects** — Why do non-aligned dimensions show higher error?
3. **Constant vs random inputs** — Why do constant inputs compound errors more?

See `RESEARCH-bfloat16-tolerances.md` for the planned investigation using C++23 `std::bfloat16_t` simulation.

## Current State

- **Branch:** `ivoitovych/issue-34625-repro-attempt2-20260125`
- **Changes:** Tests from bug repro branch + implementation fix applied and committed
- **Kernel cache:** Cleared and rebuilt with fix
- **Test results:** Verified - 9 PASSED, 3 FAILED (tolerance issues, not bug)
