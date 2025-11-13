# Root Cause Analysis: SSE Random Generation Test Crashes

## Executive Summary

**Root Cause:** Use-after-scope bug in `create_chunks()` function due to dangling reference capture in lambda.

**Impact:** Causes segmentation fault in all SSE random generation tests.

**Fix:** Change lambda capture from `[&]` (by-reference) to `[output]` (by-value).

**Severity:** Critical - causes 100% crash rate.

## Detailed Analysis

### The Bug

**File:** `/workspace/tt-metal/tt-train/sources/ttml/core/random_sse.hpp`
**Lines:** 74-82

```cpp
template <typename T>
inline auto create_chunks(std::span<T> output, size_t num_threads, size_t chunk_size) noexcept {
    return std::views::iota(0u, num_threads) | std::views::transform([&](size_t i) {  // ❌ BUG: [&] captures by reference
               const size_t offset = i * chunk_size;
               const size_t size = std::min(chunk_size, output.size() - offset);
               return std::span{output.data() + offset, size};
           }) |
           std::views::take_while([](auto chunk) { return !chunk.empty(); });
}
```

### Why It Crashes

1. **Function call:** `generate_uniform_simd_parallel()` calls `create_chunks(output, num_threads, chunk_size)` at line 344

2. **Lazy view creation:** The function returns a lazy `std::views` pipeline that captures `output` **by reference** using `[&]`

3. **Function returns:** `create_chunks()` returns, and the `output` parameter (a local variable) goes out of scope

4. **View materialization:** The for-loop at line 350 materializes the view:
   ```cpp
   for (auto chunk : chunks) {  // View is materialized HERE
   ```

5. **Dangling reference:** The lambda tries to access `output` which no longer exists → **undefined behavior** → segmentation fault

### Why It's Not Caught by Compilers

- The bug is in a template function with lazy evaluation
- The reference becomes dangling across function boundaries
- No static analyzer warnings because `std::span` itself is valid (it's copied), but what it points to isn't

### Execution Flow

```
generate_uniform_simd_parallel()
  ├─> create_chunks(output, ...)  // output is a std::span parameter
  │     └─> returns views::transform([&] { ... })  // [&] captures output by reference
  │
  ├─> create_chunks() returns  // output parameter goes out of scope
  │
  └─> for (auto chunk : chunks)  // View materialization
        └─> Lambda executes: accesses dangling output reference
              └─> SEGFAULT
```

## The Fix

### Change Required

**File:** `tt-train/sources/ttml/core/random_sse.hpp`
**Line:** 76

**Before (WRONG):**
```cpp
return std::views::iota(0u, num_threads) | std::views::transform([&](size_t i) {
```

**After (CORRECT):**
```cpp
return std::views::iota(0u, num_threads) | std::views::transform([output](size_t i) {
```

### Why This Fixes It

- Capturing `[output]` (by-value) copies the `std::span` object itself
- `std::span` is a lightweight view (just pointer + size), so copying is cheap
- The copied `std::span` remains valid after `create_chunks()` returns
- The lambda can safely access the span when the view is materialized

### Verification

The fix should be applied and tested with:

```bash
cd /workspace/tt-metal
export TT_METAL_RUNTIME_ROOT=/workspace/tt-metal

# Test 1: First crashing test
./tt-train/build/tests/ttml_tests --gtest_filter="RandomGenerationTests.ParallelUniformInitDeterminismSSE"

# Test 2: Second crashing test
./tt-train/build/tests/ttml_tests --gtest_filter="RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE"

# Test 3: Both SSE tests together
./tt-train/build/tests/ttml_tests --gtest_filter="RandomGenerationTests.*SSE"
```

**Expected result after fix:** All tests PASS ✅

## Additional Notes

### Why Basic SSE Test Worked

A minimal test with just SSE operations works fine because:
- No lazy views involved
- No multi-threading
- No dangling references

### Why Non-SSE Tests Pass

The non-SSE tests (`ttml::core::legacy::parallel_generate`) use a different implementation that doesn't have this bug.

### Similar Bugs in Codebase

The same pattern appears in:
- Line 437: `generate_normal_simd_parallel()` - uses same `create_chunks()`
- Line 501: `generate_uniform_simd_parallel_bfloat16()` - uses same `create_chunks()`

**All uses of `create_chunks()` are affected.** Fixing the function fixes all callsites.

## Impact Assessment

**Before Fix:**
- 2 SSE tests crash: 100% failure rate
- Blocks 215 tests in full test suite
- Issue #32247 remains open

**After Fix:**
- All SSE tests should pass
- Full test suite can run without exclusions
- Issue #32247 can be closed

## References

- **GitHub Issue:** #32247
- **Test File:** `tt-train/tests/core/random_tests.cpp`
- **Implementation:** `tt-train/sources/ttml/core/random_sse.hpp`
- **C++ Standard:** Capturing by-reference in lambdas with lazy views is a common pitfall
