# Fix SSE random generation crash due to dangling reference in lazy view

### Ticket
Fixes #32247

### Problem description

**Crash:** Two SSE random generation tests crash with segmentation fault (SIGSEGV, exit code 139):
- `RandomGenerationTests.ParallelUniformInitDeterminismSSE`
- `RandomGenerationTests.UniformInitsGoodMeanAndRangeSSE`

The crashes occur deterministically in all scenarios - whether tests run individually, in pairs, or as part of the full test suite. The full test suite halts at test 21/236, preventing the remaining 215 tests from running.

**Root Cause:**
Use-after-scope bug in the `create_chunks()` helper function in `random_sse.hpp`. The function returns a lazy `std::views` pipeline that captures parameters by reference using `[&]`. When the view is materialized (in the for-loop), it accesses variables that have gone out of scope, resulting in undefined behavior and segmentation faults.

**Execution Flow Leading to Crash:**
```
generate_uniform_simd_parallel()
  ├─> create_chunks(output, num_threads, chunk_size)
  │     └─> returns views::transform([&] { ... })  // ❌ Captures by reference
  │
  ├─> create_chunks() returns  // Parameters go out of scope
  │
  └─> for (auto chunk : chunks)  // View materialization
        └─> Lambda executes: accesses dangling references
              └─> SEGFAULT
```

This is a classic C++20 footgun: lazy views with reference-capturing lambdas can create dangling references when the view is materialized after the creating function returns.

### What's changed

**Fix:** Changed lambda capture in `create_chunks()` from by-reference `[&]` to by-value `[output, chunk_size]`.

**Files Modified:**
- `tt-train/sources/ttml/core/random_sse.hpp` (line 76)
- 1 line changed: `[&]` → `[output, chunk_size]`

**Technical Details:**
- Capturing `output` (a `std::span`) by value is safe and efficient - it only copies the span object (pointer + size), not the underlying data
- Capturing `chunk_size` by value ensures the lambda has its own copy that remains valid
- The fix applies to all three callsites of `create_chunks()`:
  - `generate_uniform_simd_parallel()` (line 344)
  - `generate_normal_simd_parallel()` (line 437)
  - `generate_uniform_simd_parallel_bfloat16()` (line 501)

**Impact:**
- ✅ Fixes segmentation faults in SSE random generation tests
- ✅ All 4 RandomGenerationTests now pass (previously 2/4 crashed)
- ✅ Unblocks full test suite execution (previously blocked at test 21/236)
- ✅ No performance impact - std::span is lightweight to copy

**Testing:**
- Verified both previously-crashing tests now pass:
  - `ParallelUniformInitDeterminismSSE`: ✅ PASSED (5.8s)
  - `UniformInitsGoodMeanAndRangeSSE`: ✅ PASSED (28.9s)
- Verified all 4 RandomGenerationTests pass together: ✅ 100% pass rate
- Non-SSE tests continue to pass (unchanged)

**Before Fix:**
```
Running 4 tests from RandomGenerationTests
✅ ParallelUniformInitDeterminism - PASSED
✅ UniformInitsGoodMeanAndRange - PASSED
❌ ParallelUniformInitDeterminismSSE - CRASHED (exit 139)
❌ UniformInitsGoodMeanAndRangeSSE - CRASHED (exit 139)
```

**After Fix:**
```
Running 4 tests from RandomGenerationTests
✅ ParallelUniformInitDeterminism - PASSED (2.2s)
✅ UniformInitsGoodMeanAndRange - PASSED (10.3s)
✅ ParallelUniformInitDeterminismSSE - PASSED (5.8s)
✅ UniformInitsGoodMeanAndRangeSSE - PASSED (28.9s)
```

### Additional Context

**Why Basic SSE Operations Worked:**
Minimal SSE tests without threading or lazy views worked fine, which initially suggested the issue wasn't with SSE intrinsics or CPU support. The bug was specifically in the multi-threading coordination code using lazy views.

**Why Non-SSE Variants Passed:**
The non-SSE implementation (`ttml::core::legacy::parallel_generate`) uses a different approach without lazy views, so it wasn't affected by this bug.

**Detection Difficulty:**
This bug is particularly subtle because:
- Template functions with lazy evaluation defer execution
- The reference becomes dangling across function boundaries
- No static analyzer warnings (std::span itself is valid, but what it references isn't)
- Crashes are deterministic but occur in thread execution, making debugging harder

**C++ Standards Reference:**
This pattern (reference-capturing lambdas in lazy views) is a documented pitfall in C++20 ranges. The recommended practice is to capture by value or use explicit lifetime management.
