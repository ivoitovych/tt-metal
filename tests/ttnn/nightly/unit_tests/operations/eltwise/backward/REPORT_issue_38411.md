# Reproduction Report: Issue #38411

**Issue:** [experimental::gelu_bw compute_program_hash ignores approximate parameter, causing program cache collision](https://github.com/tenstorrent/tt-metal/issues/38411)

**Tested at:** commit `004809c831` (tt-metal main, 2026-02-24)
**Hardware:** Blackhole N150
**Original report:** commit `b1668cd560`

## Finding: The hash collision does not reproduce at `004809c831`

### 1. Hash mechanism analysis

`compute_program_hash` passes `args` (type `GeluBackwardParams`) to `hash_operation`, which calls `hash_objects_with_default_seed`. The hashing code in `tt_stl/reflection.hpp:1251-1257` checks if the type satisfies `Reflectable`:

```cpp
} else if constexpr (ttsl::concepts::Reflectable<T>) {
    hash_t hash = 0;
    reflect::for_each([&hash, &object](auto I) {
        hash = hash_objects(hash, reflect::get<I>(object));
    }, object);
    return hash;
}
```

`GeluBackwardParams` is an aggregate struct (`tt_stl/concepts.hpp:15` — `std::is_aggregate_v` requirement), so `Reflectable` is satisfied. All three fields — `output_dtype`, `output_memory_config`, **and `approximate`** — are hashed via `reflect::for_each`. The hash already distinguishes between different `approximate` values without any code change.

Adding `args.approximate` explicitly as proposed in the issue would be redundant (double-hashing the same field).

### 2. Test results — "none" vs "tanh"

Ran both modes on the same device (shared program cache), same input:

```
PASSED test_gelu_bw_cache_collision_none_vs_tanh[1,1,32,32]
PASSED test_gelu_bw_cache_collision_none_vs_tanh[1,1,320,384]
```

Outputs are distinct. No cache collision. This holds **with or without** the proposed `args.approximate` addition — confirming the hash already works.

### 3. Test results — "none" vs "poly"

```
FAILED test_gelu_bw_cache_collision_none_vs_poly[1,1,32,32]
FAILED test_gelu_bw_cache_collision_none_vs_poly[1,1,320,384]
```

Outputs are bitwise-identical, but this is **not a hash collision**. The program factory (`gelu_backward_program_factory.cpp:97-100`) is a binary choice:

```cpp
args.approximate == "tanh" ? "...eltwise_bw_gelu_approx_tanh.cpp"
                           : "...eltwise_bw_gelu_approx_none.cpp"
```

There is no `"poly"` branch and no `eltwise_bw_gelu_poly.cpp` kernel file. Passing `approximate="poly"` falls through to the else branch and runs the `"none"` kernel. The hash correctly produces a different value for `"poly"` vs `"none"`, so a cache miss occurs and `create()` is called — but `create()` itself selects the same kernel for both.

### 4. Golden comparison

Each existing mode matches its PyTorch reference within BF16 tolerances (atol=0.2, rtol=0.05):

```
PASSED test_gelu_bw_approximate_golden[none, 1,1,32,32]
PASSED test_gelu_bw_approximate_golden[none, 1,1,320,384]
PASSED test_gelu_bw_approximate_golden[tanh, 1,1,32,32]
PASSED test_gelu_bw_approximate_golden[tanh, 1,1,320,384]
```

### 5. Existing test suite

All 48 existing `test_backward_gelu_fused.py` tests pass.

## Conclusion

| Claim in issue | Status at `004809c831` |
|---|---|
| `compute_program_hash` ignores `approximate` | **False** — `Reflectable` hashes all struct fields |
| "none" and "tanh" produce identical outputs | **False** — outputs are distinct, no cache collision |
| "none" and "poly" produce identical outputs | **True** — but caused by missing kernel, not hash collision |
| `eltwise_bw_gelu_poly.cpp` should exist | **Does not exist** at this commit |

The issue as described (hash collision) does not exist at `004809c831`. The "none" vs "poly" identical output is caused by the program factory having no `"poly"` code path — this is a missing feature, not a caching bug. The poly kernel implementation belongs to PR #36366.

**Recommendation:** Close #38411 as not-a-bug (at current main), or reclassify as a feature request for the poly kernel dispatch path, tracked under PR #36366.

## Test artifact

Regression test: `test_backward_gelu_fused_cache_collision.py` in the same directory as this report. Validates "none" vs "tanh" cache separation and mode-specific golden correctness.
