# PR #37166 Revision: Inherit compiler from tt-metal's CMakeCache.txt

This revision addresses all reviewer feedback from @blozano-tt and @philei-tt.

## Files Modified (3 files, all in `tt-train/`)

### 1. `tt-train/cmake/compilers.cmake` — Complete rewrite (119 lines)

| Old (PR v1) | New (this revision) |
|---|---|
| `FIND_AND_SET_CLANG20()` — hardcoded clang-20 search | `INHERIT_COMPILER_FROM_TT_METAL()` — reads compiler from tt-metal's CMakeCache.txt |
| `CHECK_COMPILERS()` — rejects GCC entirely | `CHECK_COMPILERS()` — accepts GCC 12+ (matches original pre-deletion code) |
| `ADJUST_COMPILER_WARNINGS()` — Clang-only flags | `ADJUST_COMPILER_WARNINGS()` — both Clang and GCC flags (matches original) |

### 2. `tt-train/CMakeLists.txt` — Simplified pre-`project()` block (net -2 lines)

Old: checked non-standard `$ENV{CMAKE_C_COMPILER}` env vars, then fell back to `find_and_set_clang20()`

New: single condition — if no compiler via command line, toolchain, or `CC`/`CXX` env vars, calls `inherit_compiler_from_tt_metal()`

### 3. `tt-train/build_all.sh` — Added explicit compiler flags (+3 lines)

Added `-DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20` so the standalone dev build script works without tt-metal being pre-built.

## Compiler Selection Priority

```
1. -DCMAKE_C_COMPILER=... on the cmake command line (or toolchain file)
2. CC/CXX environment variables
3. Inherited from tt-metal's CMakeCache.txt
4. CMake default search (CHECK_COMPILERS() validates the result)
```

## How Each Build Scenario Works

| Scenario | Compiler Source | How |
|---|---|---|
| **Subproject** (tt-metal parent) | Parent's toolchain | `CMAKE_C_COMPILER` already set → skip inheritance |
| **CI** (`pr-gate.yaml`) | `CC: clang-20` env var | `DEFINED ENV{CC}` → skip inheritance |
| **`pip install -e tt-train/`** | CMakeCache.txt | scikit-build-core isolates env vars, but filesystem is intact → directory inference `../` finds tt-metal → reads `build/CMakeCache.txt` |
| **`cmake -B build`** (dev) | CMakeCache.txt | Same as pip install — no env vars set, inherits from tt-metal |
| **`CC=gcc-12 cmake ...`** | User's env var | `DEFINED ENV{CC}` → skip inheritance, `CHECK_COMPILERS()` validates GCC 12 |
| **`build_all.sh`** | Explicit `-D` flags | Command-line flags take highest priority |

## Addresses All Reviewer Feedback

1. **@blozano-tt: non-standard env vars** → Removed. Now uses standard `CC`/`CXX` detection.
2. **@blozano-tt: why custom compiler search?** → Removed `FIND_AND_SET_CLANG20()`. Uses CMakeCache.txt inheritance instead.
3. **@blozano-tt: why Clang-only?** → Restored GCC 12+ support (matches original pre-deletion code).
4. **@philei-tt: why those warning flags?** → Restored from original pre-deletion code with both Clang and GCC paths.

## Graceful Degradation

`INHERIT_COMPILER_FROM_TT_METAL()` has 4 `return()` points — it never fatally errors. If tt-metal can't be found or CMakeCache.txt doesn't exist, it falls through to CMake's default compiler selection, and `CHECK_COMPILERS()` validates the result.

## Verification

Needs container build verification: build tt-metal, then run `pip install -e tt-train/` without `CC`/`CXX` env vars to confirm the CMakeCache.txt inheritance works.
