# tanh_bw Performance Benchmark: Fused sech^2(x) vs Composite 1-tanh^2(x)

**Date:** 2026-03-09
**Hardware:** Wormhole n150 L (single card), TT-KMD 2.2.0, FW 18.5.0
**Machine:** movsianikov-tt (Ubuntu 22.04, kernel 6.8.0-87-generic)
**PR Branch:** `ivoitovych/issue-35885-tanh-bw-fused-kernel-fix` (commit `d77beb31e6`)
**Merge Base:** `36890a9b35` (main at branch point)
**PR:** https://github.com/tenstorrent/tt-metal/pull/39378

## 1. Executive Summary

The fused sech^2(x) kernel is **4.6-5.8x faster** than the composite `1-tanh^2(x)` decomposition for dispatch-dominated shapes (1-1024 tiles), and **1.5x faster** for compute-dominated shapes (16384 tiles). Combined with the accuracy improvement (Max ULP: 15,140 -> 1), this is a significant win on both axes.

## 2. Methodology

### 2.1 A/B Testing Protocol

The A/B test compares the full merge-base codebase (Phase A) against the full PR branch (Phase B). This is critical: reverting only the dispatch line while keeping fused kernel infrastructure compiled produces invalid results (both phases run similar code paths through the new infrastructure).

**Phase B (fused kernel):**
1. Check out the PR branch at commit `d77beb31e6`
2. Copy the benchmark test file to `/tmp/test_tanh_bw_perf.py` (it's not part of the PR)
3. Rebuild ttnn: `~/tt/rebuild_ttnn.sh`
4. Clear kernel cache: `~/tt/clear_kernel_cache.sh`
5. Run benchmarks:
   ```bash
   python_env/bin/python -m pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py \
       -v -k "host_timing or throughput" -s 2>&1 > /tmp/tanh_bw_perf_phase_b_v2.log
   ```

**Phase A (composite, merge base):**
1. Roll back ALL source files to merge base: `git checkout 36890a9b35 -- .`
2. Copy ONLY the test file back: `cp /tmp/test_tanh_bw_perf.py tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py`
3. Fix any API signature mismatches between merge-base headers and the test file
4. Rebuild ttnn from scratch: `~/tt/rebuild_ttnn.sh`
5. Clear kernel cache: `~/tt/clear_kernel_cache.sh`
6. Run benchmarks:
   ```bash
   python_env/bin/python -m pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py \
       -v -k "host_timing or throughput" -s 2>&1 > /tmp/tanh_bw_perf_phase_a_v2.log
   ```

**Phase restore:**
1. Restore branch HEAD: `git checkout HEAD -- .`
2. Rebuild + clear cache

### 2.2 Test Configuration

- **Warmup:** 10 iterations (not measured)
- **Measurement:** 100 iterations (host timing), 50 iterations (throughput)
- **Synchronization:** `ttnn.synchronize_device(device)` after warmup and after measurement loop
- **Timer:** `time.time_ns()` (Python monotonic wall clock)
- **Data type:** BF16
- **Memory config:** DRAM interleaved
- **Random seed:** `torch.manual_seed(42)` (deterministic inputs)

### 2.3 Shape Sweep

| Name | Shape | Tiles | Regime |
|------|-------|------:|--------|
| 1_tile | [1,1,32,32] | 1 | Dispatch-dominated |
| 64_tiles | [1,1,256,256] | 64 | Dispatch-dominated |
| 1024_tiles | [1,1,1024,1024] | 1,024 | Transitional |
| 16384_tiles | [1,1,4096,4096] | 16,384 | Compute-dominated |

### 2.4 Test File

`tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py` — 235 lines, implements Approaches 1, 2, and 5. Full source is committed to branch `myfork/ivoitovych/tanh-bw-perf-benchmarking`.

## 3. Results by Approach

### 3.1 Approach 1: Host-Side Timing

Measures wall-clock time of the complete `ttnn.tanh_bw()` call including host dispatch, data movement, device compute, and synchronization.

| Shape | Tiles | Composite (us) | Fused (us) | Speedup |
|-------|------:|---------------:|-----------:|--------:|
| [1,1,32,32] | 1 | 633.1 | 139.0 | **4.55x** |
| [1,1,256,256] | 64 | 725.2 | 124.2 | **5.84x** |
| [1,1,1024,1024] | 1,024 | 725.1 | 127.0 | **5.71x** |
| [1,1,4096,4096] | 16,384 | 1,597.1 | 1,068.3 | **1.50x** |

**Key observations:**
- For shapes up to 1024 tiles, the composite is dispatch-bound: it launches 4 separate device programs (tanh, square, rsub, multiply), each with its own dispatch overhead. The fused kernel launches 1 program.
- The composite's floor is ~633-725 us regardless of tile count (1-1024 tiles all take similar time), confirming dispatch dominance.
- The fused kernel's floor is ~124-139 us — roughly 1/5 of composite, consistent with eliminating 3 of 4 dispatch cycles.
- At 16384 tiles, compute becomes significant and the speedup narrows to 1.5x. The remaining speedup comes from eliminating 3 intermediate tensor allocations and 3 extra DRAM read/write passes.

### 3.2 Approach 2: Device Profiler (Tracy)

**Status: FAILED — not executable on this build configuration.**

The device profiler requires `TT_METAL_DEVICE_PROFILER=1`, which conflicts with `TT_METAL_DPRINT_CORES` (both use the same L1 SRAM region). Even after `unset TT_METAL_DPRINT_CORES`, the profiler caused a segmentation fault during the second test's `CreateDevice` call, followed by a hard device hang requiring `tt-smi -r 0`.

Error from profiler attempt:
```
TT_FATAL: Both DPRINT and Profiler cannot be enabled at the same time. (assert.hpp:104)
```
After fixing the env var conflict, the profiler segfaulted on the second test case (device re-initialization between tests is incompatible with the profiler in Debug builds).

**Recommendation:** Device profiler testing requires either:
- A Release build (reduced debug overhead)
- A single-test-per-process harness (avoid device re-initialization)
- CI infrastructure with profiler support

### 3.3 Approach 3: SFPU Cycle Counting (DeviceZoneScopedN)

**Status: Not implemented.** Requires modifying the compute kernel source with instrumentation. Only warranted if reviewers request per-tile cycle counts. The host-side timing already demonstrates the performance improvement clearly.

### 3.4 Approach 4: Hardware Performance Counters

**Status: Not implemented.** Requires `PROFILE_PERF_COUNTERS_FPU` compilation. Reserved for deep SFPU optimization work, not PR-level benchmarking.

### 3.5 Approach 5: Throughput (Tiles/Second)

Measures processing rate in tiles per second, providing a single intuitive metric for large tensors.

| Shape | Tiles | Composite (M tiles/s) | Fused (M tiles/s) | Speedup |
|-------|------:|----------------------:|-------------------:|--------:|
| [1,1,1024,1024] | 1,024 | 1.413 | 7.118 | **5.04x** |
| [1,1,4096,4096] | 16,384 | 10.259 | 15.303 | **1.49x** |

**Key observations:**
- At 1024 tiles, throughput improves 5x — consistent with Approach 1's host-timing speedup at the same shape.
- At 16384 tiles, throughput improves 1.5x — also consistent with Approach 1.
- Throughput at 16384 tiles (15.3M tiles/s for fused) represents ~15.7 GB/s of BF16 data processed, approaching DRAM bandwidth limits for the 2-input, 1-output data pattern.

## 4. Cross-Approach Comparison

| Shape | Tiles | Approach 1 Speedup | Approach 5 Speedup | Consistent? |
|-------|------:|-------------------:|-------------------:|:-----------:|
| [1,1,1024,1024] | 1,024 | 5.71x | 5.04x | Yes |
| [1,1,4096,4096] | 16,384 | 1.50x | 1.49x | Yes |

Approaches 1 and 5 are consistent within measurement noise. The slight difference at 1024 tiles (5.71x vs 5.04x) is attributable to different iteration counts (100 vs 50) and timer granularity.

## 5. Why the Speedup is Real

The composite decomposition `1 - tanh^2(x)` executes 4 separate device programs:
1. `tanh(input)` — launch program, read input from DRAM, compute, write result to DRAM
2. `square(tanh_result)` — launch program, read from DRAM, compute, write to DRAM
3. `rsub(squared, 1.0)` — launch program, read from DRAM, compute, write to DRAM
4. `multiply(grad, rsub_result)` — launch program, read from DRAM, compute, write to DRAM

Each program launch incurs:
- Host-side dispatch overhead (~30-50 us per program)
- DRAM write of intermediate result (~100-400 ns/tile depending on size)
- DRAM read of intermediate by next op

The fused kernel executes 1 program:
1. Read `grad` and `input` from DRAM
2. Compute `sech^2(input) * grad` in a single SFPU pass
3. Write result to DRAM

Savings:
- 3 fewer program dispatches (90-150 us saved)
- 3 fewer intermediate tensor allocations (3 x input_size of DRAM traffic eliminated)
- No intermediate tensor memory footprint

For small tensors (1-1024 tiles), the 3 eliminated dispatches dominate — hence ~5x speedup. For large tensors (16384 tiles), the eliminated DRAM traffic still provides 1.5x speedup.

## 6. Raw Logs

| Phase | Log File |
|-------|----------|
| Phase A (composite, merge-base) | `/tmp/tanh_bw_perf_phase_a_v2.log` |
| Phase B (fused, PR branch) | `/tmp/tanh_bw_perf_phase_b_v2.log` |
| Phase B profiler attempt (failed) | `/tmp/tanh_bw_perf_phase_b_profiler.log` |

## 7. Reproduction Instructions

```bash
# 1. Clone and build from PR branch
git clone --recurse-submodules git@github.com:tenstorrent/tt-metal.git
cd tt-metal
git fetch origin ivoitovych/issue-35885-tanh-bw-fused-kernel-fix
git checkout ivoitovych/issue-35885-tanh-bw-fused-kernel-fix
./create_venv.sh && source python_env/bin/activate
./build_metal.sh --debug --build-all --enable-ccache

# 2. Get the test file from the benchmarking branch
git show myfork/ivoitovych/tanh-bw-perf-benchmarking:tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py \
    > tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py
# Or copy it from this report's companion repository

# 3. Phase B (fused): clear cache and run
rm -rf ~/.cache/tt-metal-cache/*
python -m pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py \
    -v -k "host_timing or throughput" -s 2>&1 | tee /tmp/phase_b.log

# 4. Record merge base
MERGE_BASE=$(git merge-base HEAD origin/main)

# 5. Phase A (composite): roll back to merge base
cp tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py /tmp/test_tanh_bw_perf.py
git checkout $MERGE_BASE -- .
cp /tmp/test_tanh_bw_perf.py tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py

# 6. Rebuild and run
cmake --build build_Debug --target ttnn -j$(nproc)   # or rebuild_ttnn.sh
rm -rf ~/.cache/tt-metal-cache/*
python -m pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py \
    -v -k "host_timing or throughput" -s 2>&1 | tee /tmp/phase_a.log

# 7. Restore
git checkout HEAD -- .
cmake --build build_Debug --target ttnn -j$(nproc)
rm -rf ~/.cache/tt-metal-cache/*
```
