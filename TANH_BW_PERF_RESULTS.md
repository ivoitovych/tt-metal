# tanh_bw Performance Benchmark: Fused sech^2(x) vs Composite 1-tanh^2(x)

**Date:** 2026-03-09
**Hardware:** Wormhole n150 L (single card), TT-KMD 2.2.0, FW 18.5.0
**Machine:** movsianikov-tt (Ubuntu 22.04, kernel 6.8.0-87-generic)
**PR Branch:** `ivoitovych/issue-35885-tanh-bw-fused-kernel-fix` (commit `d83876ac12`)
**Merge Base:** `36890a9b35` (main at branch point)
**PR:** https://github.com/tenstorrent/tt-metal/pull/39378

## 1. Executive Summary

The fused sech^2(x) kernel is **4.6-5.8x faster** than the composite `1-tanh^2(x)` decomposition for dispatch-dominated shapes (1-1024 tiles), and **1.5x faster** for compute-dominated shapes (16384 tiles). Combined with the accuracy improvement (Max ULP: 15,140 -> 1), this is a significant win on both axes.

**Note:** These performance results were obtained on Wormhole n150 bare-metal. Performance characteristics may differ on other architectures (Blackhole, Grayskull) or virtualized environments. The accuracy improvement (Max ULP 15,140 -> 1) is architecture-independent.

## 2. Reproduction

Clone tt-metal from main, build it, fetch the [benchmark test file](https://github.com/ivoitovych/tt-metal/blob/ivoitovych/tanh-bw-perf-benchmarking/tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py), run Phase A (baseline on main), then merge the PR branch, rebuild, and run Phase B (fused kernel).

```bash
# 1. Clone tt-metal from main, with submodules and LFS
git clone --recurse-submodules git@github.com:tenstorrent/tt-metal.git
cd tt-metal
git lfs pull
git submodule foreach --recursive "git lfs pull"

# 2. Install system dependencies (requires sudo)
sudo ./install_dependencies.sh

# 3. Build tt-metal and create Python venv
./build_metal.sh --debug --build-all --enable-ccache
./create_venv.sh

# 4. Get the test file from the benchmarking branch (public HTTPS, no SSH keys needed)
git fetch https://github.com/ivoitovych/tt-metal.git ivoitovych/tanh-bw-perf-benchmarking
git show FETCH_HEAD:tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py \
    > tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py
# Note: test_tanh_bw_perf.py is an untracked file — it persists across
# git checkout/reset/merge operations and is available for both phases.

# 5. Phase A (baseline — composite implementation on main): clear cache and run
rm -rf ~/.cache/tt-metal-cache/*
source python_env/bin/activate
python -m pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py \
    -v -k "host_timing or throughput" -s 2>&1 | tee /tmp/phase_a.log

# 6. Merge PR branch (brings in the fused kernel) and rebuild
git fetch origin ivoitovych/issue-35885-tanh-bw-fused-kernel-fix
git merge FETCH_HEAD --no-edit
cmake --build build_Debug --target ttnn -j$(nproc)

# 7. Phase B (fused kernel): clear cache and run
rm -rf ~/.cache/tt-metal-cache/*
python -m pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py \
    -v -k "host_timing or throughput" -s 2>&1 | tee /tmp/phase_b.log
```

## 3. Methodology

**Prerequisite:** Run on bare-metal (not a VM). Performance benchmarks measure dispatch latency differences of ~30-150 us, which can be masked by hypervisor overhead in virtualized environments. Accuracy (ULP) tests are valid on VMs.

### 3.1 Test Configuration

- **Warmup:** 10 iterations (not measured)
- **Measurement:** 100 iterations (host timing), 50 iterations (throughput)
- **Synchronization:** `ttnn.synchronize_device(device)` after warmup and after measurement loop
- **Timer:** `time.time_ns()` (Python monotonic wall clock)
- **Data type:** BF16
- **Memory config:** DRAM interleaved
- **Random seed:** `torch.manual_seed(42)` (deterministic inputs)

### 3.2 Shape Sweep

| Name | Shape | Tiles | Regime |
|------|-------|------:|--------|
| 1_tile | [1,1,32,32] | 1 | Dispatch-dominated |
| 64_tiles | [1,1,256,256] | 64 | Dispatch-dominated |
| 1024_tiles | [1,1,1024,1024] | 1,024 | Transitional |
| 16384_tiles | [1,1,4096,4096] | 16,384 | Compute-dominated |

## 4. Results

### 4.1 Host-Side Timing

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

### 4.2 Throughput (Tiles/Second)

Measures processing rate in tiles per second, providing a single intuitive metric for large tensors.

| Shape | Tiles | Composite (M tiles/s) | Fused (M tiles/s) | Speedup |
|-------|------:|----------------------:|-------------------:|--------:|
| [1,1,1024,1024] | 1,024 | 1.413 | 7.118 | **5.04x** |
| [1,1,4096,4096] | 16,384 | 10.259 | 15.303 | **1.49x** |

**Key observations:**
- At 1024 tiles, throughput improves 5x — consistent with host-side timing speedup at the same shape.
- At 16384 tiles, throughput improves 1.5x — also consistent.
- Throughput at 16384 tiles (15.3M tiles/s for fused) represents ~15.7 GB/s of BF16 data processed, approaching DRAM bandwidth limits for the 2-input, 1-output data pattern.

### 4.3 Cross-Metric Comparison

| Shape | Tiles | Host Timing Speedup | Throughput Speedup | Consistent? |
|-------|------:|-------------------:|-------------------:|:-----------:|
| [1,1,1024,1024] | 1,024 | 5.71x | 5.04x | Yes |
| [1,1,4096,4096] | 16,384 | 1.50x | 1.49x | Yes |

Both metrics are consistent within measurement noise.

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

## 6. Rerun: Paranoid A/B with Actual Branch Checkout

To validate the results above, a second A/B run was performed using actual `git checkout` of the PR branch and the merge-base commit (instead of merging). Each phase: rebuild ttnn, clear kernel cache, get test file fresh from benchmark branch.

### 6.1 Host-Side Timing (Rerun)

| Shape | Tiles | Composite (us) | Fused (us) | Speedup |
|-------|------:|---------------:|-----------:|--------:|
| [1,1,32,32] | 1 | 681.8 | 138.9 | **4.91x** |
| [1,1,256,256] | 64 | 680.3 | 170.3 | **3.99x** |
| [1,1,1024,1024] | 1,024 | 691.7 | 139.4 | **4.96x** |
| [1,1,4096,4096] | 16,384 | 1,598.9 | 1,067.9 | **1.50x** |

### 6.2 Throughput (Rerun)

| Shape | Tiles | Composite (M tiles/s) | Fused (M tiles/s) | Speedup |
|-------|------:|----------------------:|-------------------:|--------:|
| [1,1,1024,1024] | 1,024 | 1.599 | 7.904 | **4.94x** |
| [1,1,4096,4096] | 16,384 | 10.202 | 15.303 | **1.50x** |

### 6.3 Cross-Run Comparison

| Shape | Tiles | Run 1 Speedup | Run 2 Speedup | Stable? |
|-------|------:|:-------------:|:-------------:|:-------:|
| [1,1,32,32] | 1 | 4.55x | 4.91x | Yes |
| [1,1,256,256] | 64 | 5.84x | 3.99x | Noisy |
| [1,1,1024,1024] | 1,024 | 5.71x | 4.96x | Yes |
| [1,1,4096,4096] | 16,384 | 1.50x | 1.50x | Yes |

**Notes:**
- The 64-tile shape shows higher variance across runs (fused: 124 us vs 170 us). At 64 tiles, compute is negligible and timing is dominated by dispatch latency jitter. Both runs confirm the fused kernel is substantially faster.
- All other shapes are highly reproducible (< 10% variation).
- The 16384-tile speedup of 1.50x is rock-solid across both runs.
