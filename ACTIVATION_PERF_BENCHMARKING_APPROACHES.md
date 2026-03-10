# Activation Performance Benchmarking Approaches

Reusable methodology for benchmarking fused SFPU kernels vs composite decompositions.
Applicable to: tanh_bw, gelu_bw, gelu_fw, and any future activation optimizations.

## Context

When replacing a composite decomposition (N separate ops with intermediate tensors) with a fused SFPU kernel (single op, no intermediates), we need to measure the performance impact at multiple levels to give reviewers and users a complete picture.

## Approach 1: End-to-End Op Timing (Host-Side)

**What it measures:** Wall-clock time of the entire operation as seen by the host, including dispatch overhead, data movement, compute, and synchronization.

**Method:**
```python
import time
import ttnn

# Warmup (not measured)
for _ in range(warmup_iters):
    result = ttnn.op(input)
ttnn.synchronize_device(device)

# Measure
start = time.time_ns()
for _ in range(num_iters):
    result = ttnn.op(input)
ttnn.synchronize_device(device)
elapsed_ns = time.time_ns() - start

avg_ns = elapsed_ns / num_iters
```

**Shape sweep:** Single tile (32x32), medium (1024x1024), large (8192x8192)

**Pros:**
- Captures what the user actually sees
- Includes dispatch and memory savings from fusion
- Easy to write and run

**Cons:**
- Host overhead may dominate for small tensors
- Variance from OS scheduling

**When to use:** Always — this is the primary user-facing metric.

## Approach 2: Device Kernel Duration via Tracy Profiler

**What it measures:** Pure on-device kernel execution time, excluding host dispatch overhead.

**Method:**
```bash
TT_METAL_DEVICE_PROFILER=1 pytest test_file.py -k "test_name"
```

```python
# In test:
ttnn.synchronize_device(device)

# Warmup
for _ in range(warmup_iters):
    result = ttnn.op(input)
ttnn.synchronize_device(device)

# Read profiler to clear warmup data
ttnn.ReadDeviceProfiler(device)

# Measured run
for _ in range(num_iters):
    result = ttnn.op(input)
ttnn.synchronize_device(device)

# Collect profiler data
perf_data = ttnn.get_all_programs_perf_data()
# Extract DEVICE KERNEL DURATION [ns]
```

**Key metrics from profiler CSV:**
- `DEVICE KERNEL DURATION [ns]` — total kernel time on device
- `DEVICE TRISC0/1/2 KERNEL DURATION [ns]` — per-RISC compute time
- `DEVICE BRISC/NCRISC KERNEL DURATION [ns]` — data movement time
- `OP TO OP LATENCY [ns]` — inter-operation gap

**Composite vs fused comparison:**
- Composite: sum durations of all N ops in the decomposition
- Fused: single op duration
- The difference = eliminated dispatch + intermediate memory overhead

**Pros:**
- Isolates device execution from host
- Standard tt-metal methodology (same as conv2d perf tests)
- Per-RISC breakdown reveals compute vs data movement balance

**Cons:**
- Profiler overhead may perturb small-tensor measurements
- Requires `TT_METAL_DEVICE_PROFILER=1` env var
- Cannot use simultaneously with DPRINT or Watcher

**When to use:** For PR reviews — this is what Tenstorrent engineers expect.

## Approach 3: SFPU Cycle Counting via DeviceZoneScoped

**What it measures:** Exact cycle count of the compute portion only, excluding reader/writer.

**Method:** Add instrumentation to the compute kernel:
```cpp
// In compute kernel .cpp
DeviceZoneScopedN("sech2_compute");
for (uint32_t i = 0; i < num_tiles; i++) {
    // ... tile processing ...
}
```

**Pros:**
- Most precise compute measurement
- Directly answers "is the fused math faster per tile?"

**Cons:**
- Requires modifying kernel code (use a separate branch)
- Only measures compute, misses the dispatch/memory wins
- Composite requires instrumenting each of N kernels

**When to use:** Only if reviewers specifically ask about compute efficiency or when optimizing the SFPU polynomial itself.

## Approach 4: Hardware Performance Counters

**What it measures:** SFPU instruction cycles, FPU utilization, pipeline efficiency.

**Method:** Compile with `PROFILE_PERF_COUNTERS_FPU`, read counter registers.

**Pros:** Most granular — exact SFPU cycle counts.

**Cons:** Complex setup, sparse documentation, overkill for most reviews.

**When to use:** Deep optimization work, not for PR benchmarking.

## Approach 5: Throughput (Tiles/Second)

**What it measures:** Processing rate as a single intuitive number.

**Method:**
```python
num_tiles = tensor_volume / TILE_HW  # e.g., 65536 for 8192x8192
total_iters = 10

start = time.time_ns()
for _ in range(total_iters):
    result = ttnn.op(input)
ttnn.synchronize_device(device)
elapsed_s = (time.time_ns() - start) / 1e9

tiles_per_sec = (num_tiles * total_iters) / elapsed_s
```

**Pros:** Single compelling number for PR description.

**Cons:** Shape-dependent; large tensors may be memory-bound.

**When to use:** For the PR description summary — "X% throughput improvement."

## A/B Testing Protocol

For comparing old (composite) vs new (fused) implementations:

1. **Phase B (fused):** Run all benchmarks on the current branch
2. **Revert:** Change the single dispatch line in the host code to use composite
3. **Rebuild:** `cmake --build build_Debug --target unit_tests_ttnn -j$(nproc)` (or rebuild ttnn)
4. **Cache clear:** `rm -rf ~/.cache/tt-metal-cache/*`
5. **Phase A (composite):** Run all benchmarks again
6. **Restore:** Change back to fused dispatch
7. **Rebuild + cache clear** again
8. **Compare:** Phase A vs Phase B results

**Important:** Run Phase B first (fused) to confirm everything works, then revert for Phase A. This way if something breaks during revert, the fused results are already captured.

## Recommended Priority

| Priority | Approach | Effort | Value |
|----------|----------|--------|-------|
| 1 | Approach 2 (Tracy device profiler) | Medium | High — standard methodology |
| 2 | Approach 1 (host-side timing) | Low | High — user-visible metric |
| 3 | Approach 5 (throughput) | Low | Medium — compelling summary |
| 4 | Approach 3 (kernel cycles) | High | Low — only if asked |
| 5 | Approach 4 (hw counters) | Very High | Low — deep optimization only |

## Tensor Shapes for Sweep

| Name | Shape | Tiles | Purpose |
|------|-------|-------|---------|
| single_tile | [1,1,32,32] | 1 | Dispatch overhead dominance |
| small | [1,1,256,256] | 64 | Typical small activation |
| medium | [1,1,1024,1024] | 1024 | Balanced compute/dispatch |
| large | [1,1,4096,4096] | 16384 | Compute-bound regime |
| xlarge | [1,1,8192,8192] | 65536 | Maximum throughput test |
