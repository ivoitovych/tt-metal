# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmarks for ttnn.tanh_bw: fused sech²(x) kernel vs composite 1 - tanh²(x).

Implements three measurement approaches:
  - Approach 1: End-to-end host-side timing (time.time_ns)
  - Approach 2: Device profiler API (ttnn.get_all_programs_perf_data)
  - Approach 5: Throughput (tiles/second)

Usage:
  # Host-side timing (Approach 1 + 5, no special env vars needed)
  pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py -v -k "host_timing or throughput"

  # Device profiler (Approach 2, requires profiler env vars)
  TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_perf.py -v -k "device_profiler"

A/B testing:
  Run with fused kernel (current branch), record results.
  Revert unary_backward.cpp to composite, rebuild, clear cache, re-run.
  Compare results.
"""

import time
import os

import pytest
import torch
import ttnn

WARMUP_ITERS = 10
MEASURE_ITERS = 100

TILE_HW = 32 * 32  # 1024 elements per tile

# Shape sweep: from dispatch-dominated to compute-dominated
SHAPES = [
    pytest.param([1, 1, 32, 32], id="1_tile"),
    pytest.param([1, 1, 256, 256], id="64_tiles"),
    pytest.param([1, 1, 1024, 1024], id="1024_tiles"),
    pytest.param([1, 1, 4096, 4096], id="16384_tiles"),
]

# Large shapes for throughput test
THROUGHPUT_SHAPES = [
    pytest.param([1, 1, 1024, 1024], id="1024_tiles"),
    pytest.param([1, 1, 4096, 4096], id="16384_tiles"),
]


def create_input_tensors(shape, device):
    """Create grad and input tensors on device."""
    torch.manual_seed(42)
    grad_torch = torch.randn(shape, dtype=torch.bfloat16)
    input_torch = torch.randn(shape, dtype=torch.bfloat16)

    grad_tt = ttnn.from_torch(
        grad_torch,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tt = ttnn.from_torch(
        input_torch,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return grad_tt, input_tt


# =============================================================================
# Approach 1: End-to-End Host-Side Timing
# =============================================================================


@pytest.mark.parametrize("shape", SHAPES)
def test_host_timing(shape, device):
    """
    Measure wall-clock time of ttnn.tanh_bw as seen by the host.
    Includes dispatch overhead, data movement, compute, and synchronization.
    """
    grad_tt, input_tt = create_input_tensors(shape, device)
    num_tiles = 1
    for d in shape:
        num_tiles *= d
    num_tiles //= TILE_HW

    # Warmup
    for _ in range(WARMUP_ITERS):
        result = ttnn.tanh_bw(grad_tt, input_tt)
    ttnn.synchronize_device(device)

    # Measure
    start_ns = time.time_ns()
    for _ in range(MEASURE_ITERS):
        result = ttnn.tanh_bw(grad_tt, input_tt)
    ttnn.synchronize_device(device)
    elapsed_ns = time.time_ns() - start_ns

    avg_ns = elapsed_ns / MEASURE_ITERS
    avg_us = avg_ns / 1000.0

    print(f"\n{'='*60}")
    print(f"HOST TIMING — shape={shape}, tiles={num_tiles}")
    print(f"  Iterations:    {MEASURE_ITERS}")
    print(f"  Total time:    {elapsed_ns / 1e6:.3f} ms")
    print(f"  Avg per call:  {avg_us:.3f} us")
    print(f"  Avg per tile:  {avg_us / num_tiles:.3f} us/tile")
    print(f"{'='*60}")


# =============================================================================
# Approach 2: Device Profiler API
# =============================================================================


@pytest.mark.parametrize("shape", SHAPES)
def test_device_profiler(shape, device):
    """
    Measure device kernel duration using the Tracy profiler API.
    Requires: TT_METAL_DEVICE_PROFILER=1

    Extracts DEVICE KERNEL DURATION from profiler data, isolating
    on-device execution from host dispatch overhead.
    """
    profiler_enabled = os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "1"
    if not profiler_enabled:
        pytest.skip("TT_METAL_DEVICE_PROFILER=1 not set")

    grad_tt, input_tt = create_input_tensors(shape, device)
    num_tiles = 1
    for d in shape:
        num_tiles *= d
    num_tiles //= TILE_HW

    # Warmup
    for _ in range(WARMUP_ITERS):
        result = ttnn.tanh_bw(grad_tt, input_tt)
    ttnn.synchronize_device(device)

    # Clear profiler data from warmup
    ttnn.ReadDeviceProfiler(device)

    # Measured runs
    for _ in range(MEASURE_ITERS):
        result = ttnn.tanh_bw(grad_tt, input_tt)
    ttnn.synchronize_device(device)

    # Collect profiler data
    ttnn.ReadDeviceProfiler(device)
    perf_data = ttnn.get_all_programs_perf_data()

    if not perf_data:
        pytest.skip("No profiler data collected")

    # Extract durations from all programs
    durations_ns = []
    device_id = next(iter(perf_data))
    programs = perf_data[device_id]
    for program in programs:
        for analysis_name, result_data in program.program_analyses_results.items():
            if result_data.duration > 0:
                durations_ns.append(result_data.duration)

    if not durations_ns:
        pytest.skip("No duration data in profiler results")

    avg_ns = sum(durations_ns) / len(durations_ns)
    min_ns = min(durations_ns)
    max_ns = max(durations_ns)
    avg_us = avg_ns / 1000.0
    min_us = min_ns / 1000.0
    max_us = max_ns / 1000.0

    print(f"\n{'='*60}")
    print(f"DEVICE PROFILER — shape={shape}, tiles={num_tiles}")
    print(f"  Programs captured: {len(durations_ns)}")
    print(f"  Avg duration:  {avg_us:.3f} us")
    print(f"  Min duration:  {min_us:.3f} us")
    print(f"  Max duration:  {max_us:.3f} us")
    print(f"  Avg per tile:  {avg_us / num_tiles:.3f} us/tile")
    print(f"{'='*60}")


# =============================================================================
# Approach 5: Throughput (Tiles/Second)
# =============================================================================


@pytest.mark.parametrize("shape", THROUGHPUT_SHAPES)
def test_throughput(shape, device):
    """
    Measure processing throughput in tiles/second.
    Uses larger tensors to minimize dispatch overhead ratio.
    """
    grad_tt, input_tt = create_input_tensors(shape, device)
    num_tiles = 1
    for d in shape:
        num_tiles *= d
    num_tiles //= TILE_HW

    throughput_iters = 50

    # Warmup
    for _ in range(WARMUP_ITERS):
        result = ttnn.tanh_bw(grad_tt, input_tt)
    ttnn.synchronize_device(device)

    # Measure
    start_ns = time.time_ns()
    for _ in range(throughput_iters):
        result = ttnn.tanh_bw(grad_tt, input_tt)
    ttnn.synchronize_device(device)
    elapsed_ns = time.time_ns() - start_ns

    elapsed_s = elapsed_ns / 1e9
    total_tiles = num_tiles * throughput_iters
    tiles_per_sec = total_tiles / elapsed_s
    avg_us = (elapsed_ns / throughput_iters) / 1000.0

    print(f"\n{'='*60}")
    print(f"THROUGHPUT — shape={shape}, tiles={num_tiles}")
    print(f"  Iterations:      {throughput_iters}")
    print(f"  Total time:      {elapsed_s:.3f} s")
    print(f"  Avg per call:    {avg_us:.3f} us")
    print(f"  Tiles processed: {total_tiles:,}")
    print(f"  Throughput:      {tiles_per_sec:,.0f} tiles/sec")
    print(f"  Throughput:      {tiles_per_sec / 1e6:.3f} M tiles/sec")
    print(f"{'='*60}")
