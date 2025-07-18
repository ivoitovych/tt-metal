# Minimal CB Communication Demo

[![TensTorrent TT-Metal](https://img.shields.io/badge/TensTorrent-TT--Metal-orange.svg)](https://github.com/tenstorrent/tt-metal)

A minimal example demonstrating host-to-compute kernel communication using L1/SRAM circular buffers (CBs) in the TensTorrent TT-Metal framework. Data flows from host to device, through kernels, and back, with validation to ensure integrity.

## Overview
- **Purpose**: Showcase basic communication: host writes to DRAM, reader kernel moves to input CB, compute kernel processes (pass-through for demo), writer kernel moves to output DRAM, host reads and verifies.
- **Key Features**:
  - Uses single core (0,0).
  - Double buffering in CBs for overlap.
  - Tile-based data (bfloat16 format, 32x32 tiles).
  - Validation: Input == output (pass-through in compute).
- **Why This Demo?**: Simplifies starting with TT-Metal for HDL simulation acceleration PoC, focusing on data transfer without complex math.
- **Prerequisites**: TT-Metal installed (see main repo), hardware accelerator (e.g., Wormhole).

## Installation
- Clone the fork: `git clone https://github.com/ivoitovych/tt-metal.git`
- Build TT-Metal: Follow main README, then `./build_metal.sh --build-programming-examples --enable-ccache --debug`
- Navigate to this demo: `cd tt_metal/programming_examples/hdl_simulation/minimal_cb_communication`

## Usage
### Building and Running
1. Build the demo (included in programming_examples build above).
2. Run: `./build_Debug/programming_examples/hdl_simulation/minimal_cb_communication`
3. Expected Output: If successful, exits with code 0 (pass=true). Add logging for details (see code).

### Code Structure
- `minimal_cb_communication.cpp`: Host code (device init, buffers, kernels, enqueue, verify).
- `kernels/dataflow/reader_unary.cpp`: Reader kernel (DRAM to input CB).
- `kernels/compute/add_one.cpp`: Compute kernel (pass-through from input CB to output CB).
- `kernels/dataflow/writer_unary.cpp`: Writer kernel (output CB to DRAM).

Data Flow Diagram:
```
Host (CPU) --> EnqueueWriteBuffer --> DRAM Input Buffer
|
v
Reader Kernel (RISCV_0) --> NoC Read --> L1 CB In
|
v
Compute Kernel --> Tile Unpack/Copy/Pack --> L1 CB Out
|
v
Writer Kernel (RISCV_1) --> NoC Write --> DRAM Output Buffer
|
v
Host --> EnqueueReadBuffer --> Verify (input == output)
```

## Detailed Explanation
- **Communication Mechanism**: Host uses `EnqueueWriteBuffer` and `EnqueueReadBuffer` for DRAM. Kernels use runtime args for addresses, CBs for inter-kernel transfer with double buffering (2 pages per CB).
- **Tile-Based Data**: Data is bfloat16 in 32x32 tiles (num_tiles = 2 for minimal demo).
- **Validation**: Host generates random input, runs, checks output matches (pass-through ensures communication works without corruption).

Table of Components:
| Component | Description | File |
|-----------|-------------|------|
| Host Code | Buffer allocation, kernel launch, verification | minimal_cb_communication.cpp |
| Reader | DRAM to CB | kernels/dataflow/reader_unary.cpp |
| Compute | CB In to CB Out (copy) | kernels/compute/add_one.cpp |
| Writer | CB to DRAM | kernels/dataflow/writer_unary.cpp |

## Troubleshooting
- **Common Errors**:
  - Kernel path issues: Ensure paths in CreateKernel match repo structure.
  - Compilation failures: Check includes; rebuild with debug.
  - Runtime aborts: Set `TT_METAL_SLOW_DISPATCH_MODE=1` for sync debugging.
- **Logs**: Use `log_info` in host, DPRINT in kernels for diagnostics.
- **Known Issues**: Hugepage NUMA binding warning (perf impact, ignore for demo).

## Extending the Demo
- Add math in compute (e.g., SFPU add): See tt-metal examples like eltwise_unary.
- Multi-core: Use CoreRangeSet for sharded buffers.
- Performance: Increase num_tiles, benchmark transfer time.

## Contributing
- Fork and PR to https://github.com/ivoitovych/tt-metal.
- Issues: Report bugs with logs/code.

## License
Apache License (see LICENSE in main repo).
