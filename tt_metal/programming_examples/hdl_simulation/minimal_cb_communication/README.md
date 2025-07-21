# Minimal CB Communication Demo

[![TensTorrent TT-Metal](https://img.shields.io/badge/TensTorrent-TT--Metal-orange.svg)](https://github.com/tenstorrent/tt-metal)

A minimal example demonstrating host-to-compute kernel communication using L1/SRAM circular buffers (CBs) in the TensTorrent TT-Metal framework. Data flows from host to device, through kernels, and back, with comprehensive verification and debugging to ensure data integrity throughout the pipeline.

## Overview

This demo validates the complete data flow pipeline in TT-Metal by implementing a simple pass-through operation with extensive logging and verification.

### Key Features
- **Complete pipeline validation**: Host → DRAM → Reader → CB → Compute → CB → Writer → DRAM → Host
- **Single core execution**: Uses core (0,0) for simplicity
- **Double buffering**: CBs use 2 pages for overlap optimization
- **Tile-based processing**: bfloat16 format, 32x32 tiles (2048 elements per tile)
- **Comprehensive debugging**: DPRINT logs at each pipeline stage
- **Data integrity verification**: Pattern-based input with element-by-element validation
- **Performance metrics**: Reports throughput and data transfer statistics

### Why This Demo?
Perfect starting point for:
- **HDL simulation acceleration**: Understand data flow before adding complex compute
- **TT-Metal learning**: Minimal working example with all essential components
- **Debug template**: Extensive logging shows exactly what happens at each stage
- **Architecture validation**: Confirms hardware and software stack are working correctly

## Prerequisites
- TT-Metal framework installed (see main repository)
- Tenstorrent hardware accelerator (e.g., Wormhole)
- Build environment configured for TT-Metal development

## Installation
```bash
# Clone the repository
git clone https://github.com/ivoitovych/tt-metal.git
cd tt-metal

# Build TT-Metal with programming examples
./build_metal.sh --build-programming-examples --enable-ccache --debug

# Navigate to this demo
cd tt_metal/programming_examples/hdl_simulation/minimal_cb_communication
```

## Usage

### Quick Start
```bash
# Run with basic logging
./build_Debug/programming_examples/hdl_simulation/minimal_cb_communication

# Run with full kernel debug output
export TT_METAL_DPRINT_CORES=0,0
./build_Debug/programming_examples/hdl_simulation/minimal_cb_communication
```

### Expected Output
**Success case:**
```
Setting up buffers: 2 tiles, 4096 bytes total
Input data sample: 0.00 0.10 0.20 0.30 ... 4.69
Launching kernels...
Kernels completed, reading results...
Output data sample: 0.00 0.10 0.20 0.30 ... 4.69
Data transfer verification:
  Total elements: 2048
  Total bytes: 4096
  Tiles processed: 2
  Elements per tile: 1024
Verification passed: true
```

**With DPRINT enabled, you'll also see:**
```
READER: Starting with src_addr=... num_tiles=2
READER: Tile 0 first values: 0.0000 0.1000 0.2000 0.3000
COMPUTE: Starting with num_tiles=2
COMPUTE: Processed tile 0 (pass-through)
WRITER: Starting with dst_addr=... num_tiles=2
WRITER: Tile 0 first values: 0.0000 0.1000 0.2000 0.3000
```

## Architecture

### Data Flow Pipeline
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Host     │───▶│    DRAM     │───▶│   Reader    │───▶│   CB_IN     │
│   (CPU)     │    │   Input     │    │ (RISCV_0)   │    │    (L1)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                   │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Host     │◀───│    DRAM     │◀───│   Writer    │◀───│   Compute   │
│   (CPU)     │    │   Output    │    │ (RISCV_1)   │    │   (Math)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                   ▲
                                      ┌─────────────┐    ┌─────────────┐
                                      │   CB_OUT    │◀───│   CB_IN     │
                                      │    (L1)     │    │    (L1)     │
                                      └─────────────┘    └─────────────┘
```

### Component Details

| Component | Description | Processor | Function |
|-----------|-------------|-----------|----------|
| **Host Code** | Buffer management, kernel orchestration, verification | x86 CPU | Data preparation and validation |
| **Reader Kernel** | DRAM to L1 circular buffer transfer | RISCV_0 | High-bandwidth data ingestion |
| **Compute Kernel** | Tile processing and transformation | Tensix Math | Computational workload (pass-through) |
| **Writer Kernel** | L1 circular buffer to DRAM transfer | RISCV_1 | High-bandwidth data egress |

### Memory Hierarchy
- **Host Memory**: Input data generation and output verification
- **DRAM**: Device-side buffer storage (4KB total, 2KB per tile)
- **L1 SRAM**: Circular buffers with double buffering (2 pages each)
- **Compute Registers**: Tile processing workspace (16 tiles × 32×32)

## Code Structure

### Files Overview
```
minimal_cb_communication/
├── minimal_cb_communication.cpp          # Host orchestration
├── kernels/
│   ├── dataflow/
│   │   ├── reader_unary.cpp              # DRAM → CB reader
│   │   └── writer_unary.cpp              # CB → DRAM writer
│   └── compute/
│       └── add_one.cpp                   # CB → CB processor
└── README.md                             # This file
```

### Key Implementation Details

**Host Side (minimal_cb_communication.cpp):**
- Creates predictable input pattern (0.0, 0.1, 0.2, ..., 9.9, repeat)
- Configures double-buffered circular buffers (CB index 0 for input, 16 for output)
- Orchestrates kernel execution with proper synchronization
- Performs element-wise verification with detailed mismatch reporting

**Reader Kernel (reader_unary.cpp):**
- Uses `InterleavedAddrGenFast` for optimized DRAM access
- Implements proper circular buffer protocol (`cb_reserve_back` → `cb_push_back`)
- Includes DPRINT debugging to show first few elements of each tile

**Compute Kernel (add_one.cpp):**
- Implements standard compute kernel pattern (`acquire_dst` → process → `release_dst`)
- Uses `init_sfpu` for SFPU initialization with correct CB indices
- Performs tile copy operation (pass-through) for data integrity validation
- Extensible for adding actual mathematical operations

**Writer Kernel (writer_unary.cpp):**
- Implements consumer pattern (`cb_wait_front` → `cb_pop_front`)
- Uses `InterleavedAddrGenFast` for optimized DRAM writes
- Includes verification debugging to confirm data integrity before DRAM write

## Debugging and Verification

### Debug Output Control
```bash
# Enable kernel debugging (shows DPRINT output)
export TT_METAL_DPRINT_CORES=0,0

# Enable slow dispatch mode for synchronous execution
export TT_METAL_SLOW_DISPATCH_MODE=1

# Combine both for maximum debugging
export TT_METAL_DPRINT_CORES=0,0 TT_METAL_SLOW_DISPATCH_MODE=1
```

### Verification Strategy
1. **Pattern-based input**: Predictable sequence for easy manual verification
2. **Element-wise comparison**: Catches single-element corruption
3. **Pipeline stage logging**: DPRINT shows data at each transfer point
4. **Statistics reporting**: Performance and throughput metrics
5. **Error localization**: Reports first 10 mismatches with indices

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Compilation failure** | Kernel build errors | Check include paths, API usage |
| **Silent failure** | No DPRINT output | Set `TT_METAL_DPRINT_CORES=0,0` |
| **Data corruption** | Verification mismatches | Enable DPRINT, check CB indices |
| **Hang/timeout** | Program doesn't complete | Use `TT_METAL_SLOW_DISPATCH_MODE=1` |
| **Performance issues** | NUMA warnings | Ignore for functionality testing |

## Extending the Demo

### Adding Actual Computation
Replace the pass-through in `add_one.cpp` with:
```cpp
// Example: Add 1.0 to each element
copy_tile_to_dst_init_short(tt::CBIndex::c_0);
copy_tile(tt::CBIndex::c_0, 0, 0);
// Add your SFPU operations here
pack_tile(0, tt::CBIndex::c_16);
```

### Multi-Core Scaling
- Replace `CoreCoord(0, 0)` with `CoreRangeSet` for distributed processing
- Implement tile-level work distribution across cores
- Add inter-core synchronization for complex workloads

### Performance Optimization
- Increase `num_tiles` for larger datasets
- Implement pipelining with larger circular buffers
- Add bandwidth measurements and optimization

### Advanced Features
- **Sharded buffers**: For distributed memory access patterns
- **Multi-device**: Scale across multiple Tenstorrent devices
- **Custom data formats**: Beyond bfloat16 for specific use cases

## Performance Characteristics

- **Throughput**: ~4KB data transfer validated in <1 second
- **Latency**: Sub-millisecond kernel execution
- **Memory**: 8KB L1 SRAM (4KB for CBs + compute workspace)
- **Scalability**: Linear scaling with tile count and core count

## Troubleshooting

### Build Issues
```bash
# Clean and rebuild if compilation fails
make clean
./build_metal.sh --build-programming-examples --debug
```

### Runtime Issues
```bash
# Check device availability
ls /dev/tenstorrent/

# Verify driver installation
lsmod | grep tenstorrent
```

### Debug Information
```bash
# Get detailed device info
TT_METAL_LOGGER_LEVEL=Debug ./your_program

# Check memory usage
cat /proc/meminfo | grep -i huge
```

## Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** comprehensive tests for new features
4. **Include** DPRINT debugging for new kernels
5. **Update** documentation and examples
6. **Submit** a pull request with detailed description

### Development Guidelines
- Follow existing code style and patterns
- Add verification for any new data paths
- Include performance measurements for optimizations
- Test on actual hardware, not just simulation

## License

This project is licensed under the Apache License 2.0 - see the main TT-Metal repository for details.

## Acknowledgments

- TensTorrent team for the TT-Metal framework
- Community contributors for examples and documentation
- HDL simulation acceleration research community
