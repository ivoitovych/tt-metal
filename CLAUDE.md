# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# TT-Metal Codebase Architecture Guide

## Repository Overview

TT-Metal is an open-source software stack for programming Tenstorrent AI accelerators. It consists of three main components:

1. **TT-Metalium**: Low-level programming model for kernel development
2. **TT-NN**: High-level neural network operations library built on TT-Metalium
3. **TT-Train**: C++ ML training framework built on top of TT-Metal/TT-NN (PRIMARY DEVELOPMENT FOCUS)

## Important: Development Workflow for This Repository

### Primary Development Directory: tt-train

**The `tt-train/` directory is the primary focus for development work.** This is a C++-first training framework that builds on top of tt-metal and ttnn.

### Critical Build Constraints

**DO NOT run CMake commands in the root `/workspace/tt-metal` directory** - this will break the tt-metal build that tt-train depends on. The tt-metal build is pre-configured and should not be modified.

### tt-train Build Process

Clean rebuild command (uses ccache for faster rebuilds):

```bash
cd /workspace/tt-metal/tt-train/ && \
rm -rf build && \
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -B build \
      -GNinja && \
cmake --build build --config Debug --clean-first
```

Build types available: `Debug`, `Release`, `RelWithDebInfo`, `CI`

### Git Workflow

Configured remotes:
- `origin`: `git@github.com:tenstorrent/tt-metal.git` (upstream)
- `myfork`: `git@github-alt:ivoitovych/tt-metal.git` (personal fork)

### Pre-commit Requirements

**CRITICAL**: Before every commit, run pre-commit hooks repeatedly until successful:

```bash
# Run pre-commit on changed files
pre-commit run --files <changed_files>

# If it fails, re-run until it succeeds (it may auto-fix issues)
# Then stage the auto-fixed files and commit
```

The pre-commit hooks may modify files (formatting, etc.). Continue running until all checks pass, then commit the modified files.

### Commit Message Guidelines

1. **Write comprehensive commit messages** - explain what was changed and why
2. **DO NOT mention "AI" in commit messages** - a fraction of the public is skeptical of such terminology
3. Use clear, descriptive language focusing on the technical changes
4. Format:
   ```
   Brief summary of changes (50-72 chars)

   Detailed explanation of:
   - What was changed
   - Why it was changed
   - How it addresses the problem

   Technical details, implementation notes, etc.
   ```

### Example Workflow

```bash
# Make changes in tt-train
cd /workspace/tt-metal/tt-train/
# ... edit files ...

# Build and test
rm -rf build && cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja && cmake --build build --config Debug

# Run pre-commit repeatedly until clean
pre-commit run --files sources/ttml/core/file.cpp sources/ttml/core/file.hpp
# ... fix any issues, re-run ...
pre-commit run --files sources/ttml/core/file.cpp sources/ttml/core/file.hpp

# Stage changes
git add sources/ttml/core/file.cpp sources/ttml/core/file.hpp

# Commit with comprehensive message (no AI references)
git commit -m "Refactor tensor utility functions for improved performance

Reorganized tensor helper functions in core/file.cpp to reduce
unnecessary memory allocations. Key changes:

- Introduced in-place tensor operations where possible
- Reduced temporary buffer allocations by 40%
- Improved cache locality in tensor iteration patterns

This change improves training throughput by approximately 15%
on MNIST MLP benchmarks."

# Push to personal fork
git push myfork branch-name
```

## Directory Structure

### Core Components

```
tt-metal/
├── tt_metal/                 # TT-Metalium low-level core
│   ├── api/                  # Host-facing API (device, buffer, program management)
│   ├── impl/                 # Implementation details
│   │   ├── device/           # Device pool and device implementations
│   │   ├── dispatch/         # Command queue dispatch system (runtime execution)
│   │   ├── allocator/        # Memory allocation
│   │   └── buffers/          # Buffer management
│   ├── llrt/                 # Low-level runtime (HAL, ELF file handling, cluster communication)
│   ├── kernels/              # Kernel templates (compute and dataflow)
│   │   ├── compute/          # Compute kernel implementations
│   │   └── dataflow/         # Data movement kernel implementations
│   ├── include/              # Public header files
│   ├── common/               # Shared utilities
│   ├── third_party/          # UMD (User Mode Driver)
│   └── hw/                   # Hardware descriptors and specifications
│
├── ttnn/                     # TT-NN high-level library
│   ├── core/                 # Core runtime and tensor abstractions
│   │   ├── tensor/           # Tensor class, layout, memory config
│   │   ├── distributed/      # Multi-device support (fabric, topology)
│   │   └── graph/            # Graph execution and tracing
│   ├── cpp/ttnn/             # C++ operation implementations
│   │   └── operations/       # Neural network operations (30+ categories)
│   │       ├── matmul/       # Matrix multiplication
│   │       ├── conv/         # Convolutions
│   │       ├── eltwise/      # Element-wise ops (unary, binary, ternary)
│   │       ├── data_movement/ # Sharding, all-gather, etc.
│   │       ├── ccl/          # Collective communication
│   │       ├── reduction/    # Reduce ops
│   │       ├── pool/         # Pooling
│   │       ├── normalization/ # LayerNorm, BatchNorm
│   │       ├── transformer/  # Attention, linear, etc.
│   │       └── (others)      # 20+ more operation categories
│   ├── cpp/ttnn-pybind/      # Python bindings (pybind11)
│   └── ttnn/                 # Python package (_ttnn.so + Python modules)
│
├── models/                   # Pre-optimized model implementations
│   ├── demos/                # Full model demos (BERT, LLMs, CNNs, etc.)
│   ├── tt_transformers/      # Transformer models
│   └── tt_cnn/               # CNN models
│
├── tt-train/                 # C++ ML training framework (PRIMARY FOCUS)
│   ├── sources/
│   │   ├── ttml/             # Core training library
│   │   │   ├── autograd/     # Automatic differentiation (Tensor, Graph, AutoContext)
│   │   │   ├── core/         # Device management, profiler, distributed support
│   │   │   ├── modules/      # NN building blocks (Linear, Attention, LayerNorm, etc.)
│   │   │   ├── models/       # Complete models (GPT-2, Llama, MLP)
│   │   │   ├── ops/          # Operations (losses, RMSNorm, RoPE, SDPA)
│   │   │   ├── optimizers/   # Training optimizers (AdamW, SGD, RemoteOptimizer)
│   │   │   ├── schedulers/   # LR schedulers (Linear, Step, Lambda)
│   │   │   ├── datasets/     # Data handling (DataLoader, TokenDataset)
│   │   │   ├── serialization/ # Model persistence (MsgPack, SafeTensors)
│   │   │   ├── tokenizers/   # Text tokenization (BPE, Character)
│   │   │   ├── metal/        # Custom Metal kernels (RMSNorm, CE, Softmax, SDPA)
│   │   │   ├── ttnn_fixed/   # TTNN operation wrappers
│   │   │   └── nanobind/     # Python bindings
│   │   └── examples/
│   │       ├── mnist_mlp/    # MNIST digit classification
│   │       ├── nano_gpt/     # GPT training on Shakespeare
│   │       ├── linear_regression/ # Basic regression
│   │       ├── linear_regression_ddp/ # Distributed data parallel
│   │       ├── simple_cnn/   # Convolutional neural network
│   │       ├── llm_inference/ # LLM inference
│   │       └── python/       # Python examples (transformers, multihost)
│   ├── tests/                # Comprehensive test suite
│   │   ├── autograd/         # Autograd tests
│   │   ├── modules/          # Module tests (with distributed variants)
│   │   ├── ops/              # Operation tests
│   │   ├── optimizers/       # Optimizer correctness
│   │   ├── model/            # End-to-end model tests (nano_gpt, gpt2s, etc.)
│   │   └── python/           # Python API tests
│   ├── CMakeLists.txt        # Main build configuration
│   └── build/                # Build output directory (created during build)
│
├── tests/                    # Test infrastructure
│   ├── ttnn/                 # TT-NN tests
│   │   ├── unit_tests/       # Traditional unit tests (pytest)
│   │   ├── sweep_tests/      # Parametric sweep tests for coverage
│   │   ├── integration_tests/ # Full model integration tests
│   │   └── benchmark/        # Performance benchmarks
│   └── tt_metal/             # TT-Metalium tests
│
├── scripts/                  # Build and utility scripts
├── docs/                     # Documentation
└── tech_reports/             # Detailed technical documentation
    ├── LLMs/                 # LLM optimization guides
    ├── CNNs/                 # CNN optimization guides
    ├── FlashAttention/       # Attention mechanism implementation
    └── (others)              # Various technical deep dives
```

## Architectural Layers

### Layer 1: Hardware Abstraction (TT-Metalium Low Level)

**Key Files**: `tt_metal/api/tt-metalium/tt_metal.hpp`, `tt_metal/llrt/`, `tt_metal/impl/dispatch/`

The bottom layer provides direct hardware control:

- **HAL (Hardware Abstraction Layer)**: Maps generic operations to chip-specific commands
- **LLRT (Low-Level Runtime)**: Manages ELF file loading, memory mapping, cluster communication
- **Dispatch System**: Command queue execution engine that translates high-level operations into hardware commands
- **Buffer Management**: Allocates and manages L1/DRAM memory across the cluster

**Key Concept**: The dispatch system uses a hardware command queue to asynchronously execute operations on the device. The `hardware_command_queue.cpp` implements the actual device communication.

### Layer 2: Kernel Programming Model

**Key Files**: `tt_metal/kernels/`, `tt_metal/programming_examples/`

TT-Metalium kernels follow a three-kernel per Tensix core pattern:

1. **Reader Kernel** (Data Movement 0): Pulls data from DRAM via NoC0, writes to circular buffers
2. **Compute Kernel**: Performs actual computation (compiled 3x for Unpack/Math/Pack cores)
3. **Writer Kernel** (Data Movement 1): Pulls results from circular buffers, writes to DRAM via NoC1

**Key Concept**: Kernels coordinate through circular buffers in SRAM. The compute API (`compute_kernel_api.h`) automatically generates core-specific binaries from unified source.

### Layer 3: Runtime & Device Management (TT-NN Core)

**Key Files**: `ttnn/core/tensor/`, `ttnn/core/device.cpp`, `ttnn/cpp/ttnn/operations/`

The TTNN core abstracts:

- **Tensor**: Multidimensional array supporting arbitrary layouts, memory configs, and device placement
- **Device**: Represents a single Tenstorrent device with memory pools and command queue
- **Distributed Tensor**: Extends tensors across multiple devices with explicit topology
- **Operation Interface**: All ops follow a pattern of `forward()` and `backward()` methods
- **Memory Config**: Specifies DRAM vs L1, interleaved vs sharded, tile layout options

**Key Concept**: The tensor abstraction allows developers to precisely control data layout across device clusters. The `memory_config` parameter on every operation enables hardware-specific optimizations.

### Layer 4: TT-Train Training Framework (PRIMARY FOCUS)

**Key Files**: `tt-train/sources/ttml/autograd/`, `tt-train/sources/ttml/modules/`, `tt-train/sources/ttml/optimizers/`

TT-Train is a **C++-first ML training framework** that provides PyTorch-like APIs built on top of TT-Metal and TTNN:

**Core Abstractions:**

- **autograd.Tensor** (`tt-train/sources/ttml/autograd/tensor.hpp`): Wraps `tt::tt_metal::Tensor` with automatic differentiation
  - Tracks computation graph for backward pass
  - Supports `.backward()` for gradient computation
  - Maintains gradient tensors and graph nodes

- **autograd.AutoContext** (`tt-train/sources/ttml/autograd/auto_context.hpp`): Singleton managing training state
  - Device/MeshDevice lifecycle management
  - Computational graph construction and execution
  - Random number generator state
  - Gradient mode (train/eval) control
  - Distributed training context

- **modules.ModuleBase** (`tt-train/sources/ttml/modules/module_base.hpp`): Base class for neural network layers
  - Parameter registration and management
  - Train/eval mode switching
  - Hierarchical module composition
  - Named parameter iteration for serialization
  - Zero-grad functionality

**Neural Network Modules** (`tt-train/sources/ttml/modules/`):
- Linear, Embedding, LayerNorm, RMSNorm, Dropout
- RotaryEmbedding, SingleHeadAttention, MultiHeadAttention, GroupedQueryAttention
- GPTBlock, LlamaBlock, MultiLayerPerceptron, LoRALinear

**Optimizers** (`tt-train/sources/ttml/optimizers/`):
- AdamW, SGD, NoOp, RemoteOptimizer (for distributed training)
- State dict support for checkpointing

**Training Loop Components:**
1. **Data Loading**: DataLoader + Dataset (`tt-train/sources/ttml/datasets/`)
2. **Forward Pass**: Module forward() with autograd tracking
3. **Loss Computation**: Cross-entropy, MSE (`tt-train/sources/ttml/ops/losses.cpp`)
4. **Backward Pass**: tensor.backward() computes gradients via graph
5. **Optimization**: optimizer.step() updates parameters
6. **LR Scheduling**: Schedulers update learning rate (`tt-train/sources/ttml/schedulers/`)
7. **Checkpointing**: MsgPack/SafeTensors serialization (`tt-train/sources/ttml/serialization/`)

**Custom Metal Kernels** (`tt-train/sources/ttml/metal/ops/`):
- RMSNorm (forward/backward), CrossEntropy (forward/backward)
- Softmax, ScaledDotProductAttention, SiLU backward
- Each includes device operation types, program factories, compute/dataflow kernels

**Build System**:
- C++20 with Clang-17, Ninja build system
- Links against pre-built tt-metal: `TT::Metalium`, `TTNN::TTNN`
- Uses `TT_METAL_HOME` environment variable to locate tt-metal
- Dependencies: Boost, xtensor, nanobind, msgpack-c, tokenizers-cpp, GoogleTest
- Creates `_ttml` nanobind extension module for Python bindings

**Examples** (`tt-train/sources/examples/`):
- MNIST MLP, NanoGPT (Shakespeare), Linear Regression (DDP), Simple CNN
- Python examples with YAML-configured transformers, multihost training
- Demonstrates end-to-end training workflows in both C++ and Python

**Architecture Pattern**: Metal Kernels → TTNN Ops → ttml Ops → Modules → Models, with autograd providing automatic differentiation throughout the stack.

### Layer 5: Neural Network Operations (TT-NN)

**Key Files**: `ttnn/cpp/ttnn/operations/*/`

30+ operation categories, each typically containing:

```
operation_name/
├── operation_name.hpp        # Public API
├── operation_name.cpp        # Implementation
├── operation_name_pybind.cpp # Python binding
├── device/
│   ├── operation_op.hpp      # Device-side logic
│   └── kernels/              # Kernel implementations
```

**Operation Categories** (non-exhaustive):
- **matmul**: Multi-core matrix multiplication with various parallelization strategies
- **conv**: Optimized convolution with im2col or other approaches
- **eltwise**: Element-wise unary, binary, ternary operations
- **data_movement**: Sharding, all-gather, pad, reshape, etc.
- **ccl**: AllReduce, AllGather for distributed training
- **reduction**: Sum, max, mean reductions
- **normalization**: LayerNorm, BatchNorm
- **transformer**: Scaled dot product attention, linear layers
- **pool**: Max/avg pooling
- **embedding**: Embedding lookup
- **loss**: Loss functions

**Key Concept**: Each operation is a Python-exposed C++ class with device-specific kernel implementations. Operations accept `MemoryConfig`, `ComputeKernelConfig`, and device-specific optimizations.

### Layer 5: Python API & Bindings

**Key Files**: `ttnn/ttnn/__init__.py`, `ttnn/cpp/ttnn-pybind/`

The Python interface uses pybind11 to expose C++ classes and functions. The Python package (`ttnn/ttnn/`) provides:

- Python modules for operations (`operations/`)
- Distributed utilities (`distributed/`)
- Model preprocessing utilities
- Graph tracing and profiling tools
- Type definitions and decorators

**Key Concept**: The `.so` file (`_ttnn.so`) is the compiled C++ library. Pure Python code in `ttnn/` wraps and enhances the compiled operations.

## Key Architectural Patterns

### 1. Operation Implementation Pattern

Every neural network operation follows this pattern:

```cpp
// In ttnn/cpp/ttnn/operations/my_op/my_op.hpp
struct MyOp {
    static Tensor invoke(
        const Tensor& input,
        const MemoryConfig& memory_config = ...,
        const DeviceComputeKernelConfig& compute_config = ...,
        ...  // operation-specific parameters
    );
};
```

Operations are:
- **Stateless**: Pure functions taking tensors and returning tensors
- **Configurable**: Accept memory and compute configurations
- **Composable**: Can be chained together
- **Validating**: Check input shapes, dtypes, and device placement

### 2. Multi-Device Programming

Multi-device support uses:

1. **MeshDevice**: Virtualizes multiple devices as a logical unit
2. **MeshTensor**: Extends Tensor across devices with explicit `distribution_mode`
3. **DistributedTensor**: Lower-level distributed tensor primitive
4. **Fabric**: Communication layer for inter-device data movement
5. **Topology**: Describes device mesh layout and connections

**Key Concept**: The topology system allows tensor operations to automatically shard/gather data across devices.

### 3. Memory Configuration

Every operation accepts a `MemoryConfig` parameter controlling:

- **Memory Type**: DRAM vs L1 (on-chip)
- **Layout**: TILE_LAYOUT (32x32 tiles) vs ROW_MAJOR
- **Sharding**: Replicated, ShardedRow, ShardedHeight, ShardedWidth
- **Interleaving**: Across which banks/cores

**Key Concept**: The hardware is tile-based. TILE_LAYOUT is native and preferred for performance.

### 4. Compute Kernel Configuration

Operations accept `DeviceComputeKernelConfig`:

- **fp32_dest_acc_en**: Enable FP32 accumulation in matrix engine
- **math_fidelity**: LOWEST, HI_FI, HIGHEST (affects precision/speed tradeoff)
- **math_approx_mode**: Enable approximate mathematical operations
- **tile_dtype**: Output tile data format

**Key Concept**: These settings control hardware behavior at the compute level, enabling performance tuning.

## Build System

**Key Files**: `CMakeLists.txt`, `build_metal.sh`

The project uses CMake with:

- **Unity Builds**: Combine multiple source files to speed compilation
- **CPM**: Conan Package Manager for dependency management
- **Submodules**: UMD (User Mode Driver) is a git submodule
- **Fast Dispatch**: Optional optimization for command dispatch

**Build Options**:
- `--build-type [Release|Debug|RelWithDebInfo]`
- `--build-ttnn-tests`: Build TT-NN tests
- `--build-metal-tests`: Build TT-Metalium tests
- `--enable-ccache`: Speed up rebuilds
- `--ttnn-shared-sub-libs`: Use shared libraries for faster linking

**Python Packaging**:
- Builds `ttnn` wheel via `setup.py`
- Installed via `pip install ttnn`
- Includes compiled `_ttnn.so` from C++ compilation

## Testing Infrastructure

### Unit Tests
- **Location**: `tests/ttnn/unit_tests/`
- **Runner**: `pytest`
- **Structure**: Traditional test files under operation categories
- **Purpose**: Verify correctness of individual operations

### Sweep Tests
- **Location**: `tests/ttnn/sweep_tests/`
- **Purpose**: Parametric testing across input/config combinations
- **Output**: CSV reports of coverage and results
- **Format**: Special `run()` method instead of test functions
- **Run with**: `pytest tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_<operation>`

### Integration Tests
- **Location**: `tests/ttnn/integration_tests/` and `tests/tt_metal/`
- **Purpose**: Full model validation
- **Examples**: BERT, ResNet, LLM inference tests

### Benchmarks
- **Location**: `tests/ttnn/benchmark/`
- **Purpose**: Performance measurement and tracking

## Key Components to Understand

### 1. Device & Memory Management

**Files**: `tt_metal/impl/device/`, `tt_metal/impl/allocator/`

- `Device`: Single device with L1/DRAM memory pools
- `DevicePool`: Manages multiple devices
- `Allocator`: Malloc-style interface for device memory
- `Buffer`: Opaque handle to device memory

### 2. Command Queue & Dispatch

**Files**: `tt_metal/impl/dispatch/hardware_command_queue.cpp`, `tt_metal/llrt/hal.cpp`

The dispatch system is critical for performance:

- Converts high-level operations into hardware commands
- Manages kernel binaries and circular buffers
- Handles synchronization and data movement
- Supports fast dispatch for low-latency execution

### 3. Tensor Layout System

**Files**: `ttnn/core/tensor/layout/`, `ttnn/core/tensor/memory_config/`

The layout system enables:

- Tile-based computation (32x32 tiles)
- Arbitrary tensor shapes mapped to tiles
- Sharded layouts for distributed computation
- Page-table style alignment for efficient DRAM access

### 4. Graph Execution

**Files**: `ttnn/core/graph/`, `tt_metal/graph/`

Graphs enable:

- Operation tracing and profiling
- Kernel compilation caching
- Multi-device coordination
- Debugging and visualization

### 5. Distributed Support

**Files**: `ttnn/core/distributed/`, `tt_metal/distributed/`

Multi-device programming via:

- **Fabric**: Communication layer (TCP, Ethernet)
- **MeshDevice**: Logical device abstraction
- **Distribution Mode**: Defines how tensors map to devices (replicate, shard, etc.)
- **CCL Operations**: AllReduce, AllGather, Scatter, Gather

## Common Build and Test Commands

### Building the Project

```bash
# Standard build (recommended)
./build_metal.sh

# Build with specific options
./build_metal.sh --build-type Release --enable-ccache

# Manual CMake build
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja
ninja install  # Required for Python environment
```

### Setting Up Python Environment

```bash
# Create virtual environment (optional: export PYTHON_ENV_DIR=<path> first)
./create_venv.sh
source python_env/bin/activate

# For model demos, install additional dependencies
pip install -r tt_metal/python_env/requirements-dev.txt
```

### Running Tests

#### TT-NN Tests (Python)

```bash
# Run specific pytest test
pytest tests/ttnn/unit_tests/operations/test_matmul.py

# Run all unit tests for an operation
pytest tests/ttnn/unit_tests/operations/ -k "test_matmul"

# Run sweep tests (parametric coverage tests)
pytest tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_matmul

# Run with markers
pytest tests/ttnn/unit_tests/ -m post_commit

# Run a single C++ test (legacy)
./build/test/tt_metal/test_add_two_ints

# Run model demos
python models/demos/bert/demo.py
```

#### TT-Train Tests (C++)

```bash
# Navigate to tt-train directory
cd /workspace/tt-metal/tt-train/

# Run all tests with CTest
cd build && ctest --output-on-failure

# Run specific test
./build/tests/autograd/test_autograd

# Run model tests
./build/tests/model/nano_gpt_test
./build/tests/model/gpt2s_test

# Run with different log levels
TT_LOGGER_LEVEL=FATAL ./build/sources/examples/nano_gpt/nano_gpt

# Run Python tests
pytest tests/python/test_autograd.py
```

#### TT-Train Examples

```bash
# MNIST training
./build/sources/examples/mnist_mlp/mnist_mlp --model_path mnist_mlp.msgpack --num_epochs 10

# MNIST evaluation
./build/sources/examples/mnist_mlp/mnist_mlp --model_path mnist_mlp.msgpack -e 1

# NanoGPT training
TT_LOGGER_LEVEL=FATAL ./build/sources/examples/nano_gpt/nano_gpt

# NanoGPT evaluation
TT_LOGGER_LEVEL=FATAL ./build/sources/examples/nano_gpt/nano_gpt --model_path nano_gpt.msgpack -e 1 --data_path sources/examples/nano_gpt/data/shakespeare.txt

# Disable wandb (if you don't have account)
wandb offline  # Creates wandb/settings file
# Or use: -w 0
```

### Environment Variables

```bash
# Enable debug logging
export TT_LOGGER_LEVEL=Debug
export TT_LOGGER_TYPES=Op

# Disable fast dispatch (for debugging)
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'
export TT_METAL_SLOW_DISPATCH_MODE=1

# Set performance governor (for benchmarking)
sudo cpupower frequency-set -g performance
```

## Common Developer Workflows

### TT-Train Development Workflow

#### Adding a New Module

1. Create module header in `tt-train/sources/ttml/modules/my_module.hpp`
2. Implement forward/backward in `my_module.cpp`
3. Register parameters in constructor
4. Add tests in `tt-train/tests/modules/test_my_module.cpp`
5. Add Python binding in `tt-train/sources/ttml/nanobind/` if needed

#### Adding a New Operation

1. Create op in `tt-train/sources/ttml/ops/my_op.cpp`
2. Implement forward function (creates autograd graph node)
3. Implement backward function (gradient computation)
4. Add tests in `tt-train/tests/ops/test_my_op.cpp`
5. If performance-critical, consider custom Metal kernel in `tt-train/sources/ttml/metal/ops/`

#### Adding Custom Metal Kernels

1. Create operation in `tt-train/sources/ttml/metal/ops/my_kernel/`
2. Implement device operation type and program factory
3. Write compute kernels in `my_kernel_compute.cpp`
4. Write dataflow kernels (reader/writer) in `my_kernel_dataflow.cpp`
5. Add forward/backward operations
6. Test thoroughly with various input shapes

### TT-NN Development Workflow

#### Adding a New Operation

1. Create `ttnn/cpp/ttnn/operations/my_op/`
2. Implement CPU fallback and validation in `my_op.cpp`
3. Implement device kernels in `device/kernels/`
4. Write Python binding in `my_op_pybind.cpp`
5. Add tests in `tests/ttnn/unit_tests/operations/`
6. Create sweep test in `tests/ttnn/sweep_tests/`

### Optimizing an Operation

1. Profile current implementation (use `tests/ttnn/benchmark/`)
2. Adjust `MemoryConfig` (sharding strategy, layout)
3. Adjust `DeviceComputeKernelConfig` (math fidelity, accumulator precision)
4. Modify kernel implementations in `device/kernels/`
5. Benchmark against targets in `models/README.md`

### Debugging Operations

1. Enable logging: `export TT_LOGGER_TYPES=Op` and `export TT_LOGGER_LEVEL=DEBUG`
2. Disable fast dispatch: `export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'`
3. Use DPRINT in kernels for kernel-side debugging
4. Check Watcher output for hardware errors
5. Use Inspector for host runtime analysis

### Multi-Device Programming

1. Create `MeshDevice` instead of single device
2. Use `DistributionMode` to shard tensors
3. Use CCL operations for cross-device communication
4. Leverage topology awareness for topology-specific optimizations

## Performance Optimization Hierarchy

1. **Algorithmic**: Use faster algorithms (e.g., im2col convolution)
2. **Kernel-level**: Optimize compute kernel implementation
3. **Layout**: Use tile layout, proper sharding strategy
4. **Memory**: Minimize DRAM access via sharding, batching
5. **Config**: Tune math fidelity, accumulator precision
6. **Dispatch**: Use fast dispatch, trace-based execution

## Important Constants & Defaults

- **Tile Size**: 32x32 elements (native hardware size)
- **Tile Memory**: Depends on dtype (bfloat16 = 2KB per tile)
- **L1 Memory**: ~1.5MB per Tensix core
- **Default Dtype**: bfloat16 (brain float 16-bit)
- **Default Layout**: TILE_LAYOUT
- **Default Memory**: DRAM (off-chip)

## Development Tools & Debugging

- **TT-SMI**: System management interface
- **TT-Exalens**: Low-level hardware debugger
- **Tracy Profiler**: Real-time performance profiler
- **Watcher**: Firmware/kernel error monitor
- **Inspector**: Host runtime analysis
- **Model Explorer**: Graph visualization

## Related Ecosystem

- **TT-Forge**: Compiler for various ML frameworks
- **TT-Torch**: PyTorch integration
- **TT-XLA**: XLA compiler backend
- **TT-MLIR**: MLIR-based compilation
- **vLLM (Tenstorrent Fork)**: Production LLM serving

## Quick Reference: Important Files

| Purpose | File |
|---------|------|
| Main API | `tt_metal/api/tt-metalium/tt_metal.hpp` |
| Tensor Class | `ttnn/core/tensor/tensor.hpp` |
| Operation Base | `ttnn/core/operations/operation.hpp` |
| Memory Config | `ttnn/core/tensor/memory_config/memory_config.hpp` |
| Device | `tt_metal/impl/device/device.cpp` |
| Dispatch | `tt_metal/impl/dispatch/hardware_command_queue.cpp` |
| Kernels | `tt_metal/kernels/{compute,dataflow}/` |
| Python Init | `ttnn/ttnn/__init__.py` |
| Examples | `models/demos/`, `tt_metal/programming_examples/` |

## Notes for Claude

- The codebase is large: focus on the operation category you're working with
- Memory and layout configuration are critical for performance
- Always check `models/README.md` for performance targets
- Test infrastructure is extensive: unit tests, sweeps, integration tests, benchmarks
- Distributed programming requires understanding topology and fabric communication
- The tile-based architecture is fundamental to everything
- Fast dispatch is an optional but important performance optimization
- **Primary focus is tt-train**: Most development happens in the `tt-train/` directory
- **Never break tt-metal build**: The pre-built tt-metal library is a dependency for tt-train

---

## Lessons Learned

This section documents important lessons and gotchas discovered during development. Add new entries as issues are encountered and resolved.

### Build System

- **DO NOT run CMake in root `/workspace/tt-metal`**: This breaks the tt-metal build that tt-train depends on. Always work in `tt-train/` subdirectory.
- **Use ccache**: Dramatically speeds up rebuilds. Include `-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache` in CMake command.
- **Clean rebuild when in doubt**: `rm -rf build` followed by full CMake + build cycle resolves many obscure issues.

### Pre-commit Hooks

- **Run pre-commit multiple times**: The hooks auto-fix formatting issues. Run repeatedly until all checks pass, then stage the modified files.
- **Pre-commit may change files**: After running pre-commit, files may be modified (formatting, etc.). Stage these changes and commit them.

### Git Workflow

- **Never mention AI in commits**: Use technical language focused on implementation details rather than referring to automated tools.
- **Write comprehensive commit messages**: Explain what, why, and how. Include performance impacts if relevant.

### Testing

- **Nightly tests**: Some tests are marked "NIGHTLY_" and only run in nightly CI. To run locally, modify `is_nightly_tt_train_tests_enabled` in test files.
- **wandb offline mode**: If you don't have wandb account, run `wandb offline` before training or use `-w 0` flag.

### Performance

- (To be filled in as lessons are learned)

### Debugging

- (To be filled in as lessons are learned)

### Common Pitfalls

- (To be filled in as lessons are learned)
