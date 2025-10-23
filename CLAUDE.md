# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**TT-Metal** is Tenstorrent's software stack for programming AI accelerators, featuring a multi-layer architecture:

1. **TT-Metalium** (`tt_metal/`) - Low-level hardware programming model for direct control of Tensix processors
2. **TT-NN** (`ttnn/`) - High-level neural network operations library with PyTorch-like API
3. **TT-Train** (`tt-train/`) - C++ ML training framework with autograd (TTML sub-framework)
4. **Models** (`models/`) - Production ML models (LLMs, CNNs, Vision models) and demos

This is a C++ and Python codebase targeting Tenstorrent hardware (Wormhole, Blackhole processors). Each Tensix core contains 5 RISC-V CPUs, matrix/vector compute units, and 1.5MB SRAM.

### TT-Train / TTML Framework

**TT-Train** is a high-performance C++ training framework built on top of TT-NN/TT-Metalium. The **TTML** (TensTorrent ML) sub-framework provides PyTorch-like autograd capabilities for training models on Tenstorrent hardware.

**Key Features**:
- C++20 codebase with automatic differentiation (autograd)
- Module-based architecture (Linear, LayerNorm, Embedding, Attention, etc.)
- Model implementations (BERT, GPT-2, Llama, Linear Regression, MLP)
- Optimizers (Adam, AdamW, SGD)
- WandB integration for experiment tracking
- Safetensors support for model weights
- Python bindings via nanobind

## Build Commands

### TT-Train Build

TT-Train is a separate CMake project located in `tt-train/` directory.

```bash
# IMPORTANT: Always cd to tt-train directory first
# NEVER run this from tt-metal directory as it will break tt-metal build

# Recommended build command (Debug with ccache)
cd /workspace/tt-metal/tt-train && cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja && cmake --build build --config Debug --clean-first

# Release build with ccache
cd /workspace/tt-metal/tt-train && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja && cmake --build build --config Release --clean-first

# Alternative: separate configure and build steps
cd tt-train/
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build --config Debug

# Quick build script (alternative, but above command is preferred)
./build_all.sh
```

**Important Notes**:
- **CMake 3.30+** is required (newer than tt-metal's requirement)
- **Must build tt-metal first**, then tt-train
- **Always cd to `tt-train/` directory** before building
- Using `ccache` significantly speeds up rebuilds
- If you get patch errors on rebuild, remove `build/` directory completely: `rm -rf build`

### TT-Metal Build

### Initial Setup and Build
```bash
# Build the project (default: Release build)
./build_metal.sh

# Common build options
./build_metal.sh --build-type Debug              # Debug build
./build_metal.sh --build-type RelWithDebInfo     # Release with debug info
./build_metal.sh --enable-ccache                 # Enable ccache for faster rebuilds
./build_metal.sh --export-compile-commands       # Generate compile_commands.json
./build_metal.sh --clean                         # Clean all build artifacts

# Build with tests
./build_metal.sh --build-tests                   # Build all tests
./build_metal.sh --build-ttnn-tests              # Build TT-NN tests only
./build_metal.sh --build-metal-tests             # Build TT-Metalium tests only
./build_metal.sh --build-programming-examples    # Build programming examples

# Advanced options
./build_metal.sh --without-python-bindings       # C++ only, no Python
./build_metal.sh --without-distributed           # Disable multi-host support
./build_metal.sh --disable-unity-builds          # Disable unity builds (slower but better for debugging)
```

### CMake Direct Build (Alternative)
```bash
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=clang++-17
ninja
ninja install  # Required for Python environment
```

### Python Environment Setup
```bash
# Create/activate Python virtual environment
./create_venv.sh
source python_env/bin/activate

# Required environment variables
export TT_METAL_HOME=/workspace/tt-metal
export PYTHONPATH="${TT_METAL_HOME}"
```

## Running Tests

### TT-Train Tests (C++ / Google Test)

TT-Train uses Google Test (gtest) for C++ tests, not pytest.

```bash
# Navigate to tt-train directory
cd tt-train/

# Run all tests using ctest
ctest --test-dir build --output-on-failure

# Run the main ttml_tests executable (contains most tests)
./build/tests/ttml_tests

# Run specific test suites using filters
./build/tests/ttml_tests --gtest_filter=BertPolymorphismTest.*
./build/tests/ttml_tests --gtest_filter=BinaryOpsTest.*
./build/tests/ttml_tests --gtest_filter=GELUOpTest.*
./build/tests/ttml_tests --gtest_filter=ScaledDotProductAttentionTest.*
./build/tests/ttml_tests --gtest_filter=SliceRepeatOpsTest.*
./build/tests/ttml_tests --gtest_filter=EmbeddingOpTest.*

# Run specific test case
./build/tests/ttml_tests --gtest_filter=BertPolymorphismTest.BaseTransformerOperatorCall
./build/tests/ttml_tests --gtest_filter=GELUOpTest.GELU_NaNInfPropagation

# List all available tests
./build/tests/ttml_tests --gtest_list_tests

# Run tests with specific options
./build/tests/ttml_tests --gtest_brief=1                    # Only show failures
./build/tests/ttml_tests --gtest_repeat=5                   # Repeat 5 times
./build/tests/ttml_tests --gtest_shuffle                    # Randomize order
./build/tests/ttml_tests --gtest_output=xml:results.xml     # Generate XML report

# Run only non-nightly tests (exclude NIGHTLY_ prefix)
./build/tests/ttml_tests --gtest_filter=-NIGHTLY_*

# Run disabled tests
./build/tests/ttml_tests --gtest_also_run_disabled_tests

# Individual test executables (in build/tests/ops/, build/tests/autograd/, etc.)
./build/tests/ops/gelu_op_test
./build/tests/ops/binary_ops_test
./build/tests/ops/scaled_dot_product_attention_test
./build/tests/autograd/autograd_test
./build/tests/model/bert_polymorphism_test
```

**Common Test Suites in ttml_tests:**
- `AutogradTest` - Autograd tensor operations and backpropagation
- `BertPolymorphismTest` - BERT BaseTransformer interface tests
- `BinaryOpsTest` - Binary ops with broadcasting (add, subtract, multiply, divide)
- `UnaryOpsTest` - Unary ops (GELU, tanh, SiLU, etc.)
- `GELUOpTest` - Comprehensive GELU tests including edge cases (NaN/Inf)
- `ScaledDotProductAttentionTest` - SDPA forward/backward with masks
- `SliceRepeatOpsTest` - ttnn::slice and ttnn::repeat operations
- `EmbeddingOpTest` - Embedding forward/backward with layout tests
- `LinearOpTest` - Linear layer operations
- `LayerNormOpTest` - Layer normalization
- `RMSNormOpTest` - RMS normalization
- `ModuleBaseParametersTest` - Module parameter management
- `CrossEntropyForwardTest` / `CrossEntropyBackwardTest` - Loss functions

### TT-Train Example Programs

```bash
# MNIST MLP Training
cd tt-train/
./build/sources/examples/mnist_mlp/mnist_mlp --model_path mnist_mlp.msgpack --num_epochs 10

# MNIST Evaluation
./build/sources/examples/mnist_mlp/mnist_mlp --model_path mnist_mlp.msgpack -e 1

# NanoGPT Training
TT_LOGGER_LEVEL=FATAL ./build/sources/examples/nano_gpt/nano_gpt

# NanoGPT Evaluation
TT_LOGGER_LEVEL=FATAL ./build/sources/examples/nano_gpt/nano_gpt --model_path nano_gpt.msgpack -e 1 --data_path sources/examples/nano_gpt/data/shakespeare.txt

# Disable WandB logging
./build/sources/examples/mnist_mlp/mnist_mlp -w 0
# Or run: wandb offline
```

### TT-Metal Tests

### Python Tests (Pytest)

```bash
# Run specific test file
pytest tests/ttnn/unit_tests/operations/eltwise/test_add.py

# Run with verbose output
pytest tests/ttnn/unit_tests/operations/eltwise/test_add.py -vvs

# Run all tests in a directory
pytest tests/ttnn/unit_tests/operations/eltwise/

# Run tests with specific markers
pytest -m post_commit  # Post-commit tests only
pytest -m slow         # Slow tests
```

### C++ Tests (Google Test)
```bash
# Run specific C++ test binary
./build_Debug/tests/ttnn/unit_tests/gtests/tensor/test_tensor

# Run all C++ tests in directory
./build_Debug/tests/ttnn/unit_tests/gtests/
```

### Integration Tests
```bash
# Run model integration tests
pytest tests/ttnn/integration_tests/bert/
pytest tests/ttnn/integration_tests/falcon7b/

# Run specific model demo
python models/demos/bert/demo.py
```

### Sweep Tests
```bash
# Run sweep tests (comprehensive parameter coverage)
pytest tests/sweep_framework/sweeps/eltwise/unary/relu/
```

## Important Environment Variables

### Required for Development
```bash
export TT_METAL_HOME=/workspace/tt-metal  # Path to repository root
export PYTHONPATH="${TT_METAL_HOME}"      # Enable Python imports
```

### Debugging and Development
```bash
export TT_METAL_LOGGER_LEVEL=Debug        # Enable debug logging
export TT_METAL_WATCHER=10                # Enable Watcher (updates every 10s)
export TT_METAL_SLOW_DISPATCH_MODE=1      # Disable fast dispatch (debugging only)
export TT_METAL_DPRINT_CORES=(0,0)-(4,4) # Print debug output from 5x5 core grid
```

### Performance
```bash
# Set CPU governor for best performance (models/demos)
sudo cpupower frequency-set -g performance
```

## Architecture Overview

### TT-Train / TTML Architecture

**Location**: `tt-train/sources/ttml/`

TTML is organized into several key components:

#### **Autograd** (`tt-train/sources/ttml/autograd/`)
- **Tensor** (`tensor.hpp/cpp`): Autograd tensor with automatic differentiation
  - Wraps `ttnn::Tensor` with gradient tracking
  - `backward()` method for backpropagation
  - Computational graph construction and execution
- **AutoContext** (`auto_context.hpp/cpp`): Context manager for autograd operations
- **Graph** (`graph.hpp/cpp`): Computational graph for automatic differentiation

**Key Concept**: `autograd::TensorPtr` is the primary type used throughout TTML. It wraps ttnn tensors and tracks gradients automatically.

#### **Modules** (`tt-train/sources/ttml/modules/`)
PyTorch-like neural network modules:
- **module_base.hpp/cpp**: Base class for all modules (equivalent to `nn.Module`)
- **linear_module**: Fully connected layer
- **layer_norm_module**: Layer normalization
- **rms_norm_module**: RMS normalization
- **embedding_module**: Token embeddings
- **positional_embeddings**: Learned and fixed positional embeddings
- **dropout_module**: Dropout layer
- **multi_head_attention**: Multi-head attention mechanism
- **grouped_query_attention**: Grouped query attention (GQA)
- **single_head_attention**: Single attention head
- **multi_layer_perceptron**: MLP/FFN blocks
- **bert_block**: Complete BERT transformer block
- **gpt_block**: GPT transformer block
- **llama_block**: Llama transformer block
- **lora_linear_module**: LoRA (Low-Rank Adaptation) linear layer

#### **Operations** (`tt-train/sources/ttml/ops/`)
Autograd-enabled operations (forward + backward):
- **binary_ops**: Add, subtract, multiply, divide with broadcasting
- **unary_ops**: ReLU, GELU, SiLU, tanh, exp, log, sqrt, etc.
- **matmul_op**: Matrix multiplication
- **linear_op**: Linear transformation (x @ W^T + b)
- **embedding_op**: Embedding lookup
- **layernorm_op**: Layer normalization
- **rmsnorm_op**: RMS normalization
- **scaled_dot_product_attention**: Fused attention operation
- **rope_op**: Rotary position embeddings
- **losses**: Cross-entropy loss, MSE loss
- **dropout_op**: Dropout with training/eval modes

**Pattern**: Each op has forward and backward implementations. Backward ops are automatically called during `tensor.backward()`.

#### **Models** (`tt-train/sources/ttml/models/`)
Complete model implementations:
- **bert** (`bert.hpp/cpp`): BERT model with safetensors loading
  - Implements `BaseTransformer` interface
  - Supports token embeddings, position embeddings, token_type embeddings
  - Optional pooler for classification
  - Config-based initialization (YAML)
- **gpt2**: GPT-2 model
- **llama**: Llama/Llama2 model
- **mlp**: Simple multi-layer perceptron
- **linear_regression**: Basic linear regression

**BERT Specifics** (recent work by Iaroslav Voitovych):
- Polymorphic `BaseTransformer` interface for unified model handling
- Safetensors integration for loading pretrained weights
- Comprehensive embedding handling (tokens + positions + token_types)
- Layout handling fixes for proper tensor shapes

#### **Optimizers** (`tt-train/sources/ttml/optimizers/`)
- **adam_optimizer**: Adam optimizer
- **adamw_optimizer**: AdamW with weight decay
- **sgd_optimizer**: SGD with momentum

#### **Serialization** (`tt-train/sources/ttml/serialization/`)
- **msgpack**: Model serialization format
- **safetensors**: Loading pretrained weights (Hugging Face format)

#### **Core** (`tt-train/sources/ttml/core/`)
- Device management
- Tensor utilities
- Memory management

### Layer 1: TT-Metalium (Low-Level Hardware)
**Location**: `tt_metal/`

- **Host API** (`tt_metal/api/tt-metalium/`): Device management, program execution, buffer management
- **Implementation** (`tt_metal/impl/`): Device drivers, kernel compilation, memory allocation, command dispatch
- **Hardware Interface** (`tt_metal/hw/`): Firmware, compute kernels (ckernels), architecture-specific code
- **Kernels** (`tt_metal/kernels/`): Example reader/compute/writer kernels

**Key Concepts**:
- Each operation typically uses 3 kernel types: **reader** (data input), **compute** (computation), **writer** (data output)
- Kernels coordinate via circular buffers in SRAM with hardware synchronization
- Native tile-based computing: 32×32 element tiles
- No cache hierarchy - explicit SRAM management required

### Layer 2: TT-NN (Neural Network Library)
**Location**: `ttnn/`

- **Python API** (`ttnn/ttnn/`): PyTorch-like operations, tensor management, graph tracing
- **C++ Backend** (`ttnn/cpp/ttnn/`): 40+ operation implementations, tensor core, operation scheduling
- **Operations** (`ttnn/cpp/ttnn/operations/`): matmul, conv, eltwise, normalization, transformer, ccl, etc.

**Operation Structure Pattern**:
```
operations/<category>/<op_name>/
├── <op_name>.cpp/.hpp          # Operation interface
├── device/
│   ├── <op_name>_device.cpp    # Device implementation
│   └── <op_name>_program.cpp   # Program generation
└── kernel/
    ├── compute/                # Compute kernels
    └── dataflow/               # Reader/writer kernels
```

### Layer 3: Models
**Location**: `models/`

- **Transformers** (`models/tt_transformers/`): LLMs with TT-NN implementations
- **Demos** (`models/demos/`): Ready-to-run models (Llama, BERT, Falcon, YOLO, Whisper, etc.)
- **Common Utilities** (`models/common/`): Shared model utilities

### Multi-Device Support
- **Fabric** (`tt_metal/fabric/`): Chip interconnect via Ethernet
- **Distributed** (`tt_metal/distributed/`, `ttnn/core/distributed/`): Multi-device tensors
- **CCL Operations** (`ttnn/cpp/ttnn/operations/ccl/`): AllGather, AllReduce, AllToAll

## Code Patterns and Conventions

### Kernel Architecture (TT-Metalium)
Each Tensix operation uses 3 coordinating kernels:
1. **Reader kernel** (data movement kernel 0): Reads from DRAM via NoC0 into circular buffers
2. **Compute kernel**: Processes data from input CBs, writes to output CBs (uses Unpack/Math/Pack RISC-V cores)
3. **Writer kernel** (data movement kernel 1): Writes from circular buffers to DRAM via NoC1

Communication happens via circular buffers in L1 SRAM with hardware synchronization primitives (`cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`).

### SPMD Pattern
Unlike CUDA/OpenCL, Metalium doesn't provide `get_global_id()`. Instead:
- Use `split_work_to_cores()` utility to divide work across cores
- Explicitly set runtime arguments for each core with work offsets
- Each core gets unique parameters via `SetRuntimeArgs(program, kernel, core, {args...})`

### Test Organization
```
tests/ttnn/
├── unit_tests/                 # Pytest unit tests (block CI on failure)
├── sweep_tests/                # Parameterized comprehensive coverage tests
├── integration_tests/          # Full model tests
├── stress_tests/               # Extended robustness tests
├── statistical_tests/          # Numerical accuracy validation
└── multidevice_perf_tests/     # Performance benchmarks
```

### Build Artifacts
- **Primary build dir**: `build_Debug/`, `build_Release/`, or `build_RelWithDebInfo/`
- **Key outputs**:
  - `lib/libtt_metal.so` - TT-Metalium library
  - `lib/libttnn.so` - TT-NN C++ library
  - `ttnn/ttnn/_ttnn.so` - Python binding (~300MB)
  - `compile_commands.json` - For IDE integration (when using `--export-compile-commands`)

## Common Development Workflows

### TT-Train Development Workflow

#### Adding a New TTML Operation

1. **Create operation files** in `tt-train/sources/ttml/ops/`:
   ```cpp
   // my_op.hpp
   #pragma once
   #include "autograd/tensor.hpp"

   namespace ttml::ops {
   autograd::TensorPtr my_op(const autograd::TensorPtr& input);
   }
   ```

2. **Implement forward and backward**:
   ```cpp
   // my_op.cpp
   #include "my_op.hpp"
   #include "autograd/auto_context.hpp"

   autograd::TensorPtr my_op(const autograd::TensorPtr& input) {
       // Forward pass using ttnn ops
       auto output_tensor = ttnn::my_op(input->get_value());
       auto output = autograd::create_tensor(output_tensor);

       // Register backward operation if gradients needed
       if (input->get_requires_grad()) {
           autograd::GradFunction grad_fn = [input](const autograd::TensorPtr& grad_output) {
               // Compute gradient w.r.t. input
               auto grad = /* backward computation */;
               return std::vector<autograd::TensorPtr>{grad};
           };
           autograd::AutoContext::get_instance().add_backward_node(
               {input}, {output}, grad_fn);
       }
       return output;
   }
   ```

3. **Add tests** in `tt-train/tests/ops/`:
   ```cpp
   // my_op_test.cpp
   #include <gtest/gtest.h>
   #include "ops/my_op.hpp"

   TEST(MyOpTest, ForwardPass) {
       auto input = autograd::create_tensor(/* ... */);
       auto output = ops::my_op(input);
       // Assertions
   }

   TEST(MyOpTest, BackwardPass) {
       auto input = autograd::create_tensor(/* ... */);
       auto output = ops::my_op(input);
       output->backward();
       // Check gradients
   }
   ```

4. **Update CMakeLists.txt** to include new files

#### Working with BERT Model

Recent BERT work (Iaroslav Voitovych):

```cpp
// Loading BERT model
#include "models/bert.hpp"

ttml::models::bert::BertConfig config;
config.vocab_size = 30522;
config.max_sequence_length = 128;
config.embedding_dim = 768;
config.num_heads = 12;
config.num_blocks = 12;

auto model = ttml::models::bert::create(config);

// Load pretrained weights from safetensors
model->load_from_safetensors("path/to/model.safetensors");

// Forward pass
auto input_ids = /* tensor of shape [batch, seq_len] */;
auto attention_mask = /* optional mask */;
auto token_type_ids = /* optional segment ids */;

auto output = model->forward(input_ids, attention_mask, token_type_ids);

// Training loop
auto loss = /* compute loss */;
loss->backward();
optimizer.step();
```

#### Testing Autograd Operations

Key test patterns for autograd:
```cpp
// Test forward + backward
TEST(OperationTest, Gradient) {
    auto x = autograd::create_tensor(/* ... */);
    x->set_requires_grad(true);

    auto y = ops::my_operation(x);
    y->backward();

    // Verify gradient
    auto grad = x->get_grad();
    // Compare with numerical gradient or known values
}

// Test edge cases (NaN, Inf, zero)
TEST(OperationTest, EdgeCases) {
    // Test with NaN inputs
    // Test with Inf inputs
    // Test with zero values
    // Test with very large/small values
}

// Test broadcasting
TEST(BinaryOpTest, Broadcasting) {
    auto a = create_tensor(Shape{1, 64, 768});
    auto b = create_tensor(Shape{768});  // Should broadcast
    auto c = ops::add(a, b);
}
```

### TT-Metal Development Workflow

### Adding a New TT-NN Operation
1. Create operation directory: `ttnn/cpp/ttnn/operations/<category>/<op_name>/`
2. Implement operation interface and device code
3. Add compute and dataflow kernels under `kernel/`
4. Create Python wrapper in `ttnn/ttnn/operations/<category>.py`
5. Add unit tests: `tests/ttnn/unit_tests/operations/<category>/test_<op>.py`
6. Add sweep tests: `tests/sweep_framework/sweeps/<category>/<op>/`

### Writing TT-Metalium Kernels
1. Create reader kernel (`kernels/dataflow/reader.cpp`): NoC reads, populate circular buffers
2. Create compute kernel (`kernels/compute/compute.cpp`): Use compute APIs (add_tiles, pack_tile, etc.)
3. Create writer kernel (`kernels/dataflow/writer.cpp`): Read from CB, NoC writes to DRAM
4. Configure circular buffers in host code with appropriate sizes
5. Compile kernels with `CreateKernel()` and set runtime args with `SetRuntimeArgs()`

### Debugging Device Code
```bash
# Enable Watcher (monitors firmware/kernels for errors)
export TT_METAL_WATCHER=10  # Updates every 10 seconds

# Enable kernel debug printing
export TT_METAL_DPRINT_CORES=(0,0)  # Single core
export TT_METAL_DPRINT_CORES=(0,0)-(4,4)  # 5x5 grid

# Use slow dispatch mode (synchronous, easier to debug)
export TT_METAL_SLOW_DISPATCH_MODE=1

# Run with debug logging
export TT_METAL_LOGGER_LEVEL=Debug
```

### Running Single Model Demo
```bash
# Example: BERT demo
cd models/demos/bert/
python demo.py

# With performance governor
sudo cpupower frequency-set -g performance
python demo.py
```

## Key Technical Details

### Hardware Architecture
- **Tensix Core**: 5 Baby RISC-V CPUs (DM0, DM1, Unpack, Math, Pack) + FPU + SFPU + 1.5MB SRAM
- **NoC**: Two unidirectional networks (NoC0, NoC1) traversing chip in opposite directions
- **Memory**: SRAM (on-chip, explicit management) + DRAM (via NoC DMA)
- **Tile Size**: Native 32×32 element tiles for matrix/vector operations
- **Interleaved vs Sharded**: Interleaved (default, balanced) or sharded (topology-aware placement)

### Fast Dispatch
- **Default**: Async command queue, RISC-V core processes queued ops independently
- **Slow Dispatch**: Synchronous CPU-driven (set `TT_METAL_SLOW_DISPATCH_MODE=1`, debugging only)
- **Command Queues**: Queue 0 (compute), Queue 1 (data transfer), sync via events

### Compute APIs
- Abstraction layer for hardware portability across Grayskull/Wormhole/Blackhole
- FPU operations: Work directly on circular buffers (e.g., `add_tiles()`)
- SFPU operations: Require explicit Dst register management (e.g., `copy_tile()`, `sin_tile()`, `pack_tile()`)

### Multi-Device Programming
- Single mesh abstraction: Even 1 chip is a 1×1 mesh
- `MeshDevice::create_unit_mesh(device_id)` for single device
- `MeshBuffer` and `MeshWorkload` for distributed operations
- CCL operations for collective communications (AllGather, AllReduce, etc.)

## Documentation and Resources

- **TT-Metalium Guide**: `METALIUM_GUIDE.md` - Comprehensive hardware and programming model guide
- **Installation**: `INSTALLING.md` - Setup instructions for different installation methods
- **Contributing**: `CONTRIBUTING.md` - Development guidelines and CI/CD processes
- **Tech Reports**: `tech_reports/` - In-depth technical documentation on specific topics
- **Online Docs**: https://docs.tenstorrent.com/

## TT-Train Specific Notes

### Building and Dependencies
- **CMake 3.30+** is required (newer than tt-metal's 3.24)
- Must build tt-metal first, then tt-train
- TT-Train is a separate CMake project in `tt-train/` subdirectory
- Run `source ./init_repo.sh` to set up environment (direnv, clang-tidy, clang-format)
- Submodules must be initialized: `git submodule update --init --recursive`

### Testing
- Uses **Google Test (gtest)**, not pytest
- Tests are C++ executables in `build/tests/`
- Run with `ctest` or directly execute test binaries
- Use `--gtest_filter` to run specific tests
- Add `--gtest_list_tests` to see available tests

### Autograd Best Practices
- Always use `autograd::TensorPtr` for tensors that need gradients
- Call `tensor->set_requires_grad(true)` to enable gradient tracking
- Use `tensor->backward()` to compute gradients
- Check `tensor->is_grad_initialized()` before accessing gradients
- Clean computational graph with `tensor->clean_node()` if needed
- Use `AutoContext` for managing backward graph

### Module Development
- Inherit from `modules::ModuleBase`
- Override `parameters()` method to return trainable parameters
- Use `NamedParameters` for parameter management
- Modules can be nested (composition pattern)
- Call `module->train()` or `module->eval()` to set mode

### Model Loading
- Use **safetensors** format for pretrained weights (Hugging Face compatible)
- BERT model supports safetensors loading via `load_from_safetensors()`
- Use YAML configs for model configuration
- Parameter names must match safetensors keys

### WandB Integration
- Disable with `-w 0` flag or run `wandb offline`
- Creates `wandb/settings` file when offline
- Useful for tracking training metrics

### Common Pitfalls
- **Namespace changes**: Recent refactor moved from `autograd` to `modules` namespace for `ModuleBase`
- **Embedding layouts**: Recent fixes for proper tensor layout handling in embeddings
- **Broadcasting**: Ensure shapes are compatible for binary operations
- **Gradient accumulation**: Use `tensor->add_grad()` not `tensor->set_grad()` when accumulating
- **Memory**: Call `clean_node()` to free computational graph memory when done

## TT-Metal Specific Notes

### General Notes

- Always run `build_metal.sh` from repository root
- Always set `TT_METAL_HOME` and `PYTHONPATH` before running Python code
- For model development, set CPU performance governor: `sudo cpupower frequency-set -g performance`
- Use `--export-compile-commands` for IDE support (clangd, VS Code, etc.)
- Watcher is essential for device debugging - enable with `TT_METAL_WATCHER=10`
- Fast dispatch is default and much faster - only use slow dispatch for debugging dispatch issues
- All tensors are stored in tiled format (32×32) on device, padded to tile boundaries
- Circular buffers require explicit sizing - too small causes hangs, too large wastes SRAM
