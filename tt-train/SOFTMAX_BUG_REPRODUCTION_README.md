# BERT Softmax Precision Bug Reproduction Branch

**Branch**: `ivoitovych/softmax-bug-reproduction`
**Based on**: `main` commit `eca8b5a8f1`
**Purpose**: Minimal reproduction of TTNN softmax bfloat16 precision bug

---

## What This Branch Contains

This branch contains the **minimal set of files** needed to reproduce the BERT softmax precision bug:

### Bug Summary

- **Issue**: BERT models show PCC 0.81 instead of expected >0.999
- **Root cause**: TTNN bfloat16 softmax accumulation loses precision on BERT attention patterns
- **Workaround**: Using FP32 accumulation restores PCC >0.95 (but degrades performance)
- **Status**: Bug NOT fixed in TTNN, workaround is active by default

### Files Added From Feature Branch

**BERT Implementation:**
- `sources/ttml/models/bert.cpp` - BERT model implementation
- `sources/ttml/models/bert.hpp` - BERT model header
- `sources/ttml/modules/bert_block.cpp` - BERT transformer block
- `sources/ttml/modules/bert_block.hpp` - BERT block header
- `sources/ttml/modules/multi_head_attention.cpp` - Multi-head attention module
- `sources/ttml/modules/multi_head_attention.hpp` - MHA header

**Operations (with workarounds):**
- `sources/ttml/ops/unary_ops.cpp` - **Contains FP32 softmax workaround (line ~107)**
- `sources/ttml/ops/unary_ops.hpp` - Unary ops header
- `sources/ttml/ops/embedding_op.cpp` - **Contains embedding batch workaround**
- `sources/ttml/ops/scaled_dot_product_attention.cpp` - Attention implementation
- `sources/ttml/ops/multi_head_utils.cpp` - Attention utilities

**Configuration:**
- `sources/ttml/core/compute_kernel_config.cpp` - **FP32 accumulation config**
- `sources/ttml/core/compute_kernel_config.hpp` - Config header

**Python Bindings:**
- `sources/ttml/nanobind/nb_models.cpp` - BERT model Python bindings
- `sources/ttml/nanobind/nb_ops.cpp` - Ops Python bindings

**Documentation:**
- `TTNN_BUG_REPORT_SOFTMAX_BFLOAT16_PRECISION.md` - Complete bug report
- `TTNN_BUG_REPRODUCTION_SOFTMAX.md` - Detailed reproduction instructions
- `tests/python/test_bert_end_to_end_validation.py` - End-to-end validation test

**Reproduction Script:**
- `reproduce_softmax_bug.py` - **Standalone script to reproduce the bug**

---

## How to Build and Test

### Prerequisites

```bash
# Install dependencies
pip install torch transformers safetensors numpy pytest

# Set environment
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME/tt-train/build/sources:$PYTHONPATH
```

### Build

```bash
cd $TT_METAL_HOME/tt-train
rm -rf build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build
```

### Run Reproduction Script

```bash
# Test WITH workaround (should show PCC >0.95)
cd $TT_METAL_HOME/tt-train
python3 reproduce_softmax_bug.py
```

**Expected output WITH workaround**:
```
PCC: 0.95xxxx
✅ PASS: PCC >0.95 (FP32 workaround is active)
```

---

## How to Reproduce the BUG

To see the actual bug (PCC 0.81), you must **disable the FP32 workaround**:

### Step 1: Disable Workaround

Edit `sources/ttml/ops/unary_ops.cpp` around line 107:

```cpp
// BEFORE (workaround active):
auto log_softmax = ttnn::moreh_softmax(
    tensor->get_value(),
    /* axis */ dim,
    /* output */ std::nullopt,
    ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOp::LOGSOFTMAX,
    ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy::NONE,
    /* output_mem_config */ std::nullopt,
    /* compute_kernel_config */ core::ComputeKernelConfig::softmax(/* use_fp32_accumulation_workaround */ true));
    //                                                                                                         ^^^^
    //                                                                                                         Change to false
```

Change to:

```cpp
// AFTER (workaround disabled - BUG VISIBLE):
auto log_softmax = ttnn::moreh_softmax(
    tensor->get_value(),
    /* axis */ dim,
    /* output */ std::nullopt,
    ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOp::LOGSOFTMAX,
    ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy::NONE,
    /* output_mem_config */ std::nullopt,
    /* compute_kernel_config */ core::ComputeKernelConfig::softmax(/* use_fp32_accumulation_workaround */ false));
    //                                                                                                         ^^^^^
```

### Step 2: Rebuild

```bash
cd $TT_METAL_HOME/tt-train
cmake --build build
```

### Step 3: Run Reproduction

```bash
python3 reproduce_softmax_bug.py
```

**Expected output WITHOUT workaround (BUG REPRODUCED)**:
```
PCC: 0.81xxxx
❌ FAIL: PCC <0.85 (Bug reproduced! FP32 workaround not active)
```

### Step 4: Re-enable Workaround

**IMPORTANT**: Re-enable the workaround after testing!

```bash
# Change back to true in sources/ttml/ops/unary_ops.cpp
vi sources/ttml/ops/unary_ops.cpp

# Rebuild
cmake --build build
```

---

## Alternative: Run Python Test

```bash
cd $TT_METAL_HOME/tt-train
export PYTHONPATH=$TT_METAL_HOME/tt-train/build/sources:$PYTHONPATH
python3 -m pytest tests/python/test_bert_end_to_end_validation.py -v
```

---

## Understanding the Bug

### Why This Happens

1. BERT attention computes softmax over attention scores
2. TTNN's bfloat16 softmax kernel loses precision during accumulation
3. This precision loss compounds across multiple layers
4. Result: PCC drops from expected >0.999 to ~0.81

### Why Workaround Works

- FP32 accumulation (`fp32_dest_acc_en=true`) maintains precision
- But this is SLOWER than native bfloat16
- **This is NOT a fix** - it's a temporary workaround with performance penalty

### What Needs to be Fixed

TTNN team must fix the bfloat16 softmax kernel to:
- Handle attention score distributions correctly
- Maintain precision without FP32 accumulation
- Match or exceed current workaround accuracy

---

## Important Notes

⚠️ **These are WORKAROUNDS, not fixes!**

1. **FP32 softmax workaround** (sources/ttml/ops/unary_ops.cpp line ~107)
   - Degrades performance vs native bfloat16
   - Only BERT code explicitly requests FP32
   - Default is `false` to preserve TTNN framework behavior

2. **Embedding batch workaround** (sources/ttml/ops/embedding_op.cpp)
   - Processes batches separately (slower)
   - Needed because TTNN embedding returns wrong values for batch indices > 0

3. **Do NOT remove workarounds** without:
   - Confirming TTNN bugs are fixed
   - Verifying PCC >0.999 on all BERT tests
   - Testing with workarounds disabled

---

## Documentation

- **TTNN_BUG_REPORT_SOFTMAX_BFLOAT16_PRECISION.md** - Complete bug report for TTNN team
- **TTNN_BUG_REPRODUCTION_SOFTMAX.md** - Comprehensive reproduction instructions with multiple models
- See feature branch `ivoitovych/bert-model-for-ttml-task-heads-v2` for full implementation

---

## Questions?

This branch is a minimal extraction from the full BERT implementation for bug reproduction purposes.
For the complete BERT implementation with task heads, see the feature branch.
