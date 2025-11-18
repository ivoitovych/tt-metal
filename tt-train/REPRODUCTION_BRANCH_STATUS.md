# Softmax Bug Reproduction Branch - Status

**Branch**: `ivoitovych/softmax-bug-reproduction`
**Based on**: `main` commit `eca8b5a8f1`
**Created**: 2025-11-18
**Status**: ⚠️ IN PROGRESS - Build issues remain

---

## Objective

Create a minimal, self-contained branch from main that demonstrates the TTNN bfloat16 softmax precision bug in BERT models, without carrying all the task heads implementation from the full feature branch.

---

## What Was Successfully Added

### Core BERT Implementation
✅ **Models**:
- `sources/ttml/models/bert.cpp` - BERT model implementation
- `sources/ttml/models/bert.hpp` - BERT model header

✅ **Modules**:
- `sources/ttml/modules/bert_block.cpp` - BERT transformer block
- `sources/ttml/modules/bert_block.hpp` - BERT block header
- `sources/ttml/modules/multi_head_attention.cpp` - Multi-head attention (updated from feature branch)
- `sources/ttml/modules/multi_head_attention.hpp` - MHA header
- `sources/ttml/modules/layer_norm_module.cpp` - LayerNorm with epsilon control (required for BERT)
- `sources/ttml/modules/layer_norm_module.hpp` - LayerNorm header

✅ **Operations (with workarounds)**:
- `sources/ttml/ops/unary_ops.cpp` - **FP32 softmax workaround (line ~107)**
- `sources/ttml/ops/unary_ops.hpp` - Unary ops header
- `sources/ttml/ops/embedding_op.cpp` - **Embedding batch workaround**
- `sources/ttml/ops/layernorm_op.cpp` - LayerNorm op with epsilon support
- `sources/ttml/ops/layernorm_op.hpp` - LayerNorm op header
- `sources/ttml/ops/scaled_dot_product_attention.cpp` - Attention implementation
- `sources/ttml/ops/multi_head_utils.cpp` - Attention utilities

✅ **Configuration**:
- `sources/ttml/core/compute_kernel_config.cpp` - **FP32 accumulation config**
- `sources/ttml/core/compute_kernel_config.hpp` - Config header

✅ **Python Bindings** (minimal):
- `sources/ttml/nanobind/nb_models.cpp` - Basic BERT model bindings (task heads commented out)
- `sources/ttml/nanobind/nb_ops.cpp` - Ops bindings (bert_losses commented out)

✅ **Build Configuration**:
- `sources/ttml/CMakeLists.txt` - Updated to include BERT source files

✅ **Documentation**:
- `SOFTMAX_BUG_REPRODUCTION_README.md` - Complete usage guide
- `TTNN_BUG_REPORT_SOFTMAX_BFLOAT16_PRECISION.md` - Bug report for TTNN team
- `TTNN_BUG_REPRODUCTION_SOFTMAX.md` - Detailed reproduction instructions
- `REPRODUCTION_BRANCH_STATUS.md` - This file

✅ **Reproduction Script**:
- `reproduce_softmax_bug.py` - Standalone Python script
- `tests/python/test_bert_end_to_end_validation.py` - Comprehensive validation test

---

## Current Build Status

### ✅ Successfully Built
- TTML static library (`libttml.a`) - Compiles successfully
- Core BERT model and modules compile without errors

### ❌ Build Issues Remaining

**Python Module Link Error**:
```
ImportError: undefined symbol: _ZN4ttml3ops36scaled_sigmoid_dot_product_attentionE...
```

**Root Cause**: The feature branch likely added `scaled_sigmoid_dot_product_attention` function that is referenced but not included in this minimal reproduction.

**What's Missing**:
1. Potentially additional ops or utility functions from the feature branch that BERT depends on
2. The dependency chain between BERT components may require more files than initially identified

---

## How to Complete This Branch

### Option 1: Debug Missing Dependencies (Recommended)

1. **Find the missing symbol**:
   ```bash
   cd /workspace/tt-metal/tt-train
   git --no-pager diff eca8b5a8f1..ivoitovych/bert-model-for-ttml-task-heads-v2 -- sources/ttml/ops/ | grep "scaled_sigmoid"
   ```

2. **Copy any missing implementation files** from the feature branch

3. **Rebuild and test**:
   ```bash
   cmake --build build --target _ttml
   export PYTHONPATH=/workspace/tt-metal/tt-train/build/sources/ttml:$PYTHONPATH
   python3 -c "import _ttml; print(_ttml.models.bert)"
   ```

4. **Run reproduction**:
   ```bash
   python3 reproduce_softmax_bug.py
   ```

### Option 2: Use Feature Branch Directly

If minimal reproduction proves too complex due to deep dependencies, consider:
- Using the full feature branch `ivoitovych/bert-model-for-ttml-task-heads-v2` for bug reproduction
- The reproduction scripts already exist there and work
- Document that bug reproduction requires the full BERT implementation

---

## What Works Without Python Bindings

Even though Python bindings don't link yet, the **C++ implementation is complete and buildable**. You could:

1. **Write a C++ test** instead of Python:
   - Copy `tests/python/test_bert_end_to_end_validation.py` logic to C++
   - Use the C++ BERT model directly
   - No Python bindings needed

2. **Example C++ test structure**:
   ```cpp
   // tests/model/bert_softmax_bug_test.cpp
   #include "models/bert.hpp"
   #include <gtest/gtest.h>

   TEST(BertSoftmaxBug, ReproduceWithWorkaround) {
       // Create BERT model
       // Load weights
       // Run forward pass
       // Compare with HuggingFace results
   }
   ```

---

## Files to Review

### For Missing Dependencies
Check these diffs from the feature branch:
```bash
git --no-pager diff --name-only eca8b5a8f1..ivoitovych/bert-model-for-ttml-task-heads-v2 -- sources/ttml/ops/
git --no-pager diff --name-only eca8b5a8f1..ivoitovych/bert-model-for-ttml-task-heads-v2 -- sources/ttml/modules/
```

### For Alternative Approaches
- Feature branch has complete working reproduction: `ivoitovych/bert-model-for-ttml-task-heads-v2`
- All Python tests pass there (22/22 critical tests)
- C++ tests pass (53/53)

---

## Recommendations

1. **Short-term**: Use the feature branch for bug reproduction since it's already working
2. **Medium-term**: Complete this minimal branch by finding and copying missing dependencies
3. **Long-term**: Consider whether minimal reproduction is worth the effort vs using full implementation

The bug reports (`TTNN_BUG_REPORT_*.md`) and reproduction docs (`TTNN_BUG_REPRODUCTION_*.md`) are self-contained and can be used independently of which branch is used for actual testing.

---

## Next Steps

1. Decide approach: complete this branch OR use feature branch
2. If completing this branch:
   - Find `scaled_sigmoid_dot_product_attention` implementation
   - Copy any other missing ops/utils
   - Verify Python bindings link
   - Test reproduction script
3. If using feature branch:
   - Document that in bug reports
   - Ensure feature branch remains available for TTNN team

---

**Note**: All workarounds are clearly documented with ⚠️ warnings. These are NOT fixes - they cause performance degradation. The proper fix must come from the TTNN team.
