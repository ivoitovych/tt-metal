# Softmax Workaround Investigation Results

## Investigation Date
2025-11-18

## Executive Summary

After systematic testing with proper clean rebuilds, we discovered that:

1. **The "FP32 softmax workaround" was already present from the beginning** - the custom `ttml::metal::softmax` has ALWAYS used FP32 accumulation (hardcoded `fp32_dest_acc_en=true`)
2. **The workaround commits (668c173f81, 9a261d86cc) had NO EFFECT on BERT** - they only modified `ComputeKernelConfig::softmax()` and `ttnn_fixed::softmax`, which BERT doesn't use
3. **bert-small and bert-base NEVER achieved good PCC** - they still fail with PCC ~0.93 and ~0.80 respectively
4. **The test suite doesn't actually fail** when PCC < 0.95 - it prints warnings but pytest still reports PASSED

## Detailed Findings

### 1. Custom Softmax Always Had FP32

File: `sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`

Lines 293 and 302 have ALWAYS contained:
```cpp
kernels.compute_group_1 = create_compute_kernel(
    program, core_group_1, compute_group_1_args, defines,
    kComputeKernelPath, /*fp32_dest_acc_en=*/true);  // ← ALWAYS TRUE
```

This was present in the original softmax implementation (commit 6d395416f3) from the very beginning.

### 2. BERT Uses Custom Softmax, Not ttnn_fixed::softmax

File: `sources/ttml/ops/scaled_dot_product_attention.cpp`, line 185:
```cpp
auto attention_weights = ttml::metal::softmax(qk_scaled, /* axis */ 3);
```

BERT attention uses `ttml::metal::softmax`, which calls the custom implementation with hardcoded FP32.

The "workaround" commits modified:
- `ComputeKernelConfig::softmax()` - used by `ttnn_fixed::softmax`
- NOT used by BERT attention

### 3. Test Results Before and After "Workaround" Commits

Tested with proper clean rebuilds on feature branch `ivoitovych/bert-model-for-ttml-task-heads-v2`:

#### Commit 668c173f81^ (BEFORE "Enable FP32 accumulation"):
```
bert-tiny:  PCC 0.998 ✅
bert-small: PCC 0.931 ❌
bert-base:  PCC 0.805 ❌
```

#### Commit 668c173f81 (AFTER "Enable FP32 accumulation"):
```
bert-tiny:  PCC 0.998 ✅  (NO CHANGE)
bert-small: PCC 0.931 ❌  (NO CHANGE)
bert-base:  PCC 0.805 ❌  (NO CHANGE)
```

#### Commit 9a261d86cc^ (BEFORE "Add FP32 accumulation option"):
```
bert-small: PCC 0.931 ❌
```

#### Commit 9a261d86cc (AFTER "Add FP32 accumulation option"):
```
bert-small: PCC 0.931 ❌  (NO CHANGE)
```

#### Current feature branch HEAD:
```
bert-tiny:  PCC 0.998 ✅
bert-small: PCC 0.931 ❌
bert-base:  PCC 0.805 ❌
```

**Conclusion**: The "workaround" commits changed NOTHING for BERT models.

### 4. Test Suite Behavior

File: `tests/python/test_bert_end_to_end_validation.py`

Pass criteria: `pcc >= 0.95`

But the test **always reports PASSED** to pytest, even when all internal checks fail:
```
⚠️  RESULT: Some tests failed (PCC < 0.95)
PASSED
```

This explains why the documentation claims "Tests pass" when PCC values are actually failing.

### 5. Layer-by-Layer Tests Show Good PCC

Interestingly, `test_bert_isolated_layer_validation.py` shows:

**bert-base-uncased** (commit def06e1590, BEFORE any workarounds):
```
Block 0:  PCC 0.999973 ✅
Block 1:  PCC 0.999969 ✅
Block 2:  PCC 0.999971 ✅
...
Block 11: PCC 0.999969 ✅
```

**All blocks pass** even BEFORE the workaround! This suggests:
- Individual layers work correctly with the custom softmax FP32
- The precision loss happens when layers are **composed end-to-end**
- The problem is error accumulation across 4-12 layers, not a single softmax operation

### 6. Why Minimal Reproduction Branch Gets PCC 0.835

The minimal reproduction branch (`ivoitovych/softmax-bug-reproduction`) gets PCC 0.835 for bert-tiny, while the feature branch gets PCC 0.998.

Both branches have:
- ✅ Same custom softmax with FP32 hardcoded
- ✅ Same BERT model code
- ✅ Same attention mechanism

Possible causes for the difference:
1. **Different test methodology** - `reproduce_softmax_bug.py` vs pytest test suite
2. **Missing components** - other operations or workarounds present in feature branch
3. **Build differences** - compiler flags, optimization levels
4. **Input data differences** - different random seeds or data generation

## Conclusions

### What We Know:
1. ✅ Custom softmax (`ttml::metal::softmax`) ALWAYS had FP32 accumulation
2. ✅ BERT uses this custom softmax, not `ttnn_fixed::softmax`
3. ✅ The "workaround" commits (668c173f81, 9a261d86cc) had **zero effect** on BERT
4. ✅ Individual BERT layers achieve PCC >0.999 even before workarounds
5. ✅ End-to-end bert-small and bert-base FAIL with PCC 0.93 and 0.80
6. ✅ Only bert-tiny passes end-to-end (PCC 0.998)

### What This Means:
1. **There is NO softmax workaround** - FP32 was always enabled
2. **The bug is NOT in softmax alone** - individual layers work fine
3. **The problem is error accumulation** - precision loss compounds across layers
4. **bert-small and bert-base are NOT fixed** - they still fail
5. **The minimal reproduction attempt FAILED** - it doesn't properly isolate any specific bug

### Recommendations:
1. ❌ **Stop claiming there's a "softmax workaround"** - it's misleading
2. ❌ **Stop claiming bert-small and bert-base pass** - they don't (PCC < 0.95)
3. ✅ **Acknowledge only bert-tiny works** - 2 layers don't accumulate enough error
4. ✅ **Investigate error accumulation** - why does precision degrade across layers?
5. ✅ **Fix the test suite** - make it actually fail when PCC < 0.95
6. ✅ **Abandon the minimal reproduction branch** - it doesn't reproduce anything useful

## Test Commands Used

### Clean Rebuild Pattern:
```bash
cd /workspace/tt-metal/tt-train/
rm -rf build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build --config Debug --clean-first
```

### Test Commands:
```bash
export TT_METAL_HOME=/workspace/tt-metal
export LD_LIBRARY_PATH=/workspace/tt-metal/build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/tt-metal/tt-train/build/sources/ttml:$PYTHONPATH

# End-to-end test
python3 -m pytest tests/python/test_bert_end_to_end_validation.py::test_bert_end_to_end_validation[1-32-prajjwal1/bert-small] -v --tb=short

# Layer-by-layer test
python3 -m pytest tests/python/test_bert_isolated_layer_validation.py -k "bert-base" -v --tb=short
```

## Files Examined

1. `sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp` - Custom softmax with hardcoded FP32
2. `sources/ttml/ops/scaled_dot_product_attention.cpp` - BERT attention using custom softmax
3. `sources/ttml/core/compute_kernel_config.cpp` - Config modified by "workaround" commits
4. `sources/ttml/ttnn_fixed/trivial_ttnn_ops.cpp` - ttnn_fixed::softmax (not used by BERT)
5. `tests/python/test_bert_end_to_end_validation.py` - Test that doesn't actually fail

## Commits Analyzed

| Commit | Description | Actual Effect |
|--------|-------------|---------------|
| 6d395416f3 | Initial softmax implementation | FP32 already enabled |
| def06e1590 | Before "workarounds" | bert-tiny passes, others fail |
| 668c173f81 | "Enable FP32 accumulation" | ❌ No effect - wrong function |
| 9a261d86cc | "Add FP32 option" | ❌ No effect - wrong function |
| HEAD | Current feature branch | bert-tiny passes, others STILL fail |

---

**Investigation conducted by:** Claude Code
**Methodology:** Systematic testing with clean rebuilds at each commit
**Result:** The "softmax workaround" is a myth - it never existed and never worked
