# Complete Timeline of FP32 Changes in Softmax Operations

## Summary

This document traces every FP32-related change across all softmax implementations in the tt-train codebase.

**Key Finding**: The custom `ttml::metal::softmax` (used by BERT) has ALWAYS had `fp32_dest_acc_en = true` and was NEVER affected by any "workaround" commits.

---

## Timeline of FP32 Changes

### Phase 1: Initial Implementation (Before Feb 2025)

#### Commit 6d395416f3 (Original Softmax Implementation)
**Date**: Unknown (prior to Feb 2025)
**Title**: [tt-train] Softmax operation for tt-train (#22804)

**File**: `sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`

**State**:
```cpp
.fp32_dest_acc_en = true,  // ← INITIAL STATE
```

**File**: `sources/ttml/core/compute_kernel_config.cpp`

**State**:
```cpp
ttnn::WormholeComputeKernelConfig ComputeKernelConfig::softmax() {
    config.fp32_dest_acc_en = true;  // ← INITIAL STATE
}
```

**File**: `sources/ttml/ops/unary_ops.cpp` (log_softmax_moreh)

**State**:
```cpp
auto log_softmax = ttnn::moreh_softmax(
    ...
    core::ComputeKernelConfig::precise());  // ← Uses precise() which has fp32=true
```

---

### Phase 2: FP32 Turned OFF (Feb 6, 2025)

#### Commit e7e86d78be
**Date**: Feb 6, 2025
**Author**: Denys Makoviichuk
**Title**: [TT-Train] fp32 turned off for softmax (#17683)

**Problem Description** (from commit message):
> fp32_dest doesn't work for softmax and log_softmax.
> Training is exploding.

**Changes**:

1. **File**: `sources/ttml/core/compute_kernel_config.cpp`
```cpp
ttnn::WormholeComputeKernelConfig ComputeKernelConfig::softmax() {
-   config.fp32_dest_acc_en = true;
+   config.fp32_dest_acc_en = false;  // ← TURNED OFF
}
```

2. **File**: `sources/ttml/ops/unary_ops.cpp`
```cpp
auto log_softmax = ttnn::moreh_softmax(
    ...
-   core::ComputeKernelConfig::precise());
+   core::ComputeKernelConfig::softmax());  // ← Now uses softmax() which has fp32=false
```

**Files NOT Changed**:
- `sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`
  - **Still has `fp32_dest_acc_en = true`**
  - BERT continues to use FP32!

**Impact**:
- ✅ `ComputeKernelConfig::softmax()`: fp32 = false
- ✅ `log_softmax_moreh`: fp32 = false
- ❌ `ttml::metal::softmax` (BERT): **fp32 = true (unchanged)**

---

### Phase 3: "Workaround" Attempts (Nov 14, 2025)

#### Commit 668c173f81
**Date**: Nov 14, 2025
**Title**: fix: Enable FP32 accumulation in softmax for BERT precision

**Claimed**: "BREAKTHROUGH: Single-line fix resolves all BERT attention bugs!"

**Changes**:

**File**: `sources/ttml/core/compute_kernel_config.cpp`
```cpp
ttnn::WormholeComputeKernelConfig ComputeKernelConfig::softmax() {
-   config.fp32_dest_acc_en = false;
+   config.fp32_dest_acc_en = true;  // ← TURNED BACK ON
}
```

**Files NOT Changed**:
- `sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`
  - **Still has `fp32_dest_acc_en = true` (unchanged since original)**

**Impact on BERT**: **ZERO**
- BERT uses `ttml::metal::softmax` which was never affected
- This only affects `ttnn_fixed::softmax` and `log_softmax_moreh`
- BERT test results: **PCC unchanged** (0.998 for tiny, 0.931 for small, 0.805 for base)

---

#### Commit 9a261d86cc
**Date**: Nov 14, 2025
**Title**: workaround: Add FP32 accumulation option for softmax bfloat16 precision bug

**Claimed**: "Test results with workaround enabled: bert-tiny: All blocks PCC >0.999"

**Changes**:

1. **File**: `sources/ttml/core/compute_kernel_config.cpp`
```cpp
-ttnn::WormholeComputeKernelConfig ComputeKernelConfig::softmax() {
+ttnn::WormholeComputeKernelConfig ComputeKernelConfig::softmax(bool use_fp32_accumulation_workaround) {
    config.fp32_dest_acc_en = use_fp32_accumulation_workaround;  // ← Now parameterized
}
```

2. **File**: `sources/ttml/ttnn_fixed/trivial_ttnn_ops.cpp`
```cpp
-tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim) {
+tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim,
+                              bool use_fp32_accumulation_workaround) {
    return ttnn::softmax(
        t, dim, std::nullopt,
-       ttml::core::ComputeKernelConfig::softmax(),
+       ttml::core::ComputeKernelConfig::softmax(use_fp32_accumulation_workaround),
        true);
}
```

**Files NOT Changed**:
- `sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`
  - **Still has `fp32_dest_acc_en = true` (unchanged since original)**
- `sources/ttml/ops/scaled_dot_product_attention.cpp`
  - **BERT still uses `ttml::metal::softmax`**

**Impact on BERT**: **ZERO**
- BERT doesn't use `ttnn_fixed::softmax`
- BERT test results: **PCC unchanged** (0.998 for tiny, 0.931 for small, 0.805 for base)

---

#### Commit d3b81efa58
**Date**: After Nov 14, 2025
**Title**: fix: Change softmax workaround default to false - preserve TTNN framework

**Changes**:

**File**: `sources/ttml/core/compute_kernel_config.hpp`
```cpp
-static ttnn::WormholeComputeKernelConfig softmax(bool use_fp32_accumulation_workaround = true);
+static ttnn::WormholeComputeKernelConfig softmax(bool use_fp32_accumulation_workaround = false);
```

**Reason**: Changed default to `false` to preserve TTNN framework behavior

**Impact on BERT**: **ZERO** (BERT doesn't use this function)

---

## Current State (As of Investigation 2025-11-18)

### ComputeKernelConfig::softmax()
- **File**: `sources/ttml/core/compute_kernel_config.cpp`
- **State**: `fp32_dest_acc_en = use_fp32_accumulation_workaround` (default = false)
- **Used by**: `ttnn_fixed::softmax`, `log_softmax_moreh`
- **NOT used by**: BERT attention

### ttml::metal::softmax (Custom Softmax)
- **File**: `sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`
- **State**: `fp32_dest_acc_en = true` (ALWAYS, since commit 6d395416f3)
- **Used by**: **BERT attention** (`scaled_dot_product_attention.cpp`)
- **Never changed by any "workaround" commits**

---

## Test Results Across Timeline

### Before e7e86d78be (FP32 ON for everything):
- Custom softmax: fp32 = true ✅
- ComputeKernelConfig::softmax(): fp32 = true ✅
- log_softmax_moreh: uses precise() → fp32 = true ✅

**BERT Results**: Unknown (before BERT implementation)

### After e7e86d78be (FP32 OFF for ComputeKernelConfig):
- Custom softmax: fp32 = true ✅
- ComputeKernelConfig::softmax(): **fp32 = false** ❌
- log_softmax_moreh: uses softmax() → **fp32 = false** ❌

**BERT Results**: Unknown (before BERT implementation)

### After 668c173f81 ("Enable FP32 workaround"):
- Custom softmax: fp32 = true ✅ (unchanged)
- ComputeKernelConfig::softmax(): fp32 = true ✅
- log_softmax_moreh: uses softmax() → fp32 = true ✅

**BERT Results** (tested with clean rebuilds):
- bert-tiny: PCC 0.998 ✅
- bert-small: PCC 0.931 ❌
- bert-base: PCC 0.805 ❌

### After 9a261d86cc ("Add FP32 option"):
- Custom softmax: fp32 = true ✅ (unchanged)
- ComputeKernelConfig::softmax(): fp32 = parameter (default true)
- log_softmax_moreh: uses softmax() → fp32 = parameter

**BERT Results** (tested with clean rebuilds):
- bert-tiny: PCC 0.998 ✅
- bert-small: PCC 0.931 ❌ **NO CHANGE**
- bert-base: PCC 0.805 ❌ **NO CHANGE**

### Current HEAD (default = false):
- Custom softmax: fp32 = true ✅ (unchanged)
- ComputeKernelConfig::softmax(): fp32 = parameter (default **false**)
- log_softmax_moreh: uses softmax() → fp32 = parameter

**BERT Results** (tested with clean rebuilds):
- bert-tiny: PCC 0.998 ✅
- bert-small: PCC 0.931 ❌ **NO CHANGE**
- bert-base: PCC 0.805 ❌ **NO CHANGE**

---

## Conclusions

### What Actually Happened:

1. **Custom softmax ALWAYS had FP32**:
   - From commit 6d395416f3 (original implementation) onwards
   - Never touched by any "workaround" commits
   - This is what BERT actually uses

2. **ComputeKernelConfig::softmax() was toggled**:
   - Started with fp32 = true
   - Turned OFF in Feb 2025 (e7e86d78be) due to training explosion
   - Turned back ON in Nov 2025 (668c173f81) claiming to fix BERT
   - Made parameterized in Nov 2025 (9a261d86cc)
   - Default changed to false (d3b81efa58)

3. **BERT was NEVER affected by these toggles**:
   - BERT uses `ttml::metal::softmax`, not `ComputeKernelConfig::softmax()`
   - Test results prove: PCC values unchanged across all "workaround" commits
   - Only bert-tiny passes (PCC 0.998)
   - bert-small and bert-base still fail (PCC 0.93, 0.80)

### The "Workaround" Myth:

The commits claiming to add an "FP32 workaround for BERT" were:
- ❌ Modifying the wrong functions (BERT doesn't use them)
- ❌ Making no actual impact on BERT test results
- ❌ Creating false documentation claiming BERT works

The real state:
- ✅ BERT always had FP32 in the custom softmax
- ✅ Only bert-tiny works (2 layers don't accumulate enough error)
- ✅ bert-small and bert-base fail due to error accumulation across layers
- ✅ The problem is NOT a missing FP32 workaround

---

## References

### Commits Examined:
- `6d395416f3` - Initial softmax implementation
- `e7e86d78be` - FP32 turned off for ComputeKernelConfig::softmax()
- `668c173f81` - "Enable FP32 accumulation" (wrong function)
- `9a261d86cc` - "Add FP32 option" (wrong function)
- `d3b81efa58` - Change default to false

### Files Tracked:
- `sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp` (BERT's softmax)
- `sources/ttml/core/compute_kernel_config.cpp` (Config used by ttnn_fixed)
- `sources/ttml/ops/unary_ops.cpp` (log_softmax_moreh)
- `sources/ttml/ttnn_fixed/trivial_ttnn_ops.cpp` (ttnn_fixed::softmax)
- `sources/ttml/ops/scaled_dot_product_attention.cpp` (BERT attention)

### Test Commands:
```bash
# Clean rebuild at specific commit
git checkout <commit>
cd /workspace/tt-metal/tt-train/
rm -rf build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build --config Debug --clean-first

# Test BERT
export TT_METAL_HOME=/workspace/tt-metal
export LD_LIBRARY_PATH=/workspace/tt-metal/build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/tt-metal/tt-train/build/sources/ttml:$PYTHONPATH
python3 -m pytest tests/python/test_bert_end_to_end_validation.py -v
```
