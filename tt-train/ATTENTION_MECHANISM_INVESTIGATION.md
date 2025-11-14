# BERT Attention Mechanism Bug Investigation Report

**Date**: November 14, 2025
**Status**: 🔍 **INVESTIGATION IN PROGRESS**
**Branch**: `ivoitovych/bert-model-for-ttml-task-heads-v2`
**Severity**: P0 CRITICAL BLOCKER

---

## Executive Summary

Investigation into the critical bug causing 3-6% immediate error in Block 0 Attention and catastrophic error accumulation in deep BERT models. Layer-by-layer PCC analysis revealed that embeddings are perfect (PCC 1.0), but attention mechanism introduces significant errors from the very first layer.

**Key Findings**:
- ✅ **Scaling order is NOT the root cause** - PyTorch bfloat16 achieves PCC >0.9999 with both pre-scale and post-scale approaches
- 🔍 **Bug is likely in TTNN operation implementations** - Not in high-level algorithm logic
- 🔴 **Attention error manifests immediately** - Block 0 shows PCC 0.94-0.97 (3-6% error)
- 🔴 **Error compounds exponentially** - bert-base reaches negative PCC by layer 6

---

## Investigation Timeline

### 1. Initial Hypothesis: Scaling Order Bug

**Hypothesis**: Pre-scaling query before matmul `(Q * scale) @ K^T` loses precision compared to post-matmul scaling `(Q @ K^T) * scale`

**Test Method**: Created `tests/python/test_attention_scaling_order.py` to compare:
- Standard order: `(Q @ K^T) * scale`
- Pre-scale order: `(Q * scale) @ K^T` (current TTML implementation)

**Results**:
```
Embedding   Head    Scale    Standard    Pre-scale     Diff
   Dim      Dim    Factor       PCC          PCC
------------------------------------------------------------
    64        5   0.433013   0.999986     0.999990   -0.000004
   128       10   0.306186   0.999985     0.999988   -0.000003
   256       21   0.216506   0.999985     0.999988   -0.000003
   512       42   0.153093   0.999986     0.999989   -0.000003
   768       64   0.125000   0.999990     0.999990    0.000000
```

**Conclusion**: ✅ **HYPOTHESIS REJECTED**
Both scaling orders achieve PCC >0.9999 in PyTorch bfloat16. The bug is NOT in the algorithm logic.

---

## Code Analysis

### 1. Scaled Dot-Product Attention (`sources/ttml/ops/scaled_dot_product_attention.cpp`)

**Current Implementation** (lines 149-185):
```cpp
const float scale = 1.0F / std::sqrt(static_cast<float>(embedding_dim));
auto q_scaled = ttnn::multiply(query->get_value(), scale, ...);
ttnn::Tensor qk_scaled = group_shared_matmul(q_scaled, key_tensor, false, true);

if (mask) {
    // Apply mask: qk_masked = mask * qk + (mask - 1) * (1e9)
    qk_scaled = ttnn::add(
        ttnn::multiply(mask_tensor, qk_scaled, ...),
        ttnn::multiply(
            ttnn::subtract(mask_tensor, 1.F, ...),
            1e9F,
            ...),
        ...);
}

auto attention_weights = ttml::metal::softmax(qk_scaled, 3);
ttnn::Tensor attention_qkv = group_shared_matmul(attention_weights, value->get_value(), false, false);
```

**Observations**:
1. ✅ Pre-scaling is mathematically equivalent to post-scaling (verified by test)
2. ⚠️ **Mask application uses 4 operations** instead of single masked fill
   - Could compound bfloat16 rounding errors
   - Uses `1e9` instead of standard `-inf` or `-1e4`
3. ⚠️ **Softmax delegates to TTNN primitive** - cannot verify numerical stability without kernel source

### 2. Multi-Head Attention (`sources/ttml/modules/multi_head_attention.cpp`)

**Implementation** (lines 26-52):
```cpp
auto qkv = (*m_qkv_linear)(x);
auto [query_with_heads, key_with_heads, value_with_heads] = ops::heads_creation(qkv, m_num_heads);
auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);
attention = ops::heads_fusion(attention);
auto out = (*m_out_linear)(attention);
```

**Observations**:
- Uses manual head splitting/joining (workaround for TTNN bug when head_dim < 32)
- Multiple reshape + transpose operations may introduce numerical errors

### 3. Head Splitting (`sources/ttml/ops/multi_head_utils.cpp`)

**Manual Implementation** (lines 89-104):
```cpp
// [B, 1, S, E] -> [B, S, E] -> [B, S, H, E/H] -> [B, H, S, E/H]
auto q_no_channel = ttnn::reshape(q_flat, ttnn::Shape{batch_size, seq_len, embedding_dim});
auto q_with_heads = ttnn::reshape(q_no_channel, ttnn::Shape{batch_size, seq_len, num_heads, head_dim});
auto q = ttnn::transpose(q_with_heads, 1, 2);
```

**Observations**:
- ⚠️ **Transpose in tile layout** may not preserve exact numerical values
- ⚠️ **Multiple reshape operations** - each introduces potential rounding in bfloat16

### 4. Group Shared Matmul (`sources/ttml/ops/scaled_dot_product_attention.cpp:33-67`)

**For Multi-Head Attention with Grouped Queries**:
```cpp
auto query_tensor_grouped = ttnn::reshape(query_tensor, ttnn::Shape{batch_num * groups, heads / groups, seq_len, embedding_dim});
auto kv_tensor_batched = ttnn::reshape(kv_tensor, ttnn::Shape{batch_num * groups, 1U, seq_len_v, embedding_dim_v});
ttnn::Tensor kv_tensor_repeated = ttnn::repeat(kv_tensor_batched, ttnn::Shape{1U, heads / groups, 1U, 1U});
auto bcasted_mm = ttnn_fixed::matmul(query_tensor_grouped, kv_tensor_repeated, transpose_a, transpose_b);
auto reshaped_mm = ttnn::reshape(bcasted_mm, ttnn::Shape{batch_num, heads, M, N});
```

**Observations**:
- ⚠️ **Complex reshape + repeat + matmul + reshape sequence**
- ⚠️ **Each operation introduces bfloat16 rounding errors**

---

## Root Cause Analysis

### Evidence Summary

1. **Embeddings Perfect** (PCC 1.0) ✅
   - Proves embedding batch bug fix works correctly
   - Error does NOT originate from embeddings

2. **Scaling Order Not the Cause** ✅
   - PyTorch bfloat16 achieves PCC >0.9999 with both orders
   - Algorithm logic is correct

3. **Immediate Error in Block 0 Attention** 🔴
   - bert-tiny: PCC 0.9714 (2.9% error)
   - bert-small: PCC 0.9496 (5.0% error)
   - bert-base: PCC 0.9423 (5.8% error)
   - **Not gradual accumulation** - it's an immediate bug

4. **Error Scales with Model Size** 🔴
   - Larger models (more heads) = worse error
   - Suggests issue in multi-head operations or tile layout

### Hypotheses Ranked by Likelihood

#### Hypothesis 1: TTNN Transpose/Reshape Precision Loss (HIGH PROBABILITY)

**Evidence**:
- Head splitting uses multiple transpose/reshape operations
- Tile layout transformations may not preserve exact values in bfloat16
- Error scales with number of heads (more heads = more transposes)

**Test**:
- Compare TTNN transpose output vs PyTorch transpose on device
- Profile precision loss at each reshape/transpose step

#### Hypothesis 2: TTNN Matmul Numerical Precision (MEDIUM PROBABILITY)

**Evidence**:
- `ttnn_fixed::matmul` uses specific compute kernel config
- Matmul is the most numerically intensive operation
- Three matmuls per attention layer (QK, QK@V, and repeated operations in group_shared_matmul)

**Test**:
- Compare TTNN matmul vs PyTorch matmul for same inputs
- Test different compute kernel configs (precise vs default)

#### Hypothesis 3: Softmax Numerical Stability (LOW PROBABILITY)

**Evidence**:
- Softmax uses `ttml::metal::softmax` which delegates to TTNN primitive
- Code comment says "stable softmax" but cannot verify implementation
- Attention weights look reasonable in intermediate analysis

**Test**:
- Compare TTNN softmax output vs PyTorch softmax
- Check if numerical stability trick (subtract max) is implemented

#### Hypothesis 4: Attention Mask Application (LOW PROBABILITY)

**Evidence**:
- Uses 4 operations instead of single masked fill
- Could compound rounding errors
- Only affects masked positions

**Test**:
- Run attention without mask to see if error persists
- If error remains, mask is not the cause

---

## Comparison with Previous Investigation

### BERT_BATCH_PROCESSING_INVESTIGATION_REPORT.md (November 14, 2025)

**Previous Finding**:
> "Isolated layers work correctly - Each layer achieves PCC > 0.999 when fed reference inputs"

**Current Finding**:
- Block 0 Attention shows PCC 0.94-0.97, NOT > 0.999
- **Reconciliation**: Previous isolated layer tests fed HuggingFace reference inputs at each layer
- Current cumulative test uses TTML outputs as inputs to next layer
- **Conclusion**: Attention mechanism is numerically unstable and sensitive to input perturbations

---

## Next Steps

### Priority 1: Isolate TTNN Operations

1. **Test TTNN Transpose Precision**
   ```python
   # Compare TTNN transpose vs PyTorch transpose
   input_tensor = create_test_tensor()
   ttnn_output = ttnn::transpose(input_tensor, 1, 2)
   pytorch_output = torch.transpose(input_tensor, 1, 2)
   pcc = compare(ttnn_output, pytorch_output)
   ```

2. **Test TTNN Matmul Precision**
   ```python
   # Compare TTNN matmul vs PyTorch matmul
   q, k = create_test_tensors()
   ttnn_output = ttnn_fixed::matmul(q, k, transpose_b=True)
   pytorch_output = torch.matmul(q, k.transpose(-2, -1))
   pcc = compare(ttnn_output, pytorch_output)
   ```

3. **Test TTNN Reshape Precision**
   ```python
   # Test if reshape preserves numerical values
   input_tensor = create_test_tensor()
   reshaped = ttnn::reshape(input_tensor, new_shape)
   restored = ttnn::reshape(reshaped, original_shape)
   pcc = compare(input_tensor, restored)
   ```

### Priority 2: Profile Attention Step-by-Step

1. **Create Granular Attention Test**
   - Log PCC after each operation: multiply, matmul, softmax, etc.
   - Identify exactly where precision is lost

2. **Test Different Compute Kernel Configs**
   - Try `ComputeKernelConfig::precise()` instead of default
   - Compare precision vs performance trade-off

3. **Test Single-Head vs Multi-Head**
   - Test with num_heads=1 to eliminate transpose operations
   - If error disappears, confirms transpose is the issue

### Priority 3: Examine TTNN Source Code

1. **Check ttnn::transpose implementation**
   - Look for tile layout conversion code
   - Verify numerical precision handling

2. **Check ttnn::matmul kernel**
   - Verify accumulation precision
   - Check for known bugs/workarounds

---

## Test Files

### Created

1. **`tests/python/test_attention_scaling_order.py`** - Tests scaling order hypothesis
   - Result: Scaling order is NOT the root cause
   - Both orders achieve PCC >0.9999 in PyTorch bfloat16

### Existing

1. **`tests/python/test_bert_layer_pcc_report.py`** - Layer-by-layer PCC analysis
   - Identified Block 0 Attention as first error point
   - Comprehensive report across 4 model sizes

2. **`tests/python/test_bert_isolated_layer_validation.py`** - Isolated layer testing
   - Tests individual layers with reference inputs
   - Previous results showed PCC >0.999 (needs re-validation)

---

## Key Code Locations

1. **Attention Mechanism**:
   - `sources/ttml/ops/scaled_dot_product_attention.cpp:139-244` - Main attention implementation
   - `sources/ttml/modules/multi_head_attention.cpp:26-52` - Multi-head wrapper

2. **Head Operations**:
   - `sources/ttml/ops/multi_head_utils.cpp:59-143` - Manual head splitting/joining
   - Uses workaround for TTNN bug when head_dim < 32

3. **TTNN Fixed Operations**:
   - `sources/ttml/ttnn_fixed/matmuls.cpp:10-25` - Matmul with compute kernel config
   - `sources/ttml/ttnn_fixed/trivial_ttnn_ops.cpp:24-44` - Softmax implementation

4. **Softmax**:
   - `sources/ttml/metal/ops/softmax/softmax.cpp:11-13` - Delegates to TTNN primitive
   - `sources/ttml/metal/ops/softmax/device/softmax_device_operation.cpp` - Device implementation

---

## Blocker Status

🚨 **P0 CRITICAL BLOCKER** for production deployment

**Severity**: Critical
**Impact**: All models with > 2 layers are unusable in production
- bert-tiny (2 layers): Marginally acceptable (PCC 0.95)
- bert-small (4 layers): Unusable (PCC 0.67)
- bert-base (12 layers): Completely broken (PCC 0.04)

**Priority**: Immediate investigation required to identify TTNN operation causing precision loss

---

## Related Documentation

1. **`BERT_ERROR_ACCUMULATION_INVESTIGATION.md`** - Layer-by-layer PCC analysis results
2. **`BERT_BATCH_PROCESSING_INVESTIGATION_REPORT.md`** - Previous investigation (updated with new findings)
3. **`EMBEDDING_BATCH_BUG_FIX.md`** - Embedding bug fix (resolved, PCC 1.0)
4. **`BERT_LAYER_PCC_REPORT.txt`** - Raw test output from layer-by-layer analysis

---

**Report Generated**: November 14, 2025
**Investigator**: Claude Code
**Status**: Investigation in progress - Next step: Test TTNN transpose/reshape/matmul precision
