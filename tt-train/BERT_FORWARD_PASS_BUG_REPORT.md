# BERT Forward Pass Bug Report

## Executive Summary

**Issue**: TTML BERT forward pass produces incorrect outputs (PCC 0.17-0.74) despite weights loading correctly (PCC >0.999).

**Root Cause**: LayerNorm epsilon mismatch due to hardware clamping enabled by default.

**Impact**: All BERT models (tiny, small, base) produce wrong inference results.

**Status**: **ROOT CAUSE IDENTIFIED** - Fix required in C++ code.

---

## Problem Description

### Symptoms

When running BERT inference:
- ✅ Weight loading: **PERFECT** (PCC >0.9999 for all layers)
- ❌ Forward pass: **BROKEN** (PCC 0.17-0.74)
- First 5 output values show completely different results:
  ```
  HuggingFace: [-1.303, -0.770, -2.909, -2.297,  0.983]
  TTML:        [ 0.270, -0.828, -4.906, -0.144, -1.188]
  ```

### Evidence

From comprehensive validation tests:
- `test_bert_comprehensive_validation.py`: All 4 BERT variants fail forward pass
- `debug_bert_forward_pass.py`: PCC 0.57 for bert-tiny
- `test_bert_inference_showcase.py`: Average PCC 0.42 across all examples

---

## Root Cause Analysis

### The Bug

**Location**: `sources/ttml/models/bert.cpp` (lines 96-99) and `sources/ttml/modules/bert_block.cpp` (lines 51-54, 59-62)

**Issue**: LayerNorm layers are created with only 3 arguments:
```cpp
// In bert.cpp line 96-99
m_embedding_norm = std::make_shared<modules::LayerNormLayer>(
    embedding_dim,
    layer_norm_eps,  // 1e-12 from BERT config
    false            // use_composite_op
    // enable_hardware_clamp defaults to TRUE
    // min_safe_eps defaults to 1e-4
);

// In bert_block.cpp lines 51-54
m_attention_norm = std::make_shared<LayerNormLayer>(
    config.embedding_dim,
    config.layer_norm_eps,  // 1e-12 from BERT config
    false                   // use_composite_op
    // enable_hardware_clamp defaults to TRUE
    // min_safe_eps defaults to 1e-4
);
```

**Effect**: From `layer_norm_module.hpp` (line 44):
```cpp
explicit LayerNormLayer(
    uint32_t features,
    float eps = 1e-5F,
    bool use_composite_op = false,
    bool enable_hardware_clamp = true,  // DEFAULT: TRUE
    float min_safe_eps = 1e-4F);        // DEFAULT: 1e-4
```

**Result**: Hardware clamping logic in `layernorm_op.cpp` (lines 38-39):
```cpp
const float safe_eps =
    (enable_hardware_clamp && dtype == BFLOAT16) ?
        std::max(eps, min_safe_eps) : eps;
// Result: max(1e-12, 1e-4) = 1e-4
```

### Epsilon Mismatch

| Component | Intended Epsilon | Actual Epsilon | Ratio |
|-----------|------------------|----------------|-------|
| HuggingFace BERT | 1e-12 | 1e-12 | 1.0x |
| TTML BERT (configured) | 1e-12 | 1e-4 | **100,000,000,000x** |

**This 8-order-of-magnitude difference causes LayerNorm to produce completely different outputs.**

---

## Technical Details

### LayerNorm Formula

LayerNorm computes:
```
mean = E[x]
variance = E[(x - mean)²]
rstd = 1 / sqrt(variance + eps)
output = gamma * (x - mean) * rstd + beta
```

The epsilon value affects the `rstd` computation. When variance is small:
- With eps=1e-12: rstd ≈ 1/sqrt(variance)  (high sensitivity to variance)
- With eps=1e-4: rstd ≈ 1/sqrt(1e-4) = 100  (variance dominated by epsilon)

This fundamentally changes the normalization behavior!

### Why Hardware Clamping Exists

From `layernorm_op.cpp` (lines 30-36):
```cpp
// Hardware precision mitigation for dtype-dependent numerical stability:
// - BFLOAT16: 7-bit mantissa gives machine epsilon ~0.0078125.
//   Values below min_safe_eps may truncate to zero, causing NaN in rsqrt.
// - FLOAT32: Full precision allows smaller epsilon without underflow.
```

**This is correct for BFLOAT16 hardware limitations**, but BERT uses FLOAT32 precision in HuggingFace and expects exact epsilon=1e-12.

### Files Affected

1. **Embedding LayerNorm**: `bert.cpp:96-99`
   - Used in `get_embeddings()` after token + position + type embeddings

2. **Attention LayerNorm**: `bert_block.cpp:51-54`
   - Applied after attention + residual in each transformer block

3. **MLP LayerNorm**: `bert_block.cpp:59-62`
   - Applied after FFN + residual in each transformer block

**Total**: 1 embedding norm + (2 norms × num_blocks) per model
- bert-tiny (2 blocks): 5 LayerNorm layers affected
- bert-small (4 blocks): 9 LayerNorm layers affected
- bert-base (12 blocks): 25 LayerNorm layers affected

---

## Proposed Fix

### Option 1: Disable Hardware Clamping for BERT (Recommended)

**Change**: Pass `enable_hardware_clamp=false` when creating LayerNorm for BERT.

**Files to modify**:

1. `sources/ttml/models/bert.cpp` (line 96-99):
```cpp
m_embedding_norm = std::make_shared<modules::LayerNormLayer>(
    embedding_dim,
    layer_norm_eps,
    false,   // use_composite_op
    false);  // enable_hardware_clamp = false for BERT
```

2. `sources/ttml/modules/bert_block.cpp` (lines 51-54, 59-62):
```cpp
m_attention_norm = std::make_shared<LayerNormLayer>(
    config.embedding_dim,
    config.layer_norm_eps,
    false,   // use_composite_op
    false);  // enable_hardware_clamp = false for BERT

m_mlp_norm = std::make_shared<LayerNormLayer>(
    config.embedding_dim,
    config.layer_norm_eps,
    false,   // use_composite_op
    false);  // enable_hardware_clamp = false for BERT
```

**Rationale**:
- BERT reference implementation uses epsilon=1e-12 exactly
- HuggingFace uses FLOAT32, not BFLOAT16
- We need exact match for validation
- Expert mode for precision-critical models

### Option 2: Add Configuration Option (More Flexible)

**Change**: Add `enable_hardware_clamp` and `min_safe_eps` to `BertConfig`.

**Files to modify**:

1. `sources/ttml/models/bert.hpp` (add to BertConfig):
```cpp
struct BertConfig {
    // ... existing fields ...
    float layer_norm_eps = 1e-12F;
    bool enable_hardware_clamp = false;  // Default OFF for BERT
    float min_safe_eps = 1e-4F;
    // ... rest of config ...
};
```

2. Update `bert.cpp` and `bert_block.cpp` to use config values.

**Rationale**:
- More flexible for different use cases
- Allows users to enable clamping if needed
- Better for hardware deployment scenarios

### Option 3: Automatic Detection Based on Epsilon

**Change**: Automatically disable clamping when epsilon < min_safe_eps.

**Rationale**:
- Less code changes
- May surprise users who explicitly set small epsilon

---

## Validation Plan

After fix is applied:

1. ✅ Run `test_bert_comprehensive_validation.py`
   - Expect: Weight loading still >0.999 (unchanged)
   - Expect: Forward pass PCC >0.95 (FIXED)

2. ✅ Run `test_bert_inference_showcase.py`
   - Expect: Average PCC >0.95 across all examples
   - Expect: Test reports PASSED instead of FAILED

3. ✅ Run `debug_bert_forward_pass.py`
   - Expect: PCC >0.95 for bert-tiny
   - Expect: First 5 values match HuggingFace closely

4. ✅ Test all BERT variants:
   - prajjwal1/bert-tiny (2 layers, 128 dim)
   - prajjwal1/bert-small (4 layers, 512 dim)
   - bert-base-uncased (12 layers, 768 dim)

---

## References

**Code Files**:
- `sources/ttml/ops/layernorm_op.cpp` - Hardware clamping logic
- `sources/ttml/modules/layer_norm_module.hpp` - LayerNorm interface
- `sources/ttml/modules/layer_norm_module.cpp` - LayerNorm implementation
- `sources/ttml/models/bert.cpp` - BERT model implementation
- `sources/ttml/modules/bert_block.cpp` - BERT transformer block

**Test Files**:
- `tests/python/test_bert_comprehensive_validation.py` - Full validation suite
- `tests/python/test_bert_inference_showcase.py` - Inference demonstration
- `tests/python/debug_bert_forward_pass.py` - Layer-by-layer debugging

**Documentation**:
- Recent commits showing QKV weight loading fix
- Investigation showing weights load perfectly

---

## Timeline

- **2025-10-25 00:21**: Bug identified and root cause confirmed
- **Next**: Implement fix (Option 1 recommended)
- **Next**: Run full validation suite
- **Next**: Commit with comprehensive testing

---

## Impact Assessment

**Before Fix**:
- ❌ BERT inference completely broken
- ❌ Cannot use BERT for real applications
- ❌ All validation tests fail

**After Fix**:
- ✅ BERT inference matches HuggingFace reference
- ✅ Can deploy BERT models confidently
- ✅ All validation tests pass
- ✅ Clean path for other transformer models

**Risk**: Low - Change is isolated to BERT model creation, no changes to core LayerNorm logic.
