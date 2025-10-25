# BERT Forward Pass Investigation Summary

## Current Status (2025-10-25)

### What We Know

1. **Weight Loading**: ✅ **PERFECT** (PCC >0.9999)
   - All weights load correctly from HuggingFace to TTML
   - Token embeddings, QKV weights, FFN weights, LayerNorm parameters all match
   - Verified by `test_bert_comprehensive_validation.py`

2. **Forward Pass**: ❌ **BROKEN** (PCC 0.17-0.74)
   - All BERT model variants produce incorrect outputs
   - bert-tiny: PCC 0.57-0.74
   - bert-small: PCC 0.38
   - bert-base: PCC 0.17
   - First values show opposite signs and large differences

### Investigation Progress

#### Issue #1: LayerNorm Epsilon Clamping (IDENTIFIED)

**Finding**: BERT LayerNorm layers use hardware-clamped epsilon instead of BERT's intended epsilon.

**Details**:
- BERT requires epsilon = 1e-12
- TTML LayerNorm defaults to `enable_hardware_clamp=true` and `min_safe_eps=1e-4`
- For BFLOAT16 tensors, epsilon gets clamped: `max(1e-12, 1e-4) = 1e-4`
- This is **8 orders of magnitude** larger than intended!

**Fix Applied** (bert.cpp and bert_block.cpp):
```cpp
m_embedding_norm = std::make_shared<modules::LayerNormLayer>(
    embedding_dim,
    layer_norm_eps,  // 1e-12
    false,           // use_composite_op
    false);          // enable_hardware_clamp = false (NEW)
```

**Result**: ❓ **FIX DID NOT RESOLVE ISSUE**
- After applying fix and rebuilding, PCC still 0.57
- Suggests either:
  a) Tensors are not using BFLOAT16 (so clamping never activated)
  b) There are additional issues beyond epsilon
  c) The fix wasn't applied correctly

#### Observations

1. **Output Magnitude**: Outputs are completely different, not just slightly off
   ```
   HF:   [-1.303, -0.770, -2.909, -2.297,  0.983]
   TTML: [ 0.270, -0.828, -4.906, -0.144, -1.188]
   ```
   This suggests a fundamental computation difference, not just numerical precision.

2. **Sign Flips**: First value is negative in HF (-1.303) but positive in TTML (+0.270)
   - This is unlikely to be caused by epsilon alone
   - Suggests possible weight layout or computation order issue

3. **Debug Output**: MultiHeadAttention shows correct shapes
   - Input shapes match expected dimensions
   - No obvious shape mismatches

### Potential Root Causes (Beyond Epsilon)

1. **Attention Mechanism Issues**:
   - QKV weight layout verified correct (PCC >0.999)
   - But maybe QKV computation/splitting is wrong?
   - Scaled dot-product attention implementation?
   - Attention masking?

2. **Activation Functions**:
   - GELU implementation uses `ttnn::gelu()`
   - Need to verify this matches HuggingFace's GELU exactly

3. **Residual Connections**:
   - BERT uses post-norm: `LayerNorm(x + Sublayer(x))`
   - Maybe residual add is in wrong order?

4. **Precision/Dtype Issues**:
   - Maybe tensors are BFLOAT16 throughout?
   - HuggingFace uses FLOAT32
   - Large precision difference could accumulate

5. **Tensor Layout/Memory Format**:
   - TTML uses 4D tensors: `[batch, 1, seq_len, hidden_dim]`
   - HuggingFace uses 3D: `[batch, seq_len, hidden_dim]`
   - Maybe reshape/view operations are wrong?

### Next Steps

1. **Verify Epsilon Fix Actually Applied**:
   - Add logging to confirm `enable_hardware_clamp=false` is being used
   - Check actual dtype of tensors during forward pass

2. **Isolate the Divergence Point**:
   - Compare embeddings BEFORE first attention block
   - Compare first attention block output only
   - Identify exact layer where divergence starts

3. **Test Individual Components**:
   - Test LayerNorm in isolation with BERT's epsilon
   - Test GELU activation in isolation
   - Test attention mechanism with known inputs/outputs

4. **Check Tensor Dtypes**:
   - Log actual dtypes being used
   - Verify if BFLOAT16 or FLOAT32

5. **Compare Attention Implementation**:
   - Review scaled_dot_product_attention implementation
   - Compare with HuggingFace's BertSelfAttention
   - Check masking logic

### Files Modified

1. `sources/ttml/models/bert.cpp` - Added `enable_hardware_clamp=false` to embedding_norm
2. `sources/ttml/modules/bert_block.cpp` - Added `enable_hardware_clamp=false` to attention_norm and mlp_norm
3. `sources/ttml/modules/multi_head_attention.cpp` - Commented out debug prints

### Test Files Created

1. `tests/python/debug_bert_forward_pass.py` - Layer-by-layer debugging
2. `tests/python/test_layernorm_epsilon_verification.py` - Epsilon verification (incomplete - LayerNormLayer not exposed to Python)
3. `tests/python/test_bert_embeddings_only.py` - Embeddings-only test
4. `BERT_FORWARD_PASS_BUG_REPORT.md` - Comprehensive bug report (epsilon issue)
5. `BERT_INVESTIGATION_SUMMARY.md` - This file

### Questions to Answer

1. ❓ What dtype are TTML tensors actually using during forward pass?
2. ❓ Does the embedding output (before first attention) match HF?
3. ❓ Which specific layer/operation causes the first divergence?
4. ❓ Is the attention mechanism implementation correct?
5. ❓ Why did the epsilon fix not improve PCC?

### Recommendations

**Short-term**:
1. Add extensive logging to track dtypes and intermediate values
2. Create minimal reproduction case (single attention block)
3. Compare HuggingFace and TTML attention implementations line-by-line

**Long-term**:
1. Consider adding dtype configuration to BertConfig
2. Add validation tests for each BERT component in isolation
3. Create reference implementation tests comparing each operation

---

## Summary

**Root cause of epsilon issue**: IDENTIFIED and FIXED (but didn't resolve forward pass)

**Root cause of forward pass failure**: STILL UNKNOWN

The forward pass issue is more complex than just epsilon. The magnitude of the differences (opposite signs, order-of-magnitude variations) suggests a fundamental implementation difference in:
- Attention mechanism
- Tensor layouts/reshapes
- Activation functions
- Or combination of precision + accumulated errors

**Next immediate action**: Run `test_bert_embeddings_only.py` to see if embeddings match before first attention block. This will tell us if the issue is in embeddings or in attention.
