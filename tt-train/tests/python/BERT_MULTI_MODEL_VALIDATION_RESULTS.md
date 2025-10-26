# BERT Multi-Model Layer-by-Layer Validation Results

**Date**: 2025-10-26
**Purpose**: Comprehensive validation of TTML BERT implementation across 4 model variants
**Test**: `test_bert_layer_by_layer_multi_model.py`

## Executive Summary

Layer-by-layer validation across 4 BERT model variants reveals **dimension-dependent divergence**:
- **bert-tiny (128 hidden)**: Works well (avg PCC=0.952)
- **bert-small (512 hidden)**: Fails at embeddings (avg PCC=0.759)
- **bert-base (768 hidden)**: Catastrophic accumulation (avg PCC=0.458)

## Test Configuration

**Models Tested**:
1. `prajjwal1/bert-tiny`: 2 layers, 128 hidden, 2 heads
2. `prajjwal1/bert-small`: 4 layers, 512 hidden, 8 heads
3. `google/bert_uncased_L-4_H-512_A-8`: 4 layers, 512 hidden, 8 heads (official)
4. `bert-base-uncased`: 12 layers, 768 hidden, 12 heads

**Input Configuration**:
- Batch size: 1
- Sequence length: 32
- Input IDs: `[542, 67, 876, 414, 26, 335, 620, 924, 950, 113, ...]` (deterministic)
- Pass threshold: PCC ≥ 0.95

**Layers Validated Per Model**:
- Embedding layer output
- Each block's attention output (after attention + residual + norm)
- Each block's final output (after FFN + residual + norm)
- Final model output

## Detailed Results

### 1. bert-tiny (2 layers, 128 hidden, 2 heads)

**Summary**: 3/6 layers passed, avg PCC=0.952

**PCC Progression**:
1. ✅ Embeddings: 0.950881
2. ✅ Block 0 Attention: 0.961650
3. ✅ Block 0 Output: 0.967731 (best!)
4. ❌ Block 1 Attention: 0.937867 (first failure)
5. ❌ Block 1 Output: 0.947196
6. ❌ Final Output: 0.947196

**Key Observations**:
- Embeddings barely pass (0.951 vs 0.95 threshold)
- Block 0 actually IMPROVES the signal (0.968)
- Block 1 shows slight degradation
- Overall: **Mostly functional** - all PCCs very close to threshold

**Error Magnitudes**:
- Mean diff: 0.16 - 0.37
- Max diff: 0.95 - 4.92

---

### 2. bert-small (4 layers, 512 hidden, 8 heads)

**Summary**: 0/10 layers passed, avg PCC=0.759

**PCC Progression**:
1. ❌ Embeddings: 0.650450 (FIRST FAILURE)
2. ❌ Block 0 Attention: 0.858347
3. ❌ Block 0 Output: 0.808649
4. ❌ Block 1 Attention: 0.884777
5. ❌ Block 1 Output: 0.775711
6. ❌ Block 2 Attention: 0.851549
7. ❌ Block 2 Output: 0.792588
8. ❌ Block 3 Attention: 0.714213
9. ❌ Block 3 Output: 0.626709
10. ❌ Final Output: 0.626709

**Key Observations**:
- **Embedding layer fails immediately** (PCC=0.650)
- Attention layers show better PCC (0.85-0.88) than outputs (0.62-0.81)
- Progressive degradation through layers (0.858 → 0.627)
- Overall: **Significant divergence from start**

**Error Magnitudes**:
- Mean diff: 0.37 - 0.51
- Max diff: 3.12 - 17.34 (huge spikes in attention!)

---

### 3. google/bert_uncased_L-4_H-512_A-8 (4 layers, 512 hidden, 8 heads)

**Summary**: 0/10 layers passed, avg PCC=0.759

**PCC Progression**:
**IDENTICAL to bert-small** - same architecture produces identical results:
1. ❌ Embeddings: 0.650450
2-10. ❌ All other layers: same as bert-small

**Key Observations**:
- Confirms divergence is **architecture-dependent**, not model-specific
- Official Google model shows same issues as community model
- This rules out weight loading bugs (different sources, same behavior)

---

### 4. bert-base-uncased (12 layers, 768 hidden, 12 heads)

**Summary**: 1/25 layers passed, avg PCC=0.458

**PCC Progression**:
1. ✅ Embeddings: 0.976448 (GOOD!)
2. ❌ Block 0 Attention: 0.930762
3. ❌ Block 0 Output: 0.892515
4. ❌ Block 1 Attention: 0.671892
5. ❌ Block 1 Output: 0.612508
6. ❌ Block 2 Attention: 0.571166
7. ❌ Block 2 Output: 0.410529
8. ❌ Block 3 Attention: 0.439559
9. ❌ Block 3 Output: 0.290501
10. ❌ Block 4 Attention: 0.374025
11. ❌ Block 4 Output: 0.220098
12. ❌ Block 5 Attention: 0.224409
13. ❌ Block 5 Output: 0.120852
14. ❌ Block 6 Attention: 0.048697
15. ❌ Block 6 Output: **-0.008007** (NEGATIVE CORRELATION!)
16. ❌ Block 7 Attention: 0.083059
17. ❌ Block 7 Output: 0.075267
18. ❌ Block 8 Attention: 0.138311
19. ❌ Block 8 Output: 0.126827
20. ❌ Block 9 Attention: 0.152846
21. ❌ Block 9 Output: 0.143370
22. ❌ Block 10 Attention: 0.175085
23. ❌ Block 10 Output: 0.165321
24. ❌ Block 11 Attention: 0.173513
25. ❌ Block 11 Output: 0.163312
26. ❌ Final Output: 0.163312

**Key Observations**:
- Embeddings are EXCELLENT (PCC=0.976)
- **Catastrophic progressive degradation** through layers
- Block 6 output achieves **NEGATIVE correlation** (-0.008)
- After Block 6, PCC stabilizes around 0.10-0.17 (essentially random)
- Overall: **Catastrophic error accumulation**

**Error Magnitudes**:
- Mean diff: 0.06 (embeddings) → 0.86 (final)
- Max diff: 0.41 (embeddings) → 90.68 (Block 7 attention!)

---

## Cross-Model Analysis

### Divergence by Hidden Dimension

| Model | Hidden Dim | Avg PCC | Embeddings PCC | First Failure | Status |
|-------|-----------|---------|----------------|---------------|--------|
| bert-tiny | 128 | **0.952** | 0.951 | Block 1 Attention | Mostly OK |
| bert-small | 512 | 0.759 | **0.650** | Embeddings | Poor from start |
| google/bert* | 512 | 0.759 | **0.650** | Embeddings | Poor from start |
| bert-base | 768 | 0.458 | 0.976 | Block 0 Attention | Catastrophic |

\* google/bert_uncased_L-4_H-512_A-8

### Key Pattern: Dimension Dependency

**128 hidden**: Works well (~0.95 PCC throughout)
- Embeddings pass
- Blocks mostly pass or borderline
- No catastrophic accumulation

**512 hidden**: Fails at embeddings (~0.65 PCC)
- Embedding layer cannot maintain precision
- Never recovers from initial divergence
- PCC oscillates 0.62-0.88 through layers

**768 hidden**: Good start, catastrophic accumulation
- Embeddings excellent (0.976)
- Rapid degradation through layers
- By layer 6: effectively random correlation
- Suggests per-layer error compounds multiplicatively

### Attention vs Output Patterns

Across all failing models:
- **Attention outputs** show higher PCC than block outputs
- **FFN + residual + norm** appears to degrade signal further
- Suggests issue may be in:
  - FFN implementation (dense layers, GELU)
  - Layer normalization for larger dimensions
  - Residual connection accumulation

### Max Diff Spikes

All models show enormous max_diff values in attention layers:
- bert-tiny: max 4.92
- bert-small: max **17.34**
- bert-base: max **90.68**

This suggests **extreme outliers** in attention computation, possibly:
- Softmax numerical instability
- Attention score overflow
- QKV projection issues for large dimensions

## Root Cause Hypotheses

### Hypothesis 1: Numerical Precision Issues
**Evidence**:
- Larger hidden dimensions diverge more
- Error accumulates layer by layer
- Max diffs show extreme outliers

**Suspect Components**:
- Layer normalization epsilon handling (despite fix to 1e-12)
- Attention softmax precision
- Matrix multiplication accumulation

### Hypothesis 2: Operator Implementation for Large Tensors
**Evidence**:
- 128-dim works, 512/768-dim fails
- Embedding layer fails for 512-dim but passes for 768-dim
- Suggests dimension-specific code paths or optimizations

**Suspect Components**:
- Embedding lookup for large vocabularies
- Matrix operations tile/blocking strategies
- Memory layout for different tensor sizes

### Hypothesis 3: Attention Mechanism Scaling
**Evidence**:
- Attention layers show extreme max_diff spikes
- 2 heads work, 8/12 heads fail worse
- Attention outputs consistently worse than embeddings

**Suspect Components**:
- QKV projection transpose operations
- Attention score scaling (1/√d_k)
- Softmax numerical stability
- Multi-head concatenation/reshape

### Hypothesis 4: FFN/GELU Implementation
**Evidence**:
- Block outputs consistently worse than attention outputs
- FFN processes larger intermediate_size (512→2048, 768→3072)

**Suspect Components**:
- GELU activation approximation
- Dense layer implementation for large tensors
- Dropout (though set to 0.0)

## Recommended Next Steps

### Immediate Investigation (Priority 1)

1. **Isolate Embedding Layer for 512-dim**
   - Test standalone embedding layer with 512 hidden dim
   - Compare word embeddings, position embeddings, token type embeddings individually
   - Check LayerNorm on embeddings

2. **Test Attention with Synthetic Inputs**
   - Create unit test for multi-head attention with 512/768 hidden dim
   - Use known input patterns to verify QKV projections
   - Inspect intermediate attention scores and softmax outputs

3. **Verify Layer Normalization for Large Dimensions**
   - Despite epsilon fix, test LayerNorm standalone with 512/768 dims
   - Check for numerical instability in variance computation
   - Verify gamma/beta parameter loading and application

### Deep Dive (Priority 2)

4. **Profile Attention Score Distribution**
   - Log min/max/mean of attention scores before softmax
   - Check for overflow/underflow in score computation
   - Verify scaling factor (1/√d_k) is correct for each model

5. **FFN Component Testing**
   - Test GELU activation for large tensors
   - Verify dense layer weight multiplication precision
   - Check intermediate activations (embedding_dim → intermediate_size → embedding_dim)

6. **Compare Operator Precision**
   - Run matmul tests with realistic BERT dimensions
   - Test reshape/transpose for multi-head attention patterns
   - Verify concat operations maintain precision

### Validation (Priority 3)

7. **Incremental Model Testing**
   - Test intermediate sizes: 256, 384, 512 hidden dims
   - Identify exact dimension threshold where failure begins
   - Test with fewer layers to isolate accumulation effect

8. **Attention Head Scaling**
   - Test bert-base with reduced heads (3, 6 heads instead of 12)
   - Check if head count correlates with divergence

## Conclusion

The layer-by-layer validation successfully identified:
1. **Clear dimension-dependent divergence pattern**
2. **Embedding layer failure at 512-dim** (but not 768-dim - interesting!)
3. **Catastrophic error accumulation in deep models** (12 layers)
4. **Attention mechanism shows extreme outliers** (max_diff up to 90.68)

The TTML BERT implementation has a **critical dimension-dependent bug** that prevents it from matching HuggingFace reference for production model sizes (512+ hidden dimensions).

**Next Action**: Investigate embedding layer and attention mechanism for 512-dim models, as these are the first points of failure.
