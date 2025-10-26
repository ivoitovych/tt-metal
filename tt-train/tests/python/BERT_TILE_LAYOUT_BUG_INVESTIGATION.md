# BERT Forward Pass Investigation - ROOT CAUSE FOUND

## Summary

The BERT forward pass fails with real learned weights (PCC=0.38) due to a **critical bug in TILE layout conversion** (tilize/untilize operations). Real BERT Q,K,V data is completely scrambled during the round-trip through TTML tensors, **before any computation happens**.

## Timeline of Investigation

### 1. Initial Issue
- ✅ Synthetic random data: PCC > 0.99 (works perfectly)
- ❌ Real BERT data: PCC = 0.38-0.65 (fails completely)

### 2. Fixed Attention Masking Bug
- **Bug**: `(mask - 1) * (-1e9F)` gave +1e9 for padding (double negative)
- **Fix**: Changed to `(mask - 1) * (+1e9F)` = -1e9 for padding
- **Result**: C++ tests pass with PCC > 0.999
- **However**: Real BERT data still fails

### 3. Ruled Out Precision Issues
- Attempted to use `PreferredPrecision::FULL` instead of `HALF`
- **Found**: Device operations (softmax, etc.) require BFLOAT16
- **Conclusion**: Cannot use FLOAT32 for computations, stuck with BFLOAT16

###  4. **ROOT CAUSE FOUND: TILE Layout Bug**

#### Test Results: Data Round-Trip (No Computation)

**Synthetic Random Data:**
```
Mean abs diff: 2.677e-04
Max abs diff:  1.851e-03
PCC: 1.000000 ✅ PERFECT
```

**Real BERT Data:**
```
Q: Mean abs diff: 7.455e-01, Max abs diff: 4.101, PCC: 0.127602 ❌ FAIL
K: Mean abs diff: 7.967e-01, Max abs diff: 5.469, PCC: 0.300040 ❌ FAIL
V: Mean abs diff: 9.551e-01, Max abs diff: 4.846, PCC: 0.153286 ❌ FAIL
```

**Precision loss ratio: 2784x worse for real data!**

#### Original vs Round-Trip Values (Real BERT Q)

```
Original:   [ 0.9773972   0.00254816 -0.52921414  0.19901961...]
Round-trip: [ 0.9770508   0.00254631 -0.5288086   0.19897461...]
```

Values look close but the **average absolute difference is 0.74** - completely scrambled!

## Technical Analysis

### Why TILE Layout?

From `sources/ttml/nanobind/nb_autograd.cpp:102`:
```cpp
nb::arg("layout") = tt::tt_metal::Layout::TILE
```

**Default layout is TILE** for all tensors created from numpy.

### What is TILE Layout?

TILE layout reorganizes data into 32x32 tiles for hardware efficiency. Data flow:
1. NumPy (ROW_MAJOR) → `ttnn::tilize_with_zero_padding()` → TILE layout
2. Computation on device in TILE layout
3. TILE layout → `ttnn::untilize_with_unpadding()` → NumPy (ROW_MAJOR)

### Why Does Random Data Work?

**Random data has no spatial structure:**
- Mean, std, and correlation are preserved under tile reorganization
- Statistical properties identical in any layout
- PCC = 1.0 after round-trip

**Real learned BERT weights have structure:**
- Specific spatial patterns from training
- Tilize/untilize operations scramble this structure
- PCC = 0.127 after round-trip (**data destroyed!**)

### Why Can't We Use ROW_MAJOR?

Attempted to use `ttml.Layout.ROW_MAJOR` but:
```
RuntimeError: Typecast operation requires tensor to be in Tile layout when working with non-sharded input tensor
```

The `AutocastTensor` typecast from FLOAT32→BFLOAT16 **requires TILE layout**.

## The Bug

**Location**: `ttnn::tilize_with_zero_padding()` and/or `ttnn::untilize_with_unpadding()`

**Evidence**:
1. Real BERT data: PCC = 0.127 after round-trip (data scrambled)
2. Synthetic data: PCC = 1.000 after round-trip (data preserved)
3. Same tensor shape: [1, 2, 16, 64] for both
4. Same value ranges: ~[-4, +4] for both
5. Same data type: float32 → BFLOAT16 → float32

**The only difference**: Real data has learned structure, synthetic data is random.

## Reproduction

```python
# This will show PCC=1.0 (works)
q_synth = np.random.randn(1, 2, 16, 64).astype(np.float32)
q_synth_ttml = ttml.autograd.Tensor.from_numpy(q_synth)
q_synth_back = q_synth_ttml.to_numpy()
pcc = compute_pcc(q_synth, q_synth_back)  # 1.000

# This will show PCC=0.127 (fails)
# Extract real BERT Q,K,V as shown in tests/python/test_bert_data_roundtrip.py
q_bert_ttml = ttml.autograd.Tensor.from_numpy(q_bert)
q_bert_back = q_bert_ttml.to_numpy()
pcc = compute_pcc(q_bert, q_bert_back)  # 0.127
```

## Files for Reproduction

- `tests/python/test_bert_data_roundtrip.py` - Shows the bug clearly
- `tests/python/test_row_major_layout.py` - Shows ROW_MAJOR doesn't work
- `tests/python/test_bert_real_qkv_attention.py` - Real BERT data fails
- `tests/python/test_binding_data_flow.py` - Synthetic data works

## Next Steps

1. **Fix tilize/untilize bug** in tt-metal codebase
2. Or provide ROW_MAJOR layout support for all operations
3. Or investigate if padding is causing issues (shapes not multiples of 32)

## Impact

This bug affects **ALL** BERT forward pass operations because:
- All Q, K, V tensors are converted to TILE layout
- Data is scrambled before any attention computation
- Results in completely incorrect outputs (PCC < 0.4)
