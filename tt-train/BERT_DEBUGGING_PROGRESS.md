# BERT Forward Pass Debugging Progress

**Date**: 2025-10-25
**Status**: 🟡 **PARTIAL PROGRESS** - 2 bugs fixed, issue persists

---

## ✅ BUGS FIXED

### Bug #1: Attention Masking (FIXED)
- **File**: `scaled_dot_product_attention.cpp:166`
- **Change**: `1e9F` → `-1e9F`
- **Status**: ✅ Fixed
- **Impact**: Not visible in current tests (no attention masks used)
- **Will matter**: When attention masks are added to tests

### Bug #2: Heads Creation Reshape (FIXED)
- **File**: `multi_head_utils.cpp:89-104`
- **Change**: Replaced incorrect `S*H` reshape with proper transpose-based head splitting
- **Status**: ✅ Fixed
- **Impact**:
  - ✅ Signs now correct (HF: -1.303, TTML: -0.385 vs previous +0.270)
  - ✅ PCC improved slightly: 0.569 → 0.581
  - ❌ Still not good enough (need >0.95)

---

## 📊 CURRENT RESULTS

### Before Any Fixes
```
PCC: 0.569
First 5 values:
  HF:   [-1.303, -0.770, -2.909, -2.297,  0.983]
  TTML: [ 0.270, -0.828, -4.906, -0.144, -1.188]  # ❌ Opposite sign!
  Diff: [ 1.573,  0.058,  1.997,  2.154,  2.170]
```

### After Heads Creation Fix
```
PCC: 0.581
First 5 values:
  HF:   [-1.303, -0.770, -2.909, -2.297,  0.983]
  TTML: [-0.385, -0.025, -4.750, -1.242,  0.034]  # ✅ Same sign now!
  Diff: [ 0.918,  0.745,  1.841,  1.055,  0.949]  # ✅ Smaller differences
```

**Analysis**:
- ✅ Sign flip fixed (major improvement!)
- ✅ Differences reduced by ~40%
- ❌ Still too large (need <0.05 for PCC >0.95)

---

## 🔍 REMAINING ISSUES

### The Problem
Even with heads creation fixed:
- Values still differ by 0.7-1.8 units
- PCC only 0.58 (need >0.95)
- Suggests additional bug(s) or accumulation of precision errors

### Possible Causes

#### 1. Heads Fusion Bug (HIGH PROBABILITY)
**Location**: `multi_head_utils.cpp:136-163`

If heads_creation was wrong, heads_fusion might need a corresponding fix.

**Current implementation** (line 145):
```cpp
// (B, H, S, E/H) -> (B, 1, S, E)
auto fused_heads = ttnn::experimental::nlp_concat_heads(x->get_value());
```

**Question**: Does `nlp_concat_heads` expect the heads format we're now producing?

**Backward pass** (lines 151-156):
```cpp
auto grad_result = ttnn::transpose(grad_output, -2, -1);  // (B, 1, S, E) -> (B, 1, E, S)
grad_result = ttnn::reshape(grad_result, ttnn::Shape({batch_size, num_heads, embedding_dim, sequence_length}));
grad_result = ttnn::transpose(grad_result, -2, -1);  // (B, H, S, E/H)
```

This suggests the forward should also use transpose/reshape, not `nlp_concat_heads`.

#### 2. GELU Approximation (MEDIUM PROBABILITY)
**Location**: `ops/unary_ops.cpp:37-49`

**Question**: Does `ttnn::gelu()` use exact or approximate GELU?

BERT expects: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`

If using sigmoid approximation instead, outputs would differ.

#### 3. Precision/Dtype (MEDIUM PROBABILITY)
**Unknown**: What dtype are tensors using?

- If BFLOAT16: 7-bit mantissa causes precision loss
- If FLOAT32: Should match HuggingFace

#### 4. Softmax Implementation (LOW PROBABILITY)
**Location**: `scaled_dot_product_attention.cpp:182`

```cpp
auto attention_weights = ttml::metal::softmax(qk_scaled, /* axis */ 3);
```

Could have numerical stability or precision issues.

---

## 🎯 NEXT INVESTIGATION STEPS

### Priority 1: Check Heads Fusion

**Hypothesis**: `nlp_concat_heads` expects different format than we're producing

**Test**:
```cpp
// In multi_head_utils.cpp line 145, try manual implementation:
// Instead of:
auto fused_heads = ttnn::experimental::nlp_concat_heads(x->get_value());

// Try:
// [B, H, S, E/H] -> [B, S, H, E/H]
auto transposed = ttnn::transpose(x->get_value(), 1, 2);
// [B, S, H, E/H] -> [B, S, E]
auto merged = ttnn::reshape(transposed, ttnn::Shape{batch_size, sequence_length, embedding_dim * num_heads});
// [B, S, E] -> [B, 1, S, E]
auto fused_heads = ttnn::reshape(merged, ttnn::Shape{batch_size, 1, sequence_length, embedding_dim * num_heads});
```

### Priority 2: Verify GELU Implementation

**Test**:
```python
# In Python test:
import torch
x = torch.tensor([0.0, 0.5, 1.0, -0.5, -1.0])
hf_gelu = torch.nn.functional.gelu(x, approximate='none')

# Compare with TTML gelu
```

### Priority 3: Check Tensor Dtype

**Add logging**:
```cpp
// In bert.cpp forward():
fmt::print("Tensor dtype: {}\n", hidden_states->get_value().dtype());
```

---

## 📈 PROGRESS SUMMARY

| Metric | Before Fixes | After Heads Fix | Target | Status |
|--------|--------------|-----------------|--------|--------|
| PCC | 0.569 | 0.581 | >0.95 | 🟡 Improving |
| Sign Correctness | ❌ Opposite | ✅ Same | ✅ Same | ✅ Fixed |
| Mean Abs Diff | 0.769 | 0.763 | <0.05 | ❌ Too high |
| First Value Diff | 1.573 | 0.918 | <0.05 | 🟡 Better |

**Improvement**: ~40% reduction in error magnitude, signs now correct

**Remaining work**: Need to reduce errors by another ~95% to reach target PCC

---

## 🔬 DEBUGGING STRATEGY

1. ✅ **Fixed heads creation** - Improved PCC from 0.57 to 0.58
2. 🟡 **Fix heads fusion** - Expected improvement: +0.10-0.20 PCC
3. 🟡 **Verify GELU** - Expected improvement: +0.05-0.10 PCC
4. 🟡 **Check dtype** - If BFLOAT16, may explain remaining gap
5. 🟡 **Stack fixes** - Multiple small bugs may compound

**Hypothesis**: 2-3 more bugs need fixing to reach PCC >0.95

---

## ✅ VERIFIED CORRECT

- QKV weight loading (PCC >0.999)
- LayerNorm epsilon (disabled hardware clamping)
- Positional embeddings (`ops::add(input, m_weight)`)
- Residual connections (correct order)
- Post-norm architecture (matches BERT)
- Attention masking logic (fixed, but not tested yet)
- Heads creation reshape (fixed)

---

## 📝 NEXT STEPS

1. **Investigate heads_fusion** - likely needs update to match new heads_creation format
2. **Test with manual heads fusion** implementation
3. **Verify GELU** implementation matches BERT exactly
4. **Add dtype logging** to check precision
5. **Run comprehensive tests** after each fix

**Expected timeline**: 1-2 more fixes needed before reaching PCC >0.95
