# BERT Forward Pass Bug - Root Cause Identified

**Date**: 2025-10-25
**Status**: 🔴 **ROOT CAUSE IDENTIFIED** with **95% confidence**

---

## 🎯 ROOT CAUSE: Heads Creation Reshape Bug

**File**: `sources/ttml/ops/multi_head_utils.cpp`
**Lines**: 89-98
**Severity**: **CRITICAL**

### The Bug

The manual head splitting reshape **scrambles token and head assignments**:

```cpp
// Current WRONG implementation:
// Reshape: (B, 1, S, E) -> (B, 1, S*H, E/H) -> (B, H, S, E/H)
auto q_reshaped = ttnn::reshape(q_flat, ttnn::Shape{batch_size, 1, seq_len * num_heads, head_dim});
auto q = ttnn::reshape(q_reshaped, ttnn::Shape{batch_size, num_heads, seq_len, head_dim});
```

### Why This Is Wrong

For `batch=1, seq_len=32, embedding_dim=128, num_heads=2, head_dim=64`:

**Input**: `[1, 1, 32, 128]`
- Memory layout: `tok0[d0...d127], tok1[d0...d127], ..., tok31[d0...d127]`

**After first reshape to** `[1, 1, 64, 64]`:
- Reinterprets 32×128=4096 elements as 64×64
- Row 0: tok0[d0...d63]
- Row 1: tok0[d64...d127]  ← **Second half of tok0 treated as separate "token"!**
- Row 2: tok1[d0...d63]
- Row 3: tok1[d64...d127]  ← **Second half of tok1 treated as separate "token"!**
- ...

**After second reshape to** `[1, 2, 32, 64]`:
- Head 0 gets rows 0-31:
  - "Token" 0: tok0[d0...d63] ✅ Correct
  - "Token" 1: tok0[d64...d127] ❌ **WRONG! This is not a token, it's second half of tok0!**
  - "Token" 2: tok1[d0...d63] ✅ Correct dims, but labeled as token 2!
  - "Token" 3: tok1[d64...d127] ❌ **WRONG! Second half of tok1!**
  - ...

- Head 1 gets rows 32-63:
  - "Token" 0: tok16[d0...d63] ❌ **COMPLETELY WRONG! Should be tok0[d64...d127]!**
  - ...

### The Correct Logic Should Be

For proper head splitting:
```
Input:  [B, 1, S, E] where E = H × (E/H)
Output: [B, H, S, E/H]

Correct mapping:
  Token i, Head h → input[b, 1, i, h*(E/H) : (h+1)*(E/H)]

Example for tok0, head0: dims 0-63
Example for tok0, head1: dims 64-127
Example for tok1, head0: dims 0-63
Example for tok1, head1: dims 64-127
```

### What the Current Code Does Instead

```
Current mapping (WRONG):
  Head 0, "Token" 0 → tok0[d0...d63]     ✅
  Head 0, "Token" 1 → tok0[d64...d127]   ❌ (second half of tok0 as token 1!)
  Head 0, "Token" 2 → tok1[d0...d63]     ✅ (but mislabeled as token 2)
  Head 0, "Token" 3 → tok1[d64...d127]   ❌ (second half of tok1 as token 3!)
  ...
  Head 1, "Token" 0 → tok16[d0...d63]    ❌ (completely wrong token!)
```

This means:
1. Each head sees **interleaved halves** of tokens instead of proper tokens
2. **Head 1 sees completely different tokens** than Head 0 (offset by 16 positions!)
3. The self-attention mechanism computes attention between **scrambled data**

### Impact on Forward Pass

This bug explains why:
- ✅ **Weight loading works** (PCC >0.999) - weights are stored correctly
- ❌ **Forward pass fails** (PCC 0.17-0.74) - QKV heads are scrambled during computation
- ❓ **Output has wrong signs and magnitudes** - attention computed on wrong token assignments

---

## 🔧 THE FIX

### Option 1: Correct Reshape Sequence

Instead of changing sequence length, we need to split the embedding dimension:

```cpp
// PROPOSED FIX:
// We need: [B, 1, S, E] → [B, H, S, E/H]
// But we can't directly reshape E into H×(E/H) because that would require 5D: [B, 1, S, H, E/H]

// Workaround using transpose:
// Step 1: Reshape to [B, S, E]  (remove channel dim)
auto q_no_channel = ttnn::reshape(q_flat, ttnn::Shape{batch_size, seq_len, embedding_dim});

// Step 2: Reshape to [B, S, H, E/H]  (split embedding)
auto q_with_heads = ttnn::reshape(q_no_channel, ttnn::Shape{batch_size, seq_len, num_heads, head_dim});

// Step 3: Transpose to [B, H, S, E/H]
auto q = ttnn::transpose(q_with_heads, 1, 2);  // Swap dims 1 and 2
```

### Option 2: Use Slicing (More Explicit)

```cpp
// Alternative: Slice each head explicitly
std::vector<ttnn::Tensor> heads;
for (uint32_t h = 0; h < num_heads; h++) {
    // Slice dims [h * head_dim : (h+1) * head_dim] from last dimension
    auto head = ttnn::slice(
        q_flat,
        ttnn::SmallVector<uint32_t>{0, 0, 0, h * head_dim},
        ttnn::SmallVector<uint32_t>{batch_size, 1, seq_len, (h + 1) * head_dim},
        ttnn::SmallVector<uint32_t>{1, 1, 1, 1});
    heads.push_back(head);
}
// Stack along head dimension
auto q = ttnn::concat(heads, /* dim */ 1);  // Concat to create head dimension
```

---

## 📊 EVIDENCE SUPPORTING THIS DIAGNOSIS

### 1. Code Analysis ✅
- Reshape changes sequence length (`S*H`) instead of embedding dimension
- This is mathematically incorrect for head splitting

### 2. Weight Loading vs Forward Pass ✅
- Weights load correctly (PCC >0.999) → weight storage/loading is fine
- Forward fails (PCC 0.17-0.74) → computation is broken

### 3. Magnitude of Error ✅
- Outputs show **opposite signs** and **large differences**
- This matches scrambled attention: model attends to wrong tokens

### 4. Masking Fix Didn't Help ✅
- Attention masking bug fix changed nothing (PCC still 0.57)
- Because test doesn't use masks → bug must be in unmasked path
- Heads creation is in unmasked path

### 5. Pattern Consistency ✅
- ALL BERT variants fail (tiny, small, base)
- All have same heads creation code
- Failure magnitude increases with model size (more accumulated errors)

---

## 🎲 CONFIDENCE LEVEL

**95% confidence** this is the primary bug because:
1. ✅ Mathematically proven to be wrong
2. ✅ Matches observed symptoms (wrong outputs, opposite signs)
3. ✅ Explains why weight loading works but forward fails
4. ✅ Consistent across all BERT variants
5. ✅ No other bugs found that could cause this magnitude of error

**5% uncertainty** from:
- Cannot directly test heads_creation from Python
- Possible additional bugs could compound the error

---

## 🚀 NEXT STEPS

### Priority 1: Fix Heads Creation Reshape

**Recommended approach**: Option 1 (transpose-based)

**Implementation**:
1. Modify `sources/ttml/ops/multi_head_utils.cpp` lines 89-98
2. Replace current reshape with correct transpose sequence
3. Update backward pass (lines 111-118) accordingly

**Expected result**: PCC should jump from 0.57 to >0.95

### Priority 2: Verify Fix

Run tests:
```bash
python3 tests/python/debug_bert_forward_pass.py --model prajjwal1/bert-tiny
python3 -m pytest tests/python/test_bert_comprehensive_validation.py
python3 -m pytest tests/python/test_bert_inference_showcase.py
```

### Priority 3: Attention Masking (Secondary Bug)

After fixing heads creation, also fix attention masking:
- File: `sources/ttml/ops/scaled_dot_product_attention.cpp:166`
- Change: `1e9F` → `-1e9F` (already applied)
- This will fix masked attention when masks are used

---

## 📝 ADDITIONAL BUGS FOUND

### Secondary Bug: Attention Masking (Already Fixed)
- **Status**: ✅ Fixed but not tested (tests don't use masks)
- **File**: `scaled_dot_product_attention.cpp:166`
- **Change**: `1e9F` → `-1e9F`
- **Impact**: Will matter when attention masks are used

### Verified Correct:
- ✅ QKV weight loading (PCC >0.999)
- ✅ LayerNorm epsilon (fixed - disabled hardware clamping)
- ✅ Positional embeddings addition (line 103: `ops::add`)
- ✅ Residual connections (correct order)
- ✅ Post-norm architecture (matches BERT)

---

## 📖 SUMMARY

**Root Cause**: The heads creation reshape in `multi_head_utils.cpp` lines 89-98 scrambles tokens and heads by incorrectly reshaping the sequence dimension instead of the embedding dimension.

**Fix**: Replace the reshape sequence with a transpose-based approach that properly splits embedding dimensions into heads.

**Expected Outcome**: Forward pass PCC should improve from 0.17-0.74 to >0.95, matching HuggingFace reference implementation.

**Confidence**: 95%
