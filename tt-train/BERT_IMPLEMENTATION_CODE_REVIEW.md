# BERT Implementation Code Review

**Review Date**: 2025-10-25
**Reviewed Commits**: `17420e3d11^..HEAD` (BERT implementation series)
**Status**: ❌ **CRITICAL BUGS FOUND** - Forward pass broken (PCC 0.17-0.74)

---

## 🔴 CRITICAL BUGS FOUND

### **BUG #1: ATTENTION MASK PROCESSING ERROR** (HIGHEST PRIORITY)

**File**: `sources/ttml/models/bert.cpp`
**Lines**: 180-204
**Severity**: **CRITICAL** - This is likely THE primary cause of forward pass failure

```cpp
autograd::TensorPtr Bert::process_attention_mask(const autograd::TensorPtr& attention_mask) const {
    if (!attention_mask) {
        return nullptr;
    }

    // Convert attention mask from (1, 0) to (0, -10000) for additive attention
    // mask = (1 - attention_mask) * -10000
    // This makes padding tokens have very negative scores before softmax
    auto inverted_mask = ops::sub(
        autograd::create_tensor(core::ones(attention_mask->get_shape(), &autograd::ctx().get_device())),
        attention_mask);
    auto processed_mask = ops::mul(inverted_mask, -10000.0F);  // ❌ WRONG VALUE!

    // Reshape mask for broadcasting with attention scores
    // From [batch, 1, 1, seq_len] to [batch, 1, seq_len, seq_len] for self-attention
    auto mask_shape = attention_mask->get_shape();
    auto batch_size = mask_shape[0];
    auto seq_len = mask_shape[3];

    // Expand mask to [batch, 1, seq_len, seq_len] for proper broadcasting
    auto expanded_mask = ttnn::reshape(processed_mask->get_value(), ttnn::Shape{batch_size, 1, 1, seq_len});
    expanded_mask = ttnn::repeat(expanded_mask, ttnn::Shape{1, 1, seq_len, 1});

    return autograd::create_tensor(expanded_mask);
}
```

#### Issues:

1. **❌ WRONG MASK VALUE**: Uses `-10000.0` but `scaled_dot_product_attention.cpp:165` uses `1e9` (positive!)
   ```cpp
   // In scaled_dot_product_attention.cpp line 161-179:
   if (mask) {
       auto mask_tensor = mask->get_value();
       // ttnn::where when mask is not of the same shape as qk_scaled
       qk_scaled = ttnn::add(
           ttnn::multiply(mask_tensor, qk_scaled, ...),  // ← mask=1: keep score
           ttnn::multiply(
               ttnn::subtract(mask_tensor, 1.F, ...),    // ← mask=0: subtract from 1
               1e9F,  // ❌ ADDS +1e9 to masked positions!
               ...),
           ...);
   }
   ```

2. **LOGIC ERROR**:
   - HuggingFace: mask=1 means "attend", mask=0 means "ignore"
   - BERT processes mask: 1 → 0 (good), 0 → -10000 (bad but intended)
   - SDPA expects: mask=1 means "keep", mask=0 means "mask out"
   - SDPA implementation: `mask=0` → `(1-0) * 1e9 = +1e9` (ADDS huge positive number!)

3. **RESULT**: Masked (padding) positions get BOOSTED instead of suppressed!

#### Correct Logic Should Be:
```cpp
// HuggingFace BERT mask convention: 1 = attend, 0 = ignore
// SDPA expects: 1 = keep score, 0 = add -inf
// So we should pass mask AS-IS (no inversion) or fix SDPA logic
```

---

### **BUG #2: HEADS CREATION RESHAPE BUG** (HIGH PRIORITY)

**File**: `sources/ttml/ops/multi_head_utils.cpp`
**Lines**: 89-98
**Severity**: **HIGH** - Manual reshape may create wrong head layout

```cpp
// Reshape: (B, 1, S, E) -> (B, 1, S*H, E/H) -> (B, H, S, E/H)
// This works around the ttnn bug by doing reshape instead of nlp_create_qkv_heads
auto q_reshaped = ttnn::reshape(q_flat, ttnn::Shape{batch_size, 1, seq_len * num_heads, head_dim});
auto q = ttnn::reshape(q_reshaped, ttnn::Shape{batch_size, num_heads, seq_len, head_dim});
```

#### Issues:

1. **RESHAPE ORDERING**: The intermediate reshape `(B, 1, S, E) → (B, 1, S*H, E/H)` changes the memory layout
   - Original: `[tok0_dim0, tok0_dim1, ..., tok0_dimE, tok1_dim0, tok1_dim1, ...]`
   - After reshape: `[head0_tok0_dim0, ..., head0_tok0_dimH, head1_tok0_dim0, ...]`
   - This is **head-first** ordering

2. **EXPECTED LAYOUT**: BERT/HuggingFace uses **token-first** ordering:
   - `[tok0_head0, tok0_head1, ..., tok0_headN, tok1_head0, ...]`

3. **CONSEQUENCE**: Q, K, V heads may be scrambled, causing attention to compute with wrong head assignments

#### Verification Needed:
```python
# Check if heads are correctly split
# For hidden_dim=128, num_heads=2, head_dim=64:
# Token 0, dims 0-63 should map to head 0
# Token 0, dims 64-127 should map to head 1
# Current reshape might give:
# Token 0, dims 0, 2, 4, ... (even) → head 0
# Token 0, dims 1, 3, 5, ... (odd) → head 1
```

---

### **BUG #3: POSITION EMBEDDINGS ADDITION ORDER** (MEDIUM PRIORITY)

**File**: `sources/ttml/models/bert.cpp`
**Lines**: 152-156
**Severity**: MEDIUM - May cause wrong position assignment

```cpp
// Token embeddings
auto embeddings = (*m_token_embeddings)(input_ids);

// Add positional embeddings using the operator() which adds positions and applies dropout (no-op since
// dropout_prob=0)
embeddings = (*m_position_embeddings)(embeddings);
```

**File**: `sources/ttml/modules/positional_embeddings.cpp`
**Lines**: 86-100

```cpp
autograd::TensorPtr TrainablePositionalEmbedding::operator()(const autograd::TensorPtr& input) {
    auto input_tensor = input->get_value();
    auto input_shape = input_tensor.logical_shape();
    // ... validation ...

    // ❓ How does this add positions? Missing the actual addition code!
    // Expected: x = input + positional_embedding
    // But this file was truncated at line 100
```

#### Issues:

1. **CODE MISSING**: The actual position embedding addition is not visible in the truncated read
2. **ASSUMPTION**: Assuming it does `input + m_weight` element-wise
3. **POTENTIAL ISSUE**: If position indices aren't extracted from input_ids properly, wrong positions may be added

---

### **BUG #4: SCALED DOT-PRODUCT ATTENTION MASKING** (CRITICAL)

**File**: `sources/ttml/ops/scaled_dot_product_attention.cpp`
**Lines**: 158-180
**Severity**: **CRITICAL** - Masking logic is inverted

```cpp
if (mask) {
    auto mask_tensor = mask->get_value();
    // ttnn::where when mask is not of the same shape as qk_scaled
    qk_scaled = ttnn::add(
        ttnn::multiply(mask_tensor, qk_scaled, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
        ttnn::multiply(
            ttnn::subtract(mask_tensor, 1.F, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            1e9F,  // ❌ ADDS +1e9 instead of -inf!
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        none,
        none,
        none,
        false);
}
```

#### Issues:

1. **WRONG SIGN**: Should be `-1e9` or `-inf`, not `+1e9`
2. **LOGIC**: `mask=0` positions get `(1-0) * 1e9 = +1e9` added to attention scores
3. **EFFECT**: Padding tokens get MAXIMUM attention instead of being ignored!
4. **FIX**: Change `1e9F` to `-1e9F` on line 165

---

### **BUG #5: QKV WEIGHT CONCATENATION DIMENSION** (LOW - Already Verified Correct)

**File**: `sources/ttml/models/bert.cpp`
**Lines**: 549-561
**Status**: ✅ **APPEARS CORRECT** (PCC >0.999 on weight loading)

```cpp
// Convert vectors to xtensor arrays: [hidden_size, hidden_size]
// HuggingFace format: [out_features, in_features]
const auto& Q_vec = *cache.query_weight;
const auto& K_vec = *cache.key_weight;
const auto& V_vec = *cache.value_weight;

auto Q = xt::adapt(Q_vec, std::vector<size_t>{hidden_size, hidden_size});
auto K = xt::adapt(K_vec, std::vector<size_t>{hidden_size, hidden_size});
auto V = xt::adapt(V_vec, std::vector<size_t>{hidden_size, hidden_size});

// Concatenate along dim=0: cat(Q, K, V, dim=0) -> [3*hidden_size, hidden_size]
// This stacks the output features: [Q_out, K_out, V_out]
auto qkv_combined = core::concat(std::vector<xt::xarray<float>>{Q, K, V}, 0);
```

#### Notes:
- Weight loading tests show PCC >0.999, so this is correct
- Concatenates along output dimension (correct for row-major layout)
- Bias concatenation also correct (line 577-581)

---

## 🟡 POTENTIAL HOTSPOTS

### **HOTSPOT #1: LayerNorm Epsilon Clamping** (INVESTIGATED - FIXED)

**File**: `sources/ttml/modules/bert_block.cpp`, `sources/ttml/models/bert.cpp`
**Status**: ✅ **FIXED** (disabled hardware clamping)

Originally used `enable_hardware_clamp=true` which clamped epsilon from 1e-12 to 1e-4. Now fixed by passing `false`.

---

### **HOTSPOT #2: GELU Activation Implementation**

**File**: `sources/ttml/ops/unary_ops.cpp`
**Lines**: 37-49

```cpp
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::gelu(tensor->get_value()));  // ❓ Which GELU variant?
    autograd::GradFunction grad = [tensor, out]() {
        static const std::string approx_mode = "none";  // ✅ Exact GELU
        auto dL_dt = ttnn::experimental::gelu_bw(out->get_grad(), tensor->get_value(), approx_mode);
        tensor->add_grad(dL_dt);
    };
    // ...
}
```

#### Questions:
1. **Which GELU?** Does `ttnn::gelu()` use exact or approximate GELU?
2. **BERT expects**: Exact GELU: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
3. **Common variants**:
   - Exact GELU (BERT standard)
   - `tanh` approximation
   - `sigmoid` approximation

#### Verification needed:
Check if `ttnn::gelu` matches HuggingFace `torch.nn.functional.gelu(x, approximate='none')`

---

### **HOTSPOT #3: Positional Embedding Addition**

**File**: `sources/ttml/modules/positional_embeddings.cpp`
**Lines**: 86-100+ (incomplete read)

```cpp
autograd::TensorPtr TrainablePositionalEmbedding::operator()(const autograd::TensorPtr& input) {
    auto input_tensor = input->get_value();
    auto input_shape = input_tensor.logical_shape();
    // ... validation ...

    // ❓ MISSING: Actual addition code not visible
    // Expected: return ops::add(input, m_weight);
}
```

#### Need to verify:
1. Positions are added element-wise to ALL tokens
2. No position index extraction (BERT uses learned absolute positions, not relative)
3. Shape: `m_weight` should be `[1, 1, max_seq_len, embedding_dim]`
4. Broadcasting: `input [B, 1, S, E]` + `weight [1, 1, max_seq_len, E]` → `[B, 1, S, E]`

---

### **HOTSPOT #4: Tensor Dtype - BFLOAT16 vs FLOAT32**

**Files**: All forward pass operations
**Status**: ❓ **UNKNOWN** - Need to verify

#### Question:
What dtype are tensors using during forward pass?
- If BFLOAT16: Precision loss accumulates
- If FLOAT32: Should match HuggingFace

#### How to check:
```cpp
// Add logging in bert.cpp forward():
fmt::print("Tensor dtype: {}\n", hidden_states->get_value().dtype());
```

---

### **HOTSPOT #5: Residual Connection Order**

**File**: `sources/ttml/modules/bert_block.cpp`
**Lines**: 78-90

```cpp
autograd::TensorPtr BertBlock::operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& attention_mask) {
    // Self-attention with residual connection and layer norm
    // BERT uses post-norm: LayerNorm(x + Attention(x))
    auto attention_output = (*m_attention)(input, attention_mask);
    auto attention_residual = ops::add(attention_output, input);  // ✅ Correct order
    attention_residual = (*m_attention_norm)(attention_residual);

    // Feed-forward with residual connection and layer norm
    auto mlp_output = (*m_mlp)(attention_residual);
    auto mlp_residual = ops::add(mlp_output, attention_residual);  // ✅ Correct order
    mlp_residual = (*m_mlp_norm)(mlp_residual);

    return mlp_residual;
}
```

#### Analysis:
- ✅ Uses post-norm architecture (correct for BERT)
- ✅ Residual connections in correct order
- ✅ LayerNorm applied AFTER residual add (correct)

**No issues found here.**

---

### **HOTSPOT #6: Input Shape Expectations**

**File**: `sources/ttml/models/bert.cpp`
**Lines**: 146-178 (get_embeddings)

#### Current expectations:
- `input_ids`: `[batch, 1, 1, seq_len]` (4D with channel=1)
- `token_type_ids`: `[batch, 1, 1, seq_len]` (4D)

#### HuggingFace format:
- `input_ids`: `[batch, seq_len]` (2D)
- `token_type_ids`: `[batch, seq_len]` (2D)

#### Python bindings must handle:
```python
# User passes: [batch, seq_len]
# C++ expects: [batch, 1, 1, seq_len]
# Bindings must reshape
```

**Verify**: Python tests reshape correctly before calling C++ model

---

## 📋 SUMMARY OF FINDINGS

### Critical Bugs (Require Immediate Fix):

1. **❌ BUG #1**: Attention mask processing uses `-10000` but SDPA adds `+1e9` for masked positions
2. **❌ BUG #4**: SDPA masking logic adds `+1e9` instead of `-inf` to masked positions

### High Priority Issues:

3. **❓ BUG #2**: Heads creation reshape may scramble head layout (needs verification)

### Medium Priority Issues:

4. **❓ HOTSPOT #3**: Positional embedding addition (code not fully reviewed)
5. **❓ HOTSPOT #4**: Tensor dtype unknown (BFLOAT16 vs FLOAT32)

### Low Priority / Verified Correct:

6. ✅ QKV weight loading (PCC >0.999)
7. ✅ LayerNorm epsilon (fixed)
8. ✅ Residual connections (correct order)

---

## 🎯 RECOMMENDED FIX PRIORITY

### **Priority 1: Fix Attention Masking**

**Option A**: Fix SDPA implementation
```cpp
// In scaled_dot_product_attention.cpp line 165:
// Change:
1e9F,
// To:
-1e9F,
```

**Option B**: Fix BERT mask processing
```cpp
// In bert.cpp line 191:
// Change:
auto processed_mask = ops::mul(inverted_mask, -10000.0F);
// To:
auto processed_mask = ops::mul(inverted_mask, -1e9F);
// And remove mask inversion if SDPA expects 1=keep, 0=mask
```

**Recommendation**: Fix both for consistency

---

### **Priority 2: Verify Heads Creation**

Test if heads are correctly split:
```cpp
// Add logging in multi_head_utils.cpp:
fmt::print("Q before heads_creation: shape={}\n", qkv->get_value().logical_shape());
fmt::print("Q after heads_creation: shape={}\n", out_q->get_value().logical_shape());
// Verify first head contains dims 0-63, second head contains dims 64-127
```

---

### **Priority 3: Check Tensor Dtype**

Add dtype logging to identify if precision loss is contributing:
```cpp
// In bert.cpp forward():
fmt::print("Hidden states dtype: {}\n", hidden_states->get_value().dtype());
```

---

## 🔍 ROOT CAUSE HYPOTHESIS

Based on the code review, the forward pass failure (PCC 0.17-0.74) is most likely caused by:

**PRIMARY CAUSE (90% confidence)**:
**Attention masking bug** - Masked (padding) positions get boosted (+1e9) instead of suppressed (-1e9), causing attention to focus on padding tokens instead of real tokens.

**SECONDARY CAUSES (possible contributors)**:
- Heads creation reshape may scramble head assignments (30% confidence)
- Precision loss if using BFLOAT16 (20% confidence)
- GELU implementation mismatch (10% confidence)

---

## ✅ NEXT STEPS

1. **Fix attention masking bug** in `scaled_dot_product_attention.cpp:165`
2. **Rebuild and test** with debug script
3. **If still broken**: Verify heads creation reshape logic
4. **If still broken**: Check tensor dtypes and GELU implementation

**Expected outcome after fix #1**: PCC should improve significantly (from 0.57 to >0.90)
