# Operator Testing Limitations - Bottom-Up Approach

**Date**: 2025-10-25
**Status**: 🔴 **BLOCKED** - Most operations not exposed to Python

---

## Goal

Test each BERT operation in isolation with controlled random data to identify which specific operation is broken.

## Approach

1. Generate identical random inputs for PyTorch and TTML
2. Run operation through both implementations
3. Compare outputs with strict PCC threshold (>0.999)
4. Isolate broken operations

## Problem Discovered

**Most BERT internal operations are NOT exposed to Python**. They are only accessible through the full C++ BERT model.

### Operations NOT Accessible from Python:

❌ `heads_creation` - C++ only, returns tuple (Python binding issue)
❌ `heads_fusion` - C++ only
❌ `scaled_dot_product_attention` - C++ only (has Python binding but tuple return fails)
❌ LayerNorm - Module exists but named differently
❌ Direct access to MHA internals

### What IS Accessible:

✅ `ttml.models.bert.create()` - Full BERT model
✅ `ttml.modules.LinearLayer` - Linear layers
✅ `ttml.ops.unary.gelu` - GELU activation
✅ Model parameters (can set weights)

---

## Current Test Files Created

### 1. `test_bert_operator_validation.py`
- **Status**: ❌ Incomplete
- **Problem**: Uses APIs that don't exist or aren't exposed
- **Needs**: Rewrite to use only exposed APIs or add Python bindings

### 2. `test_simple_operator_validation.py`
- **Status**: ❌ Fails on first test
- **Problem**: `heads_creation` tuple return not convertible to Python
- **Error**: `TypeError: Unable to convert function return value to a Python type`

### 3. `test_mha_isolated.py`
- **Status**: ⚠️  Limited
- **Problem**: Can load weights but can't isolate MHA from full BERT block
- **Issue**: MHA wrapped in LayerNorms, residuals, etc.

---

## Options to Proceed

### Option 1: Add Python Bindings (RECOMMENDED)
**Expose internal operations to Python for testing**

```cpp
// In Python bindings file
m.def("heads_creation", &ttml::ops::heads_creation);
m.def("heads_fusion", &ttml::ops::heads_fusion);
m.def("scaled_dot_product_attention", &ttml::ops::scaled_dot_product_attention);
```

**Pros**:
- Enables true bottom-up operator testing
- Can test each operation in perfect isolation
- Aligns with user's requested approach

**Cons**:
- Requires C++ changes and rebuild
- May need to fix tuple return value conversion

### Option 2: C++ Unit Tests (ALTERNATIVE)
**Write operator tests in C++ using GTest**

Example: `tests/model/bert_weight_loading_test.cpp` already exists

```cpp
TEST(OperatorTest, HeadsCreation) {
    // Create test data
    // Run heads_creation
    // Compare with expected output
}
```

**Pros**:
- Direct access to all C++ operations
- No Python binding issues
- Fast execution

**Cons**:
- User specifically requested Python approach
- Harder to iterate/debug than Python

### Option 3: End-to-End Layer Testing (CURRENT FALLBACK)
**Test at layer level instead of operator level**

Test progression:
1. ✅ Embeddings only
2. ✅ Embeddings + LayerNorm
3. ❌ Full attention block (current PCC 0.58)
4. MLP block
5. Full model

**Pros**:
- Can use existing Python APIs
- Narrows down problem area

**Cons**:
- Not true isolation
- Multiple operations tested together
- Harder to pinpoint exact broken operation

---

## Findings from Current PCC 0.58

### What We Know:
- Embeddings work (if we could test them in isolation)
- Weight loading appears correct (weights match after loading)
- Signs are now correct (after heads_creation fix)
- Error magnitude reduced 40% (0.77 → 0.76)
- Still need 93% more improvement (PCC 0.58 → >0.95)

### Most Likely Root Causes:
1. **Precision/Dtype** - Using BFLOAT16 instead of FLOAT32
2. **Operator Implementation** - One of: matmul, softmax, layer_norm using wrong precision
3. **Numerical Stability** - Accumulation of small errors

### Cannot Test Due to API Limitations:
- Cannot isolate `heads_creation` output
- Cannot isolate attention mechanism
- Cannot test softmax in isolation
- Cannot verify matmul precision

---

## Recommendation

**Add minimal Python bindings for debugging:**

```cpp
// sources/ttml/python_bindings.cpp (or wherever bindings are)

// Expose for testing
m.def("test_heads_creation", [](TensorPtr qkv, uint32_t num_heads) {
    auto [q, k, v] = ops::heads_creation(qkv, num_heads);
    return py::make_tuple(q, k, v);  // Explicitly make Python tuple
});

m.def("test_attention", &ops::scaled_dot_product_attention);
m.def("test_softmax", [](TensorPtr x, int axis) {
    return create_tensor(metal::softmax(x->get_value(), axis));
});
```

This would enable:
✅ True operator-by-operator testing
✅ Isolation of broken operations
✅ Fast iteration on fixes
✅ User's requested bottom-up approach

---

## Alternative: Instrument Existing Code

Add logging to C++ to print intermediate values:

```cpp
// In multi_head_attention.cpp
auto qkv = (*m_qkv_linear)(x);
fmt::print("[DEBUG] QKV output: shape={}, mean={}, std={}\\n",
           qkv->get_value().logical_shape(),
           compute_mean(qkv),
           compute_std(qkv));
```

Then compare with PyTorch intermediate values.

**Pros**: No API changes needed
**Cons**: Manual, tedious, hard to automate

---

## Status

🔴 **BLOCKED** on Python API limitations

**Next Step**: Decide on approach:
1. Add Python bindings for testing
2. Write C++ unit tests
3. Continue with layer-level testing

Current recommendation: **Option 1** (Add Python bindings) to enable user's requested operator-by-operator approach.
