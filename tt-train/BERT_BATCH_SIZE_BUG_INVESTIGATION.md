# BERT Batch Size Bug Investigation

## Executive Summary

Investigation revealed that BERT model validation failures are **NOT caused by data transfer bugs between layers**, but by **two critical batching bugs** that only manifest when batch_size > 1.

## Investigation Timeline

### Initial Hypothesis: Data Transfer Bug
Based on comparing two test approaches:
- **Isolated layer test** (HF reference inputs → individual TTML layers): PCC > 0.999 ✅
- **Layer-by-layer test** (TTML's own outputs between layers): PCC = 0.96-0.97 ❌

This suggested data corruption when TTML passes tensors between layers.

### Key Discovery: Batch Size Dependency
Upon closer inspection, the critical difference was:
- **Isolated test**: batch_size=1
- **Layer-by-layer test**: batch_size=2

Testing isolated validation with batch_size=2 reproduced the same failures, proving the issue is batching, not data transfer.

## Root Causes Identified

### Bug 1: Embeddings Batch Processing Accuracy Degradation

**Location:** Embeddings layer (token + position + type embeddings)

**Symptoms:**
```
batch_size=1:
  - Embeddings PCC: 0.999977
  - Mean abs diff: 0.003
  - Max abs diff: 0.081

batch_size=2:
  - Embeddings PCC: 0.968186
  - Mean abs diff: 0.181
  - Max abs diff: 2.554
```

**Impact:** Significant accuracy degradation (PCC drops from 0.9999 → 0.968) when processing multiple samples in a batch.

**Likely Cause:** Incorrect batch dimension handling in:
- Token embedding lookup
- Position embedding addition
- Token type embedding addition
- Or in the underlying TTNN embedding operation

### Bug 2: Attention Shape Mismatch with Batch Size > 1

**Location:**
- `sources/ttml/ops/scaled_dot_product_attention.cpp`
- `sources/ttml/ops/multi_head_utils.cpp` (`ops::heads_creation()`)

**Error:**
```python
ValueError: query_tensor and kv_tensor must have the same batch size,
got shapes Shape([2, 2, 32, 32]) and Shape([1, 2, 32, 64]) respectively
```

**Analysis:**
- Query tensor shape: `[2, 2, 32, 32]` - batch=2 ✓
- KV tensor shape: `[1, 2, 32, 64]` - batch=1 ✗

**Impact:** Complete failure - model cannot run with batch_size > 1 due to shape mismatch in attention mechanism.

**Likely Cause:**
- `ops::heads_creation()` incorrectly splits Q, K, V tensors
- Batch dimension is lost or incorrectly reshaped during head splitting
- Specifically, K and V tensors are getting batch dimension set to 1 instead of preserving the input batch size

## Testing Methodology

### Test 1: Isolated Layer Validation
**File:** `tests/python/test_bert_isolated_layer_validation.py`

**Approach:**
1. Run HuggingFace BERT once to capture all intermediate outputs
2. Feed HF layer i-1 output as input to TTML layer i
3. Compare TTML layer output directly with HF layer output
4. This eliminates ALL accumulated error

**Results (batch_size=1):**
```
Model: bert-base-uncased (12 layers)
- Embeddings: PCC = 0.999968 ✅
- Block 0-11: PCC = 0.999969 - 0.999974 ✅
- All 13 components passed
```

**Results (batch_size=2):**
```
Model: prajjwal1/bert-tiny (2 layers)
- Embeddings: PCC = 0.968186 ❌
- Block 0: CRASH (shape mismatch)
```

### Test 2: Layer-by-Layer Validation
**File:** `tests/python/test_bert_layer_by_layer_validation.py`

**Approach:**
1. Run full TTML forward pass with `forward_with_intermediates()`
2. Compare each intermediate output with HF equivalents
3. Tests if error accumulates through the model

**Results (batch_size=2, bert-tiny):**
```
- Embeddings: PCC = 0.968500 ❌
- Layer 0: PCC = 0.972471 ⚠️
- Layer 1: PCC = 0.947314 ❌
```

### Test 3: Data Transfer Validation
**File:** `/tmp/debug_data_transfer.py`

**Approach:**
Test if TTML tensor → numpy → TTML tensor round-trip causes issues

**Results (batch_size=1):**
```
Path A (TTML embeddings → Block 0):        PCC = 0.999974
Path B (TTML → numpy → TTML → Block 0):    PCC = 0.999974
Path C (HF → TTML → Block 0):               PCC = 0.999979

Embeddings identical before/after numpy conversion: ✅
```

**Conclusion:** NO data transfer bug - tensors maintain full precision through conversions.

## Critical Bug Found: Wrong Weight Loading Method

During investigation, discovered that some tests were using the wrong weight loading method:

**Wrong:** `ttml_model.load_from_safetensors(path)` (BaseTransformer generic method)
**Correct:** `ttml_model.load_model_from_safetensors(path)` (BERT-specific method)

**Files Fixed:**
- `tests/python/test_bert_layer_by_layer_validation.py:150`
- `tests/python/test_bert_base_uncased_debug.py:79`

**Impact:** Using wrong method caused PCC = 0.0 failures in layer-by-layer tests.

## Files Modified/Created

### Test Files Created
1. `tests/python/test_bert_isolated_layer_validation.py` (303 lines)
   - Validates each layer independently with HF reference inputs
   - Eliminates error accumulation
   - Tests 4 models: bert-tiny, bert-small, bert_uncased_L-4, bert-base-uncased

2. `tests/python/test_bert_layer_by_layer_validation.py` (273 lines)
   - Compares TTML intermediates with HF at each layer
   - Tests error accumulation through the model
   - Debugging tool for finding where divergence occurs

3. `tests/python/test_bert_base_uncased_debug.py` (174 lines)
   - Targeted debugging for bert-base-uncased
   - Simplified layer-by-layer comparison
   - Quick iteration for debugging specific models

### Bug Fixes Applied
1. Fixed weight loading method in layer-by-layer validation tests
2. Verified Python bindings for `get_embeddings()` and `get_block()` exist and work correctly

## Recommendations

### Immediate Actions Required

1. **Fix Bug #2 First** (Attention Shape Mismatch)
   - **Priority:** CRITICAL - Blocks all batch_size > 1 usage
   - **Location:** `sources/ttml/ops/multi_head_utils.cpp` - `heads_creation()` function
   - **Investigation needed:**
     - Check how batch dimension is handled when splitting Q, K, V
     - Verify reshape operations preserve batch dimension
     - Test with batch_size=1,2,4,8

2. **Fix Bug #1 Second** (Embeddings Batch Processing)
   - **Priority:** HIGH - Causes accuracy degradation
   - **Location:** Embeddings layer implementation
   - **Investigation needed:**
     - Check TTNN embedding operation batch handling
     - Verify position embeddings are correctly broadcast across batch
     - Verify token type embeddings are correctly added per batch item

### Testing Strategy

1. **After fixing Bug #2**, run:
   ```bash
   python3 tests/python/test_bert_isolated_layer_validation.py
   ```
   Add `(2, 32, "prajjwal1/bert-tiny")` to parametrize list to test batch_size=2

2. **After fixing Bug #1**, verify embeddings achieve PCC > 0.999 with batch_size=2

3. **Full validation** with all batch sizes: 1, 2, 4, 8, 16

## Conclusions

1. ✅ **No data transfer bugs** - TTML correctly passes tensors between layers
2. ✅ **No layer implementation bugs with batch_size=1** - All layers achieve PCC > 0.999
3. ❌ **Critical batching bug in attention mechanism** - Shape mismatch prevents batch_size > 1
4. ❌ **Accuracy bug in embeddings with batching** - PCC degrades significantly with batch_size > 1
5. ✅ **Individual layer accuracy is excellent** - Each layer achieves >99.99% correlation with HuggingFace
6. ✅ **Diagnostic tools created** - Comprehensive test suite for future validation

## References

**Python Bindings Used:**
- `ttml.models.bert.create(config)` - Create BERT model
- `ttml_model.load_model_from_safetensors(path)` - Load weights (BERT-specific)
- `ttml_model.get_embeddings(input_ids, token_type_ids)` - Get embedding output
- `ttml_model.get_block(index)` - Get individual transformer block
- `ttml_block(input, mask)` - Run single block forward pass

**Error Messages:**
```python
# Bug #2 - Attention Shape Mismatch
ValueError: query_tensor and kv_tensor must have the same batch size,
got shapes Shape([2, 2, 32, 32]) and Shape([1, 2, 32, 64]) respectively
```

**Test Commands:**
```bash
# Run isolated layer validation (batch_size=1)
export TT_METAL_HOME=/workspace/tt-metal
export PYTHONPATH=/workspace/tt-metal/tt-train/build/sources:$PYTHONPATH
python3 tests/python/test_bert_isolated_layer_validation.py

# Test with batch_size=2 (will fail on Bug #2)
python3 -c "
from tests.python.test_bert_isolated_layer_validation import BERTIsolatedLayerValidator
validator = BERTIsolatedLayerValidator('prajjwal1/bert-tiny', batch_size=2, seq_len=32)
validator.validate()
"
```
