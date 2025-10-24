# BERT Test Suite Inventory

This document catalogs all test files created during the BERT weight loading investigation. These files can serve as the foundation for a comprehensive BERT test suite.

---

## Diagnostic Scripts (Standalone)

### 1. `debug_weight_loading_pipeline.py`
**Purpose:** Traces the entire weight loading pipeline step-by-step

**What it tests:**
- Roundtrip test (store → retrieve)
- Token embedding loading before/after
- QKV weight loading before/after
- Compares with HuggingFace at each stage

**Usage:**
```bash
python3 tests/python/debug_weight_loading_pipeline.py
```

**Key features:**
- Detailed statistics at each stage (mean, std, min, max)
- Point-by-point value comparison
- Safetensors validation

---

### 2. `quick_check_weights.py`
**Purpose:** Fast validation that weights load correctly

**What it tests:**
- bert-base-uncased weight loading
- Token embeddings PCC
- QKV weights PCC

**Usage:**
```bash
python3 tests/python/quick_check_weights.py
```

**Key features:**
- Quick pass/fail indicator
- PCC validation (threshold 0.999)
- Minimal output for CI/CD

---

### 3. `test_parameters_update.py`
**Purpose:** Validates TensorPtr semantics and parameter updating

**What it tests:**
- TensorPtr object identity across parameters() calls
- Weight updates persist through shared_ptr
- Loading actually modifies parameters

**Usage:**
```bash
python3 tests/python/test_parameters_update.py
```

**Key features:**
- Tests C++ shared_ptr behavior from Python
- Verifies parameter map semantics
- Useful for debugging parameter-related issues

---

### 4. `test_set_value_basic.py`
**Purpose:** Basic test of set_value() functionality

**What it tests:**
- Tensor.set_value() updates values
- core.from_vector() works correctly

**Usage:**
```bash
python3 tests/python/test_set_value_basic.py
```

**Note:** Currently incomplete (core.from_vector not exposed to Python)

---

### 5. `compare_loaded_weights.py`
**Purpose:** Direct comparison of expected vs loaded weights

**What it tests:**
- Loads expected QKV from numpy
- Loads actual QKV from TTML model
- Point-by-point comparison
- Tests transpose scenarios

**Usage:**
```bash
python3 tests/python/compare_loaded_weights.py
```

**Key features:**
- Saves weights to disk for manual inspection
- Tests multiple concatenation patterns

---

### 6. `debug_qkv_loading.py`
**Purpose:** Focused QKV weight inspection

**What it tests:**
- QKV weight shapes
- Concatenation patterns
- Direct safetensors reading

**Usage:**
```bash
python3 tests/python/debug_qkv_loading.py
```

---

### 7. `inspect_safetensors.py`
**Purpose:** Validates safetensors file contents

**What it tests:**
- Safetensors file structure
- Tensor shapes and dtypes
- Direct tensor value inspection

**Usage:**
```bash
python3 tests/python/inspect_safetensors.py
```

---

## Pytest Test Suites

### 8. `test_bert_stepwise_validation_manual.py`
**Purpose:** Manual stepwise validation comparing HF and TTML BERT

**What it tests:**
- Weight loading validation (token embeddings, QKV)
- Forward pass comparison
- Full BERT model validation

**Usage:**
```bash
python3 -m pytest tests/python/test_bert_stepwise_validation_manual.py -v -s
```

**Test cases:**
- `test_bert_manual_stepwise_validation[1-32-prajjwal1/bert-tiny]`
- Can be extended with more model sizes

**Key features:**
- Uses existing TTML Python API (no C++ changes needed)
- Validates both weights and forward pass
- Comprehensive failure reporting

---

### 9. `test_bert_stepwise_validation.py`
**Purpose:** Framework for future stepwise validation with C++ hooks

**What it tests:**
- Designed to compare intermediate outputs if C++ exposes them
- Currently only tests final output

**Usage:**
```bash
python3 -m pytest tests/python/test_bert_stepwise_validation.py -v -s
```

**Status:** Framework complete, awaiting C++ API for intermediate outputs

---

### 10. `test_bert_base_uncased_diagnostic.py`
**Purpose:** Layer-by-layer diagnostic for bert-base-uncased

**What it tests:**
- Each transformer block individually
- QKV concatenation patterns
- Intermediate attention outputs

**Usage:**
```bash
python3 -m pytest tests/python/test_bert_base_uncased_diagnostic.py -v -s
```

**Test cases:**
- `test_bert_base_uncased_layer_by_layer`

**Key features:**
- Tests all 4 QKV concatenation patterns
- Layer-by-layer comparison
- Captures hooks from HF model

---

## Existing Tests (Modified)

### 11. `test_bert_golden_reference.py` (Modified)
**Purpose:** Golden reference test comparing TTML vs HuggingFace

**What was added:**
- bert-base-uncased to parametrize list

**Test cases:**
- `test_bert_qkv_loading_golden_reference[1-32-prajjwal1/bert-tiny]`
- `test_bert_qkv_loading_golden_reference[1-32-bert-base-uncased]`
- Can be extended with (2, 64) batch/seq_len variant

---

## Test Organization Recommendations

### Phase 1: Core Validation (CI/CD)
These should run on every commit:
- `quick_check_weights.py` - Fast weight loading check
- `test_bert_stepwise_validation_manual.py` (bert-tiny only) - Full validation

### Phase 2: Extended Validation (Nightly)
Run these daily:
- `test_bert_golden_reference.py` (all models) - Golden reference
- `test_bert_stepwise_validation_manual.py` (all models) - All sizes

### Phase 3: Diagnostic Tools (On Demand)
Use these for debugging:
- `debug_weight_loading_pipeline.py`
- `test_parameters_update.py`
- `inspect_safetensors.py`
- `compare_loaded_weights.py`

---

## Future Test Suite Enhancements

### Suggested Additions

1. **Quantitative metrics test**
   - Test PCC thresholds for different model sizes
   - Track PCC over time (regression detection)

2. **Multi-batch validation**
   - Test batch sizes: 1, 2, 4, 8
   - Test sequence lengths: 32, 64, 128, 256, 512

3. **Model variants**
   - bert-tiny
   - bert-base-uncased
   - bert-base-cased
   - bert-large-uncased

4. **Weight loading edge cases**
   - Missing weights
   - Shape mismatches
   - Dtype mismatches
   - Corrupted safetensors

5. **Performance benchmarks**
   - Weight loading time
   - Forward pass time
   - Memory usage

6. **Intermediate layer validation** (requires C++ changes)
   - Attention scores
   - FFN activations
   - Layer norm outputs

---

## Test Data Management

### Safetensors Files
Current location: `/tmp/*.safetensors`

Recommendation: Move to dedicated test data directory
```
tests/data/models/
├── bert-tiny.safetensors
├── bert-base-uncased.safetensors
└── README.md
```

### Expected Outputs
Consider saving golden outputs for regression testing:
```
tests/data/golden_outputs/
├── bert-tiny_batch1_seq32.npy
├── bert-base-uncased_batch1_seq32.npy
└── README.md
```

---

## CI/CD Integration

### Suggested pytest markers

```python
@pytest.mark.weight_loading
@pytest.mark.bert
@pytest.mark.golden_reference
@pytest.mark.diagnostic
@pytest.mark.slow
```

### Example CI workflow
```yaml
# Fast checks (every commit)
pytest -m "weight_loading and not slow"

# Full validation (nightly)
pytest -m "bert"

# Diagnostics (manual trigger)
pytest -m "diagnostic"
```

---

## Maintenance Notes

- Keep diagnostic scripts even if tests pass (useful for future debugging)
- Update thresholds based on hardware/precision changes
- Document any test failures with repro steps
- Archive old test data but keep recent versions

---

## Contact

For questions about these tests, refer to:
- `WEIGHT_LOADING_INVESTIGATION_RESULTS__INTERNAL.md` - Investigation findings
- `BERT_QKV_WEIGHT_LOADING_BUG_REPORT__INTERNAL.md` - Original bug report
- Git commit history on branch `ivoitovych/bert-model-for-ttml-qkv-weight-loading-3`
