# Comprehensive Testing Plan for nlp_create_qkv_heads Bug Fix

## Bug Summary
**Issue:** Integer division in `nlp_create_qkv_heads_program_factory.cpp` causes incorrect output shapes when `head_dim < 32`.

**Root Cause:** `q_num_tiles = num_heads * (head_dim / TILE_WIDTH)` returns 0 when head_dim < 32.

**Fix:** Use ceiling division: `q_num_tiles = (num_heads * head_dim + TILE_WIDTH - 1) / TILE_WIDTH`

---

## Test Coverage Added

### **New Python Test Files:**

1. **`test_nlp_create_qkv_heads_head_dim_bug.py`** (Bug reproduction)
   - 5 test configurations × 2 dtypes × 4 memory configs = **40 test cases**
   - Focus: head_dim = 8, 16, 32 (bug cases + boundary)

2. **`test_nlp_create_qkv_heads_regression.py`** (Regression protection) ⭐ NEW
   - **6 test categories:**
     - Transpose tests: 12 configs × 2 dtypes × 4 mem = **96 test cases**
     - Separate KV tests: 3 head_dims × 2 dtypes × 4 mem = **24 test cases**
     - GQA tests: 5 configs × 1 dtype × 2 mem = **10 test cases**
     - Feature combinations: 9 configs × 1 dtype = **9 test cases**
     - Boundary (head_dim=32): 4 configs × 2 dtypes × 4 mem = **32 test cases**
     - Large dimensions: 4 configs × 1 dtype = **4 test cases**
   - **Total: 175 regression test cases**

### **New C++ Test File:**

3. **`test_nlp_create_qkv_heads.cpp`**
   - Original: 25 test cases
   - Added regression: 15 test cases
   - **Total: 40 comprehensive C++ test cases**

### **Total New Test Coverage:**
- Python: **215 test cases** (40 bug + 175 regression)
- C++: **40 test cases**
- **Grand Total: 255 new test cases** 🎉

---

## Three-Step Testing Strategy

### **STEP 1: Baseline (No Changes)**

**Goal:** Verify all existing tests pass before any changes.

#### 1a. Existing C++ Tests
```bash
cd /workspace/bert-model-for-ttml
cmake --build build -- -j

# Run existing integration test
./build/test/tt_eager/integration_tests/test_bert
```

**Expected:** ✅ All tests PASS

#### 1b. Existing Python Tests
```bash
# Run existing nlp_create_qkv_heads tests
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py -v

# Run Falcon tests
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py::test_nlp_create_qkv_heads_falcon7b_test -v

# Run generic tests
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py::test_nlp_create_qkv_heads_test -v

# Run Llama tests
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py::test_nlp_create_qkv_heads_llama_test -v

# Run sharded tests
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py::test_sharded_nlp_create_qkv_heads_test -v
```

**Expected:** ✅ All tests PASS

---

### **STEP 2: Apply Test Changes Only (Bug Should Be Visible)**

**Goal:** Prove the bug exists. Tests with head_dim < 32 should FAIL.

#### Files to Apply (NO BUG FIX YET):
```
✅ tests/tt_eager/CMakeLists.txt
✅ tests/tt_eager/ops/test_nlp_create_qkv_heads.cpp (NEW)
✅ tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_head_dim_bug.py (NEW)
✅ tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_regression.py (NEW)
✅ tt-train/tests/CMakeLists.txt (remove line 22)
✅ DELETE tt-train/tests/ttnn_fixed/nlp_create_qkv_heads_head_dim_bug_test.cpp

❌ DO NOT APPLY BUG FIX FILES YET!
```

#### 2a. Rebuild with New Tests
```bash
cmake --build build -- -j
```

#### 2b. Run NEW C++ Tests (Should Show Bug)
```bash
./build/test/tt_eager/ops/test_nlp_create_qkv_heads
```

**Expected Results:**
- ❌ `bug_regression_head_dim_16` - **FAIL** (shape mismatch)
- ❌ `bug_regression_head_dim_8` - **FAIL** (shape mismatch)
- ❌ `bug_regression_batch2_head_dim_16` - **FAIL** (shape mismatch)
- ❌ All other tests with head_dim < 32 - **FAIL**
- ✅ `regression_boundary_head32_*` - **PASS** (boundary case)
- ✅ All tests with head_dim >= 32 - **PASS**
- ✅ All regression tests (head_dim 32, 64, 96, 128) - **PASS**

**Key Observation:** Should see ~10-15 failures, all with head_dim < 32.

#### 2c. Run NEW Python Bug Tests (Should Show Bug)
```bash
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_head_dim_bug.py -v
```

**Expected Results:**
- ❌ `test_*_head16_*` - **FAIL** (assertion on shape)
- ❌ `test_*_head8_*` - **FAIL** (assertion on shape)
- ✅ `test_*_head32_*_control` - **PASS** (boundary case)

**Key Observation:** Should see ~32 failures (head_dim=16,8 × 2 dtypes × 4 mem configs).

#### 2d. Run NEW Python Regression Tests (Should ALL PASS)
```bash
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_regression.py -v
```

**Expected:** ✅ **ALL 175 tests PASS** (these use head_dim = 32, 64, 96, 128)

**CRITICAL:** If any regression test fails here, STOP! The test setup is wrong.

#### 2e. Verify Existing Tests Still Pass
```bash
# Should still pass (they use head_dim >= 64)
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py -v
./build/test/tt_eager/integration_tests/test_bert
```

**Expected:** ✅ All existing tests still PASS

---

### **STEP 3: Apply Bug Fix (Everything Should Pass)**

**Goal:** Prove the fix works. All tests should now PASS.

#### Files to Apply:
```
✅ ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/nlp_create_qkv_heads_program_factory.cpp
✅ ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_boltz/device/nlp_create_qkv_heads_boltz_program_factory.cpp
```

#### 3a. Rebuild with Bug Fix
```bash
cmake --build build -- -j
```

#### 3b. Run ALL C++ Tests
```bash
./build/test/tt_eager/ops/test_nlp_create_qkv_heads
```

**Expected:** ✅ **ALL 40 tests PASS** (including head_dim=8,16 that failed before)

#### 3c. Run ALL New Python Tests
```bash
# Bug tests (should now pass)
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_head_dim_bug.py -v

# Regression tests (should still pass)
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_regression.py -v
```

**Expected:** ✅ **ALL 215 tests PASS**

#### 3d. Final Verification - Existing Tests
```bash
# Make absolutely sure we didn't break anything
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py -v
./build/test/tt_eager/integration_tests/test_bert

# Run full test suite if desired
pytest tests/tt_eager/python_api_testing/unit_testing/misc/ -v
```

**Expected:** ✅ All existing tests still PASS

---

## Test Coverage Summary

### **Mathematical Verification**

Our fix changes:
```cpp
OLD: q_num_tiles = num_heads * (head_dim / 32)
NEW: q_num_tiles = (num_heads * head_dim + 31) / 32
```

**Verification for "safe" head_dims:**

| head_dim | OLD Formula | NEW Formula | Match? |
|----------|-------------|-------------|--------|
| 32 | `num_heads * 1` | `(num_heads * 32 + 31) / 32 = num_heads` | ✅ YES |
| 64 | `num_heads * 2` | `(num_heads * 64 + 31) / 32 = num_heads * 2` | ✅ YES |
| 96 | `num_heads * 3` | `(num_heads * 96 + 31) / 32 = num_heads * 3` | ✅ YES |
| 128 | `num_heads * 4` | `(num_heads * 128 + 31) / 32 = num_heads * 4` | ✅ YES |

**For buggy head_dims:**

| head_dim | OLD Formula | NEW Formula | Fixed? |
|----------|-------------|-------------|--------|
| 8 | `num_heads * 0` = **0** 🐛 | `(num_heads * 8 + 31) / 32` ≈ `num_heads * 0.25` | ✅ YES |
| 16 | `num_heads * 0` = **0** 🐛 | `(num_heads * 16 + 31) / 32` ≈ `num_heads * 0.5` | ✅ YES |

### **Features Tested**

| Feature | Bug Tests | Regression Tests | Total |
|---------|-----------|------------------|-------|
| head_dim < 32 | ✅ 40 | ✅ 0 | 40 |
| head_dim = 32 (boundary) | ✅ 8 | ✅ 32 | 40 |
| head_dim = 64 | ✅ 0 | ✅ 60 | 60 |
| head_dim = 96 | ✅ 0 | ✅ 30 | 30 |
| head_dim = 128 | ✅ 0 | ✅ 40 | 40 |
| transpose_k_heads | ✅ 3 | ✅ 96 | 99 |
| separate_kv | ✅ 0 | ✅ 24 | 24 |
| GQA (Q≠KV heads) | ✅ 4 | ✅ 10 | 14 |
| Memory configs | ✅ All | ✅ All | All |
| Dtypes | ✅ All | ✅ All | All |

---

## Success Criteria

### ✅ Step 1 Success:
- All existing tests pass
- No failures in baseline

### ✅ Step 2 Success:
- Bug tests FAIL for head_dim < 32 (proves bug exists)
- Regression tests ALL PASS (proves tests are correct)
- Existing tests still PASS (proves we didn't break test setup)

### ✅ Step 3 Success:
- ALL new tests PASS (proves fix works)
- ALL existing tests still PASS (proves no regressions)

---

## Quick Test Commands

```bash
# Complete test run (after Step 3)
cd /workspace/bert-model-for-ttml

# C++ tests
./build/test/tt_eager/ops/test_nlp_create_qkv_heads
./build/test/tt_eager/integration_tests/test_bert

# Python tests - all nlp_create_qkv_heads tests
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads*.py -v

# Or run individually:
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py -v              # Existing
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_head_dim_bug.py -v # Bug tests
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_regression.py -v   # Regression
```

---

## Files Modified/Added

### Modified (5 files):
1. `tests/tt_eager/CMakeLists.txt`
2. `tt-train/tests/CMakeLists.txt`
3. `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/nlp_create_qkv_heads_program_factory.cpp`
4. `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_boltz/device/nlp_create_qkv_heads_boltz_program_factory.cpp`

### New (3 files):
1. `tests/tt_eager/ops/test_nlp_create_qkv_heads.cpp` (40 C++ tests)
2. `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_head_dim_bug.py` (40 bug tests)
3. `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_regression.py` (175 regression tests)

### Deleted (1 file):
1. `tt-train/tests/ttnn_fixed/nlp_create_qkv_heads_head_dim_bug_test.cpp` (wrong location)

---

## Confidence Level: ⭐⭐⭐⭐⭐ VERY HIGH

With **255 new test cases** covering:
- ✅ Bug reproduction and fix verification
- ✅ Comprehensive regression protection
- ✅ All feature combinations
- ✅ Boundary cases
- ✅ Mathematical equivalence verification

We can be **extremely confident** that:
1. The bug is real (Step 2 will show failures)
2. The fix works (Step 3 will show all tests pass)
3. No regressions introduced (175 regression tests protect existing functionality)
