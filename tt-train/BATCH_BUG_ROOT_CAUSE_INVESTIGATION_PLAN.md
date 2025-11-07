# BERT Batch Processing Bug - Root Cause Investigation Plan

**Date**: 2025-11-07
**Branch**: `ivoitovych/bert-model-for-ttml-completeness-implementation`
**Status**: Ready to Execute

---

## Current Situation

### What We Know

**The Workaround (Band-Aid)**:
- Location: `tt-train/sources/ttml/ops/embedding_op.cpp:27-92`
- Method: Slice-and-concatenate - processes each batch sample individually
- Effect: **Masks bug symptoms** - all tests pass
- Problem: **Does NOT fix root cause** - bug is still present, just hidden

**What Works (Verified)**:
1. ✅ Direct embedding pipeline (all ops in isolation)
2. ✅ BERT.get_embeddings() method
3. ✅ Transformer encoder blocks
4. ✅ Pooler (CLS token extraction)
5. ✅ Classification head
6. ✅ Full BertForSequenceClassification (with workaround)

**What We Don't Know**:
- ❓ **Where the actual bug is** (location unknown)
- ❓ **What causes identical outputs** when batch_size > 1
- ❓ **Why workaround masks it** (what exactly is being fixed by slice-and-concat?)

### Evidence from Clean Branch

The clean bug reproduction branch (`ivoitovych/ttnn-embedding-batch-bug-reproduction`) shows:
- ttnn::embedding() works correctly with clean 2D tensors
- Direct ttnn operations handle batches correctly
- **Therefore**: Bug is NOT in ttnn library, bug is in BERT implementation

---

## Investigation Plan

### Phase 1: Baseline - Remove Workaround

**Goal**: Restore original code to reproduce bug

**Steps**:
1. Create backup of current `embedding_op.cpp` with workaround
2. Restore original code (before commit 10a9d642d2)
3. Rebuild tt-train
4. Verify bug reproduces:
   - Run Python `debug_batch_processing.py` → Should FAIL
   - Run C++ `BertBatchBugTest` → Should FAIL
   - Run C++ `BertBatchIsolationTest` → Should FAIL

**Expected Result**: Bug symptoms return - batch samples produce identical outputs

**Commit**: "test: Remove embedding workaround to expose root cause"

---

### Phase 2: Progressive Component Testing

**Goal**: Isolate exact component where bug emerges

**Test Sequence** (run each after removing workaround):

#### Test 2.1: Direct ops::embedding_op
```cpp
TEST_F(BugIsolation, DirectEmbeddingOp) {
    // Test ops::embedding_op directly with batch=2
    // Expected: PASS (works in isolation)
}
```

#### Test 2.2: ops::embedding_op → ops::add (position embeddings)
```cpp
TEST_F(BugIsolation, EmbeddingPlusPositions) {
    // Token embeddings + position embeddings
    // Expected: Check if bug appears here
}
```

#### Test 2.3: Full embedding pipeline (manual)
```cpp
TEST_F(BugIsolation, ManualEmbeddingPipeline) {
    // Token + Position + TokenType + LayerNorm + Dropout
    // This is BertEmbeddingPipelineTest::FullPipelineStepByStep
    // Expected: Check if bug appears
}
```

#### Test 2.4: BERT.get_embeddings()
```cpp
TEST_F(BugIsolation, BertGetEmbeddings) {
    // BERT model's get_embeddings() method
    // This is BertEmbeddingPipelineTest::BertModelGetEmbeddings
    // Expected: Check if bug appears in BERT's wiring
}
```

#### Test 2.5: BERT with transformer blocks
```cpp
TEST_F(BugIsolation, BertWithTransformer) {
    // BERT with 1 transformer block
    // This is BertEmbeddingPipelineTest::BertModelWithTransformerBlocks
    // Expected: Check if bug appears in attention/FFN
}
```

#### Test 2.6: Full BertForSequenceClassification
```cpp
TEST_F(BugIsolation, FullSequenceClassification) {
    // Complete model with pooler + classifier
    // This is BertBatchBugTest::DifferentInputsProduceDifferentOutputs
    // Expected: FAIL (bug manifests here)
}
```

**Analysis Method**:
- Record FIRST test that fails (outputs become identical)
- That's where the bug emerges
- Compare with previous passing test to isolate the problematic operation

**Commit**: "test: Progressive isolation shows bug location at [component]"

---

### Phase 3: Deep Dive into Failing Component

**Goal**: Understand why the specific component fails

**Investigation Areas** (based on which test fails):

#### If bug is in ops::embedding_op direct call:
- Examine tensor shapes before/after embedding
- Check autograd context state
- Verify weight tensor initialization
- Test with different input shapes

#### If bug is in position embedding addition:
- Check broadcasting behavior with actual BERT tensors
- Verify position weight tensor shape and values
- Test if ops::add behaves differently with BERT's specific shapes

#### If bug is in BERT.get_embeddings():
- Compare with manual pipeline (which works)
- Check module state (m_token_embeddings, m_position_embeddings, etc.)
- Verify operator() calls vs direct function calls
- Check if ModuleBase wrapper affects behavior

#### If bug is in transformer block:
- Test attention mechanism in isolation
- Test FFN in isolation
- Check residual connections
- Verify layer norm between components

**Tools**:
- Print tensor shapes at each step
- Print first N values of each tensor
- Compare batch[0] vs batch[1] at each operation
- Check for tensor aliasing or shared memory issues

**Commit**: "debug: Detailed analysis of [failing component]"

---

### Phase 4: Root Cause Identification

**Goal**: Identify exact line(s) of code causing the bug

**Hypotheses to Test**:

1. **Tensor Shape Issue**:
   - Bug: Incorrect tensor reshaping that collapses batch dimension
   - Test: Print shapes before/after each reshape
   - Fix: Correct reshape to preserve batch dimension

2. **Memory Aliasing**:
   - Bug: Multiple batch samples pointing to same memory
   - Test: Check tensor pointers, verify copy vs reference
   - Fix: Ensure proper tensor copies

3. **Autograd State Issue**:
   - Bug: Autograd context not properly handling batch dimension
   - Test: Run same operations with/without autograd
   - Fix: Correct autograd tensor creation/wrapping

4. **Broadcasting Bug**:
   - Bug: Broadcasting incorrectly duplicates instead of broadcasts
   - Test: Verify ttnn::add behavior with BERT's exact tensor shapes
   - Fix: Correct broadcasting operation or tensor preparation

5. **Module State Issue**:
   - Bug: Module (EmbeddingModule, etc.) maintains shared state across batch
   - Test: Create fresh module for each sample vs reuse
   - Fix: Ensure stateless execution or proper state reset

6. **Weight Tensor Issue**:
   - Bug: Weight tensor shape incompatible with batched input
   - Test: Print weight tensor shapes, verify against spec
   - Fix: Correct weight tensor initialization

**Debugging Strategy**:
```cpp
// At each operation, add:
std::cout << "Operation: [name]\n";
std::cout << "  Input shape: " << input->get_shape() << "\n";
std::cout << "  Sample 0 first value: " << vec[0] << "\n";
std::cout << "  Sample 1 first value: " << vec[seq_len * emb_dim] << "\n";
if (abs(sample0 - sample1) < 0.01) {
    std::cout << "  ❌ BUG EMERGES HERE!\n";
}
```

**Commit**: "fix: Identify root cause - [specific issue]"

---

### Phase 5: Implement Proper Fix

**Goal**: Fix the actual bug, not symptoms

**Fix Requirements**:
- Must address root cause, not mask symptoms
- Must preserve batch processing efficiency
- Must pass all tests without workarounds
- Must match HuggingFace outputs

**Testing After Fix**:
1. All C++ tests pass (without workarounds)
2. All Python tests pass (without workarounds)
3. debug_batch_processing.py passes
4. Batch and individual runs produce matching results

**Commit**: "fix: [Description of actual fix]"

---

### Phase 6: Verify Fix and Remove Workarounds

**Goal**: Clean up all band-aids

**Steps**:
1. Verify embedding_op.cpp no longer needs slice-and-concat
2. Remove workaround code completely
3. Remove workarounds from tests (if any)
4. Run full test suite
5. Verify Python integration tests
6. Update documentation to reflect fix

**Commit**: "refactor: Remove embedding workaround - bug fixed properly"

---

## Test Matrix

| Component | Test | Expected Before Fix | Expected After Fix |
|-----------|------|--------------------|--------------------|
| Direct ops | BroadcastingHypothesisTest | PASS | PASS |
| Manual pipeline | FullPipelineStepByStep | PASS | PASS |
| BERT embeddings | BertModelGetEmbeddings | **TBD** | PASS |
| BERT + transformer | BertModelWithTransformerBlocks | **TBD** | PASS |
| Full model (C++) | BertBatchBugTest | **TBD** | PASS |
| Full model (Python) | debug_batch_processing.py | **TBD** | PASS |

**TBD** = Need to test without workaround to determine actual status

---

## Success Criteria

### Must Have (Blocking)
- [ ] Root cause identified and documented
- [ ] Proper fix implemented (not workaround)
- [ ] All C++ tests pass without workarounds
- [ ] All Python tests pass without workarounds
- [ ] Batch processing matches individual processing

### Should Have
- [ ] Performance is maintained (no slice-and-concat overhead)
- [ ] Fix is minimal and targeted (not invasive refactor)
- [ ] Documentation explains what was wrong and how it was fixed

### Nice to Have
- [ ] Regression test that catches this specific bug
- [ ] Understanding of why workaround masked the symptoms

---

## Risk Assessment

### Low Risk
- Tests are comprehensive and will catch regressions
- Workaround is isolated to one file
- Bug affects batch processing only (batch=1 works)

### Medium Risk
- Root cause is unknown - investigation may take time
- Fix might require changes to core operations
- May discover additional related bugs

### High Risk
- Bug might be in ttnn library (outside our control)
  - **Mitigation**: Clean branch tests suggest bug is in BERT, not ttnn
- Fix might break other functionality
  - **Mitigation**: Comprehensive test suite will catch issues

---

## Timeline Estimate

- **Phase 1** (Remove workaround): 30 minutes
- **Phase 2** (Progressive testing): 1-2 hours
- **Phase 3** (Deep dive): 2-4 hours (depends on where bug is)
- **Phase 4** (Root cause ID): 1-2 hours (depends on complexity)
- **Phase 5** (Implement fix): 1-3 hours (depends on fix complexity)
- **Phase 6** (Cleanup & verify): 1 hour

**Total**: 6-12 hours of focused investigation

---

## Notes

- This plan assumes bug is in BERT implementation (not ttnn)
- If bug turns out to be in ttnn, will need to engage ttnn team
- All commits should be made incrementally to preserve investigation history
- Document all findings in commit messages for future reference

---

## Ready to Execute

All prerequisite tests are in place:
- ✅ BroadcastingHypothesisTest (tests individual ops)
- ✅ BertEmbeddingPipelineTest (tests pipeline and BERT components)
- ✅ BertBatchBugTest (end-to-end bug detection)
- ✅ debug_batch_processing.py (Python validation)

**Next Command**: Remove workaround and begin Phase 1
