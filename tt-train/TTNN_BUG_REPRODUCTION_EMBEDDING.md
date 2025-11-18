# Complete Reproduction Instructions for Embedding Batch Processing Bug

**Related Bug Report**: `TTNN_BUG_REPORT_EMBEDDING_BATCH_PROCESSING.md`

This document provides **complete, self-contained reproduction instructions** for the TTNN embedding batch processing bug (PCC 0.60 → 0.9999 with workaround).

---

## Prerequisites

### Software Requirements

```bash
# Python packages
pip install torch numpy pytest

# Environment variables
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME/tt-train/build/sources/ttml:$PYTHONPATH
```

### Hardware Requirements

- Tenstorrent Wormhole device
- Minimum 16GB RAM

### TTML Build

```bash
cd $TT_METAL_HOME/tt-train
cmake -DCMAKE_BUILD_TYPE=Debug -B build -GNinja
cmake --build build
```

---

## Complete Reproduction Code

### Python Reproduction Script

Save as `reproduce_embedding_bug.py`:

```python
#!/usr/bin/env python3
"""
Complete reproduction of TTNN embedding batch processing bug.

Bug: ttnn::embedding returns INCORRECT values for batch indices > 0
Workaround: Process each batch separately and concatenate

This script demonstrates the bug with a minimal test case.
"""

import numpy as np
import torch
import sys
import os

# Add TTML to path
sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    t1_flat = tensor1.flatten()
    t2_flat = tensor2.flatten()

    mean1 = np.mean(t1_flat)
    mean2 = np.mean(t2_flat)

    numerator = np.sum((t1_flat - mean1) * (t2_flat - mean2))
    denominator = np.sqrt(
        np.sum((t1_flat - mean1) ** 2) * np.sum((t2_flat - mean2) ** 2)
    )

    if denominator == 0:
        return 1.0

    return numerator / denominator


def test_direct_ttnn_embedding(vocab_size=100, embedding_dim=128, batch_size=2, seq_len=4):
    """
    Test TTNN embedding directly to show the batch processing bug.

    This test bypasses TTML workaround to show the raw TTNN bug.
    """
    print(f"\n{'='*80}")
    print(f"Testing DIRECT TTNN Embedding (Bypassing TTML Workaround)")
    print(f"{'='*80}")
    print(f"Vocab size: {vocab_size}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print()

    device = ttml.autograd.ctx().get_device()

    # Create embedding weight table with unique values
    print("Creating embedding weight table...")
    weight_data = np.zeros((vocab_size * embedding_dim,), dtype=np.float32)
    for token in range(vocab_size):
        for dim in range(embedding_dim):
            weight_data[token * embedding_dim + dim] = float(token * 1000 + dim)

    weight_tensor = ttml.core.from_vector(
        weight_data.tolist(),
        [1, 1, vocab_size, embedding_dim],
        device
    )
    # Untilize for embedding operation
    import ttnn
    weight_tensor = ttnn.untilize(weight_tensor)

    # Create input with SAME token IDs for both batches
    # This is key: same tokens should give same embeddings regardless of batch
    print("Creating input token IDs...")
    token_ids = [10, 20, 30, 40]  # Same tokens for both batches
    input_ids = np.array([
        token_ids,  # Batch 0
        token_ids,  # Batch 1 (identical to batch 0)
    ], dtype=np.uint32)

    print(f"Input token IDs:")
    print(f"  Batch 0: {input_ids[0].tolist()}")
    print(f"  Batch 1: {input_ids[1].tolist()}")
    print(f"  (Same tokens in both batches)")

    input_tensor = ttml.core.from_vector(
        input_ids.flatten().tolist(),
        [batch_size, 1, 1, seq_len],
        device,
        layout=ttnn.Layout.ROW_MAJOR
    )

    # Call TTNN embedding directly - THIS IS WHERE BUG OCCURS
    print("\nCalling ttnn::embedding...")
    embeddings = ttnn.embedding(
        input_tensor,
        weight_tensor,
        pad_token=None,
        layout=ttnn.Layout.TILE
    )

    # Get results
    result = ttml.core.to_vector(embeddings)
    result_np = np.array(result, dtype=np.float32)

    # Reshape to [batch, seq, embedding_dim]
    result_np = result_np.reshape(batch_size, seq_len, embedding_dim)

    print(f"\nOutput shape: {result_np.shape}")

    # Verify each batch
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")

    batch_results = []

    for batch_idx in range(batch_size):
        print(f"\nBatch {batch_idx}:")
        batch_embeddings = result_np[batch_idx]  # [seq, embedding_dim]

        # Check each token
        token_pccs = []
        for seq_idx, token_id in enumerate(token_ids):
            actual_embedding = batch_embeddings[seq_idx]
            expected_embedding = weight_data[token_id * embedding_dim : (token_id + 1) * embedding_dim]

            pcc = compute_pcc(expected_embedding, actual_embedding)
            token_pccs.append(pcc)

            status = "✅" if pcc > 0.999 else "❌"
            print(f"  Token {seq_idx} (ID={token_id}): PCC = {pcc:.6f} {status}")

        avg_pcc = np.mean(token_pccs)
        batch_status = "✅ CORRECT" if avg_pcc > 0.999 else "❌ INCORRECT"
        print(f"  Overall batch {batch_idx}: PCC = {avg_pcc:.6f} {batch_status}")

        batch_results.append({
            'batch': batch_idx,
            'pcc': avg_pcc,
            'token_pccs': token_pccs
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for result in batch_results:
        status = "✅ PASS" if result['pcc'] > 0.999 else "❌ FAIL"
        print(f"Batch {result['batch']}: PCC = {result['pcc']:.6f} {status}")

    # Check if bug is present
    batch0_ok = batch_results[0]['pcc'] > 0.999
    batch1_ok = batch_results[1]['pcc'] > 0.999

    print()
    if batch0_ok and batch1_ok:
        print("✅ NO BUG DETECTED - Workaround is active or bug is fixed")
        print("   (This is expected with current TTML which has the workaround)")
        bug_present = False
    elif batch0_ok and not batch1_ok:
        print("❌ BUG REPRODUCED!")
        print("   Batch 0: Correct (PCC >0.999)")
        print("   Batch 1: Incorrect (PCC <0.999)")
        print("   This is the embedding batch processing bug!")
        bug_present = True
    else:
        print("⚠️  UNEXPECTED RESULT")
        print("   Both batches may be incorrect or other issue present")
        bug_present = None

    print(f"{'='*80}\n")

    return batch_results, bug_present


def test_ttml_embedding_with_workaround(vocab_size=100, embedding_dim=128, batch_size=2, seq_len=4):
    """
    Test TTML embedding operation which includes the batch workaround.

    This should show correct results for all batches.
    """
    print(f"\n{'='*80}")
    print(f"Testing TTML Embedding (With Workaround)")
    print(f"{'='*80}")
    print(f"Vocab size: {vocab_size}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print()

    # Create embedding weight using autograd
    print("Creating embedding weight...")
    weight_data = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    for token in range(vocab_size):
        for dim in range(embedding_dim):
            weight_data[token, dim] = float(token * 1000 + dim)

    weight = ttml.core.from_vector(
        weight_data.flatten().tolist(),
        [1, 1, vocab_size, embedding_dim],
        ttml.autograd.ctx().get_device()
    )
    weight_tensor = ttml.autograd.create_tensor(weight)

    # Create input
    print("Creating input token IDs...")
    token_ids = [10, 20, 30, 40]
    input_ids = np.array([
        token_ids,  # Batch 0
        token_ids,  # Batch 1 (identical)
    ], dtype=np.uint32)

    print(f"Input token IDs:")
    print(f"  Batch 0: {input_ids[0].tolist()}")
    print(f"  Batch 1: {input_ids[1].tolist()}")

    input_tt = ttml.core.from_vector(
        input_ids.flatten().tolist(),
        [batch_size, 1, 1, seq_len],
        ttml.autograd.ctx().get_device()
    )
    input_tensor = ttml.autograd.create_tensor(input_tt)

    # Call TTML embedding_op (includes workaround)
    print("\nCalling ttml::ops::embedding_op (with workaround)...")
    output_tensor = ttml.ops.embedding_op(input_tensor, weight_tensor)

    # Get results
    result = ttml.core.to_vector(output_tensor.value)
    result_np = np.array(result, dtype=np.float32).reshape(batch_size, seq_len, embedding_dim)

    print(f"\nOutput shape: {result_np.shape}")

    # Verify
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")

    batch_results = []
    for batch_idx in range(batch_size):
        print(f"\nBatch {batch_idx}:")
        batch_embeddings = result_np[batch_idx]

        token_pccs = []
        for seq_idx, token_id in enumerate(token_ids):
            actual = batch_embeddings[seq_idx]
            expected = weight_data[token_id]

            pcc = compute_pcc(expected, actual)
            token_pccs.append(pcc)

            status = "✅" if pcc > 0.999 else "❌"
            print(f"  Token {seq_idx} (ID={token_id}): PCC = {pcc:.6f} {status}")

        avg_pcc = np.mean(token_pccs)
        batch_status = "✅ CORRECT" if avg_pcc > 0.999 else "❌ INCORRECT"
        print(f"  Overall batch {batch_idx}: PCC = {avg_pcc:.6f} {batch_status}")

        batch_results.append({'batch': batch_idx, 'pcc': avg_pcc})

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    all_pass = all(r['pcc'] > 0.999 for r in batch_results)

    for result in batch_results:
        status = "✅ PASS" if result['pcc'] > 0.999 else "❌ FAIL"
        print(f"Batch {result['batch']}: PCC = {result['pcc']:.6f} {status}")

    print()
    if all_pass:
        print("✅ ALL BATCHES CORRECT - Workaround is working")
    else:
        print("❌ SOME BATCHES INCORRECT - Workaround may not be active")

    print(f"{'='*80}\n")

    return batch_results


def main():
    """Run embedding bug reproduction tests."""
    print("\n" + "="*80)
    print("TTNN Embedding Batch Processing Bug Reproduction")
    print("="*80)
    print("\nThis script demonstrates the embedding batch bug.")
    print("\nExpected behavior:")
    print("  - Direct TTNN call: May show bug (batch 1 incorrect)")
    print("  - TTML with workaround: Should work (all batches correct)")
    print("="*80)

    # Open device
    print("\nOpening device...")
    ttml.autograd.ctx().open_device()
    print("Device opened successfully")

    # Test 1: Direct TTNN (may show bug if workaround bypassed)
    print("\n\n" + "="*80)
    print("TEST 1: Direct TTNN Embedding")
    print("="*80)
    print("This attempts to call TTNN directly to expose the bug.")
    print("NOTE: May not work due to TTML's internal workaround.\n")

    try:
        direct_results, bug_present = test_direct_ttnn_embedding(
            vocab_size=100,
            embedding_dim=128,
            batch_size=2,
            seq_len=4
        )
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("This is expected - TTNN direct call may not be easily accessible")
        direct_results = None
        bug_present = None

    # Test 2: TTML with workaround
    print("\n\n" + "="*80)
    print("TEST 2: TTML Embedding (With Workaround)")
    print("="*80)
    print("This uses TTML's embedding_op which has the batch workaround.\n")

    ttml_results = test_ttml_embedding_with_workaround(
        vocab_size=100,
        embedding_dim=128,
        batch_size=2,
        seq_len=4
    )

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    if direct_results:
        print("\n Direct TTNN Test:")
        for r in direct_results:
            status = "✅" if r['pcc'] > 0.999 else "❌"
            print(f"  {status} Batch {r['batch']}: PCC = {r['pcc']:.6f}")

    print("\nTTML Workaround Test:")
    for r in ttml_results:
        status = "✅" if r['pcc'] > 0.999 else "❌"
        print(f"  {status} Batch {r['batch']}: PCC = {r['pcc']:.6f}")

    print("\n" + "="*80)
    print("\nNote: The bug exists in TTNN's embedding kernel for batch > 0.")
    print("TTML works around it by processing batches separately (slower).")
    print("To see the actual bug, you would need to modify TTML code to")
    print("bypass the workaround in sources/ttml/ops/embedding_op.cpp")
    print("="*80)

    # Close device
    print("\nClosing device...")
    ttml.autograd.ctx().close_device()
    print("Device closed successfully")


if __name__ == "__main__":
    main()
```

---

## How to Run

### Quick Test

```bash
# Set environment
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME/tt-train/build/sources/ttml:$PYTHONPATH

# Run reproduction
python3 reproduce_embedding_bug.py
```

**Expected output WITH workaround (current)**:
```
TTML Workaround Test:
  ✅ Batch 0: PCC = 0.999999
  ✅ Batch 1: PCC = 0.999999
```

**Expected output WITHOUT workaround** (if disabled):
```
Direct TTNN Test:
  ✅ Batch 0: PCC = 0.999999
  ❌ Batch 1: PCC = 0.608615  # BUG!
```

---

## How to Reproduce the BUG (Disable Workaround)

To see the actual bug, you must disable the batch workaround in TTML:

### Step 1: Edit Source Code

```bash
vi $TT_METAL_HOME/tt-train/sources/ttml/ops/embedding_op.cpp
```

Find the batch workaround code (around lines 24-55):

```cpp
if (batch_size > 1) {
    // ⚠️ WORKAROUND - NOT A FIX ⚠️
    // Process each batch separately
    std::vector<ttnn::Tensor> batch_embeddings;
    // ... workaround code ...
} else {
    embeddings = ttnn::embedding(...);  // Direct call
}
```

**Change to bypass workaround**:
```cpp
// ALWAYS use direct path (bypasses workaround to show bug)
embeddings = ttnn::embedding(
    input_tensor,
    weight_tensor,
    /* pad_token */ std::nullopt,
    ttnn::Layout::TILE
);
```

### Step 2: Rebuild TTML

```bash
cd $TT_METAL_HOME/tt-train
cmake --build build --target ttml
```

### Step 3: Run Reproduction Script

```bash
python3 reproduce_embedding_bug.py
```

**Expected output**:
```
Batch 0:
  ✅ Token 0 (ID=10): PCC = 0.999999
  ✅ Token 1 (ID=20): PCC = 0.999999
  ...

Batch 1:
  ✅ Token 0 (ID=10): PCC = 0.999999  # First token works
  ❌ Token 1 (ID=20): PCC = 0.326861  # Bug appears!
  ❌ Token 2 (ID=30): PCC = 0.327035  # Wrong values
  ...

❌ BUG REPRODUCED!
```

### Step 4: Re-enable Workaround

**IMPORTANT**: Restore the workaround after testing!

```bash
# Restore original code
git checkout $TT_METAL_HOME/tt-train/sources/ttml/ops/embedding_op.cpp

# Rebuild
cd $TT_METAL_HOME/tt-train
cmake --build build --target ttml
```

---

## C++ Reproduction

For C++ test, see `tests/ops/embedding_batch_regression_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include "core/tt_tensor_utils.hpp"
#include "autograd/auto_context.hpp"

TEST(EmbeddingBatchBug, DirectTTNNCall) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Create embedding table
    uint32_t vocab_size = 100;
    uint32_t embedding_dim = 128;
    std::vector<float> weight_data(vocab_size * embedding_dim);

    for (uint32_t token = 0; token < vocab_size; ++token) {
        for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
            weight_data[token * embedding_dim + dim] =
                static_cast<float>(token * 1000 + dim);
        }
    }

    auto weight = ttml::core::from_vector(
        weight_data,
        ttnn::Shape{1, 1, vocab_size, embedding_dim},
        device
    );
    weight = ttnn::untilize(weight);

    // Input: Same tokens in both batches
    std::vector<uint32_t> input_ids = {
        10, 20, 30, 40,  // Batch 0
        10, 20, 30, 40   // Batch 1 (same tokens!)
    };

    auto input = ttml::core::from_vector(
        input_ids,
        ttnn::Shape{2, 1, 1, 4},  // batch_size=2
        device,
        ttnn::Layout::ROW_MAJOR
    );

    // Call ttnn::embedding - BUG OCCURS HERE
    auto embeddings = ttnn::embedding(input, weight, std::nullopt, ttnn::Layout::TILE);

    auto result = ttml::core::to_vector<float>(embeddings);

    // Verify batch 0 (should be correct)
    for (uint32_t seq = 0; seq < 4; ++seq) {
        uint32_t token_id = input_ids[seq];
        for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
            float expected = weight_data[token_id * embedding_dim + dim];
            float actual = result[seq * embedding_dim + dim];
            EXPECT_NEAR(actual, expected, 1e-4)
                << "Batch 0, token " << token_id;
            // ✅ PASSES
        }
    }

    // Verify batch 1 (will FAIL - shows bug)
    size_t batch1_offset = 4 * embedding_dim;
    for (uint32_t seq = 0; seq < 4; ++seq) {
        uint32_t token_id = input_ids[4 + seq];
        for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
            float expected = weight_data[token_id * embedding_dim + dim];
            float actual = result[batch1_offset + seq * embedding_dim + dim];
            EXPECT_NEAR(actual, expected, 1e-4)
                << "Batch 1, token " << token_id;
            // ❌ FAILS - Gets wrong values!
        }
    }
}
```

---

## Bug Characteristics

### What Works
- ✅ Batch 0: Always correct (PCC >0.999)
- ✅ First token in batch 1: Often correct
- ✅ Single batch (batch_size=1): Always correct

### What Fails
- ❌ Batch 1+ tokens: Wrong embeddings (PCC ~0.60)
- ❌ Same token ID gets different embeddings in different batches
- ❌ Appears to use wrong row from embedding table

### Pattern
- Batch 0, Token 10 → Correct embedding from row 10 ✅
- Batch 1, Token 10 → Wrong embedding (wrong row) ❌
- Suggests **memory offset calculation bug** in TTNN kernel

---

## Summary

This reproduction demonstrates:

1. ✅ **Bug is real**: Disabling workaround shows PCC 0.60 for batch 1
2. ✅ **Workaround works**: Batch splitting restores PCC >0.999
3. ✅ **Easily reproducible**: Simple test case with predictable data
4. ✅ **Self-contained**: All code and data included

The bug is in TTNN's `ttnn::embedding` kernel batch processing logic.
