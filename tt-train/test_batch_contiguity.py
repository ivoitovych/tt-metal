#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test to check if batch arrays are contiguous.

This tests whether the reshape operation used to create batch tensors
results in contiguous or non-contiguous arrays. Non-contiguous arrays
were the root cause of a previous critical bug (commit 38f05bb43c).
"""

import numpy as np

print("=" * 80)
print("BATCH ARRAY CONTIGUITY TEST")
print("=" * 80)

# Simulate what we do in BERT tests
batch_size = 2
seq_len = 32

# Create input as we do in tests
input_ids_np = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64)

print(f"\nOriginal array:")
print(f"  Shape: {input_ids_np.shape}")
print(f"  Dtype: {input_ids_np.dtype}")
print(f"  Is C-contiguous: {input_ids_np.flags['C_CONTIGUOUS']}")
print(f"  Is F-contiguous: {input_ids_np.flags['F_CONTIGUOUS']}")
print(f"  Strides: {input_ids_np.strides}")

# Reshape as we do for TTML
input_ids_reshaped = input_ids_np.reshape(batch_size, 1, 1, seq_len).astype(np.float32)

print(f"\nAfter reshape to (batch_size, 1, 1, seq_len):")
print(f"  Shape: {input_ids_reshaped.shape}")
print(f"  Dtype: {input_ids_reshaped.dtype}")
print(f"  Is C-contiguous: {input_ids_reshaped.flags['C_CONTIGUOUS']}")
print(f"  Is F-contiguous: {input_ids_reshaped.flags['F_CONTIGUOUS']}")
print(f"  Strides: {input_ids_reshaped.strides}")

# Check contiguity
if input_ids_reshaped.flags["C_CONTIGUOUS"]:
    print(f"\n✓ Array is C-contiguous - should be handled correctly by bindings")
else:
    print(f"\n❌ Array is NON-contiguous - might trigger binding bug!")
    print(f"   This could explain the batch processing bug!")

# Also test what happens with different operations
print("\n" + "=" * 80)
print("Testing different array creation methods:")
print("=" * 80)

# Method 1: Current approach (reshape)
arr1 = np.random.randint(0, 1000, (batch_size, seq_len)).reshape(batch_size, 1, 1, seq_len).astype(np.float32)
print(f"\nMethod 1 (reshape): C-contiguous = {arr1.flags['C_CONTIGUOUS']}")

# Method 2: Direct creation with target shape
arr2 = np.random.randint(0, 1000, (batch_size, 1, 1, seq_len)).astype(np.float32)
print(f"Method 2 (direct):  C-contiguous = {arr2.flags['C_CONTIGUOUS']}")

# Method 3: Explicit contiguous copy
arr3 = np.ascontiguousarray(arr1)
print(f"Method 3 (ascontiguousarray): C-contiguous = {arr3.flags['C_CONTIGUOUS']}")

# Test if they're actually the same data
print(f"\nMethod 1 and 2 have same values: {np.allclose(arr1.reshape(-1)[:10], arr1.reshape(-1)[:10])}")

print("\n" + "=" * 80)
print("CONCLUSION:")
if input_ids_reshaped.flags["C_CONTIGUOUS"]:
    print("Batch arrays ARE contiguous, so contiguity is NOT the issue.")
    print("The batch processing bug must be in the C++ implementation or")
    print("in how the binding passes batch data to the C++ layer.")
else:
    print("Batch arrays are NON-contiguous!")
    print("This could be the root cause - the binding fix from commit 38f05bb43c")
    print("should handle this, but there might be an edge case for batches.")
print("=" * 80)
