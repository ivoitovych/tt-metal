#!/usr/bin/env python3
"""
Simple verification that the masking fix is working correctly.
"""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

# Create simple test data
batch_size = 1
num_heads = 2
seq_len = 8
head_dim = 4

# Simple Q, K, V - all ones for simplicity
q_data = np.ones((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
k_data = np.ones((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
v_data = np.arange(batch_size * num_heads * seq_len * head_dim, dtype=np.float32).reshape(
    batch_size, num_heads, seq_len, head_dim
)

# Mask: first 4 positions = attend (1), last 4 positions = masked (0)
mask_data = np.array([[[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32)

print(f"Input shapes:")
print(f"  Q: {q_data.shape}")
print(f"  K: {k_data.shape}")
print(f"  V: {v_data.shape}")
print(f"  Mask: {mask_data.shape}")
print(f"\nMask values: {mask_data[0, 0, 0, :]}")
print(f"\nV values (first head, all positions):")
print(v_data[0, 0, :, :])

# Convert to TTML tensors
q_ttml = ttml.autograd.Tensor.from_numpy(q_data)
k_ttml = ttml.autograd.Tensor.from_numpy(k_data)
v_ttml = ttml.autograd.Tensor.from_numpy(v_data)
mask_ttml = ttml.autograd.Tensor.from_numpy(mask_data)

# Run attention
output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)
output_np = output_ttml.to_numpy()

print(f"\nOutput (first head, first position):")
print(output_np[0, 0, 0, :])
print(f"\nOutput (first head, all positions):")
print(output_np[0, 0, :, :])

# Expected behavior with correct masking:
# - Attention should ONLY attend to first 4 positions (0-3)
# - Should NOT attend to last 4 positions (4-7)
# - Output should be weighted average of V values for positions 0-3 ONLY

# With all Q,K = 1, and scale = 1/sqrt(4) = 0.5:
# - QK^T scores before mask = 4 (since Q @ K = [1,1,1,1] @ [1,1,1,1]^T = 4)
# - Scaled scores = 4 * 0.5 = 2.0
# - After masking: positions 0-3 get 2.0, positions 4-7 get -1e9
# - After softmax: positions 0-3 get ~0.25 each, positions 4-7 get ~0.0
# - Output should be ~0.25*V[0] + 0.25*V[1] + 0.25*V[2] + 0.25*V[3]

expected_output_pos0 = 0.25 * (v_data[0, 0, 0, :] + v_data[0, 0, 1, :] + v_data[0, 0, 2, :] + v_data[0, 0, 3, :])

print(f"\nExpected output (first head, first position): {expected_output_pos0}")
print(f"Actual output (first head, first position):   {output_np[0, 0, 0, :]}")

diff = np.abs(expected_output_pos0 - output_np[0, 0, 0, :])
print(f"Difference: {diff}")
print(f"Max difference: {diff.max():.6f}")

# Check if masked positions (4-7) contributed to output
# If they did, output would include values from V[4-7], which are larger
# If masking works, output should ONLY have values from V[0-3]
if diff.max() < 0.5:  # Allow for numerical errors
    print(f"\n✅ MASKING FIX WORKS! Masked positions properly ignored.")
else:
    print(f"\n❌ MASKING STILL BROKEN! Masked positions are contributing to output.")

    # Debug: what would output be if masking is completely broken?
    # (attending to all 8 positions equally)
    broken_output = v_data[0, 0, :, :].mean(axis=0)
    print(f"\nIf masking completely broken (attend to all): {broken_output}")
