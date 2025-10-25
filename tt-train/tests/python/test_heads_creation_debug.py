#!/usr/bin/env python3
"""Debug the heads creation reshape to see if it's scrambling data."""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

# Create a simple test case with known pattern
# For 128 embedding dim, 2 heads, head_dim=64:
# We want:
#   Head 0 to get dims 0-63
#   Head 1 to get dims 64-127

batch_size = 1
seq_len = 4  # Small for easy verification
embedding_dim = 128
num_heads = 2
head_dim = embedding_dim // num_heads

print(f"Testing heads creation with:")
print(f"  Sequence length: {seq_len}")
print(f"  Embedding dim: {embedding_dim}")
print(f"  Num heads: {num_heads}")
print(f"  Head dim: {head_dim}")

# Create QKV tensor with a known pattern
# Each token has a unique pattern so we can track where it goes
# Token i, dim j: value = i * 1000 + j
qkv_data = np.zeros((batch_size, 1, seq_len, embedding_dim * 3), dtype=np.float32)

for token_idx in range(seq_len):
    for dim_idx in range(embedding_dim):
        # Q values
        qkv_data[0, 0, token_idx, dim_idx] = token_idx * 1000 + dim_idx
        # K values
        qkv_data[0, 0, token_idx, embedding_dim + dim_idx] = token_idx * 1000 + dim_idx + 10000
        # V values
        qkv_data[0, 0, token_idx, 2 * embedding_dim + dim_idx] = token_idx * 1000 + dim_idx + 20000

print(f"\nCreated QKV tensor with pattern: token_idx * 1000 + dim_idx")
print(f"  Token 0, Q, dim 0: {qkv_data[0, 0, 0, 0]}")
print(f"  Token 0, Q, dim 63: {qkv_data[0, 0, 0, 63]}")
print(f"  Token 0, Q, dim 64: {qkv_data[0, 0, 0, 64]}")
print(f"  Token 0, Q, dim 127: {qkv_data[0, 0, 0, 127]}")
print(f"  Token 1, Q, dim 0: {qkv_data[0, 0, 1, 0]}")

# Convert to TTML tensor
qkv_ttml = ttml.autograd.Tensor.from_numpy(qkv_data)

# Call heads_creation
from ttml.ops import heads_creation

q, k, v = heads_creation(qkv_ttml, num_heads)

# Get back as numpy
q_np = q.to_numpy()  # Should be [B, H, S, E/H]
print(f"\nOutput Q shape: {q_np.shape}")
print(f"Expected shape: [1, 2, 4, 64]")

# Check what values ended up where
print(f"\n{'='*80}")
print("CHECKING HEAD ASSIGNMENTS:")
print(f"{'='*80}")

for head_idx in range(num_heads):
    for token_idx in range(seq_len):
        # Get the values for this head and token
        values = q_np[0, head_idx, token_idx, :]

        # Check first and last values
        first_val = values[0]
        last_val = values[-1]

        print(f"\nHead {head_idx}, Token {token_idx}:")
        print(f"  First value: {first_val:.1f}")
        print(f"  Last value: {last_val:.1f}")

        # What we EXPECT:
        # Head 0, Token i should have dims 0-63 from original token i
        # Head 1, Token i should have dims 64-127 from original token i
        expected_first = token_idx * 1000 + (head_idx * head_dim)
        expected_last = token_idx * 1000 + ((head_idx + 1) * head_dim - 1)

        print(f"  Expected first: {expected_first:.1f}")
        print(f"  Expected last: {expected_last:.1f}")

        if abs(first_val - expected_first) < 0.1 and abs(last_val - expected_last) < 0.1:
            print(f"  ✅ CORRECT!")
        else:
            print(f"  ❌ WRONG! Data is scrambled!")

            # Try to figure out what data ended up here
            # If first_val = X, then original was token (X // 1000), dim (X % 1000)
            if first_val >= 0:
                orig_token = int(first_val // 1000)
                orig_dim = int(first_val % 1000)
                print(f"  Actual first value comes from: Token {orig_token}, Dim {orig_dim}")

print(f"\n{'='*80}")
print("SUMMARY:")
print(f"{'='*80}")
print("If heads are CORRECT:")
print("  Head 0, Token 0 should have dims 0-63 from original Token 0")
print("  Head 1, Token 0 should have dims 64-127 from original Token 0")
print("  Head 0, Token 1 should have dims 0-63 from original Token 1")
print("  etc.")
