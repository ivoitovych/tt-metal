#!/usr/bin/env python3
"""
Test data flow through Python bindings.

This test verifies that data is correctly converted from numpy -> TTML -> computation -> numpy.
"""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def test_simple_attention_roundtrip():
    """Test that simple attention computation matches expected result."""
    print("=" * 80)
    print("TEST: Simple Attention Data Flow")
    print("=" * 80)

    # Create simple test data - same as C++ test
    batch_size = 1
    num_heads = 2
    seq_len = 8
    head_dim = 4

    # Use specific values for debugging
    np.random.seed(42)
    q_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    k_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    v_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

    print(f"\nInput data shapes:")
    print(f"  Q: {q_data.shape}, dtype: {q_data.dtype}")
    print(f"  K: {k_data.shape}, dtype: {k_data.dtype}")
    print(f"  V: {v_data.shape}, dtype: {v_data.dtype}")
    print(f"\nInput data ranges:")
    print(f"  Q: min={q_data.min():.6f}, max={q_data.max():.6f}, mean={q_data.mean():.6f}")
    print(f"  K: min={k_data.min():.6f}, max={k_data.max():.6f}, mean={k_data.mean():.6f}")
    print(f"  V: min={v_data.min():.6f}, max={v_data.max():.6f}, mean={v_data.mean():.6f}")

    # Reference computation in numpy
    scale = 1.0 / np.sqrt(head_dim)
    attn_scores = np.matmul(q_data, k_data.transpose(0, 1, 3, 2)) * scale

    # Softmax
    attn_scores_exp = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
    attn_weights = attn_scores_exp / attn_scores_exp.sum(axis=-1, keepdims=True)

    output_ref = np.matmul(attn_weights, v_data)

    print(f"\nReference output:")
    print(f"  Shape: {output_ref.shape}")
    print(f"  Min={output_ref.min():.6f}, max={output_ref.max():.6f}, mean={output_ref.mean():.6f}")

    # TTML computation
    q_ttml = ttml.autograd.Tensor.from_numpy(q_data)
    k_ttml = ttml.autograd.Tensor.from_numpy(k_data)
    v_ttml = ttml.autograd.Tensor.from_numpy(v_data)

    output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=None)
    output_ttml_np = output_ttml.to_numpy()

    print(f"\nTTML output:")
    print(f"  Shape: {output_ttml_np.shape}")
    print(f"  Min={output_ttml_np.min():.6f}, max={output_ttml_np.max():.6f}, mean={output_ttml_np.mean():.6f}")

    # Compare
    diff = np.abs(output_ref - output_ttml_np)
    print(f"\nComparison:")
    print(f"  Mean abs diff: {diff.mean():.6e}")
    print(f"  Max abs diff: {diff.max():.6e}")

    # Compute PCC
    ref_flat = output_ref.flatten()
    ttml_flat = output_ttml_np.flatten()
    corr = np.corrcoef(ref_flat, ttml_flat)
    pcc = corr[0, 1] if corr.shape == (2, 2) else 0.0
    print(f"  PCC: {pcc:.6f}")

    if pcc > 0.99:
        print(f"\n✅ PASS: Data flow is correct (PCC={pcc:.6f})")
    else:
        print(f"\n❌ FAIL: Data flow has issues (PCC={pcc:.6f})")
        print(f"\nFirst 5 reference values: {ref_flat[:5]}")
        print(f"First 5 TTML values:      {ttml_flat[:5]}")


def test_mask_shape_broadcasting():
    """Test different mask shapes to see which broadcasts correctly."""
    print("\n" + "=" * 80)
    print("TEST: Mask Shape Broadcasting")
    print("=" * 80)

    batch_size = 1
    num_heads = 2
    seq_len = 8
    head_dim = 4

    np.random.seed(42)
    q_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    k_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    v_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

    # Reference: Use full mask shape [batch, num_heads, seq_len, seq_len]
    mask_1d = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)

    # Test different mask shapes
    mask_shapes = [
        ("1D [seq_len]", mask_1d.reshape(seq_len)),
        ("2D [1, seq_len]", mask_1d.reshape(1, seq_len)),
        ("3D [1, 1, seq_len]", mask_1d.reshape(1, 1, seq_len)),
        ("4D [1, 1, 1, seq_len]", mask_1d.reshape(1, 1, 1, seq_len)),
        ("4D [batch, 1, 1, seq_len]", mask_1d.reshape(batch_size, 1, 1, seq_len)),
        (
            "4D [batch, heads, seq_len, seq_len]",
            np.broadcast_to(mask_1d.reshape(1, 1, 1, seq_len), (batch_size, num_heads, seq_len, seq_len)),
        ),
    ]

    # Compute reference with full mask
    scale = 1.0 / np.sqrt(head_dim)
    attn_scores = np.matmul(q_data, k_data.transpose(0, 1, 3, 2)) * scale

    # Apply mask in reference
    mask_ref = np.broadcast_to(mask_1d.reshape(1, 1, 1, seq_len), (batch_size, num_heads, seq_len, seq_len))
    attn_scores_masked = np.where(mask_ref == 1, attn_scores, -1e9)

    attn_scores_exp = np.exp(attn_scores_masked - attn_scores_masked.max(axis=-1, keepdims=True))
    attn_weights = attn_scores_exp / attn_scores_exp.sum(axis=-1, keepdims=True)
    output_ref = np.matmul(attn_weights, v_data)

    print(f"\nReference output (with proper masking):")
    print(f"  Mean={output_ref.mean():.6f}, std={output_ref.std():.6f}")

    # Try each mask shape
    for name, mask_data in mask_shapes:
        try:
            print(f"\n{name}: shape={mask_data.shape}")

            q_ttml = ttml.autograd.Tensor.from_numpy(q_data)
            k_ttml = ttml.autograd.Tensor.from_numpy(k_data)
            v_ttml = ttml.autograd.Tensor.from_numpy(v_data)
            mask_ttml = ttml.autograd.Tensor.from_numpy(mask_data)

            output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)
            output_ttml_np = output_ttml.to_numpy()

            diff = np.abs(output_ref - output_ttml_np)
            ref_flat = output_ref.flatten()
            ttml_flat = output_ttml_np.flatten()
            corr = np.corrcoef(ref_flat, ttml_flat)
            pcc = corr[0, 1] if corr.shape == (2, 2) else 0.0

            status = "✅" if pcc > 0.99 else "❌"
            print(f"  TTML: mean={output_ttml_np.mean():.6f}, std={output_ttml_np.std():.6f}")
            print(f"  PCC: {pcc:.6f} {status}")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")


if __name__ == "__main__":
    test_simple_attention_roundtrip()
    test_mask_shape_broadcasting()
