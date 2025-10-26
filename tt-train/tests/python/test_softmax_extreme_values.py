#!/usr/bin/env python3
"""
Test if TTML softmax handles extreme values correctly (like -1e9 from masking).
"""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml
import torch


def test_softmax_with_masking():
    """Test softmax with extreme negative values from masking."""
    print("=" * 80)
    print("TEST: Softmax with Extreme Values (Masking)")
    print("=" * 80)

    # Create scores with masking pattern
    # Realistic BERT attention scores: small values for real positions, -1e9 for masked
    scores = np.array(
        [
            [0.5, -0.3, 0.2, -1e9, -1e9, -1e9],  # 3 real, 3 masked
            [0.1, 0.8, -0.2, 0.4, -1e9, -1e9],  # 4 real, 2 masked
        ],
        dtype=np.float32,
    ).reshape(1, 1, 2, 6)

    print(f"\nInput scores shape: {scores.shape}")
    print(f"Input scores:\n{scores[0, 0]}")

    # PyTorch reference
    scores_torch = torch.from_numpy(scores)
    softmax_torch = torch.softmax(scores_torch, dim=-1).numpy()

    print(f"\nPyTorch softmax output:")
    print(softmax_torch[0, 0])
    print(f"Row sums (should be 1.0): {softmax_torch[0, 0].sum(axis=-1)}")

    # TTML softmax
    scores_ttml = ttml.autograd.Tensor.from_numpy(scores)

    # Use TTML's softmax directly
    import ttml.metal as metal

    softmax_ttml_tensor = metal.softmax(scores_ttml.get_value(), axis=3)
    softmax_ttml = ttml.autograd.create_tensor(softmax_ttml_tensor).to_numpy()

    print(f"\nTTML softmax output:")
    print(softmax_ttml[0, 0])
    print(f"Row sums (should be 1.0): {softmax_ttml[0, 0].sum(axis=-1)}")

    # Compare
    diff = np.abs(softmax_torch - softmax_ttml)
    print(f"\nComparison:")
    print(f"  Mean abs diff: {diff.mean():.6e}")
    print(f"  Max abs diff: {diff.max():.6e}")

    # Check if masked positions are near zero
    print(f"\nMasked position values (should be ~0):")
    print(f"  PyTorch row 0, positions 3-5: {softmax_torch[0, 0, 0, 3:6]}")
    print(f"  TTML row 0, positions 3-5:    {softmax_ttml[0, 0, 0, 3:6]}")

    if diff.max() < 1e-4:
        print(f"\n✅ Softmax handles extreme values correctly")
    else:
        print(f"\n❌ Softmax has issues with extreme values!")


def test_attention_with_real_bert_scores():
    """Test attention using actual BERT attention score distribution."""
    print("\n" + "=" * 80)
    print("TEST: Attention with Real BERT Score Distribution")
    print("=" * 80)

    # Simulate real BERT attention scores
    # Real positions: values around -2 to +5
    # Masked positions: -1e9
    np.random.seed(42)
    batch_size = 1
    num_heads = 2
    seq_len = 8
    head_dim = 4

    # Create Q, K that will produce realistic score range
    q_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    k_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    v_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

    # Mask: first 4 real, last 4 masked
    mask_data = np.array([[[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32)

    # PyTorch reference
    q_torch = torch.from_numpy(q_data)
    k_torch = torch.from_numpy(k_data)
    v_torch = torch.from_numpy(v_data)
    mask_torch = torch.from_numpy(mask_data)

    scale = 1.0 / np.sqrt(head_dim)
    scores_torch = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale

    print(f"\nAttention scores (before masking):")
    print(f"  Min: {scores_torch.min():.4f}, Max: {scores_torch.max():.4f}, Mean: {scores_torch.mean():.4f}")

    scores_masked_torch = scores_torch.masked_fill(mask_torch == 0, -1e9)

    print(f"Attention scores (after masking):")
    print(f"  Min: {scores_masked_torch.min():.4f}, Max: {scores_masked_torch.max():.4f}")
    print(f"  Mean: {scores_masked_torch.mean():.4f}")  # Will be very negative due to masked positions
    print(f"  First row (should have -1e9 for last 4): {scores_masked_torch[0, 0, 0, :].numpy()}")

    weights_torch = torch.softmax(scores_masked_torch, dim=-1)
    output_torch = torch.matmul(weights_torch, v_torch).numpy()

    print(f"\nAttention weights (after softmax):")
    print(f"  First row sum: {weights_torch[0, 0, 0, :].sum():.6f}")
    print(f"  First row weights: {weights_torch[0, 0, 0, :].numpy()}")

    # TTML
    q_ttml = ttml.autograd.Tensor.from_numpy(q_data)
    k_ttml = ttml.autograd.Tensor.from_numpy(k_data)
    v_ttml = ttml.autograd.Tensor.from_numpy(v_data)
    mask_ttml = ttml.autograd.Tensor.from_numpy(mask_data)

    output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)
    output_ttml_np = output_ttml.to_numpy()

    # Compare
    diff = np.abs(output_torch - output_ttml_np)
    ref_flat = output_torch.flatten()
    ttml_flat = output_ttml_np.flatten()
    corr = np.corrcoef(ref_flat, ttml_flat)
    pcc = corr[0, 1] if corr.shape == (2, 2) else 0.0

    print(f"\nComparison:")
    print(f"  Mean abs diff: {diff.mean():.6e}")
    print(f"  Max abs diff: {diff.max():.6e}")
    print(f"  PCC: {pcc:.6f}")

    if pcc > 0.99:
        print(f"\n✅ Attention works correctly with realistic BERT score distribution")
    else:
        print(f"\n❌ Attention fails with realistic BERT score distribution")


if __name__ == "__main__":
    # test_softmax_with_masking()  # Skip - can't easily test softmax in isolation from Python
    test_attention_with_real_bert_scores()
