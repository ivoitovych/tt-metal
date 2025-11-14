#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test to verify the scaling order hypothesis for attention mechanism bug.

Root Cause Hypothesis:
    The current implementation pre-scales Q before matmul: (Q * scale) @ K^T
    This may lose precision in bfloat16 compared to: (Q @ K^T) * scale

Expected Results:
    - Current implementation: PCC ~0.94-0.97 (3-6% error)
    - Post-matmul scaling: PCC >0.999 (if hypothesis is correct)
"""

import pytest
import torch
import numpy as np
from scipy.stats import pearsonr


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient between two tensors."""
    tensor1_flat = tensor1.detach().cpu().numpy().flatten()
    tensor2_flat = tensor2.detach().cpu().numpy().flatten()

    # Remove any NaN/Inf values
    mask = np.isfinite(tensor1_flat) & np.isfinite(tensor2_flat)
    tensor1_flat = tensor1_flat[mask]
    tensor2_flat = tensor2_flat[mask]

    if len(tensor1_flat) == 0:
        return 0.0

    correlation, _ = pearsonr(tensor1_flat, tensor2_flat)
    return correlation


def scaled_dot_product_attention_reference(query, key, value, scale):
    """
    Reference implementation using PyTorch (gold standard).
    Computes: softmax((Q @ K^T) * scale) @ V
    """
    # Standard order: compute QK^T first, then scale
    qk = torch.matmul(query, key.transpose(-2, -1))
    qk_scaled = qk * scale
    attention_weights = torch.softmax(qk_scaled, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output


def scaled_dot_product_attention_pre_scale(query, key, value, scale):
    """
    Current TTML implementation order (suspected bug).
    Computes: softmax((Q * scale) @ K^T) @ V
    """
    # Pre-scale Q before matmul (current implementation)
    q_scaled = query * scale
    qk = torch.matmul(q_scaled, key.transpose(-2, -1))
    attention_weights = torch.softmax(qk, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output


def test_scaling_order_hypothesis():
    """
    Test different scaling orders to identify numerical precision issues.

    This test simulates the BERT attention mechanism with different scaling orders:
    1. Reference (PyTorch standard): (Q @ K^T) * scale
    2. Pre-scaled (current TTML): (Q * scale) @ K^T

    We test with:
    - bfloat16 dtype (what TTML uses)
    - Different embedding dimensions (64, 128, 256, 512, 768)
    - Batch size and sequence length matching BERT
    """
    print("\n" + "=" * 80)
    print("ATTENTION SCALING ORDER TEST")
    print("=" * 80)

    batch_size = 2
    num_heads = 12
    seq_len = 32
    embedding_dims = [64, 128, 256, 512, 768]  # Different head dimensions

    results = []

    for embedding_dim in embedding_dims:
        head_dim = embedding_dim // num_heads
        scale = 1.0 / np.sqrt(embedding_dim / num_heads)

        print(f"\n--- Testing embedding_dim={embedding_dim}, head_dim={head_dim}, scale={scale:.6f} ---")

        # Generate random input tensors (float32 for high precision)
        torch.manual_seed(42)
        query_fp32 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key_fp32 = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_fp32 = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Convert to bfloat16 (what TTML uses)
        query_bf16 = query_fp32.to(torch.bfloat16)
        key_bf16 = key_fp32.to(torch.bfloat16)
        value_bf16 = value_fp32.to(torch.bfloat16)

        # Compute reference output (float32 for gold standard)
        output_reference_fp32 = scaled_dot_product_attention_reference(query_fp32, key_fp32, value_fp32, scale)

        # Test 1: Standard order in bfloat16 (post-matmul scaling)
        output_standard_bf16 = scaled_dot_product_attention_reference(query_bf16, key_bf16, value_bf16, scale).to(
            torch.float32
        )

        # Test 2: Pre-scaled order in bfloat16 (current TTML implementation)
        output_prescale_bf16 = scaled_dot_product_attention_pre_scale(query_bf16, key_bf16, value_bf16, scale).to(
            torch.float32
        )

        # Compute PCCs
        pcc_standard = compute_pcc(output_reference_fp32, output_standard_bf16)
        pcc_prescale = compute_pcc(output_reference_fp32, output_prescale_bf16)

        print(f"PCC (standard order):  {pcc_standard:.6f}")
        print(f"PCC (pre-scale order): {pcc_prescale:.6f}")
        print(f"Difference:            {pcc_standard - pcc_prescale:.6f}")

        results.append(
            {
                "embedding_dim": embedding_dim,
                "head_dim": head_dim,
                "scale": scale,
                "pcc_standard": pcc_standard,
                "pcc_prescale": pcc_prescale,
                "difference": pcc_standard - pcc_prescale,
            }
        )

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Embedding':>10} {'Head':>6} {'Scale':>8} {'Standard':>10} {'Pre-scale':>10} {'Diff':>8}")
    print(f"{'Dim':>10} {'Dim':>6} {'Factor':>8} {'PCC':>10} {'PCC':>10} {'':>8}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['embedding_dim']:>10} {r['head_dim']:>6} {r['scale']:>8.6f} "
            f"{r['pcc_standard']:>10.6f} {r['pcc_prescale']:>10.6f} "
            f"{r['difference']:>8.6f}"
        )

    print("=" * 80)

    # Analysis
    print("\nANALYSIS:")
    avg_pcc_standard = np.mean([r["pcc_standard"] for r in results])
    avg_pcc_prescale = np.mean([r["pcc_prescale"] for r in results])
    avg_difference = np.mean([r["difference"] for r in results])

    print(f"Average PCC (standard order):  {avg_pcc_standard:.6f}")
    print(f"Average PCC (pre-scale order): {avg_pcc_prescale:.6f}")
    print(f"Average difference:            {avg_difference:.6f}")

    if avg_difference > 0.001:
        print("\n🔴 HYPOTHESIS CONFIRMED: Pre-scaling causes significant precision loss!")
        print("   Recommendation: Change implementation to post-matmul scaling")
    else:
        print("\n✅ Hypothesis NOT confirmed: Scaling order doesn't significantly affect precision")
        print("   The bug must be elsewhere in the attention mechanism")

    print("=" * 80)

    # Test assertion: If hypothesis is correct, we should see a significant difference
    # This test is for investigation purposes, so we don't fail on assertion
    # Instead, we document the findings


def test_detailed_intermediate_analysis():
    """
    Detailed analysis of intermediate values to pinpoint where precision is lost.
    """
    print("\n" + "=" * 80)
    print("DETAILED INTERMEDIATE ANALYSIS")
    print("=" * 80)

    batch_size = 1
    num_heads = 12
    seq_len = 32
    embedding_dim = 768  # BERT-base
    head_dim = embedding_dim // num_heads
    scale = 1.0 / np.sqrt(head_dim)

    print(f"\nBERT-base configuration:")
    print(f"  embedding_dim={embedding_dim}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"  scale={scale:.6f} (1/√{head_dim})")

    torch.manual_seed(42)
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Reference computation (standard order)
    print("\n--- Standard Order: (Q @ K^T) * scale ---")
    qk_ref = torch.matmul(query, key.transpose(-2, -1))
    print(f"Q @ K^T:      min={qk_ref.min():.6f}, max={qk_ref.max():.6f}, mean={qk_ref.mean():.6f}")

    qk_scaled_ref = qk_ref * scale
    print(
        f"After scale:  min={qk_scaled_ref.min():.6f}, max={qk_scaled_ref.max():.6f}, mean={qk_scaled_ref.mean():.6f}"
    )

    attn_weights_ref = torch.softmax(qk_scaled_ref, dim=-1)
    print(
        f"After softmax: min={attn_weights_ref.min():.6f}, max={attn_weights_ref.max():.6f}, mean={attn_weights_ref.mean():.6f}"
    )

    # Pre-scaled computation (current TTML)
    print("\n--- Pre-scaled Order: (Q * scale) @ K^T ---")
    q_scaled = query * scale
    print(f"Q * scale:    min={q_scaled.min():.6f}, max={q_scaled.max():.6f}, mean={q_scaled.mean():.6f}")

    qk_pre = torch.matmul(q_scaled, key.transpose(-2, -1))
    print(f"After Q@K^T:  min={qk_pre.min():.6f}, max={qk_pre.max():.6f}, mean={qk_pre.mean():.6f}")

    attn_weights_pre = torch.softmax(qk_pre, dim=-1)
    print(
        f"After softmax: min={attn_weights_pre.min():.6f}, max={attn_weights_pre.max():.6f}, mean={attn_weights_pre.mean():.6f}"
    )

    # Compare attention weights
    pcc_attn_weights = compute_pcc(attn_weights_ref, attn_weights_pre)
    print(f"\n🔍 PCC of attention weights: {pcc_attn_weights:.6f}")

    # Final outputs
    output_ref = torch.matmul(attn_weights_ref, value)
    output_pre = torch.matmul(attn_weights_pre, value)

    pcc_output = compute_pcc(output_ref, output_pre)
    print(f"🔍 PCC of final output:      {pcc_output:.6f}")

    print("=" * 80)


if __name__ == "__main__":
    test_scaling_order_hypothesis()
    test_detailed_intermediate_analysis()
