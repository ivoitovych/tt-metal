#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test actual BERT attention operations to find where data corruption occurs.

Tests the real operations used in BERT attention:
1. heads_creation (splits embeddings into multiple heads)
2. scaled_dot_product_attention (the full attention operation)
3. heads_fusion (concatenates heads back together)

This tests the actual code paths that show PCC 0.94 in Block 0.
"""

import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')

import numpy as np
import pytest
import torch

import _ttml as ttml


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    if isinstance(tensor1, torch.Tensor):
        tensor1 = tensor1.detach().cpu().numpy()
    if isinstance(tensor2, torch.Tensor):
        tensor2 = tensor2.detach().cpu().numpy()
    if hasattr(tensor1, "to_numpy"):
        tensor1 = tensor1.to_numpy()
    if hasattr(tensor2, "to_numpy"):
        tensor2 = tensor2.to_numpy()

    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()

    mean1 = np.mean(tensor1_flat)
    mean2 = np.mean(tensor2_flat)
    numerator = np.sum((tensor1_flat - mean1) * (tensor2_flat - mean2))
    denominator = np.sqrt(np.sum((tensor1_flat - mean1) ** 2) * np.sum((tensor2_flat - mean2) ** 2))

    return numerator / denominator if denominator > 0 else (1.0 if numerator == 0 else 0.0)


def test_heads_creation():
    """Test the heads_creation operation used in multi-head attention."""
    print("\n" + "=" * 80)
    print("TEST: heads_creation")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 8
    num_heads = 2
    head_dim = embedding_dim // num_heads

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}, H={num_heads}, D={head_dim}")

    # Create input [B, 1, S, E]
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    input_np = input_pt.numpy()

    # PyTorch reference: [B,1,S,E] -> [B,S,E] -> [B,S,H,D] -> [B,H,S,D]
    pt_step1 = input_pt.squeeze(1)  # [B, S, E]
    pt_step2 = pt_step1.reshape(batch_size, seq_len, num_heads, head_dim)  # [B, S, H, D]
    pt_result = pt_step2.transpose(1, 2)  # [B, H, S, D]

    # TTML heads_creation
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_result = ttml.ops.multi_head_utils.heads_creation(ttml_input, num_heads)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ heads_creation PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if pcc < 0.9999:
        print(f"\n❌ CORRUPTION FOUND IN heads_creation!")
        print(f"   This operation splits [B,1,S,E] into [B,H,S,D]")
        print(f"   PCC {pcc:.8f} indicates numerical precision loss")

    assert pcc > 0.999, f"heads_creation failed! PCC={pcc:.8f}"


def test_scaled_dot_product_attention():
    """Test the full scaled_dot_product_attention operation."""
    print("\n" + "=" * 80)
    print("TEST: scaled_dot_product_attention")
    print("=" * 80)

    batch_size = 1
    num_heads = 2
    seq_len = 4
    head_dim = 4

    print(f"\nConfiguration: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")

    # Create Q, K, V tensors [B, H, S, D]
    torch.manual_seed(42)
    q_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # PyTorch reference
    scale = 1.0 / np.sqrt(head_dim)
    qk = torch.matmul(q_pt, k_pt.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(qk, dim=-1)
    pt_result = torch.matmul(attn_weights, v_pt)

    # TTML scaled_dot_product_attention
    ttml_q = ttml.autograd.Tensor.from_numpy(q_pt.numpy())
    ttml_k = ttml.autograd.Tensor.from_numpy(k_pt.numpy())
    ttml_v = ttml.autograd.Tensor.from_numpy(v_pt.numpy())

    ttml_result = ttml.ops.multi_head_utils.scaled_dot_product_attention(ttml_q, ttml_k, ttml_v, mask=None)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ scaled_dot_product_attention PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if pcc < 0.999:
        print(f"\n❌ CORRUPTION FOUND IN scaled_dot_product_attention!")
        print(f"   This is the core attention operation")
        print(f"   PCC {pcc:.8f} indicates significant numerical error")
        print(f"   This matches the Block 0 Attention PCC ~0.94 we observed!")

    assert pcc > 0.95, f"scaled_dot_product_attention failed! PCC={pcc:.8f}"


def test_heads_fusion():
    """Test the heads_fusion operation."""
    print("\n" + "=" * 80)
    print("TEST: heads_fusion")
    print("=" * 80)

    batch_size = 1
    num_heads = 2
    seq_len = 4
    head_dim = 4
    embedding_dim = num_heads * head_dim

    print(f"\nConfiguration: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")

    # Create input [B, H, S, D]
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    input_np = input_pt.numpy()

    # PyTorch reference: [B,H,S,D] -> [B,S,H,D] -> [B,S,E] -> [B,1,S,E]
    pt_step1 = input_pt.transpose(1, 2)  # [B, S, H, D]
    pt_step2 = pt_step1.reshape(batch_size, seq_len, embedding_dim)  # [B, S, E]
    pt_result = pt_step2.unsqueeze(1)  # [B, 1, S, E]

    # TTML heads_fusion
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_result = ttml.ops.multi_head_utils.heads_fusion(ttml_input)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ heads_fusion PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if pcc < 0.9999:
        print(f"\n❌ CORRUPTION FOUND IN heads_fusion!")
        print(f"   This operation concatenates [B,H,S,D] back to [B,1,S,E]")
        print(f"   PCC {pcc:.8f} indicates numerical precision loss")

    assert pcc > 0.999, f"heads_fusion failed! PCC={pcc:.8f}"


def test_full_attention_pipeline():
    """Test the complete attention pipeline: heads_creation -> attention -> heads_fusion."""
    print("\n" + "=" * 80)
    print("TEST: Full Attention Pipeline")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 8
    num_heads = 2
    head_dim = embedding_dim // num_heads

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}, H={num_heads}, D={head_dim}")

    # Create QKV inputs [B, 1, S, E]
    torch.manual_seed(42)
    q_input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    k_input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    v_input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    # PyTorch reference pipeline
    q_heads_pt = q_input_pt.squeeze(1).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k_heads_pt = k_input_pt.squeeze(1).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v_heads_pt = v_input_pt.squeeze(1).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    scale = 1.0 / np.sqrt(head_dim)
    qk = torch.matmul(q_heads_pt, k_heads_pt.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(qk, dim=-1)
    attn_output = torch.matmul(attn_weights, v_heads_pt)

    pt_result = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embedding_dim).unsqueeze(1)

    # TTML pipeline
    ttml_q_input = ttml.autograd.Tensor.from_numpy(q_input_pt.numpy())
    ttml_k_input = ttml.autograd.Tensor.from_numpy(k_input_pt.numpy())
    ttml_v_input = ttml.autograd.Tensor.from_numpy(v_input_pt.numpy())

    ttml_q_heads = ttml.ops.multi_head_utils.heads_creation(ttml_q_input, num_heads)
    ttml_k_heads = ttml.ops.multi_head_utils.heads_creation(ttml_k_input, num_heads)
    ttml_v_heads = ttml.ops.multi_head_utils.heads_creation(ttml_v_input, num_heads)

    ttml_attn_output = ttml.ops.multi_head_utils.scaled_dot_product_attention(
        ttml_q_heads, ttml_k_heads, ttml_v_heads, mask=None
    )

    ttml_result = ttml.ops.multi_head_utils.heads_fusion(ttml_attn_output)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ Full pipeline PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if pcc < 0.95:
        print(f"\n❌ FULL PIPELINE SHOWS CORRUPTION!")
        print(f"   This matches the Block 0 Attention PCC we observed (~0.94)")
        print(f"   The error compounds through: heads_creation -> attention -> heads_fusion")
    elif pcc < 0.999:
        print(f"\n⚠️  Pipeline shows some degradation (PCC {pcc:.8f})")
        print(f"   Individual operations may have small errors that compound")

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print("If full pipeline PCC < 0.95, the corruption matches Block 0 Attention bug!")
    print("Run individual operation tests to identify which operation introduces error.")
    print("=" * 80)
