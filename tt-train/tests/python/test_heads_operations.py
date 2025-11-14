#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test heads_creation and heads_fusion operations.

These operations transform tensors between different shapes for multi-head attention.
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


def test_heads_creation_from_qkv():
    """Test heads_creation with concatenated QKV input."""
    print("\n" + "=" * 80)
    print("TEST: heads_creation (from QKV concatenated)")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 8
    num_heads = 2
    head_dim = embedding_dim // num_heads

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}, H={num_heads}, D={head_dim}")

    # Create concatenated QKV [B, 1, S, E*3]
    torch.manual_seed(42)
    qkv_pt = torch.randn(batch_size, 1, seq_len, embedding_dim * 3, dtype=torch.float32)
    qkv_np = qkv_pt.numpy()

    # PyTorch reference: split Q, K, V and reshape
    q_pt, k_pt, v_pt = torch.chunk(qkv_pt, 3, dim=-1)  # Each [B, 1, S, E]

    # [B, 1, S, E] -> [B, S, H, D] -> [B, H, S, D]
    q_ref = q_pt.squeeze(1).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k_ref = k_pt.squeeze(1).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v_ref = v_pt.squeeze(1).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # TTML heads_creation
    ttml_qkv = ttml.autograd.Tensor.from_numpy(qkv_np)
    ttml_q, ttml_k, ttml_v = ttml.ops.multi_head_utils.heads_creation(ttml_qkv, num_heads)

    # Compare each output
    pcc_q = compute_pcc(q_ref, ttml_q)
    pcc_k = compute_pcc(k_ref, ttml_k)
    pcc_v = compute_pcc(v_ref, ttml_v)

    print(f"\n✓ heads_creation Q PCC: {pcc_q:.8f}")
    print(f"✓ heads_creation K PCC: {pcc_k:.8f}")
    print(f"✓ heads_creation V PCC: {pcc_v:.8f}")

    min_pcc = min(pcc_q, pcc_k, pcc_v)

    if min_pcc < 0.999:
        print(f"\n❌ CORRUPTION FOUND IN heads_creation!")
        print(f"   Min PCC: {min_pcc:.8f}")
        print(f"   This operation splits and reshapes QKV tensor")

    assert min_pcc > 0.999, f"heads_creation failed! Min PCC={min_pcc:.8f}"


def test_heads_fusion():
    """Test heads_fusion operation."""
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

    if pcc < 0.999:
        print(f"\n❌ CORRUPTION FOUND IN heads_fusion!")
        print(f"   This operation concatenates heads [B,H,S,D] -> [B,1,S,E]")
        print(f"   PCC {pcc:.8f} indicates numerical precision loss")

    assert pcc > 0.999, f"heads_fusion failed! PCC={pcc:.8f}"
