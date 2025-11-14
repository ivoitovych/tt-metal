#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Direct comparison of TTML SDPA vs PyTorch SDPA with identical inputs.

This is the simplest possible test to confirm the bug.
"""

import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')

import numpy as np
import pytest
import torch
import torch.nn.functional as F

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


def test_sdpa_with_bert_shapes_and_values():
    """
    Test SDPA with BERT-like data values (not just shapes).

    Use actual values from BERT forward pass to trigger the bug.
    """
    print("\n" + "=" * 80)
    print("SDPA DIRECT COMPARISON TEST")
    print("=" * 80)

    from transformers import BertModel

    # Load BERT to get realistic Q, K, V values
    model_name = "prajjwal1/bert-tiny"
    hf_model = BertModel.from_pretrained(model_name)
    hf_model.eval()

    hf_attention = hf_model.encoder.layer[0].attention.self
    config = hf_model.config

    embedding_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = embedding_dim // num_heads
    batch_size = 1
    seq_len = 4

    print(f"\nConfiguration: E={embedding_dim}, H={num_heads}, head_dim={head_dim}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}\n")

    # Create BERT-like input
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, seq_len, embedding_dim, dtype=torch.float32)

    # Get Q, K, V from real BERT
    with torch.no_grad():
        q_hf = hf_attention.query(input_pt)  # (B, S, E)
        k_hf = hf_attention.key(input_pt)
        v_hf = hf_attention.value(input_pt)

    # Reshape to heads: (B, S, E) -> (B, H, S, head_dim) -> (B, H, 1, S, head_dim)
    def reshape_to_heads(x):
        B, S, E = x.shape
        return (
            x.view(B, S, num_heads, head_dim)
            .transpose(1, 2)  # (B, H, S, head_dim)
            .unsqueeze(2)  # (B, H, 1, S, head_dim)
        )

    q_heads_pt = reshape_to_heads(q_hf)
    k_heads_pt = reshape_to_heads(k_hf)
    v_heads_pt = reshape_to_heads(v_hf)

    print(f"Q range: [{q_heads_pt.min():.4f}, {q_heads_pt.max():.4f}]")
    print(f"K range: [{k_heads_pt.min():.4f}, {k_heads_pt.max():.4f}]")
    print(f"V range: [{v_heads_pt.min():.4f}, {v_heads_pt.max():.4f}]")

    # PyTorch SDPA
    scale = 1.0 / np.sqrt(head_dim)
    qk_pt = torch.matmul(q_heads_pt, k_heads_pt.transpose(-2, -1)) * scale
    attn_weights_pt = F.softmax(qk_pt, dim=-1)
    output_pt = torch.matmul(attn_weights_pt, v_heads_pt)

    print(f"\nPyTorch SDPA output range: [{output_pt.min():.4f}, {output_pt.max():.4f}]")

    # TTML SDPA
    q_ttml = ttml.autograd.Tensor.from_numpy(q_heads_pt.numpy())
    k_ttml = ttml.autograd.Tensor.from_numpy(k_heads_pt.numpy())
    v_ttml = ttml.autograd.Tensor.from_numpy(v_heads_pt.numpy())

    # No mask (zeros = no masking)
    mask_np = np.zeros((batch_size, 1, 1, seq_len), dtype=np.float32)
    mask_ttml = ttml.autograd.Tensor.from_numpy(mask_np)

    output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask_ttml)

    output_ttml_np = output_ttml.to_numpy()
    print(f"TTML SDPA output range: [{output_ttml_np.min():.4f}, {output_ttml_np.max():.4f}]")

    # Compare
    pcc = compute_pcc(output_pt, output_ttml)
    max_diff = np.abs(output_pt.numpy() - output_ttml_np).max()
    mean_diff = np.abs(output_pt.numpy() - output_ttml_np).mean()

    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"PCC:       {pcc:.8f}")
    print(f"Max diff:  {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    print(f"{'='*80}")

    if pcc < 0.95:
        print(f"\n❌ BUG CONFIRMED! SDPA with BERT data shows PCC {pcc:.8f}")
        print(f"   Output range mismatch:")
        print(f"   Expected: [{output_pt.min():.4f}, {output_pt.max():.4f}]")
        print(f"   Actual:   [{output_ttml_np.min():.4f}, {output_ttml_np.max():.4f}]")
    elif pcc < 0.999:
        print(f"\n⚠️  Some degradation (PCC {pcc:.8f})")
    else:
        print(f"\n✅ SDPA works correctly with BERT data")


def test_sdpa_with_random_data():
    """
    Test SDPA with random data (control test - should pass).
    """
    print("\n" + "=" * 80)
    print("SDPA WITH RANDOM DATA (CONTROL TEST)")
    print("=" * 80)

    batch_size = 1
    num_heads = 2
    seq_len = 4
    head_dim = 64

    # Random data
    np.random.seed(42)
    q_np = np.random.randn(batch_size, num_heads, 1, seq_len, head_dim).astype(np.float32)
    k_np = np.random.randn(batch_size, num_heads, 1, seq_len, head_dim).astype(np.float32)
    v_np = np.random.randn(batch_size, num_heads, 1, seq_len, head_dim).astype(np.float32)

    # PyTorch
    q_pt = torch.from_numpy(q_np)
    k_pt = torch.from_numpy(k_np)
    v_pt = torch.from_numpy(v_np)

    scale = 1.0 / np.sqrt(head_dim)
    qk_pt = torch.matmul(q_pt, k_pt.transpose(-2, -1)) * scale
    attn_weights_pt = F.softmax(qk_pt, dim=-1)
    output_pt = torch.matmul(attn_weights_pt, v_pt)

    # TTML
    q_ttml = ttml.autograd.Tensor.from_numpy(q_np)
    k_ttml = ttml.autograd.Tensor.from_numpy(k_np)
    v_ttml = ttml.autograd.Tensor.from_numpy(v_np)

    mask_np = np.zeros((batch_size, 1, 1, seq_len), dtype=np.float32)
    mask_ttml = ttml.autograd.Tensor.from_numpy(mask_np)

    output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask_ttml)

    # Compare
    pcc = compute_pcc(output_pt, output_ttml)

    print(f"\nRandom data SDPA PCC: {pcc:.8f}")

    if pcc > 0.999:
        print(f"✅ SDPA works with random data (as expected)")
    else:
        print(f"❌ SDPA fails even with random data! PCC {pcc:.8f}")
