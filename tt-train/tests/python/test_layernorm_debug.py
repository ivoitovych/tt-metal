#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test LayerNorm operation to find the attention mechanism bug.

FINAL DISCOVERY: "Block 0 Attention" measurement includes:
1. MultiHeadAttention(input) → PCC >0.99999 ✅
2. add(attention_output, input) → PCC >0.99999 ✅
3. LayerNorm(residual) → ??? ← SUSPECT!

Code from bert_block.cpp:93-101:
    auto attention_output = (*m_attention)(input, attention_mask);
    auto attention_residual = ops::add(attention_output, input);
    attention_residual = (*m_attention_norm)(attention_residual);  # LayerNorm!
    outputs.attention_output = attention_residual;  # This is measured!

If LayerNorm shows low PCC, WE FOUND THE BUG!
"""

import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')

import numpy as np
import pytest
import torch
import torch.nn as nn

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


def test_layernorm_basic():
    """Test basic LayerNorm operation."""
    print("\n" + "=" * 80)
    print("TEST: Basic LayerNorm Operation")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 128
    eps = 1e-12

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}, eps={eps}")

    # Create test input
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    # Create LayerNorm weights
    gamma = torch.ones(embedding_dim, dtype=torch.float32)
    beta = torch.zeros(embedding_dim, dtype=torch.float32)

    # PyTorch reference
    pt_layernorm = nn.LayerNorm(embedding_dim, eps=eps)
    pt_layernorm.weight.data = gamma
    pt_layernorm.bias.data = beta
    pt_result = pt_layernorm(input_pt)

    # TTML
    input_np = input_pt.numpy()
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_gamma = ttml.autograd.Tensor.from_numpy(gamma.numpy())
    ttml_beta = ttml.autograd.Tensor.from_numpy(beta.numpy())

    ttml_result = ttml.ops.layernorm.layernorm(ttml_input, ttml_gamma, ttml_beta, eps=eps)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"PCC:       {pcc:.8f}")
    print(f"Max diff:  {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    print(f"{'='*80}")

    if pcc < 0.95:
        print(f"\n❌ FOUND IT! LayerNorm shows the bug!")
        print(f"   PCC {pcc:.8f} matches the Block 0 Attention error (~0.94)!")
        print(f"   THE BUG IS IN LAYERNORM!")
    elif pcc < 0.999:
        print(f"\n⚠️  LayerNorm shows some degradation (PCC {pcc:.8f})")
    else:
        print(f"\n✅ LayerNorm works correctly")

    # Don't fail yet - want to see all results
    # assert pcc > 0.999, f"LayerNorm failed! PCC={pcc:.8f}"


def test_layernorm_with_loaded_weights():
    """Test LayerNorm with actual BERT weights."""
    print("\n" + "=" * 80)
    print("TEST: LayerNorm with Loaded BERT Weights")
    print("=" * 80)

    from transformers import BertModel

    # Load HuggingFace BERT
    model_name = "prajjwal1/bert-tiny"
    hf_model = BertModel.from_pretrained(model_name)

    # Get Layer 0 attention LayerNorm weights
    layer_0_layernorm = hf_model.encoder.layer[0].attention.output.LayerNorm

    gamma = layer_0_layernorm.weight.data
    beta = layer_0_layernorm.bias.data
    eps = layer_0_layernorm.eps

    embedding_dim = gamma.shape[0]
    batch_size = 1
    seq_len = 4

    print(f"\nConfiguration: E={embedding_dim}, eps={eps}")
    print(f"Gamma range: [{gamma.min():.6f}, {gamma.max():.6f}]")
    print(f"Beta range: [{beta.min():.6f}, {beta.max():.6f}]")

    # Create test input
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    print(f"Input range: [{input_pt.min():.6f}, {input_pt.max():.6f}]")

    # PyTorch reference
    pt_result = layer_0_layernorm(input_pt)

    print(f"PyTorch output range: [{pt_result.min():.6f}, {pt_result.max():.6f}]")

    # TTML
    input_np = input_pt.numpy()
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_gamma = ttml.autograd.Tensor.from_numpy(gamma.numpy())
    ttml_beta = ttml.autograd.Tensor.from_numpy(beta.numpy())

    ttml_result = ttml.ops.layernorm.layernorm(ttml_input, ttml_gamma, ttml_beta, eps=eps)

    ttml_np = ttml_result.to_numpy()
    print(f"TTML output range: [{ttml_np.min():.6f}, {ttml_np.max():.6f}]")

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n{'='*80}")
    print(f"RESULTS (with loaded BERT weights):")
    print(f"{'='*80}")
    print(f"PCC:       {pcc:.8f}")
    print(f"Max diff:  {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    print(f"{'='*80}")

    if pcc < 0.95:
        print(f"\n❌ CRITICAL: LayerNorm with loaded weights shows the bug!")
        print(f"   PCC {pcc:.8f} matches the Block 0 Attention error!")
        print(f"   ROOT CAUSE FOUND: LayerNorm implementation or weight loading!")
    elif pcc < 0.999:
        print(f"\n⚠️  Some degradation (PCC {pcc:.8f})")
    else:
        print(f"\n✅ LayerNorm with loaded weights works correctly")


def test_full_block_simulation():
    """
    Simulate the exact computation measured as 'Block 0 Attention'.

    This is:
    1. MultiHeadAttention(input)
    2. add(attention_output, input)
    3. LayerNorm(residual)
    """
    print("\n" + "=" * 80)
    print("TEST: Full Block Simulation (What 'Block 0 Attention' Measures)")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 128
    eps = 1e-12

    print(f"\nSimulating: LayerNorm(add(attention_output, embeddings))")

    # Create test tensors
    torch.manual_seed(42)
    embeddings_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    attention_output_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    gamma = torch.ones(embedding_dim, dtype=torch.float32)
    beta = torch.zeros(embedding_dim, dtype=torch.float32)

    # PyTorch: residual = add then layernorm
    residual_pt = attention_output_pt + embeddings_pt
    pt_layernorm = nn.LayerNorm(embedding_dim, eps=eps)
    pt_layernorm.weight.data = gamma
    pt_layernorm.bias.data = beta
    pt_result = pt_layernorm(residual_pt)

    # TTML: same pattern
    embeddings_ttml = ttml.autograd.Tensor.from_numpy(embeddings_pt.numpy())
    attention_output_ttml = ttml.autograd.Tensor.from_numpy(attention_output_pt.numpy())

    # Add
    residual_ttml = ttml.ops.binary.add(attention_output_ttml, embeddings_ttml)

    # LayerNorm
    ttml_gamma = ttml.autograd.Tensor.from_numpy(gamma.numpy())
    ttml_beta = ttml.autograd.Tensor.from_numpy(beta.numpy())
    ttml_result = ttml.ops.layernorm.layernorm(residual_ttml, ttml_gamma, ttml_beta, eps=eps)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS (This is 'Block 0 Attention'):")
    print(f"{'='*80}")
    print(f"PCC:       {pcc:.8f}")
    print(f"Max diff:  {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    print(f"{'='*80}")

    if pcc < 0.95:
        print(f"\n🎯 BUG CONFIRMED!")
        print(f"   This PCC {pcc:.8f} matches 'Block 0 Attention' (~0.94)!")
        print(f"   LayerNorm is the root cause!")
    elif pcc < 0.999:
        print(f"\n⚠️  Some degradation")
    else:
        print(f"\n✅ Full block pattern works correctly")


def test_analysis_summary():
    """Print final analysis."""
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)
    print("'Block 0 Attention' in layer-by-layer test measures:")
    print("  1. attention_output = MultiHeadAttention(input)")
    print("  2. residual = add(attention_output, input)")
    print("  3. result = LayerNorm(residual)  ← FINAL SUSPECT!")
    print("")
    print("We've verified ALL operations are perfect (PCC >0.99999):")
    print("  ✅ Embeddings")
    print("  ✅ All attention operations (heads_creation, SDPA, heads_fusion)")
    print("  ✅ Linear layers (random and loaded weights)")
    print("  ✅ ADD operation")
    print("")
    print("If LayerNorm shows PCC < 0.95:")
    print("  🎯 ROOT CAUSE FOUND: LayerNorm implementation bug!")
    print("  🎯 This is the ONLY remaining operation that could cause PCC 0.94!")
    print("=" * 80)
