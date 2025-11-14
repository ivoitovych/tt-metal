#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test linear layer operations to identify the source of attention mechanism bug.

Since all core attention operations achieve PCC >0.99999, the bug must be in
the LINEAR LAYERS that surround the attention:
1. QKV linear projection (E → 3E) before attention
2. Output linear projection (E → E) after attention

This test isolates these operations to find which one introduces the ~0.94 PCC error.
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


def test_linear_layer_basic():
    """Test basic linear layer operation with random weights."""
    print("\n" + "=" * 80)
    print("TEST: Basic Linear Layer (Random Weights)")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    in_features = 8
    out_features = 16

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, in={in_features}, out={out_features}")

    # Create input [B, 1, S, E]
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, in_features, dtype=torch.float32)
    input_np = input_pt.numpy()

    # Create random weights and bias
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    bias = torch.randn(out_features, dtype=torch.float32)

    # PyTorch reference
    pt_linear = nn.Linear(in_features, out_features, bias=True)
    pt_linear.weight.data = weight
    pt_linear.bias.data = bias
    pt_result = pt_linear(input_pt)

    # TTML linear layer - construct with weight and bias tensors
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_weight = ttml.autograd.Tensor.from_numpy(weight.numpy())
    ttml_bias = ttml.autograd.Tensor.from_numpy(bias.numpy())
    ttml_linear = ttml.modules.LinearLayer(ttml_weight, ttml_bias)

    ttml_result = ttml_linear(ttml_input)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ Basic Linear Layer PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if pcc < 0.999:
        print(f"\n❌ CORRUPTION FOUND IN BASIC LINEAR LAYER!")
        print(f"   PCC {pcc:.8f} indicates the Linear module itself has a bug")
        print(f"   This is the root cause of the attention mechanism error!")

    assert pcc > 0.999, f"Basic linear layer failed! PCC={pcc:.8f}"


def test_linear_layer_no_bias():
    """Test linear layer without bias."""
    print("\n" + "=" * 80)
    print("TEST: Linear Layer Without Bias")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    in_features = 8
    out_features = 16

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, in={in_features}, out={out_features}")

    # Create input
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, in_features, dtype=torch.float32)
    input_np = input_pt.numpy()

    # Create random weights
    weight = torch.randn(out_features, in_features, dtype=torch.float32)

    # PyTorch reference
    pt_linear = nn.Linear(in_features, out_features, bias=False)
    pt_linear.weight.data = weight
    pt_result = pt_linear(input_pt)

    # TTML linear layer - construct with weight only (no bias)
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_weight = ttml.autograd.Tensor.from_numpy(weight.numpy())
    ttml_linear = ttml.modules.LinearLayer(ttml_weight, has_bias=False)
    ttml_result = ttml_linear(ttml_input)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    print(f"\n✓ Linear Layer (no bias) PCC: {pcc:.8f}")

    if pcc < 0.999:
        print(f"\n❌ CORRUPTION FOUND IN LINEAR LAYER (no bias)!")
        print(f"   PCC {pcc:.8f} indicates bug in weight-only operation")

    assert pcc > 0.999, f"Linear layer (no bias) failed! PCC={pcc:.8f}"


def test_qkv_projection_size():
    """Test linear layer with QKV projection dimensions."""
    print("\n" + "=" * 80)
    print("TEST: QKV Projection Size (E → 3E)")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 128
    qkv_dim = embedding_dim * 3  # 384

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}, QKV={qkv_dim}")

    # Create input [B, 1, S, E]
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    input_np = input_pt.numpy()

    # Create random weights and bias
    weight = torch.randn(qkv_dim, embedding_dim, dtype=torch.float32)
    bias = torch.randn(qkv_dim, dtype=torch.float32)

    # PyTorch reference
    pt_linear = nn.Linear(embedding_dim, qkv_dim, bias=True)
    pt_linear.weight.data = weight
    pt_linear.bias.data = bias
    pt_result = pt_linear(input_pt)

    # TTML linear layer
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_weight = ttml.autograd.Tensor.from_numpy(weight.numpy())
    ttml_bias = ttml.autograd.Tensor.from_numpy(bias.numpy())
    ttml_linear = ttml.modules.LinearLayer(ttml_weight, ttml_bias)
    ttml_result = ttml_linear(ttml_input)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ QKV Projection PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")
    print(f"  Output shape: {ttml_np.shape}")

    if pcc < 0.95:
        print(f"\n❌ CRITICAL: QKV PROJECTION SHOWS THE BUG!")
        print(f"   PCC {pcc:.8f} matches the Block 0 Attention error (~0.94)!")
        print(f"   This is likely the root cause of the attention mechanism bug!")
    elif pcc < 0.999:
        print(f"\n⚠️  QKV projection shows some degradation (PCC {pcc:.8f})")

    # More lenient threshold for this test since we're looking for the ~0.94 bug
    assert pcc > 0.90, f"QKV projection failed! PCC={pcc:.8f}"


def test_output_projection_size():
    """Test linear layer with output projection dimensions."""
    print("\n" + "=" * 80)
    print("TEST: Output Projection Size (E → E)")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 128

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}")

    # Create input [B, 1, S, E]
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    input_np = input_pt.numpy()

    # Create random weights and bias
    weight = torch.randn(embedding_dim, embedding_dim, dtype=torch.float32)
    bias = torch.randn(embedding_dim, dtype=torch.float32)

    # PyTorch reference
    pt_linear = nn.Linear(embedding_dim, embedding_dim, bias=True)
    pt_linear.weight.data = weight
    pt_linear.bias.data = bias
    pt_result = pt_linear(input_pt)

    # TTML linear layer
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_weight = ttml.autograd.Tensor.from_numpy(weight.numpy())
    ttml_bias = ttml.autograd.Tensor.from_numpy(bias.numpy())
    ttml_linear = ttml.modules.LinearLayer(ttml_weight, ttml_bias)
    ttml_result = ttml_linear(ttml_input)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ Output Projection PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if pcc < 0.95:
        print(f"\n❌ CRITICAL: OUTPUT PROJECTION SHOWS THE BUG!")
        print(f"   PCC {pcc:.8f} matches the Block 0 Attention error (~0.94)!")
        print(f"   This is likely the root cause of the attention mechanism bug!")
    elif pcc < 0.999:
        print(f"\n⚠️  Output projection shows some degradation (PCC {pcc:.8f})")

    # More lenient threshold for this test
    assert pcc > 0.90, f"Output projection failed! PCC={pcc:.8f}"


def test_bert_base_dimensions():
    """Test with bert-base-uncased dimensions (768)."""
    print("\n" + "=" * 80)
    print("TEST: BERT-Base Dimensions (E=768, QKV=2304)")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 768
    qkv_dim = embedding_dim * 3

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}, QKV={qkv_dim}")

    # Create input
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    input_np = input_pt.numpy()

    # Create random weights and bias
    weight = torch.randn(qkv_dim, embedding_dim, dtype=torch.float32)
    bias = torch.randn(qkv_dim, dtype=torch.float32)

    # PyTorch reference
    pt_linear = nn.Linear(embedding_dim, qkv_dim, bias=True)
    pt_linear.weight.data = weight
    pt_linear.bias.data = bias
    pt_result = pt_linear(input_pt)

    # TTML linear layer
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_weight = ttml.autograd.Tensor.from_numpy(weight.numpy())
    ttml_bias = ttml.autograd.Tensor.from_numpy(bias.numpy())
    ttml_linear = ttml.modules.LinearLayer(ttml_weight, ttml_bias)
    ttml_result = ttml_linear(ttml_input)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ BERT-Base QKV Projection PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if pcc < 0.95:
        print(f"\n❌ CRITICAL: BERT-BASE DIMENSIONS SHOW THE BUG!")
        print(f"   PCC {pcc:.8f} matches the Block 0 Attention error!")
        print(f"   Bug manifests with production model dimensions!")

    assert pcc > 0.90, f"BERT-Base dimensions failed! PCC={pcc:.8f}"


def test_analysis_summary():
    """Print analysis summary after all tests."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print("Linear layer tests help identify if the bug is in:")
    print("1. Basic Linear module implementation")
    print("2. Bias handling")
    print("3. Specific matrix dimensions (QKV or output projection)")
    print("")
    print("Expected results:")
    print("- If basic tests fail (PCC < 0.999): Linear module has fundamental bug")
    print("- If QKV/output tests fail (PCC < 0.95): Bug in specific projection")
    print("- If all pass (PCC > 0.999): Bug is elsewhere (weight loading?)")
    print("=" * 80)
