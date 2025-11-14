#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test linear layers with LOADED WEIGHTS from HuggingFace BERT model.

Since linear layers work perfectly with random weights (PCC >0.99999), but
Block 0 Attention shows PCC 0.94 with loaded weights, this test will identify
the exact weight loading bug.
"""

import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import BertModel

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


def test_qkv_linear_with_loaded_weights():
    """Test QKV linear layer with actual loaded BERT weights."""
    print("\n" + "=" * 80)
    print("TEST: QKV Linear Layer with Loaded HuggingFace Weights")
    print("=" * 80)

    # Load HuggingFace BERT model
    print("\nLoading bert-tiny model...")
    model_name = "prajjwal1/bert-tiny"
    hf_model = BertModel.from_pretrained(model_name)
    hf_model.eval()

    # Get Layer 0 attention weights
    layer_0_attn = hf_model.encoder.layer[0].attention.self

    # Get Q, K, V weights and biases
    q_weight = layer_0_attn.query.weight.data  # [E, E]
    q_bias = layer_0_attn.query.bias.data  # [E]
    k_weight = layer_0_attn.key.weight.data
    k_bias = layer_0_attn.key.bias.data
    v_weight = layer_0_attn.value.weight.data
    v_bias = layer_0_attn.value.bias.data

    embedding_dim = q_weight.shape[0]
    print(f"Embedding dimension: {embedding_dim}")

    # Combine Q, K, V weights into single QKV weight (like TTML does)
    # Shape: [3E, E]
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

    print(f"QKV weight shape: {qkv_weight.shape}")
    print(f"QKV bias shape: {qkv_bias.shape}")

    # Create test input [B, 1, S, E]
    batch_size = 1
    seq_len = 4
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    print(f"Input shape: {input_pt.shape}")

    # PyTorch reference with loaded weights
    pt_result = F.linear(input_pt, qkv_weight, qkv_bias)

    print(f"PyTorch output shape: {pt_result.shape}")
    print(f"PyTorch output range: [{pt_result.min():.6f}, {pt_result.max():.6f}]")

    # TTML with loaded weights
    input_np = input_pt.numpy()
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)

    # Create TTML linear layer with loaded weights
    ttml_weight = ttml.autograd.Tensor.from_numpy(qkv_weight.numpy())
    ttml_bias = ttml.autograd.Tensor.from_numpy(qkv_bias.numpy())
    ttml_linear = ttml.modules.LinearLayer(ttml_weight, ttml_bias)

    ttml_result = ttml_linear(ttml_input)

    ttml_np = ttml_result.to_numpy()
    print(f"TTML output shape: {ttml_np.shape}")
    print(f"TTML output range: [{ttml_np.min():.6f}, {ttml_np.max():.6f}]")

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
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
        print(f"\n❌ CRITICAL BUG FOUND!")
        print(f"   QKV linear layer with LOADED WEIGHTS shows PCC {pcc:.8f}")
        print(f"   This matches the Block 0 Attention error (~0.94)!")
        print(f"   BUG CONFIRMED: Weight loading/application for linear layers!")

        # Debug: Check weight statistics
        print(f"\n📊 Weight Statistics:")
        print(
            f"   PyTorch QKV weight: min={qkv_weight.min():.6f}, max={qkv_weight.max():.6f}, mean={qkv_weight.mean():.6f}"
        )
        print(
            f"   TTML QKV weight: min={ttml_weight.to_numpy().min():.6f}, max={ttml_weight.to_numpy().max():.6f}, mean={ttml_weight.to_numpy().mean():.6f}"
        )

        # Check if weights match
        weight_pcc = compute_pcc(qkv_weight, ttml_weight)
        print(f"   Weight PCC: {weight_pcc:.8f}")

    elif pcc < 0.999:
        print(f"\n⚠️  Degradation detected (PCC {pcc:.8f})")
        print(f"   Not catastrophic but below expected precision")
    else:
        print(f"\n✅ PASS - Weights load correctly!")

    # Don't fail the test yet - we want to see all results
    # assert pcc > 0.95, f"QKV linear with loaded weights failed! PCC={pcc:.8f}"


def test_output_linear_with_loaded_weights():
    """Test output linear layer with actual loaded BERT weights."""
    print("\n" + "=" * 80)
    print("TEST: Output Linear Layer with Loaded HuggingFace Weights")
    print("=" * 80)

    # Load HuggingFace BERT model
    print("\nLoading bert-tiny model...")
    model_name = "prajjwal1/bert-tiny"
    hf_model = BertModel.from_pretrained(model_name)
    hf_model.eval()

    # Get Layer 0 output projection weights
    layer_0_output = hf_model.encoder.layer[0].attention.output.dense

    out_weight = layer_0_output.weight.data  # [E, E]
    out_bias = layer_0_output.bias.data  # [E]

    embedding_dim = out_weight.shape[0]
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Output weight shape: {out_weight.shape}")
    print(f"Output bias shape: {out_bias.shape}")

    # Create test input [B, 1, S, E]
    batch_size = 1
    seq_len = 4
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    print(f"Input shape: {input_pt.shape}")

    # PyTorch reference with loaded weights
    pt_result = F.linear(input_pt, out_weight, out_bias)

    print(f"PyTorch output shape: {pt_result.shape}")
    print(f"PyTorch output range: [{pt_result.min():.6f}, {pt_result.max():.6f}]")

    # TTML with loaded weights
    input_np = input_pt.numpy()
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)

    # Create TTML linear layer with loaded weights
    ttml_weight = ttml.autograd.Tensor.from_numpy(out_weight.numpy())
    ttml_bias = ttml.autograd.Tensor.from_numpy(out_bias.numpy())
    ttml_linear = ttml.modules.LinearLayer(ttml_weight, ttml_bias)

    ttml_result = ttml_linear(ttml_input)

    ttml_np = ttml_result.to_numpy()
    print(f"TTML output shape: {ttml_np.shape}")
    print(f"TTML output range: [{ttml_np.min():.6f}, {ttml_np.max():.6f}]")

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
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
        print(f"\n❌ CRITICAL BUG FOUND!")
        print(f"   Output linear layer with LOADED WEIGHTS shows PCC {pcc:.8f}")
        print(f"   This matches the Block 0 Attention error (~0.94)!")
        print(f"   BUG CONFIRMED: Weight loading/application for linear layers!")

        # Debug: Check weight statistics
        print(f"\n📊 Weight Statistics:")
        print(
            f"   PyTorch output weight: min={out_weight.min():.6f}, max={out_weight.max():.6f}, mean={out_weight.mean():.6f}"
        )
        print(
            f"   TTML output weight: min={ttml_weight.to_numpy().min():.6f}, max={ttml_weight.to_numpy().max():.6f}, mean={ttml_weight.to_numpy().mean():.6f}"
        )

        # Check if weights match
        weight_pcc = compute_pcc(out_weight, ttml_weight)
        print(f"   Weight PCC: {weight_pcc:.8f}")

    elif pcc < 0.999:
        print(f"\n⚠️  Degradation detected (PCC {pcc:.8f})")
        print(f"   Not catastrophic but below expected precision")
    else:
        print(f"\n✅ PASS - Weights load correctly!")


def test_weight_transpose_hypothesis():
    """Test if the bug is in weight transpose during loading."""
    print("\n" + "=" * 80)
    print("TEST: Weight Transpose Hypothesis")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = BertModel.from_pretrained(model_name)

    layer_0_attn = hf_model.encoder.layer[0].attention.self
    q_weight = layer_0_attn.query.weight.data

    embedding_dim = q_weight.shape[0]
    batch_size = 1
    seq_len = 4

    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    q_bias = layer_0_attn.query.bias.data

    # Test 1: Normal weight
    pt_result_normal = F.linear(input_pt, q_weight, q_bias)

    input_np = input_pt.numpy()
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_weight_normal = ttml.autograd.Tensor.from_numpy(q_weight.numpy())
    ttml_bias = ttml.autograd.Tensor.from_numpy(q_bias.numpy())
    ttml_linear_normal = ttml.modules.LinearLayer(ttml_weight_normal, ttml_bias)
    ttml_result_normal = ttml_linear_normal(ttml_input)

    pcc_normal = compute_pcc(pt_result_normal, ttml_result_normal)
    print(f"Normal weight PCC: {pcc_normal:.8f}")

    # Test 2: Transposed weight
    q_weight_t = q_weight.t()  # Transpose
    pt_result_transposed = F.linear(input_pt, q_weight_t, q_bias)

    ttml_weight_t = ttml.autograd.Tensor.from_numpy(q_weight_t.numpy())
    ttml_linear_t = ttml.modules.LinearLayer(ttml_weight_t, ttml_bias)
    ttml_result_t = ttml_linear_t(ttml_input)

    pcc_transposed = compute_pcc(pt_result_transposed, ttml_result_t)
    print(f"Transposed weight PCC: {pcc_transposed:.8f}")

    print(f"\n📊 Analysis:")
    print(f"   If normal PCC is high (~0.999): Weights are loaded correctly")
    print(f"   If transposed PCC is high (~0.999): Weights need transpose during loading")

    if pcc_normal > 0.999:
        print(f"\n✅ Weights are loaded correctly (no transpose needed)")
    elif pcc_transposed > 0.999:
        print(f"\n❌ BUG: Weights need to be TRANSPOSED during loading!")
    else:
        print(f"\n❓ Neither works well - different issue")
