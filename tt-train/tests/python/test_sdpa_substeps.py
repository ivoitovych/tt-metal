#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Deep dive into SDPA sub-operations to find exactly where the bug is.

We know SDPA with real BERT data shows PCC 0.81.
This test breaks SDPA into atomic operations:
1. Q @ K^T
2. Softmax(Q @ K^T)
3. Softmax @ V

One of these must be failing.
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


def test_sdpa_substeps():
    """Test each sub-operation of SDPA separately."""
    print("\n" + "=" * 80)
    print("SDPA SUB-OPERATION DEBUGGING")
    print("=" * 80)

    from transformers import BertModel

    # Load BERT
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

    # Create input
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    # Load weights and create QKV
    q_weight = hf_attention.query.weight.data
    k_weight = hf_attention.key.weight.data
    v_weight = hf_attention.value.weight.data
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

    q_bias = hf_attention.query.bias.data
    k_bias = hf_attention.key.bias.data
    v_bias = hf_attention.value.bias.data
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

    # QKV projection (we know this works from previous tests)
    input_ttml = ttml.autograd.Tensor.from_numpy(input_pt.numpy())
    ttml_qkv_weight = ttml.autograd.Tensor.from_numpy(qkv_weight.numpy())
    ttml_qkv_bias = ttml.autograd.Tensor.from_numpy(qkv_bias.numpy())
    ttml_qkv_linear = ttml.modules.LinearLayer(ttml_qkv_weight, ttml_qkv_bias)
    qkv_ttml = ttml_qkv_linear(input_ttml)

    # Heads creation (we know this works)
    q_ttml, k_ttml, v_ttml = ttml.ops.multi_head_utils.heads_creation(qkv_ttml, num_heads)

    # Get PyTorch reference Q, K, V
    input_hf = input_pt.squeeze(1)  # (B, S, E)
    q_hf = F.linear(input_hf, q_weight, q_bias)
    k_hf = F.linear(input_hf, k_weight, k_bias)
    v_hf = F.linear(input_hf, v_weight, v_bias)

    # Reshape to heads: (B, S, E) -> (B, H, S, head_dim)
    B, S, E = q_hf.shape
    q_heads_hf = q_hf.view(B, S, num_heads, head_dim).transpose(1, 2).unsqueeze(2)  # (B, H, 1, S, head_dim)
    k_heads_hf = k_hf.view(B, S, num_heads, head_dim).transpose(1, 2).unsqueeze(2)  # (B, H, 1, S, head_dim)
    v_heads_hf = v_hf.view(B, S, num_heads, head_dim).transpose(1, 2).unsqueeze(2)  # (B, H, 1, S, head_dim)

    # ========================================================================
    # SUB-STEP 1: Q @ K^T (scaled)
    # ========================================================================
    print("SUB-STEP 1: Q @ K^T / sqrt(head_dim)")
    print("-" * 80)

    scale = 1.0 / np.sqrt(head_dim)

    # PyTorch
    # (B, H, 1, S, head_dim) @ (B, H, 1, head_dim, S) -> (B, H, 1, S, S)
    qk_hf = torch.matmul(q_heads_hf, k_heads_hf.transpose(-2, -1)) * scale

    print(f"PyTorch Q @ K^T range: [{qk_hf.min():.4f}, {qk_hf.max():.4f}]")

    # TTML - Manual implementation
    # First scale Q
    q_scaled_ttml = ttml.ops.binary.multiply(q_ttml, ttml.autograd.Tensor.from_numpy(np.array(scale, dtype=np.float32)))

    # Then Q_scaled @ K^T
    # Need to implement matrix multiplication manually
    # For now, let's check if TTML has a matmul operation we can call directly

    print("\n⚠️  Testing TTML's internal matmul operation...")

    # Try using ttnn matmul directly on the underlying tensors
    q_scaled_ttnn = q_scaled_ttml.get_value()
    k_ttnn = k_ttml.get_value()

    # Import ttnn_fixed matmul
    import sys

    sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
    from ttnn_fixed import matmuls

    qk_ttml_ttnn = matmuls.matmul(q_scaled_ttnn, k_ttnn, transpose_a=False, transpose_b=True)
    qk_ttml = ttml.autograd.create_tensor(qk_ttml_ttnn)

    pcc_qk = compute_pcc(qk_hf, qk_ttml)
    print(f"Q @ K^T PCC: {pcc_qk:.8f}")
    print(f"TTML Q @ K^T range: [{qk_ttml.to_numpy().min():.4f}, {qk_ttml.to_numpy().max():.4f}]")

    if pcc_qk < 0.95:
        print("❌ BUG FOUND IN Q @ K^T COMPUTATION!")
        print("\nDiagnostics:")
        qk_diff = np.abs(qk_hf.numpy() - qk_ttml.to_numpy())
        print(f"  Max diff: {qk_diff.max():.6f}")
        print(f"  Mean diff: {qk_diff.mean():.6f}")
        return

    # ========================================================================
    # SUB-STEP 2: Softmax(Q @ K^T)
    # ========================================================================
    print("\nSUB-STEP 2: Softmax(Q @ K^T)")
    print("-" * 80)

    # PyTorch
    attn_weights_hf = F.softmax(qk_hf, dim=-1)  # (B, H, 1, S, S)
    print(f"PyTorch Softmax range: [{attn_weights_hf.min():.4f}, {attn_weights_hf.max():.4f}]")

    # TTML
    # Use metal::softmax
    import _ttml.metal as ttml_metal

    # Need to reshape for softmax: (B, H, 1, S, S) - softmax over last dimension
    attn_weights_ttml_ttnn = ttml_metal.softmax(qk_ttml_ttnn, dim=3)
    attn_weights_ttml = ttml.autograd.create_tensor(attn_weights_ttml_ttnn)

    pcc_softmax = compute_pcc(attn_weights_hf, attn_weights_ttml)
    print(f"Softmax PCC: {pcc_softmax:.8f}")
    print(f"TTML Softmax range: [{attn_weights_ttml.to_numpy().min():.4f}, {attn_weights_ttml.to_numpy().max():.4f}]")

    if pcc_softmax < 0.95:
        print("❌ BUG FOUND IN SOFTMAX!")
        print("\nDiagnostics:")
        softmax_diff = np.abs(attn_weights_hf.numpy() - attn_weights_ttml.to_numpy())
        print(f"  Max diff: {softmax_diff.max():.6f}")
        print(f"  Mean diff: {softmax_diff.mean():.6f}")

        # Check if sum equals 1 for each row
        hf_sums = attn_weights_hf.sum(dim=-1).numpy()
        ttml_sums = attn_weights_ttml.to_numpy().sum(axis=-1)
        print(f"\nSoftmax row sums (should be ~1.0):")
        print(f"  PyTorch: min={hf_sums.min():.6f}, max={hf_sums.max():.6f}")
        print(f"  TTML: min={ttml_sums.min():.6f}, max={ttml_sums.max():.6f}")
        return

    # ========================================================================
    # SUB-STEP 3: Softmax @ V
    # ========================================================================
    print("\nSUB-STEP 3: Softmax @ V")
    print("-" * 80)

    # PyTorch
    # (B, H, 1, S, S) @ (B, H, 1, S, head_dim) -> (B, H, 1, S, head_dim)
    output_hf = torch.matmul(attn_weights_hf, v_heads_hf)
    print(f"PyTorch output range: [{output_hf.min():.4f}, {output_hf.max():.4f}]")

    # TTML
    v_ttnn = v_ttml.get_value()
    output_ttml_ttnn = matmuls.matmul(attn_weights_ttml_ttnn, v_ttnn, transpose_a=False, transpose_b=False)
    output_ttml = ttml.autograd.create_tensor(output_ttml_ttnn)

    pcc_final = compute_pcc(output_hf, output_ttml)
    print(f"Final output PCC: {pcc_final:.8f}")
    print(f"TTML output range: [{output_ttml.to_numpy().min():.4f}, {output_ttml.to_numpy().max():.4f}]")

    if pcc_final < 0.95:
        print("❌ BUG FOUND IN SOFTMAX @ V!")
        return

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Q @ K^T (scaled):  PCC = {pcc_qk:.8f} {'✅' if pcc_qk >= 0.999 else '❌'}")
    print(f"Softmax:           PCC = {pcc_softmax:.8f} {'✅' if pcc_softmax >= 0.999 else '❌'}")
    print(f"Softmax @ V:       PCC = {pcc_final:.8f} {'✅' if pcc_final >= 0.999 else '❌'}")
    print("=" * 80)

    if min(pcc_qk, pcc_softmax, pcc_final) < 0.95:
        print("\n❌ BUG CONFIRMED in one of the sub-steps above!")
    else:
        print("\n✅ All sub-steps work correctly!")
        print("   Bug must be in the integrated SDPA function (not sub-operations).")
