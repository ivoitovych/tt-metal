#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Step-by-step debugging of MultiHeadAttention to find where PCC drops.

We know:
- Individual ops with random weights: PCC >0.99999 ✅
- Full attention with loaded weights: PCC 0.115 ❌

This test checks PCC after EACH step to find where it breaks.
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


def test_attention_step_by_step():
    """Check PCC after each attention operation step."""
    print("\n" + "=" * 80)
    print("STEP-BY-STEP ATTENTION DEBUGGING")
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
    batch_size = 1
    seq_len = 4

    print(f"\nConfiguration: E={embedding_dim}, H={num_heads}, B={batch_size}, S={seq_len}\n")

    # Create input
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    attention_mask_pt = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.float32)

    # Load weights
    q_weight = hf_attention.query.weight.data
    k_weight = hf_attention.key.weight.data
    v_weight = hf_attention.value.weight.data
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

    q_bias = hf_attention.query.bias.data
    k_bias = hf_attention.key.bias.data
    v_bias = hf_attention.value.bias.data
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

    hf_output_dense = hf_model.encoder.layer[0].attention.output.dense
    output_weight = hf_output_dense.weight.data
    output_bias = hf_output_dense.bias.data

    # ========================================================================
    # STEP 1: QKV Projection
    # ========================================================================
    print("STEP 1: QKV Projection")
    print("-" * 80)

    # HuggingFace
    input_hf = input_pt.squeeze(1)  # (B, S, E)
    q_hf = torch.nn.functional.linear(input_hf, q_weight, q_bias)
    k_hf = torch.nn.functional.linear(input_hf, k_weight, k_bias)
    v_hf = torch.nn.functional.linear(input_hf, v_weight, v_bias)
    qkv_hf = torch.cat([q_hf, k_hf, v_hf], dim=-1)  # (B, S, 3E)
    qkv_hf = qkv_hf.unsqueeze(1)  # (B, 1, S, 3E)

    # TTML
    input_ttml = ttml.autograd.Tensor.from_numpy(input_pt.numpy())
    ttml_qkv_weight = ttml.autograd.Tensor.from_numpy(qkv_weight.numpy())
    ttml_qkv_bias = ttml.autograd.Tensor.from_numpy(qkv_bias.numpy())
    ttml_qkv_linear = ttml.modules.LinearLayer(ttml_qkv_weight, ttml_qkv_bias)
    qkv_ttml = ttml_qkv_linear(input_ttml)

    pcc_qkv = compute_pcc(qkv_hf, qkv_ttml)
    print(f"  QKV Projection PCC: {pcc_qkv:.8f}")
    print(f"  HF range: [{qkv_hf.min():.4f}, {qkv_hf.max():.4f}]")
    print(f"  TTML range: [{qkv_ttml.to_numpy().min():.4f}, {qkv_ttml.to_numpy().max():.4f}]")

    if pcc_qkv < 0.95:
        print(f"  ❌ BUG FOUND IN QKV PROJECTION!")
        return

    # ========================================================================
    # STEP 2: Heads Creation
    # ========================================================================
    print("\nSTEP 2: Heads Creation (QKV -> Q, K, V with heads dimension)")
    print("-" * 80)

    # HuggingFace
    head_dim = embedding_dim // num_heads
    qkv_hf_np = qkv_hf.numpy()  # (B, 1, S, 3E)
    B, _, S, _ = qkv_hf_np.shape

    # Split QKV
    q_hf_split = qkv_hf_np[:, :, :, :embedding_dim]  # (B, 1, S, E)
    k_hf_split = qkv_hf_np[:, :, :, embedding_dim : 2 * embedding_dim]
    v_hf_split = qkv_hf_np[:, :, :, 2 * embedding_dim :]

    # Reshape to (B, H, S, head_dim)
    q_heads_hf = (
        q_hf_split.reshape(B, 1, S, num_heads, head_dim).transpose(0, 3, 1, 2, 4).reshape(B, num_heads, 1, S, head_dim)
    )
    k_heads_hf = (
        k_hf_split.reshape(B, 1, S, num_heads, head_dim).transpose(0, 3, 1, 2, 4).reshape(B, num_heads, 1, S, head_dim)
    )
    v_heads_hf = (
        v_hf_split.reshape(B, 1, S, num_heads, head_dim).transpose(0, 3, 1, 2, 4).reshape(B, num_heads, 1, S, head_dim)
    )

    # TTML
    q_ttml, k_ttml, v_ttml = ttml.ops.multi_head_utils.heads_creation(qkv_ttml, num_heads)

    pcc_q = compute_pcc(q_heads_hf, q_ttml)
    pcc_k = compute_pcc(k_heads_hf, k_ttml)
    pcc_v = compute_pcc(v_heads_hf, v_ttml)

    print(f"  Q heads PCC: {pcc_q:.8f}")
    print(f"  K heads PCC: {pcc_k:.8f}")
    print(f"  V heads PCC: {pcc_v:.8f}")

    if pcc_q < 0.95 or pcc_k < 0.95 or pcc_v < 0.95:
        print(f"  ❌ BUG FOUND IN HEADS CREATION!")
        return

    # ========================================================================
    # STEP 3: Scaled Dot-Product Attention
    # ========================================================================
    print("\nSTEP 3: Scaled Dot-Product Attention")
    print("-" * 80)

    # HuggingFace
    # Compute attention scores: Q @ K^T / sqrt(head_dim)
    q_heads_hf_torch = torch.from_numpy(q_heads_hf)  # (B, H, 1, S, head_dim)
    k_heads_hf_torch = torch.from_numpy(k_heads_hf)
    v_heads_hf_torch = torch.from_numpy(v_heads_hf)

    # Reshape for matmul: (B, H, 1, S, head_dim) -> (B*H*1, S, head_dim)
    B_flat = B * num_heads * 1
    q_flat = q_heads_hf_torch.reshape(B_flat, S, head_dim)
    k_flat = k_heads_hf_torch.reshape(B_flat, S, head_dim)
    v_flat = v_heads_hf_torch.reshape(B_flat, S, head_dim)

    # Attention scores
    scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) / np.sqrt(head_dim)  # (B*H*1, S, S)

    # Apply mask (zeros means no masking in BERT)
    # HuggingFace uses -10000 for masked positions
    # scores = scores + attention_mask_pt.reshape(B, 1, 1, S)

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply to values
    attn_output_hf = torch.matmul(attn_weights, v_flat)  # (B*H*1, S, head_dim)
    attn_output_hf = attn_output_hf.reshape(B, num_heads, 1, S, head_dim)

    # TTML
    attention_mask_ttml = ttml.autograd.Tensor.from_numpy(attention_mask_pt.numpy())
    attn_output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(
        q_ttml, k_ttml, v_ttml, attention_mask_ttml
    )

    pcc_attn = compute_pcc(attn_output_hf, attn_output_ttml)
    print(f"  Attention output PCC: {pcc_attn:.8f}")
    print(f"  HF range: [{attn_output_hf.min():.4f}, {attn_output_hf.max():.4f}]")
    print(f"  TTML range: [{attn_output_ttml.to_numpy().min():.4f}, {attn_output_ttml.to_numpy().max():.4f}]")

    if pcc_attn < 0.95:
        print(f"  ❌ BUG FOUND IN SCALED DOT-PRODUCT ATTENTION!")
        return

    # ========================================================================
    # STEP 4: Heads Fusion
    # ========================================================================
    print("\nSTEP 4: Heads Fusion")
    print("-" * 80)

    # HuggingFace: (B, H, 1, S, head_dim) -> (B, 1, S, E)
    attn_output_hf_fused = attn_output_hf.transpose(0, 2, 1, 3, 4).reshape(B, 1, S, embedding_dim)

    # TTML
    attn_fused_ttml = ttml.ops.multi_head_utils.heads_fusion(attn_output_ttml)

    pcc_fusion = compute_pcc(attn_output_hf_fused, attn_fused_ttml)
    print(f"  Fused attention PCC: {pcc_fusion:.8f}")

    if pcc_fusion < 0.95:
        print(f"  ❌ BUG FOUND IN HEADS FUSION!")
        return

    # ========================================================================
    # STEP 5: Output Projection
    # ========================================================================
    print("\nSTEP 5: Output Projection")
    print("-" * 80)

    # HuggingFace
    output_hf = torch.nn.functional.linear(attn_output_hf_fused.squeeze(1), output_weight, output_bias).unsqueeze(1)

    # TTML
    ttml_output_weight = ttml.autograd.Tensor.from_numpy(output_weight.numpy())
    ttml_output_bias = ttml.autograd.Tensor.from_numpy(output_bias.numpy())
    ttml_output_linear = ttml.modules.LinearLayer(ttml_output_weight, ttml_output_bias)
    output_ttml = ttml_output_linear(attn_fused_ttml)

    pcc_output = compute_pcc(output_hf, output_ttml)
    print(f"  Final output PCC: {pcc_output:.8f}")
    print(f"  HF range: [{output_hf.min():.4f}, {output_hf.max():.4f}]")
    print(f"  TTML range: [{output_ttml.to_numpy().min():.4f}, {output_ttml.to_numpy().max():.4f}]")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Step 1 (QKV Projection):  PCC = {pcc_qkv:.8f} {'✅' if pcc_qkv >= 0.999 else '❌'}")
    print(
        f"Step 2 (Heads Creation):  PCC = {min(pcc_q, pcc_k, pcc_v):.8f} {'✅' if min(pcc_q, pcc_k, pcc_v) >= 0.999 else '❌'}"
    )
    print(f"Step 3 (SDPA):            PCC = {pcc_attn:.8f} {'✅' if pcc_attn >= 0.999 else '❌'}")
    print(f"Step 4 (Heads Fusion):    PCC = {pcc_fusion:.8f} {'✅' if pcc_fusion >= 0.999 else '❌'}")
    print(f"Step 5 (Output Proj):     PCC = {pcc_output:.8f} {'✅' if pcc_output >= 0.999 else '❌'}")
    print("=" * 80)

    if pcc_output < 0.95:
        print("\n❌ BUG CONFIRMED! One of the steps above has low PCC!")
    else:
        print("\n✅ All steps work correctly! Bug must be in something else...")
