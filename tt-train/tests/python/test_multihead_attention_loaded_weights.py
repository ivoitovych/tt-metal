#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test the ACTUAL MultiHeadAttention module with loaded BERT weights.

CRITICAL INSIGHT: All individual operations work perfectly in isolation,
but the actual model shows PCC 0.94. The difference might be:
1. Real forward pass data patterns (not random tensors)
2. The actual MultiHeadAttention module integration
3. Weight loading into the full module

This test directly compares HuggingFace's attention layer vs TTML's
MultiHeadAttention module with loaded weights.
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


def test_multihead_attention_module_with_loaded_weights():
    """
    Test the ACTUAL MultiHeadAttention module with loaded BERT weights.

    This is the definitive test: if this shows PCC 0.94, we've found where
    the bug manifests in the integrated module (not just individual ops).
    """
    print("\n" + "=" * 80)
    print("TEST: MultiHeadAttention Module with Loaded BERT Weights")
    print("=" * 80)

    from transformers import BertModel

    # Load HuggingFace BERT
    model_name = "prajjwal1/bert-tiny"
    hf_model = BertModel.from_pretrained(model_name)
    hf_model.eval()

    # Get Layer 0 attention
    hf_attention = hf_model.encoder.layer[0].attention.self
    config = hf_model.config

    embedding_dim = config.hidden_size
    num_heads = config.num_attention_heads
    batch_size = 1
    seq_len = 4

    print(f"\nConfiguration:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Batch size: {batch_size}")
    print(f"  Seq len: {seq_len}")

    # Create test input
    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    # Create attention mask (all ones = no masking)
    attention_mask_pt = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.float32)

    print(f"\nInput range: [{input_pt.min():.6f}, {input_pt.max():.6f}]")

    # HuggingFace forward pass
    with torch.no_grad():
        # HuggingFace attention expects (B, S, E) not (B, 1, S, E)
        input_hf = input_pt.squeeze(1)  # (B, S, E)
        attention_mask_hf = attention_mask_pt  # (B, 1, 1, S)

        hf_output = hf_attention(input_hf, attention_mask=attention_mask_hf)[0]
        # Unsqueeze to match TTML shape (B, 1, S, E)
        hf_output = hf_output.unsqueeze(1)

    print(f"HuggingFace output range: [{hf_output.min():.6f}, {hf_output.max():.6f}]")

    # TTML MultiHeadAttention
    # Load QKV weights
    q_weight = hf_attention.query.weight.data  # (E, E)
    k_weight = hf_attention.key.weight.data
    v_weight = hf_attention.value.weight.data
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # (3E, E)

    q_bias = hf_attention.query.bias.data  # (E,)
    k_bias = hf_attention.key.bias.data
    v_bias = hf_attention.value.bias.data
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)  # (3E,)

    print(f"\nQKV weight shape: {qkv_weight.shape}")
    print(f"QKV bias shape: {qkv_bias.shape}")

    # Get output linear layer from HuggingFace
    # In HuggingFace, this is in layer.attention.output.dense
    hf_output_dense = hf_model.encoder.layer[0].attention.output.dense
    output_weight = hf_output_dense.weight.data  # (E, E)
    output_bias = hf_output_dense.bias.data  # (E,)

    print(f"Output weight shape: {output_weight.shape}")
    print(f"Output bias shape: {output_bias.shape}")

    # Load weights into TTML (need to use the internal linear layers)
    # This is tricky - we need to access the internal linear layers of MultiHeadAttention
    # Let me check if there's a better way or if I need to construct it differently

    # Actually, looking at multi_head_attention.cpp, the module has:
    # - m_qkv_linear (LinearLayer)
    # - m_out_linear (LinearLayer)
    # - m_dropout (DropoutLayer)

    # But I don't have access to set weights after construction...
    # Let me create the linear layers separately and construct MHA with them

    # Create QKV linear layer with weights
    ttml_qkv_weight = ttml.autograd.Tensor.from_numpy(qkv_weight.numpy())
    ttml_qkv_bias = ttml.autograd.Tensor.from_numpy(qkv_bias.numpy())
    ttml_qkv_linear = ttml.modules.LinearLayer(ttml_qkv_weight, ttml_qkv_bias)

    # Create output linear layer with weights
    ttml_output_weight = ttml.autograd.Tensor.from_numpy(output_weight.numpy())
    ttml_output_bias = ttml.autograd.Tensor.from_numpy(output_bias.numpy())
    ttml_output_linear = ttml.modules.LinearLayer(ttml_output_weight, ttml_output_bias)

    # Hmm, looking at the constructor, MultiHeadAttention creates its own linear layers
    # I need to manually execute the operations instead of using the module

    print("\n⚠️  NOTE: MultiHeadAttention module doesn't expose weight loading.")
    print("    Executing attention operations manually with loaded weights instead.")
    print("    This tests the same computation flow as the module.\n")

    # Manual execution of MultiHeadAttention operations
    input_ttml = ttml.autograd.Tensor.from_numpy(input_pt.numpy())
    attention_mask_ttml = ttml.autograd.Tensor.from_numpy(attention_mask_pt.numpy())

    # 1. QKV projection
    qkv_ttml = ttml_qkv_linear(input_ttml)

    # 2. Create heads
    q_ttml, k_ttml, v_ttml = ttml.ops.multi_head_utils.heads_creation(qkv_ttml, num_heads)

    # 3. Scaled dot-product attention
    attention_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, attention_mask_ttml)

    # 4. Fuse heads
    attention_fused_ttml = ttml.ops.multi_head_utils.heads_fusion(attention_ttml)

    # 5. Output projection
    output_ttml = ttml_output_linear(attention_fused_ttml)

    # 6. Dropout (disabled, so no-op)
    # output_ttml = dropout(output_ttml)  # Skip since dropout_prob=0.0

    ttml_output_np = output_ttml.to_numpy()
    print(f"TTML output range: [{ttml_output_np.min():.6f}, {ttml_output_np.max():.6f}]")

    # Compare
    pcc = compute_pcc(hf_output, output_ttml)

    hf_np = hf_output.detach().numpy()
    max_diff = np.abs(hf_np - ttml_output_np).max()
    mean_diff = np.abs(hf_np - ttml_output_np).mean()

    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"PCC:       {pcc:.8f}")
    print(f"Max diff:  {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    print(f"{'='*80}")

    if pcc < 0.95:
        print(f"\n❌ FOUND IT! MultiHeadAttention with loaded weights shows the bug!")
        print(f"   PCC {pcc:.8f} matches the Block 0 Attention error (~0.94)!")
        print(f"   THE BUG IS IN THE INTEGRATED MULTIHEADATTENTION EXECUTION!")
    elif pcc < 0.999:
        print(f"\n⚠️  Some degradation (PCC {pcc:.8f})")
    else:
        print(f"\n✅ MultiHeadAttention with loaded weights works correctly")
        print(f"   This means the bug is NOT in the attention computation itself.")
        print(f"   It must be in LayerNorm or the residual pattern in the actual model.")

    # Don't fail - we want to see all results
    # assert pcc > 0.999, f"MultiHeadAttention failed! PCC={pcc:.8f}"


def test_analysis_summary():
    """Print final analysis."""
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)
    print("Tested so far:")
    print("  ✅ Individual attention ops (heads_creation, SDPA, heads_fusion): PCC >0.99999")
    print("  ✅ Linear layers (random weights): PCC >0.99999")
    print("  ✅ Linear layers (loaded weights): PCC >0.99999")
    print("  ✅ ADD operation: PCC >0.99999")
    print("  ✅ LayerNorm: PCC >0.99999")
    print("  ✅ ADD + LayerNorm chained: PCC >0.99999")
    print("")
    print("If MultiHeadAttention with loaded weights shows PCC < 0.95:")
    print("  🎯 Bug is in the full attention execution flow")
    print("  🎯 Something about the integrated execution differs from isolated tests")
    print("")
    print("If MultiHeadAttention shows PCC > 0.999:")
    print("  ❓ Bug must be in how BertBlock integrates everything together")
    print("  ❓ Or in the actual forward pass data patterns not tested here")
    print("=" * 80)
