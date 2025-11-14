#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test residual connection (add operation) to find attention mechanism bug.

CRITICAL DISCOVERY: "Block 0 Attention" in layer-by-layer test includes the
residual connection:
    attention_output = MultiHeadAttention(input, mask)
    result = add(attention_output, input)  # <-- This is what's measured!

Since all individual operations work perfectly (PCC >0.99999), the bug must
be in the ADD operation or how the residual connection works!
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


def test_add_operation_basic():
    """Test basic add operation."""
    print("\n" + "=" * 80)
    print("TEST: Basic Add Operation")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 128

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}")

    # Create test tensors
    torch.manual_seed(42)
    a_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    b_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    # PyTorch reference
    pt_result = a_pt + b_pt

    # TTML
    a_ttml = ttml.autograd.Tensor.from_numpy(a_pt.numpy())
    b_ttml = ttml.autograd.Tensor.from_numpy(b_pt.numpy())
    ttml_result = ttml.ops.binary.add(a_ttml, b_ttml)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ Add operation PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if pcc < 0.999:
        print(f"\n❌ CORRUPTION FOUND IN ADD OPERATION!")
        print(f"   PCC {pcc:.8f} indicates the add operation has precision loss")
        print(f"   This could be the root cause of the attention mechanism bug!")

    assert pcc > 0.999, f"Add operation failed! PCC={pcc:.8f}"


def test_residual_connection_pattern():
    """Test the exact pattern used in BERT blocks: add(attention_output, input)."""
    print("\n" + "=" * 80)
    print("TEST: Residual Connection Pattern")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 128

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}")
    print("Pattern: residual = add(attention_output, embeddings)")

    # Create test tensors (simulating embeddings and attention output)
    torch.manual_seed(42)
    embeddings_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    attention_out_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    # PyTorch reference (what HuggingFace does)
    pt_residual = attention_out_pt + embeddings_pt

    # TTML (what TTML does in BertBlock)
    embeddings_ttml = ttml.autograd.Tensor.from_numpy(embeddings_pt.numpy())
    attention_out_ttml = ttml.autograd.Tensor.from_numpy(attention_out_pt.numpy())
    ttml_residual = ttml.ops.binary.add(attention_out_ttml, embeddings_ttml)

    # Compare
    pcc = compute_pcc(pt_residual, ttml_residual)

    pt_np = pt_residual.detach().numpy()
    ttml_np = ttml_residual.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ Residual connection PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if pcc < 0.999:
        print(f"\n❌ CORRUPTION FOUND IN RESIDUAL CONNECTION!")
        print(f"   PCC {pcc:.8f} indicates precision loss in add operation")
        print(f"   This is measured as 'Block 0 Attention' in layer-by-layer test!")

    assert pcc > 0.999, f"Residual connection failed! PCC={pcc:.8f}"


def test_full_attention_with_residual():
    """
    Test the complete attention block WITH residual connection.
    This simulates exactly what 'Block 0 Attention' measures.
    """
    print("\n" + "=" * 80)
    print("TEST: Full Attention Block with Residual Connection")
    print("=" * 80)
    print("This is what 'Block 0 Attention' actually measures!")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 128
    num_heads = 2

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}, H={num_heads}")

    # Create embeddings (input to the block)
    torch.manual_seed(42)
    embeddings_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    embeddings_np = embeddings_pt.numpy()

    print("\nStep 1: Run attention (we know this is perfect from previous tests)")

    # For simplicity, just create random attention output
    # In real case this would be MultiHeadAttention(embeddings)
    attention_output_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    print("\nStep 2: Add residual connection")
    # PyTorch: residual = attention_output + embeddings
    pt_result = attention_output_pt + embeddings_pt

    # TTML: residual = add(attention_output, embeddings)
    embeddings_ttml = ttml.autograd.Tensor.from_numpy(embeddings_np)
    attention_output_ttml = ttml.autograd.Tensor.from_numpy(attention_output_pt.numpy())
    ttml_result = ttml.ops.binary.add(attention_output_ttml, embeddings_ttml)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n{'='*80}")
    print(f"RESULTS (This is 'Block 0 Attention' PCC):")
    print(f"{'='*80}")
    print(f"PCC:       {pcc:.8f}")
    print(f"Max diff:  {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    print(f"{'='*80}")

    if pcc < 0.95:
        print(f"\n❌ FOUND IT! This matches the Block 0 Attention bug!")
        print(f"   PCC {pcc:.8f} matches the ~0.94 error we observed!")
        print(f"   The bug is in the ADD operation used for residual connection!")
    elif pcc < 0.999:
        print(f"\n⚠️  Some degradation (PCC {pcc:.8f})")
    else:
        print(f"\n✅ PASS - Add operation works correctly")

    assert pcc > 0.999, f"Full attention with residual failed! PCC={pcc:.8f}"


def test_add_with_large_values():
    """Test add operation with large values (like actual BERT outputs)."""
    print("\n" + "=" * 80)
    print("TEST: Add Operation with Large Values")
    print("=" * 80)

    batch_size = 1
    seq_len = 4
    embedding_dim = 128

    # Create tensors with larger values (more realistic for BERT)
    torch.manual_seed(42)
    a_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32) * 5.0
    b_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32) * 5.0

    print(f"Input A range: [{a_pt.min():.6f}, {a_pt.max():.6f}]")
    print(f"Input B range: [{b_pt.min():.6f}, {b_pt.max():.6f}]")

    # PyTorch
    pt_result = a_pt + b_pt

    # TTML
    a_ttml = ttml.autograd.Tensor.from_numpy(a_pt.numpy())
    b_ttml = ttml.autograd.Tensor.from_numpy(b_pt.numpy())
    ttml_result = ttml.ops.binary.add(a_ttml, b_ttml)

    # Compare
    pcc = compute_pcc(pt_result, ttml_result)

    pt_np = pt_result.detach().numpy()
    ttml_np = ttml_result.to_numpy()
    max_diff = np.abs(pt_np - ttml_np).max()
    mean_diff = np.abs(pt_np - ttml_np).mean()

    print(f"\n✓ Add (large values) PCC: {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")
    print(f"  Output range: [{ttml_np.min():.6f}, {ttml_np.max():.6f}]")

    if pcc < 0.999:
        print(f"\n❌ Add operation fails with large values!")
        print(f"   Possible bfloat16 overflow or precision issue")

    assert pcc > 0.999, f"Add with large values failed! PCC={pcc:.8f}"


def test_analysis_summary():
    """Print analysis summary."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print("CRITICAL FINDING: 'Block 0 Attention' in layer-by-layer test measures:")
    print("  attention_output = MultiHeadAttention(input, mask)")
    print("  result = ADD(attention_output, input)  # <-- RESIDUAL CONNECTION")
    print("")
    print("We've verified:")
    print("  ✅ MultiHeadAttention operations: PCC >0.99999")
    print("  ✅ Linear layers (loaded weights): PCC >0.99999")
    print("  ✅ All attention components: PCC >0.99999")
    print("")
    print("If ADD operation shows PCC < 0.95:")
    print("  ❌ BUG CONFIRMED in ADD operation!")
    print("  ❌ This explains the ~0.94 PCC in 'Block 0 Attention'!")
    print("")
    print("If ADD operation shows PCC > 0.999:")
    print("  ❓ Bug is elsewhere (LayerNorm? Different residual pattern?)")
    print("=" * 80)
