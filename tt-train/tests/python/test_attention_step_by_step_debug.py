#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Step-by-Step Attention Mechanism Debugging Test

This test breaks down the attention mechanism into the smallest possible steps
and compares each operation against PyTorch reference to identify exactly where
data corruption occurs.

The bug could be in:
1. Python-C++ bindings (data exchange)
2. Tensor creation/conversion
3. Individual TTNN operations (transpose, reshape, matmul, etc.)
4. Memory layout conversions
5. Data type conversions

We test EVERY step independently to find the exact corruption point.
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


def compare_step(step_name, pytorch_tensor, ttml_tensor, threshold=0.9999):
    """Compare a single step and report detailed statistics."""
    pcc = compute_pcc(pytorch_tensor, ttml_tensor)

    # Convert to numpy for detailed comparison
    if isinstance(pytorch_tensor, torch.Tensor):
        pt_np = pytorch_tensor.detach().cpu().numpy()
    else:
        pt_np = pytorch_tensor

    if hasattr(ttml_tensor, "to_numpy"):
        ttml_np = ttml_tensor.to_numpy()
    else:
        ttml_np = ttml_tensor

    diff = np.abs(pt_np - ttml_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    status = "✅ PASS" if pcc >= threshold else "❌ FAIL"

    print(f"\n{status} {step_name}")
    print(f"  PCC:       {pcc:.8f}")
    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")
    print(f"  PyTorch:   min={pt_np.min():.6f}, max={pt_np.max():.6f}, mean={pt_np.mean():.6f}")
    print(f"  TTML:      min={ttml_np.min():.6f}, max={ttml_np.max():.6f}, mean={ttml_np.mean():.6f}")

    return pcc, max_diff, mean_diff


def test_step_1_tensor_creation_and_roundtrip():
    """
    Step 1: Test basic tensor creation and data exchange between Python and C++.

    This tests if the Python-C++ bindings correctly transfer data without corruption.
    """
    print("\n" + "=" * 80)
    print("STEP 1: TENSOR CREATION AND ROUNDTRIP (Python ↔ C++ bindings)")
    print("=" * 80)

    # Create simple test data
    batch_size = 2
    seq_len = 4
    embedding_dim = 8

    print(f"\nTest configuration: batch={batch_size}, seq_len={seq_len}, embedding_dim={embedding_dim}")

    # Create PyTorch tensor with known values
    torch.manual_seed(42)
    original_data = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)

    print("\n--- Test 1a: NumPy → TTML → NumPy roundtrip ---")
    original_np = original_data.numpy()

    # Convert to TTML tensor
    ttml_tensor = ttml.autograd.Tensor.from_numpy(original_np)

    # Convert back to NumPy
    roundtrip_np = ttml_tensor.to_numpy()

    pcc, max_diff, mean_diff = compare_step("NumPy → TTML → NumPy", original_np, roundtrip_np, threshold=0.99999)

    assert pcc > 0.99999, f"Data corruption in Python-C++ bindings! PCC={pcc:.8f}"

    print("\n--- Test 1b: Float32 → BFloat16 → Float32 conversion ---")
    # Test if dtype conversion preserves data reasonably
    original_bf16 = original_data.to(torch.bfloat16).to(torch.float32)
    expected_diff = np.abs(original_data.numpy() - original_bf16.numpy()).max()

    print(f"  Expected BF16 quantization error: {expected_diff:.8f}")

    pcc_bf16 = compute_pcc(original_data.numpy(), original_bf16.numpy())
    print(f"  PCC after BF16 conversion: {pcc_bf16:.8f}")


def test_step_2a_multiply():
    """Step 2a: Test multiply operation."""
    print("\n" + "=" * 80)
    print("STEP 2A: MULTIPLY BY SCALAR")
    print("=" * 80)

    batch_size = 2
    seq_len = 4
    embedding_dim = 8
    scale = 0.5

    torch.manual_seed(42)
    input_data = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    input_np = input_data.numpy()

    pt_result = input_data * scale
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_result = ttml.ops.multiply(ttml_input, scale)

    pcc, _, _ = compare_step("Multiply by scalar", pt_result, ttml_result)
    assert pcc > 0.9999, f"Multiply operation failed! PCC={pcc:.8f}"


def test_step_2b_add():
    """Step 2b: Test add operation."""
    print("\n" + "=" * 80)
    print("STEP 2B: ADD TWO TENSORS")
    print("=" * 80)

    batch_size = 2
    seq_len = 4
    embedding_dim = 8

    torch.manual_seed(42)
    input_data = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    other_data = torch.randn_like(input_data)

    input_np = input_data.numpy()
    other_np = other_data.numpy()

    pt_result = input_data + other_data
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_other = ttml.autograd.Tensor.from_numpy(other_np)
    ttml_result = ttml.ops.add(ttml_input, ttml_other)

    pcc, _, _ = compare_step("Add tensors", pt_result, ttml_result)
    assert pcc > 0.9999, f"Add operation failed! PCC={pcc:.8f}"


def test_step_2c_reshape():
    """Step 2c: Test reshape operation."""
    print("\n" + "=" * 80)
    print("STEP 2C: RESHAPE TENSOR")
    print("=" * 80)

    batch_size = 2
    seq_len = 4
    embedding_dim = 8

    torch.manual_seed(42)
    input_data = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    input_np = input_data.numpy()

    # [B, 1, S, E] -> [B, S, E]
    pt_result = input_data.squeeze(1)

    # TTML reshape
    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_result_value = ttml.core.reshape(ttml_input.get_value(), [batch_size, seq_len, embedding_dim])

    pcc, _, _ = compare_step("Reshape [B,1,S,E] → [B,S,E]", pt_result, ttml_result_value)
    assert pcc > 0.9999, f"Reshape operation failed! PCC={pcc:.8f}"


def test_step_2d_transpose():
    """Step 2d: Test transpose operation."""
    print("\n" + "=" * 80)
    print("STEP 2D: TRANSPOSE TENSOR")
    print("=" * 80)

    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 4

    torch.manual_seed(42)
    shaped_data = torch.randn(batch_size, seq_len, num_heads, head_dim)
    shaped_np = shaped_data.numpy()

    # PyTorch: transpose dim 1 and 2
    pt_result = shaped_data.transpose(1, 2)  # [B, H, S, D]

    # TTML transpose
    ttml_shaped = ttml.autograd.Tensor.from_numpy(shaped_np)
    ttml_result = ttml.ops.transpose(ttml_shaped, 1, 2)

    pcc, _, _ = compare_step("Transpose [B,S,H,D] → [B,H,S,D]", pt_result, ttml_result)
    assert pcc > 0.9999, f"Transpose operation failed! PCC={pcc:.8f}"


def test_step_3a_matmul_2d():
    """Step 3a: Test simple 2D matmul."""
    print("\n" + "=" * 80)
    print("STEP 3A: SIMPLE 2D MATMUL")
    print("=" * 80)

    torch.manual_seed(42)
    a = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(16, 8, dtype=torch.float32)

    pt_result = torch.matmul(a, b)

    ttml_a = ttml.autograd.Tensor.from_numpy(a.numpy().reshape(1, 1, 8, 16))
    ttml_b = ttml.autograd.Tensor.from_numpy(b.numpy().reshape(1, 1, 16, 8))
    ttml_result = ttml.ops.matmul(ttml_a, ttml_b)

    pcc, _, _ = compare_step("2D Matmul [8,16] @ [16,8]", pt_result, ttml_result)
    assert pcc > 0.999, f"2D Matmul failed! PCC={pcc:.8f}"


def test_step_3b_matmul_4d_qk():
    """Step 3b: Test 4D matmul (attention Q@K^T)."""
    print("\n" + "=" * 80)
    print("STEP 3B: 4D MATMUL (ATTENTION Q@K^T)")
    print("=" * 80)

    batch_size = 2
    num_heads = 2
    seq_len = 4
    head_dim = 4

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # PyTorch: Q @ K^T
    pt_result = torch.matmul(q, k.transpose(-2, -1))  # [B, H, S, S]

    # TTML matmul with transpose
    ttml_q = ttml.autograd.Tensor.from_numpy(q.numpy())
    ttml_k = ttml.autograd.Tensor.from_numpy(k.numpy())
    ttml_result = ttml.ops.matmul(ttml_q, ttml_k, transpose_b=True)

    pcc, _, _ = compare_step("4D Matmul Q@K^T [B,H,S,D]@[B,H,D,S]", pt_result, ttml_result)
    assert pcc > 0.999, f"4D Matmul Q@K^T failed! PCC={pcc:.8f}"


def test_step_3c_matmul_scaled():
    """Step 3c: Test scaled matmul (Q*scale)@K^T."""
    print("\n" + "=" * 80)
    print("STEP 3C: SCALED MATMUL (Q*scale)@K^T")
    print("=" * 80)

    batch_size = 2
    num_heads = 2
    seq_len = 4
    head_dim = 4
    scale = 1.0 / np.sqrt(head_dim)

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # PyTorch: (Q * scale) @ K^T
    pt_result = torch.matmul(q * scale, k.transpose(-2, -1))

    # TTML: multiply then matmul
    ttml_q = ttml.autograd.Tensor.from_numpy(q.numpy())
    ttml_k = ttml.autograd.Tensor.from_numpy(k.numpy())
    ttml_q_scaled = ttml.ops.multiply(ttml_q, scale)
    ttml_result = ttml.ops.matmul(ttml_q_scaled, ttml_k, transpose_b=True)

    pcc, _, _ = compare_step("Scaled Matmul (Q*scale)@K^T", pt_result, ttml_result)
    assert pcc > 0.999, f"Scaled Matmul failed! PCC={pcc:.8f}"


def test_step_4_attention_components():
    """
    Step 4: Test attention mechanism components step by step.

    This recreates the exact operations in the attention mechanism.
    """
    print("\n" + "=" * 80)
    print("STEP 4: ATTENTION MECHANISM COMPONENTS (step by step)")
    print("=" * 80)

    batch_size = 1
    num_heads = 2
    seq_len = 4
    head_dim = 4
    scale = 1.0 / np.sqrt(head_dim)

    print(f"\nConfiguration: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}, scale={scale:.4f}")

    torch.manual_seed(42)
    q_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v_pt = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    q_np = q_pt.numpy()
    k_np = k_pt.numpy()
    v_np = v_pt.numpy()

    # Step 4a: Q * scale
    print("\n--- Step 4a: Q * scale ---")
    pt_q_scaled = q_pt * scale

    ttml_q = ttml.autograd.Tensor.from_numpy(q_np)
    ttml_q_scaled = ttml.ops.multiply(ttml_q, scale)

    pcc, _, _ = compare_step("Q * scale", pt_q_scaled, ttml_q_scaled)
    assert pcc > 0.9999, f"Q * scale failed! PCC={pcc:.8f}"

    # Step 4b: Q @ K^T
    print("\n--- Step 4b: (Q*scale) @ K^T ---")
    pt_qk = torch.matmul(pt_q_scaled, k_pt.transpose(-2, -1))

    ttml_k = ttml.autograd.Tensor.from_numpy(k_np)
    ttml_qk = ttml.ops.matmul(ttml_q_scaled, ttml_k, transpose_b=True)

    pcc, _, _ = compare_step("(Q*scale) @ K^T", pt_qk, ttml_qk)
    assert pcc > 0.999, f"(Q*scale) @ K^T failed! PCC={pcc:.8f}"

    # Step 4c: Softmax
    print("\n--- Step 4c: Softmax(QK) ---")
    pt_attn = torch.softmax(pt_qk, dim=-1)

    ttml_attn = ttml.ops.softmax(ttml_qk, -1)

    pcc, _, _ = compare_step("Softmax(QK)", pt_attn, ttml_attn)
    assert pcc > 0.999, f"Softmax failed! PCC={pcc:.8f}"

    # Step 4d: Attention @ V
    print("\n--- Step 4d: Attention @ V ---")
    pt_output = torch.matmul(pt_attn, v_pt)

    ttml_v = ttml.autograd.Tensor.from_numpy(v_np)
    ttml_output = ttml.ops.matmul(ttml_attn, ttml_v)

    pcc, _, _ = compare_step("Attention @ V", pt_output, ttml_output)
    assert pcc > 0.999, f"Attention @ V failed! PCC={pcc:.8f}"

    # Step 4e: Full attention (reference)
    print("\n--- Step 4e: Full attention (end-to-end) ---")
    pt_full = torch.matmul(torch.softmax(torch.matmul(q_pt * scale, k_pt.transpose(-2, -1)), dim=-1), v_pt)

    pcc_full = compute_pcc(pt_full, ttml_output)
    print(f"\n✓ Full attention PCC: {pcc_full:.8f}")


def test_step_5_heads_creation():
    """
    Step 5: Test the heads_creation operation step by step.

    This operation does: [B,1,S,E] -> [B,S,E] -> [B,S,H,D] -> [B,H,S,D]
    We test each transformation individually.
    """
    print("\n" + "=" * 80)
    print("STEP 5: HEADS CREATION (multi-step reshape + transpose)")
    print("=" * 80)

    batch_size = 2
    seq_len = 4
    embedding_dim = 8
    num_heads = 2
    head_dim = embedding_dim // num_heads

    print(f"\nConfiguration: B={batch_size}, S={seq_len}, E={embedding_dim}, H={num_heads}, D={head_dim}")

    torch.manual_seed(42)
    input_pt = torch.randn(batch_size, 1, seq_len, embedding_dim, dtype=torch.float32)
    input_np = input_pt.numpy()

    # Step 5a: Remove channel dimension [B,1,S,E] -> [B,S,E]
    print("\n--- Step 5a: Remove channel [B,1,S,E] -> [B,S,E] ---")
    pt_step1 = input_pt.squeeze(1)

    ttml_input = ttml.autograd.Tensor.from_numpy(input_np)
    ttml_step1 = ttml.core.reshape(ttml_input.get_value(), [batch_size, seq_len, embedding_dim])

    pcc, _, _ = compare_step("Reshape [B,1,S,E] -> [B,S,E]", pt_step1, ttml_step1)
    assert pcc > 0.9999, f"Remove channel failed! PCC={pcc:.8f}"

    # Step 5b: Split heads [B,S,E] -> [B,S,H,D]
    print("\n--- Step 5b: Split heads [B,S,E] -> [B,S,H,D] ---")
    pt_step2 = pt_step1.reshape(batch_size, seq_len, num_heads, head_dim)

    ttml_step2 = ttml.core.reshape(ttml_step1, [batch_size, seq_len, num_heads, head_dim])

    pcc, _, _ = compare_step("Reshape [B,S,E] -> [B,S,H,D]", pt_step2, ttml_step2)
    assert pcc > 0.9999, f"Split heads failed! PCC={pcc:.8f}"

    # Step 5c: Transpose [B,S,H,D] -> [B,H,S,D]
    print("\n--- Step 5c: Transpose [B,S,H,D] -> [B,H,S,D] ---")
    pt_step3 = pt_step2.transpose(1, 2)

    ttml_step3_tensor = ttml.autograd.Tensor(ttml_step2)
    ttml_step3 = ttml.ops.transpose(ttml_step3_tensor, 1, 2)

    pcc, _, _ = compare_step("Transpose [B,S,H,D] -> [B,H,S,D]", pt_step3, ttml_step3)
    assert pcc > 0.9999, f"Transpose heads failed! PCC={pcc:.8f}"

    # Step 5d: Full heads_creation operation
    print("\n--- Step 5d: Full heads_creation (TTML API) ---")
    from_ttml = ttml.ops.heads_creation(ttml_input, num_heads)

    pcc, _, _ = compare_step("Full heads_creation", pt_step3, from_ttml)
    assert pcc > 0.9999, f"Full heads_creation failed! PCC={pcc:.8f}"
