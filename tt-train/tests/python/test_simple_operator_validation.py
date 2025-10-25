#!/usr/bin/env python3
"""
Simple Operator Validation - Bottom-Up Testing
Test each BERT operation in isolation to find which one is broken.
"""

import numpy as np
import os
import sys
import torch

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x, y):
    """Compute Pearson Correlation Coefficient."""
    x_flat, y_flat = x.flatten(), y.flatten()
    if len(x_flat) != len(y_flat):
        return 0.0
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


def print_result(name, ref, ttml_output, threshold=0.999):
    """Print comparison result."""
    ref_np = ref.detach().numpy() if torch.is_tensor(ref) else ref
    ttml_np = ttml_output.to_numpy() if hasattr(ttml_output, "to_numpy") else ttml_output

    pcc = compute_pcc(ref_np, ttml_np)
    diff = np.abs(ref_np - ttml_np)

    status = "✅ PASS" if pcc > threshold else "❌ FAIL"
    print(f"\n{'='*80}")
    print(f"{name}: PCC = {pcc:.6f} {status}")
    print(f"{'='*80}")
    print(f"Shapes: ref {ref_np.shape}, ttml {ttml_np.shape}")
    print(f"Mean abs diff: {diff.mean():.6e}, Max abs diff: {diff.max():.6e}")
    print(f"First 5 ref:  {ref_np.flatten()[:5]}")
    print(f"First 5 ttml: {ttml_np.flatten()[:5]}")
    print(f"First 5 diff: {diff.flatten()[:5]}")

    return pcc > threshold


print("\n" + "=" * 80)
print("BERT OPERATOR VALIDATION - BOTTOM-UP APPROACH")
print("=" * 80)

# =============================================================================
# TEST 1: Heads Creation
# =============================================================================
print("\n\n" + "=" * 80)
print("TEST 1: Heads Creation (QKV splitting into heads)")
print("=" * 80)

np.random.seed(42)
batch_size = 1
seq_len = 32
hidden_dim = 128
num_heads = 2
head_dim = hidden_dim // num_heads

# Generate random QKV input
qkv_data = np.random.randn(batch_size, seq_len, hidden_dim * 3).astype(np.float32)

# PyTorch reference - manual head splitting
qkv_torch = torch.from_numpy(qkv_data)
q_flat = qkv_torch[:, :, :hidden_dim]
k_flat = qkv_torch[:, :, hidden_dim : hidden_dim * 2]
v_flat = qkv_torch[:, :, hidden_dim * 2 :]

# Reshape: [batch, seq, hidden] -> [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
q_ref = q_flat.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k_ref = k_flat.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
v_ref = v_flat.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

# TTML implementation
qkv_ttml = ttml.autograd.Tensor.from_numpy(qkv_data.reshape(batch_size, 1, seq_len, hidden_dim * 3))
q_ttml, k_ttml, v_ttml = ttml.ops.multi_head_utils.heads_creation(qkv_ttml, num_heads)

# Compare
print("\n--- Q Heads ---")
q_pass = print_result("Q Heads", q_ref, q_ttml)

print("\n--- K Heads ---")
k_pass = print_result("K Heads", k_ref, k_ttml)

print("\n--- V Heads ---")
v_pass = print_result("V Heads", v_ref, v_ttml)

test1_pass = q_pass and k_pass and v_pass
print(f"\n{'='*80}")
print(f"TEST 1 RESULT: {'✅ PASS' if test1_pass else '❌ FAIL'}")
print(f"{'='*80}")


# =============================================================================
# TEST 2: Scaled Dot-Product Attention
# =============================================================================
print("\n\n" + "=" * 80)
print("TEST 2: Scaled Dot-Product Attention")
print("=" * 80)

np.random.seed(42)
q_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
k_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
v_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

# PyTorch reference
q_torch = torch.from_numpy(q_data)
k_torch = torch.from_numpy(k_data)
v_torch = torch.from_numpy(v_data)

scale = 1.0 / np.sqrt(head_dim)
attn_scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
attn_weights = torch.softmax(attn_scores, dim=-1)
output_ref = torch.matmul(attn_weights, v_torch)

# TTML implementation
q_ttml = ttml.autograd.Tensor.from_numpy(q_data)
k_ttml = ttml.autograd.Tensor.from_numpy(k_data)
v_ttml = ttml.autograd.Tensor.from_numpy(v_data)

output_ttml = ttml.ops.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=None)

# Compare
test2_pass = print_result("Attention Output", output_ref, output_ttml, threshold=0.99)

print(f"\n{'='*80}")
print(f"TEST 2 RESULT: {'✅ PASS' if test2_pass else '❌ FAIL'}")
print(f"{'='*80}")


# =============================================================================
# TEST 3: Heads Fusion
# =============================================================================
print("\n\n" + "=" * 80)
print("TEST 3: Heads Fusion (merging heads back)")
print("=" * 80)

np.random.seed(42)
heads_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

# PyTorch reference
heads_torch = torch.from_numpy(heads_data)
# [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads, head_dim] -> [batch, seq, hidden]
fused_ref = heads_torch.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

# TTML implementation
heads_ttml = ttml.autograd.Tensor.from_numpy(heads_data)
fused_ttml = ttml.ops.multi_head_utils.heads_fusion(heads_ttml)

# Compare (TTML output has shape [B, 1, S, E], ref is [B, S, E])
test3_pass = print_result("Fused Heads", fused_ref, fused_ttml)

print(f"\n{'='*80}")
print(f"TEST 3 RESULT: {'✅ PASS' if test3_pass else '❌ FAIL'}")
print(f"{'='*80}")


# =============================================================================
# TEST 4: GELU Activation
# =============================================================================
print("\n\n" + "=" * 80)
print("TEST 4: GELU Activation")
print("=" * 80)

np.random.seed(42)
gelu_input = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# PyTorch reference (exact GELU)
gelu_torch = torch.from_numpy(gelu_input)
gelu_ref = torch.nn.functional.gelu(gelu_torch, approximate="none")

# TTML implementation
gelu_ttml_input = ttml.autograd.Tensor.from_numpy(gelu_input.reshape(batch_size, 1, seq_len, hidden_dim))
gelu_ttml_output = ttml.ops.unary.gelu(gelu_ttml_input)

# Compare
test4_pass = print_result("GELU", gelu_ref, gelu_ttml_output)

print(f"\n{'='*80}")
print(f"TEST 4 RESULT: {'✅ PASS' if test4_pass else '❌ FAIL'}")
print(f"{'='*80}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print(f"TEST 1 - Heads Creation:  {'✅ PASS' if test1_pass else '❌ FAIL'}")
print(f"TEST 2 - Attention:       {'✅ PASS' if test2_pass else '❌ FAIL'}")
print(f"TEST 3 - Heads Fusion:    {'✅ PASS' if test3_pass else '❌ FAIL'}")
print(f"TEST 4 - GELU:            {'✅ PASS' if test4_pass else '❌ FAIL'}")

all_pass = test1_pass and test2_pass and test3_pass and test4_pass
print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
print("=" * 80)

if not all_pass:
    print("\n⚠️  Failed tests indicate which operations are broken!")
    print("Focus debugging effort on the failed operations.")
