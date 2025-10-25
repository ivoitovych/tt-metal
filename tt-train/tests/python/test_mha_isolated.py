#!/usr/bin/env python3
"""
Test Multi-Head Attention in isolation with controlled random data.
This will help identify if MHA is the problem or something else.
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
        print(f"Shape mismatch: {len(x_flat)} vs {len(y_flat)}")
        return 0.0
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


print("=" * 80)
print("ISOLATED MULTI-HEAD ATTENTION TEST")
print("Testing MHA with controlled random weights and data")
print("=" * 80)

# Configuration
batch_size = 1
seq_len = 32
hidden_dim = 128
num_heads = 2
head_dim = hidden_dim // num_heads

np.random.seed(42)

# Generate random input
input_data = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# Generate random QKV weights (same for Q, K, V for simplicity)
qkv_weight_data = np.random.randn(hidden_dim * 3, hidden_dim).astype(np.float32)
qkv_bias_data = np.random.randn(hidden_dim * 3).astype(np.float32)

# Generate random output projection weights
out_weight_data = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
out_bias_data = np.random.randn(hidden_dim).astype(np.float32)

print(f"\nConfiguration:")
print(f"  Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_dim}, Heads: {num_heads}")
print(f"  Input shape: {input_data.shape}")
print(f"  QKV weight shape: {qkv_weight_data.shape}")

# =============================================================================
# PyTorch Reference Implementation
# =============================================================================
print("\n" + "=" * 80)
print("PyTorch Reference")
print("=" * 80)

input_torch = torch.from_numpy(input_data)

# QKV projection
qkv_linear = torch.nn.Linear(hidden_dim, hidden_dim * 3)
qkv_linear.weight.data = torch.from_numpy(qkv_weight_data)
qkv_linear.bias.data = torch.from_numpy(qkv_bias_data)
qkv = qkv_linear(input_torch)  # [batch, seq, hidden*3]

# Split into Q, K, V and reshape into heads
q, k, v = qkv.chunk(3, dim=-1)  # Each is [batch, seq, hidden]
q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch, heads, seq, head_dim]
k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

print(f"After QKV projection and head split:")
print(f"  Q shape: {q.shape}")
print(f"  K shape: {k.shape}")
print(f"  V shape: {v.shape}")

# Scaled dot-product attention
scale = 1.0 / np.sqrt(head_dim)
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, heads, seq, seq]
attn_weights = torch.softmax(attn_scores, dim=-1)
attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

print(f"After attention:")
print(f"  Attention output shape: {attn_output.shape}")

# Merge heads back
attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

# Output projection
out_linear = torch.nn.Linear(hidden_dim, hidden_dim)
out_linear.weight.data = torch.from_numpy(out_weight_data)
out_linear.bias.data = torch.from_numpy(out_bias_data)
output_ref = out_linear(attn_output)

print(f"Final output shape: {output_ref.shape}")
print(f"Output stats: mean={output_ref.mean():.6f}, std={output_ref.std():.6f}")
print(f"First 5 values: {output_ref.flatten()[:5].numpy()}")

# =============================================================================
# TTML Implementation
# =============================================================================
print("\n" + "=" * 80)
print("TTML Implementation")
print("=" * 80)

# Create TTML model
config = ttml.models.bert.BertConfig()
config.vocab_size = 1000
config.max_sequence_length = seq_len
config.embedding_dim = hidden_dim
config.num_heads = num_heads
config.num_blocks = 1  # Just one block to test
config.intermediate_size = 512
config.dropout_prob = 0.0
config.layer_norm_eps = 1e-12

bert_ttml = ttml.models.bert.create(config)
params = bert_ttml.parameters()

print("Setting weights for block 0...")

# Set QKV weights
params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"].set_value(
    ttml.core.from_vector(
        qkv_weight_data.flatten().tolist(),
        params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"].get_value().logical_shape(),
        params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"].get_value().device(),
    )
)

params["bert/bert_block_0/attention/self_attention/qkv_linear/bias"].set_value(
    ttml.core.from_vector(
        qkv_bias_data.flatten().tolist(),
        params["bert/bert_block_0/attention/self_attention/qkv_linear/bias"].get_value().logical_shape(),
        params["bert/bert_block_0/attention/self_attention/qkv_linear/bias"].get_value().device(),
    )
)

# Set output projection weights
params["bert/bert_block_0/attention/self_attention/out_linear/weight"].set_value(
    ttml.core.from_vector(
        out_weight_data.flatten().tolist(),
        params["bert/bert_block_0/attention/self_attention/out_linear/weight"].get_value().logical_shape(),
        params["bert/bert_block_0/attention/self_attention/out_linear/weight"].get_value().device(),
    )
)

params["bert/bert_block_0/attention/self_attention/out_linear/bias"].set_value(
    ttml.core.from_vector(
        out_bias_data.flatten().tolist(),
        params["bert/bert_block_0/attention/self_attention/out_linear/bias"].get_value().logical_shape(),
        params["bert/bert_block_0/attention/self_attention/out_linear/bias"].get_value().device(),
    )
)

print("Weights set successfully!")

# Run input through just the attention part
# We can't easily isolate MHA, so let's just see what the full block produces
# and compare it qualitatively

print("\n⚠️  Note: Cannot easily isolate MHA from BERT block due to LayerNorms")
print("This test validates that the code runs but can't give perfect PCC comparison")
print("For true isolation, need to expose MHA module directly to Python")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("✅ PyTorch reference implementation ran successfully")
print("✅ TTML model created and weights loaded successfully")
print("❌ Cannot test in perfect isolation - MHA not directly exposed to Python")
print("\nRecommendation: Need C++ unit tests or expose MHA Python bindings")
