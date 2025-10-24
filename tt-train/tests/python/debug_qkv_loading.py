#!/usr/bin/env python3
"""Debug script to inspect QKV weight loading."""

import numpy as np
import torch
import transformers
from pathlib import Path
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

model_name = "bert-base-uncased"

# Load HF model
print("Loading HuggingFace BERT...")
hf_model = transformers.BertModel.from_pretrained(model_name)
hf_config = hf_model.config

# Save to safetensors
safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
if not safetensors_path.exists():
    from safetensors.torch import save_file

    save_file(hf_model.state_dict(), str(safetensors_path))

# Create TTML model
print("Creating TTML BERT...")
ttml_config = ttml.models.bert.BertConfig()
ttml_config.vocab_size = hf_config.vocab_size
ttml_config.max_sequence_length = 32
ttml_config.embedding_dim = hf_config.hidden_size
ttml_config.intermediate_size = hf_config.intermediate_size
ttml_config.num_heads = hf_config.num_attention_heads
ttml_config.num_blocks = hf_config.num_hidden_layers
ttml_config.dropout_prob = 0.0
ttml_config.layer_norm_eps = hf_config.layer_norm_eps
ttml_config.use_token_type_embeddings = True
ttml_config.use_pooler = False

bert = ttml.models.bert.create(ttml_config)
bert.load_model_from_safetensors(str(safetensors_path))

# Get weights
params = bert.parameters()

print("\n" + "=" * 80)
print("INSPECTING LAYER 0 QKV WEIGHTS")
print("=" * 80)

# TTML QKV weight
qkv_weight_ttml = params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"].to_numpy()
print(f"\nTTML QKV weight shape: {qkv_weight_ttml.shape}")
print(f"TTML QKV weight (first 10 values, flat): {qkv_weight_ttml.flat[:10]}")

# Reshape to 2D
qkv_weight_ttml_2d = qkv_weight_ttml.reshape(2304, 768)
print(f"\nTTML QKV weight (2D): {qkv_weight_ttml_2d.shape}")
print(f"TTML QKV [0, 0]: {qkv_weight_ttml_2d[0, 0]}")
print(f"TTML QKV [0, :5]: {qkv_weight_ttml_2d[0, :5]}")
print(f"TTML QKV [:5, 0]: {qkv_weight_ttml_2d[:5, 0]}")

# HF Q, K, V weights
hf_q = hf_model.encoder.layer[0].attention.self.query.weight.detach().numpy()
hf_k = hf_model.encoder.layer[0].attention.self.key.weight.detach().numpy()
hf_v = hf_model.encoder.layer[0].attention.self.value.weight.detach().numpy()

print(f"\nHF Q weight shape: {hf_q.shape}")
print(f"HF Q [0, 0]: {hf_q[0, 0]}")
print(f"HF Q [0, :5]: {hf_q[0, :5]}")
print(f"HF Q [:5, 0]: {hf_q[:5, 0]}")

print(f"\nHF K [0, 0]: {hf_k[0, 0]}")
print(f"HF V [0, 0]: {hf_v[0, 0]}")

# Test different concatenations
print("\n" + "=" * 80)
print("TESTING CONCATENATION PATTERNS")
print("=" * 80)

# Pattern 1: cat(Q, K, V, dim=0)
qkv_p1 = np.concatenate([hf_q, hf_k, hf_v], axis=0)
print(f"\nPattern 1 - cat(Q,K,V, dim=0): {qkv_p1.shape}")
print(f"P1 [0, 0] (should be Q[0,0]): {qkv_p1[0, 0]} (expected: {hf_q[0, 0]})")
print(f"P1 [768, 0] (should be K[0,0]): {qkv_p1[768, 0]} (expected: {hf_k[0, 0]})")
print(f"P1 [1536, 0] (should be V[0,0]): {qkv_p1[1536, 0]} (expected: {hf_v[0, 0]})")

# What does TTML actually have?
print(f"\nTTML [0, 0]: {qkv_weight_ttml_2d[0, 0]}")
print(f"TTML [768, 0]: {qkv_weight_ttml_2d[768, 0]}")
print(f"TTML [1536, 0]: {qkv_weight_ttml_2d[1536, 0]}")

# Check if TTML matches Pattern 1
match_p1 = np.allclose(qkv_weight_ttml_2d, qkv_p1, atol=1e-5)
print(f"\nTTML matches Pattern 1: {match_p1}")
if not match_p1:
    diff = np.abs(qkv_weight_ttml_2d - qkv_p1)
    print(f"Diff stats: mean={diff.mean():.6e}, max={diff.max():.6e}")
    print(f"First 10 diffs: {diff.flat[:10]}")

# Try row-major vs column-major
print("\n" + "=" * 80)
print("CHECKING MEMORY LAYOUT")
print("=" * 80)

# Maybe the issue is row-major vs column-major?
qkv_p1_f = np.asfortranarray(qkv_p1)  # Column-major
qkv_p1_c = np.ascontiguousarray(qkv_p1)  # Row-major

print(f"P1 C-contiguous (row-major): {qkv_p1_c.flags['C_CONTIGUOUS']}")
print(f"P1 F-contiguous (col-major): {qkv_p1_f.flags['F_CONTIGUOUS']}")

# Try transposing and flattening in different orders
qkv_p1_flat_c = qkv_p1.flatten("C")  # Row-major flatten
qkv_p1_flat_f = qkv_p1.flatten("F")  # Column-major flatten

qkv_ttml_flat = qkv_weight_ttml.flatten()

print(f"\nP1 flat C (first 10): {qkv_p1_flat_c[:10]}")
print(f"P1 flat F (first 10): {qkv_p1_flat_f[:10]}")
print(f"TTML flat (first 10): {qkv_ttml_flat[:10]}")

match_c = np.allclose(qkv_ttml_flat, qkv_p1_flat_c, atol=1e-5)
match_f = np.allclose(qkv_ttml_flat, qkv_p1_flat_f, atol=1e-5)

print(f"\nTTML flat matches P1 flat (row-major): {match_c}")
print(f"TTML flat matches P1 flat (col-major): {match_f}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
