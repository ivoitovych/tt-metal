#!/usr/bin/env python3
"""Inspect what's actually in the safetensors file."""

import numpy as np
from safetensors import safe_open

safetensors_path = "/tmp/bert-base-uncased.safetensors"

print("Opening safetensors file...")
with safe_open(safetensors_path, framework="numpy") as f:
    # List all keys
    keys = f.keys()
    print(f"Total keys: {len(list(keys))}")
    print("\nLayer 0 attention keys:")
    for key in f.keys():
        if "layer.0.attention" in key:
            print(f"  {key}")

    # Check Q, K, V for layer 0
    q_key = "encoder.layer.0.attention.self.query.weight"
    k_key = "encoder.layer.0.attention.self.key.weight"
    v_key = "encoder.layer.0.attention.self.value.weight"

    print(f"\nKey: {q_key}")
    q_weight = f.get_tensor(q_key)
    print(f"Shape: {q_weight.shape}")
    print(f"Dtype: {q_weight.dtype}")
    print(f"Q[0,0]: {q_weight[0, 0]}")
    print(f"Q[0,:5]: {q_weight[0, :5]}")
    print(f"Q[:5,0]: {q_weight[:5, 0]}")

    print(f"\nKey: {k_key}")
    k_weight = f.get_tensor(k_key)
    print(f"Shape: {k_weight.shape}")
    print(f"K[0,0]: {k_weight[0, 0]}")

    print(f"\nKey: {v_key}")
    v_weight = f.get_tensor(v_key)
    print(f"Shape: {v_weight.shape}")
    print(f"V[0,0]: {v_weight[0, 0]}")

    # Try concatenating
    qkv_concat = np.concatenate([q_weight, k_weight, v_weight], axis=0)
    print(f"\nConcatenated QKV shape: {qkv_concat.shape}")
    print(f"QKV[0,0] (Q): {qkv_concat[0, 0]}")
    print(f"QKV[768,0] (K): {qkv_concat[768, 0]}")
    print(f"QKV[1536,0] (V): {qkv_concat[1536, 0]}")

    # Save raw concatenated for comparison
    np.save("/tmp/qkv_expected.npy", qkv_concat)
    print("\nSaved expected QKV to /tmp/qkv_expected.npy")
