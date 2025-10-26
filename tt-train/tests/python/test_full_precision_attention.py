#!/usr/bin/env python3
"""
Test TTML attention with FULL precision (not HALF precision).

The issue discovered: TTML uses HALF precision (bfloat16/float16) by default,
which causes precision loss with real BERT weights.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x, y):
    """Compute Pearson correlation coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


def test_precision_settings():
    """Test if we can control precision settings."""
    print("\n" + "=" * 80)
    print("TESTING PRECISION SETTINGS")
    print("=" * 80)

    # Check if PreferredPrecision enum is accessible
    try:
        print("\nAvailable precision modes:")
        precision_enum = ttml.autograd.PreferredPrecision
        for attr in dir(precision_enum):
            if not attr.startswith("_"):
                print(f"  - {attr}")
    except Exception as e:
        print(f"Could not access PreferredPrecision: {e}")

    # Test round-trip with different values
    test_data = np.array([[0.123456789, 0.333333333, 0.987654321]], dtype=np.float32)
    print(f"\nOriginal values: {test_data[0]}")

    tensor = ttml.autograd.Tensor.from_numpy(test_data)
    result = tensor.to_numpy()
    print(f"Round-trip values: {result[0]}")

    diff = np.abs(test_data - result)
    print(f"Max precision loss: {diff.max():.10e}")


def test_attention_with_precision_hint():
    """Test attention with explicit precision control if possible."""
    print("\n" + "=" * 80)
    print("TEST: Real BERT Attention with Precision Control")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    text = "Hello world test"
    max_length = 8

    encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    input_ids = encoded["input_ids"]

    with torch.no_grad():
        embeddings = hf_model.embeddings(input_ids)
        first_layer = hf_model.encoder.layer[0]
        attention = first_layer.attention.self

        batch_size, seq_len, hidden_size = embeddings.shape
        num_heads = hf_model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        q = attention.query(embeddings)
        k = attention.key(embeddings)
        v = attention.value(embeddings)

        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Reference (PyTorch, full precision)
        scale = 1.0 / np.sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output_ref = torch.matmul(attn_weights, v).numpy()

        q_np = q.numpy().astype(np.float32)
        k_np = k.numpy().astype(np.float32)
        v_np = v.numpy().astype(np.float32)

        # Test with TTML (default precision)
        print("\n1. Testing with DEFAULT precision:")
        q_ttml = ttml.autograd.Tensor.from_numpy(q_np)
        k_ttml = ttml.autograd.Tensor.from_numpy(k_np)
        v_ttml = ttml.autograd.Tensor.from_numpy(v_np)

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=None)
        output_ttml_np = output_ttml.to_numpy()

        pcc_default = compute_pcc(output_ref, output_ttml_np)
        print(f"   PCC: {pcc_default:.6f}")

        # Try to check if we can inspect or set precision
        print("\n2. Inspecting tensor internals:")
        print(f"   Q tensor type: {type(q_ttml)}")
        print(f"   Q tensor attributes: {[attr for attr in dir(q_ttml) if not attr.startswith('_')]}")


if __name__ == "__main__":
    test_precision_settings()
    test_attention_with_precision_hint()
