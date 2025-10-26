#!/usr/bin/env python3
"""
Test if real BERT Q,K,V data survives round-trip through TTML tensors.
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


def main():
    print("\n" + "=" * 80)
    print("TEST: Real BERT Data Round-Trip Through TTML Tensors")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    text = "Hello world"
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

        q_np_orig = q.numpy().astype(np.float32)
        k_np_orig = k.numpy().astype(np.float32)
        v_np_orig = v.numpy().astype(np.float32)

        print(f"\nOriginal data (float32):")
        print(f"  Q: mean={q_np_orig.mean():.6f}, std={q_np_orig.std():.6f}")
        print(f"  K: mean={k_np_orig.mean():.6f}, std={k_np_orig.std():.6f}")
        print(f"  V: mean={v_np_orig.mean():.6f}, std={v_np_orig.std():.6f}")
        print(f"  Q first 10 values: {q_np_orig.flatten()[:10]}")

        # Convert to TTML and back
        q_ttml = ttml.autograd.Tensor.from_numpy(q_np_orig)
        k_ttml = ttml.autograd.Tensor.from_numpy(k_np_orig)
        v_ttml = ttml.autograd.Tensor.from_numpy(v_np_orig)

        q_np_back = q_ttml.to_numpy()
        k_np_back = k_ttml.to_numpy()
        v_np_back = v_ttml.to_numpy()

        print(f"\nAfter round-trip through TTML:")
        print(f"  Q: mean={q_np_back.mean():.6f}, std={q_np_back.std():.6f}")
        print(f"  K: mean={k_np_back.mean():.6f}, std={k_np_back.std():.6f}")
        print(f"  V: mean={v_np_back.mean():.6f}, std={v_np_back.std():.6f}")
        print(f"  Q first 10 values: {q_np_back.flatten()[:10]}")

        # Compute precision loss
        q_diff = np.abs(q_np_orig - q_np_back)
        k_diff = np.abs(k_np_orig - k_np_back)
        v_diff = np.abs(v_np_orig - v_np_back)

        print(f"\nPrecision loss:")
        print(f"  Q: mean={q_diff.mean():.6e}, max={q_diff.max():.6e}")
        print(f"  K: mean={k_diff.mean():.6e}, max={k_diff.max():.6e}")
        print(f"  V: mean={v_diff.mean():.6e}, max={v_diff.max():.6e}")

        q_pcc = compute_pcc(q_np_orig, q_np_back)
        k_pcc = compute_pcc(k_np_orig, k_np_back)
        v_pcc = compute_pcc(v_np_orig, v_np_back)

        print(f"\nPCC:")
        print(f"  Q: {q_pcc:.6f}")
        print(f"  K: {k_pcc:.6f}")
        print(f"  V: {v_pcc:.6f}")

        if q_pcc > 0.999 and k_pcc > 0.999 and v_pcc > 0.999:
            print(f"\n✅ Round-trip preserves data well")
        else:
            print(f"\n⚠️  Significant precision loss in round-trip")

        # Now test with synthetic random data for comparison
        print(f"\n{'=' * 80}")
        print("COMPARISON: Synthetic Random Data")
        print(f"{'=' * 80}")

        np.random.seed(42)
        q_synth = np.random.randn(*q_np_orig.shape).astype(np.float32)
        q_synth_ttml = ttml.autograd.Tensor.from_numpy(q_synth)
        q_synth_back = q_synth_ttml.to_numpy()

        q_synth_diff = np.abs(q_synth - q_synth_back)
        q_synth_pcc = compute_pcc(q_synth, q_synth_back)

        print(f"Synthetic Q precision loss: mean={q_synth_diff.mean():.6e}, max={q_synth_diff.max():.6e}")
        print(f"Synthetic Q PCC: {q_synth_pcc:.6f}")

        print(f"\n{'=' * 80}")
        print("ANALYSIS:")
        print(f"{'=' * 80}")
        print(f"Real BERT Q precision loss:  {q_diff.mean():.6e}")
        print(f"Synthetic Q precision loss:  {q_synth_diff.mean():.6e}")
        print(f"Ratio (real/synthetic):      {q_diff.mean() / q_synth_diff.mean():.2f}x")


if __name__ == "__main__":
    main()
