#!/usr/bin/env python3
"""
Test if using ROW_MAJOR layout instead of TILE layout fixes the real BERT data issue.
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
    print("TEST: Real BERT Data with ROW_MAJOR vs TILE Layout")
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

        print(f"\nOriginal Q:")
        print(f"  mean={q_np_orig.mean():.6f}, std={q_np_orig.std():.6f}")
        print(f"  First 10 values: {q_np_orig.flatten()[:10]}")

        # Test 1: Default TILE layout
        print(f"\n{'=' * 80}")
        print("Test 1: TILE Layout (default)")
        print(f"{'=' * 80}")

        q_tile = ttml.autograd.Tensor.from_numpy(q_np_orig)  # Default is TILE
        q_tile_back = q_tile.to_numpy()

        q_tile_diff = np.abs(q_np_orig - q_tile_back)
        q_tile_pcc = compute_pcc(q_np_orig, q_tile_back)

        print(f"After round-trip (TILE):")
        print(f"  mean={q_tile_back.mean():.6f}, std={q_tile_back.std():.6f}")
        print(f"  First 10 values: {q_tile_back.flatten()[:10]}")
        print(f"  Mean abs diff: {q_tile_diff.mean():.6e}")
        print(f"  Max abs diff: {q_tile_diff.max():.6e}")
        print(f"  PCC: {q_tile_pcc:.6f}")

        # Test 2: ROW_MAJOR layout
        print(f"\n{'=' * 80}")
        print("Test 2: ROW_MAJOR Layout")
        print(f"{'=' * 80}")

        q_row = ttml.autograd.Tensor.from_numpy(q_np_orig, layout=ttml.Layout.ROW_MAJOR)
        q_row_back = q_row.to_numpy()

        q_row_diff = np.abs(q_np_orig - q_row_back)
        q_row_pcc = compute_pcc(q_np_orig, q_row_back)

        print(f"After round-trip (ROW_MAJOR):")
        print(f"  mean={q_row_back.mean():.6f}, std={q_row_back.std():.6f}")
        print(f"  First 10 values: {q_row_back.flatten()[:10]}")
        print(f"  Mean abs diff: {q_row_diff.mean():.6e}")
        print(f"  Max abs diff: {q_row_diff.max():.6e}")
        print(f"  PCC: {q_row_pcc:.6f}")

        print(f"\n{'=' * 80}")
        print("SUMMARY:")
        print(f"{'=' * 80}")
        print(f"TILE layout PCC:      {q_tile_pcc:.6f} {'❌' if q_tile_pcc < 0.99 else '✅'}")
        print(f"ROW_MAJOR layout PCC: {q_row_pcc:.6f} {'❌' if q_row_pcc < 0.99 else '✅'}")

        if q_row_pcc > 0.99 and q_tile_pcc < 0.99:
            print(f"\n✅ ROW_MAJOR layout fixes the issue!")
            print(f"   The problem is in the TILE layout conversion (tilize/untilize)")
        elif q_row_pcc < 0.99:
            print(f"\n❌ ROW_MAJOR layout doesn't fix the issue")
            print(f"   The problem is elsewhere")


if __name__ == "__main__":
    main()
