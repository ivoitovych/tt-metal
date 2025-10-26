#!/usr/bin/env python3
"""
Debug Attention Masking - Inspect intermediate values

This script extracts intermediate values from the attention computation
to see exactly where the masking is going wrong.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def extract_real_qkv_and_mask():
    """Extract REAL Q, K, V tensors and attention mask from BERT."""
    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    text = "The quick brown fox jumps."
    encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

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

        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()

        return {
            "q": q.numpy().astype(np.float32),
            "k": k.numpy().astype(np.float32),
            "v": v.numpy().astype(np.float32),
            "mask": attn_mask.numpy().astype(np.float32),
            "attention_mask_1d": attention_mask.numpy().astype(np.float32),
        }


def compute_reference_attention(data):
    """Compute reference attention with mask."""
    q_torch = torch.from_numpy(data["q"])
    k_torch = torch.from_numpy(data["k"])
    v_torch = torch.from_numpy(data["v"])
    mask_torch = torch.from_numpy(data["mask"])

    head_dim = q_torch.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)

    # Compute attention scores
    attn_scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale

    print(f"\n{'='*80}")
    print(f"REFERENCE IMPLEMENTATION (PyTorch)")
    print(f"{'='*80}")
    print(f"Attention scores (before masking):")
    print(f"  Shape: {attn_scores.shape}")
    print(f"  Mean: {attn_scores.mean():.6f}, Std: {attn_scores.std():.6f}")
    print(f"  Min: {attn_scores.min():.6f}, Max: {attn_scores.max():.6f}")

    # Show a sample row from the attention scores (first head, first query position)
    print(f"\nSample attention scores (head=0, query_pos=0) - first 16 positions:")
    print(f"  {attn_scores[0, 0, 0, :16].numpy()}")

    print(f"\nMask shape: {mask_torch.shape}")
    print(f"Mask (first 16 positions): {mask_torch[0, 0, 0, :16].numpy()}")

    # Apply mask
    attn_scores_masked = attn_scores.masked_fill(mask_torch == 0, -1e9)

    print(f"\nAttention scores (after masking):")
    print(f"  Mean: {attn_scores_masked.mean():.6f}, Std: {attn_scores_masked.std():.6f}")
    print(f"  Min: {attn_scores_masked.min():.6f}, Max: {attn_scores_masked.max():.6f}")
    print(f"\nSample masked scores (head=0, query_pos=0) - first 16 positions:")
    print(f"  {attn_scores_masked[0, 0, 0, :16].numpy()}")

    # Apply softmax
    attn_weights = torch.softmax(attn_scores_masked, dim=-1)

    print(f"\nAttention weights (after softmax):")
    print(f"  Mean: {attn_weights.mean():.6f}, Std: {attn_weights.std():.6f}")
    print(f"  Min: {attn_weights.min():.6f}, Max: {attn_weights.max():.6f}")
    print(f"  Sum (should be ~1.0 per row): {attn_weights[0, 0, 0, :].sum():.6f}")
    print(f"\nSample weights (head=0, query_pos=0) - first 16 positions:")
    print(f"  {attn_weights[0, 0, 0, :16].numpy()}")

    # Compute output
    output_ref = torch.matmul(attn_weights, v_torch).numpy()

    return output_ref, attn_scores.numpy(), attn_scores_masked.numpy(), attn_weights.numpy()


def compute_ttml_attention(data):
    """Compute TTML attention with mask and inspect intermediates."""
    q_ttml = ttml.autograd.Tensor.from_numpy(data["q"])
    k_ttml = ttml.autograd.Tensor.from_numpy(data["k"])
    v_ttml = ttml.autograd.Tensor.from_numpy(data["v"])
    mask_ttml = ttml.autograd.Tensor.from_numpy(data["mask"])

    print(f"\n{'='*80}")
    print(f"TTML IMPLEMENTATION")
    print(f"{'='*80}")
    print(f"Input shapes:")
    print(f"  Q: {q_ttml.shape()}")
    print(f"  K: {k_ttml.shape()}")
    print(f"  V: {v_ttml.shape()}")
    print(f"  Mask: {mask_ttml.shape()}")

    # Unfortunately, we can't easily extract intermediate values from within the C++ implementation
    # But we can compute what SHOULD happen step by step

    # Let's manually compute what the masking SHOULD produce
    q_np = data["q"]
    k_np = data["k"]
    mask_np = data["mask"]

    head_dim = q_np.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)

    # Q @ K^T scaled
    qk = np.matmul(q_np, k_np.transpose(0, 1, 3, 2)) * scale

    print(f"\nManual computation (what TTML should do):")
    print(f"QK scores (before masking):")
    print(f"  Shape: {qk.shape}")
    print(f"  Mean: {qk.mean():.6f}, Std: {qk.std():.6f}")
    print(f"  Sample (head=0, query_pos=0) - first 16: {qk[0, 0, 0, :16]}")

    # Apply mask using the FIXED formula from the C++ code:
    # result = mask * qk + (mask - 1.0) * (+1e9)
    masked_part = mask_np * qk  # Where mask=1, this is qk. Where mask=0, this is 0.
    inverted_mask = mask_np - 1.0  # Where mask=1, this is 0. Where mask=0, this is -1.
    large_negative = inverted_mask * (+1e9)  # Where mask=1, this is 0. Where mask=0, this is -1e9 (FIXED).

    qk_masked_manual = masked_part + large_negative

    print(f"\nManual masking breakdown:")
    print(f"  mask_np[0, 0, 0, :16]: {mask_np[0, 0, 0, :16]}")
    print(f"  masked_part[0, 0, 0, :16]: {masked_part[0, 0, 0, :16]}")
    print(f"  inverted_mask[0, 0, 0, :16]: {inverted_mask[0, 0, 0, :16]}")
    print(f"  large_negative[0, 0, 0, :16]: {large_negative[0, 0, 0, :16]}")
    print(f"  qk_masked_manual[0, 0, 0, :16]: {qk_masked_manual[0, 0, 0, :16]}")

    # Now run actual TTML
    output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)

    output_ttml_np = output_ttml.to_numpy()

    print(f"\nTTML Output:")
    print(f"  Shape: {output_ttml_np.shape}")
    print(f"  Mean: {output_ttml_np.mean():.6f}, Std: {output_ttml_np.std():.6f}")
    print(f"  Min: {output_ttml_np.min():.6f}, Max: {output_ttml_np.max():.6f}")

    return output_ttml_np


def main():
    print(f"{'#'*80}")
    print(f"DEBUGGING ATTENTION MASKING")
    print(f"{'#'*80}")

    # Extract real data
    data = extract_real_qkv_and_mask()
    print(f"\nExtracted data:")
    print(f"  Q shape: {data['q'].shape}")
    print(f"  K shape: {data['k'].shape}")
    print(f"  V shape: {data['v'].shape}")
    print(f"  Mask shape: {data['mask'].shape}")
    print(f"  Attention mask (1D): {data['attention_mask_1d'][0]}")

    # Compute reference
    output_ref, scores_before, scores_after, weights_ref = compute_reference_attention(data)

    # Compute TTML
    output_ttml = compute_ttml_attention(data)

    # Compare
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    diff = np.abs(output_ref - output_ttml)
    print(f"Reference output - mean: {output_ref.mean():.6f}, std: {output_ref.std():.6f}")
    print(f"TTML output      - mean: {output_ttml.mean():.6f}, std: {output_ttml.std():.6f}")
    print(f"Absolute diff    - mean: {diff.mean():.6e}, max: {diff.max():.6e}")

    # Correlation
    ref_flat = output_ref.flatten()
    ttml_flat = output_ttml.flatten()
    corr = np.corrcoef(ref_flat, ttml_flat)
    pcc = corr[0, 1] if corr.shape == (2, 2) else 0.0
    print(f"PCC: {pcc:.6f}")


if __name__ == "__main__":
    main()
