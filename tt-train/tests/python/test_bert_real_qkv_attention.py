#!/usr/bin/env python3
"""
Test TTML attention with real BERT Q, K, V data.

This test extracts real learned Q, K, V values from prajjwal1/bert-tiny
and tests them directly with TTML's scaled_dot_product_attention to isolate
whether the issue is in the attention computation itself.
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


def test_real_bert_attention_no_mask():
    """Test with real BERT data WITHOUT masking (all real tokens)."""
    print("\n" + "=" * 80)
    print("TEST: Real BERT Attention - NO MASKING (All Real Tokens)")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    # Use a text that fills the sequence length (no padding)
    text = "The quick brown fox jumps over the lazy dog and runs through the forest"
    max_length = 16

    encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    num_real_tokens = attention_mask.sum().item()
    total_tokens = attention_mask.numel()
    padding_pct = 100.0 * (1.0 - num_real_tokens / total_tokens)

    print(f"\nText: '{text}'")
    print(f"Sequence length: {max_length}")
    print(f"Real tokens: {num_real_tokens}/{total_tokens} ({padding_pct:.1f}% padding)")
    print(f"Attention mask: {attention_mask[0].tolist()}")

    with torch.no_grad():
        # Get embeddings
        embeddings = hf_model.embeddings(input_ids)

        # Extract Q, K, V from first attention layer
        first_layer = hf_model.encoder.layer[0]
        attention = first_layer.attention.self

        batch_size, seq_len, hidden_size = embeddings.shape
        num_heads = hf_model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        q = attention.query(embeddings)
        k = attention.key(embeddings)
        v = attention.value(embeddings)

        # Reshape to multi-head format [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        print(f"\nQ shape: {q.shape}")
        print(f"K shape: {k.shape}")
        print(f"V shape: {v.shape}")
        print(f"Q stats: mean={q.mean():.6f}, std={q.std():.6f}")
        print(f"K stats: mean={k.mean():.6f}, std={k.std():.6f}")
        print(f"V stats: mean={v.mean():.6f}, std={v.std():.6f}")

        # Compute reference output using PyTorch (WITHOUT mask)
        scale = 1.0 / np.sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output_ref = torch.matmul(attn_weights, v)

        print(f"\nReference output shape: {output_ref.shape}")
        print(f"Reference output stats: mean={output_ref.mean():.6f}, std={output_ref.std():.6f}")

        # Convert to numpy
        q_np = q.numpy().astype(np.float32)
        k_np = k.numpy().astype(np.float32)
        v_np = v.numpy().astype(np.float32)
        output_ref_np = output_ref.numpy().astype(np.float32)

        # Test with TTML (NO mask)
        q_ttml = ttml.autograd.Tensor.from_numpy(q_np)
        k_ttml = ttml.autograd.Tensor.from_numpy(k_np)
        v_ttml = ttml.autograd.Tensor.from_numpy(v_np)

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=None)
        output_ttml_np = output_ttml.to_numpy()

        print(f"\nTTML output shape: {output_ttml_np.shape}")
        print(f"TTML output stats: mean={output_ttml_np.mean():.6f}, std={output_ttml_np.std():.6f}")

        # Compare
        diff = np.abs(output_ref_np - output_ttml_np)
        pcc = compute_pcc(output_ref_np, output_ttml_np)

        print(f"\n{'=' * 80}")
        print("RESULTS:")
        print(f"{'=' * 80}")
        print(f"Mean abs diff: {diff.mean():.6e}")
        print(f"Max abs diff: {diff.max():.6e}")
        print(f"PCC: {pcc:.6f}")

        threshold = 0.99
        if pcc >= threshold:
            print(f"✅ PASS: Real BERT attention works correctly (PCC={pcc:.6f} >= {threshold})")
        else:
            print(f"❌ FAIL: Real BERT attention has issues (PCC={pcc:.6f} < {threshold})")
            print(f"\nFirst 10 reference values: {output_ref_np.flatten()[:10]}")
            print(f"First 10 TTML values:      {output_ttml_np.flatten()[:10]}")

        return pcc >= threshold


def test_real_bert_attention_with_mask():
    """Test with real BERT data WITH masking (padding tokens)."""
    print("\n" + "=" * 80)
    print("TEST: Real BERT Attention - WITH MASKING (Padding Tokens)")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    # Use shorter text to have padding
    text = "The quick brown fox"
    max_length = 16

    encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    num_real_tokens = attention_mask.sum().item()
    total_tokens = attention_mask.numel()
    padding_pct = 100.0 * (1.0 - num_real_tokens / total_tokens)

    print(f"\nText: '{text}'")
    print(f"Sequence length: {max_length}")
    print(f"Real tokens: {num_real_tokens}/{total_tokens} ({padding_pct:.1f}% padding)")
    print(f"Attention mask: {attention_mask[0].tolist()}")

    with torch.no_grad():
        # Get embeddings
        embeddings = hf_model.embeddings(input_ids)

        # Extract Q, K, V from first attention layer
        first_layer = hf_model.encoder.layer[0]
        attention = first_layer.attention.self

        batch_size, seq_len, hidden_size = embeddings.shape
        num_heads = hf_model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        q = attention.query(embeddings)
        k = attention.key(embeddings)
        v = attention.value(embeddings)

        # Reshape to multi-head format [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        print(f"\nQ shape: {q.shape}")
        print(f"K shape: {k.shape}")
        print(f"V shape: {v.shape}")

        # Compute reference output using PyTorch (WITH mask)
        scale = 1.0 / np.sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply mask
        mask_for_scores = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
        attn_scores_masked = attn_scores.masked_fill(mask_for_scores == 0, -1e9)
        attn_weights = torch.softmax(attn_scores_masked, dim=-1)
        output_ref = torch.matmul(attn_weights, v)

        print(f"\nReference output shape: {output_ref.shape}")
        print(f"Reference output stats: mean={output_ref.mean():.6f}, std={output_ref.std():.6f}")

        # Convert to numpy
        q_np = q.numpy().astype(np.float32)
        k_np = k.numpy().astype(np.float32)
        v_np = v.numpy().astype(np.float32)
        mask_np = mask_for_scores.numpy().astype(np.float32)
        output_ref_np = output_ref.numpy().astype(np.float32)

        # Test with TTML (WITH mask)
        q_ttml = ttml.autograd.Tensor.from_numpy(q_np)
        k_ttml = ttml.autograd.Tensor.from_numpy(k_np)
        v_ttml = ttml.autograd.Tensor.from_numpy(v_np)
        mask_ttml = ttml.autograd.Tensor.from_numpy(mask_np)

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)
        output_ttml_np = output_ttml.to_numpy()

        print(f"\nTTML output shape: {output_ttml_np.shape}")
        print(f"TTML output stats: mean={output_ttml_np.mean():.6f}, std={output_ttml_np.std():.6f}")

        # Compare
        diff = np.abs(output_ref_np - output_ttml_np)
        pcc = compute_pcc(output_ref_np, output_ttml_np)

        print(f"\n{'=' * 80}")
        print("RESULTS:")
        print(f"{'=' * 80}")
        print(f"Mean abs diff: {diff.mean():.6e}")
        print(f"Max abs diff: {diff.max():.6e}")
        print(f"PCC: {pcc:.6f}")

        threshold = 0.99
        if pcc >= threshold:
            print(f"✅ PASS: Real BERT attention with masking works correctly (PCC={pcc:.6f} >= {threshold})")
        else:
            print(f"❌ FAIL: Real BERT attention with masking has issues (PCC={pcc:.6f} < {threshold})")
            print(f"\nFirst 10 reference values: {output_ref_np.flatten()[:10]}")
            print(f"First 10 TTML values:      {output_ttml_np.flatten()[:10]}")

        return pcc >= threshold


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("BERT REAL Q, K, V ATTENTION VALIDATION")
    print("#" * 80)

    test1_pass = test_real_bert_attention_no_mask()
    test2_pass = test_real_bert_attention_with_mask()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Test 1 (No masking):   {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (With masking): {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print("=" * 80)

    if test1_pass and test2_pass:
        print("\n✅ ALL TESTS PASSED: TTML attention correctly handles real BERT data")
    else:
        print("\n❌ SOME TESTS FAILED: Issue with real BERT data in TTML attention")
