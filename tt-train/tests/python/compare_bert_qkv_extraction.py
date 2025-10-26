#!/usr/bin/env python3
"""
Compare Q, K, V extraction methods to ensure we're getting the right values.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def main():
    print("=" * 80)
    print("COMPARING BERT Q, K, V EXTRACTION METHODS")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    text = "The quick brown fox jumps."
    encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print(f"\nInput text: '{text}'")
    print(f"Attention mask: {attention_mask[0, :10].tolist()}...")

    with torch.no_grad():
        # Get embeddings
        embeddings = hf_model.embeddings(input_ids)
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Embeddings stats: mean={embeddings.mean():.6f}, std={embeddings.std():.6f}")

        # Method 1: Extract Q, K, V manually (what we do in tests)
        first_layer = hf_model.encoder.layer[0]
        attention = first_layer.attention.self

        batch_size, seq_len, hidden_size = embeddings.shape
        num_heads = hf_model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        q_manual = attention.query(embeddings)
        k_manual = attention.key(embeddings)
        v_manual = attention.value(embeddings)

        print(f"\n--- METHOD 1: Manual Q, K, V extraction ---")
        print(f"Q (before reshape): shape={q_manual.shape}, mean={q_manual.mean():.6f}, std={q_manual.std():.6f}")
        print(f"K (before reshape): shape={k_manual.shape}, mean={k_manual.mean():.6f}, std={k_manual.std():.6f}")
        print(f"V (before reshape): shape={v_manual.shape}, mean={v_manual.mean():.6f}, std={v_manual.std():.6f}")

        # Reshape to multi-head format
        q_manual_mh = q_manual.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k_manual_mh = k_manual.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v_manual_mh = v_manual.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        print(f"Q (multi-head): shape={q_manual_mh.shape}")
        print(f"K (multi-head): shape={k_manual_mh.shape}")
        print(f"V (multi-head): shape={v_manual_mh.shape}")

        # Method 2: Let HuggingFace compute attention internally
        print(f"\n--- METHOD 2: Full HuggingFace attention ---")

        # Prepare attention mask (HF format: [batch, 1, 1, seq_len])
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Run attention through HuggingFace
        attn_output_hf = first_layer.attention(embeddings, attention_mask=extended_attention_mask)[0]

        print(f"HF attention output: shape={attn_output_hf.shape}")
        print(f"  mean={attn_output_hf.mean():.6f}, std={attn_output_hf.std():.6f}")

        # Method 3: Manually compute attention with our Q, K, V
        print(f"\n--- METHOD 3: Manual attention with extracted Q, K, V ---")

        scale = 1.0 / np.sqrt(head_dim)
        attn_scores_manual = torch.matmul(q_manual_mh, k_manual_mh.transpose(-2, -1)) * scale

        print(
            f"Attention scores (before mask): mean={attn_scores_manual.mean():.6f}, std={attn_scores_manual.std():.6f}"
        )

        # Apply mask
        mask_for_scores = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
        attn_scores_masked = attn_scores_manual.masked_fill(mask_for_scores == 0, -1e9)

        print(
            f"Attention scores (after mask): mean={attn_scores_masked.mean():.6f}, std={attn_scores_masked.std():.6f}"
        )

        # Softmax
        attn_weights_manual = torch.softmax(attn_scores_masked, dim=-1)

        print(f"Attention weights: mean={attn_weights_manual.mean():.6f}, std={attn_weights_manual.std():.6f}")
        print(f"  First position attention (should sum to 1): {attn_weights_manual[0, 0, 0, :].sum():.6f}")

        # Multiply by V
        context_layer = torch.matmul(attn_weights_manual, v_manual_mh)

        # Reshape back
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        print(f"Context layer (before output projection): shape={context_layer.shape}")
        print(f"  mean={context_layer.mean():.6f}, std={context_layer.std():.6f}")

        # CRITICAL: Apply output projection (this was missing!)
        attn_output_manual = first_layer.attention.output.dense(context_layer)

        print(f"Manual attention output (after output projection): shape={attn_output_manual.shape}")
        print(f"  mean={attn_output_manual.mean():.6f}, std={attn_output_manual.std():.6f}")

        # Compare Method 2 vs Method 3
        print(f"\n--- COMPARISON: HuggingFace vs Manual ---")
        diff = torch.abs(attn_output_hf - attn_output_manual)
        print(f"Mean abs diff: {diff.mean():.6e}")
        print(f"Max abs diff: {diff.max():.6e}")

        hf_flat = attn_output_hf.flatten().numpy()
        manual_flat = attn_output_manual.flatten().numpy()
        corr = np.corrcoef(hf_flat, manual_flat)
        pcc = corr[0, 1] if corr.shape == (2, 2) else 0.0
        print(f"PCC: {pcc:.6f}")

        if pcc > 0.999:
            print(f"✅ Manual extraction matches HuggingFace!")
        else:
            print(f"❌ Manual extraction DOESN'T match HuggingFace!")

        # Now test with TTML
        print(f"\n--- METHOD 4: TTML attention ---")

        q_np = q_manual_mh.numpy().astype(np.float32)
        k_np = k_manual_mh.numpy().astype(np.float32)
        v_np = v_manual_mh.numpy().astype(np.float32)
        mask_np = mask_for_scores.numpy().astype(np.float32)

        q_ttml = ttml.autograd.Tensor.from_numpy(q_np)
        k_ttml = ttml.autograd.Tensor.from_numpy(k_np)
        v_ttml = ttml.autograd.Tensor.from_numpy(v_np)
        mask_ttml = ttml.autograd.Tensor.from_numpy(mask_np)

        attn_output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(
            q_ttml, k_ttml, v_ttml, mask=mask_ttml
        )

        # Convert back and reshape
        attn_output_ttml_np = attn_output_ttml.to_numpy()

        # TTML output is [batch, heads, seq, head_dim], need to convert to [batch, seq, hidden]
        attn_output_ttml_reshaped = attn_output_ttml_np.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)

        print(f"TTML attention output: shape={attn_output_ttml_reshaped.shape}")
        print(f"  mean={attn_output_ttml_reshaped.mean():.6f}, std={attn_output_ttml_reshaped.std():.6f}")

        # Compare TTML vs Manual (PyTorch) - comparing CONTEXT LAYERS (before output projection)
        print(f"\n--- COMPARISON: TTML vs Manual PyTorch (context layer only) ---")
        context_layer_np = context_layer.numpy()
        diff_ttml = np.abs(context_layer_np - attn_output_ttml_reshaped)
        print(f"Mean abs diff: {diff_ttml.mean():.6e}")
        print(f"Max abs diff: {diff_ttml.max():.6e}")

        ttml_flat = attn_output_ttml_reshaped.flatten()
        context_flat = context_layer_np.flatten()
        corr_ttml = np.corrcoef(context_flat, ttml_flat)
        pcc_ttml = corr_ttml[0, 1] if corr_ttml.shape == (2, 2) else 0.0
        print(f"PCC: {pcc_ttml:.6f}")

        if pcc_ttml > 0.99:
            print(f"✅ TTML matches PyTorch reference!")
        else:
            print(f"❌ TTML DOESN'T match PyTorch reference!")
            print(f"\nFirst 10 values comparison:")
            print(f"  PyTorch (context): {context_flat[:10]}")
            print(f"  TTML:              {ttml_flat[:10]}")
            print(f"  Diff:              {np.abs(context_flat[:10] - ttml_flat[:10])}")


if __name__ == "__main__":
    main()
