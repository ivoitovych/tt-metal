#!/usr/bin/env python3
"""
Debug attention computation step-by-step with real BERT data.

This test compares each intermediate step of the attention computation
to identify exactly where the divergence occurs.
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
    print("DEBUGGING ATTENTION: Step-by-Step Intermediate Results")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    # Use a very simple short text for easier debugging
    text = "Hello world"
    max_length = 8

    encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print(f"\nText: '{text}'")
    print(f"Sequence length: {max_length}")
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

        print(f"\n{'=' * 80}")
        print("STEP 0: Input Q, K, V tensors")
        print(f"{'=' * 80}")
        print(f"Q shape: {q.shape} (batch, heads, seq, head_dim)")
        print(f"Q stats: mean={q.mean():.6f}, std={q.std():.6f}, min={q.min():.6f}, max={q.max():.6f}")
        print(f"K stats: mean={k.mean():.6f}, std={k.std():.6f}, min={k.min():.6f}, max={k.max():.6f}")
        print(f"V stats: mean={v.mean():.6f}, std={v.std():.6f}, min={v.min():.6f}, max={v.max():.6f}")
        print(f"First 5 Q values: {q.flatten()[:5].tolist()}")
        print(f"First 5 K values: {k.flatten()[:5].tolist()}")
        print(f"First 5 V values: {v.flatten()[:5].tolist()}")

        # Convert to numpy for TTML
        q_np = q.numpy().astype(np.float32)
        k_np = k.numpy().astype(np.float32)
        v_np = v.numpy().astype(np.float32)

        # STEP 1: Compute Q @ K^T (before scaling)
        print(f"\n{'=' * 80}")
        print("STEP 1: Q @ K^T (before scaling)")
        print(f"{'=' * 80}")

        qk_ref = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq, seq]
        print(f"PyTorch Q@K^T shape: {qk_ref.shape}")
        print(
            f"PyTorch Q@K^T stats: mean={qk_ref.mean():.6f}, std={qk_ref.std():.6f}, min={qk_ref.min():.6f}, max={qk_ref.max():.6f}"
        )
        print(f"First 5 values: {qk_ref.flatten()[:5].tolist()}")

        # Compute the same in numpy to verify
        qk_np = np.matmul(q_np, k_np.transpose(0, 1, 3, 2))
        print(f"NumPy Q@K^T stats: mean={qk_np.mean():.6f}, std={qk_np.std():.6f}")
        pcc_qk = compute_pcc(qk_ref.numpy(), qk_np)
        print(f"PCC (PyTorch vs NumPy): {pcc_qk:.6f}")

        # STEP 2: Scale by 1/sqrt(head_dim)
        print(f"\n{'=' * 80}")
        print("STEP 2: Scaled Q@K^T (after 1/sqrt(head_dim))")
        print(f"{'=' * 80}")

        scale = 1.0 / np.sqrt(head_dim)
        print(f"Scale factor: {scale:.6f} (1/sqrt({head_dim}))")

        qk_scaled_ref = qk_ref * scale
        print(f"PyTorch scaled shape: {qk_scaled_ref.shape}")
        print(
            f"PyTorch scaled stats: mean={qk_scaled_ref.mean():.6f}, std={qk_scaled_ref.std():.6f}, min={qk_scaled_ref.min():.6f}, max={qk_scaled_ref.max():.6f}"
        )
        print(f"First 5 values: {qk_scaled_ref.flatten()[:5].tolist()}")

        # STEP 3: Softmax
        print(f"\n{'=' * 80}")
        print("STEP 3: Softmax over last dimension")
        print(f"{'=' * 80}")

        attn_weights_ref = torch.softmax(qk_scaled_ref, dim=-1)
        print(f"PyTorch attention weights shape: {attn_weights_ref.shape}")
        print(f"PyTorch attention weights stats: mean={attn_weights_ref.mean():.6f}, std={attn_weights_ref.std():.6f}")
        print(f"First 5 values: {attn_weights_ref.flatten()[:5].tolist()}")
        print(f"Sum of first position (should be 1.0): {attn_weights_ref[0, 0, 0, :].sum():.6f}")

        # Check softmax in numpy
        qk_scaled_np = qk_np * scale
        qk_scaled_exp = np.exp(qk_scaled_np - qk_scaled_np.max(axis=-1, keepdims=True))
        attn_weights_np = qk_scaled_exp / qk_scaled_exp.sum(axis=-1, keepdims=True)
        print(f"NumPy attention weights stats: mean={attn_weights_np.mean():.6f}, std={attn_weights_np.std():.6f}")
        pcc_weights = compute_pcc(attn_weights_ref.numpy(), attn_weights_np)
        print(f"PCC (PyTorch vs NumPy): {pcc_weights:.6f}")

        # STEP 4: Multiply by V
        print(f"\n{'=' * 80}")
        print("STEP 4: Attention weights @ V")
        print(f"{'=' * 80}")

        output_ref = torch.matmul(attn_weights_ref, v)
        print(f"PyTorch output shape: {output_ref.shape}")
        print(f"PyTorch output stats: mean={output_ref.mean():.6f}, std={output_ref.std():.6f}")
        print(f"First 5 values: {output_ref.flatten()[:5].tolist()}")

        # NumPy reference
        output_np = np.matmul(attn_weights_np, v_np)
        print(f"NumPy output stats: mean={output_np.mean():.6f}, std={output_np.std():.6f}")
        pcc_output = compute_pcc(output_ref.numpy(), output_np)
        print(f"PCC (PyTorch vs NumPy): {pcc_output:.6f}")

        # NOW TEST WITH TTML
        print(f"\n{'=' * 80}")
        print("TTML: Full scaled_dot_product_attention")
        print(f"{'=' * 80}")

        q_ttml = ttml.autograd.Tensor.from_numpy(q_np)
        k_ttml = ttml.autograd.Tensor.from_numpy(k_np)
        v_ttml = ttml.autograd.Tensor.from_numpy(v_np)

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=None)
        output_ttml_np = output_ttml.to_numpy()

        print(f"TTML output shape: {output_ttml_np.shape}")
        print(f"TTML output stats: mean={output_ttml_np.mean():.6f}, std={output_ttml_np.std():.6f}")
        print(f"First 5 values: {output_ttml_np.flatten()[:5].tolist()}")

        # Compare TTML vs Reference
        diff = np.abs(output_ref.numpy() - output_ttml_np)
        pcc_ttml = compute_pcc(output_ref.numpy(), output_ttml_np)

        print(f"\n{'=' * 80}")
        print("FINAL COMPARISON: PyTorch Reference vs TTML")
        print(f"{'=' * 80}")
        print(f"Mean abs diff: {diff.mean():.6e}")
        print(f"Max abs diff: {diff.max():.6e}")
        print(f"PCC: {pcc_ttml:.6f}")

        if pcc_ttml >= 0.99:
            print(f"✅ PASS: TTML matches reference (PCC={pcc_ttml:.6f})")
        else:
            print(f"❌ FAIL: TTML diverges from reference (PCC={pcc_ttml:.6f})")

            # Detailed comparison
            print(f"\nDetailed comparison (first 20 values):")
            print(f"{'Index':<6} {'Reference':<15} {'TTML':<15} {'Diff':<15}")
            print("-" * 60)
            ref_flat = output_ref.numpy().flatten()
            ttml_flat = output_ttml_np.flatten()
            for i in range(min(20, len(ref_flat))):
                diff_val = abs(ref_flat[i] - ttml_flat[i])
                print(f"{i:<6} {ref_flat[i]:<15.6f} {ttml_flat[i]:<15.6f} {diff_val:<15.6f}")


if __name__ == "__main__":
    main()
