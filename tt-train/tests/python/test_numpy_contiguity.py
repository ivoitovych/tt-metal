#!/usr/bin/env python3
"""
Test if BERT Q,K,V numpy arrays are contiguous.

Non-contiguous arrays could cause data corruption during conversion to C++.
"""

import numpy as np
import os
import sys
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def main():
    print("\n" + "=" * 80)
    print("TEST: NumPy Array Contiguity")
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

        print(f"\nBefore reshape:")
        print(f"Q shape: {q.shape}")
        print(f"Q is C-contiguous: {q.is_contiguous()}")
        print(f"Q strides: {q.stride()}")

        # THIS IS THE KEY OPERATION - reshape and transpose
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        print(f"\nAfter reshape + transpose:")
        print(f"Q shape: {q.shape}")
        print(f"Q is C-contiguous: {q.is_contiguous()}")
        print(f"Q strides: {q.stride()}")

        q_np = q.numpy().astype(np.float32)
        k_np = k.numpy().astype(np.float32)
        v_np = v.numpy().astype(np.float32)

        print(f"\nAfter numpy conversion:")
        print(f"Q numpy shape: {q_np.shape}")
        print(f"Q numpy is C-contiguous: {q_np.flags['C_CONTIGUOUS']}")
        print(f"Q numpy is F-contiguous: {q_np.flags['F_CONTIGUOUS']}")
        print(f"Q numpy strides: {q_np.strides}")

        # Test if making it contiguous helps
        print(f"\n{'=' * 80}")
        print("TEST: Force contiguous before TTML conversion")
        print(f"{'=' * 80}")

        q_np_contiguous = np.ascontiguousarray(q_np)
        print(f"Q contiguous: {q_np_contiguous.flags['C_CONTIGUOUS']}")
        print(f"Q contiguous strides: {q_np_contiguous.strides}")

        # Test round-trip with non-contiguous
        print(f"\n{'=' * 80}")
        print("Round-trip 1: Non-contiguous array")
        print(f"{'=' * 80}")
        q_ttml_noncontig = ttml.autograd.Tensor.from_numpy(q_np)
        q_back_noncontig = q_ttml_noncontig.to_numpy()

        diff_noncontig = np.abs(q_np - q_back_noncontig)
        print(f"Mean abs diff: {diff_noncontig.mean():.6e}")
        print(f"Max abs diff: {diff_noncontig.max():.6e}")

        from test_bert_data_roundtrip import compute_pcc

        pcc_noncontig = compute_pcc(q_np, q_back_noncontig)
        print(f"PCC: {pcc_noncontig:.6f}")

        # Test round-trip with contiguous
        print(f"\n{'=' * 80}")
        print("Round-trip 2: Contiguous array")
        print(f"{'=' * 80}")
        q_ttml_contig = ttml.autograd.Tensor.from_numpy(q_np_contiguous)
        q_back_contig = q_ttml_contig.to_numpy()

        diff_contig = np.abs(q_np_contiguous - q_back_contig)
        print(f"Mean abs diff: {diff_contig.mean():.6e}")
        print(f"Max abs diff: {diff_contig.max():.6e}")

        pcc_contig = compute_pcc(q_np_contiguous, q_back_contig)
        print(f"PCC: {pcc_contig:.6f}")

        print(f"\n{'=' * 80}")
        print("RESULTS:")
        print(f"{'=' * 80}")
        print(f"Non-contiguous PCC: {pcc_noncontig:.6f}")
        print(f"Contiguous PCC:     {pcc_contig:.6f}")

        if pcc_contig > 0.99 and pcc_noncontig < 0.99:
            print(f"\n✅ BUG FOUND: Non-contiguous arrays are corrupted!")
            print(f"   Making arrays contiguous before TTML conversion fixes the issue.")
        elif pcc_noncontig < 0.99 and pcc_contig < 0.99:
            print(f"\n❌ Both fail - contiguity is not the issue")
        else:
            print(f"\n✅ Both work - not a contiguity issue")


if __name__ == "__main__":
    main()
