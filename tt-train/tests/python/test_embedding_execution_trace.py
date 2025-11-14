#!/usr/bin/env python3
"""
Trace embedding execution step-by-step to find where HF vs TTML diverges.

Tests every intermediate step:
1. Weight tensor retrieval from TTML
2. Weight tensor to numpy conversion
3. Embedding operation input preparation
4. Embedding operation execution
5. Embedding output retrieval
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoModel

sys.path.insert(0, "/workspace/tt-metal/tt-train/build/sources/ttml")
import _ttml as ttml


def compute_pcc(a, b):
    """Compute Pearson Correlation Coefficient."""
    a_flat = a.flatten()
    b_flat = b.flatten()

    if len(a_flat) != len(b_flat):
        return 0.0

    mean_a = np.mean(a_flat)
    mean_b = np.mean(b_flat)

    numerator = np.sum((a_flat - mean_a) * (b_flat - mean_b))
    denominator = np.sqrt(np.sum((a_flat - mean_a) ** 2) * np.sum((b_flat - mean_b) ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def main():
    print("=" * 80)
    print("EMBEDDING EXECUTION TRACE - STEP BY STEP VERIFICATION")
    print("=" * 80)
    print()

    # Load models
    print("Loading models...")
    hf_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

    config = ttml.models.bert.BertConfig()
    config.vocab_size = 30522
    config.max_sequence_length = 32
    config.embedding_dim = 128
    config.intermediate_size = 512
    config.num_heads = 2
    config.num_blocks = 2
    config.dropout_prob = 0.0
    config.layer_norm_eps = 1e-12
    config.use_token_type_embeddings = True

    ttml_model = ttml.models.bert.create(config)

    # Load weights
    safetensors_path = (
        f"{os.environ['HOME']}/.cache/huggingface/hub/models--prajjwal1--bert-tiny/snapshots/*/model.safetensors"
    )
    import glob

    safetensors_file = glob.glob(safetensors_path)[0]
    ttml_model.load_model_from_safetensors(safetensors_file)
    print("Models loaded\n")

    # Create test input
    batch_size = 2
    seq_len = 8  # Small for detailed inspection
    input_ids = torch.tensor(
        [[101, 2003, 2023, 1037, 3231, 102, 0, 0], [101, 7592, 2088, 102, 0, 0, 0, 0]],  # Batch 0  # Batch 1
        dtype=torch.long,
    )

    print(f"Test input shape: {input_ids.shape}")
    print(f"Batch 0 IDs: {input_ids[0].tolist()}")
    print(f"Batch 1 IDs: {input_ids[1].tolist()}")
    print()

    print("=" * 80)
    print("STEP 1: WEIGHT TENSOR RETRIEVAL")
    print("=" * 80)
    print()

    # HuggingFace weights
    hf_weights = hf_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    print(f"HF weights shape: {hf_weights.shape}")
    print(f"HF weights dtype: {hf_weights.dtype}")
    print(
        f"HF weights memory layout: C-contiguous={hf_weights.flags['C_CONTIGUOUS']}, F-contiguous={hf_weights.flags['F_CONTIGUOUS']}"
    )
    print()

    # TTML weights - get tensor object
    ttml_params = ttml_model.parameters()
    ttml_weight_tensor = ttml_params["bert/token_embeddings/weight"]

    print(f"TTML weight tensor type: {type(ttml_weight_tensor)}")
    print()

    print("=" * 80)
    print("STEP 2: WEIGHT TENSOR TO NUMPY CONVERSION")
    print("=" * 80)
    print()

    # Convert to numpy
    ttml_weights_np = ttml_weight_tensor.to_numpy()
    print(f"TTML weights numpy shape: {ttml_weights_np.shape}")
    print(f"TTML weights numpy dtype: {ttml_weights_np.dtype}")
    print(
        f"TTML weights memory layout: C-contiguous={ttml_weights_np.flags['C_CONTIGUOUS']}, F-contiguous={ttml_weights_np.flags['F_CONTIGUOUS']}"
    )

    # Reshape to 2D
    if len(ttml_weights_np.shape) == 4:
        ttml_weights_2d = ttml_weights_np[0, 0, :, :]
    else:
        ttml_weights_2d = ttml_weights_np

    print(f"TTML weights 2D shape: {ttml_weights_2d.shape}")
    print()

    # Compare weights for specific tokens in our input
    print("Weight comparison for input tokens:")
    print("-" * 80)
    test_token_ids = [101, 2003, 2023, 7592]
    for token_id in test_token_ids:
        hf_vec = hf_weights[token_id, :]
        ttml_vec = ttml_weights_2d[token_id, :]

        diff = np.abs(hf_vec - ttml_vec)
        pcc = compute_pcc(hf_vec, ttml_vec)

        print(f"Token {token_id}:")
        print(f"  HF   first 5: {hf_vec[:5]}")
        print(f"  TTML first 5: {ttml_vec[:5]}")
        print(f"  Max diff: {np.max(diff):.6f}, Mean diff: {np.mean(diff):.6f}, PCC: {pcc:.6f}")
        print()

    print("=" * 80)
    print("STEP 3: EMBEDDING OPERATION INPUT PREPARATION")
    print("=" * 80)
    print()

    # HF embedding operation
    print("HuggingFace embedding:")
    with torch.no_grad():
        hf_embedded = hf_model.embeddings.word_embeddings(input_ids).cpu().numpy()
    print(f"  Output shape: {hf_embedded.shape}")
    print(f"  Output dtype: {hf_embedded.dtype}")
    print()

    # TTML embedding operation - prepare input
    print("TTML embedding input preparation:")
    input_ids_np = input_ids.numpy().astype(np.uint32)
    print(f"  Input IDs numpy shape: {input_ids_np.shape}")
    print(f"  Input IDs numpy dtype: {input_ids_np.dtype}")

    # Reshape to TTML format [batch, 1, 1, seq_len]
    input_ids_4d = input_ids_np.reshape(batch_size, 1, 1, seq_len)
    print(f"  Input IDs 4D shape: {input_ids_4d.shape}")

    # Convert to TTML tensor
    ttml_input = ttml.autograd.Tensor.from_numpy(input_ids_4d)
    print(f"  TTML input tensor created")
    print()

    print("=" * 80)
    print("STEP 4: EMBEDDING OPERATION EXECUTION")
    print("=" * 80)
    print()

    # Execute TTML embedding
    print("Executing ttml.ops.embedding.embedding_op()...")
    ttml_embedded_tensor = ttml.ops.embedding.embedding_op(ttml_input, ttml_weight_tensor)
    print(f"  Output tensor type: {type(ttml_embedded_tensor)}")
    print()

    print("=" * 80)
    print("STEP 5: EMBEDDING OUTPUT RETRIEVAL")
    print("=" * 80)
    print()

    # Convert TTML output to numpy
    print("Converting TTML output to numpy...")
    ttml_embedded_np = ttml_embedded_tensor.to_numpy()
    print(f"  Numpy shape: {ttml_embedded_np.shape}")
    print(f"  Numpy dtype: {ttml_embedded_np.dtype}")
    print(
        f"  Memory layout: C-contiguous={ttml_embedded_np.flags['C_CONTIGUOUS']}, F-contiguous={ttml_embedded_np.flags['F_CONTIGUOUS']}"
    )
    print()

    # Reshape to match HF
    if len(ttml_embedded_np.shape) == 4:
        ttml_embedded_3d = ttml_embedded_np[:, 0, :, :]  # [batch, seq, dim]
    else:
        ttml_embedded_3d = ttml_embedded_np

    print(f"  Reshaped to 3D: {ttml_embedded_3d.shape}")
    print()

    print("=" * 80)
    print("STEP 6: TOKEN-BY-TOKEN COMPARISON")
    print("=" * 80)
    print()

    # Compare token by token
    for batch_idx in range(batch_size):
        print(f"Batch {batch_idx}:")
        print("-" * 80)
        for token_idx in range(seq_len):
            token_id = input_ids[batch_idx, token_idx].item()

            # Get embeddings for this token
            hf_token_emb = hf_embedded[batch_idx, token_idx, :]
            ttml_token_emb = ttml_embedded_3d[batch_idx, token_idx, :]

            # Compare
            diff = np.abs(hf_token_emb - ttml_token_emb)
            pcc = compute_pcc(hf_token_emb, ttml_token_emb)

            print(f"  Token {token_idx} (ID={token_id}):")
            print(f"    HF   first 5: {hf_token_emb[:5]}")
            print(f"    TTML first 5: {ttml_token_emb[:5]}")
            print(f"    Max diff: {np.max(diff):.6f}, Mean diff: {np.mean(diff):.6f}, PCC: {pcc:.6f}")

            # Also compare with weight tensor directly
            if token_id > 0:  # Skip padding
                weight_vec_hf = hf_weights[token_id, :]
                weight_vec_ttml = ttml_weights_2d[token_id, :]

                # Check if embedding output matches weight directly
                emb_vs_weight_hf = np.allclose(hf_token_emb, weight_vec_hf, rtol=1e-5)
                emb_vs_weight_ttml = np.allclose(ttml_token_emb, weight_vec_ttml, rtol=1e-5)

                print(f"    HF embedding == HF weight: {emb_vs_weight_hf}")
                print(f"    TTML embedding == TTML weight: {emb_vs_weight_ttml}")

                if not emb_vs_weight_ttml:
                    # There's a mismatch - embedding didn't retrieve the weight correctly!
                    diff_emb_weight = np.abs(ttml_token_emb - weight_vec_ttml)
                    print(f"    ⚠️ TTML embedding doesn't match weight!")
                    print(f"       Max diff emb vs weight: {np.max(diff_emb_weight):.6f}")
            print()

    print("=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)
    print()

    # Overall stats
    diff_all = np.abs(hf_embedded - ttml_embedded_3d)
    pcc_all = compute_pcc(hf_embedded, ttml_embedded_3d)

    print(f"Overall statistics:")
    print(f"  Max difference: {np.max(diff_all):.6f}")
    print(f"  Mean difference: {np.mean(diff_all):.6f}")
    print(f"  Median difference: {np.median(diff_all):.6f}")
    print(f"  PCC: {pcc_all:.6f}")
    print()

    # Per-batch comparison
    print("Per-batch PCC:")
    for batch_idx in range(batch_size):
        pcc_batch = compute_pcc(hf_embedded[batch_idx], ttml_embedded_3d[batch_idx])
        print(f"  Batch {batch_idx}: {pcc_batch:.6f}")
    print()

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    if pcc_all < 0.999:
        print("❌ EMBEDDING OUTPUT MISMATCH DETECTED")
        print()
        print("Possible causes:")
        print("1. Memory layout issue in to_numpy() conversion")
        print("2. Embedding operation using wrong tensor layout")
        print("3. Index lookup error in ttnn::embedding")
        print("4. Batch dimension handling issue")
    else:
        print("✅ EMBEDDING OUTPUT MATCHES")

    print()


if __name__ == "__main__":
    main()
