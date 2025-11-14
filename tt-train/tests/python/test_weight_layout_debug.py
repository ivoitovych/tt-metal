#!/usr/bin/env python3
"""
Debug weight layout to find if there's a transpose or reshaping issue.

Check if vocabulary entries are stored contiguously or if there's a layout problem.
"""

import os
import sys
import numpy as np
from transformers import AutoModel

sys.path.insert(0, "/workspace/tt-metal/tt-train/build/sources/ttml")
import _ttml as ttml


def main():
    print("=" * 80)
    print("WEIGHT LAYOUT DEBUG TEST")
    print("=" * 80 + "\n")

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    hf_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    hf_embeddings = hf_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    print(f"HF shape: {hf_embeddings.shape}")  # [30522, 128]

    # Load TTML model
    print("Loading TTML model...")
    config = ttml.models.bert.BertConfig()
    config.vocab_size = 30522
    config.max_sequence_length = 512
    config.embedding_dim = 128
    config.intermediate_size = 512
    config.num_heads = 2
    config.num_blocks = 2
    config.dropout_prob = 0.0
    config.layer_norm_eps = 1e-12
    config.use_token_type_embeddings = True

    model = ttml.models.bert.create(config)

    # Load weights
    safetensors_path = (
        f"{os.environ['HOME']}/.cache/huggingface/hub/models--prajjwal1--bert-tiny/snapshots/*/model.safetensors"
    )
    import glob

    safetensors_file = glob.glob(safetensors_path)[0]
    model.load_model_from_safetensors(safetensors_file)
    print("Weights loaded\n")

    # Get TTML weights
    ttml_params = model.parameters()
    ttml_weight_tensor = ttml_params["bert/token_embeddings/weight"]
    ttml_embeddings = ttml_weight_tensor.to_numpy()  # Shape: [1, 1, 30528, 128]
    ttml_embeddings_2d = ttml_embeddings[0, 0, :, :]  # [30528, 128]
    print(f"TTML shape: {ttml_embeddings_2d.shape}")  # [30528, 128]

    print("\n" + "=" * 80)
    print("TESTING WEIGHT LAYOUT")
    print("=" * 80 + "\n")

    # Test 1: Check if rows are contiguous
    print("Test 1: Check row contiguity")
    print("-" * 80)

    # Compare first row, middle row, last row
    test_indices = [0, 1, 100, 1000, 10000, 20000, 30000, 30521]

    for idx in test_indices:
        hf_row = hf_embeddings[idx, :]
        ttml_row = ttml_embeddings_2d[idx, :]

        diff = np.abs(hf_row - ttml_row)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"Row {idx:5d}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        # Show first 3 and last 3 values
        print(f"  HF   first 3: {hf_row[:3]}")
        print(f"  TTML first 3: {ttml_row[:3]}")
        print(f"  HF   last 3: {hf_row[-3:]}")
        print(f"  TTML last 3: {ttml_row[-3:]}")
        print()

    # Test 2: Check if columns might be transposed
    print("\n" + "=" * 80)
    print("Test 2: Check for possible transpose")
    print("-" * 80)

    # If weights are transposed, then TTML[i, j] should match HF[j, i]
    # Let's check a few positions
    test_positions = [(0, 0), (0, 127), (100, 50), (1000, 100), (30521, 127)]

    for i, j in test_positions:
        hf_val = hf_embeddings[i, j]
        ttml_val = ttml_embeddings_2d[i, j]
        ttml_transposed_val = ttml_embeddings_2d[j, i] if j < 30522 else None

        print(f"Position [{i:5d}, {j:3d}]:")
        print(f"  HF value:             {hf_val:.8f}")
        print(f"  TTML value:           {ttml_val:.8f}")
        if ttml_transposed_val is not None:
            print(f"  TTML transposed[{j}, {i}]: {ttml_transposed_val:.8f}")
        print(f"  Diff (normal):        {abs(hf_val - ttml_val):.8f}")
        if ttml_transposed_val is not None:
            print(f"  Diff (transposed):    {abs(hf_val - ttml_transposed_val):.8f}")
        print()

    # Test 3: Check embedding dimension continuity
    print("=" * 80)
    print("Test 3: Check embedding dimension values")
    print("-" * 80)

    # For a single vocabulary entry, check all 128 dimensions
    test_vocab_idx = 100
    hf_vec = hf_embeddings[test_vocab_idx, :]
    ttml_vec = ttml_embeddings_2d[test_vocab_idx, :]

    print(f"Vocab index {test_vocab_idx}, all 128 dimensions:")
    print(f"HF   dims 0-10:   {hf_vec[:10]}")
    print(f"TTML dims 0-10:   {ttml_vec[:10]}")
    print(f"Diffs:            {np.abs(hf_vec[:10] - ttml_vec[:10])}")
    print()
    print(f"HF   dims 118-128: {hf_vec[-10:]}")
    print(f"TTML dims 118-128: {ttml_vec[-10:]}")
    print(f"Diffs:             {np.abs(hf_vec[-10:] - ttml_vec[-10:])}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Compute overall statistics
    all_diffs = np.abs(hf_embeddings - ttml_embeddings_2d[:30522, :])

    print(f"\nOverall statistics:")
    print(f"  Max diff:    {np.max(all_diffs):.6f}")
    print(f"  Mean diff:   {np.mean(all_diffs):.6f}")
    print(f"  Median diff: {np.median(all_diffs):.6f}")
    print(f"  90%ile diff: {np.percentile(all_diffs, 90):.6f}")
    print(f"  99%ile diff: {np.percentile(all_diffs, 99):.6f}")

    # Check if errors are uniformly distributed or concentrated
    row_max_diffs = np.max(all_diffs, axis=1)
    col_max_diffs = np.max(all_diffs, axis=0)

    print(f"\nError distribution:")
    print(f"  Max diff per row - max: {np.max(row_max_diffs):.6f}, mean: {np.mean(row_max_diffs):.6f}")
    print(f"  Max diff per col - max: {np.max(col_max_diffs):.6f}, mean: {np.mean(col_max_diffs):.6f}")

    # Find which rows/cols have the largest errors
    worst_rows = np.argsort(row_max_diffs)[-5:]
    worst_cols = np.argsort(col_max_diffs)[-5:]

    print(f"\nWorst 5 vocabulary rows (indices): {worst_rows}")
    print(f"  Their max diffs: {row_max_diffs[worst_rows]}")

    print(f"\nWorst 5 embedding dimensions (indices): {worst_cols}")
    print(f"  Their max diffs: {col_max_diffs[worst_cols]}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
