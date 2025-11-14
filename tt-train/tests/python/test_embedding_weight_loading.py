#!/usr/bin/env python3
"""
Test to verify that embedding weights are loaded correctly from safetensors.

This test compares HuggingFace word embedding weights with TTML loaded weights
to identify if the bug is in the weight loading pipeline.

Expected behavior:
- HuggingFace weights should match TTML weights exactly (or very close)
- If they don't match, the bug is in weight loading (pad_vocab_embeddings, core::from_vector)
- If they do match, the bug is elsewhere
"""

import os
import sys
import pytest
import torch
import numpy as np
from transformers import AutoModel

# Import _ttml from PYTHONPATH (should be set to build/sources)
# The module will be loaded from build/sources/ttml when run with proper PYTHONPATH
import _ttml as ttml  # noqa: E402


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace BERT tiny model."""
    return AutoModel.from_pretrained("prajjwal1/bert-tiny")


@pytest.fixture(scope="module")
def ttml_model():
    """Load TTML BERT model with pre-trained weights."""
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
    config.type_vocab_size = 2

    model = ttml.models.bert.create(config)

    # Load weights from safetensors
    safetensors_path = (
        f"{os.environ['HOME']}/.cache/huggingface/hub/models--prajjwal1--bert-tiny/snapshots/*/model.safetensors"
    )
    import glob

    safetensors_file = glob.glob(safetensors_path)[0]

    model.load_model_from_safetensors(safetensors_file)

    return model


def test_word_embedding_weights_match(hf_model, ttml_model):
    """
    Test that word embedding weights loaded into TTML match HuggingFace weights.

    This verifies the weight loading pipeline (pad_vocab_embeddings, core::from_vector).
    """
    print("\n" + "=" * 80)
    print("WORD EMBEDDING WEIGHT COMPARISON TEST")
    print("=" * 80 + "\n")

    # Get HuggingFace weights
    hf_embeddings = hf_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    print(f"HuggingFace word embeddings shape: {hf_embeddings.shape}")  # Should be [30522, 128]

    # Get TTML weights
    # The weights are stored in the model parameters
    ttml_params = ttml_model.named_parameters()
    ttml_weight_tensor = None
    for name, param in ttml_params.items():
        if "token_embeddings/weight" in name:
            ttml_weight_tensor = param
            break

    assert ttml_weight_tensor is not None, "Could not find token_embeddings/weight in TTML model"

    # Convert TTML tensor to numpy
    ttml_embeddings = ttml_weight_tensor.to_numpy()
    print(f"TTML word embeddings shape: {ttml_embeddings.shape}")  # Should be [1, 1, 30528, 128] (padded)

    # TTML pads vocab to 30528 (nearest multiple of 32), so we need to handle this
    # Extract the actual vocab part (first 30522 rows)
    if len(ttml_embeddings.shape) == 4:
        # Shape is [1, 1, vocab_size_padded, embedding_dim]
        ttml_embeddings_2d = ttml_embeddings[0, 0, :, :]  # Extract to [vocab, dim]
    else:
        ttml_embeddings_2d = ttml_embeddings

    print(f"TTML embeddings reshaped: {ttml_embeddings_2d.shape}")

    # Compare first 30522 rows (actual vocab, not padding)
    vocab_size = 30522
    ttml_vocab = ttml_embeddings_2d[:vocab_size, :]

    print(f"\nComparing {vocab_size} vocabulary entries...")
    print(f"HuggingFace vocab shape: {hf_embeddings.shape}")
    print(f"TTML vocab shape: {ttml_vocab.shape}")

    # Sample specific vocabulary entries
    test_indices = [0, 100, 1000, 5000, 10000, 20000, 30000]

    print("\n" + "-" * 80)
    print("COMPARING SPECIFIC VOCABULARY ENTRIES")
    print("-" * 80)

    all_match = True
    for idx in test_indices:
        hf_row = hf_embeddings[idx, :]
        ttml_row = ttml_vocab[idx, :]

        # Compute statistics
        max_diff = np.max(np.abs(hf_row - ttml_row))
        mean_diff = np.mean(np.abs(hf_row - ttml_row))
        match = np.allclose(hf_row, ttml_row, rtol=1e-5, atol=1e-6)

        status = "✅ MATCH" if match else "❌ MISMATCH"
        all_match = all_match and match

        print(f"\nVocab[{idx:5d}]: {status}")
        print(f"  HF  first 5: {hf_row[:5]}")
        print(f"  TTML first 5: {ttml_row[:5]}")
        print(f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

    # Overall comparison
    print("\n" + "=" * 80)
    print("OVERALL WEIGHT COMPARISON")
    print("=" * 80)

    overall_max_diff = np.max(np.abs(hf_embeddings - ttml_vocab))
    overall_mean_diff = np.mean(np.abs(hf_embeddings - ttml_vocab))
    overall_match = np.allclose(hf_embeddings, ttml_vocab, rtol=1e-5, atol=1e-6)

    print(f"\nAll {vocab_size} entries:")
    print(f"  Max absolute difference: {overall_max_diff:.2e}")
    print(f"  Mean absolute difference: {overall_mean_diff:.2e}")
    print(f"  Overall match (rtol=1e-5, atol=1e-6): {overall_match}")

    if overall_match:
        print("\n✅ WEIGHTS LOADED CORRECTLY")
        print("   The bug is NOT in weight loading pipeline.")
        print("   Need to investigate embedding operation with these specific weights.")
    else:
        print("\n❌ WEIGHTS LOADED INCORRECTLY")
        print("   The bug IS in weight loading pipeline:")
        print("   - Check pad_vocab_embeddings() in bert.cpp:486-493")
        print("   - Check core::from_vector() usage")
        print("   - Check weight tensor layout/transpose")

    print("=" * 80 + "\n")

    # Assert that weights match
    assert (
        overall_match
    ), f"Word embeddings don't match! Max diff: {overall_max_diff:.2e}, Mean diff: {overall_mean_diff:.2e}"


def test_embedding_lookup_with_loaded_weights(hf_model, ttml_model):
    """
    Test embedding lookup with loaded weights to see if the operation produces correct results.

    This is a sanity check to ensure the weights work correctly in actual lookup.
    """
    print("\n" + "=" * 80)
    print("EMBEDDING LOOKUP TEST WITH LOADED WEIGHTS")
    print("=" * 80 + "\n")

    # Test with a simple input
    batch_size = 2
    seq_len = 32

    # Create identical test input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    print(f"Test input shape: {input_ids.shape}")
    print(f"First sequence IDs: {input_ids[0, :10].tolist()}")

    # HuggingFace forward pass (embeddings only)
    with torch.no_grad():
        hf_output = hf_model.embeddings.word_embeddings(input_ids).cpu().numpy()

    print(f"HF embedding output shape: {hf_output.shape}")

    # TTML forward pass
    # Convert input to TTML format: [batch, 1, 1, seq_len]
    input_ids_np = input_ids.numpy().astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    ttml_input = ttml.autograd.Tensor.from_numpy(input_ids_np)

    # Get just the word embeddings (not the full embedding layer)
    ttml_params = ttml_model.named_parameters()
    for name, param in ttml_params.items():
        if "token_embeddings/weight" in name:
            ttml_weight = param
            break

    # Perform embedding lookup
    ttml_output_tensor = ttml.ops.embedding(ttml_input, ttml_weight)
    ttml_output = ttml_output_tensor.to_numpy()

    print(f"TTML embedding output shape: {ttml_output.shape}")

    # Reshape TTML output to match HF: [batch, seq_len, dim]
    if len(ttml_output.shape) == 4:
        ttml_output_reshaped = ttml_output[:, 0, :, :]
    else:
        ttml_output_reshaped = ttml_output

    print(f"TTML output reshaped: {ttml_output_reshaped.shape}")

    # Compare outputs
    print("\n" + "-" * 80)
    print("COMPARING EMBEDDING LOOKUP OUTPUTS")
    print("-" * 80)

    # Compare batch 0 vs batch 1 in TTML output (should be different since inputs are different)
    ttml_batch0 = ttml_output_reshaped[0, :, :]
    ttml_batch1 = ttml_output_reshaped[1, :, :]

    # But let's also check: when inputs are identical, do outputs match?
    # Create a test with identical inputs
    identical_input = torch.tensor([[10, 20, 30] + [0] * 29] * 2, dtype=torch.long)
    identical_input_np = identical_input.numpy().astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    identical_ttml_input = ttml.autograd.Tensor.from_numpy(identical_input_np)

    identical_output_tensor = ttml.ops.embedding(identical_ttml_input, ttml_weight)
    identical_output = identical_output_tensor.to_numpy()

    if len(identical_output.shape) == 4:
        identical_output = identical_output[:, 0, :, :]

    # Compare batch 0 vs batch 1 (should be identical)
    batch0_identical = identical_output[0, :, :]
    batch1_identical = identical_output[1, :, :]

    # Compute PCC between batches
    def compute_pcc(arr1, arr2):
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()
        mean1 = np.mean(arr1_flat)
        mean2 = np.mean(arr2_flat)

        numerator = np.sum((arr1_flat - mean1) * (arr2_flat - mean2))
        denom = np.sqrt(np.sum((arr1_flat - mean1) ** 2) * np.sum((arr2_flat - mean2) ** 2))

        if denom == 0:
            return 0.0
        return numerator / denom

    pcc = compute_pcc(batch0_identical, batch1_identical)

    print(f"\nTest with identical inputs [10, 20, 30, 0, 0, ...]:")
    print(f"  Batch 0 first token (idx=10) first 5 values: {batch0_identical[0, :5]}")
    print(f"  Batch 1 first token (idx=10) first 5 values: {batch1_identical[0, :5]}")
    print(f"  PCC between batch 0 and batch 1: {pcc:.6f}")

    if pcc > 0.999:
        print(f"  ✅ Batches match correctly (PCC > 0.999)")
    else:
        print(f"  ❌ BUG REPRODUCED: Batches don't match (PCC = {pcc:.6f})")
        print(f"     This confirms the batch processing bug with loaded weights!")

    print("=" * 80 + "\n")

    # This test is informational - we expect it might fail due to the bug
    # So we don't assert here, just report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
