#!/usr/bin/env python3
"""
Regression test for BERT batch processing

Tests that BERT operations work correctly with batch_size > 1.
These tests serve as regression tests to ensure batch processing
maintains high accuracy (PCC > 0.999).
"""

import pytest
import torch
from transformers import AutoModel, AutoTokenizer
import ttml
import os
import numpy as np


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient"""
    flat1 = tensor1.reshape(-1).numpy() if isinstance(tensor1, torch.Tensor) else tensor1.reshape(-1)
    flat2 = tensor2.reshape(-1).numpy() if isinstance(tensor2, torch.Tensor) else tensor2.reshape(-1)

    mean1 = np.mean(flat1)
    mean2 = np.mean(flat2)

    numerator = np.sum((flat1 - mean1) * (flat2 - mean2))
    denom1 = np.sum((flat1 - mean1) ** 2)
    denom2 = np.sum((flat2 - mean2) ** 2)

    return numerator / np.sqrt(denom1 * denom2)


@pytest.mark.parametrize(
    "batch_size,seq_len,model_name",
    [
        (1, 32, "prajjwal1/bert-tiny"),
        (2, 32, "prajjwal1/bert-tiny"),
        (4, 32, "prajjwal1/bert-tiny"),
    ],
)
def test_bert_embeddings_batch_processing(batch_size, seq_len, model_name):
    """
    Test that BERT embeddings work correctly with different batch sizes.

    This is a regression test to ensure batch processing maintains high accuracy.
    Previously, there were concerns about batch processing accuracy degradation,
    but the core operations have been verified to work correctly.
    """
    print(f"\n{'='*70}")
    print(f"Testing BERT embeddings with batch_size={batch_size}, model={model_name}")
    print(f"{'='*70}\n")

    # Load HuggingFace model
    hf_model = AutoModel.from_pretrained(model_name)
    hf_model.eval()

    # Create TTML model
    config = ttml.models.bert.BertConfig(
        num_heads=hf_model.config.num_attention_heads,
        encoder_layers=hf_model.config.num_hidden_layers,
        num_embeddings=hf_model.config.vocab_size,
        encoder_hidden_size=hf_model.config.hidden_size,
        encoder_ff_dim=hf_model.config.intermediate_size,
        max_position_embeddings=hf_model.config.max_position_embeddings,
        layer_norm_eps=hf_model.config.layer_norm_eps,
        dropout_prob=0.0,
        use_token_type_embeddings=True,
    )

    ttml_model = ttml.models.bert.create(config)

    # Load weights from safetensors
    model_dir = f"/root/.cache/huggingface/hub/models--{model_name.replace('/', '--')}/snapshots"
    snapshot_dir = None
    if os.path.exists(model_dir):
        for d in os.listdir(model_dir):
            snapshot_path = os.path.join(model_dir, d)
            if os.path.isdir(snapshot_path):
                snapshot_dir = snapshot_path
                break

    assert snapshot_dir is not None, f"Could not find snapshot directory for {model_name}"

    safetensors_path = os.path.join(snapshot_dir, "model.safetensors")
    assert os.path.exists(safetensors_path), f"model.safetensors not found at {safetensors_path}"

    ttml_model.load_model_from_safetensors(safetensors_path)

    # Create test inputs
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.num_embeddings, (batch_size, seq_len))
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # Get HF embeddings
    with torch.no_grad():
        hf_embeddings = hf_model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

    # Get TTML embeddings
    ttml_input_ids = ttml.core.from_torch(input_ids.unsqueeze(1).unsqueeze(1), device=ttml.autograd.ctx().get_device())
    ttml_token_type_ids = ttml.core.from_torch(
        token_type_ids.unsqueeze(1).unsqueeze(1), device=ttml.autograd.ctx().get_device()
    )

    ttml_input = ttml.autograd.create_tensor(ttml_input_ids)
    ttml_token_types = ttml.autograd.create_tensor(ttml_token_type_ids)

    ttml_embeddings = ttml_model.get_embeddings(ttml_input, ttml_token_types)
    ttml_embeddings_cpu = ttml.core.to_torch(ttml_embeddings.get_value())

    # Compute metrics
    pcc = compute_pcc(hf_embeddings, ttml_embeddings_cpu)
    mean_abs_diff = torch.mean(torch.abs(hf_embeddings - ttml_embeddings_cpu)).item()
    max_abs_diff = torch.max(torch.abs(hf_embeddings - ttml_embeddings_cpu)).item()

    print(f"Results for batch_size={batch_size}:")
    print(f"  PCC: {pcc:.6f}")
    print(f"  Mean abs diff: {mean_abs_diff:.6f}")
    print(f"  Max abs diff: {max_abs_diff:.6f}")

    # Assert high accuracy - this should pass for all batch sizes
    assert pcc > 0.999, f"BERT embeddings PCC should be > 0.999 for batch_size={batch_size}, " f"but got {pcc:.6f}"

    print(f"✓ PASS: Embeddings work correctly with batch_size={batch_size}\n")


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_embedding_op_batch_processing(batch_size):
    """
    Test that core ops::embedding_op works correctly with different batch sizes.

    This is a low-level test that directly tests the embedding operation
    without the full BERT model overhead.
    """
    vocab_size = 128
    embedding_dim = 64
    seq_len = 32

    # Create simple sequential embedding weights
    weight_data = []
    for i in range(vocab_size):
        for j in range(embedding_dim):
            weight_data.append(float(i) * 0.1)

    weight_tensor = ttml.core.from_vector(
        weight_data, [1, 1, vocab_size, embedding_dim], device=ttml.autograd.ctx().get_device()
    )
    weight = ttml.autograd.create_tensor(weight_tensor)

    # Create input indices
    input_data = [i % vocab_size for i in range(batch_size * seq_len)]
    input_tensor = ttml.core.from_vector(
        input_data,
        [batch_size, 1, 1, seq_len],
        device=ttml.autograd.ctx().get_device(),
        dtype=ttml.core.DataType.UINT32,
        layout=ttml.core.Layout.ROW_MAJOR,
    )
    input_ids = ttml.autograd.create_tensor(input_tensor)

    # Run embedding operation
    embeddings = ttml.ops.embedding_op(input_ids, weight)
    embeddings_data = ttml.core.to_vector(embeddings.get_value())

    # Compute expected output
    expected_output = []
    for b in range(batch_size):
        for s in range(seq_len):
            idx = input_data[b * seq_len + s]
            for e in range(embedding_dim):
                expected_output.append(weight_data[idx * embedding_dim + e])

    # Compute PCC
    pcc = compute_pcc(np.array(embeddings_data), np.array(expected_output))

    mean_abs_diff = np.mean(np.abs(np.array(embeddings_data) - np.array(expected_output)))
    max_abs_diff = np.max(np.abs(np.array(embeddings_data) - np.array(expected_output)))

    print(f"\nEmbedding op batch_size={batch_size}:")
    print(f"  PCC: {pcc:.6f}")
    print(f"  Mean abs diff: {mean_abs_diff:.6f}")
    print(f"  Max abs diff: {max_abs_diff:.6f}")

    # Assert high accuracy
    assert pcc > 0.999, f"Embedding op PCC should be > 0.999 for batch_size={batch_size}, " f"but got {pcc:.6f}"

    print(f"✓ PASS: Embedding op works correctly with batch_size={batch_size}\n")


if __name__ == "__main__":
    # Run tests when executed directly
    import sys

    ttml.autograd.ctx().open_device()

    try:
        print("\n" + "=" * 70)
        print("BERT Batch Processing Regression Tests")
        print("=" * 70)

        # Test embedding op with different batch sizes
        print("\n### Testing core embedding_op ###\n")
        for batch_size in [1, 2, 4]:
            test_embedding_op_batch_processing(batch_size)

        # Test BERT embeddings with different batch sizes
        print("\n### Testing BERT embeddings ###\n")
        for batch_size, seq_len, model_name in [
            (1, 32, "prajjwal1/bert-tiny"),
            (2, 32, "prajjwal1/bert-tiny"),
            (4, 32, "prajjwal1/bert-tiny"),
        ]:
            test_bert_embeddings_batch_processing(batch_size, seq_len, model_name)

        print("\n" + "=" * 70)
        print("All batch processing tests PASSED!")
        print("=" * 70 + "\n")

    finally:
        ttml.autograd.ctx().close_device()
