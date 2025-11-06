#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Batch Processing Bug - Detailed Reproduction Script

PURPOSE:
This script provides detailed evidence of the critical batch processing bug in BERT
where batch_size > 1 produces identical outputs for all samples regardless of input.

BUG DESCRIPTION:
When processing multiple samples in a batch (batch_size > 1), the BERT model produces
IDENTICAL outputs for ALL samples, even when the inputs are completely different.

EVIDENCE PROVIDED:
1. Creates two samples with maximally different inputs:
   - Sample 0: Token IDs from first half of vocabulary (0 to vocab_size/2)
   - Sample 1: Token IDs from second half of vocabulary (vocab_size/2 to vocab_size)

2. Runs batch inference (batch_size=2) and shows outputs are identical

3. Runs individual inference (batch_size=1) for each sample and shows they differ

4. Compares batch vs individual results to isolate the bug

FINDINGS:
- Batch processing: Both samples get identical outputs (BUG)
- Individual processing: Samples get different outputs (CORRECT)
- Batch sample[0] matches individual run of sample[0] (CORRECT)
- Batch sample[1] does NOT match individual run of sample[1] (BUG)

This proves the bug is in batch processing, not in the model itself.

USAGE:
    python3 tt-train/debug_batch_processing.py

SEE ALSO:
- tt-train/debug_tensor_shapes.py: Shape inspection across batch sizes
- tt-train/tests/model/bert_batch_bug_test.cpp: C++ reproduction of bug
- tt-train/tests/model/bert_seq_cls_test.cpp: False positive test that misses bug
"""

import numpy as np
import os
import sys
import torch

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml

from transformers import BertModel, BertForSequenceClassification


def test_batch_independence():
    """Test if different inputs in a batch produce different outputs."""

    print("\n" + "=" * 80)
    print("BATCH PROCESSING DEBUG")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    batch_size = 2
    seq_len = 32
    num_labels = 2

    # Load HF model
    print(f"\nLoading HuggingFace model: {model_name}")
    hf_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    hf_model.eval()
    config = hf_model.config

    # Save to safetensors
    from pathlib import Path
    from safetensors.torch import save_file

    model_dir = Path(f"/tmp/{model_name.replace('/', '_')}_batch_debug")
    model_dir.mkdir(exist_ok=True)
    save_file(hf_model.state_dict(), str(model_dir / "model.safetensors"))

    # Create TTML model
    print(f"Creating TTML model with batch_size={batch_size}")
    ttml_config = ttml.models.bert.BertConfig()
    ttml_config.vocab_size = config.vocab_size
    ttml_config.max_sequence_length = seq_len
    ttml_config.embedding_dim = config.hidden_size
    ttml_config.intermediate_size = config.intermediate_size
    ttml_config.num_heads = config.num_attention_heads
    ttml_config.num_blocks = config.num_hidden_layers
    ttml_config.dropout_prob = 0.0
    ttml_config.layer_norm_eps = config.layer_norm_eps
    ttml_config.use_token_type_embeddings = True

    ttml_model = ttml.models.bert.create_for_sequence_classification(ttml_config, num_labels, classifier_dropout=0.0)
    ttml_model.load_from_safetensors(str(model_dir))

    # Copy classifier weights
    ttml_params = ttml_model.parameters()
    hf_weight = hf_model.classifier.weight.detach().numpy()
    hf_bias = hf_model.classifier.bias.detach().numpy()

    weight_shape = ttml_params["bert/classifier/weight"].shape()
    num_labels_aligned = weight_shape[2]

    if num_labels_aligned > num_labels:
        padded_weight = np.zeros((num_labels_aligned, config.hidden_size), dtype=np.float32)
        padded_weight[:num_labels, :] = hf_weight
        hf_weight = padded_weight

        padded_bias = np.zeros(num_labels_aligned, dtype=np.float32)
        padded_bias[:num_labels] = hf_bias
        hf_bias = padded_bias

    weight_reshaped = hf_weight.reshape(1, 1, num_labels_aligned, config.hidden_size)
    bias_reshaped = hf_bias.reshape(1, 1, 1, num_labels_aligned)

    ttml_params["bert/classifier/weight"].set_value_from_tensor(ttml.autograd.Tensor.from_numpy(weight_reshaped))
    ttml_params["bert/classifier/bias"].set_value_from_tensor(ttml.autograd.Tensor.from_numpy(bias_reshaped))

    print("\n" + "-" * 80)
    print("TEST 1: Completely different inputs (different token IDs)")
    print("-" * 80)

    # Create two VERY different inputs
    np.random.seed(42)
    input_ids_np = np.zeros((batch_size, seq_len), dtype=np.int64)
    input_ids_np[0, :] = np.random.randint(0, config.vocab_size // 2, seq_len)  # First half of vocab
    input_ids_np[1, :] = np.random.randint(config.vocab_size // 2, config.vocab_size, seq_len)  # Second half

    token_type_ids_np = np.zeros((batch_size, seq_len), dtype=np.int64)
    attention_mask_np = np.ones((batch_size, seq_len), dtype=np.int64)

    print(f"Input IDs sample 0: {input_ids_np[0, :5]}")
    print(f"Input IDs sample 1: {input_ids_np[1, :5]}")
    print(f"Are inputs different? {not np.array_equal(input_ids_np[0], input_ids_np[1])}")

    # HF forward pass
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=torch.tensor(input_ids_np),
            token_type_ids=torch.tensor(token_type_ids_np),
            attention_mask=torch.tensor(attention_mask_np),
        )
        hf_logits = hf_outputs.logits.numpy()

    print(f"\nHF output sample 0: {hf_logits[0]}")
    print(f"HF output sample 1: {hf_logits[1]}")
    print(f"HF outputs different? {not np.allclose(hf_logits[0], hf_logits[1], rtol=1e-3)}")

    # TTML forward pass
    input_ids_reshaped = input_ids_np.reshape(batch_size, 1, 1, seq_len).astype(np.float32)
    token_type_ids_reshaped = token_type_ids_np.reshape(batch_size, 1, 1, seq_len).astype(np.float32)
    attention_mask_reshaped = attention_mask_np.reshape(batch_size, 1, 1, seq_len).astype(np.float32)

    print(f"\nTTML input shape: {input_ids_reshaped.shape}")
    print(f"TTML input sample 0: {input_ids_reshaped[0, 0, 0, :5]}")
    print(f"TTML input sample 1: {input_ids_reshaped[1, 0, 0, :5]}")

    input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids_reshaped)
    token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids_reshaped)
    attention_mask_ttml = ttml.autograd.Tensor.from_numpy(attention_mask_reshaped)

    ttml_logits = ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)
    ttml_logits_np = ttml_logits.to_numpy()

    ttml_logits_np = ttml_logits_np[:, :, :, :num_labels].reshape(batch_size, num_labels)

    print(f"\nTTML output sample 0: {ttml_logits_np[0]}")
    print(f"TTML output sample 1: {ttml_logits_np[1]}")
    print(f"TTML outputs different? {not np.allclose(ttml_logits_np[0], ttml_logits_np[1], rtol=1e-3)}")

    if np.allclose(ttml_logits_np[0], ttml_logits_np[1], rtol=1e-3):
        print("\n❌ BUG CONFIRMED: TTML produces identical outputs for different inputs!")
        print(f"   Difference: {np.abs(ttml_logits_np[0] - ttml_logits_np[1])}")
    else:
        print("\n✓ TTML produces different outputs for different inputs")

    print("\n" + "-" * 80)
    print("TEST 2: Run each sample individually (batch_size=1)")
    print("-" * 80)

    # Test sample 0 alone
    input_0 = input_ids_np[0:1, :].reshape(1, 1, 1, seq_len).astype(np.float32)
    token_type_0 = token_type_ids_np[0:1, :].reshape(1, 1, 1, seq_len).astype(np.float32)
    mask_0 = attention_mask_np[0:1, :].reshape(1, 1, 1, seq_len).astype(np.float32)

    logits_0 = ttml_model(
        ttml.autograd.Tensor.from_numpy(input_0),
        ttml.autograd.Tensor.from_numpy(mask_0),
        ttml.autograd.Tensor.from_numpy(token_type_0),
    )
    logits_0_np = logits_0.to_numpy()[:, :, :, :num_labels].reshape(num_labels)

    # Test sample 1 alone
    input_1 = input_ids_np[1:2, :].reshape(1, 1, 1, seq_len).astype(np.float32)
    token_type_1 = token_type_ids_np[1:2, :].reshape(1, 1, 1, seq_len).astype(np.float32)
    mask_1 = attention_mask_np[1:2, :].reshape(1, 1, 1, seq_len).astype(np.float32)

    logits_1 = ttml_model(
        ttml.autograd.Tensor.from_numpy(input_1),
        ttml.autograd.Tensor.from_numpy(mask_1),
        ttml.autograd.Tensor.from_numpy(token_type_1),
    )
    logits_1_np = logits_1.to_numpy()[:, :, :, :num_labels].reshape(num_labels)

    print(f"Individual run sample 0: {logits_0_np}")
    print(f"Individual run sample 1: {logits_1_np}")
    print(f"Individual runs produce different outputs? {not np.allclose(logits_0_np, logits_1_np, rtol=1e-3)}")

    print(f"\nBatch output sample 0 vs individual: {np.allclose(ttml_logits_np[0], logits_0_np, rtol=1e-3)}")
    print(f"Batch output sample 1 vs individual: {np.allclose(ttml_logits_np[1], logits_1_np, rtol=1e-3)}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if np.allclose(ttml_logits_np[0], ttml_logits_np[1], rtol=1e-3):
        print("❌ Batch processing is BROKEN: All samples get identical outputs")
        if not np.allclose(logits_0_np, logits_1_np, rtol=1e-3):
            print("✓ But individual processing works correctly")
            print("⚠️  This suggests the bug is in how batches are processed")
    else:
        print("✓ Batch processing works correctly!")


if __name__ == "__main__":
    test_batch_independence()
