#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Batch Processing Bug - Tensor Shape Inspection

PURPOSE:
Inspect tensor shapes throughout the BERT forward pass at different batch sizes
to verify that the batch dimension is correctly preserved and to identify where
the batch processing bug might be occurring.

WHAT IT TESTS:
- Creates BERT model and runs inference at batch_size = 1, 2, 4
- Inspects input tensor shapes before and after conversion to TTML format
- Inspects output tensor shapes
- Verifies if all samples in a batch produce identical outputs

FINDINGS:
- All tensor shapes are CORRECT (batch dimension preserved correctly)
- Input tensors for different samples contain DIFFERENT values
- Output tensors have correct shape: [batch_size, 1, 1, num_labels]
- BUT: All samples in batch produce IDENTICAL output values

CONCLUSION:
The bug is NOT in tensor shapes or dimension handling. The shapes are all correct.
The bug is in the actual computation - something in the forward pass is causing
all samples to produce the same result despite different inputs.

USAGE:
    python3 tt-train/debug_tensor_shapes.py

SEE ALSO:
- tt-train/debug_batch_processing.py: Detailed batch processing bug reproduction
- tt-train/tests/model/bert_batch_bug_test.cpp: C++ reproduction of bug
"""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml

from transformers import BertForSequenceClassification


def inspect_shapes():
    """Inspect tensor shapes with different batch sizes."""

    print("\n" + "=" * 80)
    print("TENSOR SHAPE INSPECTION")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"
    seq_len = 32
    num_labels = 2

    # Load and save HF model
    hf_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    hf_model.eval()
    config = hf_model.config

    from pathlib import Path
    from safetensors.torch import save_file

    model_dir = Path(f"/tmp/{model_name.replace('/', '_')}_shape_debug")
    model_dir.mkdir(exist_ok=True)
    save_file(hf_model.state_dict(), str(model_dir / "model.safetensors"))

    for batch_size in [1, 2, 4]:
        print(f"\n{'='*80}")
        print(f"Testing batch_size={batch_size}")
        print("=" * 80)

        # Create TTML model
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

        ttml_model = ttml.models.bert.create_for_sequence_classification(
            ttml_config, num_labels, classifier_dropout=0.0
        )
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

        # Create test inputs
        np.random.seed(42)
        input_ids_np = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
        token_type_ids_np = np.zeros((batch_size, seq_len), dtype=np.int64)
        attention_mask_np = np.ones((batch_size, seq_len), dtype=np.int64)

        # Reshape for TTML
        input_ids_reshaped = input_ids_np.reshape(batch_size, 1, 1, seq_len).astype(np.float32)
        token_type_ids_reshaped = token_type_ids_np.reshape(batch_size, 1, 1, seq_len).astype(np.float32)
        attention_mask_reshaped = attention_mask_np.reshape(batch_size, 1, 1, seq_len).astype(np.float32)

        print(f"\nInput tensor shapes:")
        print(f"  input_ids: {input_ids_reshaped.shape}")
        print(f"  token_type_ids: {token_type_ids_reshaped.shape}")
        print(f"  attention_mask: {attention_mask_reshaped.shape}")

        # Forward pass
        input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids_reshaped)
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids_reshaped)
        attention_mask_ttml = ttml.autograd.Tensor.from_numpy(attention_mask_reshaped)

        print(f"\nTTML Tensor shapes:")
        print(f"  input_ids: {input_ids_ttml.shape()}")
        print(f"  token_type_ids: {token_type_ids_ttml.shape()}")
        print(f"  attention_mask: {attention_mask_ttml.shape()}")

        logits = ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)

        print(f"\nOutput tensor shape:")
        print(f"  logits: {logits.shape()}")

        logits_np = logits.to_numpy()
        print(f"  logits (numpy): {logits_np.shape}")

        # Check if all batch samples are identical
        if batch_size > 1:
            logits_extracted = logits_np[:, :, :, :num_labels].reshape(batch_size, num_labels)
            print(f"\nLogits values:")
            for i in range(batch_size):
                print(f"  Sample {i}: {logits_extracted[i]}")

            all_same = all(
                np.allclose(logits_extracted[0], logits_extracted[i], rtol=1e-5) for i in range(1, batch_size)
            )
            if all_same:
                print(f"\n❌ All {batch_size} samples have IDENTICAL outputs!")
            else:
                print(f"\n✓ Samples have different outputs")


if __name__ == "__main__":
    inspect_shapes()
