# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple BERT Sequence Classification Validation

Basic tests to verify BertForSequenceClassification works correctly:
- Model creation and loading
- Correct output shapes
- Forward pass executes without errors
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")


@pytest.mark.parametrize(
    "model_name,num_labels,batch_size,seq_len",
    [
        ("prajjwal1/bert-tiny", 2, 2, 32),  # Binary classification
        ("prajjwal1/bert-tiny", 3, 1, 32),  # 3-way classification
        ("prajjwal1/bert-small", 5, 2, 64),  # Multi-class, larger model
    ],
)
def test_bert_sequence_classification_basic(model_name, num_labels, batch_size, seq_len):
    """Test BertForSequenceClassification basic functionality."""
    from transformers import BertModel

    # Load HuggingFace config
    hf_model = BertModel.from_pretrained(model_name)
    config = hf_model.config

    # Save to safetensors
    model_dir = Path(f"/tmp/{model_name.replace('/', '_')}")
    model_dir.mkdir(exist_ok=True)
    safetensors_path = model_dir / "model.safetensors"

    if not safetensors_path.exists():
        from safetensors.torch import save_file

        save_file(hf_model.state_dict(), str(safetensors_path))

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

    print(f"\nCreating TTML BertForSequenceClassification with {num_labels} labels")
    ttml_model = ttml.models.bert.create_for_sequence_classification(ttml_config, num_labels, classifier_dropout=0.0)

    # Load base BERT weights
    ttml_model.load_from_safetensors(str(model_dir))

    print(f"Model created successfully. Num labels: {ttml_model.get_num_labels()}")
    assert ttml_model.get_num_labels() == num_labels

    # Create random inputs
    np.random.seed(42)
    input_ids_np = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    token_type_ids_np = np.random.randint(0, 2, (batch_size, seq_len))
    attention_mask_np = np.ones((batch_size, seq_len), dtype=np.int64)

    # Convert to TTML tensors
    from ttml.autograd import create_tensor
    from ttml.core import from_vector, get_device, to_vector

    device = get_device()

    input_ids_ttml = create_tensor(
        from_vector(
            input_ids_np.flatten().tolist(),
            [batch_size, 1, 1, seq_len],
            device,
        )
    )

    token_type_ids_ttml = create_tensor(
        from_vector(
            token_type_ids_np.flatten().tolist(),
            [batch_size, 1, 1, seq_len],
            device,
        )
    )

    attention_mask_ttml = create_tensor(
        from_vector(
            attention_mask_np.flatten().tolist(),
            [batch_size, 1, 1, seq_len],
            device,
        )
    )

    # Forward pass
    print(f"Running forward pass...")
    logits = ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)

    # Convert output to numpy
    logits_np = np.array(to_vector(logits.get_value())).reshape(batch_size, 1, 1, -1)

    # Check output shape (num_labels is aligned to 32, so we extract only actual labels)
    num_labels_aligned = logits_np.shape[-1]
    print(f"Output shape: {logits_np.shape}, num_labels_aligned: {num_labels_aligned}")

    assert num_labels_aligned >= num_labels, f"Output has fewer than {num_labels} labels"
    assert num_labels_aligned % 32 == 0, f"Output labels not aligned to 32: {num_labels_aligned}"

    # Extract actual labels
    logits_actual = logits_np[:, :, :, :num_labels]
    assert logits_actual.shape == (batch_size, 1, 1, num_labels)

    print(f"✓ Test passed: {model_name}, {num_labels} labels, batch={batch_size}, seq={seq_len}")
    print(f"  Logits shape: {logits_actual.shape}")
    print(f"  Logits range: [{np.min(logits_actual):.4f}, {np.max(logits_actual):.4f}]")


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running BERT Sequence Classification basic tests...")
    test_bert_sequence_classification_basic("prajjwal1/bert-tiny", 2, 2, 32)
    test_bert_sequence_classification_basic("prajjwal1/bert-tiny", 3, 1, 32)
    print("\n✓ All basic tests passed!")
