# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Sequence Classification PCC Validation

Two validation approaches:
1. Copy HF classifier weights → TTML, compare outputs (PCC ≥ 0.99)
2. Use finetuned model with trained weights (PCC ≥ 0.99)

This validates BertForSequenceClassification against HuggingFace reference.
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
from transformers import BertForSequenceClassification as HFBertForSequenceClassification  # noqa: E402


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()

    mean1 = np.mean(tensor1_flat)
    mean2 = np.mean(tensor2_flat)

    numerator = np.sum((tensor1_flat - mean1) * (tensor2_flat - mean2))
    denominator = np.sqrt(np.sum((tensor1_flat - mean1) ** 2) * np.sum((tensor2_flat - mean2) ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


@pytest.mark.parametrize(
    "model_name,num_labels",
    [
        ("prajjwal1/bert-tiny", 2),  # Binary classification
        ("prajjwal1/bert-tiny", 3),  # 3-way classification
    ],
)
def test_option1_copy_weights_pcc(model_name, num_labels):
    """
    Option 1: Copy HF classifier weights to TTML, validate PCC ≥ 0.99

    This ensures identical weights produce identical outputs.
    """
    from transformers import BertModel

    batch_size = 2
    seq_len = 32

    # Load HuggingFace base model
    print(f"\n=== Option 1: Copy Weights Test ===")
    print(f"Model: {model_name}, num_labels: {num_labels}")

    hf_base_model = BertModel.from_pretrained(model_name)
    hf_base_model.eval()
    config = hf_base_model.config

    # Save base weights
    model_dir = Path(f"/tmp/{model_name.replace('/', '_')}")
    model_dir.mkdir(exist_ok=True)
    safetensors_path = model_dir / "model.safetensors"

    if not safetensors_path.exists():
        from safetensors.torch import save_file

        save_file(hf_base_model.state_dict(), str(safetensors_path))

    # Create HF classification model with random head
    hf_model = HFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    hf_model.eval()

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

    ttml_model = ttml.models.bert.create_for_sequence_classification(ttml_config, num_labels, classifier_dropout=0.0)
    ttml_model.load_from_safetensors(str(model_dir))

    # Copy HF classifier weights to TTML
    print("Copying HF classifier weights to TTML...")
    hf_classifier_weight = hf_model.classifier.weight.detach().numpy()  # [num_labels, hidden_size]
    hf_classifier_bias = hf_model.classifier.bias.detach().numpy()  # [num_labels]

    ttml_params = ttml_model.parameters()

    # Get shapes using to_numpy()
    classifier_weight_ttml = ttml_params["bert/classifier/weight"].to_numpy()
    classifier_bias_ttml = ttml_params["bert/classifier/bias"].to_numpy()

    weight_shape = classifier_weight_ttml.shape  # [1, 1, num_labels_aligned, hidden_size]
    bias_shape = classifier_bias_ttml.shape  # [1, 1, 1, num_labels_aligned]

    num_labels_aligned = weight_shape[2]

    print(f"  Weight shape: {weight_shape}, aligned labels: {num_labels_aligned}")

    # Pad if needed
    if num_labels_aligned > num_labels:
        padded_weight = np.zeros((num_labels_aligned, config.hidden_size), dtype=np.float32)
        padded_weight[:num_labels, :] = hf_classifier_weight
        hf_classifier_weight = padded_weight

        padded_bias = np.zeros(num_labels_aligned, dtype=np.float32)
        padded_bias[:num_labels] = hf_classifier_bias
        hf_classifier_bias = padded_bias

    # Reshape to TTML format [1, 1, out_features, in_features]
    weight_reshaped = hf_classifier_weight.reshape(1, 1, num_labels_aligned, config.hidden_size)
    bias_reshaped = hf_classifier_bias.reshape(1, 1, 1, num_labels_aligned)

    # Set values using from_vector
    device = ttml.core.get_device()

    ttml_params["bert/classifier/weight"].set_value(
        ttml.core.from_vector(weight_reshaped.flatten().tolist(), weight_shape, device)
    )
    ttml_params["bert/classifier/bias"].set_value(
        ttml.core.from_vector(bias_reshaped.flatten().tolist(), bias_shape, device)
    )

    print("  ✓ Weights copied successfully")

    # Generate test inputs
    np.random.seed(42)
    torch.manual_seed(42)

    input_ids_np = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    token_type_ids_np = np.random.randint(0, 2, (batch_size, seq_len))
    attention_mask_np = np.ones((batch_size, seq_len), dtype=np.int64)

    # HF forward pass
    with torch.no_grad():
        input_ids_torch = torch.tensor(input_ids_np, dtype=torch.long)
        token_type_ids_torch = torch.tensor(token_type_ids_np, dtype=torch.long)
        attention_mask_torch = torch.tensor(attention_mask_np, dtype=torch.long)

        hf_outputs = hf_model(
            input_ids=input_ids_torch,
            token_type_ids=token_type_ids_torch,
            attention_mask=attention_mask_torch,
        )
        hf_logits = hf_outputs.logits.numpy()

    # TTML forward pass
    input_ids_ttml = ttml.autograd.create_tensor(
        ttml.core.from_vector(input_ids_np.flatten().tolist(), [batch_size, 1, 1, seq_len], device)
    )
    token_type_ids_ttml = ttml.autograd.create_tensor(
        ttml.core.from_vector(token_type_ids_np.flatten().tolist(), [batch_size, 1, 1, seq_len], device)
    )
    attention_mask_ttml = ttml.autograd.create_tensor(
        ttml.core.from_vector(attention_mask_np.flatten().tolist(), [batch_size, 1, 1, seq_len], device)
    )

    ttml_logits = ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)
    ttml_logits_np = np.array(ttml.core.to_vector(ttml_logits.get_value())).reshape(batch_size, 1, 1, -1)

    # Extract actual labels (not padding)
    ttml_logits_np = ttml_logits_np[:, :, :, :num_labels]

    # Compute PCC
    hf_logits_flat = hf_logits.reshape(batch_size, num_labels)
    ttml_logits_flat = ttml_logits_np.reshape(batch_size, num_labels)

    pcc = compute_pcc(hf_logits_flat, ttml_logits_flat)

    abs_diff = np.abs(hf_logits_flat - ttml_logits_flat)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"\nResults:")
    print(f"  PCC: {pcc:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  HF logits sample: {hf_logits_flat[0]}")
    print(f"  TTML logits sample: {ttml_logits_flat[0]}")

    # Assert PCC ≥ 0.99
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99 (failed accuracy threshold)"
    print(f"\n✓ Option 1 passed: PCC {pcc:.6f} ≥ 0.99")


@pytest.mark.parametrize(
    "model_name,expected_num_labels",
    [
        ("textattack/bert-base-uncased-SST-2", 2),  # Sentiment (positive/negative)
        ("textattack/bert-base-uncased-MNLI", 3),  # NLI (entailment/neutral/contradiction)
    ],
)
def test_option2_finetuned_model_pcc(model_name, expected_num_labels):
    """
    Option 2: Use finetuned model with trained weights, validate PCC ≥ 0.99

    This validates against real-world finetuned models.
    """
    batch_size = 2
    seq_len = 128  # SST-2 and MNLI typically use longer sequences

    print(f"\n=== Option 2: Finetuned Model Test ===")
    print(f"Model: {model_name}")

    # Load finetuned HF model
    hf_model = HFBertForSequenceClassification.from_pretrained(model_name)
    hf_model.eval()
    config = hf_model.config

    actual_num_labels = hf_model.num_labels
    print(f"Num labels: {actual_num_labels} (expected: {expected_num_labels})")
    assert actual_num_labels == expected_num_labels

    # Save complete model (including classifier head)
    model_dir = Path(f"/tmp/{model_name.replace('/', '_')}_finetuned")
    model_dir.mkdir(exist_ok=True)
    safetensors_path = model_dir / "model.safetensors"

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

    ttml_model = ttml.models.bert.create_for_sequence_classification(
        ttml_config, actual_num_labels, classifier_dropout=0.0
    )

    # Load weights (including classifier from finetuned model)
    print("Loading finetuned weights including classifier...")
    ttml_model.load_from_safetensors(str(model_dir))

    # Copy classifier weights from saved model
    classifier_state = {
        "classifier.weight": hf_model.classifier.weight.detach().numpy(),
        "classifier.bias": hf_model.classifier.bias.detach().numpy(),
    }

    ttml_params = ttml_model.parameters()
    classifier_weight_ttml = ttml_params["bert/classifier/weight"].to_numpy()
    weight_shape = classifier_weight_ttml.shape
    num_labels_aligned = weight_shape[2]

    # Pad and copy weights
    hf_classifier_weight = classifier_state["classifier.weight"]  # [num_labels, hidden]
    hf_classifier_bias = classifier_state["classifier.bias"]  # [num_labels]

    if num_labels_aligned > actual_num_labels:
        padded_weight = np.zeros((num_labels_aligned, config.hidden_size), dtype=np.float32)
        padded_weight[:actual_num_labels, :] = hf_classifier_weight
        hf_classifier_weight = padded_weight

        padded_bias = np.zeros(num_labels_aligned, dtype=np.float32)
        padded_bias[:actual_num_labels] = hf_classifier_bias
        hf_classifier_bias = padded_bias

    device = ttml.core.get_device()

    weight_reshaped = hf_classifier_weight.reshape(1, 1, num_labels_aligned, config.hidden_size)
    bias_reshaped = hf_classifier_bias.reshape(1, 1, 1, num_labels_aligned)

    ttml_params["bert/classifier/weight"].set_value(
        ttml.core.from_vector(weight_reshaped.flatten().tolist(), weight_shape, device)
    )
    bias_shape = ttml_params["bert/classifier/bias"].to_numpy().shape
    ttml_params["bert/classifier/bias"].set_value(
        ttml.core.from_vector(bias_reshaped.flatten().tolist(), bias_shape, device)
    )

    print(f"  ✓ Finetuned weights loaded")

    # Generate test inputs
    np.random.seed(42)
    torch.manual_seed(42)

    input_ids_np = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    token_type_ids_np = np.random.randint(0, 2, (batch_size, seq_len))
    attention_mask_np = np.ones((batch_size, seq_len), dtype=np.int64)

    # HF forward pass
    with torch.no_grad():
        input_ids_torch = torch.tensor(input_ids_np, dtype=torch.long)
        token_type_ids_torch = torch.tensor(token_type_ids_np, dtype=torch.long)
        attention_mask_torch = torch.tensor(attention_mask_np, dtype=torch.long)

        hf_outputs = hf_model(
            input_ids=input_ids_torch,
            token_type_ids=token_type_ids_torch,
            attention_mask=attention_mask_torch,
        )
        hf_logits = hf_outputs.logits.numpy()

    # TTML forward pass
    input_ids_ttml = ttml.autograd.create_tensor(
        ttml.core.from_vector(input_ids_np.flatten().tolist(), [batch_size, 1, 1, seq_len], device)
    )
    token_type_ids_ttml = ttml.autograd.create_tensor(
        ttml.core.from_vector(token_type_ids_np.flatten().tolist(), [batch_size, 1, 1, seq_len], device)
    )
    attention_mask_ttml = ttml.autograd.create_tensor(
        ttml.core.from_vector(attention_mask_np.flatten().tolist(), [batch_size, 1, 1, seq_len], device)
    )

    ttml_logits = ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)
    ttml_logits_np = np.array(ttml.core.to_vector(ttml_logits.get_value())).reshape(batch_size, 1, 1, -1)

    # Extract actual labels
    ttml_logits_np = ttml_logits_np[:, :, :, :actual_num_labels]

    # Compute PCC
    hf_logits_flat = hf_logits.reshape(batch_size, actual_num_labels)
    ttml_logits_flat = ttml_logits_np.reshape(batch_size, actual_num_labels)

    pcc = compute_pcc(hf_logits_flat, ttml_logits_flat)

    abs_diff = np.abs(hf_logits_flat - ttml_logits_flat)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"\nResults:")
    print(f"  PCC: {pcc:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  HF logits sample: {hf_logits_flat[0]}")
    print(f"  TTML logits sample: {ttml_logits_flat[0]}")

    # Assert PCC ≥ 0.99
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99 (failed accuracy threshold)"
    print(f"\n✓ Option 2 passed: PCC {pcc:.6f} ≥ 0.99 for finetuned model")


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running BERT Sequence Classification PCC validation...")
    print("\n" + "=" * 70)
    test_option1_copy_weights_pcc("prajjwal1/bert-tiny", 2)
    print("\n" + "=" * 70)
    test_option2_finetuned_model_pcc("textattack/bert-base-uncased-SST-2", 2)
    print("\n✓ All PCC validation tests passed!")
