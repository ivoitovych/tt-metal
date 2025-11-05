# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERT Sequence Classification Validation

Tests BertForSequenceClassification against HuggingFace reference:
- Multiple model sizes (tiny, small)
- Multiple configurations (binary, multi-class)
- PCC ≥ 0.99 target for production quality

This validates the first task-specific head implementation for BERT.
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


class BERTSequenceClassificationValidator:
    """Validates BERT sequence classification against HuggingFace."""

    def __init__(self, model_name: str, num_labels: int, batch_size: int = 2, seq_len: int = 32):
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Load HuggingFace model (without task head initially)
        print(f"\nLoading HuggingFace BERT base model: {model_name}")
        from transformers import BertModel

        hf_base_model = BertModel.from_pretrained(model_name)
        hf_base_model.eval()
        self.config = hf_base_model.config

        print(
            f"Model config: {self.config.num_hidden_layers} layers, "
            f"{self.config.hidden_size} hidden dim, {self.config.num_attention_heads} heads"
        )

        # Save base BERT weights to safetensors (without task head)
        # Create a directory for the model
        self.model_dir = Path(f"/tmp/{model_name.replace('/', '_')}")
        self.model_dir.mkdir(exist_ok=True)
        self.safetensors_path = self.model_dir / "model.safetensors"

        if not self.safetensors_path.exists():
            from safetensors.torch import save_file

            save_file(hf_base_model.state_dict(), str(self.safetensors_path))

        # Create HuggingFace classification model with random head
        print(f"Creating HuggingFace sequence classification model with {num_labels} labels")
        self.hf_model = HFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.hf_model.eval()

        # Create TTML model
        ttml_config = ttml.models.bert.BertConfig()
        ttml_config.vocab_size = self.config.vocab_size
        ttml_config.max_sequence_length = seq_len
        ttml_config.embedding_dim = self.config.hidden_size
        ttml_config.intermediate_size = self.config.intermediate_size
        ttml_config.num_heads = self.config.num_attention_heads
        ttml_config.num_blocks = self.config.num_hidden_layers
        ttml_config.dropout_prob = 0.0  # Disable dropout for validation
        ttml_config.layer_norm_eps = self.config.layer_norm_eps
        ttml_config.use_token_type_embeddings = True

        print(f"Creating TTML sequence classification model")
        self.ttml_model = ttml.models.bert.create_for_sequence_classification(
            ttml_config, num_labels, classifier_dropout=0.0
        )
        self.ttml_model.load_from_safetensors(str(self.model_dir))

        # Copy classifier weights from HuggingFace to TTML for identical comparison
        self._copy_classifier_weights()

    def _copy_classifier_weights(self):
        """Copy classifier head weights from HuggingFace to TTML for validation."""
        print("Copying classifier weights from HuggingFace to TTML...")

        # Get HF classifier weights
        hf_classifier_weight = self.hf_model.classifier.weight.detach().numpy()  # [num_labels, hidden_size]
        hf_classifier_bias = self.hf_model.classifier.bias.detach().numpy()  # [num_labels]

        # Get TTML parameters
        ttml_params = self.ttml_model.parameters()

        # Set classifier weight
        classifier_weight_param = ttml_params["bert/classifier/weight"]
        # TTML expects [1, 1, out_features, in_features], HF provides [out_features, in_features]
        weight_shape = classifier_weight_param.get_value().logical_shape()

        # Pad num_labels to aligned size if needed
        num_labels_aligned = weight_shape[-2]
        if num_labels_aligned > self.num_labels:
            print(f"  Padding classifier weights from {self.num_labels} to {num_labels_aligned}")
            padded_weight = np.zeros((num_labels_aligned, weight_shape[-1]), dtype=np.float32)
            padded_weight[: self.num_labels, :] = hf_classifier_weight
            hf_classifier_weight = padded_weight

            padded_bias = np.zeros(num_labels_aligned, dtype=np.float32)
            padded_bias[: self.num_labels] = hf_classifier_bias
            hf_classifier_bias = padded_bias

        from ttml.core import from_vector

        classifier_weight_param.set_value(
            from_vector(
                hf_classifier_weight.flatten().tolist(),
                weight_shape,
                classifier_weight_param.get_value().device(),
            )
        )

        # Set classifier bias
        classifier_bias_param = ttml_params["bert/classifier/bias"]
        bias_shape = classifier_bias_param.get_value().logical_shape()
        classifier_bias_param.set_value(
            from_vector(
                hf_classifier_bias.tolist(),
                bias_shape,
                classifier_bias_param.get_value().device(),
            )
        )

        print("  ✓ Classifier weights copied successfully")

    def compute_pcc(self, tensor1, tensor2):
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

    def validate_single_run(self, input_ids_np, token_type_ids_np, attention_mask_np, seed: int):
        """Run single validation with given inputs."""

        # HuggingFace forward pass
        with torch.no_grad():
            input_ids_torch = torch.tensor(input_ids_np, dtype=torch.long)
            token_type_ids_torch = torch.tensor(token_type_ids_np, dtype=torch.long)
            attention_mask_torch = torch.tensor(attention_mask_np, dtype=torch.long)

            hf_outputs = self.hf_model(
                input_ids=input_ids_torch,
                token_type_ids=token_type_ids_torch,
                attention_mask=attention_mask_torch,
            )
            hf_logits = hf_outputs.logits.numpy()

        # TTML forward pass
        from ttml.autograd import create_tensor
        from ttml.core import from_vector, get_device

        device = get_device()

        # Convert inputs to TTML tensors
        input_ids_ttml = create_tensor(
            from_vector(
                input_ids_np.flatten().tolist(),
                [self.batch_size, 1, 1, self.seq_len],
                device,
            )
        )

        token_type_ids_ttml = create_tensor(
            from_vector(
                token_type_ids_np.flatten().tolist(),
                [self.batch_size, 1, 1, self.seq_len],
                device,
            )
        )

        attention_mask_ttml = create_tensor(
            from_vector(
                attention_mask_np.flatten().tolist(),
                [self.batch_size, 1, 1, self.seq_len],
                device,
            )
        )

        # Forward pass
        ttml_logits = self.ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)

        # Convert TTML output to numpy
        from ttml.core import to_vector

        ttml_logits_np = np.array(to_vector(ttml_logits.get_value())).reshape(self.batch_size, 1, 1, -1)

        # Extract only actual labels (not padding)
        ttml_logits_np = ttml_logits_np[:, :, :, : self.num_labels]

        # Reshape for comparison: [batch_size, num_labels]
        hf_logits_reshaped = hf_logits.reshape(self.batch_size, self.num_labels)
        ttml_logits_reshaped = ttml_logits_np.reshape(self.batch_size, self.num_labels)

        # Compute PCC
        pcc = self.compute_pcc(hf_logits_reshaped, ttml_logits_reshaped)

        # Compute element-wise differences
        abs_diff = np.abs(hf_logits_reshaped - ttml_logits_reshaped)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        return {
            "pcc": pcc,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "hf_logits": hf_logits_reshaped,
            "ttml_logits": ttml_logits_reshaped,
        }

    def validate(self, num_runs: int = 3):
        """Run multiple validation runs and report statistics."""
        print(f"\nRunning {num_runs} validation runs...")

        results = []
        for run_idx in range(num_runs):
            seed = 42 + run_idx
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Generate random inputs
            input_ids_np = np.random.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
            token_type_ids_np = np.random.randint(0, 2, (self.batch_size, self.seq_len))
            attention_mask_np = np.ones((self.batch_size, self.seq_len), dtype=np.int64)

            # Optionally mask some tokens
            if run_idx % 2 == 1:
                mask_length = self.seq_len // 4
                attention_mask_np[:, -mask_length:] = 0

            result = self.validate_single_run(input_ids_np, token_type_ids_np, attention_mask_np, seed)
            results.append(result)

            print(
                f"  Run {run_idx + 1}: PCC={result['pcc']:.6f}, "
                f"Max diff={result['max_diff']:.6f}, Mean diff={result['mean_diff']:.6f}"
            )

        # Compute statistics
        pccs = [r["pcc"] for r in results]
        max_diffs = [r["max_diff"] for r in results]
        mean_diffs = [r["mean_diff"] for r in results]

        print(f"\nStatistics over {num_runs} runs:")
        print(f"  PCC: mean={np.mean(pccs):.6f}, min={np.min(pccs):.6f}, max={np.max(pccs):.6f}")
        print(f"  Max diff: mean={np.mean(max_diffs):.6f}, max={np.max(max_diffs):.6f}")
        print(f"  Mean diff: mean={np.mean(mean_diffs):.6f}, max={np.max(mean_diffs):.6f}")

        return results


@pytest.mark.parametrize(
    "model_name,num_labels",
    [
        ("prajjwal1/bert-tiny", 2),  # Binary classification
        ("prajjwal1/bert-tiny", 3),  # 3-way classification
        ("prajjwal1/bert-small", 2),  # Binary with larger model
    ],
)
def test_bert_sequence_classification_pcc(model_name, num_labels):
    """Test BertForSequenceClassification matches HuggingFace (PCC ≥ 0.99)."""
    validator = BERTSequenceClassificationValidator(
        model_name=model_name, num_labels=num_labels, batch_size=2, seq_len=32
    )

    results = validator.validate(num_runs=3)

    # Assert PCC ≥ 0.99 for all runs
    for idx, result in enumerate(results):
        pcc = result["pcc"]
        print(f"Run {idx + 1}: PCC = {pcc:.6f}")
        assert pcc >= 0.99, f"Run {idx + 1} failed: PCC {pcc:.6f} < 0.99"

    print(f"✓ All runs passed: PCC ≥ 0.99 for {model_name} with {num_labels} labels")


@pytest.mark.parametrize(
    "model_name,num_labels,batch_size,seq_len",
    [
        ("prajjwal1/bert-tiny", 2, 1, 32),  # Single sample
        ("prajjwal1/bert-tiny", 5, 4, 64),  # Larger batch and sequence
        ("prajjwal1/bert-small", 10, 2, 128),  # Many labels
    ],
)
def test_bert_sequence_classification_shapes(model_name, num_labels, batch_size, seq_len):
    """Test BertForSequenceClassification output shapes are correct."""
    validator = BERTSequenceClassificationValidator(
        model_name=model_name, num_labels=num_labels, batch_size=batch_size, seq_len=seq_len
    )

    # Single validation run
    np.random.seed(42)
    torch.manual_seed(42)

    input_ids_np = np.random.randint(0, validator.config.vocab_size, (batch_size, seq_len))
    token_type_ids_np = np.random.randint(0, 2, (batch_size, seq_len))
    attention_mask_np = np.ones((batch_size, seq_len), dtype=np.int64)

    result = validator.validate_single_run(input_ids_np, token_type_ids_np, attention_mask_np, seed=42)

    # Check shapes
    assert result["hf_logits"].shape == (batch_size, num_labels)
    assert result["ttml_logits"].shape == (batch_size, num_labels)

    print(f"✓ Shape validation passed: ({batch_size}, {num_labels}) for {model_name}")


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running BERT Sequence Classification validation...")
    test_bert_sequence_classification_pcc("prajjwal1/bert-tiny", 2)
    test_bert_sequence_classification_pcc("prajjwal1/bert-tiny", 3)
    test_bert_sequence_classification_shapes("prajjwal1/bert-tiny", 2, 2, 32)
    print("\n✓ All tests passed!")
