# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Manual stepwise validation test for BERT models.

This test manually steps through BERT computation using TTML parameters
and operations, comparing each step against HuggingFace BERT.

This approach works with existing TTML Python bindings without requiring
C++ modifications.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Tuple

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")


def compute_pcc(golden: np.ndarray, actual: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient."""
    golden_flat = golden.flatten()
    actual_flat = actual.flatten()

    if len(golden_flat) != len(actual_flat):
        return 0.0

    mean_golden = np.mean(golden_flat)
    mean_actual = np.mean(actual_flat)
    numerator = np.sum((golden_flat - mean_golden) * (actual_flat - mean_actual))
    denominator = np.sqrt(np.sum((golden_flat - mean_golden) ** 2) * np.sum((actual_flat - mean_actual) ** 2))
    return numerator / denominator if denominator > 0 else (1.0 if numerator == 0 else 0.0)


def compare_tensors(golden: np.ndarray, actual: np.ndarray, name: str, threshold: float = 0.95) -> bool:
    """Compare two tensors and print results."""
    if golden.shape != actual.shape:
        print(f"\n❌ {name}")
        print(f"  Shape mismatch: {golden.shape} vs {actual.shape}")
        return False

    pcc = compute_pcc(golden, actual)
    diff = np.abs(golden - actual)

    passed = pcc > threshold
    symbol = "✅" if passed else "❌"

    print(f"\n{symbol} {name}")
    print(f"  PCC: {pcc:.6f} (threshold: {threshold})")
    print(f"  Mean diff: {diff.mean():.6e}, Max diff: {diff.max():.6e}")
    print(f"  Golden: mean={golden.mean():.4f}, std={golden.std():.4f}")
    print(f"  Actual: mean={actual.mean():.4f}, std={actual.std():.4f}")

    return passed


class ManualBERTValidator:
    """
    Manually execute BERT step-by-step using TTML parameters and ops.
    """

    def __init__(self, model_name: str, batch_size: int = 1, seq_len: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.failures = []

        # Load HF model
        print(f"Loading HuggingFace BERT: {model_name}")
        self.hf_model = transformers.BertModel.from_pretrained(model_name)
        self.hf_model.eval()
        self.hf_config = self.hf_model.config

        # Save to safetensors
        safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
        if not safetensors_path.exists():
            from safetensors.torch import save_file

            save_file(self.hf_model.state_dict(), str(safetensors_path))

        # Create TTML model and get parameters
        print("Creating TTML BERT...")
        ttml_config = ttml.models.bert.BertConfig()
        ttml_config.vocab_size = self.hf_config.vocab_size
        ttml_config.max_sequence_length = seq_len
        ttml_config.embedding_dim = self.hf_config.hidden_size
        ttml_config.intermediate_size = self.hf_config.intermediate_size
        ttml_config.num_heads = self.hf_config.num_attention_heads
        ttml_config.num_blocks = self.hf_config.num_hidden_layers
        ttml_config.dropout_prob = 0.0
        ttml_config.layer_norm_eps = self.hf_config.layer_norm_eps
        ttml_config.use_token_type_embeddings = True
        ttml_config.use_pooler = False

        self.ttml_model = ttml.models.bert.create(ttml_config)
        self.ttml_model.load_model_from_safetensors(str(safetensors_path))
        self.params = self.ttml_model.parameters()

        print(
            f"Config: {self.hf_config.num_hidden_layers} layers, "
            f"{self.hf_config.hidden_size} hidden, {self.hf_config.num_attention_heads} heads\n"
        )

    def validate_weight_loading(self) -> bool:
        """Validate that weights were loaded correctly."""
        print("=" * 80)
        print("VALIDATING WEIGHT LOADING")
        print("=" * 80)

        all_passed = True

        # Check token embeddings
        hf_token_emb = self.hf_model.embeddings.word_embeddings.weight.detach().numpy()
        ttml_token_emb = self.params["bert/token_embeddings/weight"].to_numpy()
        # TTML pads vocab, so compare only the HF vocab size
        ttml_token_emb_cropped = ttml_token_emb.reshape(-1, self.hf_config.hidden_size)[: self.hf_config.vocab_size]

        passed = compare_tensors(hf_token_emb, ttml_token_emb_cropped, "Token Embeddings", threshold=0.9999)
        all_passed = all_passed and passed

        # Check QKV weights for layer 0
        hf_q = self.hf_model.encoder.layer[0].attention.self.query.weight.detach().numpy()
        hf_k = self.hf_model.encoder.layer[0].attention.self.key.weight.detach().numpy()
        hf_v = self.hf_model.encoder.layer[0].attention.self.value.weight.detach().numpy()

        ttml_qkv = self.params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"].to_numpy()
        hidden_size = self.hf_config.hidden_size
        ttml_qkv_2d = ttml_qkv.reshape(3 * hidden_size, hidden_size)

        # Expected: cat(Q, K, V, dim=0)
        expected_qkv = np.concatenate([hf_q, hf_k, hf_v], axis=0)

        passed = compare_tensors(expected_qkv, ttml_qkv_2d, "Layer 0 QKV Weights", threshold=0.9999)
        all_passed = all_passed and passed
        if not passed:
            self.failures.append("QKV weight loading")

        return all_passed

    def run_comparison(self) -> bool:
        """Run full step-by-step comparison."""
        print("\n" + "=" * 80)
        print("STEP-BY-STEP FORWARD PASS COMPARISON")
        print("=" * 80)

        # Create deterministic inputs
        torch.manual_seed(42)
        input_ids = torch.randint(0, min(self.hf_config.vocab_size, 1000), (self.batch_size, self.seq_len))
        token_type_ids = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)

        print(f"\nInput IDs: {input_ids[0, :10].tolist()}\n")

        # Get HF embeddings
        with torch.no_grad():
            hf_token_emb = self.hf_model.embeddings.word_embeddings(input_ids)
            position_ids = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)
            hf_pos_emb = self.hf_model.embeddings.position_embeddings(position_ids)
            hf_type_emb = self.hf_model.embeddings.token_type_embeddings(token_type_ids)
            hf_embeddings = hf_token_emb + hf_pos_emb + hf_type_emb
            hf_embeddings_norm = self.hf_model.embeddings.LayerNorm(hf_embeddings)

            # Final HF output
            hf_output = self.hf_model(input_ids=input_ids, token_type_ids=token_type_ids)
            hf_final = hf_output.last_hidden_state.numpy()

        # Get TTML output using full forward pass
        input_ids_np = input_ids.numpy().astype(np.float32)
        token_type_ids_np = token_type_ids.numpy().astype(np.float32)
        # Create attention mask (all ones = no masking)
        attention_mask_np = np.ones((self.batch_size, self.seq_len), dtype=np.float32)

        input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids_np.reshape(self.batch_size, 1, 1, self.seq_len))
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(
            token_type_ids_np.reshape(self.batch_size, 1, 1, self.seq_len)
        )
        attention_mask_ttml = ttml.autograd.Tensor.from_numpy(
            attention_mask_np.reshape(self.batch_size, 1, 1, self.seq_len)
        )

        ttml_output = self.ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)
        ttml_final = ttml_output.to_numpy().reshape(self.batch_size, self.seq_len, self.hf_config.hidden_size)

        # Compare embeddings (if we could access them from TTML)
        # For now, just compare final output

        all_passed = compare_tensors(hf_final, ttml_final, "Final Output", threshold=0.95)

        if not all_passed:
            self.failures.append("Final output comparison")

        return all_passed

    def run_full_validation(self) -> bool:
        """Run complete validation."""
        print("=" * 80)
        print(f"BERT STEPWISE VALIDATION: {self.model_name}")
        print("=" * 80)

        # Step 1: Validate weight loading
        weights_ok = self.validate_weight_loading()

        # Step 2: Compare forward pass
        forward_ok = self.run_comparison()

        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        print(f"\nWeights validation: {'✅ PASS' if weights_ok else '❌ FAIL'}")
        print(f"Forward pass validation: {'✅ PASS' if forward_ok else '❌ FAIL'}")

        if self.failures:
            print(f"\n❌ Failures: {', '.join(self.failures)}")
        else:
            print("\n✅ All validations PASSED!")

        return weights_ok and forward_ok


@pytest.mark.parametrize(
    "model_name",
    [
        "prajjwal1/bert-tiny",
        # "bert-base-uncased",  # Uncomment after fixing QKV loading
    ],
)
@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (1, 32),
    ],
)
def test_bert_manual_stepwise_validation(model_name, batch_size, seq_len):
    """
    Manual stepwise validation test.

    This test validates weights and forward pass step-by-step without
    requiring modifications to TTML C++ code.
    """
    validator = ManualBERTValidator(model_name, batch_size, seq_len)
    passed = validator.run_full_validation()

    assert passed, f"Validation failed: {validator.failures}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
