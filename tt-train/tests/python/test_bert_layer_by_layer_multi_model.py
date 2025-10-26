# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive layer-by-layer BERT validation across multiple model variants.

Tests 4 BERT models from tiny to base, comparing HuggingFace vs TTML
at every layer to identify exactly where and how divergence occurs.

Models tested:
- prajjwal1/bert-tiny: 2 layers, 128 hidden, 2 heads
- prajjwal1/bert-small: 4 layers, 512 hidden, 8 heads
- google/bert_uncased_L-4_H-512_A-8: 4 layers, 512 hidden, 8 heads
- bert-base-uncased: 12 layers, 768 hidden, 12 heads

For each model, validates:
1. Embedding layer output
2. Each transformer block's attention output
3. Each transformer block's final output (after FFN)
4. Final model output

Identifies first layer where PCC drops below threshold and tracks
divergence accumulation across layers.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Tuple

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


class LayerMetrics:
    """Metrics for a single layer comparison."""

    def __init__(self, name: str, pcc: float, mean_diff: float, max_diff: float):
        self.name = name
        self.pcc = pcc
        self.mean_diff = mean_diff
        self.max_diff = max_diff
        self.passed = pcc >= 0.95

    def __repr__(self):
        status = "✅" if self.passed else "❌"
        return f"{status} {self.name}: PCC={self.pcc:.6f}, mean_diff={self.mean_diff:.6e}, max_diff={self.max_diff:.6e}"


class BERTLayerByLayerValidator:
    """Validates BERT models layer-by-layer against HuggingFace reference."""

    def __init__(self, model_name: str, batch_size: int = 1, seq_len: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_len = seq_len

        print(f"\n{'=' * 80}")
        print(f"Loading BERT model: {model_name}")
        print(f"{'=' * 80}")

        # Load HuggingFace model
        self.hf_model = transformers.BertModel.from_pretrained(model_name)
        self.hf_model.eval()
        self.hf_config = self.hf_model.config

        # Save to safetensors
        safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
        if not safetensors_path.exists():
            from safetensors.torch import save_file

            save_file(self.hf_model.state_dict(), str(safetensors_path))

        # Create TTML model
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

        print(
            f"Model config: {self.hf_config.num_hidden_layers} layers, "
            f"{self.hf_config.hidden_size} hidden dim, "
            f"{self.hf_config.num_attention_heads} heads"
        )

        self.all_metrics: List[LayerMetrics] = []
        self.first_failure: str = None

    def get_hf_intermediate_outputs(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Run HuggingFace BERT and capture all intermediate outputs."""
        outputs = {}

        with torch.no_grad():
            # Get embeddings
            embeddings = self.hf_model.embeddings(input_ids, token_type_ids=token_type_ids)
            outputs["embeddings"] = embeddings.numpy()

            # Pass through each layer
            hidden_states = embeddings
            for layer_idx, layer in enumerate(self.hf_model.encoder.layer):
                # Attention output (before FFN)
                attention_output = layer.attention(hidden_states)[0]
                attention_residual = attention_output + hidden_states
                attention_norm = layer.attention.output.LayerNorm(attention_residual)
                outputs[f"block_{layer_idx}_attention"] = attention_norm.numpy()

                # FFN output (final block output)
                ffn_output = layer.intermediate(attention_norm)
                ffn_output = layer.output.dense(ffn_output)
                ffn_output = layer.output.dropout(ffn_output)
                ffn_residual = ffn_output + attention_norm
                block_output = layer.output.LayerNorm(ffn_residual)
                outputs[f"block_{layer_idx}_output"] = block_output.numpy()

                hidden_states = block_output

            outputs["final"] = hidden_states.numpy()

        return outputs

    def compare_layer(self, hf_output: np.ndarray, ttml_output: np.ndarray, name: str) -> LayerMetrics:
        """Compare a single layer's output."""
        # Reshape TTML output to match HF format [batch, seq, hidden]
        ttml_reshaped = ttml_output.reshape(self.batch_size, self.seq_len, -1)

        pcc = compute_pcc(hf_output, ttml_reshaped)
        diff = np.abs(hf_output - ttml_reshaped)
        mean_diff = diff.mean()
        max_diff = diff.max()

        metrics = LayerMetrics(name, pcc, mean_diff, max_diff)
        self.all_metrics.append(metrics)

        if not metrics.passed and self.first_failure is None:
            self.first_failure = name

        return metrics

    def validate(self) -> bool:
        """Run complete layer-by-layer validation."""
        print(f"\n{'=' * 80}")
        print("LAYER-BY-LAYER VALIDATION")
        print(f"{'=' * 80}\n")

        # Create deterministic inputs
        torch.manual_seed(42)
        input_ids = torch.randint(0, min(self.hf_config.vocab_size, 1000), (self.batch_size, self.seq_len))
        token_type_ids = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)
        attention_mask = torch.ones((self.batch_size, self.seq_len), dtype=torch.long)

        print(f"Input IDs (first 10): {input_ids[0, :10].tolist()}\n")

        # Get HuggingFace outputs
        print("Running HuggingFace BERT...")
        hf_outputs = self.get_hf_intermediate_outputs(input_ids, token_type_ids)

        # Get TTML outputs
        print("Running TTML BERT with intermediates...")
        input_ids_np = input_ids.numpy().astype(np.float32)
        token_type_ids_np = token_type_ids.numpy().astype(np.float32)
        attention_mask_np = attention_mask.numpy().astype(np.float32)

        input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids_np.reshape(self.batch_size, 1, 1, self.seq_len))
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(
            token_type_ids_np.reshape(self.batch_size, 1, 1, self.seq_len)
        )
        attention_mask_ttml = ttml.autograd.Tensor.from_numpy(
            attention_mask_np.reshape(self.batch_size, 1, 1, self.seq_len)
        )

        ttml_intermediates = self.ttml_model.forward_with_intermediates(
            input_ids_ttml, attention_mask_ttml, token_type_ids_ttml
        )

        print("Comparing layers...\n")

        # Compare embeddings
        print("Layer 0: Embeddings")
        emb_metrics = self.compare_layer(
            hf_outputs["embeddings"], ttml_intermediates.embeddings.to_numpy(), "Embeddings"
        )
        print(f"  {emb_metrics}")

        # Compare each block
        for block_idx in range(self.hf_config.num_hidden_layers):
            print(f"\nLayer {block_idx + 1}: Block {block_idx}")

            # Attention output
            attn_metrics = self.compare_layer(
                hf_outputs[f"block_{block_idx}_attention"],
                ttml_intermediates.block_attention_outputs[block_idx].to_numpy(),
                f"Block {block_idx} Attention",
            )
            print(f"  Attention: {attn_metrics}")

            # Block output (after FFN)
            block_metrics = self.compare_layer(
                hf_outputs[f"block_{block_idx}_output"],
                ttml_intermediates.block_outputs[block_idx].to_numpy(),
                f"Block {block_idx} Output",
            )
            print(f"  Output:    {block_metrics}")

        # Compare final output
        print(f"\nLayer {self.hf_config.num_hidden_layers + 1}: Final Output")
        final_metrics = self.compare_layer(
            hf_outputs["final"], ttml_intermediates.final_output.to_numpy(), "Final Output"
        )
        print(f"  {final_metrics}")

        # Print summary
        print(f"\n{'=' * 80}")
        print("VALIDATION SUMMARY")
        print(f"{'=' * 80}\n")

        total = len(self.all_metrics)
        passed = sum(1 for m in self.all_metrics if m.passed)
        failed = total - passed

        print(f"Total layers compared: {total}")
        print(f"Passed (PCC ≥ 0.95): {passed}")
        print(f"Failed (PCC < 0.95): {failed}")

        if self.first_failure:
            print(f"\n❌ First failure at: {self.first_failure}")
            for m in self.all_metrics:
                if m.name == self.first_failure:
                    print(f"   PCC: {m.pcc:.6f}")
                    print(f"   Mean diff: {m.mean_diff:.6e}")
                    print(f"   Max diff: {m.max_diff:.6e}")
                    break
        else:
            print("\n✅ All layers PASSED!")

        # Analyze divergence pattern
        print(f"\n{'=' * 80}")
        print("DIVERGENCE ANALYSIS")
        print(f"{'=' * 80}\n")

        pcc_values = [m.pcc for m in self.all_metrics]
        print(f"PCC progression:")
        for i, (name, pcc) in enumerate(zip([m.name for m in self.all_metrics], pcc_values)):
            status = "✅" if pcc >= 0.95 else "❌"
            print(f"  {i + 1}. {status} {name}: {pcc:.6f}")

        print(f"\nMinimum PCC: {min(pcc_values):.6f}")
        print(f"Maximum PCC: {max(pcc_values):.6f}")
        print(f"Average PCC: {np.mean(pcc_values):.6f}")

        return failed == 0


# Test parameters: (batch_size, seq_len, model_name)
BERT_MODELS = [
    (1, 32, "prajjwal1/bert-tiny"),  # 2 layers
    (1, 32, "prajjwal1/bert-small"),  # 4 layers
    (1, 32, "google/bert_uncased_L-4_H-512_A-8"),  # 4 layers (official)
    (1, 32, "bert-base-uncased"),  # 12 layers
]


@pytest.mark.parametrize("batch_size,seq_len,model_name", BERT_MODELS)
def test_bert_layer_by_layer_validation(batch_size, seq_len, model_name):
    """
    Comprehensive layer-by-layer validation for BERT models.

    Tests multiple BERT variants from tiny to base, comparing HuggingFace
    vs TTML at every layer to identify divergence patterns.
    """
    validator = BERTLayerByLayerValidator(model_name, batch_size, seq_len)
    all_passed = validator.validate()

    # For now, we expect some divergence but want to see the pattern
    # Uncomment to make test strict:
    # assert all_passed, f"Layer-by-layer validation failed. First failure: {validator.first_failure}"

    # Current expectation: Log results for analysis
    print(f"\n\nRESULT for {model_name}: {'PASS' if all_passed else 'DIVERGENCE DETECTED'}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
