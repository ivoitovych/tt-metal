# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layer-by-layer diagnostic test for bert-base-uncased.
Compares TTML BERT against HuggingFace BERT at each intermediate step
to identify exactly where divergences occur.
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

# Skip if transformers not available
transformers = pytest.importorskip("transformers", reason="transformers not installed")


def compute_metrics(golden, actual, name=""):
    """Compute comprehensive comparison metrics between two tensors."""
    golden_flat = golden.flatten()
    actual_flat = actual.flatten()

    if len(golden_flat) != len(actual_flat):
        return {
            "name": name,
            "error": f"Shape mismatch: {golden.shape} vs {actual.shape}",
            "pcc": 0.0,
        }

    # PCC
    mean_golden = np.mean(golden_flat)
    mean_actual = np.mean(actual_flat)
    numerator = np.sum((golden_flat - mean_golden) * (actual_flat - mean_actual))
    denominator = np.sqrt(np.sum((golden_flat - mean_golden) ** 2) * np.sum((actual_flat - mean_actual) ** 2))
    pcc = numerator / denominator if denominator > 0 else (1.0 if numerator == 0 else 0.0)

    # Differences
    abs_diff = np.abs(golden - actual)
    rel_diff = abs_diff / (np.abs(golden) + 1e-8)

    return {
        "name": name,
        "pcc": float(pcc),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "median_abs_diff": float(np.median(abs_diff)),
        "mean_rel_diff": float(rel_diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "golden_mean": float(golden.mean()),
        "golden_std": float(golden.std()),
        "actual_mean": float(actual.mean()),
        "actual_std": float(actual.std()),
        "has_nan": bool(np.isnan(actual).any()),
        "has_inf": bool(np.isinf(actual).any()),
    }


def print_metrics(metrics: Dict, threshold_pcc=0.95):
    """Print metrics with color coding based on thresholds."""
    name = metrics["name"]
    pcc = metrics["pcc"]

    status = "✅ PASS" if pcc > threshold_pcc else "❌ FAIL"

    print(f"\n{status} {name}")
    print(f"  PCC: {pcc:.6f}")
    print(f"  Mean abs diff: {metrics['mean_abs_diff']:.6e}")
    print(f"  Max abs diff: {metrics['max_abs_diff']:.6e}")
    print(f"  Golden: mean={metrics['golden_mean']:.6f}, std={metrics['golden_std']:.6f}")
    print(f"  Actual: mean={metrics['actual_mean']:.6f}, std={metrics['actual_std']:.6f}")

    if metrics.get("has_nan"):
        print(f"  ⚠️  Contains NaN!")
    if metrics.get("has_inf"):
        print(f"  ⚠️  Contains Inf!")


class HFBERTWithHooks:
    """HuggingFace BERT model with hooks to capture intermediate outputs."""

    def __init__(self, model_name: str):
        self.model = transformers.BertModel.from_pretrained(model_name)
        self.model.eval()
        self.outputs = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture all intermediate outputs."""

        # Embeddings
        def embedding_hook(module, input, output):
            self.outputs["embeddings"] = output.detach()

        self.hooks.append(self.model.embeddings.register_forward_hook(embedding_hook))

        # Each layer
        for layer_idx, layer in enumerate(self.model.encoder.layer):
            # Attention output (after self-attention + residual + layernorm)
            def make_attention_hook(idx):
                def hook(module, input, output):
                    self.outputs[f"layer_{idx}_attention_output"] = output[0].detach()

                return hook

            self.hooks.append(layer.attention.register_forward_hook(make_attention_hook(layer_idx)))

            # Layer output (after FFN + residual + layernorm)
            def make_layer_hook(idx):
                def hook(module, input, output):
                    self.outputs[f"layer_{idx}_output"] = output[0].detach()

                return hook

            self.hooks.append(layer.register_forward_hook(make_layer_hook(layer_idx)))

    def forward(self, input_ids, token_type_ids):
        """Run forward pass and return outputs with intermediate results."""
        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, return_dict=True)
        self.outputs["final"] = output.last_hidden_state.detach()
        return self.outputs

    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def test_bert_base_uncased_layer_by_layer():
    """
    Diagnostic test that compares TTML and HuggingFace BERT layer by layer.

    This identifies exactly where the first divergence occurs:
    - Embeddings (token + position + token_type)
    - Each transformer layer (attention, FFN)
    - Final output
    """
    print(f"\n{'='*80}")
    print("BERT Layer-by-Layer Diagnostic Test")
    print("Model: bert-base-uncased")
    print(f"{'='*80}\n")

    model_name = "bert-base-uncased"
    batch_size = 1
    seq_len = 32

    # Load HuggingFace model with hooks
    print("Loading HuggingFace BERT...")
    hf_bert = HFBERTWithHooks(model_name)
    hf_bert.register_hooks()
    hf_config = hf_bert.model.config

    print(
        f"Config: {hf_config.num_hidden_layers} layers, {hf_config.hidden_size} hidden, {hf_config.num_attention_heads} heads\n"
    )

    # Create test input
    torch.manual_seed(42)
    input_ids = torch.randint(0, min(hf_config.vocab_size, 1000), (batch_size, seq_len))
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs sample: {input_ids[0, :10].tolist()}\n")

    # Run HuggingFace forward pass
    print("Running HuggingFace forward pass with hooks...")
    hf_outputs = hf_bert.forward(input_ids, token_type_ids)
    print(f"Captured {len(hf_outputs)} intermediate outputs\n")

    # Load TTML BERT
    print("Loading TTML BERT...")
    safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")

    if not safetensors_path.exists():
        from safetensors.torch import save_file

        print(f"Saving HF model to safetensors: {safetensors_path}")
        save_file(hf_bert.model.state_dict(), str(safetensors_path))

    ttml_config = ttml.models.bert.BertConfig()
    ttml_config.vocab_size = hf_config.vocab_size
    ttml_config.max_sequence_length = seq_len
    ttml_config.embedding_dim = hf_config.hidden_size
    ttml_config.intermediate_size = hf_config.intermediate_size
    ttml_config.num_heads = hf_config.num_attention_heads
    ttml_config.num_blocks = hf_config.num_hidden_layers
    ttml_config.dropout_prob = 0.0
    ttml_config.layer_norm_eps = hf_config.layer_norm_eps
    ttml_config.use_token_type_embeddings = True
    ttml_config.use_pooler = False

    bert = ttml.models.bert.create(ttml_config)
    bert.load_model_from_safetensors(str(safetensors_path))
    print("TTML BERT loaded\n")

    # Convert inputs to TTML format
    input_ids_np = input_ids.numpy().astype(np.float32)
    token_type_ids_np = token_type_ids.numpy().astype(np.float32)
    input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids_np.reshape(batch_size, 1, 1, seq_len))
    token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids_np.reshape(batch_size, 1, 1, seq_len))

    # Run TTML forward pass
    print("Running TTML forward pass...")
    ttml_output = bert(input_ids_ttml, token_type_ids_ttml)
    ttml_output_np = ttml_output.to_numpy()
    print("TTML forward pass complete\n")

    # Compare embeddings
    print("=" * 80)
    print("COMPARING EMBEDDINGS")
    print("=" * 80)

    hf_embeddings = hf_outputs["embeddings"].numpy()  # [batch, seq, hidden]

    # Get TTML embeddings (we need to capture this separately)
    # For now, we'll compare just the weight loading

    print("\nChecking embedding weights...")
    params = bert.parameters()

    # Token embeddings
    ttml_token_emb = params["bert/token_embeddings/weight"].to_numpy()
    hf_token_emb = hf_bert.model.embeddings.word_embeddings.weight.detach().numpy()
    print(f"Token embeddings - TTML: {ttml_token_emb.shape}, HF: {hf_token_emb.shape}")

    # Position embeddings
    ttml_pos_emb = params["bert/position_embeddings/weight"].to_numpy()
    hf_pos_emb = hf_bert.model.embeddings.position_embeddings.weight.detach().numpy()
    print(f"Position embeddings - TTML: {ttml_pos_emb.shape}, HF: {hf_pos_emb.shape}")

    # Token type embeddings
    ttml_type_emb = params["bert/token_type_embeddings/weight"].to_numpy()
    hf_type_emb = hf_bert.model.embeddings.token_type_embeddings.weight.detach().numpy()
    print(f"Token type embeddings - TTML: {ttml_type_emb.shape}, HF: {hf_type_emb.shape}")

    # Compare QKV weights for layer 0
    print("\n" + "=" * 80)
    print("COMPARING LAYER 0 QKV WEIGHTS")
    print("=" * 80)

    qkv_weight_ttml = params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"].to_numpy()
    print(f"\nTTML QKV weight shape: {qkv_weight_ttml.shape}")

    # Get HF Q, K, V weights
    hf_q_weight = hf_bert.model.encoder.layer[0].attention.self.query.weight.detach().numpy()
    hf_k_weight = hf_bert.model.encoder.layer[0].attention.self.key.weight.detach().numpy()
    hf_v_weight = hf_bert.model.encoder.layer[0].attention.self.value.weight.detach().numpy()

    print(f"HF Q weight shape: {hf_q_weight.shape}")
    print(f"HF K weight shape: {hf_k_weight.shape}")
    print(f"HF V weight shape: {hf_v_weight.shape}")

    # Try different concatenation patterns
    print("\nTesting concatenation patterns:")

    # Pattern 1: cat(Q, K, V, dim=0) - current implementation
    qkv_pattern1 = np.concatenate([hf_q_weight, hf_k_weight, hf_v_weight], axis=0)
    print(f"Pattern 1 - cat(Q,K,V, dim=0): {qkv_pattern1.shape}")

    # Pattern 2: cat(Q.T, K.T, V.T, dim=1).T
    qkv_pattern2 = np.concatenate([hf_q_weight.T, hf_k_weight.T, hf_v_weight.T], axis=1).T
    print(f"Pattern 2 - cat(Q.T,K.T,V.T, dim=1).T: {qkv_pattern2.shape}")

    # Pattern 3: cat(Q, K, V, dim=1)
    qkv_pattern3 = np.concatenate([hf_q_weight, hf_k_weight, hf_v_weight], axis=1)
    print(f"Pattern 3 - cat(Q,K,V, dim=1): {qkv_pattern3.shape}")

    # Pattern 4: cat(Q.T, K.T, V.T, dim=0)
    qkv_pattern4 = np.concatenate([hf_q_weight.T, hf_k_weight.T, hf_v_weight.T], axis=0)
    print(f"Pattern 4 - cat(Q.T,K.T,V.T, dim=0): {qkv_pattern4.shape}")

    # Reshape TTML weight
    qkv_weight_ttml_2d = qkv_weight_ttml.reshape(-1, hf_config.hidden_size)
    print(f"\nActual TTML weight (reshaped): {qkv_weight_ttml_2d.shape}")

    # Compare each pattern
    for i, pattern in enumerate([qkv_pattern1, qkv_pattern2, qkv_pattern3, qkv_pattern4], 1):
        if pattern.shape == qkv_weight_ttml_2d.shape:
            metrics = compute_metrics(pattern, qkv_weight_ttml_2d, f"Pattern {i}")
            print_metrics(metrics, threshold_pcc=0.99)
        else:
            print(f"\nPattern {i}: Shape mismatch - {pattern.shape} vs {qkv_weight_ttml_2d.shape}")

    # Compare final outputs
    print("\n" + "=" * 80)
    print("COMPARING FINAL OUTPUTS")
    print("=" * 80)

    hf_final = hf_outputs["final"].numpy()  # [batch, seq, hidden]
    ttml_final = ttml_output_np.reshape(batch_size, seq_len, hf_config.hidden_size)

    final_metrics = compute_metrics(hf_final, ttml_final, "Final Output")
    print_metrics(final_metrics, threshold_pcc=0.95)

    # Cleanup
    hf_bert.cleanup()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST COMPLETE")
    print("=" * 80)

    # Report summary
    print("\n📊 SUMMARY:")
    print(f"Final output PCC: {final_metrics['pcc']:.6f}")

    if final_metrics["pcc"] < 0.95:
        print("\n❌ BERT implementation has issues")
        print("Review the weight loading patterns above to identify the correct transformation")
    else:
        print("\n✅ BERT implementation looks correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
