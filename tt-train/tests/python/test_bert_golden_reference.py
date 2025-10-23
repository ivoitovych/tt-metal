# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Golden reference test comparing TTML BERT against HuggingFace BERT.
This test validates that QKV weight loading correctly transforms HuggingFace's
separate Q, K, V weights into TTML's combined QKV format.
"""

import numpy as np
import pytest
import os
import sys
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml  # noqa: E402

# Skip if transformers not available
transformers = pytest.importorskip("transformers", reason="transformers not installed")


def compute_pcc(golden, actual):
    """Compute Pearson Correlation Coefficient between two tensors."""
    golden_flat = golden.flatten()
    actual_flat = actual.flatten()

    if len(golden_flat) != len(actual_flat):
        return 0.0

    mean_golden = np.mean(golden_flat)
    mean_actual = np.mean(actual_flat)

    numerator = np.sum((golden_flat - mean_golden) * (actual_flat - mean_actual))
    denominator = np.sqrt(np.sum((golden_flat - mean_golden) ** 2) * np.sum((actual_flat - mean_actual) ** 2))

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return numerator / denominator


def save_hf_bert_to_safetensors(model_name, output_path):
    """Download and save HuggingFace BERT model to safetensors format."""
    from safetensors.torch import save_file

    print(f"Loading HuggingFace BERT model: {model_name}")
    model = transformers.BertModel.from_pretrained(model_name)

    # Extract state dict
    state_dict = model.state_dict()

    # Save to safetensors
    print(f"Saving to: {output_path}")
    save_file(state_dict, output_path)

    return model


def get_hf_bert_output(model, input_ids, token_type_ids=None):
    """Get output from HuggingFace BERT model."""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, return_dict=True)
    return outputs.last_hidden_state


@pytest.mark.parametrize(
    "model_name",
    [
        "prajjwal1/bert-tiny",  # 2 layers, 128 hidden, 2 heads - small and fast
    ],
)
@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (1, 32),  # Minimal test case (must be multiple of TILE_HEIGHT=32)
        (2, 64),  # Small batch
    ],
)
def test_bert_qkv_loading_golden_reference(model_name, batch_size, seq_len):
    """
    Golden reference test that compares TTML BERT output against HuggingFace BERT.

    This validates:
    1. QKV weight loading correctly combines separate Q, K, V weights
    2. Forward pass produces numerically correct results
    3. The cat(Q, K, V, dim=0) transformation is correct
    """
    print(f"\n{'='*80}")
    print(f"Golden Reference Test: {model_name}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"{'='*80}\n")

    # Load HuggingFace model
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_config = hf_model.config

    print(f"HF Config:")
    print(f"  Vocab size: {hf_config.vocab_size}")
    print(f"  Hidden size: {hf_config.hidden_size}")
    print(f"  Num layers: {hf_config.num_hidden_layers}")
    print(f"  Num heads: {hf_config.num_attention_heads}")
    print(f"  Intermediate size: {hf_config.intermediate_size}")
    print(f"  Max position embeddings: {hf_config.max_position_embeddings}\n")

    # Create test input
    torch.manual_seed(42)
    input_ids = torch.randint(0, min(hf_config.vocab_size, 1000), (batch_size, seq_len))
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs sample: {input_ids[0, :5].tolist()}\n")

    # Get HuggingFace output (golden reference)
    print("Running HuggingFace BERT forward pass...")
    hf_output = get_hf_bert_output(hf_model, input_ids, token_type_ids)
    hf_output_np = hf_output.numpy()

    print(f"HF output shape: {hf_output_np.shape}")
    print(f"HF output stats:")
    print(f"  Mean: {hf_output_np.mean():.6f}")
    print(f"  Std: {hf_output_np.std():.6f}")
    print(f"  Min: {hf_output_np.min():.6f}")
    print(f"  Max: {hf_output_np.max():.6f}")
    print(f"  Contains NaN: {np.isnan(hf_output_np).any()}")
    print(f"  Contains Inf: {np.isinf(hf_output_np).any()}\n")

    # Save HuggingFace model to safetensors
    safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
    safetensors_path.parent.mkdir(parents=True, exist_ok=True)

    if not safetensors_path.exists():
        from safetensors.torch import save_file

        print(f"Saving HF model to safetensors: {safetensors_path}")
        save_file(hf_model.state_dict(), str(safetensors_path))

    # Create TTML BERT config matching HuggingFace config
    print("Creating TTML BERT model...")
    ttml_config = ttml.models.bert.BertConfig()
    ttml_config.vocab_size = hf_config.vocab_size
    ttml_config.max_sequence_length = seq_len  # Use actual test sequence length
    ttml_config.embedding_dim = hf_config.hidden_size
    ttml_config.intermediate_size = hf_config.intermediate_size
    ttml_config.num_heads = hf_config.num_attention_heads
    ttml_config.num_blocks = hf_config.num_hidden_layers
    ttml_config.dropout_prob = 0.0  # Disable for inference
    ttml_config.layer_norm_eps = hf_config.layer_norm_eps
    ttml_config.use_token_type_embeddings = True
    ttml_config.use_pooler = False

    print(f"TTML Config:")
    print(f"  Vocab size: {ttml_config.vocab_size}")
    print(f"  Embedding dim: {ttml_config.embedding_dim}")
    print(f"  Num blocks: {ttml_config.num_blocks}")
    print(f"  Num heads: {ttml_config.num_heads}")
    print(f"  Intermediate size: {ttml_config.intermediate_size}")
    print(f"  Layer norm eps: {ttml_config.layer_norm_eps}\n")

    # Create TTML BERT model
    bert = ttml.models.bert.create(ttml_config)

    # Load weights from safetensors
    print(f"Loading weights from safetensors into TTML BERT...")
    bert.load_model_from_safetensors(str(safetensors_path))
    print("Weights loaded successfully!\n")

    # Convert inputs to TTML tensors
    input_ids_np = input_ids.numpy().astype(np.float32)
    token_type_ids_np = token_type_ids.numpy().astype(np.float32)

    input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids_np.reshape(batch_size, 1, 1, seq_len))
    token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids_np.reshape(batch_size, 1, 1, seq_len))

    # Run TTML BERT forward pass
    print("Running TTML BERT forward pass...")
    ttml_output = bert(input_ids_ttml, token_type_ids_ttml)
    ttml_output_np = ttml_output.to_numpy()

    print(f"TTML output shape: {ttml_output_np.shape}")
    print(f"TTML output stats:")
    print(f"  Mean: {ttml_output_np.mean():.6f}")
    print(f"  Std: {ttml_output_np.std():.6f}")
    print(f"  Min: {ttml_output_np.min():.6f}")
    print(f"  Max: {ttml_output_np.max():.6f}")
    print(f"  Contains NaN: {np.isnan(ttml_output_np).any()}")
    print(f"  Contains Inf: {np.isinf(ttml_output_np).any()}\n")

    # Reshape TTML output to match HF output [batch, seq, hidden]
    # TTML output is [batch, 1, seq, hidden]
    ttml_output_reshaped = ttml_output_np.reshape(batch_size, seq_len, hf_config.hidden_size)

    # Compare outputs
    print("Comparing outputs...")

    # Check for NaN/Inf
    assert not np.isnan(ttml_output_reshaped).any(), "TTML output contains NaN"
    assert not np.isinf(ttml_output_reshaped).any(), "TTML output contains Inf"

    # Compute differences
    abs_diff = np.abs(hf_output_np - ttml_output_reshaped)
    rel_diff = abs_diff / (np.abs(hf_output_np) + 1e-8)

    print(f"Absolute difference:")
    print(f"  Mean: {abs_diff.mean():.6e}")
    print(f"  Max: {abs_diff.max():.6e}")
    print(f"  Median: {np.median(abs_diff):.6e}")

    print(f"Relative difference:")
    print(f"  Mean: {rel_diff.mean():.6e}")
    print(f"  Max: {rel_diff.max():.6e}")
    print(f"  Median: {np.median(rel_diff):.6e}\n")

    # Compute PCC
    pcc = compute_pcc(hf_output_np, ttml_output_reshaped)
    print(f"Pearson Correlation Coefficient: {pcc:.6f}\n")

    # Assertions
    # PCC should be very high (>0.99) for correct implementation
    assert pcc > 0.99, f"PCC too low: {pcc:.6f} (expected >0.99)"

    # Mean absolute error should be reasonable for bfloat16 precision
    # TTML uses bfloat16 internally, so we expect ~1e-3 error
    assert abs_diff.mean() < 1e-2, f"Mean absolute error too high: {abs_diff.mean():.6e}"

    print(f"{'='*80}")
    print("✅ Golden reference test PASSED!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run with: python test_bert_golden_reference.py
    pytest.main([__file__, "-v", "-s"])
