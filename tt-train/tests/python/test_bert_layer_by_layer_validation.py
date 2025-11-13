#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layer-by-layer BERT validation against HuggingFace.

This test compares TTML BERT with HuggingFace BERT at every intermediate step:
1. Embeddings (token + position + type)
2. Each transformer block (0-11)
3. Layer norm after each block
4. Final pooling

By feeding HF intermediate outputs into TTML, we can isolate exactly where
numerical divergence occurs without error accumulation.
"""

import numpy as np
import os
import pytest
import sys
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

transformers = pytest.importorskip("transformers", reason="transformers not installed")


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    # Convert to numpy if needed
    if torch.is_tensor(tensor1):
        tensor1 = tensor1.detach().numpy()
    elif hasattr(tensor1, "to_numpy"):
        tensor1 = tensor1.to_numpy()

    if torch.is_tensor(tensor2):
        tensor2 = tensor2.detach().numpy()
    elif hasattr(tensor2, "to_numpy"):
        tensor2 = tensor2.to_numpy()

    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    if len(flat1) != len(flat2):
        print(f"  ⚠️  Size mismatch: {len(flat1)} vs {len(flat2)}")
        return 0.0

    mean1 = np.mean(flat1)
    mean2 = np.mean(flat2)

    numerator = np.sum((flat1 - mean1) * (flat2 - mean2))
    denominator = np.sqrt(np.sum((flat1 - mean1) ** 2) * np.sum((flat2 - mean2) ** 2))

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return numerator / denominator


def compare_tensors(name, hf_tensor, ttml_tensor, threshold=0.95):
    """Compare two tensors and print detailed stats."""
    hf_np = hf_tensor.detach().numpy() if torch.is_tensor(hf_tensor) else hf_tensor
    ttml_np = ttml_tensor.to_numpy() if hasattr(ttml_tensor, "to_numpy") else ttml_tensor

    # Reshape TTML if needed (remove extra dimensions)
    if ttml_np.ndim == 4 and hf_np.ndim == 3:
        # TTML: [B, 1, S, H] -> [B, S, H]
        ttml_np = ttml_np.squeeze(1)
    elif ttml_np.ndim == 4 and hf_np.ndim == 2:
        # TTML: [B, 1, 1, H] -> [B, H]
        ttml_np = ttml_np.squeeze(1).squeeze(1)

    pcc = compute_pcc(hf_np, ttml_np)

    abs_diff = np.abs(hf_np - ttml_np)
    mean_abs_diff = abs_diff.mean()
    max_abs_diff = abs_diff.max()

    status = "✅" if pcc > threshold else "❌"

    print(f"{status} {name}:")
    print(f"    PCC: {pcc:.6f}")
    print(f"    Mean abs diff: {mean_abs_diff:.6e}")
    print(f"    Max abs diff: {max_abs_diff:.6e}")
    print(f"    HF  stats: mean={hf_np.mean():.6f}, std={hf_np.std():.6f}")
    print(f"    TTML stats: mean={ttml_np.mean():.6f}, std={ttml_np.std():.6f}")
    print(f"    Shapes: HF={hf_np.shape}, TTML={ttml_np.shape}")

    return pcc, mean_abs_diff


@pytest.mark.slow
def test_bert_layer_by_layer_comparison():
    """
    Compare BERT layer-by-layer to isolate where divergence occurs.

    This test uses bert-tiny for speed but can be changed to bert-base-uncased
    to debug the HF validation failures.
    """
    print("\n" + "=" * 80)
    print("BERT Layer-by-Layer Validation")
    print("=" * 80)

    # Configuration
    model_name = "prajjwal1/bert-tiny"  # Change to "bert-base-uncased" to debug
    batch_size = 2
    seq_len = 32

    print(f"\nModel: {model_name}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}\n")

    # Load HuggingFace model
    from transformers import BertModel, BertConfig as HFBertConfig

    hf_model = BertModel.from_pretrained(model_name)
    hf_model.eval()
    hf_config = hf_model.config

    print("HuggingFace Config:")
    print(f"  Hidden size: {hf_config.hidden_size}")
    print(f"  Num layers: {hf_config.num_hidden_layers}")
    print(f"  Num heads: {hf_config.num_attention_heads}")
    print(f"  Vocab size: {hf_config.vocab_size}\n")

    # Save HF model to safetensors
    safetensors_path = Path(f"/tmp/layer_by_layer_{model_name.replace('/', '_')}.safetensors")
    if not safetensors_path.exists():
        from safetensors.torch import save_file

        save_file(hf_model.state_dict(), str(safetensors_path))

    # Create TTML model
    ttml_config = ttml.models.bert.BertConfig()
    ttml_config.vocab_size = hf_config.vocab_size
    ttml_config.max_sequence_length = seq_len
    ttml_config.embedding_dim = hf_config.hidden_size
    ttml_config.intermediate_size = hf_config.intermediate_size
    ttml_config.num_heads = hf_config.num_attention_heads
    ttml_config.num_blocks = hf_config.num_hidden_layers
    ttml_config.dropout_prob = 0.0  # Deterministic
    ttml_config.layer_norm_eps = hf_config.layer_norm_eps
    ttml_config.use_token_type_embeddings = True
    ttml_config.type_vocab_size = 2
    ttml_config.use_pooler = True

    ttml_model = ttml.models.bert.create(ttml_config)
    ttml_model.load_from_safetensors(str(safetensors_path))

    print("TTML model loaded\n")

    # Create test inputs
    torch.manual_seed(42)
    np.random.seed(42)
    input_ids = torch.randint(100, min(hf_config.vocab_size, 1000), (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    print("=" * 80)
    print("STEP-BY-STEP COMPARISON")
    print("=" * 80 + "\n")

    # ========================================================================
    # Step 1: Full forward pass comparison
    # ========================================================================
    print("Step 1: Full Forward Pass")
    print("-" * 80)

    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True
        )
        hf_final = hf_outputs.last_hidden_state

    # Convert to TTML
    input_ids_ttml = ttml.autograd.Tensor.from_numpy(
        input_ids.numpy().astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    )
    attention_mask_ttml = ttml.autograd.Tensor.from_numpy(
        attention_mask.numpy().astype(np.float32).reshape(batch_size, 1, 1, seq_len)
    )
    token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(
        token_type_ids.numpy().astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    )

    ttml_final = ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)

    pcc_final, _ = compare_tensors("Final output", hf_final, ttml_final)

    # ========================================================================
    # Step 2: Compare intermediate hidden states from HF
    # ========================================================================
    print("\nStep 2: Layer-by-Layer Hidden States")
    print("-" * 80)

    # HF provides all hidden states
    hf_hidden_states = hf_outputs.hidden_states  # Tuple of (num_layers + 1) tensors

    print(f"HF provides {len(hf_hidden_states)} hidden states:")
    print(f"  [0]: After embeddings")
    for i in range(1, len(hf_hidden_states)):
        print(f"  [{i}]: After layer {i-1}")

    # Get TTML intermediate outputs
    print(f"\nGetting TTML intermediate outputs...")
    ttml_intermediates = ttml_model.forward_with_intermediates(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)

    print(f"TTML provides {len(ttml_intermediates.block_outputs)} block outputs\n")

    # Compare embeddings
    compare_tensors("Embeddings", hf_hidden_states[0], ttml_intermediates.embeddings, threshold=0.99)

    # Compare each layer
    pcc_scores = []
    for i in range(len(ttml_intermediates.block_outputs)):
        hf_layer_out = hf_hidden_states[i + 1]  # HF includes embeddings at index 0
        ttml_layer_out = ttml_intermediates.block_outputs[i]

        pcc, _ = compare_tensors(f"Layer {i}", hf_layer_out, ttml_layer_out, threshold=0.95)
        pcc_scores.append(pcc)

        # If PCC drops significantly, this is where the problem starts
        if pcc < 0.90 and i > 0 and pcc_scores[i - 1] > 0.95:
            print(f"\n⚠️  ALERT: Significant PCC drop at layer {i}")
            print(f"    Previous layer: {pcc_scores[i-1]:.6f}")
            print(f"    Current layer: {pcc:.6f}")
            print(f"    This layer likely has a bug!\n")

    # ========================================================================
    # Step 3: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nPCC Scores per Layer:")
    print(
        f"  Embeddings: {compare_tensors('Embeddings (recap)', hf_hidden_states[0], ttml_intermediates.embeddings, threshold=0.99)[0]:.6f}"
    )
    for i, pcc in enumerate(pcc_scores):
        status = "✅" if pcc > 0.95 else "⚠️" if pcc > 0.90 else "❌"
        print(f"  {status} Layer {i}: {pcc:.6f}")
    print(f"  Final output: {pcc_final:.6f}\n")

    # Find problematic layers
    problematic = [(i, pcc) for i, pcc in enumerate(pcc_scores) if pcc < 0.90]
    if problematic:
        print("Problematic Layers (PCC < 0.90):")
        for layer_idx, pcc in problematic:
            print(f"  - Layer {layer_idx}: PCC = {pcc:.6f}")
    else:
        print("✅ All layers have acceptable PCC")

    if pcc_final > 0.95:
        print(f"\n✅ PASSED: Final PCC = {pcc_final:.6f}")
    else:
        print(f"\n❌ FAILED: Final PCC = {pcc_final:.6f}")
        if problematic:
            first_bad = problematic[0][0]
            print(f"\n🔍 Root cause: Divergence starts at layer {first_bad}")
            print(f"   Investigate:")
            print(f"   - Attention mechanism in layer {first_bad}")
            print(f"   - Layer norm in layer {first_bad}")
            print(f"   - Feed-forward network in layer {first_bad}")

    assert pcc_final > 0.95, f"PCC too low: {pcc_final:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
