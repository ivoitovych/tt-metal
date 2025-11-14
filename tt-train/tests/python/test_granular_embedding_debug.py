#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Granular Embedding Decomposition Test

Purpose: Identify which specific embedding component introduces the error that
         causes PCC degradation in end-to-end BERT execution.

Tests each intermediate embedding tensor:
1. Word (token) embeddings
2. After adding positional embeddings
3. Token type embeddings (if used)
4. After adding token type embeddings
5. After LayerNorm
6. After dropout (final embeddings)
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml  # noqa: E402

from transformers import BertModel  # noqa: E402


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    # Handle both torch tensors and numpy arrays
    if isinstance(tensor1, torch.Tensor):
        tensor1 = tensor1.detach().numpy()
    if hasattr(tensor2, "to_numpy"):
        tensor2 = tensor2.to_numpy()
    elif isinstance(tensor2, ttml.autograd.Tensor):
        tensor2 = tensor2.to_numpy()

    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    mean1 = np.mean(flat1)
    mean2 = np.mean(flat2)

    numerator = np.sum((flat1 - mean1) * (flat2 - mean2))
    denominator = np.sqrt(np.sum((flat1 - mean1) ** 2) * np.sum((flat2 - mean2) ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def test_granular_embedding_decomposition():
    """
    Test each embedding component individually to find where error is introduced.
    """
    print("\n" + "=" * 80)
    print("GRANULAR EMBEDDING DECOMPOSITION TEST")
    print("=" * 80)
    print("\nPurpose: Identify which embedding component introduces PCC degradation")
    print("Expected: Find the exact operation causing PCC < 0.999\n")

    # Test configurations
    model_name = "prajjwal1/bert-tiny"
    batch_size = 2  # Use batch_size=2 since that's where we see degradation
    seq_len = 32

    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}\n")

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    hf_model = BertModel.from_pretrained(model_name)
    hf_model.eval()
    hf_config = hf_model.config

    # Save to safetensors
    safetensors_path = Path(f"/tmp/granular_debug_{model_name.replace('/', '_')}.safetensors")
    if not safetensors_path.exists():
        from safetensors.torch import save_file

        save_file(hf_model.state_dict(), str(safetensors_path))

    # Create TTML model
    print("Creating TTML model...")
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
    ttml_config.type_vocab_size = 2

    ttml_model = ttml.models.bert.create(ttml_config)
    ttml_model.load_model_from_safetensors(str(safetensors_path))
    print("TTML model loaded\n")

    # Create test inputs
    torch.manual_seed(42)
    np.random.seed(42)
    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Token type IDs shape: {token_type_ids.shape}\n")

    # Get HuggingFace embedding components
    print("=" * 80)
    print("EXTRACTING HUGGINGFACE EMBEDDING COMPONENTS")
    print("=" * 80 + "\n")

    with torch.no_grad():
        embeddings_module = hf_model.embeddings

        # 1. Word embeddings
        word_embeddings = embeddings_module.word_embeddings(input_ids)
        print(f"1. Word embeddings shape: {word_embeddings.shape}")

        # 2. Position embeddings
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = embeddings_module.position_embeddings(position_ids)
        print(f"2. Position embeddings shape: {position_embeddings.shape}")

        # Word + position
        after_position = word_embeddings + position_embeddings
        print(f"3. After adding position: {after_position.shape}")

        # 3. Token type embeddings
        token_type_embeddings = embeddings_module.token_type_embeddings(token_type_ids)
        print(f"4. Token type embeddings shape: {token_type_embeddings.shape}")

        # Word + position + token_type
        after_token_type = after_position + token_type_embeddings
        print(f"5. After adding token type: {after_token_type.shape}")

        # 4. After LayerNorm
        after_layer_norm = embeddings_module.LayerNorm(after_token_type)
        print(f"6. After LayerNorm: {after_layer_norm.shape}")

        # 5. After dropout (final) - dropout is no-op during eval
        final_embeddings = embeddings_module.dropout(after_layer_norm)
        print(f"7. Final embeddings (after dropout): {final_embeddings.shape}\n")

    # Get TTML embedding components
    print("=" * 80)
    print("EXTRACTING TTML EMBEDDING COMPONENTS")
    print("=" * 80 + "\n")

    input_ids_ttml = ttml.autograd.Tensor.from_numpy(
        input_ids.numpy().astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    )
    token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(
        token_type_ids.numpy().astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    )

    ttml_intermediates = ttml_model.get_embeddings_with_intermediates(input_ids_ttml, token_type_ids_ttml)

    print("TTML intermediate tensors retrieved:")
    print(f"  - word_embeddings: {ttml_intermediates.word_embeddings is not None}")
    print(f"  - after_position: {ttml_intermediates.after_position is not None}")
    print(f"  - token_type_embeddings: {ttml_intermediates.token_type_embeddings is not None}")
    print(f"  - after_token_type: {ttml_intermediates.after_token_type is not None}")
    print(f"  - after_layer_norm: {ttml_intermediates.after_layer_norm is not None}")
    print(f"  - after_dropout: {ttml_intermediates.after_dropout is not None}\n")

    # Compare each component
    print("=" * 80)
    print("COMPONENT-BY-COMPONENT PCC COMPARISON")
    print("=" * 80 + "\n")

    results = []

    # Helper to compute and display PCC
    def compare_component(name, hf_tensor, ttml_tensor, threshold=0.999):
        pcc = compute_pcc(hf_tensor, ttml_tensor)
        mean_diff = np.mean(
            np.abs(
                hf_tensor.numpy()
                if isinstance(hf_tensor, torch.Tensor)
                else hf_tensor - (ttml_tensor.to_numpy() if hasattr(ttml_tensor, "to_numpy") else ttml_tensor)
            )
        )
        max_diff = np.max(
            np.abs(
                hf_tensor.numpy()
                if isinstance(hf_tensor, torch.Tensor)
                else hf_tensor - (ttml_tensor.to_numpy() if hasattr(ttml_tensor, "to_numpy") else ttml_tensor)
            )
        )

        passed = pcc >= threshold
        status = "✅" if passed else "❌"

        print(f"{status} {name}:")
        print(f"   PCC: {pcc:.6f}")
        print(f"   Mean abs diff: {mean_diff:.6e}")
        print(f"   Max abs diff: {max_diff:.6e}")

        if not passed:
            print(f"   ⚠️  DEGRADATION DETECTED AT THIS STAGE!")

        print()

        results.append({"component": name, "pcc": pcc, "passed": passed, "mean_diff": mean_diff, "max_diff": max_diff})

        return passed

    # Test each component
    compare_component("1. Word (token) embeddings", word_embeddings, ttml_intermediates.word_embeddings)
    compare_component("2. After adding position embeddings", after_position, ttml_intermediates.after_position)
    compare_component(
        "3. Token type embeddings (standalone)", token_type_embeddings, ttml_intermediates.token_type_embeddings
    )
    compare_component("4. After adding token type", after_token_type, ttml_intermediates.after_token_type)
    compare_component("5. After LayerNorm", after_layer_norm, ttml_intermediates.after_layer_norm)
    compare_component("6. Final embeddings (after dropout)", final_embeddings, ttml_intermediates.after_dropout)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)

    print(f"Components passing (PCC ≥ 0.999): {passed_count}/{total_count}\n")

    # Find first failure
    first_failure = None
    for i, r in enumerate(results):
        if not r["passed"]:
            first_failure = r
            break

    if first_failure:
        print(f"🔍 FIRST FAILURE DETECTED:")
        print(f"   Component: {first_failure['component']}")
        print(f"   PCC: {first_failure['pcc']:.6f}")
        print(f"\n   This is where the error is introduced!\n")
    else:
        print("✅ All components passed! Error must be in layer processing, not embeddings.\n")

    # Show PCC progression
    print("PCC Progression:")
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {r['component']}: {r['pcc']:.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_granular_embedding_decomposition()
