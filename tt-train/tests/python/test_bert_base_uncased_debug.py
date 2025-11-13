#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Debug bert-base-uncased specifically to find HF validation failures.

This uses the layer-by-layer approach to pinpoint where bert-base-uncased diverges.
"""

import sys

sys.path.insert(0, "tests/python")

# Import and run test_bert_layer_by_layer_validation but with bert-base-uncased
from test_bert_layer_by_layer_validation import *

# Override model_name
import pytest


@pytest.mark.slow
def test_bert_base_uncased_layer_by_layer():
    """Run layer-by-layer comparison with bert-base-uncased."""
    import numpy as np
    import os
    import torch
    from pathlib import Path

    sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
    import _ttml as ttml  # noqa: E402

    from transformers import BertModel

    print("\n" + "=" * 80)
    print("BERT BASE UNCASED - Layer-by-Layer Debug")
    print("=" * 80)

    model_name = "bert-base-uncased"  # THE PROBLEMATIC MODEL
    batch_size = 2
    seq_len = 32

    print(f"\nModel: {model_name}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}\n")

    # Load HuggingFace model
    hf_model = BertModel.from_pretrained(model_name)
    hf_model.eval()
    hf_config = hf_model.config

    print("HuggingFace Config:")
    print(f"  Hidden size: {hf_config.hidden_size}")
    print(f"  Num layers: {hf_config.num_hidden_layers}")
    print(f"  Num heads: {hf_config.num_attention_heads}")
    print(f"  Vocab size: {hf_config.vocab_size}\n")

    # Save HF model
    safetensors_path = Path(f"/tmp/debug_{model_name.replace('/', '_')}.safetensors")
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
    ttml_config.dropout_prob = 0.0
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
    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    # HF forward pass
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True
        )

    # TTML forward pass
    input_ids_ttml = ttml.autograd.Tensor.from_numpy(
        input_ids.numpy().astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    )
    attention_mask_ttml = ttml.autograd.Tensor.from_numpy(
        attention_mask.numpy().astype(np.float32).reshape(batch_size, 1, 1, seq_len)
    )
    token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(
        token_type_ids.numpy().astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    )

    ttml_intermediates = ttml_model.forward_with_intermediates(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)

    # Compare layer by layer
    print("=" * 80)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 80 + "\n")

    hf_hidden_states = hf_outputs.hidden_states

    # Embeddings
    print("Comparing Embeddings...")
    pcc_emb = compute_pcc(hf_hidden_states[0], ttml_intermediates.embeddings)
    print(f"  PCC: {pcc_emb:.6f}\n")

    # Each layer
    pcc_scores = []
    for i in range(len(ttml_intermediates.block_outputs)):
        hf_layer = hf_hidden_states[i + 1]
        ttml_layer = ttml_intermediates.block_outputs[i]

        pcc = compute_pcc(hf_layer, ttml_layer)
        pcc_scores.append(pcc)

        status = "✅" if pcc > 0.95 else "⚠️" if pcc > 0.90 else "❌"
        print(f"{status} Layer {i}: PCC = {pcc:.6f}")

        if pcc < 0.50:  # Very low PCC
            print(f"   ⚠️ CRITICAL: Layer {i} has severe divergence!")
            hf_np = hf_layer.detach().numpy()
            ttml_np = ttml_layer.to_numpy()

            # Reshape for comparison
            if ttml_np.ndim == 4:
                ttml_np = ttml_np.squeeze(1)

            print(f"   HF mean: {hf_np.mean():.6f}, std: {hf_np.std():.6f}")
            print(f"   TTML mean: {ttml_np.mean():.6f}, std: {ttml_np.std():.6f}")
            print(f"   Are signs flipped? {(hf_np.mean() * ttml_np.mean()) < 0}")

            # Check if it's a simple sign flip
            pcc_flipped = compute_pcc(hf_layer, -ttml_layer.to_numpy())
            print(f"   PCC if signs flipped: {pcc_flipped:.6f}\n")
            break  # Stop at first critical failure

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY FOR BERT-BASE-UNCASED")
    print("=" * 80)

    print(f"\nEmbeddings PCC: {pcc_emb:.6f}")
    print(f"Layer PCCs:")
    for i, pcc in enumerate(pcc_scores):
        status = "✅" if pcc > 0.95 else "⚠️" if pcc > 0.90 else "❌"
        print(f"  {status} Layer {i}: {pcc:.6f}")

    # Find first bad layer
    for i, pcc in enumerate(pcc_scores):
        if pcc < 0.90:
            print(f"\n🔍 First problematic layer: {i} (PCC = {pcc:.6f})")
            if i == 0:
                print("   Problem starts immediately after embeddings")
                print("   Check: Layer norm, attention mechanism")
            break


if __name__ == "__main__":
    test_bert_base_uncased_layer_by_layer()
