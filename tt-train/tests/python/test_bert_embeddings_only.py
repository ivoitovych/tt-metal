#!/usr/bin/env python3
"""Test BERT embeddings layer-by-layer to identify where divergence starts."""

import numpy as np
import os
import sys
from pathlib import Path
import torch
import transformers

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    mean_x = np.mean(x_flat)
    mean_y = np.mean(y_flat)
    numerator = np.sum((x_flat - mean_x) * (y_flat - mean_y))
    denominator = np.sqrt(np.sum((x_flat - mean_x) ** 2) * np.sum((y_flat - mean_y) ** 2))
    return numerator / denominator if denominator > 0 else 0.0


def test_embeddings():
    model_name = "prajjwal1/bert-tiny"
    test_text = "The quick brown fox."

    print(f"Testing BERT embeddings for: {model_name}")
    print(f"Input text: '{test_text}'")

    # Load HuggingFace model
    hf_model = transformers.BertModel.from_pretrained(model_name)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    hf_config = hf_model.config

    # Tokenize
    encoded = tokenizer(test_text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    input_ids = encoded["input_ids"]
    token_type_ids = torch.zeros_like(input_ids)

    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0, :10].tolist()}")

    # Load TTML model
    safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
    if not safetensors_path.exists():
        from safetensors.torch import save_file

        save_file(hf_model.state_dict(), str(safetensors_path))

    ttml_config = ttml.models.bert.BertConfig()
    ttml_config.vocab_size = hf_config.vocab_size
    ttml_config.max_sequence_length = 32
    ttml_config.embedding_dim = hf_config.hidden_size
    ttml_config.intermediate_size = hf_config.intermediate_size
    ttml_config.num_heads = hf_config.num_attention_heads
    ttml_config.num_blocks = hf_config.num_hidden_layers
    ttml_config.dropout_prob = 0.0
    ttml_config.layer_norm_eps = hf_config.layer_norm_eps
    ttml_config.use_token_type_embeddings = True
    ttml_config.use_pooler = False

    ttml_model = ttml.models.bert.create(ttml_config)
    ttml_model.load_model_from_safetensors(str(safetensors_path))

    print(f"\n{'='*80}")
    print("STEP-BY-STEP EMBEDDING COMPARISON")
    print(f"{'='*80}")

    with torch.no_grad():
        # Step 1: Token embeddings only
        print(f"\n1. Token Embeddings:")
        hf_token_emb = hf_model.embeddings.word_embeddings(input_ids)
        print(f"   HF shape: {hf_token_emb.shape}")
        print(f"   HF mean: {hf_token_emb.mean():.6f}, std: {hf_token_emb.std():.6f}")
        print(f"   HF first 5: {hf_token_emb[0, 0, :5].numpy()}")

        # Step 2: Add position embeddings
        print(f"\n2. Token + Position Embeddings:")
        hf_pos_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        hf_pos_emb = hf_model.embeddings.position_embeddings(hf_pos_ids)
        hf_combined = hf_token_emb + hf_pos_emb
        print(f"   HF mean: {hf_combined.mean():.6f}, std: {hf_combined.std():.6f}")
        print(f"   HF first 5: {hf_combined[0, 0, :5].numpy()}")

        # Step 3: Add token type embeddings
        print(f"\n3. Token + Position + Type Embeddings:")
        hf_type_emb = hf_model.embeddings.token_type_embeddings(token_type_ids)
        hf_combined_all = hf_combined + hf_type_emb
        print(f"   HF mean: {hf_combined_all.mean():.6f}, std: {hf_combined_all.std():.6f}")
        print(f"   HF first 5: {hf_combined_all[0, 0, :5].numpy()}")

        # Step 4: Apply LayerNorm
        print(f"\n4. After Embedding LayerNorm:")
        hf_normed = hf_model.embeddings.LayerNorm(hf_combined_all)
        print(f"   HF mean: {hf_normed.mean():.6f}, std: {hf_normed.std():.6f}")
        print(f"   HF first 5: {hf_normed[0, 0, :5].numpy()}")
        print(f"   HF epsilon: {hf_model.embeddings.LayerNorm.eps}")

        # Step 5: Get TTML embeddings (including LayerNorm)
        print(f"\n5. TTML Full Embeddings (including LayerNorm):")
        input_ids_np = input_ids.numpy().astype(np.float32)
        token_type_ids_np = token_type_ids.numpy().astype(np.float32)

        input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids_np.reshape(1, 1, 1, 32))
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids_np.reshape(1, 1, 1, 32))

        # Get just embeddings (before first attention block)
        # We'll need to run full forward and compare
        ttml_output_full = ttml_model(input_ids_ttml, token_type_ids_ttml)
        ttml_output_np = ttml_output_full.to_numpy().reshape(1, 32, hf_config.hidden_size)

        # Get HF full output for comparison
        hf_output_full = hf_model(input_ids=input_ids, token_type_ids=token_type_ids)
        hf_output_np = hf_output_full.last_hidden_state.numpy()

        print(f"\n{'='*80}")
        print("EMBEDDING OUTPUT COMPARISON")
        print(f"{'='*80}")
        print(f"\nAfter Embedding LayerNorm:")
        print(f"  HF:   mean={hf_normed.mean():.6f}, std={hf_normed.std():.6f}")
        print(f"  HF first 5: {hf_normed[0, 0, :5].numpy()}")

        print(f"\nAfter Full Forward Pass:")
        print(f"  HF:   mean={hf_output_np.mean():.6f}, std={hf_output_np.std():.6f}")
        print(f"  TTML: mean={ttml_output_np.mean():.6f}, std={ttml_output_np.std():.6f}")
        print(f"  HF first 5:   {hf_output_np[0, 0, :5]}")
        print(f"  TTML first 5: {ttml_output_np[0, 0, :5]}")
        pcc = compute_pcc(hf_output_np, ttml_output_np)
        print(f"  PCC: {pcc:.6f} {'✅' if pcc > 0.95 else '❌'}")

        # Check variance and mean of embeddings before LayerNorm
        print(f"\n{'='*80}")
        print("LAYERNORM INPUT ANALYSIS")
        print(f"{'='*80}")
        print(f"Embeddings before LayerNorm (combined token+pos+type):")
        print(f"  HF variance along last dim (first token): {hf_combined_all[0, 0, :].var().item():.6e}")
        print(f"  This matters because LayerNorm divides by sqrt(variance + eps)")
        print(f"  With eps={hf_model.embeddings.LayerNorm.eps:.2e}, the effect is:")
        print(
            f"    sqrt(variance + eps) = sqrt({hf_combined_all[0, 0, :].var().item():.6e} + {hf_model.embeddings.LayerNorm.eps:.2e})"
        )
        print(
            f"                         = {np.sqrt(hf_combined_all[0, 0, :].var().item() + hf_model.embeddings.LayerNorm.eps):.6f}"
        )

        # If we used 1e-4 instead:
        wrong_eps = 1e-4
        print(f"\n  If eps was {wrong_eps:.2e} (hardware clamped):")
        print(f"    sqrt(variance + eps) = sqrt({hf_combined_all[0, 0, :].var().item():.6e} + {wrong_eps:.2e})")
        print(f"                         = {np.sqrt(hf_combined_all[0, 0, :].var().item() + wrong_eps):.6f}")
        print(
            f"  Difference in normalization: {(np.sqrt(hf_combined_all[0, 0, :].var().item() + wrong_eps) - np.sqrt(hf_combined_all[0, 0, :].var().item() + hf_model.embeddings.LayerNorm.eps)) / np.sqrt(hf_combined_all[0, 0, :].var().item() + hf_model.embeddings.LayerNorm.eps) * 100:.3f}%"
        )


if __name__ == "__main__":
    test_embeddings()
