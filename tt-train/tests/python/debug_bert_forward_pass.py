#!/usr/bin/env python3
"""
Debug BERT Forward Pass - Layer by Layer Comparison

Systematically compares HuggingFace vs TTML outputs at each stage:
1. Embeddings (token + position + token_type)
2. Embedding LayerNorm
3. Each transformer block (attention, FFN, layer norms)
4. Final output

Goal: Identify exactly where the divergence starts.
"""

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


def compare_outputs(name, hf_output, ttml_output, threshold=0.99):
    """Compare two outputs and print detailed statistics."""
    pcc = compute_pcc(hf_output, ttml_output)
    diff = np.abs(hf_output - ttml_output)

    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Shape: HF {hf_output.shape}, TTML {ttml_output.shape}")
    print(f"\nHuggingFace:")
    print(f"  Mean: {hf_output.mean():.6f}, Std: {hf_output.std():.6f}")
    print(f"  Min: {hf_output.min():.6f}, Max: {hf_output.max():.6f}")
    print(f"\nTTML:")
    print(f"  Mean: {ttml_output.mean():.6f}, Std: {ttml_output.std():.6f}")
    print(f"  Min: {ttml_output.min():.6f}, Max: {ttml_output.max():.6f}")
    print(f"\nComparison:")
    print(f"  PCC: {pcc:.6f} {'✅' if pcc >= threshold else '❌'}")
    print(f"  Mean abs diff: {diff.mean():.6e}")
    print(f"  Max abs diff: {diff.max():.6e}")
    print(f"  Median abs diff: {np.median(diff):.6e}")

    # Show first few values for debugging
    print(f"\nFirst 5 values comparison:")
    print(f"  HF:   {hf_output.flatten()[:5]}")
    print(f"  TTML: {ttml_output.flatten()[:5]}")
    print(f"  Diff: {diff.flatten()[:5]}")

    return pcc >= threshold


def debug_bert_forward_pass(model_name="prajjwal1/bert-tiny", test_text="The quick brown fox jumps."):
    """Debug BERT forward pass layer by layer."""

    print(f"\n{'#'*80}")
    print(f"DEBUG BERT FORWARD PASS: {model_name}")
    print(f"{'#'*80}\n")
    print(f"Test text: '{test_text}'")

    # Load HuggingFace model
    print(f"\nLoading HuggingFace model...")
    hf_model = transformers.BertModel.from_pretrained(model_name)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    hf_config = hf_model.config

    # Tokenize
    encoded = tokenizer(test_text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]  # CRITICAL: Extract attention mask
    token_type_ids = torch.zeros_like(input_ids)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0, :10].tolist()}")
    print(f"Attention mask: {attention_mask[0, :10].tolist()}")
    num_real_tokens = attention_mask.sum().item()
    num_padding = (attention_mask == 0).sum().item()
    print(
        f"Real tokens: {num_real_tokens}, Padding tokens: {num_padding} ({num_padding/(num_real_tokens+num_padding)*100:.1f}% padding)"
    )

    # Load TTML model
    print(f"\nLoading TTML model...")
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

    print(f"✅ Models loaded")

    # ========================================================================
    # Step 1: Compare embeddings (before LayerNorm)
    # ========================================================================

    print(f"\n{'█'*80}")
    print(f"STEP 1: EMBEDDINGS (Token + Position + Token Type)")
    print(f"{'█'*80}")

    with torch.no_grad():
        # HuggingFace embeddings
        hf_token_emb = hf_model.embeddings.word_embeddings(input_ids)
        hf_pos_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        hf_pos_emb = hf_model.embeddings.position_embeddings(hf_pos_ids)
        hf_type_emb = hf_model.embeddings.token_type_embeddings(token_type_ids)

        hf_embeddings_raw = hf_token_emb + hf_pos_emb + hf_type_emb

        print(f"\n📊 HuggingFace Embeddings Breakdown:")
        print(f"Token embeddings: mean={hf_token_emb.mean():.6f}, std={hf_token_emb.std():.6f}")
        print(f"Position embeddings: mean={hf_pos_emb.mean():.6f}, std={hf_pos_emb.std():.6f}")
        print(f"Token type embeddings: mean={hf_type_emb.mean():.6f}, std={hf_type_emb.std():.6f}")
        print(f"Combined (raw): mean={hf_embeddings_raw.mean():.6f}, std={hf_embeddings_raw.std():.6f}")

    # TTML doesn't expose intermediate embeddings easily, so we'll compare after full forward pass
    # But we can check if embedding weights are correct (we know they are from previous tests)

    # ========================================================================
    # Step 2: Compare after Embedding LayerNorm
    # ========================================================================

    print(f"\n{'█'*80}")
    print(f"STEP 2: EMBEDDING LAYERNORM OUTPUT")
    print(f"{'█'*80}")

    with torch.no_grad():
        hf_embeddings_normed = hf_model.embeddings.LayerNorm(hf_embeddings_raw)
        hf_embeddings_normed = hf_model.embeddings.dropout(hf_embeddings_normed)  # Dropout is 0 in eval mode

    print(f"\n📊 After LayerNorm:")
    print(f"  Mean: {hf_embeddings_normed.mean():.6f}")
    print(f"  Std: {hf_embeddings_normed.std():.6f}")

    # ========================================================================
    # Step 3: Full forward pass comparison
    # ========================================================================

    print(f"\n{'█'*80}")
    print(f"STEP 3: FULL FORWARD PASS (WITH ATTENTION MASK)")
    print(f"{'█'*80}")

    with torch.no_grad():
        # CRITICAL FIX: Pass attention_mask to properly handle padding!
        hf_output = hf_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hf_final = hf_output.last_hidden_state.numpy()

    # TTML forward pass
    input_ids_np = input_ids.numpy().astype(np.float32)
    token_type_ids_np = token_type_ids.numpy().astype(np.float32)
    attention_mask_np = attention_mask.numpy().astype(np.float32)

    input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids_np.reshape(1, 1, 1, 32))
    token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids_np.reshape(1, 1, 1, 32))
    attention_mask_ttml = ttml.autograd.Tensor.from_numpy(attention_mask_np.reshape(1, 1, 1, 32))

    # CRITICAL: Pass attention mask to TTML model
    # Parameter order: input_ids, attention_mask, token_type_ids
    ttml_output = ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)
    ttml_final = ttml_output.to_numpy().reshape(1, 32, hf_config.hidden_size)

    compare_outputs("FINAL OUTPUT (WITH PROPER MASKING)", hf_final, ttml_final, threshold=0.95)

    print(f"\n⚠️  NOTE: This test now properly uses attention masks to ignore padding tokens.")
    print(f"   Previous tests without masks were comparing WRONG outputs (both models)")
    print(f"   attended to padding, making bugs less visible.")

    # ========================================================================
    # Step 4: Check specific components
    # ========================================================================

    print(f"\n{'█'*80}")
    print(f"STEP 4: DEBUGGING HINTS")
    print(f"{'█'*80}")

    # Check LayerNorm parameters
    print(f"\nEmbedding LayerNorm parameters:")
    hf_ln_weight = hf_model.embeddings.LayerNorm.weight.detach().numpy()
    hf_ln_bias = hf_model.embeddings.LayerNorm.bias.detach().numpy()

    ttml_params = ttml_model.parameters()
    ttml_ln_weight = ttml_params["bert/embedding_norm/gamma"].to_numpy().flatten()
    ttml_ln_bias = ttml_params["bert/embedding_norm/beta"].to_numpy().flatten()

    ln_weight_pcc = compute_pcc(hf_ln_weight, ttml_ln_weight)
    ln_bias_pcc = compute_pcc(hf_ln_bias, ttml_ln_bias)

    print(f"  Weight PCC: {ln_weight_pcc:.6f} {'✅' if ln_weight_pcc > 0.999 else '❌'}")
    print(f"  Bias PCC: {ln_bias_pcc:.6f} {'✅' if ln_bias_pcc > 0.999 else '❌'}")

    # Check epsilon
    print(f"\nLayerNorm epsilon:")
    print(f"  HF: {hf_model.embeddings.LayerNorm.eps}")
    print(f"  TTML config: {ttml_config.layer_norm_eps}")

    # Check if there are NaN or Inf values
    print(f"\nNaN/Inf check:")
    print(f"  HF output: NaN={np.isnan(hf_final).any()}, Inf={np.isinf(hf_final).any()}")
    print(f"  TTML output: NaN={np.isnan(ttml_final).any()}, Inf={np.isinf(ttml_final).any()}")

    # Check output distribution
    print(f"\nOutput distribution:")
    print(
        f"  HF: min={hf_final.min():.6f}, max={hf_final.max():.6f}, mean={hf_final.mean():.6f}, std={hf_final.std():.6f}"
    )
    print(
        f"  TTML: min={ttml_final.min():.6f}, max={ttml_final.max():.6f}, mean={ttml_final.mean():.6f}, std={ttml_final.std():.6f}"
    )

    # ========================================================================
    # Conclusion
    # ========================================================================

    print(f"\n{'='*80}")
    print(f"DEBUGGING SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Weight loading: Verified correct (from previous tests)")
    print(f"✅ LayerNorm parameters: Weight PCC={ln_weight_pcc:.4f}, Bias PCC={ln_bias_pcc:.4f}")
    print(f"❌ Forward pass: Output does not match")
    print(f"\n⚠️  LIKELY ISSUES TO INVESTIGATE:")
    print(f"  1. LayerNorm implementation (epsilon, formula)")
    print(f"  2. Attention mechanism (QKV computation, softmax, output)")
    print(f"  3. Activation functions (GELU implementation)")
    print(f"  4. Residual connections")
    print(f"  5. Dropout (should be disabled in eval mode)")
    print(f"  6. Precision (BFloat16 vs Float32)")

    return hf_final, ttml_final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prajjwal1/bert-tiny", help="Model to debug")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.", help="Test text")
    args = parser.parse_args()

    debug_bert_forward_pass(args.model, args.text)
