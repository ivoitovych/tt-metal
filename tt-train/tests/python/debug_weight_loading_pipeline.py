#!/usr/bin/env python3
"""
Debug script to trace weight loading corruption through the entire pipeline.

This script tests each stage of weight loading to identify where corruption occurs:
1. Safetensors file reading
2. Python float vector
3. core::from_vector storage
4. to_numpy() retrieval
"""

import numpy as np
import torch
import transformers
from pathlib import Path
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

from safetensors import safe_open


def analyze_weights(weights: np.ndarray, name: str):
    """Print detailed statistics about weights."""
    print(f"\n{name}:")
    print(f"  Shape: {weights.shape}")
    print(f"  Dtype: {weights.dtype}")
    print(f"  Mean: {weights.mean():.6f}")
    print(f"  Std: {weights.std():.6f}")
    print(f"  Min: {weights.min():.6f}")
    print(f"  Max: {weights.max():.6f}")
    print(f"  First 10 values: {weights.flat[:10]}")
    print(f"  [0,0]: {weights.reshape(-1, weights.shape[-1])[0, 0]}")


def test_roundtrip():
    """Test if TTML can store and retrieve weights correctly."""
    print("=" * 80)
    print("TESTING TTML ROUNDTRIP (store → retrieve)")
    print("=" * 80)

    # Create known test data
    test_data = np.random.randn(768, 768).astype(np.float32)
    print("\nOriginal test data:")
    analyze_weights(test_data, "Test Data (Random)")

    # Create a simple TTML parameter
    test_tensor = ttml.autograd.Tensor.from_numpy(test_data.reshape(1, 1, 768, 768))

    # Retrieve it back
    retrieved = test_tensor.to_numpy()
    retrieved_2d = retrieved.reshape(768, 768)

    print("\nRetrieved from TTML:")
    analyze_weights(retrieved_2d, "Retrieved Data")

    # Compare
    diff = np.abs(test_data - retrieved_2d)
    print(f"\nRoundtrip comparison:")
    print(f"  Mean diff: {diff.mean():.6e}")
    print(f"  Max diff: {diff.max():.6e}")
    print(f"  Match (atol=1e-5): {np.allclose(test_data, retrieved_2d, atol=1e-5)}")

    return np.allclose(test_data, retrieved_2d, atol=1e-5)


def test_token_embeddings():
    """Test token embedding loading step by step."""
    print("\n" + "=" * 80)
    print("TESTING TOKEN EMBEDDING LOADING PIPELINE")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"

    # Load HF model
    print(f"\nLoading HuggingFace model: {model_name}")
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_config = hf_model.config

    # Get HF token embeddings
    hf_embeddings = hf_model.embeddings.word_embeddings.weight.detach().numpy()
    print("\nHuggingFace token embeddings:")
    analyze_weights(hf_embeddings, "HF Embeddings")

    # Load from safetensors directly
    safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
    if not safetensors_path.exists():
        from safetensors.torch import save_file

        save_file(hf_model.state_dict(), str(safetensors_path))

    with safe_open(str(safetensors_path), framework="numpy") as f:
        st_embeddings = f.get_tensor("embeddings.word_embeddings.weight")
        print("\nSafetensors token embeddings:")
        analyze_weights(st_embeddings, "Safetensors Embeddings")

        # Verify safetensors matches HF
        print(f"\nSafetensors vs HF match: {np.allclose(st_embeddings, hf_embeddings, atol=1e-5)}")

    # Create TTML model and load
    print("\nCreating TTML model...")
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

    bert = ttml.models.bert.create(ttml_config)

    # BEFORE loading weights
    params_before = bert.parameters()
    ttml_emb_before = params_before["bert/token_embeddings/weight"].to_numpy()
    print("\nTTML embeddings BEFORE loading:")
    analyze_weights(ttml_emb_before, "TTML Before")

    # Load weights
    print("\nLoading weights from safetensors...")
    bert.load_model_from_safetensors(str(safetensors_path))

    # AFTER loading weights
    params_after = bert.parameters()
    ttml_emb_after = params_after["bert/token_embeddings/weight"].to_numpy()
    print("\nTTML embeddings AFTER loading:")
    analyze_weights(ttml_emb_after, "TTML After")

    # Compare
    # TTML pads vocab, so compare only HF vocab size
    ttml_emb_cropped = ttml_emb_after.reshape(-1, hf_config.hidden_size)[: hf_config.vocab_size]

    diff = np.abs(hf_embeddings - ttml_emb_cropped)
    print(f"\nTTML vs HF comparison:")
    print(f"  Mean diff: {diff.mean():.6e}")
    print(f"  Max diff: {diff.max():.6e}")
    print(
        f"  Std ratio: {ttml_emb_cropped.std():.6f} / {hf_embeddings.std():.6f} = {ttml_emb_cropped.std() / hf_embeddings.std():.4f}"
    )

    # Check specific values
    print(f"\nPoint-by-point comparison:")
    for i in range(5):
        for j in range(5):
            print(
                f"  [{i},{j}]: HF={hf_embeddings[i,j]:.6f}, TTML={ttml_emb_cropped[i,j]:.6f}, diff={abs(hf_embeddings[i,j]-ttml_emb_cropped[i,j]):.6f}"
            )


def test_qkv_weights():
    """Test QKV weight loading step by step."""
    print("\n" + "=" * 80)
    print("TESTING QKV WEIGHT LOADING PIPELINE")
    print("=" * 80)

    model_name = "prajjwal1/bert-tiny"

    # Load HF model
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_config = hf_model.config

    # Get HF Q, K, V weights
    hf_q = hf_model.encoder.layer[0].attention.self.query.weight.detach().numpy()
    hf_k = hf_model.encoder.layer[0].attention.self.key.weight.detach().numpy()
    hf_v = hf_model.encoder.layer[0].attention.self.value.weight.detach().numpy()

    print("\nHuggingFace Q, K, V weights:")
    analyze_weights(hf_q, "HF Q")
    analyze_weights(hf_k, "HF K")
    analyze_weights(hf_v, "HF V")

    # Expected concatenation
    expected_qkv = np.concatenate([hf_q, hf_k, hf_v], axis=0)
    print("\nExpected QKV (cat dim=0):")
    analyze_weights(expected_qkv, "Expected QKV")

    # Load from safetensors
    safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")

    with safe_open(str(safetensors_path), framework="numpy") as f:
        st_q = f.get_tensor("encoder.layer.0.attention.self.query.weight")
        st_k = f.get_tensor("encoder.layer.0.attention.self.key.weight")
        st_v = f.get_tensor("encoder.layer.0.attention.self.value.weight")

        print("\nSafetensors Q, K, V:")
        analyze_weights(st_q, "ST Q")

        # Verify safetensors matches HF
        print(f"\nSafetensors Q vs HF Q match: {np.allclose(st_q, hf_q, atol=1e-5)}")

    # Create TTML model and load
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

    bert = ttml.models.bert.create(ttml_config)
    bert.load_model_from_safetensors(str(safetensors_path))

    # Get TTML QKV
    params = bert.parameters()
    ttml_qkv = params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"].to_numpy()
    ttml_qkv_2d = ttml_qkv.reshape(3 * hf_config.hidden_size, hf_config.hidden_size)

    print("\nTTML QKV:")
    analyze_weights(ttml_qkv_2d, "TTML QKV")

    # Compare
    diff = np.abs(expected_qkv - ttml_qkv_2d)
    print(f"\nTTML vs Expected QKV comparison:")
    print(f"  Mean diff: {diff.mean():.6e}")
    print(f"  Max diff: {diff.max():.6e}")
    print(
        f"  Std ratio: {ttml_qkv_2d.std():.6f} / {expected_qkv.std():.6f} = {ttml_qkv_2d.std() / expected_qkv.std():.4f}"
    )

    # Check specific values
    print(f"\nPoint-by-point comparison:")
    print(f"  Q section [0,0]: Expected={expected_qkv[0,0]:.6f}, TTML={ttml_qkv_2d[0,0]:.6f}")
    print(f"  K section [128,0]: Expected={expected_qkv[128,0]:.6f}, TTML={ttml_qkv_2d[128,0]:.6f}")
    print(f"  V section [256,0]: Expected={expected_qkv[256,0]:.6f}, TTML={ttml_qkv_2d[256,0]:.6f}")


if __name__ == "__main__":
    print("BERT WEIGHT LOADING PIPELINE DEBUGGING")
    print("=" * 80)

    # Test 1: Basic roundtrip
    roundtrip_ok = test_roundtrip()
    print(f"\n{'✅' if roundtrip_ok else '❌'} Roundtrip test: {'PASS' if roundtrip_ok else 'FAIL'}")

    # Test 2: Token embeddings
    test_token_embeddings()

    # Test 3: QKV weights
    test_qkv_weights()

    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)
