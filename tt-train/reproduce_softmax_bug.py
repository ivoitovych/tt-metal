#!/usr/bin/env python3
"""
Minimal Softmax Precision Bug Reproduction

This script demonstrates the BERT softmax precision bug with bfloat16 accumulation.

Bug: BERT models show PCC 0.81 with bfloat16 softmax accumulation
Workaround: FP32 accumulation restores PCC >0.95

CRITICAL: The workaround is enabled by default in unary_ops.cpp line ~107.
To reproduce the BUG, you must disable the workaround in the source code:
  1. Edit sources/ttml/ops/unary_ops.cpp line ~107
  2. Change: softmax(/* use_fp32_accumulation_workaround */ true)
  3. To: softmax(/* use_fp32_accumulation_workaround */ false)
  4. Rebuild TTML
  5. Re-run this script
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add TTML to path
sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources')
import ttml

from transformers import BertModel
from safetensors.torch import save_file


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    t1_flat = tensor1.flatten()
    t2_flat = tensor2.flatten()

    mean1 = np.mean(t1_flat)
    mean2 = np.mean(t2_flat)

    numerator = np.sum((t1_flat - mean1) * (t2_flat - mean2))
    denominator = np.sqrt(np.sum((t1_flat - mean1) ** 2) * np.sum((t2_flat - mean2) ** 2))

    if denominator == 0:
        return 1.0

    return numerator / denominator


def reproduce_bug(model_name="prajjwal1/bert-tiny", batch_size=1, seq_len=32):
    """
    Reproduce the softmax precision bug with full BERT model.

    Returns:
        dict with PCC results
    """
    print(f"\n{'='*80}")
    print(f"Testing BERT Model: {model_name}")
    print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}")
    print(f"{'='*80}\n")

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    hf_model = BertModel.from_pretrained(model_name)
    hf_model.eval()
    config = hf_model.config

    print(f"Model architecture:")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden dim: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Vocab size: {config.vocab_size}")

    # Save to safetensors
    safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
    if not safetensors_path.exists():
        print(f"Saving weights to {safetensors_path}")
        save_file(hf_model.state_dict(), str(safetensors_path))

    # Create TTML model
    print("\nCreating TTML model...")
    ttml_config = ttml.models.bert.BertConfig()
    ttml_config.vocab_size = config.vocab_size
    ttml_config.max_sequence_length = seq_len
    ttml_config.embedding_dim = config.hidden_size
    ttml_config.intermediate_size = config.intermediate_size
    ttml_config.num_heads = config.num_attention_heads
    ttml_config.num_blocks = config.num_hidden_layers
    ttml_config.dropout_prob = 0.0
    ttml_config.layer_norm_eps = config.layer_norm_eps
    ttml_config.use_token_type_embeddings = True
    ttml_config.use_pooler = False

    ttml_model = ttml.models.bert.create(ttml_config)
    ttml_model.load_model_from_safetensors(str(safetensors_path))
    print("Weights loaded successfully")

    # Create test inputs
    print("\nCreating test inputs...")
    np.random.seed(42)

    # Token IDs: random valid tokens
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=np.uint32)

    # Token type IDs: 0 for first half, 1 for second half
    token_type_ids = np.zeros((batch_size, seq_len), dtype=np.uint32)
    token_type_ids[:, seq_len // 2 :] = 1

    # Attention mask: all 1s (no padding)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.uint32)

    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  token_type_ids: {token_type_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")

    # HuggingFace forward pass (reference)
    print("\nRunning HuggingFace forward pass...")
    with torch.no_grad():
        input_ids_torch = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids_torch = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask_torch = torch.tensor(attention_mask, dtype=torch.long)

        hf_outputs = hf_model(
            input_ids=input_ids_torch,
            token_type_ids=token_type_ids_torch,
            attention_mask=attention_mask_torch,
        )
        hf_result = hf_outputs.last_hidden_state.cpu().numpy()

    print(f"HuggingFace output shape: {hf_result.shape}")
    print(f"HuggingFace output range: [{hf_result.min():.4f}, {hf_result.max():.4f}]")

    # TTML forward pass
    print("\nRunning TTML forward pass...")

    # Reshape inputs for TTML
    input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids.reshape(batch_size, 1, 1, seq_len))
    token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids.reshape(batch_size, 1, 1, seq_len))
    attention_mask_ttml = ttml.autograd.Tensor.from_numpy(
        attention_mask.astype(np.float32).reshape(batch_size, 1, 1, seq_len)
    )

    ttml_output = ttml_model(input_ids_ttml, attention_mask_ttml, token_type_ids_ttml)
    ttml_result = ttml_output.to_numpy()

    print(f"TTML output shape: {ttml_result.shape}")
    print(f"TTML output range: [{ttml_result.min():.4f}, {ttml_result.max():.4f}]")

    # Compute PCC
    pcc = compute_pcc(hf_result, ttml_result)

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"PCC: {pcc:.6f}")

    if pcc > 0.95:
        print("✅ PASS: PCC >0.95 (FP32 workaround is active)")
        status = "PASS_WITH_WORKAROUND"
    elif pcc > 0.85:
        print("⚠️  MARGINAL: 0.85 < PCC < 0.95")
        status = "MARGINAL"
    else:
        print("❌ FAIL: PCC <0.85 (Bug reproduced! FP32 workaround not active)")
        status = "BUG_REPRODUCED"

    print(f"\nMean absolute difference: {np.mean(np.abs(hf_result - ttml_result)):.6f}")
    print(f"Max absolute difference: {np.max(np.abs(hf_result - ttml_result)):.6f}")
    print(f"{'='*80}\n")

    return {
        "model": model_name,
        "pcc": pcc,
        "status": status,
        "mean_diff": np.mean(np.abs(hf_result - ttml_result)),
        "max_diff": np.max(np.abs(hf_result - ttml_result)),
    }


def main():
    """Run reproduction test."""
    print("\n" + "=" * 80)
    print("BERT Softmax Precision Bug Reproduction")
    print("=" * 80)
    print("\nThis script tests BERT models with the current TTML build.")
    print("Expected behavior:")
    print("  - WITH FP32 workaround (current default): PCC >0.95 ✅")
    print("  - WITHOUT workaround (if disabled): PCC ~0.81 ❌")
    print("\nTo test WITHOUT workaround, you must:")
    print("  1. Edit sources/ttml/ops/unary_ops.cpp line ~107")
    print("  2. Change: softmax(/* use_fp32_accumulation_workaround */ true)")
    print("  3. To: softmax(/* use_fp32_accumulation_workaround */ false)")
    print("  4. Rebuild TTML: cmake --build build")
    print("=" * 80)

    # Test bert-tiny (fastest)
    result = reproduce_bug("prajjwal1/bert-tiny", batch_size=1, seq_len=32)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    status_symbol = "✅" if result["status"] == "PASS_WITH_WORKAROUND" else "❌"
    print(f"{status_symbol} {result['model']}: PCC = {result['pcc']:.6f} ({result['status']})")
    print("=" * 80)

    # Conclusion
    if result["status"] == "PASS_WITH_WORKAROUND":
        print("\n✅ Test PASS - FP32 workaround is active and effective")
        print("\nNote: To reproduce the BUG (PCC ~0.81), you must:")
        print("  1. Disable FP32 workaround in sources/ttml/ops/unary_ops.cpp")
        print("  2. Rebuild TTML")
        print("  3. Re-run this script")
    else:
        print("\n❌ Test FAILED - Bug may be reproduced or workaround not working")


if __name__ == "__main__":
    main()
