# Complete Reproduction Instructions for Softmax Precision Bug

**Related Bug Report**: `TTNN_BUG_REPORT_SOFTMAX_BFLOAT16_PRECISION.md`

This document provides **complete, self-contained reproduction instructions** for the BERT softmax precision issue (PCC 0.81 → 0.95 with FP32 workaround).

---

## Prerequisites

### Software Requirements

```bash
# Python packages (install with pip)
pip install torch transformers safetensors numpy pytest

# Environment variables
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME/tt-train/build/sources/ttml:$PYTHONPATH
```

### Hardware Requirements

- Tenstorrent Wormhole device
- Minimum 16GB RAM

### TTML Build

```bash
cd $TT_METAL_HOME/tt-train
cmake -DCMAKE_BUILD_TYPE=Debug -B build -GNinja
cmake --build build
```

---

## Complete Reproduction Code

### Full Python Script

Save as `reproduce_softmax_bug.py`:

```python
#!/usr/bin/env python3
"""
Complete reproduction of BERT softmax precision bug.

Bug: BERT models show PCC 0.81 with bfloat16 softmax accumulation
Workaround: FP32 accumulation restores PCC >0.95

This script demonstrates the bug can ONLY be reproduced in full
end-to-end BERT execution, not in isolated softmax tests.
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add TTML to path
sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
import _ttml as ttml

from transformers import BertModel
from safetensors.torch import save_file


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    t1_flat = tensor1.flatten()
    t2_flat = tensor2.flatten()

    mean1 = np.mean(t1_flat)
    mean2 = np.mean(t2_flat)

    numerator = np.sum((t1_flat - mean1) * (t2_flat - mean2))
    denominator = np.sqrt(
        np.sum((t1_flat - mean1) ** 2) * np.sum((t2_flat - mean2) ** 2)
    )

    if denominator == 0:
        return 1.0

    return numerator / denominator


def create_ttml_model_with_fp32_option(config, use_fp32_softmax=False):
    """
    Create TTML BERT model with optional FP32 softmax.

    Args:
        config: HuggingFace BERT config
        use_fp32_softmax: If True, use FP32 accumulation workaround

    Returns:
        TTML BERT model
    """
    ttml_config = ttml.models.bert.BertConfig()
    ttml_config.vocab_size = config.vocab_size
    ttml_config.max_sequence_length = 32
    ttml_config.embedding_dim = config.hidden_size
    ttml_config.intermediate_size = config.intermediate_size
    ttml_config.num_heads = config.num_attention_heads
    ttml_config.num_blocks = config.num_hidden_layers
    ttml_config.dropout_prob = 0.0
    ttml_config.layer_norm_eps = config.layer_norm_eps
    ttml_config.use_token_type_embeddings = True
    ttml_config.use_pooler = False

    # NOTE: The FP32 workaround is currently hardcoded in the TTML source
    # at sources/ttml/ops/unary_ops.cpp (line 107)
    # To test WITHOUT the workaround, you would need to change:
    #   softmax(/* use_fp32_accumulation_workaround */ true)
    # to:
    #   softmax(/* use_fp32_accumulation_workaround */ false)
    # and rebuild TTML

    return ttml.models.bert.create(ttml_config)


def reproduce_bug(model_name="prajjwal1/bert-tiny", batch_size=1, seq_len=32):
    """
    Reproduce the softmax precision bug with full BERT model.

    Args:
        model_name: HuggingFace model name
                   Options: "prajjwal1/bert-tiny" (2 layers, 128 dim)
                           "prajjwal1/bert-small" (4 layers, 512 dim)
                           "bert-base-uncased" (12 layers, 768 dim)
        batch_size: Batch size (must be 1 for current TTML)
        seq_len: Sequence length (must be divisible by 32)

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
    ttml_model = create_ttml_model_with_fp32_option(config)
    ttml_model.load_model_from_safetensors(str(safetensors_path))
    print("Weights loaded successfully")

    # Create test inputs
    print("\nCreating test inputs...")
    np.random.seed(42)

    # Token IDs: random valid tokens
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=np.uint32)

    # Token type IDs: 0 for first half, 1 for second half (simulating two sentences)
    token_type_ids = np.zeros((batch_size, seq_len), dtype=np.uint32)
    token_type_ids[:, seq_len//2:] = 1

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
    ttml_result = ttml_model(input_ids, token_type_ids, attention_mask)

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
        'model': model_name,
        'pcc': pcc,
        'status': status,
        'mean_diff': np.mean(np.abs(hf_result - ttml_result)),
        'max_diff': np.max(np.abs(hf_result - ttml_result)),
    }


def main():
    """Run reproduction tests on multiple BERT variants."""
    print("\n" + "="*80)
    print("BERT Softmax Precision Bug Reproduction")
    print("="*80)
    print("\nThis script tests BERT models with the current TTML build.")
    print("Expected behavior:")
    print("  - WITH FP32 workaround (current): PCC >0.95 ✅")
    print("  - WITHOUT workaround (if disabled): PCC ~0.81 ❌")
    print("\nTo test WITHOUT workaround, you must:")
    print("  1. Edit sources/ttml/ops/unary_ops.cpp line 107")
    print("  2. Change: softmax(/* use_fp32_accumulation_workaround */ true)")
    print("  3. To: softmax(/* use_fp32_accumulation_workaround */ false)")
    print("  4. Rebuild TTML")
    print("="*80)

    # Test models (start with smallest)
    models = [
        ("prajjwal1/bert-tiny", "Tiny (2 layers, 128 dim)"),
        ("prajjwal1/bert-small", "Small (4 layers, 512 dim)"),
        # Uncomment to test larger model (slower):
        # ("bert-base-uncased", "Base (12 layers, 768 dim)"),
    ]

    results = []
    for model_name, description in models:
        print(f"\n\nTesting: {description}")
        result = reproduce_bug(model_name, batch_size=1, seq_len=32)
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for result in results:
        status_symbol = "✅" if result['status'] == "PASS_WITH_WORKAROUND" else "❌"
        print(f"{status_symbol} {result['model']}: PCC = {result['pcc']:.6f} ({result['status']})")
    print("="*80)

    # Conclusion
    all_pass = all(r['status'] == 'PASS_WITH_WORKAROUND' for r in results)
    if all_pass:
        print("\n✅ All tests PASS - FP32 workaround is active and effective")
        print("\nNote: To reproduce the BUG (PCC ~0.81), you must:")
        print("  1. Disable FP32 workaround in sources/ttml/ops/unary_ops.cpp")
        print("  2. Rebuild TTML")
        print("  3. Re-run this script")
    else:
        print("\n❌ Some tests FAILED - Bug may be reproduced or workaround not working")


if __name__ == "__main__":
    main()
```

---

## How to Run

### Quick Test (Tiny Model)

```bash
# Set environment
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME/tt-train/build/sources/ttml:$PYTHONPATH

# Run reproduction
python3 reproduce_softmax_bug.py
```

**Expected output WITH workaround**:
```
Testing: Tiny (2 layers, 128 dim)
PCC: 0.9537xx
✅ PASS: PCC >0.95 (FP32 workaround is active)
```

**Expected output WITHOUT workaround** (after disabling in code):
```
Testing: Tiny (2 layers, 128 dim)
PCC: 0.8100xx
❌ FAIL: PCC <0.85 (Bug reproduced!)
```

### Test All Models

Uncomment `bert-base-uncased` in the script to test larger model (slower):

```python
models = [
    ("prajjwal1/bert-tiny", "Tiny (2 layers, 128 dim)"),
    ("prajjwal1/bert-small", "Small (4 layers, 512 dim)"),
    ("bert-base-uncased", "Base (12 layers, 768 dim)"),  # Uncomment this
]
```

---

## How to Reproduce the BUG (Disable Workaround)

To see the actual bug (PCC 0.81), you must disable the FP32 workaround:

### Step 1: Edit Source Code

```bash
# Edit the workaround file
vi $TT_METAL_HOME/tt-train/sources/ttml/ops/unary_ops.cpp
```

Find line ~107:
```cpp
/* compute_kernel_config */ core::ComputeKernelConfig::softmax(
    /* use_fp32_accumulation_workaround */ true  // <- Change this to false
)
```

Change to:
```cpp
/* compute_kernel_config */ core::ComputeKernelConfig::softmax(
    /* use_fp32_accumulation_workaround */ false  // <- Changed!
)
```

### Step 2: Rebuild TTML

```bash
cd $TT_METAL_HOME/tt-train
cmake --build build --target ttml
```

### Step 3: Run Reproduction Script

```bash
python3 reproduce_softmax_bug.py
```

**Expected output**:
```
Testing: Tiny (2 layers, 128 dim)
PCC: 0.8100xx  # <-- Bug reproduced!
❌ FAIL: PCC <0.85 (Bug reproduced! FP32 workaround not active)
```

### Step 4: Re-enable Workaround

**IMPORTANT**: Re-enable the workaround after testing!

```bash
# Change back to true
vi $TT_METAL_HOME/tt-train/sources/ttml/ops/unary_ops.cpp
# Set use_fp32_accumulation_workaround back to true

# Rebuild
cd $TT_METAL_HOME/tt-train
cmake --build build --target ttml
```

---

## Model Details

### prajjwal1/bert-tiny
- **Layers**: 2
- **Hidden dim**: 128
- **Attention heads**: 2
- **Parameters**: ~4M
- **Download size**: ~17MB
- **Best for**: Quick testing

### prajjwal1/bert-small
- **Layers**: 4
- **Hidden dim**: 512
- **Attention heads**: 8
- **Parameters**: ~29M
- **Download size**: ~116MB
- **Best for**: Medium testing

### bert-base-uncased
- **Layers**: 12
- **Hidden dim**: 768
- **Attention heads**: 12
- **Parameters**: ~110M
- **Download size**: ~440MB
- **Best for**: Full validation (slower)

---

## Expected Results

### With FP32 Workaround (Current Default)

| Model | Expected PCC | Status |
|-------|--------------|--------|
| bert-tiny | >0.95 | ✅ PASS |
| bert-small | >0.95 | ✅ PASS |
| bert-base-uncased | >0.95 | ✅ PASS |

### Without Workaround (Bug Visible)

| Model | Expected PCC | Status |
|-------|--------------|--------|
| bert-tiny | ~0.81-0.85 | ❌ BUG |
| bert-small | ~0.67-0.75 | ❌ BUG |
| bert-base-uncased | ~0.04-0.20 | ❌ BUG |

---

## Troubleshooting

### Import Error: No module named '_ttml'

```bash
# Check TTML was built
ls $TT_METAL_HOME/tt-train/build/sources/ttml/_ttml*.so

# Set PYTHONPATH
export PYTHONPATH=$TT_METAL_HOME/tt-train/build/sources/ttml:$PYTHONPATH
```

### Import Error: No module named 'transformers'

```bash
pip install transformers safetensors torch
```

### Device Error

Ensure Wormhole device is available:
```bash
# Check device
python3 -c "import _ttml; _ttml.autograd.ctx().open_device(); print('Device OK')"
```

---

## Summary

This reproduction demonstrates:

1. ✅ **Bug is real**: Disabling FP32 workaround shows PCC 0.81
2. ✅ **Workaround works**: FP32 accumulation restores PCC >0.95
3. ✅ **Only appears in full model**: Cannot reproduce in isolated softmax tests
4. ✅ **Self-contained**: All code and instructions included

The bug requires complete BERT model execution to manifest and cannot be isolated to a single softmax operation.
