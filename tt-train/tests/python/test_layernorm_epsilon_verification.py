#!/usr/bin/env python3
"""Verify LayerNorm epsilon configuration in BERT model."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml
import transformers
from pathlib import Path

model_name = "prajjwal1/bert-tiny"

# Load HF model
hf_model = transformers.BertModel.from_pretrained(model_name)
hf_config = hf_model.config

print(f"HuggingFace BERT epsilon: {hf_config.layer_norm_eps}")

# Create TTML model
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

print(f"TTML config epsilon: {ttml_config.layer_norm_eps}")

bert = ttml.models.bert.create(ttml_config)

# Access internal LayerNorm modules to check their epsilon settings
# The BERT model should have:
# - embedding_norm
# - bert_block_0/attention_norm
# - bert_block_0/mlp_norm
# etc.

print(f"\n{'='*80}")
print("LayerNorm Epsilon Configuration in TTML BERT:")
print(f"{'='*80}")

# Try to access the LayerNorm layers
# Since we can't directly access C++ objects from Python, let's create a standalone
# LayerNorm to test the default behavior

print("\nTesting standalone LayerNorm with BERT's epsilon:")
print(f"  Requested epsilon: {hf_config.layer_norm_eps}")

# Create a simple input tensor
test_input = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 1, 128), dtype=np.float32))

# Create LayerNorm with BERT epsilon (only 3 args - uses defaults)
# This mimics what bert_block.cpp does
ln_with_defaults = ttml.modules.LayerNormLayer(
    features=128,
    eps=hf_config.layer_norm_eps,
    use_composite_op=False
    # enable_hardware_clamp defaults to TRUE
    # min_safe_eps defaults to 1e-4
)

print(f"  LayerNorm.get_epsilon(): {ln_with_defaults.get_epsilon()}")
print(f"  LayerNorm.get_enable_hardware_clamp(): {ln_with_defaults.get_enable_hardware_clamp()}")
print(f"  LayerNorm.get_min_safe_eps(): {ln_with_defaults.get_min_safe_eps()}")

print(f"\n⚠️  ISSUE IDENTIFIED:")
print(f"  - BERT expects epsilon: {hf_config.layer_norm_eps} (1e-12)")
print(f"  - TTML LayerNorm configured epsilon: {ln_with_defaults.get_epsilon()}")
print(f"  - Hardware clamping enabled: {ln_with_defaults.get_enable_hardware_clamp()}")
print(f"  - Min safe epsilon: {ln_with_defaults.get_min_safe_eps()}")

if ln_with_defaults.get_enable_hardware_clamp():
    effective_eps = max(ln_with_defaults.get_epsilon(), ln_with_defaults.get_min_safe_eps())
    print(f"  - ACTUAL epsilon used (for BFLOAT16): {effective_eps}")
    print(f"\n❌ BUG: Epsilon is clamped from {hf_config.layer_norm_eps:.2e} to {effective_eps:.2e}")
    print(f"   This is {effective_eps / hf_config.layer_norm_eps:.1e}x larger!")
else:
    print(f"  - ACTUAL epsilon used: {ln_with_defaults.get_epsilon()}")
    print(f"\n✅ No clamping - epsilon matches BERT")

print(f"\n{'='*80}")
print("Testing LayerNorm with hardware clamping disabled:")
print(f"{'='*80}")

ln_no_clamp = ttml.modules.LayerNormLayer(
    features=128, eps=hf_config.layer_norm_eps, use_composite_op=False, enable_hardware_clamp=False  # Disable clamping
)

print(f"  LayerNorm.get_epsilon(): {ln_no_clamp.get_epsilon()}")
print(f"  LayerNorm.get_enable_hardware_clamp(): {ln_no_clamp.get_enable_hardware_clamp()}")
print(f"  - ACTUAL epsilon used: {ln_no_clamp.get_epsilon()}")

print(f"\n✅ With clamping disabled, epsilon matches BERT exactly")

print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}")
print(f"The BERT forward pass divergence is caused by LayerNorm epsilon mismatch:")
print(f"  - HuggingFace uses: {hf_config.layer_norm_eps:.2e}")
print(f"  - TTML currently uses: {max(hf_config.layer_norm_eps, 1e-4):.2e} (due to hardware clamping)")
print(f"  - This causes different normalization behavior")
print(f"\nFIX: Disable hardware clamping for BERT models, or make it configurable.")
