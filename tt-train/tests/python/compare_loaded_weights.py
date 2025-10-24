#!/usr/bin/env python3
"""Compare what TTML loaded vs what it should have loaded."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

# Load expected QKV
qkv_expected = np.load("/tmp/qkv_expected.npy")
print(f"Expected QKV shape: {qkv_expected.shape}")
print(f"Expected QKV[0,0]: {qkv_expected[0,0]}")
print(f"Expected QKV[768,0]: {qkv_expected[768,0]}")
print(f"Expected QKV[1536,0]: {qkv_expected[1536,0]}")

# Load TTML model (already created with correct weights)
import transformers
from pathlib import Path

model_name = "bert-base-uncased"
hf_model = transformers.BertModel.from_pretrained(model_name)
hf_config = hf_model.config

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

safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
bert.load_model_from_safetensors(str(safetensors_path))

params = bert.parameters()
qkv_ttml = params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"].to_numpy()
qkv_ttml_2d = qkv_ttml.reshape(2304, 768)

print(f"\nActual TTML QKV shape: {qkv_ttml_2d.shape}")
print(f"Actual TTML QKV[0,0]: {qkv_ttml_2d[0,0]}")
print(f"Actual TTML QKV[768,0]: {qkv_ttml_2d[768,0]}")
print(f"Actual TTML QKV[1536,0]: {qkv_ttml_2d[1536,0]}")

# Direct comparison
diff = np.abs(qkv_expected - qkv_ttml_2d)
print(f"\nDifference stats:")
print(f"  Mean: {diff.mean():.6e}")
print(f"  Max: {diff.max():.6e}")
print(f"  Min: {diff.min():.6e}")

# Check if maybe transposed
qkv_expected_T = qkv_expected.T
diff_T = np.abs(qkv_expected_T - qkv_ttml_2d)
print(f"\nDifference with transpose:")
print(f"  Mean: {diff_T.mean():.6e}")
print(f"  Max: {diff_T.max():.6e}")

# Save both for manual inspection
np.save("/tmp/qkv_actual.npy", qkv_ttml_2d)
print("\nSaved actual TTML QKV to /tmp/qkv_actual.npy")
print("Expected and actual arrays saved for comparison")
