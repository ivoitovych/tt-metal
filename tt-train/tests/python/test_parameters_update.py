#!/usr/bin/env python3
"""Test if parameters() returns references to the same tensor objects."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml
import transformers
from pathlib import Path

# Create a simple BERT model
print("Creating BERT model...")
config = ttml.models.bert.BertConfig()
config.vocab_size = 100
config.max_sequence_length = 32
config.embedding_dim = 64
config.intermediate_size = 256
config.num_heads = 2
config.num_blocks = 1
config.dropout_prob = 0.0

bert = ttml.models.bert.create(config)

# Get parameters twice
print("\nGetting parameters twice...")
params1 = bert.parameters()
params2 = bert.parameters()

# Get the same parameter from both
param1 = params1["bert/token_embeddings/weight"]
param2 = params2["bert/token_embeddings/weight"]

# Check if they're the same object (same memory address)
print(f"\nparam1 id: {id(param1)}")
print(f"param2 id: {id(param2)}")
print(f"Same Python object? {param1 is param2}")

# Get values
val1 = param1.to_numpy()
val2 = param2.to_numpy()

print(f"\nval1 mean: {val1.mean():.6f}")
print(f"val2 mean: {val2.mean():.6f}")
print(f"Values equal? {np.allclose(val1, val2)}")

# Try loading weights using the actual loading function
print("\n" + "=" * 80)
print("Testing actual weight loading...")
print("=" * 80)

# Create HF model and save
hf_model = transformers.BertModel.from_pretrained("prajjwal1/bert-tiny")
safetensors_path = Path("/tmp/test_params_update.safetensors")
from safetensors.torch import save_file

save_file(hf_model.state_dict(), str(safetensors_path))

# Create TTML model matching HF
hf_config = hf_model.config
ttml_config = ttml.models.bert.BertConfig()
ttml_config.vocab_size = hf_config.vocab_size
ttml_config.max_sequence_length = 32
ttml_config.embedding_dim = hf_config.hidden_size
ttml_config.intermediate_size = hf_config.intermediate_size
ttml_config.num_heads = hf_config.num_attention_heads
ttml_config.num_blocks = hf_config.num_hidden_layers
ttml_config.dropout_prob = 0.0
ttml_config.use_token_type_embeddings = True
ttml_config.use_pooler = False

bert2 = ttml.models.bert.create(ttml_config)

# Get parameter BEFORE loading
print("\nGetting parameter BEFORE loading...")
params_before = bert2.parameters()
emb_before = params_before["bert/token_embeddings/weight"]
val_before = emb_before.to_numpy()
print(f"Mean BEFORE: {val_before.mean():.6f}")
print(f"TensorPtr id BEFORE: {id(emb_before)}")

# Load weights
print("\nLoading weights...")
bert2.load_model_from_safetensors(str(safetensors_path))

# Get parameter AFTER loading - use the SAME TensorPtr object
print("\nChecking SAME TensorPtr object after loading...")
val_after_same = emb_before.to_numpy()
print(f"Mean AFTER (same ptr): {val_after_same.mean():.6f}")
print(f"Did it change? {not np.allclose(val_before, val_after_same)}")

# Get parameter AFTER loading - get NEW TensorPtr
print("\nGetting NEW parameter AFTER loading...")
params_after = bert2.parameters()
emb_after = params_after["bert/token_embeddings/weight"]
val_after_new = emb_after.to_numpy()
print(f"Mean AFTER (new ptr): {val_after_new.mean():.6f}")
print(f"TensorPtr id AFTER: {id(emb_after)}")
print(f"Did it change? {not np.allclose(val_before, val_after_new)}")

# Expected value from HF
hf_emb = hf_model.embeddings.word_embeddings.weight.detach().numpy()
print(f"\nExpected mean (HF): {hf_emb.mean():.6f}")
print(
    f"Match AFTER (same ptr)? {np.allclose(val_after_same.reshape(-1, hf_config.hidden_size)[:hf_config.vocab_size], hf_emb, atol=1e-3)}"
)
print(
    f"Match AFTER (new ptr)? {np.allclose(val_after_new.reshape(-1, hf_config.hidden_size)[:hf_config.vocab_size], hf_emb, atol=1e-3)}"
)
