#!/usr/bin/env python3
"""Quick check that bert-base-uncased weights load correctly."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml
import transformers
from pathlib import Path

model_name = "bert-base-uncased"

# Load HF model
hf_model = transformers.BertModel.from_pretrained(model_name)
hf_config = hf_model.config

# Save to safetensors
safetensors_path = Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")
if not safetensors_path.exists():
    from safetensors.torch import save_file

    save_file(hf_model.state_dict(), str(safetensors_path))

# Create TTML model
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

bert = ttml.models.bert.create(ttml_config)

# Check BEFORE loading
params_before = bert.parameters()
emb_before = params_before["bert/token_embeddings/weight"].to_numpy()
mean_before = emb_before.mean()

# Load weights
bert.load_model_from_safetensors(str(safetensors_path))

# Check AFTER loading
params_after = bert.parameters()
emb_after = params_after["bert/token_embeddings/weight"].to_numpy()
mean_after = emb_after.mean()

# Expected from HF
hf_emb = hf_model.embeddings.word_embeddings.weight.detach().numpy()
mean_expected = hf_emb.mean()

print(f"bert-base-uncased weight loading check:")
print(f"  BEFORE loading: mean = {mean_before:.6f}")
print(f"  AFTER loading:  mean = {mean_after:.6f}")
print(f"  Expected (HF):  mean = {mean_expected:.6f}")
print(f"  Match? {np.abs(mean_after - mean_expected) < 0.001}")

# Check QKV
hf_q = hf_model.encoder.layer[0].attention.self.query.weight.detach().numpy()
hf_k = hf_model.encoder.layer[0].attention.self.key.weight.detach().numpy()
hf_v = hf_model.encoder.layer[0].attention.self.value.weight.detach().numpy()
expected_qkv = np.concatenate([hf_q, hf_k, hf_v], axis=0)

ttml_qkv = params_after["bert/bert_block_0/attention/self_attention/qkv_linear/weight"].to_numpy()
ttml_qkv_2d = ttml_qkv.reshape(3 * hf_config.hidden_size, hf_config.hidden_size)

from scipy.stats import pearsonr

pcc_token_emb = pearsonr(
    hf_emb.flatten(), emb_after.reshape(-1, hf_config.hidden_size)[: hf_config.vocab_size].flatten()
)[0]
pcc_qkv = pearsonr(expected_qkv.flatten(), ttml_qkv_2d.flatten())[0]

print(f"\nPCC values:")
print(f"  Token embeddings: {pcc_token_emb:.6f}")
print(f"  QKV weights:      {pcc_qkv:.6f}")
print(f"  {'✅ PASS' if pcc_token_emb > 0.999 and pcc_qkv > 0.999 else '❌ FAIL'}")
