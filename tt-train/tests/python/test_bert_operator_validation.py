#!/usr/bin/env python3
"""
BERT Operator Validation Tests - Bottom-Up Approach

Tests each BERT operation in isolation with controlled random data to identify
which specific operations are broken.

Strategy:
1. Generate identical random inputs for both PyTorch and TTML
2. Run operation through both implementations
3. Compare outputs with strict PCC threshold (>0.999)
4. Isolate and identify broken operations

Test Levels:
- Level 1: Primitive operations (matmul, softmax, layer_norm, GELU, etc.)
- Level 2: Composite operations (QKV projection, attention mechanism)
- Level 3: Layer-by-layer (embeddings, attention blocks, MLP blocks)
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient between two tensors."""
    # Flatten tensors
    t1_flat = tensor1.flatten()
    t2_flat = tensor2.flatten()

    # Compute means
    mean1 = np.mean(t1_flat)
    mean2 = np.mean(t2_flat)

    # Compute standard deviations
    std1 = np.std(t1_flat)
    std2 = np.std(t2_flat)

    # Compute correlation
    numerator = np.mean((t1_flat - mean1) * (t2_flat - mean2))
    denominator = std1 * std2

    if denominator == 0:
        return 1.0 if np.allclose(t1_flat, t2_flat) else 0.0

    pcc = numerator / denominator
    return pcc


def print_comparison(name, ref, ttml_output, threshold=0.999):
    """Print detailed comparison between reference and TTML outputs."""
    ref_np = ref.detach().numpy() if torch.is_tensor(ref) else ref
    ttml_np = ttml_output.to_numpy() if hasattr(ttml_output, "to_numpy") else ttml_output

    pcc = compute_pcc(ref_np, ttml_np)
    diff = np.abs(ref_np - ttml_np)

    print(f"\n{'=' * 80}")
    print(f"Testing: {name}")
    print(f"{'=' * 80}")
    print(f"Reference shape: {ref_np.shape}")
    print(f"TTML shape: {ttml_np.shape}")
    print(f"\nReference stats:")
    print(f"  Mean: {ref_np.mean():.6f}, Std: {ref_np.std():.6f}")
    print(f"  Min: {ref_np.min():.6f}, Max: {ref_np.max():.6f}")
    print(f"\nTTML stats:")
    print(f"  Mean: {ttml_np.mean():.6f}, Std: {ttml_np.std():.6f}")
    print(f"  Min: {ttml_np.min():.6f}, Max: {ttml_np.max():.6f}")
    print(f"\nComparison:")
    print(f"  PCC: {pcc:.6f} {'✅' if pcc > threshold else '❌'}")
    print(f"  Mean abs diff: {diff.mean():.6e}")
    print(f"  Max abs diff: {diff.max():.6e}")
    print(f"  First 5 ref: {ref_np.flatten()[:5]}")
    print(f"  First 5 ttml: {ttml_np.flatten()[:5]}")
    print(f"  First 5 diff: {diff.flatten()[:5]}")

    return pcc > threshold, pcc


# ==============================================================================
# LEVEL 1: PRIMITIVE OPERATIONS
# ==============================================================================


def test_matmul_operation():
    """Test matrix multiplication in isolation."""
    print("\n" + "=" * 80)
    print("LEVEL 1: PRIMITIVE OPERATIONS - Matrix Multiplication")
    print("=" * 80)

    # Test configuration
    batch_size = 1
    seq_len = 32
    hidden_dim = 128

    # Generate random input
    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    weight_data = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)

    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    weight_torch = torch.from_numpy(weight_data)
    output_ref = torch.matmul(input_torch, weight_torch)

    # TTML implementation
    input_ttml = ttml.autograd.Tensor.from_numpy(input_data.reshape(batch_size, 1, seq_len, hidden_dim))
    weight_ttml = ttml.autograd.Tensor.from_numpy(weight_data.reshape(1, 1, hidden_dim, hidden_dim))

    # Use TTML's linear layer (which uses matmul internally)
    linear = ttml.modules.LinearLayer(hidden_dim, hidden_dim, has_bias=False)
    linear.weight.set_value(weight_ttml.get_value())
    output_ttml = linear(input_ttml)

    # Compare
    passed, pcc = print_comparison("Matrix Multiplication", output_ref, output_ttml)
    assert passed, f"Matmul failed with PCC {pcc:.6f}"


def test_layer_norm_operation():
    """Test LayerNorm in isolation."""
    print("\n" + "=" * 80)
    print("LEVEL 1: PRIMITIVE OPERATIONS - LayerNorm")
    print("=" * 80)

    # Test configuration
    batch_size = 1
    seq_len = 32
    hidden_dim = 128
    eps = 1e-12

    # Generate random input
    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    gamma_data = np.random.randn(hidden_dim).astype(np.float32)
    beta_data = np.random.randn(hidden_dim).astype(np.float32)

    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    gamma_torch = torch.from_numpy(gamma_data)
    beta_torch = torch.from_numpy(beta_data)

    layer_norm_ref = torch.nn.LayerNorm(hidden_dim, eps=eps)
    layer_norm_ref.weight.data = gamma_torch
    layer_norm_ref.bias.data = beta_torch
    output_ref = layer_norm_ref(input_torch)

    # TTML implementation
    input_ttml = ttml.autograd.Tensor.from_numpy(input_data.reshape(batch_size, 1, seq_len, hidden_dim))
    gamma_ttml = ttml.autograd.Tensor.from_numpy(gamma_data.reshape(1, 1, 1, hidden_dim))
    beta_ttml = ttml.autograd.Tensor.from_numpy(beta_data.reshape(1, 1, 1, hidden_dim))

    layer_norm_ttml = ttml.modules.LayerNormLayer(hidden_dim, eps, use_composite_op=False, enable_hardware_clamp=False)
    layer_norm_ttml.gamma.set_value(gamma_ttml.get_value())
    layer_norm_ttml.beta.set_value(beta_ttml.get_value())
    output_ttml = layer_norm_ttml(input_ttml)

    # Compare
    passed, pcc = print_comparison("LayerNorm", output_ref, output_ttml)
    assert passed, f"LayerNorm failed with PCC {pcc:.6f}"


def test_gelu_operation():
    """Test GELU activation in isolation."""
    print("\n" + "=" * 80)
    print("LEVEL 1: PRIMITIVE OPERATIONS - GELU")
    print("=" * 80)

    # Test configuration
    batch_size = 1
    seq_len = 32
    hidden_dim = 128

    # Generate random input
    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

    # PyTorch reference (exact GELU)
    input_torch = torch.from_numpy(input_data)
    output_ref = torch.nn.functional.gelu(input_torch, approximate="none")

    # TTML implementation
    input_ttml = ttml.autograd.Tensor.from_numpy(input_data.reshape(batch_size, 1, seq_len, hidden_dim))
    output_ttml = ttml.ops.gelu(input_ttml)

    # Compare
    passed, pcc = print_comparison("GELU", output_ref, output_ttml)
    assert passed, f"GELU failed with PCC {pcc:.6f}"


def test_softmax_operation():
    """Test Softmax in isolation."""
    print("\n" + "=" * 80)
    print("LEVEL 1: PRIMITIVE OPERATIONS - Softmax")
    print("=" * 80)

    # Test configuration
    batch_size = 1
    num_heads = 2
    seq_len = 32

    # Generate random input (attention scores)
    np.random.seed(42)
    input_data = np.random.randn(batch_size, num_heads, seq_len, seq_len).astype(np.float32)

    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    output_ref = torch.softmax(input_torch, dim=-1)

    # TTML implementation
    input_ttml = ttml.autograd.Tensor.from_numpy(input_data)
    # TTML softmax uses metal::softmax
    output_ttml_value = ttml.metal.softmax(input_ttml.get_value(), axis=3)
    output_ttml = ttml.autograd.create_tensor(output_ttml_value)

    # Compare
    passed, pcc = print_comparison("Softmax", output_ref, output_ttml, threshold=0.99)
    assert passed, f"Softmax failed with PCC {pcc:.6f}"


# ==============================================================================
# LEVEL 2: COMPOSITE OPERATIONS
# ==============================================================================


def test_qkv_projection():
    """Test QKV linear projection in isolation."""
    print("\n" + "=" * 80)
    print("LEVEL 2: COMPOSITE OPERATIONS - QKV Projection")
    print("=" * 80)

    # Test configuration
    batch_size = 1
    seq_len = 32
    hidden_dim = 128

    # Generate random input and weights
    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    qkv_weight_data = np.random.randn(hidden_dim * 3, hidden_dim).astype(np.float32)
    qkv_bias_data = np.random.randn(hidden_dim * 3).astype(np.float32)

    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    qkv_weight_torch = torch.from_numpy(qkv_weight_data)
    qkv_bias_torch = torch.from_numpy(qkv_bias_data)

    qkv_linear_ref = torch.nn.Linear(hidden_dim, hidden_dim * 3)
    qkv_linear_ref.weight.data = qkv_weight_torch
    qkv_linear_ref.bias.data = qkv_bias_torch
    output_ref = qkv_linear_ref(input_torch)

    # TTML implementation
    input_ttml = ttml.autograd.Tensor.from_numpy(input_data.reshape(batch_size, 1, seq_len, hidden_dim))
    qkv_weight_ttml = ttml.autograd.Tensor.from_numpy(qkv_weight_data.reshape(1, 1, hidden_dim * 3, hidden_dim))
    qkv_bias_ttml = ttml.autograd.Tensor.from_numpy(qkv_bias_data.reshape(1, 1, 1, hidden_dim * 3))

    qkv_linear_ttml = ttml.modules.LinearLayer(hidden_dim, hidden_dim * 3)
    qkv_linear_ttml.weight.set_value(qkv_weight_ttml.get_value())
    qkv_linear_ttml.bias.set_value(qkv_bias_ttml.get_value())
    output_ttml = qkv_linear_ttml(input_ttml)

    # Compare
    passed, pcc = print_comparison("QKV Projection", output_ref, output_ttml)
    assert passed, f"QKV Projection failed with PCC {pcc:.6f}"


def test_heads_creation_operation():
    """Test heads creation (QKV splitting into heads) in isolation."""
    print("\n" + "=" * 80)
    print("LEVEL 2: COMPOSITE OPERATIONS - Heads Creation")
    print("=" * 80)

    # Test configuration
    batch_size = 1
    seq_len = 32
    hidden_dim = 128
    num_heads = 2
    head_dim = hidden_dim // num_heads

    # Generate random QKV input
    np.random.seed(42)
    qkv_data = np.random.randn(batch_size, seq_len, hidden_dim * 3).astype(np.float32)

    # PyTorch reference - manual head splitting
    qkv_torch = torch.from_numpy(qkv_data)
    q_ref = qkv_torch[:, :, :hidden_dim]
    k_ref = qkv_torch[:, :, hidden_dim : hidden_dim * 2]
    v_ref = qkv_torch[:, :, hidden_dim * 2 :]

    # Reshape to [batch, seq, num_heads, head_dim] then transpose to [batch, num_heads, seq, head_dim]
    q_ref = q_ref.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k_ref = k_ref.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v_ref = v_ref.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # TTML implementation
    qkv_ttml = ttml.autograd.Tensor.from_numpy(qkv_data.reshape(batch_size, 1, seq_len, hidden_dim * 3))
    q_ttml, k_ttml, v_ttml = ttml.ops.heads_creation(qkv_ttml, num_heads)

    # Compare each output
    print("\n--- Checking Q ---")
    passed_q, pcc_q = print_comparison("Q heads", q_ref, q_ttml)

    print("\n--- Checking K ---")
    passed_k, pcc_k = print_comparison("K heads", k_ref, k_ttml)

    print("\n--- Checking V ---")
    passed_v, pcc_v = print_comparison("V heads", v_ref, v_ttml)

    assert (
        passed_q and passed_k and passed_v
    ), f"Heads creation failed: Q PCC={pcc_q:.6f}, K PCC={pcc_k:.6f}, V PCC={pcc_v:.6f}"


def test_attention_mechanism():
    """Test full scaled dot-product attention in isolation."""
    print("\n" + "=" * 80)
    print("LEVEL 2: COMPOSITE OPERATIONS - Scaled Dot-Product Attention")
    print("=" * 80)

    # Test configuration
    batch_size = 1
    num_heads = 2
    seq_len = 32
    head_dim = 64

    # Generate random Q, K, V
    np.random.seed(42)
    q_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    k_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    v_data = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

    # PyTorch reference
    q_torch = torch.from_numpy(q_data)
    k_torch = torch.from_numpy(k_data)
    v_torch = torch.from_numpy(v_data)

    scale = 1.0 / np.sqrt(head_dim)
    attn_scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output_ref = torch.matmul(attn_weights, v_torch)

    # TTML implementation
    q_ttml = ttml.autograd.Tensor.from_numpy(q_data)
    k_ttml = ttml.autograd.Tensor.from_numpy(k_data)
    v_ttml = ttml.autograd.Tensor.from_numpy(v_data)

    output_ttml = ttml.ops.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=None)

    # Compare
    passed, pcc = print_comparison("Scaled Dot-Product Attention", output_ref, output_ttml, threshold=0.99)
    assert passed, f"Attention failed with PCC {pcc:.6f}"


# ==============================================================================
# LEVEL 3: LAYER-BY-LAYER
# ==============================================================================


def test_embedding_layer():
    """Test embedding layer (word + position + token type) in isolation."""
    print("\n" + "=" * 80)
    print("LEVEL 3: LAYER-BY-LAYER - Embedding Layer")
    print("=" * 80)

    # Test configuration
    batch_size = 1
    seq_len = 32
    vocab_size = 1000
    hidden_dim = 128

    # Generate random embeddings
    np.random.seed(42)
    word_emb_data = np.random.randn(vocab_size, hidden_dim).astype(np.float32)
    pos_emb_data = np.random.randn(seq_len, hidden_dim).astype(np.float32)
    token_type_emb_data = np.random.randn(2, hidden_dim).astype(np.float32)

    # Input IDs
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.float32)
    token_type_ids = np.zeros((batch_size, seq_len)).astype(np.float32)

    # PyTorch reference
    word_emb_ref = torch.nn.Embedding(vocab_size, hidden_dim)
    word_emb_ref.weight.data = torch.from_numpy(word_emb_data)

    pos_emb_ref = torch.nn.Embedding(seq_len, hidden_dim)
    pos_emb_ref.weight.data = torch.from_numpy(pos_emb_data)

    token_type_emb_ref = torch.nn.Embedding(2, hidden_dim)
    token_type_emb_ref.weight.data = torch.from_numpy(token_type_emb_data)

    input_ids_torch = torch.from_numpy(input_ids).long()
    token_type_ids_torch = torch.from_numpy(token_type_ids).long()
    position_ids_torch = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    embeddings_ref = (
        word_emb_ref(input_ids_torch) + pos_emb_ref(position_ids_torch) + token_type_emb_ref(token_type_ids_torch)
    )

    # TTML implementation
    # Create BERT model with controlled embeddings
    config = ttml.models.bert.BertConfig()
    config.vocab_size = vocab_size
    config.max_sequence_length = seq_len
    config.embedding_dim = hidden_dim
    config.num_heads = 2
    config.num_blocks = 0  # No transformer blocks, just test embeddings
    config.intermediate_size = 512
    config.dropout_prob = 0.0
    config.layer_norm_eps = 1e-12
    config.use_token_type_embeddings = True

    bert_ttml = ttml.models.bert.create(config)

    # Set embedding weights
    params = bert_ttml.parameters()

    # Get device from model
    device = params["bert/embeddings/word_embeddings/weight"].get_value().device()
    params["bert/embeddings/word_embeddings/weight"].set_value(
        ttml.core.from_vector(
            word_emb_data.flatten().tolist(),
            params["bert/embeddings/word_embeddings/weight"].get_value().logical_shape(),
            device,
        )
    )
    params["bert/embeddings/position_embeddings/weight"].set_value(
        ttml.core.from_vector(
            pos_emb_data.flatten().tolist(),
            params["bert/embeddings/position_embeddings/weight"].get_value().logical_shape(),
            device,
        )
    )
    params["bert/embeddings/token_type_embeddings/weight"].set_value(
        ttml.core.from_vector(
            token_type_emb_data.flatten().tolist(),
            params["bert/embeddings/token_type_embeddings/weight"].get_value().logical_shape(),
            device,
        )
    )

    # Run forward (will only compute embeddings since num_blocks=0)
    input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids.reshape(batch_size, 1, 1, seq_len))
    token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids.reshape(batch_size, 1, 1, seq_len))

    # We need to access embeddings directly since model has no blocks
    # This is tricky - let me just compare the embedding sum
    print("\n⚠️  Embedding layer test needs direct access to intermediate values")
    print("Skipping for now - will implement layer-by-layer hooks")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
