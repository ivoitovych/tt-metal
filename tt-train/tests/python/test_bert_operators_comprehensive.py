#!/usr/bin/env python3
"""
Comprehensive BERT Operator Validation Tests

Tests each BERT operation in isolation with controlled random data.
Compares PyTorch reference implementation vs TTML implementation.

This is a HEAVY test suite implementing Option 1 from OPERATOR_TESTING_LIMITATIONS.md
"""

import numpy as np
import os
import sys
import torch
import pytest

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    if len(x_flat) != len(y_flat):
        return 0.0
    corr = np.corrcoef(x_flat, y_flat)
    return corr[0, 1] if corr.shape == (2, 2) else 0.0


def print_comparison(name: str, ref: np.ndarray, ttml_output: np.ndarray, threshold: float = 0.999):
    """Print detailed comparison statistics."""
    pcc = compute_pcc(ref, ttml_output)
    diff = np.abs(ref - ttml_output)

    status = "✅ PASS" if pcc >= threshold else "❌ FAIL"
    print(f"\n{'='*80}")
    print(f"{name}: PCC = {pcc:.6f} {status}")
    print(f"{'='*80}")
    print(f"Shapes: ref {ref.shape}, ttml {ttml_output.shape}")
    print(f"Mean abs diff: {diff.mean():.6e}, Max abs diff: {diff.max():.6e}")
    print(f"Ref  - mean: {ref.mean():.6e}, std: {ref.std():.6e}, min: {ref.min():.6e}, max: {ref.max():.6e}")
    print(
        f"TTML - mean: {ttml_output.mean():.6e}, std: {ttml_output.std():.6e}, min: {ttml_output.min():.6e}, max: {ttml_output.max():.6e}"
    )
    print(f"First 5 ref:  {ref.flatten()[:5]}")
    print(f"First 5 ttml: {ttml_output.flatten()[:5]}")
    print(f"First 5 diff: {diff.flatten()[:5]}")

    return pcc >= threshold


class TestBERTOperators:
    """Comprehensive test suite for BERT operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test configuration."""
        self.batch_size = 1
        self.seq_len = 32
        self.hidden_dim = 128
        self.num_heads = 2
        self.head_dim = self.hidden_dim // self.num_heads
        np.random.seed(42)
        torch.manual_seed(42)

    def test_heads_creation(self):
        """Test QKV head splitting operation."""
        print("\n" + "=" * 80)
        print("TEST: Heads Creation (QKV splitting into heads)")
        print("=" * 80)

        # Generate random QKV input [B, S, E*3]
        qkv_data = np.random.randn(self.batch_size, self.seq_len, self.hidden_dim * 3).astype(np.float32)

        # PyTorch reference
        qkv_torch = torch.from_numpy(qkv_data)
        q_flat = qkv_torch[:, :, : self.hidden_dim]
        k_flat = qkv_torch[:, :, self.hidden_dim : self.hidden_dim * 2]
        v_flat = qkv_torch[:, :, self.hidden_dim * 2 :]

        # Reshape: [B, S, E] -> [B, S, H, E/H] -> [B, H, S, E/H]
        q_ref = q_flat.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2).numpy()
        k_ref = k_flat.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2).numpy()
        v_ref = v_flat.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2).numpy()

        # TTML implementation (needs [B, 1, S, E*3] shape)
        qkv_ttml = ttml.autograd.Tensor.from_numpy(
            qkv_data.reshape(self.batch_size, 1, self.seq_len, self.hidden_dim * 3)
        )
        q_ttml, k_ttml, v_ttml = ttml.ops.multi_head_utils.heads_creation(qkv_ttml, self.num_heads)

        # Compare
        q_pass = print_comparison("Q Heads", q_ref, q_ttml.to_numpy())
        k_pass = print_comparison("K Heads", k_ref, k_ttml.to_numpy())
        v_pass = print_comparison("V Heads", v_ref, v_ttml.to_numpy())

        assert q_pass and k_pass and v_pass, "Heads creation test failed"

    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention operation."""
        print("\n" + "=" * 80)
        print("TEST: Scaled Dot-Product Attention")
        print("=" * 80)

        # Generate random Q, K, V inputs [B, H, S, E/H]
        q_data = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim).astype(np.float32)
        k_data = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim).astype(np.float32)
        v_data = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim).astype(np.float32)

        # PyTorch reference
        q_torch = torch.from_numpy(q_data)
        k_torch = torch.from_numpy(k_data)
        v_torch = torch.from_numpy(v_data)

        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output_ref = torch.matmul(attn_weights, v_torch).numpy()

        # TTML implementation
        q_ttml = ttml.autograd.Tensor.from_numpy(q_data)
        k_ttml = ttml.autograd.Tensor.from_numpy(k_data)
        v_ttml = ttml.autograd.Tensor.from_numpy(v_data)

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=None)

        # Compare (slightly lower threshold due to softmax numerical differences)
        passed = print_comparison("Attention Output", output_ref, output_ttml.to_numpy(), threshold=0.99)
        assert passed, "Scaled dot-product attention test failed"

    def test_scaled_dot_product_attention_with_mask(self):
        """Test scaled dot-product attention with masking."""
        print("\n" + "=" * 80)
        print("TEST: Scaled Dot-Product Attention with Mask")
        print("=" * 80)

        # Generate random Q, K, V inputs [B, H, S, E/H]
        q_data = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim).astype(np.float32)
        k_data = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim).astype(np.float32)
        v_data = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim).astype(np.float32)

        # Create attention mask (1 = attend, 0 = mask out)
        # Mask out last 8 positions
        mask_data = np.ones((self.batch_size, 1, 1, self.seq_len), dtype=np.float32)
        mask_data[:, :, :, -8:] = 0.0

        # PyTorch reference
        q_torch = torch.from_numpy(q_data)
        k_torch = torch.from_numpy(k_data)
        v_torch = torch.from_numpy(v_data)
        mask_torch = torch.from_numpy(mask_data)

        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale

        # Apply mask: where mask=0, set to large negative value
        attn_scores = attn_scores.masked_fill(mask_torch == 0, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        output_ref = torch.matmul(attn_weights, v_torch).numpy()

        # TTML implementation
        q_ttml = ttml.autograd.Tensor.from_numpy(q_data)
        k_ttml = ttml.autograd.Tensor.from_numpy(k_data)
        v_ttml = ttml.autograd.Tensor.from_numpy(v_data)
        mask_ttml = ttml.autograd.Tensor.from_numpy(mask_data)

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)

        # Compare
        passed = print_comparison("Attention Output (Masked)", output_ref, output_ttml.to_numpy(), threshold=0.99)
        assert passed, "Scaled dot-product attention with mask test failed"

    def test_heads_fusion(self):
        """Test head fusion operation."""
        print("\n" + "=" * 80)
        print("TEST: Heads Fusion (merging heads back)")
        print("=" * 80)

        # Generate random heads data [B, H, S, E/H]
        heads_data = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim).astype(np.float32)

        # PyTorch reference
        heads_torch = torch.from_numpy(heads_data)
        # [B, H, S, E/H] -> [B, S, H, E/H] -> [B, S, E]
        fused_ref = heads_torch.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.hidden_dim).numpy()

        # TTML implementation
        heads_ttml = ttml.autograd.Tensor.from_numpy(heads_data)
        fused_ttml = ttml.ops.multi_head_utils.heads_fusion(heads_ttml)

        # TTML output is [B, 1, S, E], reshape to [B, S, E] for comparison
        fused_ttml_reshaped = fused_ttml.to_numpy().reshape(self.batch_size, self.seq_len, self.hidden_dim)

        # Compare
        passed = print_comparison("Fused Heads", fused_ref, fused_ttml_reshaped)
        assert passed, "Heads fusion test failed"

    def test_gelu_activation(self):
        """Test GELU activation function."""
        print("\n" + "=" * 80)
        print("TEST: GELU Activation")
        print("=" * 80)

        # Generate random input
        input_data = np.random.randn(self.batch_size, self.seq_len, self.hidden_dim).astype(np.float32)

        # PyTorch reference (exact GELU)
        input_torch = torch.from_numpy(input_data)
        output_ref = torch.nn.functional.gelu(input_torch, approximate="none").numpy()

        # TTML implementation (needs [B, 1, S, E] shape)
        input_ttml = ttml.autograd.Tensor.from_numpy(
            input_data.reshape(self.batch_size, 1, self.seq_len, self.hidden_dim)
        )
        output_ttml = ttml.ops.unary.gelu(input_ttml)

        # Reshape TTML output back to [B, S, E]
        output_ttml_reshaped = output_ttml.to_numpy().reshape(self.batch_size, self.seq_len, self.hidden_dim)

        # Compare (GELU can have slight numerical differences)
        passed = print_comparison("GELU", output_ref, output_ttml_reshaped, threshold=0.999)
        assert passed, "GELU activation test failed"

    def test_layernorm(self):
        """Test LayerNorm operation."""
        print("\n" + "=" * 80)
        print("TEST: LayerNorm")
        print("=" * 80)

        # Generate random input
        input_data = np.random.randn(self.batch_size, self.seq_len, self.hidden_dim).astype(np.float32)
        gamma_data = np.random.randn(self.hidden_dim).astype(np.float32)
        beta_data = np.random.randn(self.hidden_dim).astype(np.float32)
        eps = 1e-12

        # PyTorch reference
        input_torch = torch.from_numpy(input_data)
        gamma_torch = torch.from_numpy(gamma_data)
        beta_torch = torch.from_numpy(beta_data)

        # Manually compute LayerNorm to match BERT's exact implementation
        mean = input_torch.mean(dim=-1, keepdim=True)
        var = input_torch.var(dim=-1, unbiased=False, keepdim=True)
        output_ref = (input_torch - mean) / torch.sqrt(var + eps)
        output_ref = output_ref * gamma_torch + beta_torch
        output_ref = output_ref.numpy()

        # TTML implementation (needs [B, 1, S, E] shape)
        input_ttml = ttml.autograd.Tensor.from_numpy(
            input_data.reshape(self.batch_size, 1, self.seq_len, self.hidden_dim)
        )
        gamma_ttml = ttml.autograd.Tensor.from_numpy(gamma_data.reshape(1, 1, 1, self.hidden_dim))
        beta_ttml = ttml.autograd.Tensor.from_numpy(beta_data.reshape(1, 1, 1, self.hidden_dim))

        # Note: Testing with hardware clamp disabled to match exact epsilon
        output_ttml = ttml.ops.layernorm.layernorm(
            input_ttml, gamma_ttml, beta_ttml, eps=eps, enable_hardware_clamp=False
        )

        # Reshape TTML output back to [B, S, E]
        output_ttml_reshaped = output_ttml.to_numpy().reshape(self.batch_size, self.seq_len, self.hidden_dim)

        # Compare (LayerNorm can have numerical differences due to epsilon handling)
        passed = print_comparison("LayerNorm", output_ref, output_ttml_reshaped, threshold=0.99)
        assert passed, "LayerNorm test failed"

    def test_linear_layer(self):
        """Test linear layer (matmul + bias) operation."""
        print("\n" + "=" * 80)
        print("TEST: Linear Layer")
        print("=" * 80)

        out_features = 256

        # Generate random input and weights
        input_data = np.random.randn(self.batch_size, self.seq_len, self.hidden_dim).astype(np.float32)
        weight_data = np.random.randn(out_features, self.hidden_dim).astype(np.float32)
        bias_data = np.random.randn(out_features).astype(np.float32)

        # PyTorch reference
        input_torch = torch.from_numpy(input_data)
        weight_torch = torch.from_numpy(weight_data)
        bias_torch = torch.from_numpy(bias_data)

        linear = torch.nn.Linear(self.hidden_dim, out_features)
        linear.weight.data = weight_torch
        linear.bias.data = bias_torch
        output_ref = linear(input_torch).detach().numpy()

        # TTML implementation (needs [B, 1, S, E] shape)
        input_ttml = ttml.autograd.Tensor.from_numpy(
            input_data.reshape(self.batch_size, 1, self.seq_len, self.hidden_dim)
        )
        weight_ttml = ttml.autograd.Tensor.from_numpy(weight_data)
        bias_ttml = ttml.autograd.Tensor.from_numpy(bias_data)

        output_ttml = ttml.ops.linear.linear_op(input_ttml, weight_ttml, bias_ttml)

        # Reshape TTML output back to [B, S, out_features]
        output_ttml_reshaped = output_ttml.to_numpy().reshape(self.batch_size, self.seq_len, out_features)

        # Compare
        passed = print_comparison("Linear Layer", output_ref, output_ttml_reshaped, threshold=0.999)
        assert passed, "Linear layer test failed"

    def test_matmul(self):
        """Test matrix multiplication operation."""
        print("\n" + "=" * 80)
        print("TEST: Matrix Multiplication")
        print("=" * 80)

        dim_k = 64

        # Generate random matrices
        a_data = np.random.randn(self.batch_size, self.seq_len, self.hidden_dim).astype(np.float32)
        b_data = np.random.randn(self.batch_size, self.hidden_dim, dim_k).astype(np.float32)

        # PyTorch reference
        a_torch = torch.from_numpy(a_data)
        b_torch = torch.from_numpy(b_data)
        output_ref = torch.matmul(a_torch, b_torch).numpy()

        # TTML implementation (needs [B, 1, S, E] shape)
        a_ttml = ttml.autograd.Tensor.from_numpy(a_data.reshape(self.batch_size, 1, self.seq_len, self.hidden_dim))
        b_ttml = ttml.autograd.Tensor.from_numpy(b_data.reshape(self.batch_size, 1, self.hidden_dim, dim_k))

        output_ttml = ttml.ops.matmul.matmul_op(a_ttml, b_ttml)

        # Reshape TTML output back to [B, S, dim_k]
        output_ttml_reshaped = output_ttml.to_numpy().reshape(self.batch_size, self.seq_len, dim_k)

        # Compare
        passed = print_comparison("Matmul", output_ref, output_ttml_reshaped, threshold=0.999)
        assert passed, "Matrix multiplication test failed"

    def test_complete_mha_pipeline(self):
        """Test complete multi-head attention pipeline."""
        print("\n" + "=" * 80)
        print("TEST: Complete MHA Pipeline")
        print("=" * 80)

        # Generate random input
        input_data = np.random.randn(self.batch_size, self.seq_len, self.hidden_dim).astype(np.float32)
        qkv_weight_data = np.random.randn(self.hidden_dim * 3, self.hidden_dim).astype(np.float32)
        qkv_bias_data = np.random.randn(self.hidden_dim * 3).astype(np.float32)
        out_weight_data = np.random.randn(self.hidden_dim, self.hidden_dim).astype(np.float32)
        out_bias_data = np.random.randn(self.hidden_dim).astype(np.float32)

        # PyTorch reference
        input_torch = torch.from_numpy(input_data)

        # QKV projection
        qkv_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        qkv_linear.weight.data = torch.from_numpy(qkv_weight_data)
        qkv_linear.bias.data = torch.from_numpy(qkv_bias_data)
        qkv = qkv_linear(input_torch)

        # Split into Q, K, V and reshape into heads
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads back
        attn_output = attn_output.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.hidden_dim)

        # Output projection
        out_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        out_linear.weight.data = torch.from_numpy(out_weight_data)
        out_linear.bias.data = torch.from_numpy(out_bias_data)
        output_ref = out_linear(attn_output).detach().numpy()

        # TTML implementation (using operators)
        input_ttml = ttml.autograd.Tensor.from_numpy(
            input_data.reshape(self.batch_size, 1, self.seq_len, self.hidden_dim)
        )
        qkv_weight_ttml = ttml.autograd.Tensor.from_numpy(qkv_weight_data)
        qkv_bias_ttml = ttml.autograd.Tensor.from_numpy(qkv_bias_data)

        # QKV projection
        qkv_ttml = ttml.ops.linear.linear_op(input_ttml, qkv_weight_ttml, qkv_bias_ttml)

        # Split into heads
        q_ttml, k_ttml, v_ttml = ttml.ops.multi_head_utils.heads_creation(qkv_ttml, self.num_heads)

        # Attention
        attn_output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml)

        # Merge heads
        merged_ttml = ttml.ops.multi_head_utils.heads_fusion(attn_output_ttml)

        # Output projection
        out_weight_ttml = ttml.autograd.Tensor.from_numpy(out_weight_data)
        out_bias_ttml = ttml.autograd.Tensor.from_numpy(out_bias_data)
        output_ttml = ttml.ops.linear.linear_op(merged_ttml, out_weight_ttml, out_bias_ttml)

        # Reshape TTML output back to [B, S, E]
        output_ttml_reshaped = output_ttml.to_numpy().reshape(self.batch_size, self.seq_len, self.hidden_dim)

        # Compare
        passed = print_comparison("Complete MHA Pipeline", output_ref, output_ttml_reshaped, threshold=0.98)
        assert passed, "Complete MHA pipeline test failed"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
