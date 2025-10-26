#!/usr/bin/env python3
"""
BERT Operator Validation with REAL BERT Data and REAL Attention Masks

This test addresses the CRITICAL FLAW: Previous tests used random data, not real BERT outputs.

This test:
1. Loads a real pre-trained BERT model
2. Runs forward pass to get REAL intermediate values (Q, K, V from actual embeddings)
3. Extracts REAL attention masks from actual inputs
4. Tests TTML operators with these REAL values
5. Compares against HuggingFace reference with PROPER MASKING

Test Scenarios:
- Minimal padding (90% real tokens, 10% padding)
- Medium padding (50% real tokens, 50% padding)
- Heavy padding (20% real tokens, 80% padding)
- No padding (100% real tokens)
"""

import numpy as np
import os
import sys
import torch
import pytest
import transformers

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


def print_comparison(name: str, ref: np.ndarray, ttml_output: np.ndarray, threshold: float = 0.99):
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

    return pcc >= threshold


class TestBERTOperatorsWithRealData:
    """Test BERT operators using REAL data from actual BERT forward pass."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup: Load real BERT model."""
        self.model_name = "prajjwal1/bert-tiny"
        print(f"\n{'='*80}")
        print(f"Loading REAL BERT model: {self.model_name}")
        print(f"{'='*80}")

        self.hf_model = transformers.BertModel.from_pretrained(self.model_name)
        self.hf_model.eval()
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
        self.config = self.hf_model.config

        print(f"Model config:")
        print(f"  hidden_size: {self.config.hidden_size}")
        print(f"  num_attention_heads: {self.config.num_attention_heads}")
        print(f"  num_hidden_layers: {self.config.num_hidden_layers}")

    def extract_real_qkv_and_mask(self, text: str, max_length: int = 32):
        """
        Extract REAL Q, K, V tensors and attention mask from actual BERT forward pass.

        This gives us REAL data that BERT actually processes, not random noise.
        """
        print(f"\n{'='*80}")
        print(f"Extracting REAL QKV from BERT forward pass")
        print(f"{'='*80}")
        print(f"Input text: '{text}'")

        # Tokenize with attention mask
        encoded = self.tokenizer(
            text, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        num_real = attention_mask.sum().item()
        num_padding = (attention_mask == 0).sum().item()
        padding_pct = num_padding / (num_real + num_padding) * 100

        print(f"Sequence length: {max_length}")
        print(f"Real tokens: {num_real}, Padding: {num_padding} ({padding_pct:.1f}% padding)")
        print(f"Attention mask: {attention_mask[0, :min(15, max_length)].tolist()}...")

        with torch.no_grad():
            # Get embeddings
            embeddings = self.hf_model.embeddings(input_ids)

            # Get first layer's attention
            # We'll hook into the first attention layer to extract Q, K, V
            first_layer = self.hf_model.encoder.layer[0]
            attention = first_layer.attention.self

            # Compute Q, K, V using actual BERT weights on actual embeddings
            batch_size, seq_len, hidden_size = embeddings.shape
            num_heads = self.config.num_attention_heads
            head_dim = hidden_size // num_heads

            # Apply QKV linear transformations (REAL BERT WEIGHTS!)
            q = attention.query(embeddings)
            k = attention.key(embeddings)
            v = attention.value(embeddings)

            # Reshape to [batch, num_heads, seq_len, head_dim]
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            # Convert attention mask to attention scores mask
            # HF format: 1 = attend, 0 = mask out
            # Our format: [batch, 1, 1, seq_len]
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()

            print(f"\nExtracted REAL tensors:")
            print(f"  Q: {q.shape} - mean={q.mean():.6f}, std={q.std():.6f}")
            print(f"  K: {k.shape} - mean={k.mean():.6f}, std={k.std():.6f}")
            print(f"  V: {v.shape} - mean={v.mean():.6f}, std={v.std():.6f}")
            print(f"  Mask: {attn_mask.shape}")

            return {
                "q": q.numpy().astype(np.float32),
                "k": k.numpy().astype(np.float32),
                "v": v.numpy().astype(np.float32),
                "mask": attn_mask.numpy().astype(np.float32),
                "num_real_tokens": num_real,
                "padding_pct": padding_pct,
            }

    def test_attention_with_real_bert_minimal_padding(self):
        """Test with REAL BERT data - minimal padding (90% real tokens)."""
        print(f"\n{'#'*80}")
        print(f"TEST: REAL BERT Attention - Minimal Padding (90% real)")
        print(f"{'#'*80}")

        # Long text to minimize padding
        text = "The quick brown fox jumps over the lazy dog and runs through the forest with great speed and agility while chasing rabbits."
        data = self.extract_real_qkv_and_mask(text, max_length=32)

        # PyTorch reference with REAL mask
        q_torch = torch.from_numpy(data["q"])
        k_torch = torch.from_numpy(data["k"])
        v_torch = torch.from_numpy(data["v"])
        mask_torch = torch.from_numpy(data["mask"])

        head_dim = q_torch.shape[-1]
        scale = 1.0 / np.sqrt(head_dim)

        # Compute attention with mask
        attn_scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
        # Apply mask: where mask=0, set to -1e9
        attn_scores = attn_scores.masked_fill(mask_torch == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output_ref = torch.matmul(attn_weights, v_torch).numpy()

        # TTML implementation with REAL data
        q_ttml = ttml.autograd.Tensor.from_numpy(data["q"])
        k_ttml = ttml.autograd.Tensor.from_numpy(data["k"])
        v_ttml = ttml.autograd.Tensor.from_numpy(data["v"])
        mask_ttml = ttml.autograd.Tensor.from_numpy(data["mask"])

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)

        # Compare
        passed = print_comparison(
            f"Real BERT Attention ({data['padding_pct']:.1f}% padding)",
            output_ref,
            output_ttml.to_numpy(),
            threshold=0.99,
        )
        assert passed, f"Failed with {data['padding_pct']:.1f}% padding"

    def test_attention_with_real_bert_medium_padding(self):
        """Test with REAL BERT data - medium padding (50% real tokens)."""
        print(f"\n{'#'*80}")
        print(f"TEST: REAL BERT Attention - Medium Padding (50% real)")
        print(f"{'#'*80}")

        # Medium length text
        text = "Machine learning models process natural language efficiently."
        data = self.extract_real_qkv_and_mask(text, max_length=32)

        # PyTorch reference
        q_torch = torch.from_numpy(data["q"])
        k_torch = torch.from_numpy(data["k"])
        v_torch = torch.from_numpy(data["v"])
        mask_torch = torch.from_numpy(data["mask"])

        head_dim = q_torch.shape[-1]
        scale = 1.0 / np.sqrt(head_dim)

        attn_scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
        attn_scores = attn_scores.masked_fill(mask_torch == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output_ref = torch.matmul(attn_weights, v_torch).numpy()

        # TTML implementation
        q_ttml = ttml.autograd.Tensor.from_numpy(data["q"])
        k_ttml = ttml.autograd.Tensor.from_numpy(data["k"])
        v_ttml = ttml.autograd.Tensor.from_numpy(data["v"])
        mask_ttml = ttml.autograd.Tensor.from_numpy(data["mask"])

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)

        # Compare
        passed = print_comparison(
            f"Real BERT Attention ({data['padding_pct']:.1f}% padding)",
            output_ref,
            output_ttml.to_numpy(),
            threshold=0.99,
        )
        assert passed, f"Failed with {data['padding_pct']:.1f}% padding"

    def test_attention_with_real_bert_heavy_padding(self):
        """Test with REAL BERT data - heavy padding (80% padding, like original test)."""
        print(f"\n{'#'*80}")
        print(f"TEST: REAL BERT Attention - Heavy Padding (80% padding)")
        print(f"{'#'*80}")

        # Short text (like original flawed test)
        text = "The quick brown fox jumps."
        data = self.extract_real_qkv_and_mask(text, max_length=32)

        # PyTorch reference
        q_torch = torch.from_numpy(data["q"])
        k_torch = torch.from_numpy(data["k"])
        v_torch = torch.from_numpy(data["v"])
        mask_torch = torch.from_numpy(data["mask"])

        head_dim = q_torch.shape[-1]
        scale = 1.0 / np.sqrt(head_dim)

        attn_scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
        attn_scores = attn_scores.masked_fill(mask_torch == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output_ref = torch.matmul(attn_weights, v_torch).numpy()

        # TTML implementation
        q_ttml = ttml.autograd.Tensor.from_numpy(data["q"])
        k_ttml = ttml.autograd.Tensor.from_numpy(data["k"])
        v_ttml = ttml.autograd.Tensor.from_numpy(data["v"])
        mask_ttml = ttml.autograd.Tensor.from_numpy(data["mask"])

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)

        # Compare
        passed = print_comparison(
            f"Real BERT Attention ({data['padding_pct']:.1f}% padding)",
            output_ref,
            output_ttml.to_numpy(),
            threshold=0.99,
        )
        assert passed, f"Failed with {data['padding_pct']:.1f}% padding"

    def test_attention_with_real_bert_no_padding(self):
        """Test with REAL BERT data - NO padding (100% real tokens)."""
        print(f"\n{'#'*80}")
        print(f"TEST: REAL BERT Attention - No Padding (100% real)")
        print(f"{'#'*80}")

        # Exact length text
        text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 1 2 3 4"
        data = self.extract_real_qkv_and_mask(text, max_length=32)

        # PyTorch reference
        q_torch = torch.from_numpy(data["q"])
        k_torch = torch.from_numpy(data["k"])
        v_torch = torch.from_numpy(data["v"])
        mask_torch = torch.from_numpy(data["mask"])

        head_dim = q_torch.shape[-1]
        scale = 1.0 / np.sqrt(head_dim)

        attn_scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
        attn_scores = attn_scores.masked_fill(mask_torch == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output_ref = torch.matmul(attn_weights, v_torch).numpy()

        # TTML implementation
        q_ttml = ttml.autograd.Tensor.from_numpy(data["q"])
        k_ttml = ttml.autograd.Tensor.from_numpy(data["k"])
        v_ttml = ttml.autograd.Tensor.from_numpy(data["v"])
        mask_ttml = ttml.autograd.Tensor.from_numpy(data["mask"])

        output_ttml = ttml.ops.multi_head_utils.scaled_dot_product_attention(q_ttml, k_ttml, v_ttml, mask=mask_ttml)

        # Compare
        passed = print_comparison(
            f"Real BERT Attention ({data['padding_pct']:.1f}% padding)",
            output_ref,
            output_ttml.to_numpy(),
            threshold=0.99,
        )
        assert passed, f"Failed with {data['padding_pct']:.1f}% padding"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
