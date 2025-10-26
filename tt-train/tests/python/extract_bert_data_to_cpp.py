#!/usr/bin/env python3
"""
Extract real BERT Q, K, V data and output as C++ arrays.
This allows us to test with real data directly in C++ without any Python bindings.
"""

import numpy as np
import torch
import transformers


def array_to_cpp(arr: np.ndarray, name: str) -> str:
    """Convert numpy array to C++ initializer list."""
    flat = arr.flatten()

    # Format as C++ array with proper line breaks
    values = []
    for i, val in enumerate(flat):
        if i > 0 and i % 8 == 0:
            values.append(f"\n        {val:.8f}F")
        else:
            values.append(f"{val:.8f}F")

    cpp_code = f"    // Shape: {list(arr.shape)}\n"
    cpp_code += f"    std::vector<float> {name} = {{\n        "
    cpp_code += ", ".join(values)
    cpp_code += "\n    };\n"

    return cpp_code


def main():
    print("Extracting REAL BERT data for C++ test...")

    model_name = "prajjwal1/bert-tiny"
    hf_model = transformers.BertModel.from_pretrained(model_name)
    hf_model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    # Use a simple short text to minimize data size
    text = "The quick brown fox"
    max_length = 16  # Keep it small for C++ test

    encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print(f"Text: '{text}'")
    print(f"Tokens: {input_ids[0, :].tolist()}")
    print(f"Attention mask: {attention_mask[0, :].tolist()}")

    with torch.no_grad():
        # Get embeddings
        embeddings = hf_model.embeddings(input_ids)

        # Get Q, K, V from first layer
        first_layer = hf_model.encoder.layer[0]
        attention = first_layer.attention.self

        batch_size, seq_len, hidden_size = embeddings.shape
        num_heads = hf_model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        q = attention.query(embeddings)
        k = attention.key(embeddings)
        v = attention.value(embeddings)

        # Reshape to multi-head format [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Convert to numpy
        q_np = q.numpy().astype(np.float32)
        k_np = k.numpy().astype(np.float32)
        v_np = v.numpy().astype(np.float32)
        mask_np = attention_mask.unsqueeze(1).unsqueeze(2).numpy().astype(np.float32)

        print(f"\nQ shape: {q_np.shape}")
        print(f"K shape: {k_np.shape}")
        print(f"V shape: {v_np.shape}")
        print(f"Mask shape: {mask_np.shape}")

        # Compute reference output using PyTorch
        scale = 1.0 / np.sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask_for_scores = attention_mask[:, None, None, :]
        attn_scores_masked = attn_scores.masked_fill(mask_for_scores == 0, -1e9)
        attn_weights = torch.softmax(attn_scores_masked, dim=-1)
        output_ref = torch.matmul(attn_weights, v).numpy().astype(np.float32)

        print(f"Reference output shape: {output_ref.shape}")
        print(f"Reference output stats: mean={output_ref.mean():.6f}, std={output_ref.std():.6f}")

        # Generate C++ code
        print("\n" + "=" * 80)
        print("C++ CODE - Copy this into a test file:")
        print("=" * 80)
        print()

        print("// Real BERT data extracted from prajjwal1/bert-tiny")
        print(f'// Text: "{text}"')
        print(f"// Sequence length: {seq_len}, Num heads: {num_heads}, Head dim: {head_dim}")
        print()

        print(array_to_cpp(q_np, "q_real_bert"))
        print(array_to_cpp(k_np, "k_real_bert"))
        print(array_to_cpp(v_np, "v_real_bert"))
        print(array_to_cpp(mask_np, "mask_real_bert"))
        print(array_to_cpp(output_ref, "expected_output_real_bert"))

        print(f"\n    // Dimensions")
        print(f"    const uint32_t batch_size = {batch_size};")
        print(f"    const uint32_t num_heads = {num_heads};")
        print(f"    const uint32_t seq_len = {seq_len};")
        print(f"    const uint32_t head_dim = {head_dim};")


if __name__ == "__main__":
    main()
