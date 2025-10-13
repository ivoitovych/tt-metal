# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for nlp_create_qkv_heads bug when head_dim < TILE_WIDTH (32)

BUG: Integer division bug in nlp_create_qkv_heads_program_factory.cpp causes
incorrect output shapes when head_dim < 32.

Root cause: q_num_tiles = num_heads * (head_dim / TILE_WIDTH)
When head_dim=16: q_num_tiles = num_heads * 0 = 0 (reads zero tiles!)

Expected: q_num_tiles = (num_heads * head_dim + TILE_WIDTH - 1) / TILE_WIDTH

This test demonstrates the bug with head_dim=16 and verifies the fix.
"""

import pytest
from loguru import logger
import torch
import ttnn
from models.common.utility_functions import tt2torch_tensor, comp_pcc


def run_nlp_create_qkv_heads_small_head_dim_test(
    batch,
    seq_len,
    head_dim,
    num_heads,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    """
    Test nlp_create_qkv_heads with head_dim < 32 (TILE_WIDTH)

    This reproduces a bug where integer division causes incorrect tile calculations.
    """
    torch.manual_seed(1234)

    embedding_dim = num_heads * head_dim
    qkv_width = embedding_dim * 3  # Q + K + V concatenated

    logger.info(f"Testing head_dim={head_dim} < TILE_WIDTH(32) bug fix")
    logger.info(f"batch={batch}, seq_len={seq_len}, num_heads={num_heads}, embedding_dim={embedding_dim}")

    # Create fused QKV tensor: [batch, 1, seq_len, embedding_dim * 3]
    in0_shape = [batch, 1, seq_len, qkv_width]
    A = torch.randn(in0_shape)
    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)

    # Call nlp_create_qkv_heads - this triggers the bug when head_dim < 32
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        None,  # No separate KV tensor
        num_heads=num_heads,
        num_kv_heads=num_heads,
        transpose_k_heads=False,
        memory_config=out_mem_config,
    )

    # Check memory configurations
    assert in0_t.memory_config().buffer_type == in_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type

    # CRITICAL: Check output shapes - this is where the bug manifests
    # Expected: [batch, num_heads, seq_len, head_dim]
    expected_shape = [batch, num_heads, seq_len, head_dim]

    logger.info(f"Expected shape: {expected_shape}")
    logger.info(f"Q output shape: {list(q.padded_shape)}")
    logger.info(f"K output shape: {list(k.padded_shape)}")
    logger.info(f"V output shape: {list(v.padded_shape)}")

    # Verify shapes are correct
    assert list(q.padded_shape) == expected_shape, \
        f"Q shape mismatch! Expected {expected_shape}, got {list(q.padded_shape)}"
    assert list(k.padded_shape) == expected_shape, \
        f"K shape mismatch! Expected {expected_shape}, got {list(k.padded_shape)}"
    assert list(v.padded_shape) == expected_shape, \
        f"V shape mismatch! Expected {expected_shape}, got {list(v.padded_shape)}"

    # Verify numerical correctness
    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    # Split reference QKV
    (ref_q, ref_k, ref_v) = torch.split(
        A, [embedding_dim, embedding_dim, embedding_dim], dim=-1
    )

    # Reshape to match expected output: [batch, num_heads, seq_len, head_dim]
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.info(f"Q PCC: {output_pcc_q} (passing={passing_pcc_q})")
    assert passing_pcc_q, f"Q tensor PCC check failed: {output_pcc_q}"

    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.info(f"K PCC: {output_pcc_k} (passing={passing_pcc_k})")
    assert passing_pcc_k, f"K tensor PCC check failed: {output_pcc_k}"

    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.info(f"V PCC: {output_pcc_v} (passing={passing_pcc_v})")
    assert passing_pcc_v, f"V tensor PCC check failed: {output_pcc_v}"

    logger.info("✅ Test passed! Bug is fixed.")


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_heads",
    (
        # Critical test cases: head_dim < 32 (TILE_WIDTH)
        (1, 32, 16, 4),    # head_dim=16: Triggers bug
        (1, 64, 16, 4),    # head_dim=16: Different seq_len
        (1, 32, 8, 8),     # head_dim=8: Even smaller
        (2, 32, 16, 2),    # head_dim=16: Different batch/heads
        # Edge case: head_dim=32 (should work regardless)
        (1, 32, 32, 4),    # head_dim=32: Control case
    ),
    ids=[
        "batch1_seq32_head16_nheads4",
        "batch1_seq64_head16_nheads4",
        "batch1_seq32_head8_nheads8",
        "batch2_seq32_head16_nheads2",
        "batch1_seq32_head32_nheads4_control",
    ],
)
def test_nlp_create_qkv_heads_small_head_dim(
    batch,
    seq_len,
    head_dim,
    num_heads,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    """
    Test nlp_create_qkv_heads with head_dim < TILE_WIDTH (32)

    This test specifically targets the bug where integer division
    q_num_tiles = num_heads * (head_dim / TILE_WIDTH) returns 0
    when head_dim < 32.
    """
    run_nlp_create_qkv_heads_small_head_dim_test(
        batch,
        seq_len,
        head_dim,
        num_heads,
        dtype,
        in_mem_config,
        out_mem_config,
        device,
    )
