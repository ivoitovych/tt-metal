# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for https://github.com/tenstorrent/tt-metal/issues/38411

compute_program_hash in gelu_backward_device_operation.cpp did not include
the `approximate` parameter, causing program cache collisions between
"none" and "tanh" modes. When both modes were called with the same input
shape/dtype in the same device session, the second call silently reused
the first call's compiled kernel.

These tests verify:
- "none" and "tanh" produce distinct outputs when sharing a device session
  (i.e. the program cache does not collide).
- Each mode individually matches its own PyTorch golden reference (with
  tolerances appropriate for the current BF16 kernels).
"""

import torch
import pytest
import ttnn


def _gelu_bw_golden(grad, x, approximate="none"):
    """PyTorch reference for GELU backward with explicit approximate mode."""
    x = x.clone().detach().float().requires_grad_(True)
    grad = grad.clone().detach().float()
    y = torch.nn.functional.gelu(x, approximate=approximate)
    y.backward(gradient=grad)
    return x.grad.bfloat16()


INPUT_SHAPES = (
    torch.Size([1, 1, 32, 32]),
    torch.Size([1, 1, 320, 384]),
)


@pytest.mark.parametrize("input_shapes", INPUT_SHAPES)
def test_gelu_bw_cache_collision_none_vs_tanh(input_shapes, device):
    """
    Issue #38411: "none" and "tanh" must produce different outputs even when
    run sequentially on the same device (shared program cache). A hash
    collision would cause the second call to silently reuse the first kernel.
    """
    torch.manual_seed(42)
    pt_input = torch.rand(input_shapes).bfloat16() * 200 - 100
    pt_grad = torch.rand(input_shapes).bfloat16() * 10 - 5

    input_tensor = ttnn.from_torch(pt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(pt_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result_none = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="none")
    result_tanh = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="tanh")

    out_none = ttnn.to_torch(result_none[0])
    out_tanh = ttnn.to_torch(result_tanh[0])

    assert not torch.equal(out_none, out_tanh), (
        "gelu_bw approximate='none' and approximate='tanh' produced identical outputs. "
        "This indicates a program cache collision (issue #38411)."
    )


@pytest.mark.parametrize("input_shapes", INPUT_SHAPES)
@pytest.mark.parametrize("approximate", ("none", "tanh"))
def test_gelu_bw_approximate_golden(input_shapes, approximate, device):
    """
    Each approximate mode must match its own PyTorch golden reference.

    Unlike the existing test_bw_gelu which uses a golden that ignores the
    approximate parameter, this test computes a mode-specific golden.
    Tolerances are wide enough to accept the current BF16 kernel precision.
    """
    torch.manual_seed(42)
    pt_input = torch.rand(input_shapes).bfloat16() * 200 - 100
    pt_grad = torch.rand(input_shapes).bfloat16() * 10 - 5

    input_tensor = ttnn.from_torch(pt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(pt_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate=approximate)
    tt_out = ttnn.to_torch(result[0])

    golden = _gelu_bw_golden(pt_grad, pt_input, approximate=approximate)

    assert torch.allclose(tt_out.float(), golden.float(), atol=0.2, rtol=0.05), (
        f"gelu_bw(approximate='{approximate}') does not match PyTorch golden.\n"
        f"Max abs diff: {(tt_out.float() - golden.float()).abs().max().item():.6f}"
    )
