# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Reproduction test for https://github.com/tenstorrent/tt-metal/issues/38411

ttnn.experimental.gelu_bw compute_program_hash ignores the `approximate`
parameter, causing a program cache collision.  When two calls with different
`approximate` values ("none" vs "tanh") share the same input shapes and
dtypes, the second call silently reuses the kernel compiled for the first
call instead of compiling a new one.

The test runs both modes on the same input (within the same device session
so the program cache is populated) and asserts the outputs differ.
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
def test_gelu_bw_cache_collision(input_shapes, device):
    """
    Issue #38411: approximate modes must produce distinct outputs.

    Runs gelu_bw with approximate="none" and approximate="tanh" on the same
    input and asserts the results are different.  A cache collision causes
    both calls to return the same tensor.
    """
    torch.manual_seed(42)
    pt_input = torch.rand(input_shapes).bfloat16() * 200 - 100  # range [-100, 100]
    pt_grad = torch.rand(input_shapes).bfloat16() * 10 - 5  # range [-5, 5]

    input_tensor = ttnn.from_torch(pt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(pt_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run both modes in the same session so program cache is shared
    result_none = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="none")
    result_tanh = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="tanh")

    out_none = ttnn.to_torch(result_none[0])
    out_tanh = ttnn.to_torch(result_tanh[0])

    # Core assertion: outputs must differ between modes
    assert not torch.equal(out_none, out_tanh), (
        "gelu_bw with approximate='none' and approximate='tanh' produced identical outputs. "
        "This indicates a program cache collision (issue #38411)."
    )


@pytest.mark.parametrize("input_shapes", INPUT_SHAPES)
@pytest.mark.parametrize("approximate", ("none", "tanh"))
def test_gelu_bw_approximate_golden(input_shapes, approximate, device):
    """
    Each approximate mode must match its own PyTorch golden reference.

    Unlike the existing test_bw_gelu which uses a golden that ignores the
    approximate parameter, this test computes a mode-specific golden.
    """
    torch.manual_seed(42)
    pt_input = torch.rand(input_shapes).bfloat16() * 200 - 100
    pt_grad = torch.rand(input_shapes).bfloat16() * 10 - 5

    input_tensor = ttnn.from_torch(pt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(pt_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate=approximate)
    tt_out = ttnn.to_torch(result[0])

    golden = _gelu_bw_golden(pt_grad, pt_input, approximate=approximate)

    # Use allclose with BF16-appropriate tolerances
    assert torch.allclose(tt_out.float(), golden.float(), atol=0.2, rtol=0.05), (
        f"gelu_bw(approximate='{approximate}') does not match PyTorch golden.\n"
        f"Max abs diff: {(tt_out.float() - golden.float()).abs().max().item():.6f}"
    )
