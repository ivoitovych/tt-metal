# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Reproduction test for https://github.com/tenstorrent/tt-metal/issues/38411

ttnn.experimental.gelu_bw compute_program_hash ignores the `approximate`
parameter, causing a program cache collision. When two calls with different
`approximate` values share the same input shapes and dtypes, the second call
silently reuses the kernel compiled for the first call.

The bug has two facets:
1. The program factory (gelu_backward_program_factory.cpp) only distinguishes
   "tanh" from everything else — passing "poly" silently falls through to
   the "none" kernel.
2. compute_program_hash does not include `approximate` in the hash, so even
   after adding a "poly" kernel path, the cache would serve the wrong program.

These tests verify:
- "none" vs "tanh" produce distinct outputs (currently passes).
- "none" vs "poly" produce distinct outputs (currently FAILS — reproduces
  the bug: "poly" silently runs the "none" kernel).
- Each mode matches its own PyTorch golden reference (the "poly" golden test
  currently FAILS because the "poly" kernel does not exist yet).
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
    Sanity check: "none" and "tanh" use different kernels and must produce
    different outputs even when run sequentially on the same device (shared
    program cache).
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
def test_gelu_bw_cache_collision_none_vs_poly(input_shapes, device):
    """
    Issue #38411 core reproduction: "none" and "poly" must produce different
    outputs. Currently FAILS because:
    - The program factory has no "poly" branch (falls through to "none").
    - compute_program_hash ignores approximate, so even with a "poly" kernel
      the cache would return the "none" program.
    """
    torch.manual_seed(42)
    pt_input = torch.rand(input_shapes).bfloat16() * 200 - 100
    pt_grad = torch.rand(input_shapes).bfloat16() * 10 - 5

    input_tensor = ttnn.from_torch(pt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(pt_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result_none = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="none")
    result_poly = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="poly")

    out_none = ttnn.to_torch(result_none[0])
    out_poly = ttnn.to_torch(result_poly[0])

    assert not torch.equal(out_none, out_poly), (
        "gelu_bw approximate='none' and approximate='poly' produced identical outputs. "
        "This confirms issue #38411: the 'poly' mode silently runs the 'none' kernel "
        "due to missing kernel path and/or program cache hash collision."
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

    assert torch.allclose(tt_out.float(), golden.float(), atol=0.2, rtol=0.05), (
        f"gelu_bw(approximate='{approximate}') does not match PyTorch golden.\n"
        f"Max abs diff: {(tt_out.float() - golden.float()).abs().max().item():.6f}"
    )


@pytest.mark.parametrize("input_shapes", INPUT_SHAPES)
def test_gelu_bw_poly_golden(input_shapes, device):
    """
    The "poly" mode must match PyTorch's GELU backward with approximate="none"
    computed via polynomial approximation (higher precision than "tanh").

    Currently FAILS because the "poly" kernel does not exist — the factory
    falls through to the "none" (erf-based) kernel. Once the poly kernel is
    added (PR #36366), this test should pass.

    Note: PyTorch does not have a "poly" GELU mode. The "poly" kernel is a
    Tenstorrent-specific polynomial approximation that should closely match
    the exact ("none") result. We compare against the exact golden here.
    """
    torch.manual_seed(42)
    pt_input = torch.rand(input_shapes).bfloat16() * 200 - 100
    pt_grad = torch.rand(input_shapes).bfloat16() * 10 - 5

    input_tensor = ttnn.from_torch(pt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(pt_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run "none" first to populate cache, then "poly" — if cache collision
    # exists, "poly" will silently return "none" results
    result_none = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="none")
    result_poly = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="poly")

    out_none = ttnn.to_torch(result_none[0])
    out_poly = ttnn.to_torch(result_poly[0])

    golden_none = _gelu_bw_golden(pt_grad, pt_input, approximate="none")

    # The poly kernel should produce results close to the exact golden
    # but NOT bitwise-identical to the "none" kernel output (different
    # compute path). If they are identical, the poly kernel was never run.
    assert torch.allclose(out_poly.float(), golden_none.float(), atol=0.2, rtol=0.05), (
        f"gelu_bw(approximate='poly') does not match expected golden.\n"
        f"Max abs diff: {(out_poly.float() - golden_none.float()).abs().max().item():.6f}"
    )
    assert not torch.equal(out_none, out_poly), (
        "gelu_bw 'poly' output is bitwise-identical to 'none' output. "
        "The poly kernel was not dispatched (issue #38411)."
    )
