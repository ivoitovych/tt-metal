# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
GELU Implementation Validation Tests

This module validates the Tenstorrent GELU implementations against expected formulas.
These tests serve as:
1. Documentation of hardware behavior
2. Regression tests to detect changes in implementation
3. Verification that the expected code path is executed
4. Reference for expected vs actual behavior

Known Characteristics:
1. TT Accurate (Chebyshev): Floor-value bug for tiny inputs (~1e-38 to ~1e-10)
   - Returns constant 2.98e-05 instead of ~0.5*x
   - Root cause: Polynomial c0 term dominates for tiny x

2. TT Fast (LUT): Correctly handles large negative inputs
   - Formula: GELU(x) = 0.5*x + (A*|x| + B)  [lut2_sign with SGN_UPDATE]
   - Minor bug: Returns ~-0.000104 for x=0 instead of exactly 0

Reference: tt-train/tests/ops/gelu_implementation_reference.md
"""

import pytest
import torch
import ttnn
import numpy as np
from loguru import logger


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Standard GELU formula for computing golden values."""
    import math

    return 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))


# =============================================================================
# TT Accurate Mode: Floor-Value Bug Tests
# =============================================================================


class TestGeluAccurateFloorValueBug:
    """
    Tests for the floor-value bug in TT Accurate (Chebyshev) GELU mode.

    Bug Description:
    For tiny positive inputs (~1e-38 to ~1e-10), the hardware returns a constant
    floor value of approximately 2.98e-05 instead of the expected ~0.5*x.

    Root Cause:
    The 15th-degree Chebyshev polynomial has a constant term c0 = 2.98325768482e-05.
    For tiny inputs, all higher-order terms (c1*x, c2*x^2, ...) become negligible,
    leaving only the c0 floor value.

    Source: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
    """

    # The floor value in BFloat16: 0x37F9 = 2.980232e-05
    FLOOR_VALUE_BF16 = 2.980232e-05

    @pytest.mark.parametrize(
        "input_value,description",
        [
            (1e-10, "tiny normal"),
            (1e-20, "very tiny"),
            (1e-30, "extremely tiny"),
            (1e-38, "near subnormal boundary"),
        ],
    )
    def test_floor_value_bug_demonstration(self, device, input_value, description):
        """
        Demonstrates that tiny positive inputs produce the floor value instead of ~0.5*x.

        This test PASSES when the bug is present, documenting the current behavior.
        """
        # Create input tensor with the tiny value
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)

        # Expected correct result: GELU(x) ≈ 0.5*x for tiny x
        expected_correct = 0.5 * input_value

        # Transfer to device and compute GELU (accurate mode, default)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        result = ttnn.to_torch(tt_result)
        actual_value = result.item()

        # Log the discrepancy
        logger.info(f"Input: {input_value:.2e} ({description})")
        logger.info(f"  Expected (correct): {expected_correct:.6e}")
        logger.info(f"  Actual (hardware):  {actual_value:.6e}")
        logger.info(f"  Floor value:        {self.FLOOR_VALUE_BF16:.6e}")

        # The bug: actual value should be close to floor value, NOT to expected
        # This assertion documents the bug - it passes when bug is present
        assert abs(actual_value - self.FLOOR_VALUE_BF16) < 1e-6, (
            f"Bug may be fixed! Expected floor value {self.FLOOR_VALUE_BF16:.6e}, " f"got {actual_value:.6e}"
        )

        # Log the error magnitude
        if expected_correct != 0:
            relative_error = abs(actual_value - expected_correct) / abs(expected_correct)
            orders_of_magnitude_error = np.log10(relative_error) if relative_error > 0 else 0
            logger.info(f"  Relative error: {relative_error:.2e} ({orders_of_magnitude_error:.0f} orders of magnitude)")

    def test_floor_value_range(self, device):
        """
        Tests that the floor value bug affects a range of tiny inputs.
        """
        # Test values from 1e-38 to 1e-10 (logarithmically spaced)
        exponents = np.linspace(-38, -10, 29)  # 29 points
        input_values = np.power(10.0, exponents).astype(np.float32)

        # Pad to tile size (32x32 = 1024)
        padded_size = 1024
        padded_input = np.zeros(padded_size, dtype=np.float32)
        padded_input[: len(input_values)] = input_values

        torch_input = torch.tensor(padded_input, dtype=torch.bfloat16).reshape(1, 1, 32, 32)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        result = ttnn.to_torch(tt_result).flatten().float().numpy()  # Convert to float for numpy

        # Check that all tiny values produce the floor value
        floor_count = 0
        for i, (inp, out) in enumerate(zip(input_values, result[: len(input_values)])):
            if abs(out - self.FLOOR_VALUE_BF16) < 1e-6:
                floor_count += 1

        logger.info(f"Floor value bug affects {floor_count}/{len(input_values)} tiny inputs")

        # Assert that the bug affects most/all tiny inputs
        assert floor_count >= len(input_values) * 0.9, (
            f"Floor value bug should affect most tiny inputs, but only {floor_count}/{len(input_values)} "
            "produced the floor value. Bug may be partially fixed."
        )

    @pytest.mark.xfail(reason="Known bug: floor value returned for tiny inputs", strict=True)
    def test_tiny_input_correctness(self, device):
        """
        This test expects correct GELU behavior for tiny inputs.
        It is marked xfail because the bug is known - it will FAIL when bug is present.

        When the bug is fixed, this test should PASS and the xfail should be removed.
        """
        input_value = 1e-20
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)

        # Expected correct result
        expected = 0.5 * input_value

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual = ttnn.to_torch(tt_result).item()

        # For correctness, result should be close to expected, not floor value
        # This will fail because of the bug
        relative_tolerance = 0.1  # 10% tolerance
        assert (
            abs(actual - expected) < abs(expected) * relative_tolerance
        ), f"Expected GELU({input_value:.2e}) ≈ {expected:.6e}, got {actual:.6e}"


# =============================================================================
# TT Fast Mode: Large Negative Value Bug Tests
# =============================================================================


class TestGeluFastModeCharacteristics:
    """
    Tests documenting the TT Fast (LUT) GELU mode characteristics.

    FINDING: The initially suspected "large negative value bug" does NOT exist.
    Hardware testing shows that fast mode correctly returns ~0 for large negative inputs.

    Observed characteristics:
    1. Large negative x (x < -3): Returns ~0 (CORRECT)
    2. Large positive x (x > 3): Returns ~x (CORRECT)
    3. x = 0: Returns -0.000104 instead of 0 (MINOR BUG)

    Source: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h
    """

    def test_fast_mode_detailed_values(self, device):
        """
        Detailed test showing exact values from fast GELU mode.
        This test documents the actual hardware behavior for analysis.
        """
        import math

        test_values = [-10.0, -5.0, -4.0, -3.5, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 10.0]

        logger.info("=" * 80)
        logger.info("GELU Fast Mode (LUT) - Detailed Value Analysis")
        logger.info("=" * 80)
        logger.info(f"{'x':>8} | {'fast':>12} | {'accurate':>12} | {'exact':>12} | {'fast-exact':>12}")
        logger.info("-" * 80)

        for val in test_values:
            torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            # Fast mode
            tt_result_fast = ttnn.gelu(tt_input, fast_and_approximate_mode=True)
            actual_fast = ttnn.to_torch(tt_result_fast).item()

            # Accurate mode
            tt_result_acc = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
            actual_acc = ttnn.to_torch(tt_result_acc).item()

            # Exact (golden)
            expected = 0.5 * val * (1 + math.erf(val / math.sqrt(2)))

            diff = actual_fast - expected

            logger.info(f"{val:8.2f} | {actual_fast:12.6f} | {actual_acc:12.6f} | {expected:12.6f} | {diff:12.6f}")

        logger.info("=" * 80)
        # This test always passes - it's for documentation
        assert True

    @pytest.mark.parametrize(
        "input_value",
        [-10.0, -5.0, -4.0, -3.5, -3.0],
    )
    def test_large_negative_correctly_returns_near_zero(self, device, input_value):
        """
        Verifies that fast mode correctly returns ~0 for large negative inputs.

        This disproves the initially suspected bug - the implementation is correct.
        """
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=True)
        actual = ttnn.to_torch(tt_result).item()

        # Fast mode should return ~0 for large negative inputs
        logger.info(f"GELU_fast({input_value}) = {actual}")
        assert abs(actual) < 0.01, f"Fast mode should return ~0 for x={input_value}, got {actual}"

    @pytest.mark.parametrize(
        "input_value",
        [3.0, 3.5, 4.0, 5.0, 10.0],
    )
    def test_large_positive_correctly_returns_identity(self, device, input_value):
        """
        Verifies that fast mode correctly returns ~x for large positive inputs.
        """
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=True)
        actual = ttnn.to_torch(tt_result).item()

        # Fast mode should return ~x for large positive inputs
        logger.info(f"GELU_fast({input_value}) = {actual}")
        assert (
            abs(actual - input_value) < 0.01
        ), f"Fast mode should return ~{input_value} for x={input_value}, got {actual}"

    def test_zero_offset_bug(self, device):
        """
        Tests the minor bug where fast mode returns -0.000104 for x=0 instead of exactly 0.

        This is a minor precision issue in the LUT implementation.
        """
        torch_input = torch.tensor([[0.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=True)
        actual = ttnn.to_torch(tt_result).item()

        # Expected: 0.0, but fast mode returns ~-0.000104
        logger.info(f"GELU_fast(0.0) = {actual} (expected: 0.0)")

        # Document the bug: result is not exactly 0
        assert actual != 0.0, "Bug may be fixed! Expected non-zero offset, got exactly 0"
        assert abs(actual) < 0.001, f"Offset should be small, got {actual}"

    def test_asymmetry_correct(self, device):
        """
        Verifies that fast mode correctly handles GELU asymmetry.

        GELU is asymmetric:
        - For x >> 0: GELU(x) ≈ x (identity)
        - For x << 0: GELU(x) ≈ 0 (saturation)

        This test confirms that the fast mode preserves this asymmetry.
        """
        test_values = [-10.0, -5.0, 5.0, 10.0]

        # Pad to tile size
        padded_size = 1024
        padded_input = np.zeros(padded_size, dtype=np.float32)
        padded_input[: len(test_values)] = test_values

        torch_input = torch.tensor(padded_input, dtype=torch.bfloat16).reshape(1, 1, 32, 32)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=True)
        results = ttnn.to_torch(tt_result).flatten().float().numpy()[: len(test_values)]  # Convert to float for numpy

        logger.info("Testing GELU asymmetry (fast mode):")
        for inp, out in zip(test_values, results):
            expected = gelu_exact(np.array([inp]))[0]
            logger.info(f"  x={inp:6.1f}: GELU_exact={expected:8.4f}, GELU_fast={out:8.4f}")

        # Verify asymmetry: negative -> ~0, positive -> ~x
        for inp, out in zip(test_values, results):
            if inp < -3:
                assert abs(out) < 0.01, f"For x={inp}, expected ~0, got {out}"
            elif inp > 3:
                assert abs(out - inp) < 0.01, f"For x={inp}, expected ~{inp}, got {out}"


# =============================================================================
# Accurate Mode: Correct Behavior Tests (Comparison)
# =============================================================================


class TestGeluAccurateCorrectBehavior:
    """
    Tests demonstrating that TT Accurate mode handles large negative values correctly.

    This provides a comparison to show that the large negative bug is specific
    to the fast/LUT mode, not the accurate/Chebyshev mode.
    """

    @pytest.mark.parametrize(
        "input_value",
        [-10.0, -5.0, -4.0, -3.5, -3.0],
    )
    def test_accurate_mode_large_negative_correct(self, device, input_value):
        """
        Verifies that TT Accurate mode correctly returns ~0 for large negative inputs.
        """
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)

        expected = gelu_exact(np.array([input_value]))[0]

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        # Use accurate mode (default)
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual = ttnn.to_torch(tt_result).item()

        logger.info(f"Accurate mode: GELU({input_value}) = {actual:.6f} (expected: {expected:.6f})")

        # Accurate mode should return correct result (~0 for large negative)
        assert abs(actual - expected) < 0.1, f"Accurate mode should return ~0 for x={input_value}, got {actual:.6f}"


# =============================================================================
# Summary Report Test
# =============================================================================


def test_gelu_known_issues_summary(device):
    """
    Generates a summary report of all known GELU characteristics.

    This test always passes and provides a consolidated view.
    """
    logger.info("=" * 70)
    logger.info("GELU Implementation Characteristics Summary")
    logger.info("=" * 70)

    # Test 1: Floor value bug (Accurate mode)
    logger.info("\n1. TT Accurate Mode: Floor-Value Bug")
    logger.info("-" * 50)

    tiny_values = [1e-10, 1e-20, 1e-30]
    for val in tiny_values:
        torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual = ttnn.to_torch(tt_result).item()
        expected = 0.5 * val
        logger.info(f"   GELU({val:.0e}): expected={expected:.2e}, actual={actual:.2e}")

    # Test 2: Fast mode behavior (correct, not a bug)
    logger.info("\n2. TT Fast Mode: Large Input Behavior (CORRECT)")
    logger.info("-" * 50)

    large_neg_values = [-3.0, -5.0, -10.0]
    for val in large_neg_values:
        torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        # Fast mode
        tt_result_fast = ttnn.gelu(tt_input, fast_and_approximate_mode=True)
        actual_fast = ttnn.to_torch(tt_result_fast).item()

        # Accurate mode (for comparison)
        tt_result_acc = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual_acc = ttnn.to_torch(tt_result_acc).item()

        expected = gelu_exact(np.array([val]))[0]
        logger.info(
            f"   GELU({val:5.1f}): expected={expected:7.4f}, fast={actual_fast:7.4f}, accurate={actual_acc:7.4f}"
        )

    logger.info("\n" + "=" * 70)
    logger.info("Reference: tt-train/tests/ops/gelu_implementation_reference.md")
    logger.info("=" * 70)

    # This test always passes - it's for documentation
    assert True


# =============================================================================
# Formula Verification Tests
# =============================================================================


# LUT coefficients from ckernel_sfpu_gelu.h
# Format: (max_abs_x, slope_A, intercept_B)
LUT_SEGMENTS = [
    (0.5, 0.1928, -0.000104),  # B = 0x86D8 (actual loaded value)
    (1.0, 0.4939, -0.1605),
    (1.5, 0.6189, -0.2797),
    (2.0, 0.6099, -0.2635),
    (3.0, 0.5402, -0.1194),
    (float("inf"), 0.5000, 0.0),
]


def gelu_lut_formula(x: float) -> float:
    """
    Compute expected GELU using the documented LUT formula.

    Formula: GELU(x) = 0.5*x + (A*|x| + B)

    This formula is derived from analyzing the actual hardware implementation:
    - lut2_sign() uses SFPLUTFP32_MOD0_SGN_UPDATE
    - The sign comes from the computation result, NOT from the input
    - A*|x| + B is always positive (for the documented coefficients)
    """
    abs_x = abs(x)

    # Find appropriate segment
    A, B = 0.5, 0.0
    for max_val, slope, intercept in LUT_SEGMENTS:
        if abs_x < max_val:
            A = slope
            B = intercept
            break

    # lut2_sign result = A*|x| + B (always positive due to SGN_UPDATE)
    lut_result = A * abs_x + B

    # result = 0.5*x + lut_result
    return 0.5 * x + lut_result


class TestFormulaVerification:
    """
    Tests that verify the exact code path is executed by comparing
    hardware outputs against values computed from the documented formulas.
    """

    def test_fast_mode_matches_lut_formula(self, device):
        """
        Verify that fast mode outputs match the documented LUT formula.

        Formula: GELU(x) = 0.5*x + (A*|x| + B)

        This confirms the code path uses lut2_sign with SGN_UPDATE,
        not sign(x) * (A*|x| + B).
        """
        test_values = [-10.0, -5.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

        logger.info("=" * 80)
        logger.info("Fast Mode vs LUT Formula Verification")
        logger.info("Formula: GELU(x) = 0.5*x + (A*|x| + B)")
        logger.info("=" * 80)
        logger.info(f"{'x':>8} | {'hardware':>12} | {'formula':>12} | {'diff':>12} | {'segment'}")
        logger.info("-" * 80)

        max_diff = 0.0
        for val in test_values:
            torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=True)
            hardware_output = ttnn.to_torch(tt_result).item()

            expected_from_formula = gelu_lut_formula(val)

            diff = abs(hardware_output - expected_from_formula)
            max_diff = max(max_diff, diff)

            # Determine segment for logging
            abs_val = abs(val)
            segment = "[3.0, inf)" if abs_val >= 3.0 else f"[{abs_val:.1f}, ...)"

            logger.info(
                f"{val:8.2f} | {hardware_output:12.6f} | {expected_from_formula:12.6f} | {diff:12.6f} | {segment}"
            )

        logger.info("-" * 80)
        logger.info(f"Maximum difference: {max_diff:.6f}")
        logger.info("=" * 80)

        # Allow small tolerance for BF16 precision
        assert max_diff < 0.01, f"Hardware output differs too much from formula. Max diff: {max_diff}"

    def test_fast_mode_large_negative_uses_correct_formula(self, device):
        """
        Verify that large negative inputs (|x| >= 3) produce ~0, NOT x.

        This confirms: GELU(-10) = 0.5*(-10) + 0.5*|-10| = -5 + 5 = 0

        NOT the wrong formula: GELU(-10) = 0.5*(-10) + sign(-10)*(0.5*10) = -5 - 5 = -10
        """
        test_values = [-3.0, -5.0, -10.0, -100.0]

        for val in test_values:
            torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=True)
            hardware_output = ttnn.to_torch(tt_result).item()

            # Correct formula: 0.5*x + 0.5*|x| = 0 for negative x
            expected_correct = 0.5 * val + 0.5 * abs(val)  # = 0

            # Wrong formula: 0.5*x + sign(x)*0.5*|x| = x for any x
            expected_wrong = val

            logger.info(
                f"x={val}: hardware={hardware_output:.4f}, correct={expected_correct:.4f}, wrong={expected_wrong:.4f}"
            )

            # Hardware should match correct formula (near 0), NOT wrong formula (equal to x)
            assert (
                abs(hardware_output - expected_correct) < 0.01
            ), f"Hardware should return ~0 for x={val}, got {hardware_output}"
            assert (
                abs(hardware_output - expected_wrong) > 0.1
            ), f"Hardware should NOT return x for x={val}, but got {hardware_output}"

    def test_fast_mode_large_positive_uses_correct_formula(self, device):
        """
        Verify that large positive inputs (|x| >= 3) produce ~x.

        This confirms: GELU(10) = 0.5*(10) + 0.5*|10| = 5 + 5 = 10

        Both formulas give the same result for positive x, so this just confirms behavior.
        """
        test_values = [3.0, 5.0, 10.0, 100.0]

        for val in test_values:
            torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=True)
            hardware_output = ttnn.to_torch(tt_result).item()

            # For |x| >= 3: result = 0.5*x + 0.5*|x| = x for positive x
            expected = val

            logger.info(f"x={val}: hardware={hardware_output:.4f}, expected={expected:.4f}")

            assert (
                abs(hardware_output - expected) < 0.1 * val
            ), f"Hardware should return ~{val} for x={val}, got {hardware_output}"

    def test_accurate_mode_identity_for_large_positive(self, device):
        """
        Verify that accurate mode returns identity (x) for x >= 3.0.

        This verifies the code path:
        if (x >= 3.0f): result = x  // identity for large positive
        """
        test_values = [3.0, 3.5, 4.0, 5.0, 10.0]

        for val in test_values:
            torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
            hardware_output = ttnn.to_torch(tt_result).item()

            logger.info(f"Accurate mode: GELU({val}) = {hardware_output:.6f} (expected: {val})")

            # For x >= 3.0, accurate mode returns exactly x
            assert (
                abs(hardware_output - val) < 0.01
            ), f"Accurate mode should return {val} for x={val}, got {hardware_output}"

    def test_accurate_mode_zero_for_large_negative(self, device):
        """
        Verify that accurate mode returns ~0 for x < -5.5.

        This verifies the code path in calculate_gelu_chebyshev:
        if (x < -5.5f): result = 0  // from the v_if block
        """
        test_values = [-5.5, -6.0, -10.0, -100.0]

        for val in test_values:
            torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
            hardware_output = ttnn.to_torch(tt_result).item()

            expected = gelu_exact(np.array([val]))[0]

            logger.info(f"Accurate mode: GELU({val}) = {hardware_output:.6f} (exact: {expected:.10f})")

            # For x < -5.5, result should be ~0
            assert abs(hardware_output) < 0.001, f"Accurate mode should return ~0 for x={val}, got {hardware_output}"
