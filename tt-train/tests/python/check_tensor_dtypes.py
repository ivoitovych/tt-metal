#!/usr/bin/env python3
"""
Check what data types TTML tensors are using internally.
"""

import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml


def main():
    print("\n" + "=" * 80)
    print("CHECKING TTML TENSOR DATA TYPES")
    print("=" * 80)

    # Create a simple float32 array
    data_np = np.array([[1.0, 2.5, 3.7, 4.123456789]], dtype=np.float32)
    print(f"\nNumPy input:")
    print(f"  dtype: {data_np.dtype}")
    print(f"  values: {data_np}")

    # Convert to TTML tensor
    tensor_ttml = ttml.autograd.Tensor.from_numpy(data_np)
    print(f"\nTTML tensor created")

    # Try to inspect the tensor
    print(f"  Tensor type: {type(tensor_ttml)}")
    if hasattr(tensor_ttml, "get_value"):
        value = tensor_ttml.get_value()
        print(f"  get_value() type: {type(value)}")
        if hasattr(value, "dtype"):
            print(f"  dtype: {value.dtype}")

    # Convert back to numpy
    data_back = tensor_ttml.to_numpy()
    print(f"\nConverted back to NumPy:")
    print(f"  dtype: {data_back.dtype}")
    print(f"  values: {data_back}")

    # Check if values changed
    diff = np.abs(data_np - data_back)
    print(f"\nRound-trip error:")
    print(f"  Max abs diff: {diff.max():.10f}")
    print(f"  Mean abs diff: {diff.mean():.10f}")

    # Test with high-precision values
    print("\n" + "=" * 80)
    print("Testing precision loss with specific values")
    print("=" * 80)

    test_values = np.array(
        [0.123456789, 0.987654321, 1.111111111, 0.333333333, 0.142857143], dtype=np.float32  # 1/7
    ).reshape(1, -1)

    print(f"\nOriginal float32 values:")
    print(test_values[0])

    tensor = ttml.autograd.Tensor.from_numpy(test_values)
    result = tensor.to_numpy()

    print(f"\nAfter TTML round-trip:")
    print(result[0])

    print(f"\nDifferences:")
    for i in range(len(test_values[0])):
        orig = test_values[0, i]
        conv = result[0, i]
        diff_val = abs(orig - conv)
        print(f"  {i}: {orig:.10f} -> {conv:.10f} (diff: {diff_val:.10e})")

    # Simulate bfloat16 conversion
    print("\n" + "=" * 80)
    print("Comparing to expected bfloat16 precision")
    print("=" * 80)

    # bfloat16 has 7 bits of mantissa (vs 23 for float32)
    # Simulate by converting to float32, then to bfloat16 range
    # We can approximate this by checking if precision loss matches bfloat16
    def to_bfloat16_approx(x):
        """Approximate bfloat16 by rounding to ~3-4 decimal places."""
        # bfloat16 has ~2-3 decimal digits of precision
        import struct

        # Convert to bytes, truncate mantissa, convert back
        f32_bytes = struct.pack(">f", x)
        # Zero out lower 16 bits (keep sign + exponent + 7 mantissa bits)
        bf16_approx_bytes = f32_bytes[:2] + b"\x00\x00"
        return struct.unpack(">f", bf16_approx_bytes)[0]

    print(f"\nBfloat16 approximation test:")
    for i in range(len(test_values[0])):
        orig = test_values[0, i]
        ttml_val = result[0, i]
        bf16_val = to_bfloat16_approx(orig)
        print(f"  Original: {orig:.10f}")
        print(f"  TTML:     {ttml_val:.10f}")
        print(f"  bfloat16: {bf16_val:.10f}")
        print(f"  TTML matches bf16: {abs(ttml_val - bf16_val) < 1e-7}")
        print()


if __name__ == "__main__":
    main()
