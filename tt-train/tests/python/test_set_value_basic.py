#!/usr/bin/env python3
"""Test if set_value() actually updates tensor values."""

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

# Create a simple tensor
print("Creating tensor with ones...")
data = np.ones((4, 4), dtype=np.float32)
tensor = ttml.autograd.Tensor.from_numpy(data.reshape(1, 1, 4, 4))

print("Initial values:")
retrieved = tensor.to_numpy().reshape(4, 4)
print(f"  Mean: {retrieved.mean():.6f}")
print(f"  [0,0]: {retrieved[0,0]}")

# Try to set new value
print("\nCalling set_value with zeros...")
new_data = np.zeros((4, 4), dtype=np.float32)
tensor.set_value(
    ttml.core.from_vector(new_data.flatten().tolist(), tensor.get_value().logical_shape(), tensor.get_value().device())
)

print("After set_value:")
retrieved_after = tensor.to_numpy().reshape(4, 4)
print(f"  Mean: {retrieved_after.mean():.6f}")
print(f"  [0,0]: {retrieved_after[0,0]}")

if retrieved_after.mean() == 0.0:
    print("\n✅ set_value() WORKS!")
else:
    print("\n❌ set_value() FAILED! Values didn't change!")
