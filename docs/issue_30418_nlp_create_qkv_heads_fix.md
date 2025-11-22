# Workaround for nlp_create_qkv_heads with head_dim < 32

## Issue Reference
- **GitHub Issue**: #30418
- **Branch**: `ivoitovych/fix-nlp-create-qkv-heads-small-head-dim-v2`
- **Remote**: Pushed to `myfork` (NOT origin)

## Important Note

**This is a workaround, not a full fix.** The implementation uses high-level TTNN operations which may have performance degradation compared to a proper kernel-level implementation. A real fix would require modifying the reader/writer kernels to handle sub-tile head dimensions with proper tile packing/unpacking.

## Problem Description

The `nlp_create_qkv_heads` operation produces wrong output shapes when `head_dim < TILE_WIDTH (32)`.

### Root Cause

Integer division truncation in the kernel program factory:
```cpp
uint32_t q_out_w_tiles = head_dim / TILE_WIDTH;  // = 0 when head_dim = 16
```

When head_dim < 32, this results in 0 output tiles per head, causing incorrect output.

### Tile Packing Issue

- Input tiles contain multiple heads packed together (e.g., 2 heads per tile when head_dim=16)
- Output expects one tile per head with padding to 32
- Simple ceiling division doesn't work because reader/writer kernels expect 1:1 tile mapping

## Solution Approach (Workaround)

Implement a fallback path using high-level TTNN operations for head_dim < 32. This workaround trades performance for correctness by using general-purpose operations instead of optimized kernels:

1. Untilize to ROW_MAJOR for data manipulation
2. Slice to extract Q, K, V portions
3. Squeeze dimension 1 (size-1 dimension)
4. Reshape and permute to reorganize dimensions
5. Tilize back to TILE layout

## Implementation Details

### File Modified
`/workspace/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp`

### Required Includes
```cpp
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
```

### Fallback Function

```cpp
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> nlp_create_qkv_heads_small_head_dim(
    const Tensor& input_tensor,
    const std::optional<Tensor>& input_tensor_kv,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config)
```

### Key Transformation Steps

For fused QKV input `[B, 1, S, (num_q + 2*num_kv) * head_dim]`:

1. **Untilize**: `ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt)`
2. **Slice**: Use `ttnn::slice()` with `std::array<uint32_t, 4>` for indices
3. **Squeeze**: `ttnn::squeeze(q_rm, 1)` to remove dimension 1
4. **Reshape**: `ttnn::reshape(q_squeezed, ttnn::Shape({batch, seq_len, num_q_heads, head_dim}))`
5. **Permute**: `ttnn::permute(q_reshaped, SmallVector<int64_t>{0, 2, 1, 3})`
6. **Tilize**: `ttnn::to_layout(q_permuted, Layout::TILE, std::nullopt, mem_config)`

### Routing Logic

```cpp
// In NlpCreateHeadsOperation::invoke()
if (head_dim < tt::constants::TILE_WIDTH) {
    return nlp_create_qkv_heads_small_head_dim(
        input_tensor_q, input_tensor_kv, num_q_heads, num_kv_heads_val,
        head_dim, transpose_k_heads, memory_config);
}
```

## TTNN API Notes

### Compilation Issues Encountered and Fixed

1. **`ttnn::to_layout` signature**: Takes 4 arguments max (tensor, layout, dtype, memory_config)
2. **`ttnn::permute` ambiguity**: Use simple 2-argument version without std::nullopt
3. **`ttnn::slice` types**: Use explicit `std::array<uint32_t, 4>{}` for indices
4. **`ttnn::reshape` with dimension mismatch**: Must use `ttnn::squeeze()` to remove size-1 dimensions before reshaping to different rank

### Critical Bug Fix

The reshape from `[B, 1, S, W]` to `[B, S, num_heads, head_dim]` does NOT work correctly in TTNN like PyTorch's `.view()`. You MUST explicitly squeeze the size-1 dimension first:

```cpp
// Wrong - causes data layout issues
auto q_reshaped = ttnn::reshape(q_rm, ttnn::Shape({batch, seq_len, num_q_heads, head_dim}));

// Correct - squeeze first, then reshape
auto q_squeezed = ttnn::squeeze(q_rm, 1);  // [B, 1, S, W] -> [B, S, W]
auto q_reshaped = ttnn::reshape(q_squeezed, ttnn::Shape({batch, seq_len, num_q_heads, head_dim}));
```

## Testing

### Test File
`/workspace/tt-metal/test_nlp_create_qkv_heads_small_head_dim.py`

### Usage
```bash
# Test with head_dim=16 (fallback path)
python test_nlp_create_qkv_heads_small_head_dim.py 16

# Test with head_dim=32 (original kernel path - baseline)
python test_nlp_create_qkv_heads_small_head_dim.py 32
```

### Test Verification

The test verifies:
1. Output shapes are correct (fallback preserves actual head_dim)
2. Data transformation matches PyTorch reference (PCC = 1.000000)

Expected shapes for head_dim=16 (fallback path):
- Q: `[B, num_q_heads, S, 16]` (actual head_dim preserved)
- K: `[B, num_kv_heads, S, 16]` (actual head_dim preserved)
- V: `[B, num_kv_heads, S, 16]` (actual head_dim preserved)

Note: The fallback path using high-level ops preserves the exact head dimension without tile padding, unlike the kernel path which pads to tile boundaries.

## Build Commands

```bash
# Build ttnn (from tt-metal root)
cmake --build build -- -j$(nproc) ttnn

# Run test
python test_nlp_create_qkv_heads_small_head_dim.py 16
```

## Current Status

- [x] Identified root cause (integer division truncation)
- [x] Designed fallback implementation using high-level ops
- [x] Fixed compilation errors with TTNN API
- [x] Added squeeze operation to fix dimension handling
- [x] Verified test passes with PCC = 1.000000 for Q, K, V
- [ ] Run pre-commit checks
- [ ] Create commit
- [ ] Push to myfork

## Important Build/Test Notes

### Library Loading Issue

After building with cmake, the library is created at `build/ttnn/_ttnncpp.so`. However, Python may load from:
- `build/lib/_ttnncpp.so`
- `build_Debug/lib/_ttnncpp.so`

**ALWAYS copy the built library to these locations after building:**
```bash
cp build/ttnn/_ttnncpp.so build/lib/_ttnncpp.so
cp build/ttnn/_ttnn.so build/lib/_ttnn.so
```

### Test Results (Successfully Passed)

Test: `test_nlp_create_qkv_heads_small_head_dim.py` with head_dim=16

Output shapes:
- Q: `[1, 4, 32, 16]` (preserves actual head_dim, no padding)
- K: `[1, 4, 32, 16]`
- V: `[1, 4, 32, 16]`

PCC Results:
- Q PCC: 1.000000
- K PCC: 1.000000
- V PCC: 1.000000

### Backup Location

Working files backed up to: `/workspace/tt-metal/backup_fix_nlp_create_qkv_heads/`

## Debug Tips

If PCC is low but first values match:
- Check if dimension ordering is wrong in reshape
- Verify squeeze is used before reshaping 4D -> 4D with different dims
- Print values at multiple positions to identify patterns:
  - `[0, 0, 0, :]` - head 0, seq 0
  - `[0, 0, 1, :]` - head 0, seq 1
  - `[0, 1, 0, :]` - head 1, seq 0

If actual head N data appears at expected head N+1:
- The reshape is interpreting dimensions incorrectly
- Use squeeze to make dimension transformations explicit
