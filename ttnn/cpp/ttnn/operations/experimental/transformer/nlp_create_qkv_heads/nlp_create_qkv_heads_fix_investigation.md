# Investigation: nlp_create_qkv_heads head_dim < 32 Bug

## Issue Reference
https://github.com/tenstorrent/tt-metal/issues/30418

## Problem Summary

The `ttnn::experimental::nlp_create_qkv_heads` operation produces wrong output shapes when `head_dim < TILE_WIDTH (32)`. This affects small transformer models like BERT with embedding_dim=64 and num_heads=4 (head_dim=16).

## Root Cause Identified

Integer division truncation in `nlp_create_qkv_heads_program_factory.cpp`:

### Interleaved Version (line 62)
```cpp
uint32_t q_out_w_tiles = head_dim / TILE_WIDTH;  // tiles along head_dim
```

### Sharded Version (line 339)
```cpp
uint32_t head_tiles = head_dim / TILE_WIDTH;
```

When `head_dim = 16` and `TILE_WIDTH = 32`:
- `q_out_w_tiles = 16 / 32 = 0` (truncates to zero!)
- Kernel processes zero tiles per head
- Output shape becomes `[1, 4, 32, 32]` instead of `[1, 4, 32, 16]`

## Files to Modify

### Primary Files (Program Factory)
1. `/workspace/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/nlp_create_qkv_heads_program_factory.cpp`

### Device Operation (Output Shape Computation)
2. `/workspace/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/nlp_create_qkv_heads_device_operation.cpp`

### Kernel Code (Data Movement)
3. Reader kernel: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp`
4. Writer kernel: `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`

## Understanding the Operation

### Purpose
Takes fused QKV tensor `[B, 1, S, (num_q_heads + 2*num_kv_heads) * head_dim]` and splits it into:
- Q: `[B, num_q_heads, S, head_dim]`
- K: `[B, num_kv_heads, S, head_dim]`
- V: `[B, num_kv_heads, S, head_dim]`

### Tile-Based Processing
- TT hardware processes data in tiles: `TILE_WIDTH = 32`, `TILE_HEIGHT = 32`
- All dimensions must be tile-aligned (padded to multiples of 32)
- When `head_dim < 32`, the output still needs 1 tile per head (with padding)

## Proposed Fix Strategy

### Option 1: Round Up Tiles (Simple Fix)
```cpp
uint32_t q_out_w_tiles = (head_dim + TILE_WIDTH - 1) / TILE_WIDTH;  // At least 1 tile
```

**Issue**: This changes compile-time args which affects kernel behavior. Need to verify kernel handles partial tiles correctly.

### Option 2: Calculate Total Tiles (Bug Report Suggestion)
```cpp
uint32_t q_num_tiles = (num_q_heads * head_dim + TILE_WIDTH - 1) / TILE_WIDTH;
uint32_t kv_num_tiles = (num_kv_heads * head_dim + TILE_WIDTH - 1) / TILE_WIDTH;
```

**Issue**: This fundamentally changes the tile organization - instead of tiles per head, we'd have total tiles. This would require more extensive kernel changes.

## Key Variables Affected

### Interleaved Version
- `q_out_w_tiles` (line 62) - tiles per head dimension
- `q_out_HtWt` (line 63) - tiles per head (height * width)
- `q_out_CHtWt` (line 64) - total Q tiles
- `kv_out_CHtWt` (line 65) - total KV tiles
- `q_num_tiles` (line 66) - tiles to process for Q
- `kv_num_tiles` (line 67) - tiles to process for KV

### Writer Compile Time Args (line 104-110)
```cpp
std::vector<uint32_t> writer_compile_time_args = {
    (std::uint32_t)q_out_h_tiles,
    (std::uint32_t)q_out_w_tiles,    // THIS IS 0 when head_dim < 32!
    (std::uint32_t)q_out_HtWt,       // THIS IS 0 when head_dim < 32!
    (std::uint32_t)num_q_heads,
    (std::uint32_t)num_kv_heads,
};
```

## Complete Analysis (2025-11-22)

### Device Operation Analysis

The device operation (`nlp_create_qkv_heads_device_operation.cpp`) correctly pads head_dim for output shapes (lines 173-175):
```cpp
if (head_dim % TILE_WIDTH != 0) {
    head_dim = (head_dim / TILE_WIDTH + 1) * TILE_WIDTH;  // Pads to next multiple of 32
}
```

So when head_dim=16, output tensors get shape with head_dim=32 (padded). This is correct.

### Program Factory Bug

The program factory uses the ORIGINAL head_dim (from operation_attributes), not the padded one:

**Interleaved Version (line 62):**
```cpp
uint32_t q_out_w_tiles = head_dim / TILE_WIDTH;  // = 16/32 = 0!
uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;  // = 0!
uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;   // = 0!
```

**Sharded Version (line 339):**
```cpp
uint32_t head_tiles = head_dim / TILE_WIDTH;  // = 0!
uint32_t head_size = head_tiles * single_tile_size;  // = 0!
```

### Kernel Behavior with Zero Tiles

**Writer Kernel** (lines 55, 78, 101):
```cpp
for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
    // Never executes when q_out_w_tiles = 0
    noc_async_write_tile(...);
}
```

**Reader Kernel** (lines 40, 50, 65):
```cpp
for (uint32_t i = 0; i < q_num_tiles; i++) {
    // Never executes when q_num_tiles = 0
    noc_async_read_tile(...);
}
```

Result: Reader reads zero tiles, writer writes zero tiles, but output tensors are allocated with padded shape [B, heads, S, 32].

### Fix Strategy

Use ceiling division to ensure at least 1 tile per head:

**Interleaved (line 62):**
```cpp
uint32_t q_out_w_tiles = (head_dim + TILE_WIDTH - 1) / TILE_WIDTH;  // At least 1 tile
```

**Sharded (line 339):**
```cpp
uint32_t head_tiles = (head_dim + TILE_WIDTH - 1) / TILE_WIDTH;  // At least 1 tile
```

This fix is safe because:
1. Device operation already allocates output with padded size
2. Kernels handle partial tiles (they just copy data into padded positions)
3. For head_dim >= 32, ceiling division gives same result as floor division

## Investigation Tasks

1. [x] Read the reader/writer kernel code to understand tile processing
2. [x] Check device operation for output shape computation
3. [ ] Understand tt-metal partial rebuild process
4. [ ] Create test case reproducing the issue
5. [ ] Implement and test fix
6. [ ] Check if similar variants have same bug (decode, vit, falcon7b, segformer, boltz)

## Building tt-metal Partial Rebuild

For changes to ttnn operations, we can rebuild just the affected components:

```bash
# Rebuild just ttnn library
cmake --build build --target ttnn

# Or rebuild specific test
cmake --build build --target test_nlp_create_qkv_heads
```

## Session History

### 2025-11-21: Initial Investigation
- Created branch: `ivoitovych/fix-nlp-create-qkv-heads-small-head-dim`
- Read bug report and identified root cause
- Found affected code in program factory
- Identified that both Interleaved and Sharded versions have the bug

### 2025-11-22: Complete Analysis
- Examined kernel code (reader and writer)
- Understood that nested loops skip execution when tile counts are zero
- Analyzed device operation - confirms output shapes are padded correctly
- Determined fix: use ceiling division for tile calculations
- Fix preserves backward compatibility for head_dim >= 32

### 2025-11-22: Deep Dive into Fix Complexity
- Attempted simple ceiling division fix - doesn't work because:
  - Input: multiple heads share tiles when head_dim < 32 (packed)
  - Output: each head has its own tile (unpacked with padding)
  - Reader reads 2 tiles for Q (correct), but writer tries to write 4 tiles
  - This requires unpacking tiles, which the kernel doesn't support
- Added validation to reject head_dim < 32 in:
  - `nlp_create_qkv_heads_device_operation.cpp` (TT_FATAL)
  - `nlp_create_qkv_heads.cpp` (TT_FATAL in invoke)
- Added regression test `test_nlp_create_qkv_heads_small_head_dim_validation`
- Validation code compiled into `_ttnncpp.so` but not triggering (build cache issue?)

## Conclusion

The `head_dim < 32` case requires significant kernel changes to support tile unpacking:
1. Input tile contains multiple heads' data (e.g., 2 heads per tile when head_dim=16)
2. Output needs one tile per head (padded to 32)
3. This requires:
   - A compute kernel to split input tiles and add padding
   - Or modified reader/writer with partial tile handling

This is beyond a simple fix and requires architectural changes to the operation.

### Recommended Next Steps
1. Fix the validation triggering issue (possibly clear build cache)
2. Consider adding a compute kernel for the unpacking case
3. Or document the limitation clearly in the API

## Notes

- Previous attempts to fix this issue failed after many hours
- The fix requires understanding tile-based computation on TT hardware
- Simple ceiling division fix aligns tile calculations with padded output shapes
- Must maintain backward compatibility for head_dim >= 32 cases (floor div == ceiling div when divisible)
