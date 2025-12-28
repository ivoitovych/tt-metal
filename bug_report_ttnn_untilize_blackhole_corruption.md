# Bug Report: ttnn::untilize Data Corruption on Blackhole P150

**Status**: ONGOING INVESTIGATION - Interface 0-specific bug at z=1, y≥5
**Severity**: Critical
**Component**: tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h
**Hardware**: Blackhole P150
**Branch**: `ivoitovych/tt-train-untilize-blackhole-bug-2`
**Date**: 2025-12-28 (Updated - Session 13)

---

## Executive Summary

The `ttnn::untilize` operation produces corrupted output data when converting tensors from TILE layout to ROW_MAJOR layout on Blackhole P150 hardware. The corruption manifests as a progressive "read even, skip odd" pattern starting at row 3, with severity increasing for later rows. This bug does **not** occur on Wormhole hardware.

**WORKAROUND AVAILABLE (2025-12-17)**: A working workaround has been found! Use `ttnn::untilize(tensor, std::nullopt, true, false)` (set `use_pack_untilize=false`) to use the slow path which produces correct results.

Key findings:
1. **WORKAROUND FOUND**: The "slow path" (`use_pack_untilize=False`) uses `llk_unpack_untilize` + regular `llk_pack` instead of `llk_pack_untilize` and produces correct results
2. **BFloat16 precision was masking the real bug**: Initial "corruption" pattern (300→300, 301→300) was actually BFloat16 precision loss, not corruption
3. **Real bug (Float32)**: Fast path shows face interleaving corruption - data from adjacent rows incorrectly mixed into wrong face positions
4. **Second bug identified**: `ttnn::argmax` fails on untilized tensors for rows ≥ 21 (separate issue)
5. **`program_packer_untilized_destination` is EMPTY on Blackhole** - the entire function body is commented out
6. **Both DST_ACCESS_STRIDED_MODE and DST_ACCESS_NORMAL_MODE produce identical corruption** - ruling out DST access mode as the cause

---

## Table of Contents

1. [Symptoms and Impact](#symptoms-and-impact)
2. [Reproduction Steps](#reproduction-steps)
3. [Environment](#environment)
4. [Corruption Pattern Analysis](#corruption-pattern-analysis)
5. [Technical Background](#technical-background)
6. [Root Cause Investigation](#root-cause-investigation)
7. [Fix Attempts](#fix-attempts)
8. [Key Findings](#key-findings)
9. [Comparison: Blackhole vs Wormhole](#comparison-blackhole-vs-wormhole)
10. [Recommendations](#recommendations)
11. [Appendix: Code References](#appendix-code-references)

---

## Symptoms and Impact

### Primary Symptom
Data corruption when calling `ttnn::untilize()` on tensors stored in TILE layout.

### Observed Behavior
- Argmax operations return garbage values (e.g., `3201515335` instead of expected values < 64)
- Sequential data becomes corrupted with "even indices duplicated, odd indices skipped" pattern
- Corruption is **progressive** - rows 0-2 are often correct, row 3+ shows increasing corruption

### Impact
- Any operation chain involving `ttnn::untilize` on Blackhole produces incorrect results
- Affects tt-train training on Blackhole P150/P300 hardware
- Blocks adoption of Blackhole for production workloads requiring layout conversion

### Originally Failing Test
```
TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask
```

---

## Reproduction Steps

### Build
```bash
cd ~/tt/tt-metal
./build_metal.sh -b Release --build-tt-train
```

### Run Debug Tests
```bash
# Run all debug untilize tests
./build_Release/tt-train/tests/ttml_tests --gtest_filter="DebugUntilizeTest.*"

# Run specific test showing corruption
./build_Release/tt-train/tests/ttml_tests --gtest_filter="DebugUntilizeTest.UntilizeOnly8Rows"
```

### Minimal Reproduction Code
```cpp
#include <gtest/gtest.h>
#include <core/ttnn_all_includes.hpp>
#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

TEST(UntilizeBug, MinimalRepro) {
    ttml::autograd::ctx().open_device();
    auto* device = &ttml::autograd::ctx().get_device();

    // Create tensor with sequential values
    xt::xarray<float>::shape_type shape = {1, 1, 8, 64};
    xt::xarray<float> a = xt::zeros<float>(shape);
    for (size_t row = 0; row < 8; ++row) {
        for (size_t col = 0; col < 64; ++col) {
            a(0, 0, row, col) = static_cast<float>(row * 100 + col);
        }
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);  // Creates TILE layout
    auto untilized = ttnn::untilize(tensor_a);            // BUG: Corrupts data
    auto vec = ttml::core::to_vector(untilized);

    // Check output - rows 3+ will be corrupted
    for (size_t row = 0; row < 8; ++row) {
        std::cout << "Row " << row << ": ";
        for (size_t col = 0; col < 10; ++col) {
            std::cout << vec[row * 64 + col] << " ";
        }
        std::cout << std::endl;
    }

    ttml::autograd::ctx().close_device();
}
```

---

## Environment

| Property | Value |
|----------|-------|
| Hardware | Blackhole P150 (PCI device 1e52:b140) |
| Host OS | Ubuntu 22.04.5 LTS, Kernel 5.15.0-164-generic |
| Machine | vm3 |
| Firmware | 19.3.0 |
| ETH FW | 1.7.1 |
| KMD version | 2.6.1 |
| tt-smi version | 3.0.39 |
| tt-metal branch | `ivoitovych/tt-train-untilize-blackhole-bug-2` |
| Original commit | `403df4beb0f31a2b771349e58a02a98d859b9039` (when bug was discovered) |

---

## Corruption Pattern Analysis

**Note (Session 5)**: The pattern below was observed with BFloat16 data and was initially attributed to untilize corruption. However, investigation revealed this is actually **BFloat16 precision loss** (odd values rounding to even), NOT corruption. See [Session 5: Workaround Discovery](#session-5-workaround-discovery-2025-12-17) for the actual Float32 corruption pattern (face interleaving).

### Input Data (Sequential Values)
```
Row 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12...
Row 1: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109...
Row 2: 200, 201, 202, 203, 204, 205, 206, 207, 208, 209...
Row 3: 300, 301, 302, 303, 304, 305, 306, 307, 308, 309...
Row 4: 400, 401, 402, 403, 404, 405, 406, 407, 408, 409...
Row 5: 500, 501, 502, 503, 504, 505, 506, 507, 508, 509...
Row 6: 600, 601, 602, 603, 604, 605, 606, 607, 608, 609...
Row 7: 700, 701, 702, 703, 704, 705, 706, 707, 708, 709...
```

### Output Data (CORRUPTED)
```
Row 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9           (CORRECT)
Row 1: 100, 101, 102, 103, 104, 105...        (CORRECT)
Row 2: 200, 201, 202, 203, 204, 205...        (CORRECT)
Row 3: 300, 300, 302, 304, 304, 304, 306, 308, 308, 308  (CORRUPTED)
Row 4: 400, 400, 402, 404, 404, 404, 406, 408, 408, 408  (CORRUPTED)
Row 5: 500, 500, 502, 504, 504, 504, 506, 508, 508, 508  (CORRUPTED)
Row 6: 600, 600, 600, 604, 604, 604, 608, 608, 608, 608  (CORRUPTED - worse)
Row 7: 700, 700, 704, 704, 704, 704, 704, 708, 708, 708  (CORRUPTED - even worse)
```

### Pattern Characteristics
1. **Rows 0-2**: Always correct
2. **Row 3+**: Progressive corruption begins
3. **Pattern**: "Read even index, skip odd index, duplicate previous"
4. **Severity**: Increases with row number (more positions affected)
5. **Hypothesis**: Accumulating address calculation error starting at row 3

### Tile/Face Boundary Analysis
- TILE_R_DIM = 32, TILE_C_DIM = 32
- FACE_R_DIM = 16, FACE_C_DIM = 16
- Row 3 is within Face 0 (rows 0-15) - corruption starts before face boundary
- This suggests the issue is within single-face processing, not face transitions

---

## Technical Background

### DEST Memory Layout (32x32 Tile with 4 Faces)
```
+------------------+------------------+
|     Face 0       |     Face 1       |
|  rows 0-15       |  rows 0-15       |
|  cols 0-15       |  cols 16-31      |
+------------------+------------------+
|     Face 2       |     Face 3       |
|  rows 16-31      |  rows 16-31      |
|  cols 0-15       |  cols 16-31      |
+------------------+------------------+
```

### Packer Operation
The untilize operation reads data from DEST registers and writes to L1 memory in row-major order. Key components:

1. **MOP (Micro-Operation) Template**: Controls the packing loop structure
2. **PACR Instruction**: Performs the actual pack operation
3. **Address Counters (X, Y, Z, W)**: Track source (DEST) and destination (L1) addresses
4. **Stride Registers**: Define address increments for multi-dimensional access

### TT_OP_PACR Instruction Format

**Blackhole (12 parameters)**:
```cpp
TT_OP_PACR(
    p_pacr::CFG_CTXT_0,           // Config context
    p_pacr::NO_ROW_PAD_ZERO,      // Row padding
    p_pacr::DST_ACCESS_STRIDED_MODE,  // or DST_ACCESS_NORMAL_MODE
    ADDR_MOD_0,                   // Address modifier
    p_pacr::ADDR_CNT_CTXT_0,      // Address counter context
    ZERO_OUTPUT_FLAG,             // Zero output
    PACK_INTF_SEL,                // Packer interface selection
    0,                            // Reserved
    MEGAROW,                      // Face concatenation mode
    p_pacr::NO_CTXT_CTRL,         // Context control
    0,                            // Reserved
    last_bit                      // Last datum in row
)
```

**Wormhole (7 parameters)**:
```cpp
TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0)
```

### DataFormat and Strides
```cpp
// DataFormat enum (Blackhole)
enum class DataFormat : std::uint8_t {
    Float32   = 0,
    Float16   = 1,
    Float16_b = 5,  // BFloat16
    // ...
};

// For BFloat16: pack_src_format & 0x3 = 5 & 0x3 = 1 (matches Float16)
// Therefore:
//   x_stride = 2 bytes
//   y_stride = FACE_C_DIM * x_stride = 16 * 2 = 32 bytes
//   z_stride = 2 * FACE_R_DIM * y_stride = 2 * 16 * 32 = 1024 bytes
```

---

## Root Cause Investigation

### Investigated Hypotheses

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| DST_ACCESS_STRIDED_MODE bug | **Disproved** | NORMAL_MODE shows identical corruption |
| MEGAROW not set correctly | **Disproved** | Adding MEGAROW didn't fix corruption |
| Wrong stride calculation | **Possible** | Strides appear correct for data format |
| Address modifier configuration | **Possible** | Different from Wormhole |
| MOP loop structure | **Possible** | Different outer/inner loop semantics |
| `program_packer_untilized_destination` empty | **Likely** | Function is completely commented out |
| Hardware/firmware bug | **Possible** | May need HW team input |

### Critical Finding: Empty `program_packer_untilized_destination`

**Location**: `tt_llk_blackhole/common/inc/cpack_common.h:520-543`

The Blackhole implementation has this function **completely commented out**:

```cpp
template <uint32_t block_ct_dim, uint32_t full_ct_dim, bool diagonal = false>
inline void program_packer_untilized_destination(const uint32_t addr, const uint32_t pack_dst_format)
{
    // ALL CODE IS COMMENTED OUT!
    // const uint32_t block_size = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * TILE_C_DIM * (TILE_R_DIM/4));
    // constexpr uint32_t offset0 = 0;
    // const uint32_t offset1 = (1*block_size)/16;
    // ... (all TT_SETDMAREG and TTI_WRCFG calls commented out)
}
```

**Wormhole version** (working) properly configures 4 packer L1 destination addresses:
```cpp
// Calculates offsets for 4 packer interfaces
const uint32_t offset0 = 0;
const uint32_t offset1 = (1 * row_num_datums * block_size) / 16 / TILE_C_DIM;
const uint32_t offset2 = (2 * row_num_datums * block_size) / 16 / TILE_C_DIM;
const uint32_t offset3 = (3 * row_num_datums * block_size) / 16 / TILE_C_DIM;

// Programs all 4 packer output addresses
TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset0), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 0));
// ... (sets OUTPUT_ADDR + 1, 2, 3)

// Writes to hardware registers
TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32, p_gpr_pack::OUTPUT_ADDR);
TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32, p_gpr_pack::OUTPUT_ADDR + 1);
TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_L1_Dest_addr_ADDR32, p_gpr_pack::OUTPUT_ADDR + 2);
TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_L1_Dest_addr_ADDR32, p_gpr_pack::OUTPUT_ADDR + 3);
```

This missing configuration is likely a significant contributor to the corruption.

---

## Fix Attempts

### Attempt 1: Change `untilize=false` to `untilize=true` in API
- **File**: `pack_untilize.h` line 68
- **Rationale**: Ensure untilize mode flag is set
- **Result**: No change, same corruption pattern
- **Status**: Reverted

### Attempt 2: Add y_stride programming to `_llk_pack_untilize_init_`
- **File**: `llk_pack_untilize.h`
- **Rationale**: Ensure stride registers are properly configured
- **Result**: No change, same corruption pattern
- **Status**: Reverted

### Attempt 3: Port Wormhole-style explicit row iteration
- **Changes**:
  - Changed MOP outer loop to iterate over tiles (not rows)
  - Added explicit C++ for loop for row iteration
  - Changed from DST_ACCESS_STRIDED_MODE to DST_ACCESS_NORMAL_MODE
- **Result**: Different corruption pattern - face offset issue
  - Row 0: correct
  - Row 1: getting face 1 data (offset +32)
  - Row 2: getting row 1 data
  - Row 3: getting row 1 face 1 data
- **Status**: Reverted

### Attempt 4: MEGAROW fix with template parameter
- **Changes**:
  - Added `is_fp32_dest_acc_en` template parameter to match API
  - Added `const uint MEGAROW = (num_faces > 1) ? 1 : 0;`
  - Updated both PACR instructions to use MEGAROW
- **Result**: Same corruption pattern - MEGAROW alone doesn't fix it
- **Status**: Partially kept (MEGAROW variable added)

### Attempt 5: Switch to DST_ACCESS_NORMAL_MODE
- **Changes**: Changed both PACR instructions from STRIDED_MODE to NORMAL_MODE
- **Result**: **Same corruption pattern as STRIDED_MODE!**
- **Status**: Currently active for testing

### Attempt 6: Address modifier configuration
- **Changes**: Tried different ADDR_MOD configurations matching Wormhole
- **Result**: Various corruption patterns, none correct
- **Status**: Reverted

### Attempt 7 (2025-12-17): Full Wormhole-style port - DEVICE HUNG
- **Changes**:
  - **Enabled `program_packer_untilized_destination`**: Uncommented and adapted the function body in `cpack_common.h`. Added high bit for L1 validity flag, set up all 4 OUTPUT_ADDR registers with proper offsets, wrote all 4 THCON registers.
  - **Updated replay buffer**: Changed from 4 to 12 instructions. Added updates for OUTPUT_ADDR+0 through OUTPUT_ADDR+3. Added WRCFG for all 4 THCON L1 destination registers.
  - **Changed MOP structure**: Changed from outer=rows, inner=tiles to outer=tiles, inner=1. Added explicit C++ row iteration in `_llk_pack_untilize_`. Added Wormhole-style address modifiers (y_src.incr=15).
  - **Updated API wrapper**: Changed `_llk_pack_untilize_init_` call in `llk_pack_api.h` to match new signature.
- **Files Modified**:
  - `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h`
  - `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h`
  - `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_api.h`
- **Result**: **DEVICE HUNG** - test execution hung requiring process kill
- **Conclusion**: Blackhole packer hardware has fundamentally different constraints than Wormhole. Direct port is not viable.
- **Status**: All changes reverted using `git restore`

---

## Session 5: Workaround Discovery (2025-12-17)

### BFloat16 Precision Masking the Real Bug

The original "corruption" pattern observed with BFloat16 data was **NOT actually corruption** - it was BFloat16 precision loss:
- BFloat16 has 7 mantissa bits
- For values 256-512, precision is ~2 units
- Odd numbers round to nearest even: 301→300, 303→304, etc.

This was masking the real untilize bug. Testing with Float32 reveals the actual issue.

### REAL pack_untilize Bug (Float32)

With Float32, the fast path (`use_pack_untilize=True`) shows **face interleaving corruption**:

```
Expected Row 0: [0-15], [16-31], [32-47], [48-63]
Actual Row 0:   [0-15], [100-115], [32-47], [132-147]
```

- Face pairs from adjacent rows are being incorrectly interleaved
- Rows 3 and 7 have ZEROS in second half (data missing)
- 288 out of 512 values incorrect in 8x64 tensor test

### WORKAROUND: Slow Path Works!

**`ttnn::untilize(tensor, std::nullopt, true, false)` produces CORRECT results!**

Test with 8x64 Float32 tensor:
- Fast path (`use_pack_untilize=true`): 288/512 mismatches
- Slow path (`use_pack_untilize=false`): **0/512 mismatches**

The slow path uses `llk_unpack_untilize` + regular `llk_pack` instead of the buggy `llk_pack_untilize`.

### C++ API Usage

```cpp
// CORRECT: Use slow path on Blackhole
auto untilized = ttnn::untilize(tensor, std::nullopt, true, false);
//                                       memory_config, multicore, use_pack_untilize=FALSE

// BUGGY: Default fast path (corrupts data on Blackhole)
auto untilized = ttnn::untilize(tensor);  // use_pack_untilize defaults to true
```

### Python API Usage

```python
# CORRECT: Use slow path on Blackhole
untilized = ttnn.untilize(tensor, use_pack_untilize=False)

# BUGGY: Default fast path
untilized = ttnn.untilize(tensor)  # use_pack_untilize defaults to True
```

### SECOND BUG: Argmax on Untilized Tensor

Even when untilize produces correct data, `ttnn::argmax` fails for rows ≥ 21:

```
Untilized data: All correct (verified by reading to CPU)
Argmax results: [10,10,10...10,10,0,0,0,0,0,0,0,0,0,0,0]
                         ↑ rows 0-20 correct, rows 21-31 wrong
```

**WORKAROUND**: Round-trip through CPU fixes argmax:
```python
# Python - round-trip through CPU
data = ttnn.to_torch(untilized_tensor)
fresh = ttnn.from_torch(data, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
result = ttnn.argmax(fresh, ...)  # Now works correctly for all rows
```

```cpp
// C++ - round-trip through CPU using ttml helpers
auto vec = ttml::core::to_vector(untilized_tensor);
auto fresh = ttml::core::from_vector(vec, shape, device, ttnn::Layout::ROW_MAJOR);
auto result = ttnn::argmax(fresh, ...);  // Now works correctly for all rows
```

This second bug appears to be related to tensor metadata or internal state after untilize, not the actual data values.

---

## Key Findings

### Finding 1: DST Access Mode is NOT the Root Cause
Both `DST_ACCESS_STRIDED_MODE` and `DST_ACCESS_NORMAL_MODE` produce **identical corruption patterns**. This definitively rules out the DST access mode as the cause.

### Finding 2: Corruption Starts at Row 3 Consistently
Regardless of configuration changes, rows 0-2 are always correct and corruption begins at row 3. This suggests:
- The bug manifests after a specific number of PACR operations
- Possibly related to address counter overflow or wraparound
- May involve state accumulation across rows

### Finding 3: Progressive Corruption Pattern
The "read even, skip odd" pattern worsens with increasing row number, suggesting an accumulating error in address calculation rather than a fixed offset bug.

### Finding 4: Missing Packer Destination Configuration
The `program_packer_untilized_destination` function is completely non-functional on Blackhole, while Wormhole uses it to configure 4 separate packer output addresses. This may explain why multi-row output is corrupted.

### Finding 5: Architectural Differences
| Aspect | Wormhole | Blackhole |
|--------|----------|-----------|
| Row iteration | Explicit C++ loop | MOP outer loop |
| MOP outer loop | Iterates over tiles | Iterates over rows (face_r_dim=16) |
| DEST access mode | NORMAL | STRIDED (or NORMAL - both broken) |
| PACR format | 7 parameters | 12 parameters |
| PACK_SEL macro | Available | **NOT available** |
| `program_packer_untilized_destination` | Functional | **Empty/commented out** |

### Finding 6 (2025-12-17): Wormhole-Style Port Causes Device Hang
Attempting a comprehensive port of the Wormhole implementation to Blackhole resulted in the device hanging during test execution. This was attempted with:
- Enabling `program_packer_untilized_destination` (uncommenting function body)
- Updating replay buffer to configure all 4 L1 destination registers
- Changing MOP structure to match Wormhole (explicit row iteration)
- Configuring Wormhole-style address modifiers

**Conclusion**: The Blackhole packer hardware has fundamentally different constraints than Wormhole. The Wormhole approach cannot be directly ported. **This investigation is now BLOCKED** and requires Tenstorrent hardware engineer involvement.

---

## Comparison: Blackhole vs Wormhole

### MOP Configuration

**Wormhole** (`llk_pack_untilize.h`):
```cpp
constexpr uint MOP_INNER_LOOP = 1;
constexpr uint MOP_OUTER_LOOP = block_ct_dim;

// MOP iterates over tiles, explicit loop for rows
for (std::uint32_t row = 0; row < num_rows; row++) {
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_dst_offset);
    ckernel::ckernel_template::run();
    TTI_ADDRCRXY(p_setadc::PAC, 0, 0, 1, 0, 0b0010);  // Advance row
}
```

**Blackhole** (`llk_pack_untilize.h`):
```cpp
constexpr uint MOP_INNER_LOOP = block_ct_dim;
const uint MOP_OUTER_LOOP = face_r_dim;  // 16 rows

// MOP handles rows internally, face loop is explicit
for (std::uint32_t face = 0; face < num_faces_per_rdim_tile; face++) {
    ckernel::ckernel_template::run();
    TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
}
```

### Address Modifier Configuration

**Wormhole**:
```cpp
addr_mod_pack_t {
    .y_src = {.incr = 15},  // +15, combined with INCADCXY +1 = +16 (next face)
}.set(ADDR_MOD_0);

addr_mod_pack_t {
    .y_src = {.incr = 0, .clr = 0, .cr = 1},  // Carry/reset
}.set(ADDR_MOD_1);

addr_mod_pack_t {
    .y_src = {.incr = 0, .clr = 1, .cr = 0},  // Clear
}.set(ADDR_MOD_2);
```

**Blackhole**:
```cpp
addr_mod_pack_t {
    .y_src = {.incr = 0, .clr = 0},  // No increment!
}.set(ADDR_MOD_0);
// Only ADDR_MOD_0 is configured
```

---

## Recommendations

### Status: WORKAROUND AVAILABLE

A working workaround has been found! The fast path bug still needs to be fixed for performance, but production code can use the slow path.

### Immediate Workaround

**Use `use_pack_untilize=false` in all `ttnn::untilize` calls on Blackhole:**

```cpp
// C++
auto untilized = ttnn::untilize(tensor, std::nullopt, true, false);
```

```python
# Python
untilized = ttnn.untilize(tensor, use_pack_untilize=False)
```

This uses the slow path (`llk_unpack_untilize` + regular `llk_pack`) which produces correct results.

### Performance Impact

The slow path has higher latency than the optimized `llk_pack_untilize` fast path. For performance-critical applications, the fast path bug should still be fixed. Consider making `use_pack_untilize=False` the **default for Blackhole** until the fast path is fixed.

### Required Actions (Still Needed for Performance Fix)

1. **File Bug with Tenstorrent LLK/Hardware Team** (Priority: MEDIUM - workaround available)
   - The `program_packer_untilized_destination` function is completely empty on Blackhole
   - The Wormhole approach cannot be directly ported without causing device hang
   - Request documentation on Blackhole-specific packer architecture differences
   - Request guidance on how `llk_pack_untilize` should be implemented for Blackhole

2. **Consider Platform-Specific Default**
   - Make `use_pack_untilize=False` the default for Blackhole hardware
   - Keep `use_pack_untilize=True` as default for Wormhole where it works correctly

3. **Investigate Second Bug (argmax on untilized tensors)**
   - `ttnn::argmax` fails on untilized tensors for rows ≥ 21
   - Appears to be tensor metadata/internal state issue, not data corruption
   - File separate bug report for this issue

### Summary of Bugs Identified

| Bug | Description | Workaround |
|-----|-------------|------------|
| pack_untilize corruption | Face interleaving corruption on Blackhole fast path | Use `use_pack_untilize=False` |
| argmax on untilized | Fails for rows ≥ 21 on untilized tensors | Round-trip through CPU |

---

## Appendix: Code References

### Key Files

**LLK (Low-Level Kernel) Layer - Where the bug manifests:**
| File | Purpose |
|------|---------|
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h` | Main Blackhole implementation |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_pack_untilize.h` | Working Wormhole reference |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h` | Packer common functions (program_packer_untilized_destination is EMPTY here) |
| `tt_metal/include/compute_kernel_api/pack_untilize.h` | API layer |

**TTNN Layer - Higher-level untilize operation:**
| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/` | TTNN untilize kernel implementation |
| `ttnn/cpp/ttnn/operations/data_movement/untilize/untilize.cpp` | TTNN untilize operation entry point |

**Test Files:**
| File | Purpose |
|------|---------|
| `tt-train/tests/ttnn_fixed/debug_untilize_test.cpp` | Debug test suite for isolation |
| `tt-train/tests/ttnn_fixed/trivial_ttnn_ops_test.cpp` | Original failing test (line 277-298) |

### Test File Location
```
tt-metal/tt-train/tests/ttnn_fixed/debug_untilize_test.cpp
```

### Build and Test Commands

**Option 1: Full tt-metal build** (slower, builds entire stack)
```bash
# Build
./build_metal.sh -b Release --build-tt-train

# Run tests
./build_Release/tt-train/tests/ttml_tests --gtest_filter="DebugUntilizeTest.*"
./build_Release/tt-train/tests/ttml_tests --gtest_filter="TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask"
```

**Option 2: Standalone tt-train build** (faster, especially with ccache)
```bash
# Build
cd tt-train
cmake -DCMAKE_BUILD_TYPE=Release -B build -GNinja
cmake --build build

# Run tests
./tt-train/build/tests/ttml_tests --gtest_filter="DebugUntilizeTest.*"
./tt-train/build/tests/ttml_tests --gtest_filter="TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask"
```

### Git Information
```
Branch: ivoitovych/tt-train-untilize-blackhole-bug-2
Repository: tenstorrent/tt-metal
Submodule: tt_metal/third_party/tt_llk (separate repo)
```

### Original Test Error Output

The original failing test (`TestSamplingPositiveTemperatureWithMask`) produced these garbage argmax values due to untilize corruption:

```
/home/ivoitovych/tt/tt-metal/tt-train/tests/ttnn_fixed/trivial_ttnn_ops_test.cpp:296: Failure
Expected: (v) < (64), actual: 3201515335 vs 64
Expected: (v) < (64), actual: 3217342221 vs 64
Expected: (v) < (64), actual: 1066712917 vs 64
Expected: (v) < (64), actual: 3200859828 vs 64
Expected: (v) < (64), actual: 3202563797 vs 64
Expected: (v) < (64), actual: 3207970364 vs 64
Expected: (v) < (64), actual: 3217375136 vs 64
Expected: (v) < (64), actual: 3204431830 vs 64
Expected: (v) < (64), actual: 1072873412 vs 64
Expected: (v) < (64), actual: 3207184383 vs 64
Expected: (v) < (64), actual: 1058192921 vs 64
```

These values (e.g., 3201515335) are the result of argmax operating on corrupted untilized data, where BFloat16 values were incorrectly packed, causing argmax to return indices outside the valid range.

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-12-15 | Investigation Team | Initial discovery and Phase 1-2 analysis |
| 2025-12-16 | Investigation Team | Fix attempts 4-6, comprehensive documentation |
| 2025-12-17 | Investigation Team | **Session 4**: Attempt 7 (Wormhole-style port) - DEVICE HUNG. Investigation blocked. |
| 2025-12-17 | Investigation Team | **Session 5**: **WORKAROUND FOUND!** Discovered BFloat16 precision masking real bug. Slow path (`use_pack_untilize=False`) works correctly. Identified second bug (argmax on untilized tensors). Status changed from BLOCKED to WORKAROUND AVAILABLE. |
| 2025-12-17 | Investigation Team | **Session 6**: Started fixing fast path instead of using workaround. |
| 2025-12-18 | Investigation Team | **Session 7-8**: Partial fix achieved - dual L1 addresses + set_packer_strides makes face pair 0 work. Face pair 1 rows 21-31 still corrupted. DST_ACCESS_STRIDED_MODE confirmed required for multi-tile cases. |
| 2025-12-27 | Investigation Team | **Session 9-10**: Exhaustive testing of all remaining software fixes. All failed. Strong evidence points to Blackhole packer hardware bug for counter combination z=1, y>=5. Status changed to LIKELY HARDWARE BUG. |
| 2025-12-28 | Investigation Team | **Session 13**: Key discovery - corruption is INTERFACE 0 SPECIFIC at z=1! Only 176 errors = 11 rows × 16 cols = interface 0 only. Interface 1 works correctly at z=1. Tested ch1 counter sync (failed - made worse) and 8-row model (no effect). Status changed to ONGOING INVESTIGATION. |

---

## Session 6-10 Updates: Partial Fix and Hardware Bug Evidence

### Partial Fix Achieved (Session 7)

Applied fixes to `llk_pack_untilize.h`:
1. **Dual L1 addresses**: Set up OUTPUT_ADDR and OUTPUT_ADDR+1 for both packer interfaces
2. **Proper z_stride**: Added `set_packer_strides<true, false>()` call
3. **Dual L1 update in replay buffer**: Updated both THCON_SEC0_REG1 and THCON_SEC0_REG8

**Results after partial fix:**
| Test Case | Result | Details |
|-----------|--------|---------|
| 8x64 (2 tiles) | **PASS** (0 errors) | Only uses face pair 0 (8 rows) |
| 32x32 (1 tile, 4 faces) | FAIL (176 errors) | Rows 21-31 corrupted |
| 32x64 (2 tiles) | FAIL (360 errors) | Rows 20-31 corrupted |

Face pair 0 (rows 0-15) works completely. Face pair 1 rows 16-20 work, but rows 21-31 have "skip odd, duplicate even" pattern.

### Session 10 Exhaustive Testing (All Failed)

1. ✗ Set up all 4 L1 addresses (SEC0 + SEC1) - no change
2. ✗ Update replay buffer for all 4 addresses - no change
3. ✗ Configure channel 1 z_stride - no change
4. ✗ Use regular pack's addr_mod configuration - no change
5. ✗ Complete counter reset before face pair 1 - no change

### Evidence Supporting Hardware Bug

1. **Corruption is position-specific** (z=1, y>=5), not configuration-dependent
2. **All software fixes have no effect** on the corruption pattern
3. Face pair 0 (z=0) works correctly for all 16 rows
4. Face pair 1 (z=1) works for rows 0-4 but fails at row 5+
5. **The slow path works correctly** (uses different HW path: unpack+pack instead of pack_untilize)

### Conclusion

The remaining corruption in face pair 1 rows 5-15 (global rows 21-31) appears to be a **hardware bug in the Blackhole packer** where specific counter value combinations (z=1 AND y>=5) cause incorrect DEST addressing during TWO_INTFS_ACTIVE mode.

### Recommendations

1. **Use workaround**: `use_pack_untilize=False` for Blackhole with 4-face tiles
2. **File hardware bug report** with Tenstorrent describing the specific counter combination
3. **Investigate if newer Blackhole firmware/silicon has a fix**

---

## Session 13 Updates: Interface 0-Specific Bug Identified (2025-12-28)

### Key Discovery

**Critical Finding**: The 176 errors for 32x32 test = 11 rows × 16 columns!

This arithmetic reveals:
- **Interface 0** (face 2, columns 0-15) has corruption at z=1, y≥5
- **Interface 1** (face 3, columns 16-31) works **correctly** at z=1!

The issue is NOT a general hardware bug - it's specific to interface 0's DEST addressing when z=1.

### Fix Attempts (Session 13)

**Attempt 16: Channel 1 Counter Synchronization**
- Added ch1_y increment: `TT_OP_INCADCXY(p_setadc::PAC, 1, 0, 1, 0)` (Ch1_Y=1)
- Reset both ch0_y and ch1_y between face pairs: bitmask 0b1010
- Reset all XY and ZW counters at start: bitmask 0b1111
- **Result: MUCH WORSE** - Row 0 had Y-stride values (0, 16, 32, 48) instead of X values
- **Reverted** - Only ch0 counters should be modified

**Attempt 17: 8-Row Model (Like Wormhole)**
- Changed MOP_OUTER_LOOP from 16 to 8 rows
- Run MOP twice per 16-row face pair (first 8 rows, then next 8)
- **Result: NO CHANGE** - Same 176/360 error pattern

### Technical Analysis

**Why ch1 counter sync failed:**
- ch0 and ch1 are address counter contexts, NOT packer interfaces
- Both interfaces likely use ch0 for DEST addressing
- Modifying ch1 caused incorrect addressing

**Why 8-row model didn't help:**
- The corruption starts at y=5 within face pair 1, not at an 8-row boundary
- The issue is position-specific (z=1, y≥5), not structure-specific

### Current Understanding

| Aspect | Interface 0 | Interface 1 |
|--------|------------|-------------|
| Face at z=0 | Face 0 (cols 0-15) | Face 1 (cols 16-31) |
| Face at z=1 | Face 2 (cols 0-15) | Face 3 (cols 16-31) |
| L1 Register (z=1) | SEC1_REG1 | SEC1_REG8 |
| Status at z=1, y<5 | WORKS | WORKS |
| Status at z=1, y≥5 | **CORRUPTED** | WORKS |

### Areas Still to Investigate

1. **Why does interface 0 fail but interface 1 work at z=1?**
   - Different DEST offset calculation for face 2 vs face 3?
   - Different register behavior based on interface ID?

2. **What makes y=5 special?**
   - Not an 8-row boundary (that would be y=8)
   - Could be internal counter wrap or carry behavior
   - May be related to how strides interact with counter values

3. **SEC1_REG1 vs SEC1_REG8 behavior**
   - Interface 0 uses SEC1_REG1 for z=1
   - Interface 1 uses SEC1_REG8 for z=1
   - Maybe SEC1_REG1 has different behavior or requires different configuration?

### Next Steps

1. Try using SEC0 registers for all face pairs (reconfigure L1 addresses between face pairs)
2. Investigate if SEC1_REG1 requires different configuration than SEC1_REG8
3. Compare regular llk_pack with untilize=true (which works correctly)
4. Look for interface-specific DEST offset calculations

### Status: Ongoing investigation - interface 0-specific issue at z=1, y≥5
