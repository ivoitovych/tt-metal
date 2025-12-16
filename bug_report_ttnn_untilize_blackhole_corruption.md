# Bug Report: ttnn::untilize Data Corruption on Blackhole P150

**Status**: OPEN - Under Investigation
**Severity**: Critical
**Component**: tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h
**Hardware**: Blackhole P150
**Branch**: `ivoitovych/tt-train-untilize-blackhole-bug-2`
**Date**: 2025-12-16

---

## Executive Summary

The `ttnn::untilize` operation produces corrupted output data when converting tensors from TILE layout to ROW_MAJOR layout on Blackhole P150 hardware. The corruption manifests as a progressive "read even, skip odd" pattern starting at row 3, with severity increasing for later rows. This bug does **not** occur on Wormhole hardware.

Multiple fix attempts targeting different aspects of the LLK packer implementation have been unsuccessful. A critical finding is that **both DST_ACCESS_STRIDED_MODE and DST_ACCESS_NORMAL_MODE produce identical corruption patterns**, suggesting the root cause lies elsewhere in the packer configuration or hardware.

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
| Hardware | Blackhole P150 |
| Host OS | Linux 5.15.0-164-generic |
| Machine | vm3 |
| Firmware | 19.3.0 |
| ETH FW | 1.7.1 |
| tt-metal branch | `ivoitovych/tt-train-untilize-blackhole-bug-2` |

---

## Corruption Pattern Analysis

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

### Immediate Actions

1. **Implement `program_packer_untilized_destination` for Blackhole**
   - Uncomment and adapt the code in `cpack_common.h`
   - Verify register names/offsets for Blackhole architecture
   - Test with both STRIDED and NORMAL modes

2. **Align Address Modifier Configuration**
   - Configure ADDR_MOD_0, ADDR_MOD_1, ADDR_MOD_2 similar to Wormhole
   - Pay attention to y_src increment values for face transitions

3. **Review MOP Loop Structure**
   - Consider switching to Wormhole-style explicit row iteration
   - May require significant refactoring of the MOP template

### Escalation Path

If the above fixes don't resolve the issue:

1. **File Bug with LLK Team**
   - The tt_llk repository is a git submodule
   - LLK team has deeper knowledge of packer hardware

2. **Hardware Team Consultation**
   - Corruption pattern suggests possible HW bug
   - May need firmware update or hardware workaround

3. **Alternative Implementation**
   - Consider using a different untilize path (e.g., data movement instead of packer)
   - May have performance implications

---

## Appendix: Code References

### Key Files

| File | Purpose |
|------|---------|
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h` | Main Blackhole implementation |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_pack_untilize.h` | Working Wormhole reference |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h` | Packer common functions |
| `tt_metal/include/compute_kernel_api/pack_untilize.h` | API layer |
| `tt-train/tests/ttnn_fixed/debug_untilize_test.cpp` | Debug test suite |

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

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-12-15 | Investigation Team | Initial discovery and Phase 1-2 analysis |
| 2025-12-16 | Investigation Team | Fix attempts 4-6, comprehensive documentation |
