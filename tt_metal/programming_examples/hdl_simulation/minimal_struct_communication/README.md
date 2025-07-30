### Project Purpose and Current Implementation

The example demonstrates end-to-end communication of arbitrary (non-tensor) data structures between the host and a Tenstorrent device kernel pipeline. Specifically, it sends `InputStruct` instances (16 bytes each: float, int32_t, uint16_t, int8_t, padded) from host to device DRAM, processes them through a reader-compute-writer pipeline on a single core (0,0), and returns `OutputStruct` instances (similar layout, interpreted as float result, int32_t status, uint16_t iterations, uint8_t flags, padded) back to host for verification.

The pipeline treats data as raw bytes (`DataFormat::UInt8` in CBs) to handle arbitrary formats, avoiding tensor-specific assumptions. Data is divided into "tiles" (blocks of 1024 structs = 16 KB each), with 2 tiles total (2048 structs, 32 KB). The reader transfers DRAM to L1 CB, compute "processes" (currently attempts a copy), and writer transfers L1 CB to DRAM.

The code is well-structured:
- **Host (minimal_struct_communication.cpp)**: Sets up device, program, interleaved DRAM buffers, CBs (page size 16 KB, 2 pages each for double-buffering), kernels, args, and launches. Generates patterned input, verifies output against an expected transformation (e.g., result = float * 2, status = int, iterations = short % 100, flags = char & 0xFF), reports detailed stats.
- **Reader (reader_struct.cpp)**: Transfers DRAM pages (tiles) to input CB via NOC async reads, prints sample struct[0] for debug.
- **Writer (writer_struct.cpp)**: Transfers output CB pages to DRAM via NOC async writes, prints sample struct[0].
- **Compute (process_struct.cpp)**: Attempts to copy input CB pages to output CB using tile copy APIs, but fails (outputs zeros).

DPRINT logs confirm reader sees correct input, compute "copies" tiles, writer sees zeros, and host verification detects 100% corruption.

### Assessment of Remaining Issues

The pipeline initializes and executes without crashes, transferring data correctly in reader/writer, but data is lost in compute due to mismatches in handling raw byte data with tensor-oriented APIs. Key issues:

1. **Compute Kernel Fails to Propagate Data (Critical)**:
   - The kernel uses `ckernel::copy_tile_init` and `ckernel::copy_tile` (from `tile_move_copy.h`), designed for tensor tiles in floating-point formats (e.g., BFloat16, Float32). These assume standard tile sizes (e.g., 2048 bytes for BFloat16) and unpack/pack data into tile registers for math ops.
   - With `DataFormat::UInt8`, hardware tile size is 32x32x1 byte = 1024 bytes. But CB page size is 16 KB (1024 structs x 16 bytes), so each page holds 16 hardware tiles.
   - The kernel processes only the first hardware tile (1024 bytes, ~64 structs) per page via one `copy_tile(0)` call, leaving the remaining 15 KB unprocessed (likely zeroed or garbage in output CB).
   - Even the first tile's data is zeroed in output, likely because `UInt8` is unsupported or partially supported for unpacker/packer in copy_tile (API docs indicate support for UInt8 is limited; preferred for raw data is direct memory access).
   - No actual transformation is implemented; it's a failed copy, but verification expects transformation, guaranteeing failure even if copy succeeded.

2. **Struct Handling and Alignment**:
   - Structs are packed (16 bytes), but reinterpret_cast in kernels assumes little-endian and aligned memory, which is fine on RISC-V, but debug prints show correct input in reader, zeros in writer.
   - Padding is zeroed on host, but not verified; potential for padding corruption if byte copies misalign.

3. **Multi-Tile/Page Mismatch**:
   - Reader/writer correctly handle large pages (16 KB transfers via `noc_async_read_tile`/`write_tile` with `page_size=16 KB`).
   - Compute must process full pages but uses single-tile logic, causing partial copies.

4. **Verification Mismatch**:
   - Expects transformation, but kernel does copy. Samples show mismatches (e.g., input -5.0/0/0/0 vs. output 0/0/0/0x00, but expected would be -10.0/0/0/0x00 if transformed).
   - Edge cases (e.g., zero test) fail due to zeros.
   - High "errors" (e.g., max 2.12e20) from interpreting zeros as floats/ints.

5. **HDL Simulation Integration (Branch Goal, Incomplete)**:
   - Branch `hdl_simulation_structure` aims to enable HDL simulation (e.g., Verilator for Tensix cores/NoC). Current code runs on hardware only; no simulation hooks, testbenches, or configs.
   - Remaining: Add HDL models (from tt-metal/hw?), simulation build targets, switch (e.g., via env var) between hardware/sim.

6. **Minor/Performance Issues**:
   - NUMA/hugepage warnings: Non-critical, but bind to device NUMA for better Host↔Device perf.
   - No program cache: Enable for repeats.
   - Fixed single core; scalable to multi-core for larger data.
   - No error handling for CB overflows/underflows.

The project is ~70% complete: Setup, data transfer, and debug work; compute and verification need fixes. Task is "deadly hard" due to low-level API subtleties for non-tensor data.

### Recommendations and Fixes

To resolve, modify compute to handle raw bytes/structs directly (avoid tensor APIs for arbitrary data). Implement the transformation. Test incrementally (e.g., num_tiles=1, small elements_per_tile).

- **Fix Compute Kernel (process_struct.cpp)**:
  - Pass tile_bytes as arg: In host, `SetRuntimeArgs(..., {num_tiles, sizeof(InputStruct), sizeof(OutputStruct), elements_per_tile * sizeof(InputStruct)});`
  - In kernel:
    ```
    #include "compute_kernel_api/common_globals.h"  // For get_read_ptr, etc.

    namespace NAMESPACE {
    void MAIN {
        uint32_t num_tiles = get_arg_val<uint32_t>(0);
        uint32_t input_struct_size = get_arg_val<uint32_t>(1);
        uint32_t output_struct_size = get_arg_val<uint32_t>(2);
        uint32_t tile_bytes = get_arg_val<uint32_t>(3);

        constexpr auto cb_in = tt::CB::c_in0;
        constexpr auto cb_out = tt::CB::c_out16;

        DPRINT_MATH(...);  // Keep debug

        for (uint32_t i = 0; i < num_tiles; i++) {
            cb_wait_front(cb_in, 1);
            cb_reserve_back(cb_out, 1);

            volatile tt_l1_ptr uint8_t* in_ptr = (volatile tt_l1_ptr uint8_t*) get_read_ptr(cb_in);
            volatile tt_l1_ptr uint8_t* out_ptr = (volatile tt_l1_ptr uint8_t*) get_write_ptr(cb_out);

            // Process as structs for transformation
            uint32_t num_structs = tile_bytes / input_struct_size;
            InputStruct* in_s = reinterpret_cast<InputStruct*>(in_ptr);
            OutputStruct* out_s = reinterpret_cast<OutputStruct*>(out_ptr);

            for (uint32_t s = 0; s < num_structs; s++) {
                out_s[s].result_value = in_s[s].float_value * 2.0f;
                out_s[s].status_code = in_s[s].int_value;
                out_s[s].iteration_count = in_s[s].short_value % 100;
                out_s[s].flags = in_s[s].char_value & 0xFF;
                std::memset(out_s[s].padding, 0, sizeof(out_s[s].padding));
            }

            DPRINT_MATH(DPRINT << "COMPUTE: Processed tile " << i << " (transformed data)" << ENDL());

            cb_pop_front(cb_in, 1);
            cb_push_back(cb_out, 1);
        }

        DPRINT_MATH(DPRINT << "COMPUTE: Completed all tiles" << ENDL());
    }
    }  // namespace NAMESPACE
    ```
  - This copies/transforms byte-by-byte or struct-by-struct, works for any format/size, avoids tile APIs. For pure copy (if needed), use a byte loop instead of struct loop.

- **Update Host**:
  - Add tile_bytes arg to compute SetRuntimeArgs.
  - If keeping copy for testing, adjust verification to expect copy (remove *2, %100, &0xFF).

- **Debug Tips**:
  - Run with smaller data (num_tiles=1, elements_per_tile=64) to isolate.
  - Add DPRINT in compute to print sample input/output structs (like reader/writer).
  - Export TT_METAL_DPRINT_CORES=0,0 for logs.

- **For HDL Simulation**:
  - Integrate Verilator: Add build scripts to compile HDL (from tt-metal/hw/) + kernels into sim executable.
  - Use tt-metal's simulation mode (if available in v6.0.0; check docs/repo).
  - Add conditional (e.g., #ifdef SIM) for hardware vs. sim addresses/APIs.

- **Enhancements**:
  - Scale: Shard across cores for larger data.
  - Optimize: Use NOC multicasts if multi-core.
  - Docs: Add README with build/run/sim instructions.

With these fixes, the example should pass verification. If issues persist, share updated logs/output.
