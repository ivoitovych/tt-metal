#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t input_struct_size = get_arg_val<uint32_t>(1);
    uint32_t output_struct_size = get_arg_val<uint32_t>(2);

    DPRINT_MATH(
        DPRINT << "COMPUTE: Starting with num_tiles=" << num_tiles << " input_size=" << input_struct_size
               << " output_size=" << output_struct_size << ENDL());

    // Define circular buffer indices
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Initialize the copy tile operation
    ckernel::copy_tile_init(cb_in);

    // Process each tile
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Wait for input
        cb_wait_front(cb_in, 1);

        // Reserve output space
        cb_reserve_back(cb_out, 1);

        // Acquire tile registers for processing
        tile_regs_acquire();

        // Copy the input data to the output using the proper function from the header
        ckernel::copy_tile(cb_in, 0, 0);  // Copy from input CB to tile register 0

        // Signal that we're done with computation
        tile_regs_commit();

        // Wait for packer
        tile_regs_wait();

        // Pack the result to output CB
        pack_tile(0, cb_out);

        // Release tile registers
        tile_regs_release();

        DPRINT_MATH(DPRINT << "COMPUTE: Processed tile " << i << " (copied data)" << ENDL());

        // Release buffers
        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }

    DPRINT_MATH(DPRINT << "COMPUTE: Completed all tiles" << ENDL());
}
}  // namespace NAMESPACE
