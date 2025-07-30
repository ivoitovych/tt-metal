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

    // Process each tile
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Wait for input
        cb_wait_front(tt::CBIndex::c_0, 1);

        // Reserve output
        cb_reserve_back(tt::CBIndex::c_16, 1);

        // Simple copy operation - just copy the entire tile
        acquire_dst();
        copy_tile_to_dst_init_short(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0);
        pack_tile(0, tt::CBIndex::c_16);
        release_dst();

        DPRINT_MATH(DPRINT << "COMPUTE: Processed tile " << i << " (pass-through)" << ENDL());

        // Release buffers
        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_16, 1);
    }

    DPRINT_MATH(DPRINT << "COMPUTE: Completed all tiles" << ENDL());
}
}  // namespace NAMESPACE
