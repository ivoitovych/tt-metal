#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    DPRINT_MATH(DPRINT << "COMPUTE: Starting with num_tiles=" << num_tiles << ENDL());

    // Initialize with both input and output circular buffer indices
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();

        // Wait for input
        cb_wait_front(tt::CBIndex::c_0, 1);

        // Reserve output
        cb_reserve_back(tt::CBIndex::c_16, 1);

        // Copy tile (pass-through for now - we'll add computation later)
        copy_tile_to_dst_init_short(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0);
        pack_tile(0, tt::CBIndex::c_16);

        DPRINT_MATH(DPRINT << "COMPUTE: Processed tile " << i << " (pass-through)" << ENDL());

        // Release buffers
        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_16, 1);

        release_dst();
    }
    DPRINT_MATH(DPRINT << "COMPUTE: Completed all tiles" << ENDL());
}
}  // namespace NAMESPACE
