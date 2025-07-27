#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    DPRINT_MATH(DPRINT << "COMPUTE: Starting REAL computation with num_tiles=" << num_tiles << ENDL());

    // Initialize for unary operations (we'll add 1.0 to each element)
    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    add_bcast_scalar_init(DSTMODE::HALF);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst(tt::DstMode::Half);

        // Wait for input tile
        cb_wait_front(tt::CBIndex::c_0, 1);

        // Reserve space for output tile
        cb_reserve_back(tt::CBIndex::c_16, 1);

        // Unpack input tile into destination register
        unpack_reconfig_data_format(tt::CBIndex::c_0, tt::CBIndex::c_16);
        copy_tile_to_dst_init_short(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        // REAL COMPUTATION: Add 1.0 to every element in the tile
        // This uses the SFPU (Special Function Processing Unit) to add a scalar to all elements
        add_bcast_scalar(0, 1.0f);

        // Pack the result to output circular buffer
        pack_tile(0, tt::CBIndex::c_16);

        DPRINT_MATH(DPRINT << "COMPUTE: Added 1.0 to tile " << i << " (REAL COMPUTATION)" << ENDL());

        // Release resources
        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_16, 1);

        release_dst(tt::DstMode::Half);
    }
    DPRINT_MATH(DPRINT << "COMPUTE: Completed REAL computation on all tiles" << ENDL());
}
}  // namespace NAMESPACE
