#include "compute_kernel_api.h"

void MAIN {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    const uint32_t cb_id_in = 0;
    const uint32_t cb_id_out = 16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        cb_wait_front(cb_id_in, 1);
        cb_reserve_back(cb_id_out, 1);

        tile_regs_acquire();
        copy_tile_init();
        copy_tile(cb_id_in, i, 0);  // Copy input tile to dst reg 0 (pass-through)
        tile_regs_commit();
        tile_regs_release();

        pack_tile(0, cb_id_out);  // Pack to output CB
        cb_push_back(cb_id_out, 1);
        cb_pop_front(cb_id_in, 1);
        release_dst();
    }
}
