#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);  // DRAM output addr from host
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_out = 16;  // Output CB
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    // Address generator for interleaved DRAM
    const InterleavedAddrGenFast<true> dgen = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    // Loop: Pop from CB → write to DRAM
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_out, 1);  // Wait for tile from compute
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_tile(i, dgen, l1_read_addr);  // Async NoC write
        noc_async_write_barrier();                    // Sync
        cb_pop_front(cb_id_out, 1);                   // Clear slot
    }
}
