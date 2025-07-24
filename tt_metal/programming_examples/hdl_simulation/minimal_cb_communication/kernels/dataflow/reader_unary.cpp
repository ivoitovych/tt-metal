#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);  // DRAM input addr from host
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in = 0;  // Input CB index
    const uint32_t tile_bytes = get_tile_size(cb_id_in);
    const DataFormat data_format = get_dataformat(cb_id_in);

    DPRINT << "READER: Starting computation verification" << ENDL();
    DPRINT << "READER: src_addr=" << src_addr << " num_tiles=" << num_tiles << ENDL();

    // Address generator for interleaved DRAM
    const InterleavedAddrGenFast<true> sgen = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    // Loop: Read tile from DRAM → push to CB
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id_in, 1);  // Reserve 1 tile in CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);

        noc_async_read_tile(i, sgen, l1_write_addr);  // Async NoC read
        noc_async_read_barrier();                     // Sync

        // Debug: Print sample values for computation verification
        uint16_t* data_ptr = (uint16_t*)l1_write_addr;
        DPRINT << "READER: Tile " << i << " input samples: " << BF16(data_ptr[0]) << " " << BF16(data_ptr[1]) << " "
               << BF16(data_ptr[50]) << " (expecting +1.0 in output)" << ENDL();

        cb_push_back(cb_id_in, 1);                    // Push to compute kernel
    }
    DPRINT << "READER: Completed - ready for computation (+1.0)" << ENDL();
}
