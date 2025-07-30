#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);     // DRAM input addr from host
    uint32_t num_tiles = get_arg_val<uint32_t>(1);    // Number of tiles to process
    uint32_t struct_size = get_arg_val<uint32_t>(2);  // Size of input structure

    constexpr uint32_t cb_id_in = 0;  // Input CB index
    const uint32_t tile_bytes = get_tile_size(cb_id_in);
    const DataFormat data_format = get_dataformat(cb_id_in);

    DPRINT << "READER: Starting arbitrary data structure pipeline" << ENDL();
    DPRINT << "READER: src_addr=" << src_addr << " num_tiles=" << num_tiles << " struct_size=" << struct_size
           << " bytes" << ENDL();

    // Address generator for interleaved DRAM
    const InterleavedAddrGenFast<true> sgen = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    // Loop: Read tile from DRAM → push to CB
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id_in, 1);  // Reserve 1 tile in CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);

        noc_async_read_tile(i, sgen, l1_write_addr);  // Async NoC read
        noc_async_read_barrier();                     // Sync

        // Debug: Print sample values for pipeline verification
        // Interpret first bytes as float and int for debugging
        float* float_ptr = reinterpret_cast<float*>(l1_write_addr);
        int32_t* int_ptr = reinterpret_cast<int32_t*>(l1_write_addr + 4);
        uint16_t* short_ptr = reinterpret_cast<uint16_t*>(l1_write_addr + 8);
        int8_t* char_ptr = reinterpret_cast<int8_t*>(l1_write_addr + 10);

        DPRINT << "READER: Tile " << i << " input struct[0]: float=" << *float_ptr << " int=" << *int_ptr
               << " short=" << *short_ptr << " char=" << static_cast<int>(*char_ptr) << ENDL();

        cb_push_back(cb_id_in, 1);  // Push to compute kernel
    }
    DPRINT << "READER: Completed - all input structures sent to compute" << ENDL();
}
