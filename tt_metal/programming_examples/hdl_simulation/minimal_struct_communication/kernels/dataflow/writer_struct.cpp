#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);     // DRAM output addr from host
    uint32_t num_tiles = get_arg_val<uint32_t>(1);    // Number of tiles to process
    uint32_t struct_size = get_arg_val<uint32_t>(2);  // Size of output structure

    constexpr uint32_t cb_id_out = 16;  // Output CB
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    DPRINT << "WRITER: Starting arbitrary data structure result writing" << ENDL();
    DPRINT << "WRITER: dst_addr=" << dst_addr << " num_tiles=" << num_tiles << " struct_size=" << struct_size
           << " bytes" << ENDL();

    // Address generator for interleaved DRAM
    const InterleavedAddrGenFast<true> dgen = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    // Loop: Pop from CB → write to DRAM
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_out, 1);  // Wait for tile from compute
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        // Debug: Print output structure values to verify pipeline integrity
        // Interpret first bytes as float and int for debugging
        float* result_ptr = reinterpret_cast<float*>(l1_read_addr);
        int32_t* status_ptr = reinterpret_cast<int32_t*>(l1_read_addr + 4);
        uint16_t* iter_ptr = reinterpret_cast<uint16_t*>(l1_read_addr + 8);
        uint8_t* flags_ptr = reinterpret_cast<uint8_t*>(l1_read_addr + 10);

        DPRINT << "WRITER: Tile " << i << " output struct[0]: result=" << *result_ptr << " status=" << *status_ptr
               << " iterations=" << *iter_ptr << " flags=0x" << static_cast<int>(*flags_ptr) << ENDL();

        noc_async_write_tile(i, dgen, l1_read_addr);  // Async NoC write
        noc_async_write_barrier();                    // Sync
        cb_pop_front(cb_id_out, 1);                   // Clear slot
    }
    DPRINT << "WRITER: Output structures written to DRAM" << ENDL();
}
