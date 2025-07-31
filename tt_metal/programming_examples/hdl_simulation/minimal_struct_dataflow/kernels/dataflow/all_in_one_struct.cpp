#include "dataflow_api.h"
#include "debug/dprint.h"
#include <cstring>

// Define structs in kernel (must match host)
struct InputStruct {
    float float_value;
    int32_t int_value;
    uint16_t short_value;
    int8_t char_value;
    uint8_t padding[5];
};

struct OutputStruct {
    float result_value;
    int32_t status_code;
    uint16_t iteration_count;
    uint8_t flags;
    uint8_t padding[5];
};

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);            // Input DRAM address
    uint32_t dst_addr = get_arg_val<uint32_t>(1);            // Output DRAM address
    uint32_t num_tiles = get_arg_val<uint32_t>(2);           // Number of tiles
    uint32_t input_struct_size = get_arg_val<uint32_t>(3);   // Input struct size
    uint32_t output_struct_size = get_arg_val<uint32_t>(4);  // Output struct size
    uint32_t tile_bytes = get_arg_val<uint32_t>(5);          // Bytes per tile

    DPRINT << "ALL-IN-ONE: Starting pipeline with " << num_tiles << " tiles, tile_bytes=" << tile_bytes << ENDL();
    DPRINT << "ALL-IN-ONE: src_addr=0x" << src_addr << " dst_addr=0x" << dst_addr << ENDL();
    DPRINT << "ALL-IN-ONE: input_size=" << input_struct_size << " output_size=" << output_struct_size << ENDL();

    // Calculate addresses and formats for interleaved DRAM access
    const uint32_t page_size = tile_bytes;
    const DataFormat data_format = DataFormat::UInt8;

    // Address generators for input and output
    const InterleavedAddrGenFast<true> input_gen = {
        .bank_base_address = src_addr, .page_size = page_size, .data_format = data_format};

    const InterleavedAddrGenFast<true> output_gen = {
        .bank_base_address = dst_addr, .page_size = page_size, .data_format = data_format};

    // L1 temporary buffer for processing (use fixed L1 address)
    constexpr uint32_t l1_buffer_addr = 0x10000;  // Fixed L1 address for buffer

    uint32_t num_structs_per_tile = tile_bytes / input_struct_size;

    DPRINT << "ALL-IN-ONE: Processing " << num_structs_per_tile << " structs per tile" << ENDL();

    // Process each tile
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        DPRINT << "ALL-IN-ONE: Processing tile " << tile_idx << ENDL();

        // Step 1: Read input tile from DRAM to L1
        noc_async_read_tile(tile_idx, input_gen, l1_buffer_addr);
        noc_async_read_barrier();

        // Step 2: Process data in L1 (transformation)
        volatile tt_l1_ptr InputStruct* input_structs =
            reinterpret_cast<volatile tt_l1_ptr InputStruct*>(l1_buffer_addr);
        volatile tt_l1_ptr OutputStruct* output_structs =
            reinterpret_cast<volatile tt_l1_ptr OutputStruct*>(l1_buffer_addr);  // Reuse same buffer

        // Transform each struct
        for (uint32_t s = 0; s < num_structs_per_tile; s++) {
            // Read input values
            float in_float = input_structs[s].float_value;
            int32_t in_int = input_structs[s].int_value;
            uint16_t in_short = input_structs[s].short_value;
            int8_t in_char = input_structs[s].char_value;

            // Apply transformation (overwrite in place)
            output_structs[s].result_value = in_float * 2.0f;
            output_structs[s].status_code = in_int;
            output_structs[s].iteration_count = in_short % 100;
            output_structs[s].flags = static_cast<uint8_t>(in_char & 0xFF);

            // Clear padding
            for (int p = 0; p < 5; p++) {
                output_structs[s].padding[p] = 0;
            }
        }

        // Debug: Print sample results
        if (num_structs_per_tile > 0) {
            DPRINT << "ALL-IN-ONE: Tile " << tile_idx << " sample[0]: " << input_structs[0].float_value << " -> "
                   << output_structs[0].result_value << ENDL();
        }

        // Step 3: Write transformed data from L1 to output DRAM
        noc_async_write_tile(tile_idx, output_gen, l1_buffer_addr);
        noc_async_write_barrier();
    }

    DPRINT << "ALL-IN-ONE: Completed processing all " << num_tiles << " tiles" << ENDL();
}
