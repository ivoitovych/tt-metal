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
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t input_struct_size = get_arg_val<uint32_t>(3);
    uint32_t output_struct_size = get_arg_val<uint32_t>(4);
    uint32_t tile_bytes = get_arg_val<uint32_t>(5);

    DPRINT << "ALL-IN-ONE: Starting pipeline with " << num_tiles << " tiles, tile_bytes=" << tile_bytes << ENDL();

    // Use separate L1 buffers for input and output
    constexpr uint32_t l1_input_buffer_addr = 0x10000;
    constexpr uint32_t l1_output_buffer_addr = 0x20000;

    uint32_t num_structs_per_tile = tile_bytes / input_struct_size;

    const DataFormat data_format = DataFormat::UInt8;

    // Create interleaved address generators with correct page size
    const InterleavedAddrGenFast<true> input_gen = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,  // Each tile is one page
        .data_format = data_format};

    const InterleavedAddrGenFast<true> output_gen = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,  // Each tile is one page
        .data_format = data_format};

    // Process each tile
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        DPRINT << "ALL-IN-ONE: Processing tile " << tile_idx << ENDL();

        // Step 1: Read entire tile from DRAM to L1
        // For interleaved buffers, we need to use the proper NOC address calculation
        uint64_t src_noc_addr = get_noc_addr(tile_idx, input_gen);
        noc_async_read_tile(tile_idx, input_gen, l1_input_buffer_addr);
        noc_async_read_barrier();

        // Step 2: Process data (transformation)
        volatile tt_l1_ptr InputStruct* input_structs =
            reinterpret_cast<volatile tt_l1_ptr InputStruct*>(l1_input_buffer_addr);
        volatile tt_l1_ptr OutputStruct* output_structs =
            reinterpret_cast<volatile tt_l1_ptr OutputStruct*>(l1_output_buffer_addr);

        // Transform each struct
        for (uint32_t s = 0; s < num_structs_per_tile; s++) {
            // Apply transformation
            output_structs[s].result_value = input_structs[s].float_value * 2.0f;
            output_structs[s].status_code = input_structs[s].int_value;
            output_structs[s].iteration_count = input_structs[s].short_value % 100;
            output_structs[s].flags = static_cast<uint8_t>(input_structs[s].char_value & 0xFF);

            // Clear padding
            for (int p = 0; p < 5; p++) {
                output_structs[s].padding[p] = 0;
            }
        }

        // Debug: Print first few structs with actual values
        DPRINT << "Tile " << tile_idx << " samples:" << ENDL();
        for (uint32_t i = 0; i < 3 && i < num_structs_per_tile; i++) {
            DPRINT << "  [" << i << "]: float=" << input_structs[i].float_value << " int=" << input_structs[i].int_value
                   << " -> result=" << output_structs[i].result_value << " status=" << output_structs[i].status_code
                   << ENDL();
        }

        // Step 3: Write entire tile from L1 to output DRAM
        noc_async_write_tile(l1_output_buffer_addr, output_gen, tile_idx);
        noc_async_write_barrier();
    }

    DPRINT << "ALL-IN-ONE: Completed processing all " << num_tiles << " tiles" << ENDL();
}
