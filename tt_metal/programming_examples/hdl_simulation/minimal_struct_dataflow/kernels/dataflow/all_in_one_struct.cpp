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
    DPRINT << "ALL-IN-ONE: Runtime args - src_addr=0x" << src_addr << " dst_addr=0x" << dst_addr << ENDL();
    DPRINT << "ALL-IN-ONE: input_size=" << input_struct_size << " output_size=" << output_struct_size << ENDL();

    // Use separate L1 buffers for input and output to avoid overlap
    constexpr uint32_t l1_input_buffer_addr = 0x10000;   // 64KB offset
    constexpr uint32_t l1_output_buffer_addr = 0x20000;  // 128KB offset

    uint32_t num_structs_per_tile = tile_bytes / input_struct_size;

    DPRINT << "ALL-IN-ONE: Processing " << num_structs_per_tile << " structs per tile" << ENDL();

    // Process each tile
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        DPRINT << "ALL-IN-ONE: ===== Processing tile " << tile_idx << " =====" << ENDL();

        // Step 1: Calculate DRAM addresses correctly
        // The key insight: use the InterleavedAddrGen for proper addressing
        constexpr uint32_t data_format = static_cast<uint32_t>(DataFormat::UInt8);

        // For interleaved buffers, we need to use the tile/page-based addressing
        // Each tile is a separate "page" in the interleaved buffer
        const InterleavedAddrGenFast<true> src_gen = {
            .bank_base_address = src_addr,
            .page_size = tile_bytes,
            .data_format = static_cast<DataFormat>(data_format)};

        const InterleavedAddrGenFast<true> dst_gen = {
            .bank_base_address = dst_addr,
            .page_size = tile_bytes,
            .data_format = static_cast<DataFormat>(data_format)};

        DPRINT << "ALL-IN-ONE: Tile " << tile_idx << " using InterleavedAddrGen:" << ENDL();
        DPRINT << "  src_gen: bank_base=0x" << src_addr << " page_size=" << tile_bytes << ENDL();
        DPRINT << "  dst_gen: bank_base=0x" << dst_addr << " page_size=" << tile_bytes << ENDL();

        // Step 2: Read input tile from DRAM to L1 using proper tile addressing
        DPRINT << "ALL-IN-ONE: Reading tile " << tile_idx << " from DRAM..." << ENDL();
        noc_async_read_tile(tile_idx, src_gen, l1_input_buffer_addr);
        noc_async_read_barrier();
        DPRINT << "ALL-IN-ONE: Read complete." << ENDL();

        // Step 3: Debug - examine raw input data
        volatile tt_l1_ptr InputStruct* input_structs =
            reinterpret_cast<volatile tt_l1_ptr InputStruct*>(l1_input_buffer_addr);

        DPRINT << "ALL-IN-ONE: Raw input data from tile " << tile_idx << ":" << ENDL();
        for (uint32_t i = 0; i < 5 && i < num_structs_per_tile; i++) {
            DPRINT << "  Raw[" << i << "]: float=" << input_structs[i].float_value
                   << " int=" << input_structs[i].int_value << " short=" << input_structs[i].short_value
                   << " char=" << static_cast<int>(input_structs[i].char_value) << ENDL();
        }

        // Step 4: Process data (transformation)
        volatile tt_l1_ptr OutputStruct* output_structs =
            reinterpret_cast<volatile tt_l1_ptr OutputStruct*>(l1_output_buffer_addr);

        DPRINT << "ALL-IN-ONE: Processing " << num_structs_per_tile << " structures..." << ENDL();

        // Transform each struct
        for (uint32_t s = 0; s < num_structs_per_tile; s++) {
            // Read input values (use volatile to ensure proper reads)
            volatile float in_float = input_structs[s].float_value;
            volatile int32_t in_int = input_structs[s].int_value;
            volatile uint16_t in_short = input_structs[s].short_value;
            volatile int8_t in_char = input_structs[s].char_value;

            // Apply transformation
            output_structs[s].result_value = in_float * 2.0f;
            output_structs[s].status_code = in_int;
            output_structs[s].iteration_count = in_short % 100;
            output_structs[s].flags = static_cast<uint8_t>(in_char & 0xFF);

            // Clear padding explicitly
            output_structs[s].padding[0] = 0;
            output_structs[s].padding[1] = 0;
            output_structs[s].padding[2] = 0;
            output_structs[s].padding[3] = 0;
            output_structs[s].padding[4] = 0;
        }

        // Step 5: Debug - examine processed output data
        DPRINT << "ALL-IN-ONE: Processed output data for tile " << tile_idx << ":" << ENDL();
        for (uint32_t i = 0; i < 5 && i < num_structs_per_tile; i++) {
            DPRINT << "  Out[" << i << "]: result=" << output_structs[i].result_value
                   << " status=" << output_structs[i].status_code << " iterations=" << output_structs[i].iteration_count
                   << " flags=" << static_cast<int>(output_structs[i].flags) << ENDL();
        }

        // Step 6: Write transformed data from L1 to output DRAM using proper tile addressing
        DPRINT << "ALL-IN-ONE: Writing tile " << tile_idx << " to DRAM..." << ENDL();
        noc_async_write_tile(tile_idx, dst_gen, l1_output_buffer_addr);
        noc_async_write_barrier();
        DPRINT << "ALL-IN-ONE: Write complete." << ENDL();

        DPRINT << "ALL-IN-ONE: ===== Tile " << tile_idx << " complete =====" << ENDL();
    }

    DPRINT << "ALL-IN-ONE: ===== PIPELINE COMPLETE =====" << ENDL();
    DPRINT << "ALL-IN-ONE: Processed " << num_tiles << " tiles successfully" << ENDL();
}
