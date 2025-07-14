// kernels/simple_communication_reader.cpp
#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t block_size_bytes = get_arg_val<uint32_t>(2);
    uint32_t total_bytes = get_arg_val<uint32_t>(3);
    uint32_t padding_bytes = get_arg_val<uint32_t>(4);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_0;

    const uint32_t tile_size_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGen<src_is_dram> s = {.bank_base_address = src_addr, .page_size = block_size_bytes};

    uint32_t curr_offset = 0;

    for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        cb_reserve_back(cb_id_out, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_out);

        // Read actual data
        if (curr_offset < total_bytes) {
            uint32_t bytes_to_read =
                (curr_offset + block_size_bytes <= total_bytes) ? block_size_bytes : (total_bytes - curr_offset);

            uint64_t src_noc_addr = get_noc_addr(block_idx, s);
            noc_async_read(src_noc_addr, l1_write_addr, bytes_to_read);

            // Pad remaining bytes if necessary
            if (bytes_to_read < tile_size_bytes) {
                // Clear padding area
                volatile uint32_t* pad_ptr = (volatile uint32_t*)(l1_write_addr + bytes_to_read);
                uint32_t pad_words = (tile_size_bytes - bytes_to_read) / sizeof(uint32_t);
                for (uint32_t i = 0; i < pad_words; i++) {
                    pad_ptr[i] = 0;
                }
            }

            curr_offset += bytes_to_read;
        }

        noc_async_read_barrier();
        cb_push_back(cb_id_out, 1);
    }
}
