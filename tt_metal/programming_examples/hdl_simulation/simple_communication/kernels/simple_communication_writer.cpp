// kernels/simple_communication_writer.cpp
#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t block_size_bytes = get_arg_val<uint32_t>(2);
    uint32_t checksum_addr = get_arg_val<uint32_t>(3);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_in = tt::CBIndex::c_16;

    const uint32_t tile_size_bytes = get_tile_size(cb_id_in);

    const InterleavedAddrGen<dst_is_dram> d = {.bank_base_address = dst_addr, .page_size = block_size_bytes};

    uint32_t running_checksum = 0;

    for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        cb_wait_front(cb_id_in, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in);

        // Calculate checksum for verification
        volatile uint32_t* data_ptr = (volatile uint32_t*)l1_read_addr;
        uint32_t words_in_block = block_size_bytes / sizeof(uint32_t);
        for (uint32_t i = 0; i < words_in_block; i++) {
            running_checksum ^= data_ptr[i];
        }

        // Write block to destination
        uint64_t dst_noc_addr = get_noc_addr(block_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, block_size_bytes);

        cb_pop_front(cb_id_in, 1);
    }

    noc_async_write_barrier();

    // Write final checksum for verification
    const InterleavedAddrGen<dst_is_dram> checksum_gen = {
        .bank_base_address = checksum_addr, .page_size = sizeof(uint32_t)};
    uint64_t checksum_noc_addr = get_noc_addr(0, checksum_gen);
    noc_async_write((uint32_t)&running_checksum, checksum_noc_addr, sizeof(uint32_t));
    noc_async_write_barrier();
}
