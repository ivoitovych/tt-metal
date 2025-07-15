// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Get runtime arguments
    uint32_t src_addr = get_arg_val<uint32_t>(0);      // Input buffer address
    uint32_t dst_addr = get_arg_val<uint32_t>(1);      // Output buffer address
    uint32_t start_idx = get_arg_val<uint32_t>(2);     // Starting index
    uint32_t num_commands = get_arg_val<uint32_t>(3);  // Number of commands

    // Get compile-time arguments
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t max_commands = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);

    DPRINT << "DEBUG Kernel Started" << ENDL();
    DPRINT << "src_addr=" << HEX() << src_addr << " dst_addr=" << dst_addr << DEC() << ENDL();
    DPRINT << "page_size=" << page_size << " src_is_dram=" << src_is_dram << ENDL();
    DPRINT << "start_idx=" << start_idx << " num_commands=" << num_commands << ENDL();

    // Let's try multiple addressing approaches and see what works

    // Method 1: Direct address read (what we've been trying)
    DPRINT << "=== Method 1: Direct addressing ===" << ENDL();
    uint32_t test_data[4];
    noc_async_read(src_addr, reinterpret_cast<uint32_t>(&test_data[0]), 16);
    noc_async_read_barrier();
    DPRINT << "Direct read: " << HEX() << test_data[0] << " " << test_data[1] << " " << test_data[2] << " "
           << test_data[3] << DEC() << ENDL();

    // Method 2: Try InterleavedAddrGen with large page
    DPRINT << "=== Method 2: InterleavedAddrGen large page ===" << ENDL();
    const InterleavedAddrGen<src_is_dram> src_gen_large = {.bank_base_address = src_addr, .page_size = page_size};

    noc_async_read(get_noc_addr(0, src_gen_large), reinterpret_cast<uint32_t>(&test_data[0]), 16);
    noc_async_read_barrier();
    DPRINT << "Large page read: " << HEX() << test_data[0] << " " << test_data[1] << " " << test_data[2] << " "
           << test_data[3] << DEC() << ENDL();

    // Method 3: Try InterleavedAddrGen with small page
    DPRINT << "=== Method 3: InterleavedAddrGen small page ===" << ENDL();
    const InterleavedAddrGen<src_is_dram> src_gen_small = {.bank_base_address = src_addr, .page_size = 4};

    // Read first 4 words using individual page access
    for (uint32_t i = 0; i < 4; i++) {
        noc_async_read(get_noc_addr(i, src_gen_small), reinterpret_cast<uint32_t>(&test_data[i]), 4);
    }
    noc_async_read_barrier();
    DPRINT << "Small page read: " << HEX() << test_data[0] << " " << test_data[1] << " " << test_data[2] << " "
           << test_data[3] << DEC() << ENDL();

    // Method 4: Try with different offsets from base address
    DPRINT << "=== Method 4: Offset testing ===" << ENDL();
    for (uint32_t offset = 0; offset < 64; offset += 16) {
        noc_async_read(src_addr + offset, reinterpret_cast<uint32_t>(&test_data[0]), 4);
        noc_async_read_barrier();
        if (test_data[0] == 0) {  // Look for our first command (cmd=0)
            DPRINT << "Found cmd=0 at offset " << offset << ": " << HEX() << test_data[0] << DEC() << ENDL();
            break;
        }
    }

    // Let's write a simple test pattern to output to verify write works
    DPRINT << "=== Testing write capability ===" << ENDL();
    uint32_t write_test[4] = {0x12345678, 0x9ABCDEF0, 0xDEADBEEF, 0xCAFEBABE};

    // Try direct write
    noc_async_write(reinterpret_cast<uint32_t>(&write_test[0]), dst_addr, 16);
    noc_async_write_barrier();
    DPRINT << "Wrote test pattern to output buffer" << ENDL();

    DPRINT << "Debug kernel complete!" << ENDL();
}
