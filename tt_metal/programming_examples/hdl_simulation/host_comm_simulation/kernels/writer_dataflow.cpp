// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_values = get_arg_val<uint32_t>(2);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id = get_compile_time_arg_val(1);

    uint32_t value_size = sizeof(uint32_t);

    const InterleavedAddrGen<dst_is_dram> dst_gen = {.bank_base_address = dst_addr, .page_size = value_size};

    // Read results from compute kernel via CB and write to host memory
    for (uint32_t i = 0; i < num_values; i++) {
        // Wait for data from compute kernel
        cb_wait_front(cb_id, 1);
        uint32_t read_addr = get_read_ptr(cb_id);

        // Write data from CB to DRAM/L1
        noc_async_write(read_addr, get_noc_addr(start_idx + i, dst_gen), value_size);
        noc_async_write_barrier();

        // Remove data from CB
        cb_pop_front(cb_id, 1);
    }
}
