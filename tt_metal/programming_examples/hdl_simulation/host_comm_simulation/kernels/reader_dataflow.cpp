// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_values = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id = get_compile_time_arg_val(1);

    uint32_t value_size = sizeof(uint32_t);

    const InterleavedAddrGen<src_is_dram> src_gen = {.bank_base_address = src_addr, .page_size = value_size};

    // Read simulation commands/data from host and forward to compute via CB
    for (uint32_t i = 0; i < num_values; i++) {
        // Reserve space in circular buffer
        cb_reserve_back(cb_id, 1);
        uint32_t write_addr = get_write_ptr(cb_id);

        // Read data from DRAM/L1 and write to CB
        noc_async_read(get_noc_addr(start_idx + i, src_gen), write_addr, value_size);
        noc_async_read_barrier();

        // Push data to compute kernel
        cb_push_back(cb_id, 1);
    }
}
