// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t start_idx = get_arg_val<uint32_t>(2);
    uint32_t num_values = get_arg_val<uint32_t>(3);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id = tt::CBIndex::c_0;

    uint32_t value_size = sizeof(uint32_t);

    const InterleavedAddrGen<src_is_dram> src_gen = {.bank_base_address = src_addr, .page_size = value_size};

    const InterleavedAddrGen<src_is_dram> dst_gen = {.bank_base_address = dst_addr, .page_size = value_size};

    // Read simulation commands/data and forward to compute
    for (uint32_t i = 0; i < num_values; i++) {
        cb_reserve_back(cb_id, 1);
        uint32_t write_addr = get_write_ptr(cb_id);

        noc_async_read(get_noc_addr(start_idx + i, src_gen), write_addr, value_size);
        noc_async_read_barrier();

        cb_push_back(cb_id, 1);
    }

    // Read back results from compute and write to output
    for (uint32_t i = 0; i < num_values; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t read_addr = get_read_ptr(cb_id);

        noc_async_write(read_addr, get_noc_addr(i, dst_gen), value_size);
        noc_async_write_barrier();

        cb_pop_front(cb_id, 1);
    }
}
