// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_values = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    // Wait for compute kernel to finish
    cb_wait_front(cb_id, num_values);

    // Get processed data from circular buffer
    uint32_t cb_addr = get_read_ptr(cb_id);
    uint32_t* cb_ptr = (uint32_t*)cb_addr;
    uint32_t* l1_ptr = (uint32_t*)l1_addr;

    // Copy from CB to L1
    for (uint32_t i = 0; i < num_values; i++) {
        l1_ptr[i] = cb_ptr[i];
    }

    cb_pop_front(cb_id, num_values);

    // Write data from L1 to DRAM
    uint32_t data_size = num_values * sizeof(uint32_t);
    uint64_t dst_noc_addr = get_noc_addr(dst_addr, 0, dst_is_dram);

    noc_async_write(l1_addr, dst_noc_addr, data_size);
    noc_async_write_barrier();
}
