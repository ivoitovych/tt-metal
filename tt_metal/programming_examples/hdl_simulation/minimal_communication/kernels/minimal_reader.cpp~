// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_values = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id = get_compile_time_arg_val(1);

    // Read data from DRAM to L1
    uint32_t data_size = num_values * sizeof(uint32_t);
    uint64_t src_noc_addr = get_noc_addr(src_addr, 0, src_is_dram);

    noc_async_read(src_noc_addr, l1_addr, data_size);
    noc_async_read_barrier();

    // Push data to circular buffer for compute kernel
    cb_reserve_back(cb_id, num_values);
    uint32_t cb_addr = get_write_ptr(cb_id);

    // Copy from L1 to CB
    uint32_t* l1_ptr = (uint32_t*)l1_addr;
    uint32_t* cb_ptr = (uint32_t*)cb_addr;
    for (uint32_t i = 0; i < num_values; i++) {
        cb_ptr[i] = l1_ptr[i];
    }

    cb_push_back(cb_id, num_values);
}
