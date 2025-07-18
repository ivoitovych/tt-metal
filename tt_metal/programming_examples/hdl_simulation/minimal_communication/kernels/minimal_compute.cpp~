// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_values = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);

    // Wait for data in circular buffer
    cb_wait_front(cb_id, num_values);

    // Get pointer to data
    uint32_t cb_addr = get_read_ptr(cb_id);
    uint32_t* data = (uint32_t*)cb_addr;

    // Simple processing: multiply each value by 2
    for (uint32_t i = 0; i < num_values; i++) {
        data[i] = data[i] * 2;
    }

    // Data remains in place for writer kernel
    // No need to pop since writer will handle it
}
}  // namespace NAMESPACE
