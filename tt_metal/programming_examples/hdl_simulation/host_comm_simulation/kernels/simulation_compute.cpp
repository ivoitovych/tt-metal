// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_values = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id = tt::CBIndex::c_0;

    // Simple HDL module simulation with memory
    uint32_t memory[256] = {0};  // Simulated memory
    uint32_t status_reg = 0;

    DPRINT << "HDL Simulation Engine Started" << ENDL();

    for (uint32_t i = 0; i < num_values; i += 4) {
        // Wait for command packet (4 words)
        cb_wait_front(cb_id, 4);

        // Read command packet
        uint32_t* cmd_ptr = (uint32_t*)get_read_ptr(cb_id);
        uint32_t cmd = cmd_ptr[0];
        uint32_t addr = cmd_ptr[1];
        uint32_t data = cmd_ptr[2];
        uint32_t reserved = cmd_ptr[3];

        // Process command (simulate HDL module behavior)
        uint32_t result_status = 0;
        uint32_t result_addr = addr;
        uint32_t result_data = 0;

        switch (cmd & 0xFF) {
            case 0:  // Write
                if ((addr >> 2) < 256) {
                    memory[addr >> 2] = data;
                    result_status = 1;  // Success
                    result_data = data;
                }
                break;

            case 1:  // Read
                if ((addr >> 2) < 256) {
                    result_data = memory[addr >> 2];
                    result_status = 1;  // Success
                }
                break;

            case 2:  // Read-Modify-Write (XOR)
                if ((addr >> 2) < 256) {
                    uint32_t old_val = memory[addr >> 2];
                    memory[addr >> 2] = old_val ^ data;
                    result_data = memory[addr >> 2];
                    result_status = 1;
                }
                break;

            default:
                result_status = 0xFF;  // Error
                break;
        }

        // Write results back
        uint32_t* result_ptr = (uint32_t*)get_write_ptr(cb_id);
        result_ptr[0] = result_status;
        result_ptr[1] = result_addr;
        result_ptr[2] = result_data;
        result_ptr[3] = cmd;  // Echo command

        cb_pop_front(cb_id, 4);
        cb_push_back(cb_id, 4);

        // Debug output for first few commands
        if (i < 40) {
            DPRINT << "Cmd " << (i / 4) << ": op=" << (cmd & 0xFF) << " addr=" << HEX() << addr << " data=" << data
                   << " result=" << result_data << DEC() << ENDL();
        }
    }

    DPRINT << "Simulation complete. Processed " << (num_values / 4) << " commands" << ENDL();
}
}  // namespace NAMESPACE
