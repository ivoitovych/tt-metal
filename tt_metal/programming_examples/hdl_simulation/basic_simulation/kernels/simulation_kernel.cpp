// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_cycles = get_compile_time_arg_val(0);
    constexpr uint32_t seed = get_compile_time_arg_val(1);

    // Simple HDL simulation example - simulating a counter with enable
    uint32_t counter = 0;
    uint32_t enable = 1;
    uint32_t lfsr = seed;  // Linear feedback shift register for pseudo-random

    DPRINT << "Starting HDL Simulation with " << num_cycles << " cycles" << ENDL();

    for (uint32_t cycle = 0; cycle < num_cycles; cycle++) {
        // Simulate some HDL logic
        if (enable) {
            counter++;
        }

        // LFSR for pseudo-random enable signal
        uint32_t bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
        lfsr = (lfsr >> 1) | (bit << 31);
        enable = lfsr & 1;

        // Print state every 100 cycles
        if (cycle % 100 == 0) {
            DPRINT << "Cycle " << cycle << ": counter=" << counter << " enable=" << enable << " lfsr=" << HEX() << lfsr
                   << DEC() << ENDL();
        }
    }

    DPRINT << "Simulation complete. Final counter value: " << counter << ENDL();
}
}  // namespace NAMESPACE
