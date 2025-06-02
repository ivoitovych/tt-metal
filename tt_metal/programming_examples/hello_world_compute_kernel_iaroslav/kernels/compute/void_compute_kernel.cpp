// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

// RTL Simulation State Structure
struct RTLSimState {
    uint8_t clk;
    uint8_t rst;
    uint8_t div1;
    uint8_t div2;
    uint8_t div3;
    uint8_t div4;
    uint32_t cycle_count;
};

// Initialize simulation state
inline void init_sim_state(RTLSimState* state) {
    state->clk = 0;
    state->rst = 1;  // Start with reset asserted
    state->div1 = 0;
    state->div2 = 0;
    state->div3 = 0;
    state->div4 = 0;
    state->cycle_count = 0;
}

// Toggle clock
inline void toggle_clock(RTLSimState* state) { state->clk = !state->clk; }

// Evaluate combinational and sequential logic on positive clock edge
inline void eval_posedge_clk(RTLSimState* state) {
    if (state->rst) {
        // Reset logic
        state->div1 = 0;
        state->div2 = 0;
        state->div3 = 0;
        state->div4 = 0;
        state->cycle_count = 0;
    } else {
        // Normal operation
        state->cycle_count++;

        // Divider chain logic (frequency dividers)
        // div1 toggles every clock
        state->div1 = !state->div1;

        // div2 toggles when div1 goes high
        if (state->div1) {
            state->div2 = !state->div2;
        }

        // div3 toggles when both div1 and div2 are high
        if (state->div1 && state->div2) {
            state->div3 = !state->div3;
        }

        // div4 toggles when div1, div2, and div3 are all high
        if (state->div1 && state->div2 && state->div3) {
            state->div4 = !state->div4;
        }
    }
}

// Print current state using DPRINT
inline void print_state(RTLSimState* state, uint32_t time_ns) {
    DPRINT << "time=" << time_ns << " ns  cycle=" << state->cycle_count << "  div1=" << (uint32_t)state->div1
           << "  div2=" << (uint32_t)state->div2 << "  div3=" << (uint32_t)state->div3
           << "  div4=" << (uint32_t)state->div4 << ENDL();
}

// RTL simulation logic
void run_rtl_simulation() {
    // Initialize simulation state
    RTLSimState sim_state;
    init_sim_state(&sim_state);

    DPRINT << "Starting RTL simulation on TT-Metal kernel" << ENDL();
    DPRINT << "===========================================>" << ENDL();

    // Simulation parameters
    constexpr uint32_t CLOCK_PERIOD_NS = 100;   // 50ns high, 50ns low = 100ns period
    constexpr uint32_t RESET_DURATION_NS = 80;  // Reset for 80ns
    constexpr uint32_t MAX_CYCLES = 49;         // Run for 49 cycles (matching original)

    uint32_t simulation_time_ns = 0;
    uint32_t cycles_completed = 0;

    // Main simulation loop
    while (cycles_completed < MAX_CYCLES) {
        // Rising edge of clock
        toggle_clock(&sim_state);

        if (sim_state.clk) {  // Positive edge
            // Check if we should deassert reset
            if (simulation_time_ns >= RESET_DURATION_NS && sim_state.rst) {
                sim_state.rst = 0;
                DPRINT << "Reset deasserted at time " << simulation_time_ns << " ns" << ENDL();
            }

            // Evaluate logic on positive clock edge
            eval_posedge_clk(&sim_state);

            // Print state after evaluation
            print_state(&sim_state, simulation_time_ns);

            if (!sim_state.rst) {
                cycles_completed++;
            }
        }

        // Advance simulation time by half clock period
        simulation_time_ns += CLOCK_PERIOD_NS / 2;

        // Falling edge of clock
        if (sim_state.clk) {
            toggle_clock(&sim_state);
            simulation_time_ns += CLOCK_PERIOD_NS / 2;
        }
    }

    DPRINT << "===========================================>" << ENDL();
    DPRINT << "RTL simulation completed after " << cycles_completed << " cycles" << ENDL();
    DPRINT << "Final simulation time: " << simulation_time_ns << " ns" << ENDL();
    DPRINT << "Final state: div1=" << (uint32_t)sim_state.div1 << " div2=" << (uint32_t)sim_state.div2
           << " div3=" << (uint32_t)sim_state.div3 << " div4=" << (uint32_t)sim_state.div4 << ENDL();
}

// Required TT-Metal compute kernel functions
namespace NAMESPACE {

void unpack_main() {
    // Empty unpack function - no data movement needed for this simulation
}

void math_main() {
    // Run the RTL simulation in the math stage
    run_rtl_simulation();
}

void pack_main() {
    // Empty pack function - no data movement needed for this simulation
}

}  // namespace NAMESPACE
