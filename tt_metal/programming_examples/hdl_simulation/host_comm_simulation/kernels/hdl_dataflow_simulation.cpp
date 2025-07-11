// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

// HDL Module Simulation State
struct HDLSimulationState {
    uint32_t memory[256];    // Simulated memory array
    uint32_t registers[32];  // Simulated registers
    uint32_t status_reg;     // Status register
    uint32_t cycle_count;    // Simulation cycle counter

    // Constructor
    HDLSimulationState() {
        for (int i = 0; i < 256; i++) {
            memory[i] = 0;
        }
        for (int i = 0; i < 32; i++) {
            registers[i] = 0;
        }
        status_reg = 0;
        cycle_count = 0;
    }
};

// HDL simulation functions
uint32_t simulate_memory_read(HDLSimulationState& state, uint32_t addr) {
    uint32_t word_addr = addr >> 2;
    if (word_addr < 256) {
        return state.memory[word_addr];
    }
    return 0xDEADDEAD;
}

void simulate_memory_write(HDLSimulationState& state, uint32_t addr, uint32_t data) {
    uint32_t word_addr = addr >> 2;
    if (word_addr < 256) {
        state.memory[word_addr] = data;
    }
}

uint32_t simulate_register_access(HDLSimulationState& state, uint32_t reg_num, uint32_t data, bool is_write) {
    if (reg_num < 32) {
        if (is_write) {
            state.registers[reg_num] = data;
            return data;
        } else {
            return state.registers[reg_num];
        }
    }
    return 0xBADDEAD;
}

// Main HDL simulation logic
uint32_t process_hdl_command(HDLSimulationState& state, uint32_t cmd, uint32_t addr, uint32_t data) {
    state.cycle_count++;

    uint32_t result = 0;
    uint32_t opcode = cmd & 0xFF;
    uint32_t flags = (cmd >> 8) & 0xFF;

    switch (opcode) {
        case 0:  // Memory Write
            simulate_memory_write(state, addr, data);
            result = data;
            state.status_reg = 1;
            break;

        case 1:  // Memory Read
            result = simulate_memory_read(state, addr);
            state.status_reg = (result != 0xDEADDEAD) ? 1 : 0xFF;
            break;

        case 2:  // Read-Modify-Write (XOR)
        {
            uint32_t old_val = simulate_memory_read(state, addr);
            if (old_val != 0xDEADDEAD) {
                uint32_t new_val = old_val ^ data;
                simulate_memory_write(state, addr, new_val);
                result = new_val;
                state.status_reg = 1;
            } else {
                result = 0xDEADDEAD;
                state.status_reg = 0xFF;
            }
        } break;

        case 3:  // Register Write
            result = simulate_register_access(state, addr & 0x1F, data, true);
            state.status_reg = (result != 0xBADDEAD) ? 1 : 0xFF;
            break;

        case 4:  // Register Read
            result = simulate_register_access(state, addr & 0x1F, 0, false);
            state.status_reg = (result != 0xBADDEAD) ? 1 : 0xFF;
            break;

        case 5:  // Status Read
            result = state.status_reg | (state.cycle_count << 16);
            state.status_reg = 1;
            break;

        case 6:  // Memory Pattern Fill
        {
            uint32_t start_addr = addr;
            uint32_t pattern = data;
            uint32_t count = (flags == 0) ? 1 : flags;

            for (uint32_t i = 0; i < count && i < 64; i++) {
                simulate_memory_write(state, start_addr + (i * 4), pattern + i);
            }
            result = count;
            state.status_reg = 1;
        } break;

        case 7:  // Memory Checksum
        {
            uint32_t start_addr = addr >> 2;
            uint32_t count = (data == 0) ? 16 : (data & 0xFF);
            uint32_t checksum = 0;

            for (uint32_t i = 0; i < count && (start_addr + i) < 256; i++) {
                checksum ^= state.memory[start_addr + i];
                checksum = (checksum << 1) | (checksum >> 31);
            }
            result = checksum;
            state.status_reg = 1;
        } break;

        default:
            result = 0xBADC0DE;
            state.status_reg = 0xFF;
            break;
    }

    return result;
}

void kernel_main() {
    // Get runtime arguments
    uint32_t src_addr = get_arg_val<uint32_t>(0);      // Input buffer address
    uint32_t dst_addr = get_arg_val<uint32_t>(1);      // Output buffer address
    uint32_t start_idx = get_arg_val<uint32_t>(2);     // Starting index (word offset)
    uint32_t num_commands = get_arg_val<uint32_t>(3);  // Number of commands

    // Get compile-time arguments
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t max_commands = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);

    // Initialize HDL simulation state
    HDLSimulationState hdl_state;

    DPRINT << "HDL Dataflow Simulation Started (FINAL FIX)" << ENDL();
    DPRINT << "src_addr=" << HEX() << src_addr << " dst_addr=" << dst_addr << DEC() << ENDL();
    DPRINT << "page_size=" << page_size << " num_commands=" << num_commands << ENDL();
    DPRINT << "Using large page addressing for contiguous access" << ENDL();

    // FINAL FIX: Use direct byte addressing within the large page
    // Since page_size = buffer_size, we can address directly as byte offsets
    for (uint32_t cmd_idx = 0; cmd_idx < num_commands; cmd_idx++) {
        // Calculate byte offset for this command (4 words * 4 bytes each = 16 bytes per command)
        uint32_t cmd_byte_offset = cmd_idx * 16;

        // Read command directly from buffer using byte addressing
        uint32_t cmd_data[4];

        // Read the entire command in one NOC operation (16 bytes)
        noc_async_read(src_addr + cmd_byte_offset, reinterpret_cast<uint32_t>(&cmd_data[0]),
                       16);  // 4 words = 16 bytes
        noc_async_read_barrier();

        // Extract command components
        uint32_t cmd = cmd_data[0];
        uint32_t addr = cmd_data[1];
        uint32_t data = cmd_data[2];
        uint32_t sequence = cmd_data[3];

        // Process HDL command through simulation
        uint32_t result_data = process_hdl_command(hdl_state, cmd, addr, data);

        // Prepare result packet
        uint32_t result[4];
        result[0] = hdl_state.status_reg;  // Status
        result[1] = addr;                  // Address (echo)
        result[2] = result_data;           // Result data
        result[3] = sequence;              // Sequence (echo for verification)

        // Write result back using direct byte addressing
        noc_async_write(reinterpret_cast<uint32_t>(&result[0]), dst_addr + cmd_byte_offset,
                        16);  // 4 words = 16 bytes
        noc_async_write_barrier();

        // Debug output for first few commands
        if (cmd_idx < 10) {
            DPRINT << "Cmd " << cmd_idx << ": READ op=" << (cmd & 0xFF) << " addr=" << HEX() << addr << " data=" << data
                   << " seq=" << sequence << " -> result=" << result_data << " status=" << DEC() << hdl_state.status_reg
                   << ENDL();
        }
    }

    DPRINT << "HDL Simulation complete!" << ENDL();
    DPRINT << "Successfully processed " << num_commands << " commands" << ENDL();
    DPRINT << "Total simulation cycles: " << hdl_state.cycle_count << ENDL();
    DPRINT << "Real host communication: " << (num_commands * 8) << " words transferred" << ENDL();
}
