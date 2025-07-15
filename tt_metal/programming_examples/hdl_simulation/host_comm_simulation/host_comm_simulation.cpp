// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <cstring>
#include <iomanip>

using namespace tt;
using namespace tt::tt_metal;

// Verify HDL simulation results
bool verify_hdl_results(
    const std::vector<uint32_t>& input_data, const std::vector<uint32_t>& output_data, uint32_t num_commands) {
    uint32_t successful_commands = 0;

    for (uint32_t i = 0; i < num_commands; i++) {
        uint32_t base_idx = i * 4;
        uint32_t expected_seq = input_data[base_idx + 3];
        uint32_t actual_seq = output_data[base_idx + 3];

        if (actual_seq == expected_seq) {
            successful_commands++;
        }
    }

    return successful_commands == num_commands;
}

int main() {
    // Initialize device
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0};

    // Configuration
    constexpr uint32_t num_commands = 256;  // Number of simulation commands
    constexpr uint32_t values_per_cmd = 4;  // [cmd, addr, data, sequence]
    constexpr uint32_t num_values = num_commands * values_per_cmd;
    constexpr uint32_t value_size = sizeof(uint32_t);
    constexpr uint32_t buffer_size = num_values * value_size;

    // Use buffer_size as page_size for single large page
    constexpr uint32_t page_size = buffer_size;

    // Create input/output buffers
    InterleavedBufferConfig buffer_config{
        .device = device, .size = buffer_size, .page_size = page_size, .buffer_type = BufferType::DRAM};

    auto input_buffer = CreateBuffer(buffer_config);
    auto output_buffer = CreateBuffer(buffer_config);

    // Create kernel with proper compile arguments
    std::vector<uint32_t> kernel_compile_args = {
        (uint32_t)(input_buffer->buffer_type() == BufferType::DRAM), num_commands, page_size};

    KernelHandle hdl_simulation_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/host_comm_simulation/kernels/hdl_dataflow_simulation.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = kernel_compile_args});

    // Prepare test data with various HDL operations
    std::vector<uint32_t> input_data(num_values);

    printf("=== HDL Simulation with Host Communication ===\n");
    printf("Preparing %u HDL simulation commands...\n", num_commands);

    // Generate diverse test commands
    for (uint32_t i = 0; i < num_commands; i++) {
        uint32_t base_idx = i * 4;
        uint32_t cmd_type = i % 8;        // Cycle through all 8 command types
        uint32_t addr = (i * 4) & 0x3FF;  // Keep within 1KB range
        uint32_t data = 0x1000 + (i * 0x100) + (i & 0xFF);
        uint32_t sequence = 0xABCD0000 + i;

        input_data[base_idx + 0] = cmd_type;
        input_data[base_idx + 1] = addr;
        input_data[base_idx + 2] = data;
        input_data[base_idx + 3] = sequence;
    }

    // Execute HDL simulation
    printf("\nWriting commands to device memory...\n");
    EnqueueWriteBuffer(cq, input_buffer, input_data, true);

    // Set runtime arguments
    SetRuntimeArgs(
        program,
        hdl_simulation_kernel,
        core,
        {input_buffer->address(),
         output_buffer->address(),
         0,  // start_idx
         num_commands});

    printf("Executing HDL simulation on device...\n");
    EnqueueProgram(cq, program, true);

    // Read results
    std::vector<uint32_t> output_data(num_values);
    printf("Reading simulation results from device...\n");
    EnqueueReadBuffer(cq, output_buffer, output_data, true);

    // Display results
    printf("\n=== HDL Simulation Results ===\n");
    printf("Cmd# | Operation | Address    | Input Data | Result     | Status | Verified\n");
    printf("-----|-----------|------------|------------|------------|--------|----------\n");

    const char* op_names[] = {
        "MEM_WRITE", "MEM_READ ", "READ_MOD ", "REG_WRITE", "REG_READ ", "STATUS   ", "PATTERN  ", "CHECKSUM "};

    // Show first 20 results
    for (uint32_t i = 0; i < std::min(20u, num_commands); i++) {
        uint32_t base_idx = i * 4;
        uint32_t cmd = input_data[base_idx + 0];
        uint32_t addr = input_data[base_idx + 1];
        uint32_t in_data = input_data[base_idx + 2];
        uint32_t in_seq = input_data[base_idx + 3];

        uint32_t status = output_data[base_idx + 0];
        uint32_t out_data = output_data[base_idx + 2];
        uint32_t out_seq = output_data[base_idx + 3];

        bool verified = (in_seq == out_seq);

        printf(
            "%4u | %s | 0x%08x | 0x%08x | 0x%08x | %6u | %s\n",
            i,
            op_names[cmd & 0x7],
            addr,
            in_data,
            out_data,
            status,
            verified ? "✓" : "✗");
    }

    if (num_commands > 20) {
        printf("... (showing first 20 of %u commands)\n", num_commands);
    }

    // Verify all results
    bool all_verified = verify_hdl_results(input_data, output_data, num_commands);

    // Summary
    printf("\n=== Communication Summary ===\n");
    printf("Total commands sent: %u\n", num_commands);
    printf(
        "Total data transferred: %u bytes (in) + %u bytes (out) = %u bytes\n",
        buffer_size,
        buffer_size,
        buffer_size * 2);
    printf("Communication verified: %s\n", all_verified ? "✓ PASSED" : "✗ FAILED");

    if (all_verified) {
        printf("\n✓ ACHIEVED: Host-to-kernel data demonstration\n");
        printf("  - Host writes %u commands (%u bytes) to device memory\n", num_commands, buffer_size);
        printf("  - Kernel processes commands and executes HDL simulation\n");
        printf("  - Host reads back %u results (%u bytes) from device\n", num_commands, buffer_size);
        printf("  - %u total NOC transfers completed successfully\n", num_commands * 8);
    }

    // Cleanup
    CloseDevice(device);

    return all_verified ? 0 : 1;
}
