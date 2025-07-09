// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <vector>
#include <cstring>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Initialize device
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Target single core
    constexpr CoreCoord core = {0, 0};

    // Create buffers for host-device communication
    constexpr uint32_t num_commands = 256;  // Number of simulation commands
    constexpr uint32_t values_per_cmd = 4;  // [cmd, addr, data, reserved]
    constexpr uint32_t num_values = num_commands * values_per_cmd;
    constexpr uint32_t value_size = sizeof(uint32_t);
    constexpr uint32_t buffer_size = num_values * value_size;

    // Input buffer - simulation commands from host
    tt_metal::InterleavedBufferConfig input_config{
        .device = device, .size = buffer_size, .page_size = 4096, .buffer_type = tt_metal::BufferType::DRAM};
    auto input_buffer = CreateBuffer(input_config);

    // Output buffer - simulation results to host
    tt_metal::InterleavedBufferConfig output_config{
        .device = device, .size = buffer_size, .page_size = 4096, .buffer_type = tt_metal::BufferType::DRAM};
    auto output_buffer = CreateBuffer(output_config);

    // Create unified dataflow + simulation kernel
    // This kernel will read from host, simulate HDL, and write results back
    std::vector<uint32_t> kernel_compile_args = {
        (uint32_t)(input_buffer->buffer_type() == BufferType::DRAM), num_commands};

    KernelHandle hdl_simulation_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/host_comm_simulation/kernels/hdl_dataflow_simulation.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = kernel_compile_args});

    // Prepare simulation input data
    std::vector<uint32_t> input_data(num_values);

    // Create diverse HDL test commands to demonstrate communication
    for (uint32_t i = 0; i < num_commands; i++) {
        uint32_t base_idx = i * 4;

        // Create different command types for interesting simulation
        uint32_t cmd_type = i % 8;                          // 8 different command types
        uint32_t addr = i * 4;                              // Word-aligned addresses
        uint32_t data = 0x1000 + (i * 0x100) + (i & 0xFF);  // Varying data patterns

        input_data[base_idx + 0] = cmd_type;
        input_data[base_idx + 1] = addr;
        input_data[base_idx + 2] = data;
        input_data[base_idx + 3] = 0xABCD0000 + i;  // Sequence marker
    }

    // Write input commands to device
    printf("Writing %u simulation commands to device...\n", num_commands);
    EnqueueWriteBuffer(cq, input_buffer, input_data, false);

    // Set runtime arguments
    SetRuntimeArgs(
        program,
        hdl_simulation_kernel,
        core,
        {input_buffer->address(),
         output_buffer->address(),
         0,  // start_idx
         num_commands});

    // Run HDL simulation with host communication
    printf("Starting HDL simulation with host-to-kernel communication...\n");
    printf("Each command will be read from host memory, processed, and results written back\n");

    EnqueueProgram(cq, program, false);

    // Read simulation results back from device
    std::vector<uint32_t> output_data(num_values);
    EnqueueReadBuffer(cq, output_buffer, output_data, true);

    printf("HDL simulation completed!\n\n");

    // Analyze and display results to show the communication worked
    printf("Host-to-Kernel Communication Results:\n");
    printf("=====================================\n");
    printf("Cmd# | Input Cmd | Input Addr | Input Data | Result Status | Result Data | Sequence\n");
    printf("-----|-----------|------------|------------|---------------|-------------|----------\n");

    uint32_t successful_commands = 0;
    for (uint32_t i = 0; i < std::min(20u, num_commands); i++) {
        uint32_t in_base = i * 4;
        uint32_t out_base = i * 4;

        uint32_t input_cmd = input_data[in_base + 0];
        uint32_t input_addr = input_data[in_base + 1];
        uint32_t input_data_val = input_data[in_base + 2];
        uint32_t input_seq = input_data[in_base + 3];

        uint32_t result_status = output_data[out_base + 0];
        uint32_t result_addr = output_data[out_base + 1];
        uint32_t result_data = output_data[out_base + 2];
        uint32_t result_seq = output_data[out_base + 3];

        printf(
            "%4u | %9u | 0x%08x | 0x%08x | %13u | 0x%08x | %08x\n",
            i,
            input_cmd,
            input_addr,
            input_data_val,
            result_status,
            result_data,
            result_seq);

        if (result_status == 1) {
            successful_commands++;
        }
    }

    if (num_commands > 20) {
        printf("... (showing first 20 of %u commands)\n", num_commands);
    }

    printf("\nCommunication Summary:\n");
    printf("- Commands sent to device: %u\n", num_commands);
    printf("- Commands processed successfully: %u\n", successful_commands);
    printf("- Total data transferred: %u bytes input + %u bytes output\n", buffer_size, buffer_size);

    // Verify communication integrity by checking sequence markers
    bool communication_verified = true;
    for (uint32_t i = 0; i < num_commands; i++) {
        uint32_t expected_seq = 0xABCD0000 + i;
        uint32_t actual_seq = output_data[i * 4 + 3];
        if (actual_seq != expected_seq) {
            communication_verified = false;
            break;
        }
    }

    printf("- Communication integrity: %s\n", communication_verified ? "VERIFIED ✓" : "FAILED ✗");

    if (communication_verified) {
        printf("\n✓ SUCCESS: Host-to-kernel communication working correctly!\n");
        printf("  The device successfully read commands from host memory,\n");
        printf("  processed them through HDL simulation, and wrote results back.\n");
    } else {
        printf("\n✗ FAILURE: Communication integrity check failed!\n");
    }

    // Cleanup
    CloseDevice(device);

    return communication_verified ? 0 : 1;
}
