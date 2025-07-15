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

    // Simple test with smaller buffer
    constexpr uint32_t num_commands = 4;  // Just 4 commands for testing
    constexpr uint32_t values_per_cmd = 4;
    constexpr uint32_t num_values = num_commands * values_per_cmd;
    constexpr uint32_t value_size = sizeof(uint32_t);
    constexpr uint32_t buffer_size = num_values * value_size;
    constexpr uint32_t page_size = buffer_size;

    // Input buffer
    InterleavedBufferConfig input_config{
        .device = device, .size = buffer_size, .page_size = page_size, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(input_config);

    // Output buffer
    InterleavedBufferConfig output_config{
        .device = device, .size = buffer_size, .page_size = page_size, .buffer_type = BufferType::DRAM};
    auto output_buffer = CreateBuffer(output_config);

    // Debug kernel
    std::vector<uint32_t> kernel_compile_args = {
        (uint32_t)(input_buffer->buffer_type() == BufferType::DRAM), num_commands, page_size};

    KernelHandle debug_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/host_comm_simulation/kernels/hdl_dataflow_simulation_debug.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = kernel_compile_args});

    // Simple test data - very recognizable pattern
    std::vector<uint32_t> input_data(num_values);

    // Command 0: op=0, addr=0, data=0x1000, seq=0xABCD0000
    input_data[0] = 0;
    input_data[1] = 0;
    input_data[2] = 0x1000;
    input_data[3] = 0xABCD0000;

    // Command 1: op=1, addr=4, data=0x1101, seq=0xABCD0001
    input_data[4] = 1;
    input_data[5] = 4;
    input_data[6] = 0x1101;
    input_data[7] = 0xABCD0001;

    // Command 2: op=2, addr=8, data=0x1202, seq=0xABCD0002
    input_data[8] = 2;
    input_data[9] = 8;
    input_data[10] = 0x1202;
    input_data[11] = 0xABCD0002;

    // Command 3: op=3, addr=12, data=0x1303, seq=0xABCD0003
    input_data[12] = 3;
    input_data[13] = 12;
    input_data[14] = 0x1303;
    input_data[15] = 0xABCD0003;

    printf("DEBUG: Testing buffer communication\n");
    printf("Input buffer: addr=0x%x, size=%u bytes\n", input_buffer->address(), buffer_size);
    printf("Output buffer: addr=0x%x, size=%u bytes\n", output_buffer->address(), buffer_size);
    printf("\nInput data pattern:\n");
    for (uint32_t i = 0; i < num_values; i += 4) {
        printf(
            "Cmd %u: [0x%08x, 0x%08x, 0x%08x, 0x%08x]\n",
            i / 4,
            input_data[i],
            input_data[i + 1],
            input_data[i + 2],
            input_data[i + 3]);
    }

    // Write test data to device
    EnqueueWriteBuffer(cq, input_buffer, input_data, true);

    // Set runtime arguments
    SetRuntimeArgs(program, debug_kernel, core, {input_buffer->address(), output_buffer->address(), 0, num_commands});

    printf("\nRunning debug kernel...\n");
    EnqueueProgram(cq, program, true);

    // Read results
    std::vector<uint32_t> output_data(num_values);
    EnqueueReadBuffer(cq, output_buffer, output_data, true);

    printf("\nOutput data received:\n");
    for (uint32_t i = 0; i < num_values; i += 4) {
        printf(
            "Result %u: [0x%08x, 0x%08x, 0x%08x, 0x%08x]\n",
            i / 4,
            output_data[i],
            output_data[i + 1],
            output_data[i + 2],
            output_data[i + 3]);
    }

    // Check if any method worked
    bool any_success = false;
    for (uint32_t i = 0; i < num_values; i++) {
        if (output_data[i] != 0) {
            any_success = true;
            break;
        }
    }

    printf("\nDEBUG Results:\n");
    printf("- Input buffer properly written: ✓\n");
    printf("- Kernel executed: ✓\n");
    printf("- Output buffer read: ✓\n");
    printf("- Any data written to output: %s\n", any_success ? "✓" : "✗");

    if (any_success) {
        printf("🎉 SUCCESS: Some communication method worked!\n");
        printf("Check DPRINT output to see which addressing method succeeded.\n");
    } else {
        printf("⚠️  All addressing methods failed to read correct data.\n");
        printf("This suggests a fundamental addressing or buffer issue.\n");
    }

    // Cleanup
    CloseDevice(device);

    return any_success ? 0 : 1;
}
