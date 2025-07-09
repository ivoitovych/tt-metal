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
    constexpr uint32_t num_values = 1024;  // 1K values
    constexpr uint32_t value_size = sizeof(uint32_t);
    constexpr uint32_t buffer_size = num_values * value_size;

    // Input buffer - simulation commands/data from host
    tt_metal::InterleavedBufferConfig input_config{
        .device = device, .size = buffer_size, .page_size = buffer_size, .buffer_type = tt_metal::BufferType::DRAM};
    auto input_buffer = CreateBuffer(input_config);

    // Output buffer - simulation results to host
    auto output_buffer = CreateBuffer(input_config);

    // Create circular buffer for data movement
    uint32_t cb_index = CBIndex::c_0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(buffer_size, {{cb_index, tt::DataFormat::UInt32}}).set_page_size(cb_index, value_size);
    auto cb_data = CreateCircularBuffer(program, core, cb_config);

    // Create reader kernel
    std::vector<uint32_t> reader_compile_args = {(uint32_t)(input_buffer->buffer_type() == BufferType::DRAM)};

    KernelHandle reader_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/host_comm_simulation/kernels/reader_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_args});

    // Create compute kernel for simulation
    KernelHandle compute_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/host_comm_simulation/kernels/simulation_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {num_values}});

    // Prepare simulation input data
    std::vector<uint32_t> input_data(num_values);

    // Example: Create a simple HDL test pattern
    // Format: [command, address, data, expected_result, ...]
    for (uint32_t i = 0; i < num_values; i += 4) {
        input_data[i] = i / 4;               // Command ID
        input_data[i + 1] = i * 16;          // Address
        input_data[i + 2] = 0xDEADBEEF + i;  // Write data
        input_data[i + 3] = 0;               // Reserved/expected
    }

    // Write input data
    EnqueueWriteBuffer(cq, input_buffer, input_data, false);

    // Set runtime args
    SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {input_buffer->address(),
         output_buffer->address(),
         0,  // start_addr
         num_values});

    SetRuntimeArgs(program, compute_kernel, core, {});

    // Run simulation
    printf("Starting HDL simulation with host communication...\n");
    printf("Processing %u commands\n", num_values / 4);

    EnqueueProgram(cq, program, false);

    // Read results
    std::vector<uint32_t> output_data(num_values);
    EnqueueReadBuffer(cq, output_buffer, output_data, true);

    // Print some results
    printf("\nSimulation Results (first 10 entries):\n");
    printf("Cmd# | Status | Address | Data\n");
    printf("-----|--------|---------|----------\n");

    for (uint32_t i = 0; i < std::min(40u, num_values); i += 4) {
        printf("%4u | %6u | %7x | %08x\n", i / 4, output_data[i], output_data[i + 1], output_data[i + 2]);
    }

    // Cleanup
    CloseDevice(device);

    return 0;
}
