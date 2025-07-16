// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/command_queue.hpp>
#include <vector>
#include <iostream>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Initialize device
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();

    // Create program
    Program program = CreateProgram();

    // Use single core for simplicity
    constexpr CoreCoord core = {0, 0};

    // Create input data - simple array of 4 uint32_t values
    constexpr uint32_t data_size = 4 * sizeof(uint32_t);
    std::vector<uint32_t> input_data = {10, 20, 30, 40};

    // Create DRAM buffers for input and output
    InterleavedBufferConfig dram_config{
        .device = device, .size = data_size, .page_size = data_size, .buffer_type = BufferType::DRAM};

    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    // Create L1 buffer for intermediate storage
    InterleavedBufferConfig l1_config{
        .device = device, .size = data_size, .page_size = data_size, .buffer_type = BufferType::L1};
    auto l1_buffer = CreateBuffer(l1_config);

    // Create circular buffer for compute kernel
    uint32_t cb_index = CBIndex::c_0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(data_size, {{cb_index, DataFormat::UInt32}}).set_page_size(cb_index, sizeof(uint32_t));
    auto cb = CreateCircularBuffer(program, core, cb_config);

    // Compile time args
    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    // Create reader kernel - reads from DRAM to L1/CB
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_communication/kernels/minimal_reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {(uint32_t)input_is_dram, cb_index}});

    // Create compute kernel - processes data
    auto compute_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_communication/kernels/minimal_compute.cpp",
        core,
        ComputeConfig{.compile_args = {cb_index}});

    // Create writer kernel - writes from CB to DRAM
    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_communication/kernels/minimal_writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {cb_index, (uint32_t)output_is_dram}});

    // Set runtime arguments
    SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            input_buffer->address(),
            l1_buffer->address(),
            4  // number of uint32_t values
        });

    SetRuntimeArgs(
        program,
        compute_kernel,
        core,
        {
            4  // number of values to process
        });

    SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            output_buffer->address(),
            l1_buffer->address(),
            4  // number of values
        });

    // Write input data to device
    std::cout << "Sending data to device: ";
    for (auto val : input_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    EnqueueWriteBuffer(cq, input_buffer, input_data.data(), false);

    // Execute program
    EnqueueProgram(cq, program, false);

    // Read results back
    std::vector<uint32_t> output_data(4);
    EnqueueReadBuffer(cq, output_buffer, output_data.data(), true);

    // Print results
    std::cout << "Received data from device: ";
    for (auto val : output_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CloseDevice(device);

    return 0;
}
