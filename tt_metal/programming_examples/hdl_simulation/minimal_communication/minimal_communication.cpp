// minimal_communication.cpp

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/command_queue.hpp>
#include <vector>
#include <iostream>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Input/output data
    std::vector<uint32_t> input_data = {10, 20, 30, 40};
    constexpr uint32_t num_values = 4;
    constexpr uint32_t data_size = num_values * sizeof(uint32_t);

    // Buffers
    auto dram_cfg =
        BufferConfig{.device = device, .size = data_size, .page_size = data_size, .buffer_type = BufferType::DRAM};
    auto input_buf = CreateBuffer(dram_cfg);
    auto output_buf = CreateBuffer(dram_cfg);

    // CB for scalar data
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    CircularBufferConfig cb_in_cfg =
        CircularBufferConfig(data_size, {{cb_in, DataFormat::UInt32}}).set_page_size(cb_in, data_size);
    CircularBufferConfig cb_out_cfg =
        CircularBufferConfig(data_size, {{cb_out, DataFormat::UInt32}}).set_page_size(cb_out, data_size);
    auto cb_in_handle = CreateCircularBuffer(program, core, cb_in_cfg);
    auto cb_out_handle = CreateCircularBuffer(program, core, cb_out_cfg);

    // Reader kernel (RISCV0)
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_communication/kernels/minimal_reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {cb_in}});

    // Compute kernel (TRISC0)
    auto compute_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_communication/kernels/minimal_compute.cpp",
        core,
        ComputeConfig{.compile_args = {cb_in, cb_out}});

    // Writer kernel (RISCV1)
    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_communication/kernels/minimal_writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {cb_out}});

    // Set runtime args
    SetRuntimeArgs(program, reader_kernel, core, {input_buf->address(), num_values});
    SetRuntimeArgs(program, compute_kernel, core, {num_values});
    SetRuntimeArgs(program, writer_kernel, core, {output_buf->address(), num_values});

    // Write input, run, read output
    std::cout << "Sending data to device: ";
    for (auto v : input_data) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    EnqueueWriteBuffer(cq, input_buf, input_data.data(), false);
    EnqueueProgram(cq, program, false);
    std::vector<uint32_t> output_data(num_values);
    EnqueueReadBuffer(cq, output_buf, output_data.data(), true);

    std::cout << "Received data from device: ";
    for (auto v : output_data) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    CloseDevice(device);
    return 0;
}
