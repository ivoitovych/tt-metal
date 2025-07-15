#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Silicon accelerator setup
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Define core coordinates
    constexpr CoreCoord core = {0, 0};

    // Define buffer and circular buffer configuration
    constexpr uint32_t single_tile_size = 2 * 1024;
    constexpr uint32_t num_tiles = 1;

    // Create DRAM buffer
    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    // Create circular buffers
    constexpr uint32_t src_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src_config = CircularBufferConfig(single_tile_size, {{src_cb_index, DataFormat::Float16_b}})
                                             .set_page_size(src_cb_index, single_tile_size);
    auto cb_src = CreateCircularBuffer(program, core, cb_src_config);

    constexpr uint32_t dst_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_dst_config = CircularBufferConfig(single_tile_size, {{dst_cb_index, DataFormat::Float16_b}})
                                             .set_page_size(dst_cb_index, single_tile_size);
    auto cb_dst = CreateCircularBuffer(program, core, cb_dst_config);

    // Create data movement kernels
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/simplest_communication/kernels/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/simplest_communication/kernels/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Create compute kernel
    auto compute_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/simplest_communication/kernels/compute.cpp",
        core,
        ComputeConfig{});

    // Prepare source data
    std::vector<uint32_t> src_data = create_constant_vector_of_bfloat16(single_tile_size, 42.0f);

    // Write source data to DRAM
    EnqueueWriteBuffer(cq, src_dram_buffer, src_data, false);

    // Set runtime arguments
    SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            src_dram_buffer->address(),
            0  // bank_id
        });

    SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            dst_dram_buffer->address(),
            0  // bank_id
        });

    // Enqueue and execute program
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Read result back
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    // Print result
    printf("Result = %f\n", bfloat16(result_vec[0]).to_float());

    CloseDevice(device);
    return 0;
}
