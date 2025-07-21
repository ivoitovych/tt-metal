#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <iostream>

using namespace tt::tt_metal;
using namespace tt::constants;

int main(int argc, char** argv) {
    bool pass = true;

    // Initialize device
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Data params: Small tensor (1x1x32x64 for 2 tiles)
    constexpr uint32_t num_tiles = 2;
    constexpr uint32_t dram_buffer_size = TILE_HEIGHT * TILE_WIDTH * num_tiles * sizeof(bfloat16);

    log_info(tt::LogTest, "Setting up buffers: {} tiles, {} bytes total", num_tiles, dram_buffer_size);

    // DRAM buffers for host I/O
    InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size / num_tiles,
        .buffer_type = BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> input_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = CreateBuffer(dram_config);

    // Host → Device: Generate and write input data with known pattern for easier verification
    std::vector<bfloat16> input_vec(TILE_HEIGHT * TILE_WIDTH * num_tiles);

    // Create a predictable pattern instead of random for easier debugging
    for (size_t i = 0; i < input_vec.size(); ++i) {
        input_vec[i] = bfloat16(static_cast<float>(i % 100) / 10.0f);  // 0.0, 0.1, 0.2, ..., 9.9, 0.0, ...
    }

    log_info(
        tt::LogTest,
        "Input data sample: {:.2f} {:.2f} {:.2f} {:.2f} ... {:.2f}",
        input_vec[0].to_float(),
        input_vec[1].to_float(),
        input_vec[2].to_float(),
        input_vec[3].to_float(),
        input_vec[input_vec.size() - 1].to_float());

    EnqueueWriteBuffer(cq, *input_dram_buffer, input_vec, false);

    // Circular Buffers setup
    constexpr uint32_t cb_page_size = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(cb_page_size * 2, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, cb_page_size);
    CBHandle cb_in = CreateCircularBuffer(program, CoreRange(CoreCoord(0, 0)), cb_in_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(cb_page_size * 2, {{16, tt::DataFormat::Float16_b}}).set_page_size(16, cb_page_size);
    CBHandle cb_out = CreateCircularBuffer(program, CoreRange(CoreCoord(0, 0)), cb_out_config);

    // Kernels on single core {0,0}
    CoreCoord core = {0, 0};

    // Reader kernel: DRAM → cb_in
    KernelHandle reader_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_cb_communication/kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Compute kernel: cb_in → process → cb_out
    KernelHandle compute_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_cb_communication/kernels/compute/add_one.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

    // Writer kernel: cb_out → DRAM
    KernelHandle writer_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_cb_communication/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Set runtime args
    SetRuntimeArgs(program, reader_id, core, {input_dram_buffer->address(), num_tiles});
    SetRuntimeArgs(program, compute_id, core, {num_tiles});
    SetRuntimeArgs(program, writer_id, core, {output_dram_buffer->address(), num_tiles});

    log_info(tt::LogTest, "Launching kernels...");

    // Launch and sync
    EnqueueProgram(cq, program, false);
    Finish(cq);

    log_info(tt::LogTest, "Kernels completed, reading results...");

    // Device → Host: Read output and verify
    std::vector<bfloat16> result_vec;
    EnqueueReadBuffer(cq, *output_dram_buffer, result_vec, true);

    log_info(
        tt::LogTest,
        "Output data sample: {:.2f} {:.2f} {:.2f} {:.2f} ... {:.2f}",
        result_vec[0].to_float(),
        result_vec[1].to_float(),
        result_vec[2].to_float(),
        result_vec[3].to_float(),
        result_vec[result_vec.size() - 1].to_float());

    // Detailed verification - expect output = input (pass-through)
    uint32_t mismatches = 0;
    const uint32_t max_mismatches_to_show = 10;

    for (size_t i = 0; i < input_vec.size(); ++i) {
        float input_val = input_vec[i].to_float();
        float output_val = result_vec[i].to_float();
        float expected_val = input_val;  // Expect pass-through from compute kernel

        if (std::abs(output_val - expected_val) > 1e-6f) {  // Allow small floating point errors
            if (mismatches < max_mismatches_to_show) {
                log_info(
                    tt::LogTest,
                    "Mismatch at index {}: input={:.6f}, expected={:.6f}, output={:.6f}",
                    i,
                    input_val,
                    expected_val,
                    output_val);
            }
            mismatches++;
            pass = false;
        }
    }

    if (mismatches > 0) {
        log_info(tt::LogTest, "Total mismatches: {} out of {} elements", mismatches, input_vec.size());
    }

    // Statistics
    log_info(tt::LogTest, "Data transfer verification:");
    log_info(tt::LogTest, "  Total elements: {}", input_vec.size());
    log_info(tt::LogTest, "  Total bytes: {}", input_vec.size() * sizeof(bfloat16));
    log_info(tt::LogTest, "  Tiles processed: {}", num_tiles);
    log_info(tt::LogTest, "  Elements per tile: {}", TILE_HEIGHT * TILE_WIDTH);

    log_info(tt::LogTest, "Verification passed: {}", pass);
    pass &= CloseDevice(device);
    return pass ? 0 : 1;
}
