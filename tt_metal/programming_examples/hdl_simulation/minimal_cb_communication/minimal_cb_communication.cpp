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

    // Data params: Small tensor (2 tiles for verification)
    constexpr uint32_t num_tiles = 2;
    constexpr uint32_t dram_buffer_size = TILE_HEIGHT * TILE_WIDTH * num_tiles * sizeof(bfloat16);

    log_info(tt::LogTest, "=== TT-Metal Pipeline Verification Demo ===");
    log_info(tt::LogTest, "Setting up buffers: {} tiles, {} bytes total", num_tiles, dram_buffer_size);

    // DRAM buffers for host I/O
    InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size / num_tiles,
        .buffer_type = BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> input_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = CreateBuffer(dram_config);

    // Host → Device: Generate test data with known pattern
    std::vector<bfloat16> input_vec(TILE_HEIGHT * TILE_WIDTH * num_tiles);

    // Create specific test patterns for better verification
    for (size_t i = 0; i < input_vec.size(); ++i) {
        // Use a mix of positive, negative, zero, and fractional values
        if (i % 100 == 0) {
            input_vec[i] = bfloat16(0.0f);  // Test zero
        } else if (i % 100 == 50) {
            input_vec[i] = bfloat16(-1.0f);  // Test negative
        } else {
            input_vec[i] = bfloat16(static_cast<float>(i % 100) / 10.0f - 5.0f);  // -5.0 to +4.9
        }
    }

    log_info(
        tt::LogTest,
        "Input sample: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} ... {:.2f}",
        input_vec[0].to_float(),
        input_vec[1].to_float(),
        input_vec[2].to_float(),
        input_vec[49].to_float(),
        input_vec[50].to_float(),
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

    // Compute kernel: cb_in → pass-through → cb_out
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

    log_info(tt::LogTest, "Launching pipeline (operation: pass-through verification)...");

    // Launch and sync
    EnqueueProgram(cq, program, false);
    Finish(cq);

    log_info(tt::LogTest, "Kernels completed, reading and verifying results...");

    // Device → Host: Read output and verify computation
    std::vector<bfloat16> result_vec;
    EnqueueReadBuffer(cq, *output_dram_buffer, result_vec, true);

    log_info(
        tt::LogTest,
        "Output sample: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} ... {:.2f}",
        result_vec[0].to_float(),
        result_vec[1].to_float(),
        result_vec[2].to_float(),
        result_vec[49].to_float(),
        result_vec[50].to_float(),
        result_vec[result_vec.size() - 1].to_float());

    // Detailed pipeline verification
    uint32_t mismatches = 0;
    uint32_t perfect_matches = 0;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    const uint32_t max_mismatches_to_show = 10;

    for (size_t i = 0; i < input_vec.size(); ++i) {
        float input_val = input_vec[i].to_float();
        float output_val = result_vec[i].to_float();
        float expected_val = input_val;  // Expected: pass-through (input == output)
        float error = std::abs(output_val - expected_val);

        avg_error += error;
        max_error = std::max(max_error, error);

        // Allow for small bfloat16 precision errors
        if (error > 1e-3f) {  // bfloat16 has limited precision
            if (mismatches < max_mismatches_to_show) {
                log_info(
                    tt::LogTest,
                    "Mismatch[{}]: input={:.6f}, expected={:.6f}, output={:.6f}, error={:.6f}",
                    i,
                    input_val,
                    expected_val,
                    output_val,
                    error);
            }
            mismatches++;
            pass = false;
        } else {
            perfect_matches++;
        }
    }

    avg_error /= input_vec.size();

    // Comprehensive verification report
    log_info(tt::LogTest, "=== PIPELINE VERIFICATION RESULTS ===");
    log_info(tt::LogTest, "Total elements processed: {}", input_vec.size());
    log_info(
        tt::LogTest, "Perfect matches: {} ({:.1f}%)", perfect_matches, 100.0f * perfect_matches / input_vec.size());
    log_info(tt::LogTest, "Data corruption errors: {} ({:.1f}%)", mismatches, 100.0f * mismatches / input_vec.size());
    log_info(tt::LogTest, "Maximum error: {:.6f}", max_error);
    log_info(tt::LogTest, "Average error: {:.6f}", avg_error);

    if (mismatches > max_mismatches_to_show) {
        log_info(tt::LogTest, "... and {} more mismatches not shown", mismatches - max_mismatches_to_show);
    }

    // Performance and throughput stats
    log_info(tt::LogTest, "=== PERFORMANCE CHARACTERISTICS ===");
    log_info(tt::LogTest, "Data processed: {} KB", (input_vec.size() * sizeof(bfloat16)) / 1024);
    log_info(tt::LogTest, "Tiles processed: {}", num_tiles);
    log_info(tt::LogTest, "Elements per tile: {}", TILE_HEIGHT * TILE_WIDTH);
    log_info(tt::LogTest, "Operation: pass-through (data integrity verification)");

    // Test specific edge cases
    log_info(tt::LogTest, "=== EDGE CASE VERIFICATION ===");
    bool edge_cases_pass = true;

    // Find and verify some specific test cases
    for (size_t i = 0; i < input_vec.size() && i < 200; ++i) {
        float input_val = input_vec[i].to_float();
        float output_val = result_vec[i].to_float();

        if (std::abs(input_val - 0.0f) < 1e-6f) {
            // Test: 0 → 0 (pass-through)
            if (std::abs(output_val - 0.0f) > 1e-3f) {
                log_info(tt::LogTest, "FAIL: Zero test: 0.0 → {:.6f} (expected 0.0)", output_val);
                edge_cases_pass = false;
            }
        }
        if (std::abs(input_val - (-1.0f)) < 1e-6f) {
            // Test: -1 → -1 (pass-through)
            if (std::abs(output_val - (-1.0f)) > 1e-3f) {
                log_info(tt::LogTest, "FAIL: Negative test: -1.0 → {:.6f} (expected -1.0)", output_val);
                edge_cases_pass = false;
            }
        }
    }

    if (edge_cases_pass) {
        log_info(tt::LogTest, "PASS: Edge case verification (zero, negative values preserved)");
    }

    pass = pass && edge_cases_pass;

    log_info(tt::LogTest, "=== FINAL RESULT ===");
    log_info(tt::LogTest, "Overall verification: {}", pass ? "PASS ✓" : "FAIL ✗");

    if (pass) {
        log_info(tt::LogTest, "🎉 SUCCESS: TT-Metal pipeline verified!");
        log_info(tt::LogTest, "✓ Data integrity maintained through full pipeline");
        log_info(tt::LogTest, "✓ Reader kernel: DRAM → L1 CB transfer working");
        log_info(tt::LogTest, "✓ Compute kernel: CB → CB pass-through working");
        log_info(tt::LogTest, "✓ Writer kernel: L1 CB → DRAM transfer working");
        log_info(tt::LogTest, "✓ Pipeline ready for computational workloads");
        log_info(tt::LogTest, "");
        log_info(tt::LogTest, "🚀 Ready for next steps:");
        log_info(tt::LogTest, "   • Add actual computation in compute kernel");
        log_info(tt::LogTest, "   • Scale to multiple cores/tiles");
        log_info(tt::LogTest, "   • Integrate with HDL simulation acceleration");
    } else {
        log_info(tt::LogTest, "❌ FAILURE: Pipeline verification failed");
        log_info(tt::LogTest, "Check DPRINT output for debugging (export TT_METAL_DPRINT_CORES=0,0)");
        log_info(tt::LogTest, "Data corruption detected - investigate pipeline integrity");
    }

    pass &= CloseDevice(device);
    return pass ? 0 : 1;
}
