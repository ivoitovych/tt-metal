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
#include <cstring>

using namespace tt::tt_metal;
using namespace tt::constants;

// Input structure - can contain arbitrary data
struct InputStruct {
    float float_value;
    int32_t int_value;
    uint16_t short_value;
    int8_t char_value;
    // Padding to ensure 16-byte alignment (5 bytes padding)
    uint8_t padding[5];
};

// Output structure - different from input
struct OutputStruct {
    float result_value;
    int32_t status_code;
    uint16_t iteration_count;
    uint8_t flags;
    // Padding to ensure 16-byte alignment (5 bytes padding)
    uint8_t padding[5];
};

// Ensure both structures are 16 bytes
static_assert(sizeof(InputStruct) == 16, "InputStruct must be 16 bytes");
static_assert(sizeof(OutputStruct) == 16, "OutputStruct must be 16 bytes");

int main(int argc, char** argv) {
    bool pass = true;

    // Initialize device
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Data params: Small tensor (2 tiles for verification)
    constexpr uint32_t num_tiles = 2;
    constexpr uint32_t elements_per_tile = TILE_HEIGHT * TILE_WIDTH;
    constexpr uint32_t dram_buffer_size = elements_per_tile * num_tiles * sizeof(InputStruct);
    constexpr uint32_t output_buffer_size = elements_per_tile * num_tiles * sizeof(OutputStruct);

    log_info(tt::LogTest, "=== TT-Metal Arbitrary Data Structure Communication Demo ===");
    log_info(
        tt::LogTest,
        "Setting up buffers: {} tiles, {} bytes input, {} bytes output",
        num_tiles,
        dram_buffer_size,
        output_buffer_size);
    log_info(
        tt::LogTest,
        "InputStruct size: {} bytes, OutputStruct size: {} bytes",
        sizeof(InputStruct),
        sizeof(OutputStruct));

    // DRAM buffers for host I/O
    InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size / num_tiles,
        .buffer_type = BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> input_dram_buffer = CreateBuffer(input_dram_config);

    InterleavedBufferConfig output_dram_config{
        .device = device,
        .size = output_buffer_size,
        .page_size = output_buffer_size / num_tiles,
        .buffer_type = BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = CreateBuffer(output_dram_config);

    // Host → Device: Generate test data with custom structures
    std::vector<InputStruct> input_vec(elements_per_tile * num_tiles);

    // Create specific test patterns for better verification
    for (size_t i = 0; i < input_vec.size(); ++i) {
        input_vec[i].float_value = static_cast<float>(i % 100) / 10.0f - 5.0f;  // -5.0 to +4.9
        input_vec[i].int_value = static_cast<int32_t>(i);
        input_vec[i].short_value = static_cast<uint16_t>(i % 65535);
        input_vec[i].char_value = static_cast<int8_t>(i % 127);
        // Initialize padding to zero for consistency
        std::memset(input_vec[i].padding, 0, sizeof(input_vec[i].padding));
    }

    log_info(
        tt::LogTest,
        "Input sample[0]: float={:.2f}, int={}, short={}, char={}",
        input_vec[0].float_value,
        input_vec[0].int_value,
        input_vec[0].short_value,
        static_cast<int>(input_vec[0].char_value));

    log_info(
        tt::LogTest,
        "Input sample[50]: float={:.2f}, int={}, short={}, char={}",
        input_vec[50].float_value,
        input_vec[50].int_value,
        input_vec[50].short_value,
        static_cast<int>(input_vec[50].char_value));

    // Write input structures to device DRAM
    EnqueueWriteBuffer(cq, *input_dram_buffer, input_vec, false);

    // Circular Buffers setup - using raw bytes for arbitrary data structures
    constexpr uint32_t cb_page_size = elements_per_tile * sizeof(InputStruct);
    constexpr uint32_t cb_out_page_size = elements_per_tile * sizeof(OutputStruct);

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(cb_page_size * 2, {{0, tt::DataFormat::UInt8}}).set_page_size(0, cb_page_size);
    CBHandle cb_in = CreateCircularBuffer(program, CoreRange(CoreCoord(0, 0)), cb_in_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(cb_out_page_size * 2, {{16, tt::DataFormat::UInt8}}).set_page_size(16, cb_out_page_size);
    CBHandle cb_out = CreateCircularBuffer(program, CoreRange(CoreCoord(0, 0)), cb_out_config);

    // Kernels on single core {0,0}
    CoreCoord core = {0, 0};

    // Reader kernel: DRAM → cb_in
    KernelHandle reader_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_struct_communication/kernels/dataflow/reader_struct.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Compute kernel: cb_in → process → cb_out
    KernelHandle compute_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_struct_communication/kernels/compute/process_struct.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

    // Writer kernel: cb_out → DRAM
    KernelHandle writer_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_struct_communication/kernels/dataflow/writer_struct.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Set runtime args
    SetRuntimeArgs(program, reader_id, core, {input_dram_buffer->address(), num_tiles, sizeof(InputStruct)});
    SetRuntimeArgs(program, compute_id, core, {num_tiles, sizeof(InputStruct), sizeof(OutputStruct)});
    SetRuntimeArgs(program, writer_id, core, {output_dram_buffer->address(), num_tiles, sizeof(OutputStruct)});

    log_info(tt::LogTest, "Launching pipeline (operation: struct transformation)...");

    // Launch and sync
    EnqueueProgram(cq, program, false);
    Finish(cq);

    log_info(tt::LogTest, "Kernels completed, reading and verifying results...");

    // Device → Host: Read output and verify computation
    std::vector<OutputStruct> result_vec;
    EnqueueReadBuffer(cq, *output_dram_buffer, result_vec, true);

    log_info(
        tt::LogTest,
        "Output sample[0]: result={:.2f}, status={}, iterations={}, flags=0x{:02x}",
        result_vec[0].result_value,
        result_vec[0].status_code,
        result_vec[0].iteration_count,
        result_vec[0].flags);

    log_info(
        tt::LogTest,
        "Output sample[50]: result={:.2f}, status={}, iterations={}, flags=0x{:02x}",
        result_vec[50].result_value,
        result_vec[50].status_code,
        result_vec[50].iteration_count,
        result_vec[50].flags);

    // Detailed pipeline verification
    uint32_t mismatches = 0;
    uint32_t perfect_matches = 0;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    const uint32_t max_mismatches_to_show = 10;

    for (size_t i = 0; i < input_vec.size(); ++i) {
        // Expected transformation: result = float_value * 2.0f, status = int_value,
        // iterations = short_value % 100, flags = char_value & 0xFF
        float input_val = input_vec[i].float_value;
        float output_val = result_vec[i].result_value;
        float expected_val = input_val * 2.0f;  // Simple transformation for verification
        float error = std::abs(output_val - expected_val);

        avg_error += error;
        max_error = std::max(max_error, error);

        // Verify all fields with appropriate tolerances
        bool float_match = error <= 1e-3f;
        bool int_match = result_vec[i].status_code == input_vec[i].int_value;
        bool short_match = result_vec[i].iteration_count == (input_vec[i].short_value % 100);
        bool char_match = result_vec[i].flags == (input_vec[i].char_value & 0xFF);

        bool all_match = float_match && int_match && short_match && char_match;

        if (!all_match) {
            if (mismatches < max_mismatches_to_show) {
                log_info(
                    tt::LogTest,
                    "Mismatch[{}]: input={:.6f}/{}/{}/{}, output={:.6f}/{}/{}/0x{:02x}",
                    i,
                    input_val,
                    input_vec[i].int_value,
                    input_vec[i].short_value,
                    static_cast<int>(input_vec[i].char_value),
                    output_val,
                    result_vec[i].status_code,
                    result_vec[i].iteration_count,
                    result_vec[i].flags);
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
    log_info(tt::LogTest, "Input data processed: {} KB", (input_vec.size() * sizeof(InputStruct)) / 1024);
    log_info(tt::LogTest, "Output data generated: {} KB", (result_vec.size() * sizeof(OutputStruct)) / 1024);
    log_info(tt::LogTest, "Tiles processed: {}", num_tiles);
    log_info(tt::LogTest, "Elements per tile: {}", elements_per_tile);
    log_info(tt::LogTest, "Operation: struct transformation (arbitrary data processing)");

    // Test specific edge cases
    log_info(tt::LogTest, "=== EDGE CASE VERIFICATION ===");
    bool edge_cases_pass = true;

    // Find and verify some specific test cases
    for (size_t i = 0; i < input_vec.size() && i < 200; ++i) {
        if (std::abs(input_vec[i].float_value - 0.0f) < 1e-6f) {
            // Test: 0.0 → 0.0 (after multiplication by 2.0)
            if (std::abs(result_vec[i].result_value - 0.0f) > 1e-3f) {
                log_info(tt::LogTest, "FAIL: Zero test: 0.0 → {:.6f} (expected 0.0)", result_vec[i].result_value);
                edge_cases_pass = false;
            }
        }
        if (input_vec[i].int_value == 0) {
            // Test: int 0 → status 0
            if (result_vec[i].status_code != 0) {
                log_info(tt::LogTest, "FAIL: Zero int test: 0 → {} (expected 0)", result_vec[i].status_code);
                edge_cases_pass = false;
            }
        }
    }

    if (edge_cases_pass) {
        log_info(tt::LogTest, "PASS: Edge case verification (zero values preserved)");
    }

    pass = pass && edge_cases_pass;

    log_info(tt::LogTest, "=== FINAL RESULT ===");
    log_info(tt::LogTest, "Overall verification: {}", pass ? "PASS ✓" : "FAIL ✗");

    if (pass) {
        log_info(tt::LogTest, "🎉 SUCCESS: TT-Metal arbitrary data structure pipeline verified!");
        log_info(tt::LogTest, "✓ Data integrity maintained through full pipeline");
        log_info(tt::LogTest, "✓ Reader kernel: DRAM → L1 CB transfer working with custom structures");
        log_info(tt::LogTest, "✓ Compute kernel: Structure transformation working");
        log_info(tt::LogTest, "✓ Writer kernel: L1 CB → DRAM transfer working with different structures");
        log_info(tt::LogTest, "✓ Pipeline ready for arbitrary data processing workloads");
        log_info(tt::LogTest, "");
        log_info(tt::LogTest, "🚀 Ready for next steps:");
        log_info(tt::LogTest, "   • Add more complex structure transformations");
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
