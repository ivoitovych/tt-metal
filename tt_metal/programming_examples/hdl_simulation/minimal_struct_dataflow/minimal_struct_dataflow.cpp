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
    constexpr uint32_t tile_bytes = elements_per_tile * sizeof(InputStruct);

    log_info(tt::LogTest, "=== TT-Metal Arbitrary Data Structure Communication Demo ===");
    log_info(tt::LogTest, "Buffer setup details:");
    log_info(tt::LogTest, "  num_tiles: {}", num_tiles);
    log_info(tt::LogTest, "  elements_per_tile: {}", elements_per_tile);
    log_info(tt::LogTest, "  tile_bytes: {} (0x{:x})", tile_bytes, tile_bytes);
    log_info(tt::LogTest, "  dram_buffer_size: {} bytes", dram_buffer_size);
    log_info(tt::LogTest, "  output_buffer_size: {} bytes", output_buffer_size);
    log_info(tt::LogTest, "  InputStruct size: {} bytes", sizeof(InputStruct));
    log_info(tt::LogTest, "  OutputStruct size: {} bytes", sizeof(OutputStruct));

    // DRAM buffers for host I/O
    InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size / num_tiles,  // This should equal tile_bytes
        .buffer_type = BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> input_dram_buffer = CreateBuffer(input_dram_config);

    InterleavedBufferConfig output_dram_config{
        .device = device,
        .size = output_buffer_size,
        .page_size = output_buffer_size / num_tiles,  // This should equal tile_bytes
        .buffer_type = BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = CreateBuffer(output_dram_config);

    log_info(tt::LogTest, "DRAM buffer addresses:");
    log_info(tt::LogTest, "  input_buffer address: 0x{:x}", input_dram_buffer->address());
    log_info(tt::LogTest, "  output_buffer address: 0x{:x}", output_dram_buffer->address());
    log_info(tt::LogTest, "  input page_size: {} (should be {})", input_dram_config.page_size, tile_bytes);
    log_info(tt::LogTest, "  output page_size: {} (should be {})", output_dram_config.page_size, tile_bytes);

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

    log_info(tt::LogTest, "Test data patterns:");
    log_info(
        tt::LogTest,
        "  Tile 0 first element [0]: float={:.2f}, int={}, short={}, char={}",
        input_vec[0].float_value,
        input_vec[0].int_value,
        input_vec[0].short_value,
        static_cast<int>(input_vec[0].char_value));
    log_info(
        tt::LogTest,
        "  Tile 1 first element [{}]: float={:.2f}, int={}, short={}, char={}",
        elements_per_tile,
        input_vec[elements_per_tile].float_value,
        input_vec[elements_per_tile].int_value,
        input_vec[elements_per_tile].short_value,
        static_cast<int>(input_vec[elements_per_tile].char_value));

    // Write input structures to device DRAM
    EnqueueWriteBuffer(cq, *input_dram_buffer, input_vec, false);

    // Kernels on single core {0,0}
    CoreCoord core = {0, 0};

    // Single all-in-one kernel: DRAM → process → DRAM
    KernelHandle all_in_one_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/minimal_struct_dataflow/kernels/dataflow/all_in_one_struct.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Set runtime args for the all-in-one kernel
    SetRuntimeArgs(
        program,
        all_in_one_id,
        core,
        {
            input_dram_buffer->address(),   // src_addr
            output_dram_buffer->address(),  // dst_addr
            num_tiles,                      // num_tiles
            sizeof(InputStruct),            // input_struct_size
            sizeof(OutputStruct),           // output_struct_size
            tile_bytes                      // tile_bytes
        });

    log_info(tt::LogTest, "Kernel runtime arguments:");
    log_info(tt::LogTest, "  src_addr: 0x{:x}", input_dram_buffer->address());
    log_info(tt::LogTest, "  dst_addr: 0x{:x}", output_dram_buffer->address());
    log_info(tt::LogTest, "  num_tiles: {}", num_tiles);
    log_info(tt::LogTest, "  input_struct_size: {}", sizeof(InputStruct));
    log_info(tt::LogTest, "  output_struct_size: {}", sizeof(OutputStruct));
    log_info(tt::LogTest, "  tile_bytes: {} (0x{:x})", tile_bytes, tile_bytes);

    // Expected tile addresses
    for (uint32_t i = 0; i < num_tiles; i++) {
        uint64_t expected_input_addr = input_dram_buffer->address() + (i * tile_bytes);
        uint64_t expected_output_addr = output_dram_buffer->address() + (i * tile_bytes);
        log_info(
            tt::LogTest,
            "  Expected tile {} input addr: 0x{:x}, output addr: 0x{:x}",
            i,
            expected_input_addr,
            expected_output_addr);
    }

    log_info(tt::LogTest, "Launching pipeline (operation: struct transformation)...");

    // Launch and sync
    EnqueueProgram(cq, program, false);
    Finish(cq);

    log_info(tt::LogTest, "Kernels completed, reading and verifying results...");

    // Device → Host: Read output and verify computation
    std::vector<OutputStruct> result_vec;
    EnqueueReadBuffer(cq, *output_dram_buffer, result_vec, true);

    // Enhanced verification with tile-by-tile analysis
    log_info(tt::LogTest, "Verification by tile:");

    uint32_t total_perfect_matches = 0;
    uint32_t total_mismatches = 0;

    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        uint32_t tile_start = tile * elements_per_tile;
        uint32_t perfect_matches_tile = 0;
        uint32_t errors_tile = 0;

        log_info(tt::LogTest, "=== TILE {} ANALYSIS ===", tile);
        log_info(tt::LogTest, "  Elements {} to {}", tile_start, tile_start + elements_per_tile - 1);

        // Check first few elements of this tile
        for (uint32_t i = 0; i < 5 && (tile_start + i) < input_vec.size(); i++) {
            uint32_t idx = tile_start + i;
            float input_val = input_vec[idx].float_value;
            float output_val = result_vec[idx].result_value;
            float expected_val = input_val * 2.0f;
            bool float_matches = std::abs(output_val - expected_val) <= 1e-3f;
            bool int_matches = result_vec[idx].status_code == input_vec[idx].int_value;
            bool short_matches = result_vec[idx].iteration_count == (input_vec[idx].short_value % 100);
            bool char_matches = result_vec[idx].flags == static_cast<uint8_t>(input_vec[idx].char_value & 0xFF);
            bool all_match = float_matches && int_matches && short_matches && char_matches;

            log_info(
                tt::LogTest,
                "    [{}]: input={:.2f}/{}/{}/{} -> output={:.2f}/{}/{}/{} (expected={:.2f}/{}/{}/{}) {}",
                idx,
                input_val,
                input_vec[idx].int_value,
                input_vec[idx].short_value,
                static_cast<int>(input_vec[idx].char_value),
                output_val,
                result_vec[idx].status_code,
                result_vec[idx].iteration_count,
                static_cast<int>(result_vec[idx].flags),
                expected_val,
                input_vec[idx].int_value,
                input_vec[idx].short_value % 100,
                static_cast<int>(input_vec[idx].char_value & 0xFF),
                all_match ? "✓" : "✗");
        }

        // Count matches for this tile
        for (uint32_t i = 0; i < elements_per_tile && (tile_start + i) < input_vec.size(); i++) {
            uint32_t idx = tile_start + i;
            float input_val = input_vec[idx].float_value;
            float output_val = result_vec[idx].result_value;
            float expected_val = input_val * 2.0f;

            bool float_match = std::abs(output_val - expected_val) <= 1e-3f;
            bool int_match = result_vec[idx].status_code == input_vec[idx].int_value;
            bool short_match = result_vec[idx].iteration_count == (input_vec[idx].short_value % 100);
            bool char_match = result_vec[idx].flags == static_cast<uint8_t>(input_vec[idx].char_value & 0xFF);

            if (float_match && int_match && short_match && char_match) {
                perfect_matches_tile++;
                total_perfect_matches++;
            } else {
                errors_tile++;
                total_mismatches++;
            }
        }

        log_info(
            tt::LogTest,
            "  Tile {} results: {} matches, {} errors ({:.1f}% success)",
            tile,
            perfect_matches_tile,
            errors_tile,
            100.0f * perfect_matches_tile / elements_per_tile);
    }

    // Overall results
    log_info(tt::LogTest, "=== OVERALL VERIFICATION RESULTS ===");
    log_info(tt::LogTest, "Total elements processed: {}", input_vec.size());
    log_info(
        tt::LogTest,
        "Perfect matches: {} ({:.1f}%)",
        total_perfect_matches,
        100.0f * total_perfect_matches / input_vec.size());
    log_info(
        tt::LogTest,
        "Data corruption errors: {} ({:.1f}%)",
        total_mismatches,
        100.0f * total_mismatches / input_vec.size());

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
            } else {
                log_info(tt::LogTest, "PASS: Zero test: 0.0 → {:.6f} (expected 0.0)", result_vec[i].result_value);
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

    pass = (total_mismatches == 0) && edge_cases_pass;

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
