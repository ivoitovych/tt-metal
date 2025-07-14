// simple_communication_example.cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <random>
#include <iomanip>

using namespace tt;
using namespace tt::tt_metal;

// Test configuration for different block sizes
struct TestConfig {
    uint32_t num_elements;
    uint32_t block_size;
    std::string description;
};

uint32_t calculate_checksum(const std::vector<uint32_t>& data) {
    uint32_t checksum = 0;
    for (auto val : data) {
        checksum ^= val;
    }
    return checksum;
}

void run_simple_communicationulation(
    IDevice* device, const TestConfig& config, uint32_t operation_type, bool verbose = false) {
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Calculate sizes
    const uint32_t element_size = sizeof(uint32_t);
    const uint32_t total_size_bytes = config.num_elements * element_size;
    const uint32_t tile_size_bytes = tt::constants::TILE_HW * sizeof(bfloat16);

    // Calculate padding for NPOT sizes
    uint32_t padded_block_size = (config.block_size + tile_size_bytes - 1) / tile_size_bytes * tile_size_bytes;
    uint32_t num_blocks = (total_size_bytes + config.block_size - 1) / config.block_size;

    std::cout << "\n=== HDL Simulation Test: " << config.description << " ===" << std::endl;
    std::cout << "Elements: " << config.num_elements << ", Block size: " << config.block_size
              << " bytes, Num blocks: " << num_blocks << std::endl;

    // Create buffers
    uint32_t src_buffer_size = align(total_size_bytes, config.block_size);
    InterleavedBufferConfig src_config{
        .device = device, .size = src_buffer_size, .page_size = config.block_size, .buffer_type = BufferType::DRAM};
    auto src_buffer = CreateBuffer(src_config);

    InterleavedBufferConfig dst_config{
        .device = device, .size = src_buffer_size, .page_size = config.block_size, .buffer_type = BufferType::DRAM};
    auto dst_buffer = CreateBuffer(dst_config);

    // Checksum buffer for verification
    InterleavedBufferConfig checksum_config{
        .device = device, .size = sizeof(uint32_t), .page_size = sizeof(uint32_t), .buffer_type = BufferType::DRAM};
    auto checksum_buffer = CreateBuffer(checksum_config);

    // Create test data with pattern
    std::vector<uint32_t> input_data(config.num_elements);
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    for (uint32_t i = 0; i < config.num_elements; i++) {
        // Create a pattern that simulates HDL signals
        input_data[i] = (i << 16) | (i & 0xFFFF);  // Upper 16 bits: counter, lower 16 bits: inverted
    }

    // Core setup
    CoreCoord core = {0, 0};

    // Create circular buffers
    uint32_t cb_tiles = 2;  // Double buffering
    uint32_t cb_size = cb_tiles * tile_size_bytes;

    CircularBufferConfig cb_src_config = CircularBufferConfig(cb_size, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                                             .set_page_size(CBIndex::c_0, tile_size_bytes);
    auto cb_src = CreateCircularBuffer(program, core, cb_src_config);

    CircularBufferConfig cb_dst_config = CircularBufferConfig(cb_size, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                                             .set_page_size(CBIndex::c_16, tile_size_bytes);
    auto cb_dst = CreateCircularBuffer(program, core, cb_dst_config);

    // Create kernels
    std::vector<uint32_t> reader_compile_args = {(uint32_t)(src_buffer->buffer_type() == BufferType::DRAM)};

    std::vector<uint32_t> writer_compile_args = {(uint32_t)(dst_buffer->buffer_type() == BufferType::DRAM)};

    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/simple_communication/kernels/simple_communication_reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_args});

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/simple_communication/kernels/simple_communication_writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_args});

    auto compute_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/simple_communication/kernels/simple_communication_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    // Set runtime args
    SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {src_buffer->address(),
         num_blocks,
         config.block_size,
         total_size_bytes,
         padded_block_size - config.block_size});

    SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {dst_buffer->address(), num_blocks, config.block_size, checksum_buffer->address()});

    SetRuntimeArgs(program, compute_kernel, core, {num_blocks, config.block_size, operation_type});

    // Execute
    EnqueueWriteBuffer(cq, src_buffer, input_data.data(), false);
    EnqueueProgram(cq, program, false);

    // Read results
    std::vector<uint32_t> output_data(config.num_elements);
    EnqueueReadBuffer(cq, dst_buffer, output_data.data(), false);

    std::vector<uint32_t> device_checksum(1);
    EnqueueReadBuffer(cq, checksum_buffer, device_checksum.data(), true);

    // Verify results
    uint32_t host_checksum = calculate_checksum(output_data);
    bool checksum_match = (host_checksum == device_checksum[0]);

    std::cout << "Verification: " << (checksum_match ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Host checksum: 0x" << std::hex << host_checksum << ", Device checksum: 0x" << device_checksum[0]
              << std::dec << std::endl;

    if (verbose && config.num_elements <= 16) {
        std::cout << "Data samples (first 8 elements):" << std::endl;
        for (uint32_t i = 0; i < std::min(8u, config.num_elements); i++) {
            std::cout << "  [" << i << "] In: 0x" << std::hex << input_data[i] << " -> Out: 0x" << output_data[i]
                      << std::dec << std::endl;
        }
    }
}

int main() {
    try {
        // Initialize device
        int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        // Test configurations with different block sizes including NPOT
        std::vector<TestConfig> test_configs = {
            {1024, 64, "Small aligned blocks"},
            {1000, 64, "NPOT total size"},
            {2048, 123, "NPOT block size"},
            {4096, 256, "Large aligned blocks"},
            {100, 17, "Small NPOT blocks"},
            {8192, 512, "Large simulation"}};

        // Test different HDL operations
        std::vector<std::pair<uint32_t, std::string>> operations = {
            {0, "Shift Register"}, {1, "Accumulator"}, {2, "XOR Logic"}};

        // Run tests
        for (const auto& config : test_configs) {
            for (const auto& [op_type, op_name] : operations) {
                std::cout << "\nOperation: " << op_name << std::endl;
                run_simple_communicationulation(device, config, op_type, true);
            }
        }

        // Cleanup
        CloseDevice(device);

        std::cout << "\nAll HDL simulation tests completed!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
