// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Initialize device
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Target single core for simulation
    constexpr CoreCoord core = {0, 0};

    // Create compute kernel for HDL simulation
    std::vector<uint32_t> simulation_args = {
        1000,  // num_cycles to simulate
        42     // seed for simulation
    };

    KernelHandle simulation_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hdl_simulation/basic_simulation/kernels/simulation_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = simulation_args});

    // Set runtime args (could be used for dynamic simulation parameters)
    SetRuntimeArgs(program, simulation_kernel, core, {});

    // Run the simulation
    EnqueueProgram(cq, program, false);

    printf("HDL Simulation started on Core {%zu, %zu}...\n", core.x, core.y);
    printf("Check DPRINT output for simulation results (export TT_METAL_DPRINT_CORES=0,0)\n");

    // Wait for completion
    Finish(cq);

    printf("HDL Simulation completed.\n");

    // Cleanup
    CloseDevice(device);

    return 0;
}
