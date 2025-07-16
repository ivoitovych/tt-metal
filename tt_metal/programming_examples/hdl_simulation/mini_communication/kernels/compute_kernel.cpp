#include <tt_metal/hw/inc/compute.h>
#include <tt_metal/hw/inc/trace.h>

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);   // Input L1 address
    uint32_t output_addr = get_arg_val<uint32_t>(1);  // Output L1 address
    uint32_t size = get_arg_val<uint32_t>(2);         // Size in bytes

    DPRINT << "Compute Kernel: Processing " << size << " bytes from L1 0x" << input_addr << " to L1 0x" << output_addr
           << "\n";

    // Process BFloat16 data element-wise
    uint32_t num_elements = size / sizeof(uint16_t);
    for (uint32_t i = 0; i < num_elements; ++i) {
        uint16_t* input_ptr = (uint16_t*)(input_addr + i * sizeof(uint16_t));
        uint16_t* output_ptr = (uint16_t*)(output_addr + i * sizeof(uint16_t));
        *output_ptr = *input_ptr + 0x3F80;  // Add 1.0 (BFloat16)
    }

    DPRINT << "Compute Kernel: Computation completed\n";
}
