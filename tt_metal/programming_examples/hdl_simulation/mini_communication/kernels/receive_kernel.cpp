#include <tt_metal/hw/inc/data_movement.h>
#include <tt_metal/hw/inc/trace.h>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);  // DRAM source address
    uint32_t l1_addr = get_arg_val<uint32_t>(1);   // L1 destination address
    uint32_t size = get_arg_val<uint32_t>(2);      // Size to transfer

    DPRINT << "Receive Kernel: Reading " << size << " bytes from DRAM 0x" << src_addr << " to L1 0x" << l1_addr << "\n";

    noc_async_read(src_addr, l1_addr, size);
    noc_async_read_barrier();  // Ensure read completes

    DPRINT << "Receive Kernel: Read completed\n";
}
