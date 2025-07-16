#include <tt_metal/hw/inc/data_movement.h>
#include <tt_metal/hw/inc/trace.h>

void kernel_main() {
    uint32_t l1_addr = get_arg_val<uint32_t>(0);   // L1 source address
    uint32_t dst_addr = get_arg_val<uint32_t>(1);  // DRAM destination address
    uint32_t size = get_arg_val<uint32_t>(2);      // Size to transfer

    DPRINT << "Send Kernel: Writing " << size << " bytes from L1 0x" << l1_addr << " to DRAM 0x" << dst_addr << "\n";

    noc_async_write(l1_addr, dst_addr, size);
    noc_async_write_barrier();  // Ensure write completes

    DPRINT << "Send Kernel: Write completed\n";
}
