#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_values = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    cb_wait_front(cb_out, 1);
    uint32_t cb_addr = get_read_ptr(cb_out);

    // Write to DRAM
    uint64_t noc_addr = get_noc_addr(dst_addr, 0, true);
    noc_async_write(cb_addr, noc_addr, num_values * sizeof(uint32_t));
    noc_async_write_barrier();

    cb_pop_front(cb_out, 1);
}
