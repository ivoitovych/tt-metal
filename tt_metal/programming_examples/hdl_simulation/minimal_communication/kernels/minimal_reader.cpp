#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_values = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);

    cb_reserve_back(cb_in, 1);
    uint32_t cb_addr = get_write_ptr(cb_in);

    // Read from DRAM to CB (L1)
    uint64_t noc_addr = get_noc_addr(src_addr, 0, true);
    noc_async_read(noc_addr, cb_addr, num_values * sizeof(uint32_t));
    noc_async_read_barrier();

    cb_push_back(cb_in, 1);
}
