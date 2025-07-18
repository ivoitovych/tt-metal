#include <cstdint>
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_values = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);

    // Wait for input
    cb_wait_front(cb_in, 1);
    uint32_t* in_data = (uint32_t*)get_read_ptr(cb_in);

    // Reserve output
    cb_reserve_back(cb_out, 1);
    uint32_t* out_data = (uint32_t*)get_write_ptr(cb_out);

    // Process
    for (uint32_t i = 0; i < num_values; ++i) {
        out_data[i] = in_data[i] * 2;
    }

    cb_pop_front(cb_in, 1);
    cb_push_back(cb_out, 1);
}
}  // namespace NAMESPACE
