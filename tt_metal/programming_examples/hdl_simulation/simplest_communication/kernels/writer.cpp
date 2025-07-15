#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    uint32_t tile_bytes = get_tile_size(cb_id_out);

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_addr);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);

    cb_wait_front(cb_id_out, 1);
    noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out, 1);
}
