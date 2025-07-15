#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in = tt::CBIndex::c_0;
    uint32_t tile_bytes = get_tile_size(cb_id_in);

    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, src_addr);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in);

    cb_reserve_back(cb_id_in, 1);
    noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in, 1);
}
