#include "ops/ring_footprint.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

namespace infernix::test {
namespace {

// One warp per row: whether some and whether every element of its K and of its V row differ.
__global__ void ring_row_flags_kernel(const std::uint16_t* k, const std::uint16_t* v, int head_dim, std::int64_t rows,
                                      std::uint16_t sentinel, std::uint8_t* flags) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) { return; }
    const int lane = static_cast<int>(threadIdx.x % 32);
    bool key_some = false, key_all = true, value_some = false, value_all = true;
    for (int dim = lane; dim < head_dim; dim += 32) {
        const std::int64_t index = row * head_dim + dim;
        const bool key_changed   = k[index] != sentinel;
        const bool value_changed = v[index] != sentinel;
        key_some |= key_changed;
        key_all &= key_changed;
        value_some |= value_changed;
        value_all &= value_changed;
    }
    constexpr unsigned kWarp = 0xffffffffU;
    key_some   = __any_sync(kWarp, key_some);
    key_all    = __all_sync(kWarp, key_all);
    value_some = __any_sync(kWarp, value_some);
    value_all  = __all_sync(kWarp, value_all);
    if (lane == 0) {
        flags[row] = static_cast<std::uint8_t>((key_some ? kRingKeySome : 0) | (key_all ? kRingKeyAll : 0) |
                                               (value_some ? kRingValueSome : 0) | (value_all ? kRingValueAll : 0));
    }
}

__global__ void ring_rows_kernel(const std::uint16_t* ring, int head_dim, const std::int64_t* rows, int count,
                                 std::uint16_t* out) {
    const int item = blockIdx.x;
    if (item >= count) { return; }
    for (int dim = static_cast<int>(threadIdx.x); dim < head_dim; dim += static_cast<int>(blockDim.x)) {
        out[static_cast<std::int64_t>(item) * head_dim + dim] = ring[rows[item] * head_dim + dim];
    }
}

} // namespace

std::vector<std::uint8_t> ring_row_flags(const void* k, const void* v, int head_dim, std::int64_t rows,
                                         std::uint16_t sentinel) {
    std::vector<std::uint8_t> flags(static_cast<std::size_t>(rows));
    if (rows == 0) { return flags; }
    DeviceBuffer device(flags.size());
    constexpr int kRowsPerBlock = 8;
    ring_row_flags_kernel<<<static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock), 32 * kRowsPerBlock>>>(
        static_cast<const std::uint16_t*>(k), static_cast<const std::uint16_t*>(v), head_dim, rows, sentinel,
        static_cast<std::uint8_t*>(device.p));
    cuda_check_last_launch("ring_row_flags");
    device.copy_to_host(flags.data(), flags.size());
    return flags;
}

std::vector<std::uint16_t> ring_rows(const void* ring, int head_dim, const std::vector<std::int64_t>& rows) {
    std::vector<std::uint16_t> out(rows.size() * static_cast<std::size_t>(head_dim));
    if (rows.empty()) { return out; }
    const DeviceBuffer indices = to_device(rows);
    DeviceBuffer device(out.size() * sizeof(std::uint16_t));
    ring_rows_kernel<<<static_cast<unsigned>(rows.size()), 128>>>(static_cast<const std::uint16_t*>(ring), head_dim,
                                                                 static_cast<const std::int64_t*>(indices.p),
                                                                 static_cast<int>(rows.size()),
                                                                 static_cast<std::uint16_t*>(device.p));
    cuda_check_last_launch("ring_rows");
    device.copy_to_host(out.data(), out.size() * sizeof(std::uint16_t));
    return out;
}

} // namespace infernix::test
