#pragma once

// Device-side reads of a KV ring [head_dim, rows] of 16-bit elements (row = index / head_dim), so a test
// checks a ring's write footprint without copying the whole ring to the host.

#include <cstdint>
#include <vector>

namespace infernix::test {

// Per row of a K/V ring pair, which of its elements differ from `sentinel`.
enum RingRowFlags : std::uint8_t {
    kRingKeySome   = 1, // some K element differs
    kRingKeyAll    = 2, // every K element differs
    kRingValueSome = 4,
    kRingValueAll  = 8,
};

// The flags of every row of the rings `k` and `v` (device, `rows` x `head_dim` elements each).
std::vector<std::uint8_t> ring_row_flags(const void* k, const void* v, int head_dim, std::int64_t rows,
                                         std::uint16_t sentinel);

// The listed rows of a ring (device), `head_dim` elements each, in list order.
std::vector<std::uint16_t> ring_rows(const void* ring, int head_dim, const std::vector<std::int64_t>& rows);

} // namespace infernix::test
