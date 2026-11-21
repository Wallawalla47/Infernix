#pragma once

// A call column's lane, position and short-conv history rule (include/infernix/ops/inkling2_visit.h), shared by the
// visit-chain and attention kernels.

#include "infernix/ops/inkling2_visit.h"

#include <cstddef>
#include <cstdint>

namespace infernix::ops::inkling2::detail {

struct Column {
    int t, lane, pos, first;
};

__device__ __forceinline__ Column column_of(const ColumnMap& map, int t) {
    Column c{t, map.lane[t], map.position[t], 0};
    if (c.lane >= 0) { c.first = map.first[c.lane]; }
    return c;
}

// The ring row of a lane's position (channel 0).
__device__ __forceinline__ std::uint16_t* ring_row(const Ring& ring, int lane, int pos) {
    return ring.base + lane * ring.lane_stride + static_cast<std::size_t>(pos & (kRingPositions - 1)) * ring.width;
}

// Where a read of a position before the lane's first column in the call finds it: the history snapshot the call
// passes, else the ring.
__device__ __forceinline__ const std::uint16_t* history_row(const Ring& ring, int lane, int pos) {
    if (ring.history != nullptr) {
        return ring.history + lane * ring.history_lane_stride +
               static_cast<std::size_t>(pos & (kHistorySlots - 1)) * ring.width;
    }
    return ring_row(ring, lane, pos);
}

// Whether a column writes its raw input to the ring: it is among its lane's last kRingPositions columns in the call.
__device__ __forceinline__ bool writes_ring(const ColumnMap& map, const Column& col) {
    if (col.lane < 0) { return false; }
    const int later = col.t + kRingPositions;
    return later >= map.columns || map.lane[later] != col.lane;
}

// Where the value k positions back comes from: 0 (padding, or before the lane's first position), the call's column
// t - k (same lane), or the ring.
enum class Source { Zero, Call, Ring };
__device__ __forceinline__ Source history_source(const ColumnMap& map, const Column& col, int k) {
    if (col.lane < 0 || col.pos - k < col.first) { return Source::Zero; }
    if (col.t - k >= 0 && map.lane[col.t - k] == col.lane) { return Source::Call; }
    return Source::Ring;
}

} // namespace infernix::ops::inkling2::detail
