#pragma once

#include "ops/softmax_attention/dense/causal_cache/int8/schedule.cuh"

namespace infernix::ops::detail {

template <class G, int Tokens>
struct Int8KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles  = (Tokens * G::GroupSize + 15) / 16;
    // Rows of two or more columns of the 24/4 geometry use grouped_pipelined.cuh (in-register V
    // decode, double-buffered K/V, bitwise equal); single-column rows keep the staged kernel.
    static constexpr bool kPipelined = G::QHeads == 24 && Tokens >= 2;
    static constexpr bool kWideKeys = G::GroupSize == 8 && kRowTiles >= 3;
    using Schedule                  = Int8KvGroupedMmaSchedule<Tokens,
                                              kWideKeys        ? 4 * kRowTiles
                                                               : kRowTiles <= 2 ? 8
                                                                                : 2 * kRowTiles,
                                              kWideKeys ? 64 : 32, kWideKeys ? 1 : 2>;
    using Merge                     = Int8KvMergeSchedule<G::QHeads == 24 ? 256 : 64>;
};

using Int8KvTiledInstance = Int8KvTiledMmaSchedule<64, 64, 128>;

} // namespace infernix::ops::detail
