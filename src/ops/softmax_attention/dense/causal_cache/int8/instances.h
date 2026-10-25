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
    // The 16/2 geometry at two to four columns (16-32 rows, where the staged kernel keeps 32-key tiles,
    // so the pipelined kernel's bits are its own): pipelined under the temporary
    // INFERNIX_TMP_INT8_PIPELINED16 (candidates KW-10 (b)). From five columns the staged kernel takes
    // 64-key tiles, whose bits differ.
    static constexpr bool kPipelinable16 = G::QHeads == 16 && Tokens >= 2 && Tokens <= 4;
    // 48 or 64 query rows of the two-KV-head geometries (group 8 at six to eight columns, group 16 at
    // three or four): 64-key tiles, four warps per row tile.
    static constexpr bool kWideKeys = G::GroupSize >= 8 && kRowTiles >= 3;
    using Schedule                  = Int8KvGroupedMmaSchedule<Tokens,
                                              kWideKeys        ? 4 * kRowTiles
                                                               : kRowTiles <= 2 ? 8
                                                                                : 2 * kRowTiles,
                                              kWideKeys ? 64 : 32, kWideKeys ? 1 : 2>;
    using Merge                     = Int8KvMergeSchedule<G::QHeads == 24 ? 256 : 64>;
};

using Int8KvTiledInstance = Int8KvTiledMmaSchedule<64, 64, 128>;

} // namespace infernix::ops::detail
