#pragma once

// Folded gate/up GEMM on the row-split staging ring (ops/common/rowsplit_ring.cuh): the tile, the row mapping and
// the MMAs of q4_linear_swiglu_mma_split_half_pair_kernel (q4_linear_swiglu_gemm_mma.cuh), with the staging and
// the Q4 decode on producer warps. A 64-row warp tile is 32 gate rows followed by their 32 up rows, which pair in
// the SiLU epilogue; the weights decode exactly as Q4MmaDecodeAtom::decode_pair decodes them, so the bits are the
// plain kernel's.

#include "ops/common/math.cuh"
#include "ops/common/rowsplit_ring.cuh"
#include "ops/linear/q4/q4_ring_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

template <class Cfg, int Stages, int ProducerWarps, bool FullTiles>
__global__ __launch_bounds__(RowSplitRing<Cfg, Stages, ProducerWarps>::kThreads, Cfg::MIN_BLOCKS)
void q4_linear_swiglu_ring_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
                                  std::int32_t intermediate, std::int32_t k, std::int32_t t, std::int32_t padded_k) {
    using Ring       = RowSplitRing<Cfg, Stages, ProducerWarps>;
    constexpr int BN = Cfg::BN, WN = Cfg::WN, MT = Cfg::MT, NT = Cfg::NT, PM = Cfg::BM / 2;
    static_assert(Cfg::BM % 64 == 0 && Cfg::WM == 64 && MT == 4, "folded gate/up mapping requires 64-row warp tiles");
    Ring::init();

    const int NKT = padded_k / Cfg::BK;
    const int m0  = static_cast<int>(blockIdx.x) * PM;
    const int t0  = static_cast<int>(blockIdx.y) * BN;

    if (Ring::producer()) {
        // Local row -> its 64-row tile, then gate (first 32) or up (last 32) row of the tile's 32 outputs.
        const auto row_of = [&](int row) -> std::int64_t {
            const int output = m0 + (row >> 6) * 32 + (row & 31);
            if (!FullTiles && output >= intermediate) return -1;
            return output + static_cast<std::int64_t>((row >> 5) & 1) * intermediate;
        };
        Ring::produce(Q4RingCodec<decltype(row_of)>{codes, scales, padded_k >> 6, row_of}, x, k, t0,
                      FullTiles ? BN : min(BN, t - t0), NKT);
        return;
    }

    float acc[MT][NT][4] = {};
    Ring::consume(acc, NKT);

    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int wm = warp / Cfg::WARPS_N, wn = warp % Cfg::WARPS_N;
    const int gid = lane >> 2, lid = lane & 3;
#pragma unroll
    for (int mi = 0; mi < MT / 2; ++mi) {
        const int r0 = m0 + wm * 32 + mi * 16 + gid;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int cc0 = t0 + wn * WN + ni * 8 + 2 * lid;
            const int cc1 = cc0 + 1;
            auto store    = [&](int col, int row, float gv, float uv) {
                if (FullTiles || (row < intermediate && col < t)) {
                    out[static_cast<std::int64_t>(col) * intermediate + row] = __float2bfloat16_rn(silu(gv) * uv);
                }
            };
            store(cc0, r0, acc[mi][ni][0], acc[mi + MT / 2][ni][0]);
            store(cc1, r0, acc[mi][ni][1], acc[mi + MT / 2][ni][1]);
            store(cc0, r1, acc[mi][ni][2], acc[mi + MT / 2][ni][2]);
            store(cc1, r1, acc[mi][ni][3], acc[mi + MT / 2][ni][3]);
        }
    }
}

} // namespace infernix::ops::detail
