#pragma once

// Folded gate/up GEMM on the row-split staging ring (ops/common/rowsplit_ring.cuh): the tile, the row mapping and
// the MMAs of q4_linear_swiglu_mma_split_half_pair_kernel (q4_linear_swiglu_gemm_mma.cuh), with the staging and
// the Q4 decode on producer warps. A 64-row warp tile is 32 gate rows followed by their 32 up rows, which pair in
// the SiLU epilogue; the weights decode exactly as Q4MmaDecodeAtom::decode_pair decodes them, so the bits are the
// plain kernel's.

#include "ops/common/math.cuh"
#include "ops/common/rowsplit_ring.cuh"
#include "ops/linear/q4/q4_ring_codec.cuh"
#include "ops/lora/lora_device.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

// `lora`: LoraNone, or the term over the gate/up parent's rows (gate rows [0, intermediate), up rows after) and the
// launch's columns (lora_device.cuh).
template <class Cfg, int Stages, int ProducerWarps, bool FullTiles, class Lora = LoraNone>
__global__ __launch_bounds__(RowSplitRing<Cfg, Stages, ProducerWarps>::kThreads, Cfg::MIN_BLOCKS)
void q4_linear_swiglu_ring_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
                                  std::int32_t intermediate, std::int32_t k, std::int32_t t, std::int32_t padded_k,
                                  Lora lora) {
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

    // --lora: a fragment mm of warp row wm holds outputs m0 + 32 wm + 16 (mm & 1) + i, gate (mm < 2) or up rows.
    const auto lora_row = [m0, intermediate](int wm, int mm, int i) {
        const int output = m0 + wm * 32 + (mm & 1) * 16 + i;
        return (output < intermediate ? output : intermediate - 1) + (mm >> 1) * intermediate;
    };
    (void)lora_row;
    if constexpr (kLoraTerm<Lora>) {
        // The term's B rows and U columns into L2 while the ring runs (consumer threads).
        lora_prefetch_tile(
            lora, Cfg::BM, [&](int i) { return lora_row(i >> 6, (i & 63) >> 4, i & 15); }, t0,
            FullTiles ? BN : min(BN, t - t0), static_cast<int>(threadIdx.x), Cfg::THREADS);
    }
    float acc[MT][NT][4] = {};
    Ring::consume(acc, NKT);

    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int wm = warp / Cfg::WARPS_N, wn = warp % Cfg::WARPS_N;
    const int gid = lane >> 2, lid = lane & 3;
    if constexpr (kLoraTerm<Lora>) {
        // --lora: the term on the gate and up fragments (canonical chains), before the SwiGLU.
        if constexpr ((Cfg::BM + BN) * (2 * Lora::kMaxRank + 16) <= Ring::kBarrierOffset) {
            // B's rows and U's columns staged in the free ring (every consumer warp busy each pass). The tile's local
            // row 64 wm + 16 mm + i is fragment mm's row i of warp row wm.
            lora_add_rows_tile_staged<Cfg::BM, BN, Ring::kBarrierOffset>(
                lora, acc, [&](int mm, int i) { return wm * 64 + mm * 16 + i; },
                [&](int mn, int j) { return wn * WN + mn * 8 + j; },
                [&](int i) { return lora_row(i >> 6, (i & 63) >> 4, i & 15); }, t0, t, Ring::storage(), tid,
                Cfg::THREADS, [] { asm volatile("bar.sync 1, %0;" ::"r"(Cfg::THREADS) : "memory"); });
        } else {
            lora_add_rows_tile<MT, NT>(
                lora, acc, [&](int mm, int i) { return lora_row(wm, mm, i); },
                [&](int mn, int j) {
                    const int c = t0 + wn * WN + mn * 8 + j;
                    return c < t ? c : -1;
                });
        }
    }
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
