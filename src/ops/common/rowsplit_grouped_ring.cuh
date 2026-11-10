#pragma once

// The grouped Q4 / Q5 row-split GEMM (rowsplit_grouped_mma.cuh: two or four weight matrices over one activation,
// each CTA one row tile of one job) on the staging ring (rowsplit_ring.cuh). The jobs, the tile, the MMAs and the
// BF16 weights (Q4MmaDecodeAtom / Q5MmaDecodeAtom's decode) are the barrier-synchronised kernel's, so the bits are
// too.

#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/rowsplit_ring.cuh"
#include "ops/linear/q4/q4_ring_codec.cuh"
#include "ops/linear/q5/q5_ring_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

// One job's weights: Q4, or Q5 with its high-bit plane (the job says which).
template <class RowMap>
struct RowSplitGroupedRingCodec {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    int groups_per_row;
    bool q5;
    RowMap row_of;

    using Raw = typename Q5RingCodec<RowMap>::Raw;

    __device__ __forceinline__ Raw load(int row, int k_tile, int half) const {
        Raw raw{make_uint4(0U, 0U, 0U, 0U), 0U, 0.0f};
        const std::int64_t global = row_of(row);
        if (global >= 0) {
            const std::int64_t group = global * groups_per_row + k_tile;
            raw.codes                = load_vec<uint4>(codes + group * 32 + half * 16);
            if (q5) { raw.high = *reinterpret_cast<const std::uint32_t*>(high + group * 8 + half * 4); }
            raw.scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group * 2)));
        }
        return raw;
    }

    __device__ __forceinline__ void decode(const Raw& raw, unsigned (&pairs)[16]) const {
        const unsigned words[4] = {raw.codes.x, raw.codes.y, raw.codes.z, raw.codes.w};
        // Q4: the code is the nibble's four-bit two's complement; Q5: the nibble with its plane bit, five bits.
        const int sign = q5 ? 0x10 : 0x08;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned packed = (words[i >> 2] >> (8 * (i & 3))) & 0xffU;
            const int q0 =
                ((static_cast<int>(packed & 0x0fU) | static_cast<int>(((raw.high >> (2 * i)) & 1U) << 4)) ^ sign) - sign;
            const int q1 =
                ((static_cast<int>(packed >> 4) | static_cast<int>(((raw.high >> (2 * i + 1)) & 1U) << 4)) ^ sign) - sign;
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * raw.scale,
                                                               static_cast<float>(q1) * raw.scale);
            pairs[i]                   = *reinterpret_cast<const unsigned*>(&value);
        }
    }
};

template <class Tile, int Stages, int ProducerWarps, bool FullTiles, int Jobs = 4>
__global__ __launch_bounds__(RowSplitRing<Tile, Stages, ProducerWarps>::kThreads, Tile::MIN_BLOCKS)
void rowsplit_grouped_ring_kernel(const __nv_bfloat16* __restrict__ x, RowSplitGroupedMmaJob job0,
                                  RowSplitGroupedMmaJob job1, RowSplitGroupedMmaJob job2, RowSplitGroupedMmaJob job3,
                                  std::int32_t k, std::int32_t t, std::int32_t padded_k) {
    using Ring = RowSplitRing<Tile, Stages, ProducerWarps>;
    constexpr int BM = Tile::BM, BN = Tile::BN, MT = Tile::MT, NT = Tile::NT;
    static_assert(Jobs == 2 || Jobs == 4, "grouped input GEMM supports two or four jobs");
    Ring::init();

    const int tiles0 = div_up(job0.n, BM);
    int tile         = static_cast<int>(blockIdx.x);
    RowSplitGroupedMmaJob job;
    if constexpr (Jobs == 2) {
        if (tile < tiles0) {
            job = job0;
        } else {
            tile -= tiles0;
            job = job1;
        }
    } else {
        const int tiles1 = div_up(job1.n, BM);
        const int tiles2 = div_up(job2.n, BM);
        if (tile < tiles0) {
            job = job0;
        } else if ((tile -= tiles0) < tiles1) {
            job = job1;
        } else if ((tile -= tiles1) < tiles2) {
            job = job2;
        } else {
            tile -= tiles2;
            job = job3;
        }
    }
    const int NKT = padded_k / Tile::BK;
    const int m0  = tile * BM;
    const int t0  = static_cast<int>(blockIdx.y) * BN;

    if (Ring::producer()) {
        const auto row_of = [&](int row) -> std::int64_t { return FullTiles || m0 + row < job.n ? m0 + row : -1; };
        // A Q4 job has no plane: its high bits read as zero.
        Ring::produce(RowSplitGroupedRingCodec<decltype(row_of)>{job.codes, job.high, job.scales, padded_k >> 6, job.q5,
                                                                 row_of},
                      x, k, t0, FullTiles ? BN : min(BN, t - t0), NKT);
        return;
    }

    float acc[MT][NT][4] = {};
    Ring::consume(acc, NKT);

    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int wm = warp / Tile::WARPS_N, wn = warp % Tile::WARPS_N;
    const int gid = lane >> 2, lid = lane & 3;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int r0 = m0 + wm * Tile::WM + mi * 16 + gid;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int cc0 = t0 + wn * Tile::WN + ni * 8 + 2 * lid;
            const int cc1 = cc0 + 1;
            auto store    = [&](int col, int row, float value) {
                if (FullTiles || (row < job.n && col < t)) {
                    job.out[static_cast<std::int64_t>(col) * job.out_ld + job.out_row_offset + row] =
                        __float2bfloat16_rn(value);
                }
            };
            store(cc0, r0, acc[mi][ni][0]);
            store(cc1, r0, acc[mi][ni][1]);
            store(cc0, r1, acc[mi][ni][2]);
            store(cc1, r1, acc[mi][ni][3]);
        }
    }
}

} // namespace infernix::ops::detail
