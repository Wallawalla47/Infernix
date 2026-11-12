#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <cstdint>

namespace infernix::ops::detail {
// Matching gate/up fragments belong to the same warp. The persistent weight
// planes are unchanged; only the CTA's logical load rows are reordered.
template <class Schedule>
struct SwiGluTokenMajorMmaRows {
    static_assert(Schedule::kWarpRows % 16 == 0);
    static constexpr bool kPaired          = true;
    static constexpr bool kWarpPaired      = true;
    static constexpr bool kContiguous      = false;
    static constexpr bool kContiguousPairs = true;

    __device__ __forceinline__ int weight_row(int begin, int local_row, int rows) const {
        constexpr int half_warp = Schedule::kWarpRows / 2;
        const int warp          = local_row / Schedule::kWarpRows;
        const int within        = local_row % Schedule::kWarpRows;
        return begin + warp * half_warp + within % half_warp + (within >= half_warp ? rows / 2 : 0);
    }
};

// The TMA schedules' producer threads, which take no part in the epilogue.
template <class S>
struct SwiGluProducerThreads {
    static constexpr int value = 0;
};
template <class S>
    requires requires { S::kProducerThreads; }
struct SwiGluProducerThreads<S> {
    static constexpr int value = S::kProducerThreads;
};

struct SwiGluTokenMajorMmaEpilogue {
    __device__ __forceinline__ float apply_pair(int, int, float gate, float up) const {
        return silu(gate) * up;
    }

    template <class Schedule>
    static constexpr int kSharedBytes = Schedule::kBlockTokens * (Schedule::kBlockRows / 2 + 8) * 2;

    template <class S>
    static constexpr int kProducers = SwiGluProducerThreads<S>::value;

    template <class S>
    __device__ __forceinline__ static void synchronize() {
        constexpr int producers = kProducers<S>;
        if constexpr (producers)
            asm volatile("bar.sync 1, %0;" ::"r"(S::kThreads - producers) : "memory");
        else
            __syncthreads();
    }

    // Writes silu(gate) * up of the tile, rounded to BF16, to `scratch` as [token][row] with a row stride of
    // kBlockRows / 2 + 8 elements, and synchronizes the consumer threads after it.
    template <class S>
    __device__ __forceinline__ static void stage(unsigned char* scratch,
                                                 float (&acc)[S::kMmaTokens][S::kMmaRows][4]) {
        static_assert(S::kMmaRows % 2 == 0);
        constexpr int stride         = S::kBlockRows / 2 + 8;
        constexpr int gate_fragments = S::kMmaRows / 2;
        const int tid = static_cast<int>(threadIdx.x) - kProducers<S>, warp = tid >> 5, lane = tid & 31;
        const int wt = warp / S::kWarpsRows, wr = warp % S::kWarpsRows;
        auto* final_tile = reinterpret_cast<__nv_bfloat16*>(scratch);
        synchronize<S>();
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
            const int token = wt * S::kWarpTokens + mt * 16 + (lane >> 2);
#pragma unroll
            for (int mr = 0; mr < gate_fragments; ++mr) {
                const int row    = wr * (S::kWarpRows / 2) + mr * 8 + 2 * (lane & 3);
                const auto& gate = acc[mt][mr];
                const auto& up   = acc[mt][mr + gate_fragments];
                *reinterpret_cast<__nv_bfloat162*>(final_tile + token * stride + row) =
                    __floats2bfloat162_rn(silu(gate[0]) * up[0], silu(gate[1]) * up[1]);
                *reinterpret_cast<__nv_bfloat162*>(final_tile + (token + 8) * stride + row) =
                    __floats2bfloat162_rn(silu(gate[2]) * up[2], silu(gate[3]) * up[3]);
            }
        }
        synchronize<S>();
    }

    template <class S, bool Full, class Output>
    __device__ __forceinline__ void
    finish_tile(Output output, unsigned char* scratch, float (&acc)[S::kMmaTokens][S::kMmaRows][4],
                int row_begin, int token_begin, int, int token_end) const {
        constexpr int rows    = S::kBlockRows / 2;
        constexpr int stride  = rows + 8;
        constexpr int threads = S::kThreads - kProducers<S>;
        const int tid         = static_cast<int>(threadIdx.x) - kProducers<S>;
        stage<S>(scratch, acc);
        const auto* final_tile = reinterpret_cast<const __nv_bfloat16*>(scratch);
        for (int item = tid; item < S::kBlockTokens * (rows / 8); item += threads) {
            const int token = item / (rows / 8), row = (item % (rows / 8)) * 8;
            if (Full || token_begin + token < token_end)
                linear_store_bf16_vector(output, row_begin + row, token_begin + token,
                                         load_vec<uint4>(final_tile + token * stride + row));
        }
    }
};

// The fused dense MLP (SI-02+): instead of storing the BF16 product, each 16-row group of a token's staged
// product is encoded as the down projection's A4 activation planes (quantize_nvfp4_terms_k16 with the down
// projection's activation divisor): the bytes nvfp4_a4_residual_quantize_kernel (Terms > 1) or
// nvfp4_a4_quantize_kernel (Terms = 1: term 0 is quantize_nvfp4_k16) writes from the stored product, scales row-major
// or in the TMA route's tiles (`Layout`). A CTA's product rows are 16-aligned whole groups. A tiled plane is read in
// whole token tiles, so the tokens of a CTA's tile past the call get zero scales, as the quantizer writes them (their
// codes are past the activation map's rows, which the TMA reads as zeroes).
template <int Terms, Nvfp4ScaleLayout Layout = Nvfp4ScaleLayout::RowMajor>
struct SwiGluA4PlanesEpilogue : SwiGluTokenMajorMmaEpilogue {
    std::uint8_t* codes;
    std::uint8_t* scales;
    std::int64_t code_plane;
    std::int64_t scale_plane;
    float input_scale_divisor;
    int down_input_rows; // the product's rows (the down projection's K)
    // A tiled plane's columns from here (a whole-tile boundary) are written row-major instead, for the down
    // projection's MMA call over the remainder past its TMA route's whole tiles (ZK-09, nvfp4_tma_tail.h); their
    // offsets are then the row-major ones, which start where the whole tiles' tiled scales end.
    int row_major_from = 0x7fffffff;

    // nvfp4_tiled_scale_offset's tiles for a K of `groups_per_row` groups, or the row-major offset.
    __device__ __forceinline__ std::int64_t scale_offset(int column, int group, int groups_per_row) const {
        if constexpr (Layout == Nvfp4ScaleLayout::RowMajor) {
            return static_cast<std::int64_t>(column) * groups_per_row + group;
        } else {
            if (column >= row_major_from) { return static_cast<std::int64_t>(column) * groups_per_row + group; }
            constexpr int kTileTokens = kNvfp4ScaleTileTokens<Layout>;
            const int token_tile = column / kTileTokens, group_tile = group / kNvfp4ScaleTileGroups;
            const std::int64_t tile = static_cast<std::int64_t>(token_tile) * (groups_per_row / kNvfp4ScaleTileGroups) + group_tile;
            return tile * kTileTokens * kNvfp4ScaleTileGroups +
                   static_cast<std::int64_t>(column - token_tile * kTileTokens) * kNvfp4ScaleTileGroups +
                   (group - group_tile * kNvfp4ScaleTileGroups);
        }
    }

    template <class S, bool Full, class Output>
    __device__ __forceinline__ void
    finish_tile(Output, unsigned char* scratch, float (&acc)[S::kMmaTokens][S::kMmaRows][4], int row_begin,
                int token_begin, int, int token_end) const {
        constexpr int rows    = S::kBlockRows / 2;
        constexpr int stride  = rows + 8;
        constexpr int threads = S::kThreads - kProducers<S>;
        static_assert(rows % 16 == 0 && (stride * 2) % 16 == 0);
        static_assert(threads % 32 == 0);
        static_assert(Layout == Nvfp4ScaleLayout::RowMajor || S::kBlockTokens % kNvfp4ScaleTileTokens<Layout> == 0,
                      "a tiled plane's padding is written by the CTA whose token tile holds it");
        const int tid = static_cast<int>(threadIdx.x) - kProducers<S>;
        stage<S>(scratch, acc);
        const auto* final_tile = reinterpret_cast<const __nv_bfloat16*>(scratch);
        const int groups_per_row = down_input_rows / 16;
        // Four lanes a group (quantize_nvfp4_terms_k16_quad): lane q encodes the group's elements 4q..4q+3.
        const int q          = tid & 3;
        const unsigned quad  = 0xFU << (static_cast<unsigned>(threadIdx.x) & 28U);
        for (int item = tid >> 2; item < S::kBlockTokens * (rows / 16); item += threads / 4) {
            const int token = item / (rows / 16), local = (item % (rows / 16)) * 16;
            const int column = token_begin + token;
            const int group  = (row_begin + local) / 16;
            if (!Full && column >= token_end) { // the same for the quad's lanes
                if constexpr (Layout != Nvfp4ScaleLayout::RowMajor) {
                    if (q == 0 && column < row_major_from) {
#pragma unroll
                        for (int term = 0; term < Terms; ++term) {
                            scales[term * scale_plane + scale_offset(column, group, groups_per_row)] = 0;
                        }
                    }
                }
                continue;
            }
            std::uint16_t group_codes[Terms];
            std::uint8_t group_scales[Terms];
            quantize_nvfp4_terms_k16_quad<Terms>(final_tile + token * stride + local + 4 * q, input_scale_divisor, quad,
                                                 group_codes, group_scales);
            const std::int64_t code_offset = static_cast<std::int64_t>(column) * (down_input_rows / 2) + group * 8 + 2 * q;
            const std::int64_t scale_at    = scale_offset(column, group, groups_per_row);
#pragma unroll
            for (int term = 0; term < Terms; ++term) {
                *reinterpret_cast<std::uint16_t*>(codes + term * code_plane + code_offset) = group_codes[term];
                if (q == 0) { scales[term * scale_plane + scale_at] = group_scales[term]; }
            }
        }
    }
};
} // namespace infernix::ops::detail
