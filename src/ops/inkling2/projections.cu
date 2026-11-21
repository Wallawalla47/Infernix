// Inkling 2's few-input projections in one launch (include/infernix/ops/inkling2_projections.h): the BF16 skinny
// GEMV's arithmetic (ops/linear/bf16/bf16_skinny.cuh) over several matrices, each row group of a CTA in one segment.
// A warp owns RowsPerWarp rows; lane l folds chunks l, l + 32, ... in order, kAhead chunks of each row in flight
// (the first kAhead loaded before the PDL wait), every column of the pass from one load of the chunk.

#include "infernix/ops/inkling2_projections.h"

#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace infernix::ops::inkling2 {
namespace {

constexpr int kAhead   = 4;
constexpr int kMaxPass = 16;

struct Params {
    const std::uint16_t* x[3];
    ProjectionSegment seg[kMaxProjectionSegments];
    std::int32_t cta_begin[kMaxProjectionSegments + 1];
    std::int32_t count;
    std::int32_t k;
    std::int32_t c0;   // the pass's first column
    std::int32_t cols; // its columns (<= Tokens)
};

template <int Tokens, int Warps, int RowsPerWarp, class Out>
__global__ void __launch_bounds__(Warps * 32) projections_kernel(const Params p) {
    const int lane = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
    const int cta  = static_cast<int>(blockIdx.x);
    int s          = 0;
    while (s + 1 < p.count && cta >= p.cta_begin[s + 1]) { ++s; }
    const ProjectionSegment seg = p.seg[s];
    const int first             = ((cta - p.cta_begin[s]) * Warps + warp) * RowsPerWarp;
    const int chunks            = p.k / 8;
    const uint4* rows[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        rows[r] = reinterpret_cast<const uint4*>(seg.weight + static_cast<std::size_t>(min(first + r, seg.rows - 1)) * p.k);
    }
    uint4 w[kAhead][RowsPerWarp];
#pragma unroll
    for (int a = 0; a < kAhead; ++a) {
        const int chunk = lane + 32 * a;
        if (chunk < chunks) {
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) { w[a][r] = __ldcs(rows[r] + chunk); }
        }
    }
    pdl::enter_streaming();

    const std::uint16_t* x = p.x[seg.input] + static_cast<std::size_t>(p.c0) * p.k;
    float sums[RowsPerWarp][Tokens] = {};
    for (int base = 0; base < chunks; base += 32 * kAhead) {
#pragma unroll
        for (int a = 0; a < kAhead; ++a) {
            const int chunk = base + lane + 32 * a;
            if (chunk >= chunks) { break; }
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                if (t >= p.cols) { break; }
                const uint4 xp = *(reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * p.k) + chunk);
                const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
                    const __nv_bfloat162* wv = reinterpret_cast<const __nv_bfloat162*>(&w[a][r]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 wf = __bfloat1622float2(wv[i]);
                        const float2 xf = __bfloat1622float2(xv[i]);
                        sums[r][t]      = fmaf(wf.x, xf.x, sums[r][t]);
                        sums[r][t]      = fmaf(wf.y, xf.y, sums[r][t]);
                    }
                }
            }
            const int next = chunk + 32 * kAhead;
            if (next < chunks) {
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) { w[a][r] = __ldcs(rows[r] + next); }
            }
        }
    }
    pdl::trigger_dependents();
    Out* out = static_cast<Out*>(seg.out) + static_cast<std::size_t>(p.c0) * seg.rows;
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            if (t >= p.cols) { break; }
            float v = sums[r][t];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && first + r < seg.rows) {
                if constexpr (std::is_same_v<Out, float>) {
                    out[static_cast<std::size_t>(t) * seg.rows + first + r] = v;
                } else {
                    out[static_cast<std::size_t>(t) * seg.rows + first + r] = __float2bfloat16_rn(v);
                }
            }
        }
    }
}

template <int Warps, int RowsPerWarp, class Out>
void launch_pass(const Params& p, int tokens, unsigned grid, cudaStream_t stream) {
    const pdl::LaunchConfig config{dim3(grid), dim3(Warps * 32), 0, stream};
    switch (tokens) {
    case 1: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<1, Warps, RowsPerWarp, Out>, p)); break;
    case 2: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<2, Warps, RowsPerWarp, Out>, p)); break;
    case 4: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<4, Warps, RowsPerWarp, Out>, p)); break;
    case 8: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<8, Warps, RowsPerWarp, Out>, p)); break;
    default: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<16, Warps, RowsPerWarp, Out>, p)); break;
    }
}

template <class P>
bool aligned16(const P* p) {
    return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

} // namespace

void projections(const ProjectionArgs& a, cudaStream_t stream) {
    if (a.columns <= 0 || a.k <= 0 || a.k % 8 != 0 || a.count <= 0 || a.count > kMaxProjectionSegments) {
        throw std::invalid_argument("inkling2 projections: 1-8 segments, positive columns, K a multiple of 8");
    }
    std::int32_t widest = 0;
    for (int s = 0; s < a.count; ++s) {
        const ProjectionSegment& g = a.segments[s];
        if (g.weight == nullptr || g.rows <= 0 || g.input < 0 || g.input > 2 || a.x[g.input] == nullptr ||
            g.out == nullptr || !aligned16(g.weight) || !aligned16(a.x[g.input])) {
            throw std::invalid_argument("inkling2 projections: a segment's weight, input or output is missing or misaligned");
        }
        widest = std::max(widest, g.rows);
    }
    // Small matrices only: one row a warp, two warps a CTA (the BF16 skinny's small-N shape); otherwise 2 rows a warp,
    // 8 warps. The arithmetic of a row is the same in both.
    const bool small = widest <= 256;
    const int rows_per_cta = small ? 2 : 16;
    Params p{};
    for (int i = 0; i < 3; ++i) { p.x[i] = a.x[i]; }
    p.count = a.count;
    p.k     = a.k;
    std::int32_t ctas = 0;
    for (int s = 0; s < a.count; ++s) {
        p.seg[s]       = a.segments[s];
        p.cta_begin[s] = ctas;
        ctas += (a.segments[s].rows + rows_per_cta - 1) / rows_per_cta;
    }
    p.cta_begin[a.count] = ctas;
    for (std::int32_t c0 = 0; c0 < a.columns; c0 += kMaxPass) {
        p.c0   = c0;
        p.cols = std::min(kMaxPass, a.columns - c0);
        const int tokens = p.cols == 1 ? 1 : p.cols == 2 ? 2 : p.cols <= 4 ? 4 : p.cols <= 8 ? 8 : 16;
        const auto grid  = static_cast<unsigned>(ctas);
        if (small) {
            a.fp32_out ? launch_pass<2, 1, float>(p, tokens, grid, stream)
                       : launch_pass<2, 1, __nv_bfloat16>(p, tokens, grid, stream);
        } else {
            a.fp32_out ? launch_pass<8, 2, float>(p, tokens, grid, stream)
                       : launch_pass<8, 2, __nv_bfloat16>(p, tokens, grid, stream);
        }
    }
}

} // namespace infernix::ops::inkling2
