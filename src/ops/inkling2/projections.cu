// Inkling 2's few-input projections in one launch (include/infernix/ops/inkling2_projections.h): the BF16 skinny
// GEMV's arithmetic (ops/linear/bf16/bf16_skinny.cuh) over several matrices, each row group of a CTA in one segment.
// A warp owns RowsPerWarp rows; lane l folds chunks l, l + 32, ... in order, kAhead chunks of each row in flight
// (the first kAhead loaded before the PDL wait), every column of the pass from one load of the chunk.

#include "infernix/ops/inkling2_projections.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"

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
    std::int32_t cols; // its columns (<= Tokens; a wide launch: the call's)
    std::int32_t block_columns; // a wide launch: the columns each CTA row (grid y) takes, a multiple of kMaxPass
};

// A chunk of 8 weights as stored: BF16 words, or bf16_p12 code bytes and exponent nibbles.
struct P12Raw {
    uint2 code;
    unsigned nib;
};

template <int Tokens, int Warps, int RowsPerWarp, bool P12, class Out>
__global__ void __launch_bounds__(Warps * 32) projections_kernel(const Params p) {
    using Raw      = std::conditional_t<P12, P12Raw, uint4>;
    const int lane = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
    const int cta  = static_cast<int>(blockIdx.x);
    int s          = 0;
    while (s + 1 < p.count && cta >= p.cta_begin[s + 1]) { ++s; }
    const ProjectionSegment seg = p.seg[s];
    const int first             = ((cta - p.cta_begin[s]) * Warps + warp) * RowsPerWarp;
    const int chunks            = p.k / 8;
    std::size_t base[RowsPerWarp];
    ::infernix::ops::detail::P12Row prow[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = min(first + r, seg.rows - 1);
        base[r]       = static_cast<std::size_t>(row) * chunks;
        if constexpr (P12) { prow[r] = ::infernix::ops::detail::p12_row(seg.words, row); }
    }
    const auto load = [&](int r, int chunk) {
        if constexpr (P12) {
            return P12Raw{__ldcs(static_cast<const uint2*>(seg.codes) + base[r] + chunk),
                          __ldcs(static_cast<const unsigned*>(seg.nibbles) + base[r] + chunk)};
        } else {
            return __ldcs(reinterpret_cast<const uint4*>(seg.weight) + base[r] + chunk);
        }
    };
    const auto decode = [&](const Raw& raw, int r, int chunk) {
        if constexpr (P12) {
            return ::infernix::ops::detail::p12_decode_chunk(raw.code, raw.nib, prow[r], chunk, seg.escapes);
        } else {
            (void)r, (void)chunk;
            return raw;
        }
    };
    Raw w[kAhead][RowsPerWarp];
#pragma unroll
    for (int a = 0; a < kAhead; ++a) {
        const int chunk = lane + 32 * a;
        if (chunk < chunks) {
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) { w[a][r] = load(r, chunk); }
        }
    }
    pdl::enter_streaming();

    const std::uint16_t* x = p.x[seg.input] + static_cast<std::size_t>(p.c0) * p.k;
    float sums[RowsPerWarp][Tokens] = {};
    for (int start = 0; start < chunks; start += 32 * kAhead) {
#pragma unroll
        for (int a = 0; a < kAhead; ++a) {
            const int chunk = start + lane + 32 * a;
            if (chunk >= chunks) { break; }
            uint4 wd[RowsPerWarp];
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) { wd[r] = decode(w[a][r], r, chunk); }
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                if (t >= p.cols) { break; }
                const uint4 xp = *(reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * p.k) + chunk);
                const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
                    const __nv_bfloat162* wv = reinterpret_cast<const __nv_bfloat162*>(&wd[r]);
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
                for (int r = 0; r < RowsPerWarp; ++r) { w[a][r] = load(r, next); }
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

// A wide call (more than kMaxPass columns, BF16 storage): each lane holds every chunk of its warp's rows in registers
// (Chunks a row), loaded once before the PDL wait, and the launch runs the call's passes of kMaxPass columns itself.
// A (row, column)'s fold is projections_kernel's (chunks l, l + 32, ... in order, 8 FMAs in k order a chunk, the same
// butterfly), so are its bits; only the weights' reuse across passes differs.
// The small instance (the gates' downs) at most 128 registers a thread, eight 2-warp CTAs an SM (measured faster:
// 128 -> 97 ms of a 2,048-column call's gates); the large one unbounded (bounding it spilled W_o's and left K2's
// time unchanged).
template <int Chunks, int Warps, int RowsPerWarp, class Out>
__global__ void __launch_bounds__(Warps * 32, Warps == 2 ? 8 : 1) projections_wide_kernel(const Params p) {
    const int lane = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
    const int cta  = static_cast<int>(blockIdx.x);
    int s          = 0;
    while (s + 1 < p.count && cta >= p.cta_begin[s + 1]) { ++s; }
    const ProjectionSegment seg = p.seg[s];
    const int first             = ((cta - p.cta_begin[s]) * Warps + warp) * RowsPerWarp;
    const int chunks            = p.k / 8;
    uint4 w[Chunks][RowsPerWarp];
#pragma unroll
    for (int c = 0; c < Chunks; ++c) {
        const int chunk = lane + 32 * c;
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            const int row = min(first + r, seg.rows - 1);
            w[c][r]       = chunk < chunks ? __ldcs(reinterpret_cast<const uint4*>(seg.weight) +
                                                    static_cast<std::size_t>(row) * chunks + chunk)
                                           : make_uint4(0U, 0U, 0U, 0U);
        }
    }
    pdl::enter_streaming();
    const std::uint16_t* x = p.x[seg.input];
    Out* out               = static_cast<Out*>(seg.out);
    // This CTA's columns: grid y splits the call's columns (each column whole in one CTA, so its bits are kept).
    const int from = static_cast<int>(blockIdx.y) * p.block_columns, to = min(p.cols, from + p.block_columns);
    for (int c0 = from; c0 < to; c0 += kMaxPass) {
        const int n = min(kMaxPass, to - c0);
        float sums[RowsPerWarp][kMaxPass] = {};
#pragma unroll
        for (int c = 0; c < Chunks; ++c) {
            const int chunk = lane + 32 * c;
            if (chunk >= chunks) { break; }
            // The chunk's weights as FP32 once (an exact conversion), for every column of the pass.
            float2 wf[RowsPerWarp][4];
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
                const __nv_bfloat162* wv = reinterpret_cast<const __nv_bfloat162*>(&w[c][r]);
#pragma unroll
                for (int i = 0; i < 4; ++i) { wf[r][i] = __bfloat1622float2(wv[i]); }
            }
#pragma unroll
            for (int t = 0; t < kMaxPass; ++t) {
                if (t >= n) { break; }
                const uint4 xp = *(reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(c0 + t) * p.k) + chunk);
                const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(&xp);
                float2 xf[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) { xf[i] = __bfloat1622float2(xv[i]); }
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sums[r][t] = fmaf(wf[r][i].x, xf[i].x, sums[r][t]);
                        sums[r][t] = fmaf(wf[r][i].y, xf[i].y, sums[r][t]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
            for (int t = 0; t < kMaxPass; ++t) {
                if (t >= n) { break; }
                float v = sums[r][t];
                for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
                if (lane == 0 && first + r < seg.rows) {
                    const std::size_t at = static_cast<std::size_t>(c0 + t) * seg.rows + first + r;
                    if constexpr (std::is_same_v<Out, float>) {
                        out[at] = v;
                    } else {
                        out[at] = __float2bfloat16_rn(v);
                    }
                }
            }
        }
    }
    pdl::trigger_dependents();
}

// The wide kernel for K = 8 * 32 * Chunks at most (Chunks of 1, 9, 12 and 18: the forget/gain, K2, W_o and gate-down
// shapes); false for another K (the call then runs narrow passes).
template <int Warps, int RowsPerWarp, class Out>
bool launch_wide(const Params& p, dim3 grid, cudaStream_t stream) {
    const pdl::LaunchConfig config{grid, dim3(Warps * 32), 0, stream};
    switch ((p.k / 8 + 31) / 32) {
    case 1: CUDA_CHECK(pdl::launch_consumer(config, projections_wide_kernel<1, Warps, RowsPerWarp, Out>, p)); return true;
    case 9: CUDA_CHECK(pdl::launch_consumer(config, projections_wide_kernel<9, Warps, RowsPerWarp, Out>, p)); return true;
    case 12: CUDA_CHECK(pdl::launch_consumer(config, projections_wide_kernel<12, Warps, RowsPerWarp, Out>, p)); return true;
    case 18: CUDA_CHECK(pdl::launch_consumer(config, projections_wide_kernel<18, Warps, RowsPerWarp, Out>, p)); return true;
    default: return false;
    }
}

template <int Warps, int RowsPerWarp, bool P12, class Out>
void launch_pass(const Params& p, int tokens, unsigned grid, cudaStream_t stream) {
    const pdl::LaunchConfig config{dim3(grid), dim3(Warps * 32), 0, stream};
    switch (tokens) {
    case 1: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<1, Warps, RowsPerWarp, P12, Out>, p)); break;
    case 2: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<2, Warps, RowsPerWarp, P12, Out>, p)); break;
    case 4: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<4, Warps, RowsPerWarp, P12, Out>, p)); break;
    case 8: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<8, Warps, RowsPerWarp, P12, Out>, p)); break;
    default: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<16, Warps, RowsPerWarp, P12, Out>, p)); break;
    }
}

template <bool P12>
void launch_storage(const Params& p, bool small, bool fp32, int tokens, unsigned grid, cudaStream_t stream) {
    if (small) {
        fp32 ? launch_pass<2, 1, P12, float>(p, tokens, grid, stream)
             : launch_pass<2, 1, P12, __nv_bfloat16>(p, tokens, grid, stream);
    } else {
        fp32 ? launch_pass<8, 2, P12, float>(p, tokens, grid, stream)
             : launch_pass<8, 2, P12, __nv_bfloat16>(p, tokens, grid, stream);
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
    const bool packed   = a.segments[0].codes != nullptr;
    for (int s = 0; s < a.count; ++s) {
        const ProjectionSegment& g = a.segments[s];
        const bool storage = packed ? g.codes != nullptr && g.nibbles != nullptr && g.words != nullptr &&
                                          g.escapes != nullptr && g.weight == nullptr
                                    : g.codes == nullptr && g.weight != nullptr && aligned16(g.weight);
        if (!storage || g.rows <= 0 || g.input < 0 || g.input > 2 || a.x[g.input] == nullptr || g.out == nullptr ||
            !aligned16(a.x[g.input])) {
            throw std::invalid_argument("inkling2 projections: a segment's weight (one storage for the call), input or "
                                        "output is missing or misaligned");
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
    const auto grid      = static_cast<unsigned>(ctas);
    if (!packed && a.columns > kMaxPass) { // one launch over every pass, the weights held in registers
        // The narrow kernel's CTA shapes (2 rows a warp, 8 warps; one row a warp halved x's reuse and ran K2 1.5x
        // slower); the columns split into blocks (grid y) so the launch has about kWideCtas CTAs whatever its rows: a
        // matrix of few rows (the gates' downs) still fills the GPU; each column stays whole in one CTA.
        constexpr int kWideCtas = 1024;
        const int wide_rows = small ? 2 : 16;
        Params w            = p;
        std::int32_t wctas  = 0;
        for (int s = 0; s < a.count; ++s) {
            w.cta_begin[s] = wctas;
            wctas += (a.segments[s].rows + wide_rows - 1) / wide_rows;
        }
        w.cta_begin[a.count] = wctas;
        const int passes   = (a.columns + kMaxPass - 1) / kMaxPass;
        const int blocks   = std::clamp((kWideCtas + wctas - 1) / wctas, 1, passes);
        w.c0               = 0;
        w.cols             = a.columns;
        w.block_columns    = (passes + blocks - 1) / blocks * kMaxPass;
        const dim3 wgrid(static_cast<unsigned>(wctas),
                         static_cast<unsigned>((a.columns + w.block_columns - 1) / w.block_columns));
        const bool wide = small ? (a.fp32_out ? launch_wide<2, 1, float>(w, wgrid, stream)
                                              : launch_wide<2, 1, __nv_bfloat16>(w, wgrid, stream))
                                : (a.fp32_out ? launch_wide<8, 2, float>(w, wgrid, stream)
                                              : launch_wide<8, 2, __nv_bfloat16>(w, wgrid, stream));
        if (wide) { return; }
    }
    for (std::int32_t c0 = 0; c0 < a.columns; c0 += kMaxPass) {
        p.c0   = c0;
        p.cols = std::min(kMaxPass, a.columns - c0);
        const int tokens = p.cols == 1 ? 1 : p.cols == 2 ? 2 : p.cols <= 4 ? 4 : p.cols <= 8 ? 8 : 16;
        packed ? launch_storage<true>(p, small, a.fp32_out, tokens, grid, stream)
               : launch_storage<false>(p, small, a.fp32_out, tokens, grid, stream);
    }
}

} // namespace infernix::ops::inkling2
