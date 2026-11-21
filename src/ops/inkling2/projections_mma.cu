// Inkling 2's few-input projections on tensor cores (include/infernix/ops/inkling2_projections.h projections_mma):
// several BF16 [rows, K] matrices in one launch, each bound to one of up to three inputs, in ops::linear's MMA
// arithmetic (src/ops/linear/bf16/bf16_a16_mma.cuh): a matrix's rows are the A operand of mma.m16n8k16 and the columns
// the B operand, and every (row, column) accumulates the k16 steps in ascending K from zero in FP32 through the MMA's
// accumulator, without a split of K. A (row, column)'s bits therefore depend on its row and its column alone: not on
// the call's width, its other columns or matrices, nor on the tile shape (each tile runs the same step sequence).
//
// A CTA owns BM rows of one segment and BN columns; a cp.async ring of S stages of BK; the products by ops::linear's
// stage (bf16_mma_compute_stage); the tile leaves through shared memory as 16-byte rows of the [T][rows] output.

#include "infernix/ops/inkling2_projections.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/dynamic_shared.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16/bf16_schedule.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace infernix::ops::inkling2 {
namespace {

namespace d = ::infernix::ops::detail;

struct MmaSegment {
    const __nv_bfloat16* weight;
    const __nv_bfloat16* x;
    void* out;
    std::int32_t rows;
    // Q8 storage (Q8_G32_FP16 RowSplit): int8 codes [rows][stride] and FP16 scales [rows][stride / 32].
    const std::uint8_t* codes   = nullptr;
    const std::uint16_t* scales = nullptr;
    std::int32_t stride         = 0;
};

// A Q8 stage's weights: codes [BM][kQ8Row] (64 a row used; the padding spreads the fragment rows over the banks) and
// scales [BM][2] (the stage's two 32-groups), inside the BF16 stage's region.
constexpr int kQ8Row = 80;

struct MmaParams {
    MmaSegment seg[kMaxProjectionSegments];
    std::int32_t tile_begin[kMaxProjectionSegments + 1]; // each segment's first row tile (grid x)
    std::int32_t count;
    std::int32_t k;
    std::int32_t columns;
};

template <int BM, int BN, int WM, int WN, int S, int MinBlocks>
using Schedule = d::Bf16A16MmaSchedule<BM, BN, 64, WM, WN, S, MinBlocks, Cache::cg, Cache::cg,
                                       d::Bf16MmaFragmentPipeline::PingPong, d::Bf16MmaRaster::TokenFast,
                                       d::Bf16MmaSwizzle::Xor64>;

// The output tile staged as FP32 [BN][BM + 4] (the padding spreads a fragment's 8 rows over the banks).
template <class Sched>
constexpr int staging_bytes() {
    return Sched::kBlockTokens * (Sched::kBlockRows + 4) * 4;
}
template <class Sched>
constexpr int shared_bytes() {
    return std::max(Sched::kSharedBytes, staging_bytes<Sched>());
}

// A Q8 stage (BK = 64: two 32-groups): per group a chain of two m16n8k16 MMAs from zero over the codes (as BF16, exact)
// and the columns' BF16 inputs, then acc = fma(group sum, the row's group scale, acc). The B fragments as
// bf16_mma_compute_stage reads them.
template <class Sched>
__device__ __forceinline__ void q8_compute_stage(const std::uint8_t* codes, const std::uint16_t* scales,
                                                 const __nv_bfloat16* Bs,
                                                 float (&accum)[Sched::kMmaRows][Sched::kMmaTokens][4], int warp,
                                                 int lane) {
    constexpr int BK = Sched::kBlockK, WM = Sched::kWarpRows, WN = Sched::kWarpTokens;
    constexpr int MT = Sched::kMmaRows, NT = Sched::kMmaTokens;
    static_assert(BK == 64, "a Q8 stage is two 32-groups");
    const int wm = warp / Sched::kWarpsTokens, wn = warp % Sched::kWarpsTokens;
    const int g = lane >> 2, t = lane & 3;
    const int b_inner_row = lane & 7, b_k_offset = ((lane >> 3) & 1) << 3;
    const auto pair = [&](int row, int k) {
        const unsigned v = *reinterpret_cast<const std::uint16_t*>(codes + row * kQ8Row + k);
        const __nv_bfloat162 b = __floats2bfloat162_rn(static_cast<float>(static_cast<signed char>(v & 0xFFU)),
                                                       static_cast<float>(static_cast<signed char>(v >> 8)));
        return *reinterpret_cast<const unsigned*>(&b);
    };
#pragma unroll
    for (int grp = 0; grp < 2; ++grp) {
        float tmp[MT][NT][4] = {};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int k = 32 * grp + 16 * ks;
            unsigned a[MT][4];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int r = wm * WM + mi * 16 + g;
                a[mi][0] = pair(r, k + 2 * t);
                a[mi][1] = pair(r + 8, k + 2 * t);
                a[mi][2] = pair(r, k + 8 + 2 * t);
                a[mi][3] = pair(r + 8, k + 8 + 2 * t);
            }
            unsigned b[NT][2];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = wn * WN + ni * 8 + b_inner_row;
                const int col = k + b_k_offset;
                ldmatrix_x2(b[ni][0], b[ni][1], smem_addr(&Bs[row * BK + d::bf16_mma_shared_col<Sched>(row, col)]));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(tmp[mi][ni][0], tmp[mi][ni][1], tmp[mi][ni][2], tmp[mi][ni][3], a[mi][0], a[mi][1],
                             a[mi][2], a[mi][3], b[ni][0], b[ni][1]);
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int r = wm * WM + mi * 16 + g;
            const float s0 = __half2float(__ushort_as_half(scales[r * 2 + grp]));
            const float s1 = __half2float(__ushort_as_half(scales[(r + 8) * 2 + grp]));
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                accum[mi][ni][0] = fmaf(tmp[mi][ni][0], s0, accum[mi][ni][0]);
                accum[mi][ni][1] = fmaf(tmp[mi][ni][1], s0, accum[mi][ni][1]);
                accum[mi][ni][2] = fmaf(tmp[mi][ni][2], s1, accum[mi][ni][2]);
                accum[mi][ni][3] = fmaf(tmp[mi][ni][3], s1, accum[mi][ni][3]);
            }
        }
    }
}

template <class Sched, class Out, bool Q8 = false>
__global__ void __launch_bounds__(Sched::kThreads, Sched::kMinBlocksPerSm) projections_mma_kernel(const MmaParams p) {
    constexpr int BM = Sched::kBlockRows, BN = Sched::kBlockTokens, BK = Sched::kBlockK, S = Sched::kStages;
    constexpr int THREADS = Sched::kThreads;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* As = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* Bs = As + S * BM * BK;
    // Q8: stage s's codes at shared_raw + s * BM * kQ8Row, its scales after every stage's codes.
    static_assert(!Q8 || S * BM * (kQ8Row + 4) <= S * BM * BK * 2, "a Q8 stage fits the BF16 stage's region");
    auto* q8_codes  = reinterpret_cast<std::uint8_t*>(shared_raw);
    auto* q8_scales = reinterpret_cast<std::uint16_t*>(shared_raw + S * BM * kQ8Row);

    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int tile = static_cast<int>(blockIdx.x);
    int s          = 0;
    while (s + 1 < p.count && tile >= p.tile_begin[s + 1]) { ++s; }
    const MmaSegment seg = p.seg[s];
    const int K = p.k, M = seg.rows, T = p.columns;
    const int m0 = (tile - p.tile_begin[s]) * BM;
    const int n0 = static_cast<int>(blockIdx.y) * BN;
    const int kTiles = K / BK;

    // Weights first (immutable: before the PDL wait), then the inputs of the same stages.
    const auto stage_weights = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
        if constexpr (Q8) {
            std::uint8_t* codes    = q8_codes + stage * BM * kQ8Row;
            std::uint16_t* scales  = q8_scales + stage * BM * 2;
#pragma unroll 1
            for (int item = tid; item < BM * (BK / 16 + 1); item += THREADS) {
                const int row = item / (BK / 16 + 1), part = item - row * (BK / 16 + 1);
                const bool valid = m0 + row < M;
                const std::size_t r = valid ? static_cast<std::size_t>(m0 + row) : 0;
                if (part < BK / 16) {
                    ops::cp_async_zfill<16, Cache::cg>(codes + row * kQ8Row + 16 * part,
                                                      seg.codes + r * seg.stride + k0 + 16 * part, valid ? 16 : 0);
                } else {
                    ops::cp_async_zfill<4>(scales + row * 2, seg.scales + r * (seg.stride / 32) + k0 / 32,
                                           valid ? 4 : 0);
                }
            }
            return;
        }
        auto* a      = As + stage * BM * BK;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 8); item += THREADS) {
            const int row = item / (BK / 8), kk = (item - row * (BK / 8)) * 8;
            const bool valid = m0 + row < M;
            const std::size_t src = valid ? static_cast<std::size_t>(m0 + row) * K + k0 + kk : 0;
            ops::cp_async_zfill<16, Cache::cg>(&a[row * BK + d::bf16_mma_shared_col<Sched>(row, kk)], seg.weight + src,
                                              valid ? 16 : 0);
        }
    };
    const auto stage_inputs = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
        auto* b      = Bs + stage * BN * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int col = item / (BK / 8), kk = (item - col * (BK / 8)) * 8;
            const bool valid = n0 + col < T;
            const std::size_t src = valid ? static_cast<std::size_t>(n0 + col) * K + k0 + kk : 0;
            ops::cp_async_zfill<16, Cache::cg>(&b[col * BK + d::bf16_mma_shared_col<Sched>(col, kk)], seg.x + src,
                                              valid ? 16 : 0);
        }
    };

    // The prologue: S groups of weights, then (after the PDL wait) one group of the first stages' inputs. The first
    // step waits for all of them; every later step's stage came from the prologue or from the refill committed S
    // steps earlier, with at most S - 1 groups (one a step, empty ones included) after it.
#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < kTiles) { stage_weights(stage, stage); }
        ops::cp_commit();
    }
    pdl::enter_streaming();
#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < kTiles) { stage_inputs(stage, stage); }
    }
    ops::cp_commit();

    float accum[Sched::kMmaRows][Sched::kMmaTokens][4] = {};
#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile == 0) {
            ops::cp_wait<0>();
        } else {
            ops::cp_wait<S - 1>();
        }
        __syncthreads();
        if constexpr (Q8) {
            q8_compute_stage<Sched>(q8_codes + stage * BM * kQ8Row, q8_scales + stage * BM * 2, Bs + stage * BN * BK,
                                    accum, warp, lane);
        } else {
            d::bf16_mma_compute_stage<Sched>(As + stage * BM * BK, Bs + stage * BN * BK, accum, warp, lane);
        }
        __syncthreads();
        const int next = k_tile + S;
        if (next < kTiles) {
            stage_weights(stage, next);
            stage_inputs(stage, next);
        }
        ops::cp_commit();
    }
    ops::cp_wait<0>();
    __syncthreads();
    pdl::trigger_dependents();

    // The tile through shared memory: FP32 [BN][BM + 4], then 16-byte rows of the output.
    constexpr int LD = BM + 4;
    float* tile_out  = reinterpret_cast<float*>(shared_raw);
    const int wm = warp / Sched::kWarpsTokens, wn = warp % Sched::kWarpsTokens;
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mi = 0; mi < Sched::kMmaRows; ++mi) {
#pragma unroll
        for (int ni = 0; ni < Sched::kMmaTokens; ++ni) {
            const int r = wm * Sched::kWarpRows + mi * 16 + g, c = wn * Sched::kWarpTokens + ni * 8 + 2 * t;
            tile_out[c * LD + r]           = accum[mi][ni][0];
            tile_out[(c + 1) * LD + r]     = accum[mi][ni][1];
            tile_out[c * LD + r + 8]       = accum[mi][ni][2];
            tile_out[(c + 1) * LD + r + 8] = accum[mi][ni][3];
        }
    }
    __syncthreads();
    constexpr int kPer = std::is_same_v<Out, float> ? 4 : 8; // values a 16-byte store
    constexpr int kChunks = BM / kPer;
    Out* out = static_cast<Out*>(seg.out);
#pragma unroll 1
    for (int item = tid; item < BN * kChunks; item += THREADS) {
        const int c = item / kChunks, r = (item - c * kChunks) * kPer;
        if (n0 + c >= T || m0 + r >= M) { continue; }
        const float* v = tile_out + c * LD + r;
        Out* dst       = out + static_cast<std::size_t>(n0 + c) * M + m0 + r;
        if constexpr (std::is_same_v<Out, float>) {
            *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
        } else {
            uint4 packed;
            unsigned* words = reinterpret_cast<unsigned*>(&packed);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __nv_bfloat162 pair = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
                words[i]                  = *reinterpret_cast<const unsigned*>(&pair);
            }
            *reinterpret_cast<uint4*>(dst) = packed;
        }
    }
}

template <class Sched, class Out, bool Q8 = false>
void launch(const MmaParams& p, std::int32_t tiles, cudaStream_t stream) {
    constexpr auto kernel = projections_mma_kernel<Sched, Out, Q8>;
    constexpr int bytes   = shared_bytes<Sched>();
    const int dynamic     = d::prepare_dynamic_shared<bytes, kernel, true>();
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>((p.columns + Sched::kBlockTokens - 1) / Sched::kBlockTokens));
    const pdl::LaunchConfig config{grid, dim3(Sched::kThreads), static_cast<std::size_t>(dynamic), stream};
    CUDA_CHECK(pdl::launch_consumer(config, kernel, p));
}

// Tile shapes (the arithmetic is the same in each): wide calls 128 x 128 (8 warps of 64 x 32, three 32 KB stages);
// few-row matrices (<= 256 rows: the gates' downs, the router) 64 x 64 so the launch still fills the GPU; narrow
// calls (<= 16 columns) 64 rows x 16 columns.
template <class Out, bool Q8 = false>
void launch_shape(const MmaParams& p, bool small, cudaStream_t stream, std::int32_t& tiles_out,
                  const ProjectionArgs& a) {
    const auto tiles_for = [&](int bm, MmaParams& q) {
        std::int32_t tiles = 0;
        for (int s = 0; s < a.count; ++s) {
            q.tile_begin[s] = tiles;
            tiles += (a.segments[s].rows + bm - 1) / bm;
        }
        q.tile_begin[a.count] = tiles;
        return tiles;
    };
    MmaParams q = p;
    if (a.columns <= 8) { // decode widths: one warp of 16 rows a CTA, eight stages in flight
        using Sched = Schedule<16, 8, 16, 8, 8, 4>;
        tiles_out   = tiles_for(Sched::kBlockRows, q);
        launch<Sched, Out, Q8>(q, tiles_out, stream);
    } else if (a.columns <= 16) {
        using Sched = Schedule<16, 16, 16, 16, 8, 4>;
        tiles_out   = tiles_for(Sched::kBlockRows, q);
        launch<Sched, Out, Q8>(q, tiles_out, stream);
    } else if (small) {
        using Sched = Schedule<64, 64, 32, 32, 3, 2>;
        tiles_out   = tiles_for(Sched::kBlockRows, q);
        launch<Sched, Out, Q8>(q, tiles_out, stream);
    } else {
        // Q8: 16 warps of 32 x 32 (each 32-group's sums beside the accumulators).
        using Sched = std::conditional_t<Q8, Schedule<128, 128, 32, 32, 3, 1>, Schedule<128, 128, 64, 32, 3, 1>>;
        tiles_out   = tiles_for(Sched::kBlockRows, q);
        launch<Sched, Out, Q8>(q, tiles_out, stream);
    }
}

// ---------------------------------------------------------------------------- decode-width MMA projections

// projections_dmma (calls of up to 16 columns): a CTA per 16-row tile of Splits warps; warp q accumulates the q-th of
// Splits equal runs of K's 32-blocks as a chain of m16n8k16 MMAs from zero, blocks in ascending order, two MMAs a block
// (k-step 0: elements 8t .. 8t + 3 of the block into slots 2t, 2t + 1, 2t + 8, 2t + 9, t = lane % 4; k-step 1: elements
// 8t + 4 .. 8t + 7), so a thread's weights and inputs of a block are one 16-byte load each; then the runs' partials
// added in order. A column's bits depend on its row and its input alone (the same at every width up to 16).
constexpr int kDmmaAhead = 4; // blocks in flight a warp

template <int Splits, int NT, class Out>
__global__ void __launch_bounds__(Splits * 32) projections_dmma_kernel(const MmaParams p) {
    __shared__ float partial[Splits][32][NT * 4];
    const int lane = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
    const int g = lane >> 2, t = lane & 3;
    const int tile = static_cast<int>(blockIdx.x);
    int s          = 0;
    while (s + 1 < p.count && tile >= p.tile_begin[s + 1]) { ++s; }
    const MmaSegment seg = p.seg[s];
    const int K = p.k, M = seg.rows, T = p.columns;
    const int m0 = (tile - p.tile_begin[s]) * 16;
    const int r0 = min(m0 + g, M - 1), r1 = min(m0 + g + 8, M - 1);
    const int blocks = K / 32 / Splits, b0 = warp * blocks;
    const uint4* w0 = reinterpret_cast<const uint4*>(seg.weight + static_cast<std::size_t>(r0) * K) + t;
    const uint4* w1 = reinterpret_cast<const uint4*>(seg.weight + static_cast<std::size_t>(r1) * K) + t;
    uint4 a0[kDmmaAhead], a1[kDmmaAhead];
#pragma unroll
    for (int i = 0; i < kDmmaAhead; ++i) {
        if (i < blocks) {
            a0[i] = __ldcs(w0 + 4 * (b0 + i));
            a1[i] = __ldcs(w1 + 4 * (b0 + i));
        }
    }
    pdl::enter_streaming();
    // The thread's input columns: g (and g + 8 with NT = 2); past the call's columns, zeros.
    const uint4* x[NT];
    bool live[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int c = g + 8 * n;
        live[n]     = c < T;
        x[n]        = reinterpret_cast<const uint4*>(seg.x + static_cast<std::size_t>(live[n] ? c : 0) * K) + t;
    }
    float acc[NT][4] = {};
    for (int i = 0; i < blocks; i += kDmmaAhead) {
#pragma unroll
        for (int j = 0; j < kDmmaAhead; ++j) {
            const int b = i + j;
            if (b >= blocks) { break; }
            uint4 xv[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) { xv[n] = live[n] ? x[n][4 * (b0 + b)] : make_uint4(0U, 0U, 0U, 0U); }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], a0[j].x, a1[j].x, a0[j].y, a1[j].y, xv[n].x, xv[n].y);
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], a0[j].z, a1[j].z, a0[j].w, a1[j].w, xv[n].z, xv[n].w);
            }
            const int next = b + kDmmaAhead;
            if (next < blocks) {
                a0[j] = __ldcs(w0 + 4 * (b0 + next));
                a1[j] = __ldcs(w1 + 4 * (b0 + next));
            }
        }
    }
    pdl::trigger_dependents();
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { partial[warp][lane][4 * n + e] = acc[n][e]; }
    }
    __syncthreads();
    if (warp != 0) { return; }
    Out* out = static_cast<Out*>(seg.out);
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float v = partial[0][lane][4 * n + e];
#pragma unroll
            for (int q = 1; q < Splits; ++q) { v = v + partial[q][lane][4 * n + e]; }
            const int row = m0 + g + (e >= 2 ? 8 : 0), col = 8 * n + 2 * t + (e & 1);
            if (row >= M || col >= T) { continue; }
            if constexpr (std::is_same_v<Out, float>) {
                out[static_cast<std::size_t>(col) * M + row] = v;
            } else {
                out[static_cast<std::size_t>(col) * M + row] = __float2bfloat16_rn(v);
            }
        }
    }
}

template <int Splits, int NT, class Out>
void launch_dmma(const MmaParams& p, std::int32_t tiles, cudaStream_t stream) {
    const pdl::LaunchConfig config{dim3(static_cast<unsigned>(tiles)), dim3(Splits * 32), 0, stream};
    CUDA_CHECK(pdl::launch_consumer(config, projections_dmma_kernel<Splits, NT, Out>, p));
}

// The runs of K a call splits into: a function of its segments' shapes only (never its width), enough warps for ~2,048
// in the launch (loads in flight on every SM), each run at least 4 blocks.
int dmma_splits(std::int32_t k, std::int32_t tiles) {
    const int blocks = k / 32;
    int splits       = 1;
    while (splits < 16 && tiles * splits < 2048 && blocks % (2 * splits) == 0 && blocks / (2 * splits) >= 4) {
        splits *= 2;
    }
    return splits;
}

template <int NT, class Out>
void dmma_width(const MmaParams& p, std::int32_t tiles, int splits, cudaStream_t stream) {
    switch (splits) {
    case 1: launch_dmma<1, NT, Out>(p, tiles, stream); break;
    case 2: launch_dmma<2, NT, Out>(p, tiles, stream); break;
    case 4: launch_dmma<4, NT, Out>(p, tiles, stream); break;
    case 8: launch_dmma<8, NT, Out>(p, tiles, stream); break;
    default: launch_dmma<16, NT, Out>(p, tiles, stream); break;
    }
}

template <class Out>
void dmma_shape(const MmaParams& p, std::int32_t tiles, cudaStream_t stream) {
    const int splits = dmma_splits(p.k, tiles);
    p.columns <= 8 ? dmma_width<1, Out>(p, tiles, splits, stream) : dmma_width<2, Out>(p, tiles, splits, stream);
}

MmaParams mma_params(const ProjectionArgs& a, int k_multiple, const char* what) {
    if (a.columns <= 0 || a.k <= 0 || a.k % k_multiple != 0 || a.count <= 0 || a.count > kMaxProjectionSegments) {
        throw std::invalid_argument(std::string("inkling2 ") + what + ": 1-8 segments, positive columns, K a multiple of " +
                                    std::to_string(k_multiple));
    }
    MmaParams p{};
    p.count   = a.count;
    p.k       = a.k;
    p.columns = a.columns;
    for (int s = 0; s < a.count; ++s) {
        const ProjectionSegment& g = a.segments[s];
        const auto* x              = (g.input >= 0 && g.input <= 2) ? a.x[g.input] : nullptr;
        const std::size_t word     = a.fp32_out ? 4 : 2;
        if (g.weight == nullptr || g.codes != nullptr || g.rows <= 0 || x == nullptr || g.out == nullptr ||
            reinterpret_cast<std::uintptr_t>(g.weight) % 16 != 0 || reinterpret_cast<std::uintptr_t>(x) % 16 != 0 ||
            reinterpret_cast<std::uintptr_t>(g.out) % 16 != 0 || (g.rows * word) % 16 != 0) {
            throw std::invalid_argument(std::string("inkling2 ") + what +
                                        ": BF16 segments only, with 16-byte aligned weights, inputs, outputs and output "
                                        "rows");
        }
        p.seg[s] = {reinterpret_cast<const __nv_bfloat16*>(g.weight), reinterpret_cast<const __nv_bfloat16*>(x), g.out,
                    g.rows};
    }
    return p;
}

} // namespace

void projections_dmma(const ProjectionArgs& a, cudaStream_t stream) {
    MmaParams p = mma_params(a, 32, "projections_dmma");
    std::int32_t tiles = 0;
    for (int s = 0; s < a.count; ++s) {
        p.tile_begin[s] = tiles;
        tiles += (a.segments[s].rows + 15) / 16;
    }
    p.tile_begin[a.count] = tiles;
    // A call wider than 16 columns runs launches of 16 (each column's arithmetic is its own: the same bits).
    const std::size_t word = a.fp32_out ? 4 : 2;
    for (std::int32_t c0 = 0; c0 < a.columns; c0 += 16) {
        MmaParams q = p;
        q.columns   = std::min<std::int32_t>(16, a.columns - c0);
        for (int s = 0; s < a.count; ++s) {
            q.seg[s].x   = p.seg[s].x + static_cast<std::size_t>(c0) * a.k;
            q.seg[s].out = static_cast<std::uint8_t*>(p.seg[s].out) + static_cast<std::size_t>(c0) * p.seg[s].rows * word;
        }
        a.fp32_out ? dmma_shape<float>(q, tiles, stream) : dmma_shape<__nv_bfloat16>(q, tiles, stream);
    }
}

void projections_mma(const ProjectionArgs& a, cudaStream_t stream) {
    if (a.columns <= 0 || a.k <= 0 || a.k % 64 != 0 || a.count <= 0 || a.count > kMaxProjectionSegments) {
        throw std::invalid_argument("inkling2 projections_mma: 1-8 segments, positive columns, K a multiple of 64");
    }
    MmaParams p{};
    p.count   = a.count;
    p.k       = a.k;
    p.columns = a.columns;
    std::int32_t widest = 0;
    // One storage for the call: BF16, or Q8 (codes and scales, stride a multiple of 128).
    const bool q8 = a.segments[0].q8_scales != nullptr;
    for (int s = 0; s < a.count; ++s) {
        const ProjectionSegment& g = a.segments[s];
        const auto* x              = (g.input >= 0 && g.input <= 2) ? a.x[g.input] : nullptr;
        const std::size_t word     = a.fp32_out ? 4 : 2;
        const bool storage = q8 ? (g.weight == nullptr && g.codes != nullptr && g.q8_scales != nullptr &&
                                   g.nibbles == nullptr && g.q8_stride >= a.k && g.q8_stride % 128 == 0 &&
                                   reinterpret_cast<std::uintptr_t>(g.codes) % 16 == 0 &&
                                   reinterpret_cast<std::uintptr_t>(g.q8_scales) % 4 == 0)
                                : (g.weight != nullptr && g.codes == nullptr &&
                                   reinterpret_cast<std::uintptr_t>(g.weight) % 16 == 0);
        if (!storage || g.rows <= 0 || x == nullptr || g.out == nullptr || reinterpret_cast<std::uintptr_t>(x) % 16 != 0 ||
            reinterpret_cast<std::uintptr_t>(g.out) % 16 != 0 || (g.rows * word) % 16 != 0) {
            throw std::invalid_argument("inkling2 projections_mma: BF16 or Q8 segments (one storage a call), with 16-byte "
                                        "aligned weights, inputs, outputs and output rows");
        }
        p.seg[s] = {reinterpret_cast<const __nv_bfloat16*>(g.weight), reinterpret_cast<const __nv_bfloat16*>(x), g.out,
                    g.rows};
        if (q8) {
            p.seg[s].codes  = static_cast<const std::uint8_t*>(g.codes);
            p.seg[s].scales = g.q8_scales;
            p.seg[s].stride = g.q8_stride;
        }
        widest = std::max(widest, g.rows);
    }
    std::int32_t tiles = 0;
    const bool small   = widest <= 256;
    if (q8) {
        a.fp32_out ? launch_shape<float, true>(p, small, stream, tiles, a)
                   : launch_shape<__nv_bfloat16, true>(p, small, stream, tiles, a);
        return;
    }
    a.fp32_out ? launch_shape<float>(p, small, stream, tiles, a) : launch_shape<__nv_bfloat16>(p, small, stream, tiles, a);
}

} // namespace infernix::ops::inkling2
