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
#include <type_traits>

namespace infernix::ops::inkling2 {
namespace {

namespace d = ::infernix::ops::detail;

struct MmaSegment {
    const __nv_bfloat16* weight;
    const __nv_bfloat16* x;
    void* out;
    std::int32_t rows;
};

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

template <class Sched, class Out>
__global__ void __launch_bounds__(Sched::kThreads, Sched::kMinBlocksPerSm) projections_mma_kernel(const MmaParams p) {
    constexpr int BM = Sched::kBlockRows, BN = Sched::kBlockTokens, BK = Sched::kBlockK, S = Sched::kStages;
    constexpr int THREADS = Sched::kThreads;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* As = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* Bs = As + S * BM * BK;

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
        d::bf16_mma_compute_stage<Sched>(As + stage * BM * BK, Bs + stage * BN * BK, accum, warp, lane);
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

template <class Sched, class Out>
void launch(const MmaParams& p, std::int32_t tiles, cudaStream_t stream) {
    constexpr auto kernel = projections_mma_kernel<Sched, Out>;
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
template <class Out>
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
        launch<Sched, Out>(q, tiles_out, stream);
    } else if (a.columns <= 16) {
        using Sched = Schedule<16, 16, 16, 16, 8, 4>;
        tiles_out   = tiles_for(Sched::kBlockRows, q);
        launch<Sched, Out>(q, tiles_out, stream);
    } else if (small) {
        using Sched = Schedule<64, 64, 32, 32, 3, 2>;
        tiles_out   = tiles_for(Sched::kBlockRows, q);
        launch<Sched, Out>(q, tiles_out, stream);
    } else {
        using Sched = Schedule<128, 128, 64, 32, 3, 1>;
        tiles_out   = tiles_for(Sched::kBlockRows, q);
        launch<Sched, Out>(q, tiles_out, stream);
    }
}

} // namespace

void projections_mma(const ProjectionArgs& a, cudaStream_t stream) {
    if (a.columns <= 0 || a.k <= 0 || a.k % 64 != 0 || a.count <= 0 || a.count > kMaxProjectionSegments) {
        throw std::invalid_argument("inkling2 projections_mma: 1-8 segments, positive columns, K a multiple of 64");
    }
    MmaParams p{};
    p.count   = a.count;
    p.k       = a.k;
    p.columns = a.columns;
    std::int32_t widest = 0;
    for (int s = 0; s < a.count; ++s) {
        const ProjectionSegment& g = a.segments[s];
        const auto* x              = (g.input >= 0 && g.input <= 2) ? a.x[g.input] : nullptr;
        const std::size_t word     = a.fp32_out ? 4 : 2;
        if (g.weight == nullptr || g.codes != nullptr || g.rows <= 0 || x == nullptr || g.out == nullptr ||
            reinterpret_cast<std::uintptr_t>(g.weight) % 16 != 0 || reinterpret_cast<std::uintptr_t>(x) % 16 != 0 ||
            reinterpret_cast<std::uintptr_t>(g.out) % 16 != 0 || (g.rows * word) % 16 != 0) {
            throw std::invalid_argument("inkling2 projections_mma: BF16 segments only, with 16-byte aligned weights, "
                                        "inputs, outputs and output rows");
        }
        p.seg[s] = {reinterpret_cast<const __nv_bfloat16*>(g.weight), reinterpret_cast<const __nv_bfloat16*>(x), g.out,
                    g.rows};
        widest = std::max(widest, g.rows);
    }
    std::int32_t tiles = 0;
    const bool small   = widest <= 256;
    a.fp32_out ? launch_shape<float>(p, small, stream, tiles, a) : launch_shape<__nv_bfloat16>(p, small, stream, tiles, a);
}

} // namespace infernix::ops::inkling2
