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
#include "core/tensor.h"
#include "ops/common/dynamic_shared.cuh"
#include "ops/common/memory.cuh"
#include "ops/ggml/ggml_a8.h"
#include "ops/ggml/ggml_a8_mma.cuh"
#include "ops/ggml/ggml_linear_a8.h"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16/bf16_schedule.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
    // A8 (with Q8 storage): the input's encoded columns (a8_encode_columns), MmaParams::x8_stride words apart.
    const std::int32_t* x8 = nullptr;
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
    std::int32_t x8_stride = 0; // A8: words between encoded columns
};

template <int BM, int BN, int WM, int WN, int S, int MinBlocks>
using Schedule = d::Bf16A16MmaSchedule<BM, BN, 64, WM, WN, S, MinBlocks, Cache::cg, Cache::cg,
                                       d::Bf16MmaFragmentPipeline::PingPong, d::Bf16MmaRaster::TokenFast,
                                       d::Bf16MmaSwizzle::Xor64>;

// The output tile staged as FP32 [BN][BM + 4] (the padding spreads a fragment's 8 rows over the banks), in one pass or,
// a tile wider than kStagingLimit allows, two passes of half its columns.
constexpr int kStagingLimit = 96 * 1024;
template <class Sched>
inline constexpr int kStorePasses = Sched::kBlockTokens * (Sched::kBlockRows + 4) * 4 <= kStagingLimit ? 1 : 2;
template <class Sched>
constexpr int staging_bytes() {
    return Sched::kBlockTokens / kStorePasses<Sched> * (Sched::kBlockRows + 4) * 4;
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

// The tile through shared memory (every copy landed and every thread past its reads): FP32 [BN][BM + 4], then 16-byte
// rows of the [T][rows] output. Accumulator (mi, ni, q) of warp (wm, wn): row wm WM + 16 mi + g + 8 (q >> 1), column
// wn WN + 8 ni + 2t + (q & 1).
template <class Sched, class Out>
__device__ __forceinline__ void store_tile(unsigned char* shared_raw,
                                           const float (&accum)[Sched::kMmaRows][Sched::kMmaTokens][4],
                                           const MmaSegment& seg, int m0, int n0, int T) {
    constexpr int BM = Sched::kBlockRows, BN = Sched::kBlockTokens, THREADS = Sched::kThreads, LD = BM + 4;
    constexpr int kPasses = kStorePasses<Sched>, CW = BN / kPasses; // columns a pass stages
    static_assert(Sched::kWarpsTokens % kPasses == 0, "a pass stages whole warps' columns");
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int M      = seg.rows;
    float* tile_out  = reinterpret_cast<float*>(shared_raw);
    const int wm = warp / Sched::kWarpsTokens, wn = warp % Sched::kWarpsTokens;
    const int g = lane >> 2, t = lane & 3;
    constexpr int kPer = std::is_same_v<Out, float> ? 4 : 8; // values a 16-byte store
    constexpr int kChunks = BM / kPer;
    Out* out = static_cast<Out*>(seg.out);
#pragma unroll 1
    for (int pass = 0; pass < kPasses; ++pass) {
        if (pass > 0) { __syncthreads(); } // the previous pass's stores have read the staging
        if (wn * Sched::kWarpTokens / CW == pass) {
#pragma unroll
            for (int mi = 0; mi < Sched::kMmaRows; ++mi) {
#pragma unroll
                for (int ni = 0; ni < Sched::kMmaTokens; ++ni) {
                    const int r = wm * Sched::kWarpRows + mi * 16 + g;
                    const int c = wn * Sched::kWarpTokens + ni * 8 + 2 * t - pass * CW;
                    tile_out[c * LD + r]           = accum[mi][ni][0];
                    tile_out[(c + 1) * LD + r]     = accum[mi][ni][1];
                    tile_out[c * LD + r + 8]       = accum[mi][ni][2];
                    tile_out[(c + 1) * LD + r + 8] = accum[mi][ni][3];
                }
            }
        }
        __syncthreads();
        const int c0 = n0 + pass * CW;
        // Output rows of a 16-byte multiple from a 16-byte aligned start take 16-byte stores; others (the head's FP32
        // logits rows) one value a store.
        const bool vector = (static_cast<std::size_t>(M) * sizeof(Out)) % 16 == 0 &&
                            reinterpret_cast<std::uintptr_t>(seg.out) % 16 == 0;
#pragma unroll 1
        for (int item = tid; item < CW * kChunks; item += THREADS) {
            const int c = item / kChunks, r = (item - c * kChunks) * kPer;
            if (c0 + c >= T || m0 + r >= M) { continue; }
            const float* v = tile_out + c * LD + r;
            Out* dst       = out + static_cast<std::size_t>(c0 + c) * M + m0 + r;
            if (!vector) {
                for (int i = 0; i < kPer && m0 + r + i < M; ++i) {
                    if constexpr (std::is_same_v<Out, float>) {
                        dst[i] = v[i];
                    } else {
                        dst[i] = __float2bfloat16_rn(v[i]);
                    }
                }
            } else if constexpr (std::is_same_v<Out, float>) {
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
    store_tile<Sched, Out>(shared_raw, accum, seg, m0, n0, T);
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

// ---------------------------------------------------------------------------- 8-bit activations (A8)

// projections_mma over Q8 segments with 8-bit inputs (ProjectionArgs::x8, `--a8-activations`: LOSSY, opt-in): the wide
// 8-bit route's arithmetic (ops/ggml/ggml_a8_tile.cuh) over Q8_G32_FP16 rows read in place. Per 32-slice s of K, P_s is
// the exact int32 dot of the row's codes with the column's (a8_encode_columns: ops/ggml/ggml_a8.h's codec), w_s the
// row's FP16 group scale and d_s the column's slice scale: acc = +0, then for s = 0 .. K / 32 - 1 in order
// acc = fma(fl32(P_s w_s), d_s, acc), with fl32(P_s w_s) from an MMA started at M = 1.5 2^23 (ggml_a8_mma.cuh: exact,
// since an FP16 scale has at most 11 significant bits, and no conversion instruction); the output is acc rounded once.
// One thread adds an output's terms, so its bits depend only on its row, its column and K: not on the call's width,
// other columns or segments, nor on the tile shape.
//
// A CTA computes BM rows x BN columns with warps of WM x WN, over a cp.async ring of S stages of BK = 128 elements of K
// (four slices; K a multiple of 64, a last stage of two slices zero-filling the other two). A stage holds the rows' codes [BM][128] and the columns' [BN][128], each row's or column's 128 bytes
// one line of global memory (one L2 request: 2026-11-23, ncu, K2 at 4,096 columns, a 64-element stage split into
// slice-major halves and 4-byte scale copies issued ~1 sector a request and held L2's tags at 80 % while its data path
// ran at 21 %), 16-byte chunks XOR-swizzled by (row & 3) << 1 so that the 8 bytes a lane reads of 8 rows cover the banks
// once; then the rows' four FP16 scales [BM][4] and the columns' four FP32 scales [BN][4] (one 8- and one 16-byte copy).
// The MMA takes the ggml tiles' permuted K labelling: lane (g, t) holds bytes 8t .. 8t + 7 of a slice of rows g and
// g + 8 (A: row g's word 2t, row g + 8's, row g's word 2t + 1, row g + 8's) and of column g (B), one 8-byte load each.
template <int BM_, int BN_, int WM_, int WN_, int S_, int MinBlocks_>
struct A8Schedule {
    static constexpr int kBlockRows = BM_, kBlockTokens = BN_, kWarpRows = WM_, kWarpTokens = WN_, kStages = S_;
    static constexpr int kMinBlocksPerSm = MinBlocks_;
    static constexpr int kWarpsTokens    = BN_ / WN_;
    static constexpr int kThreads        = 32 * (BM_ / WM_) * kWarpsTokens;
    static constexpr int kMmaRows = WM_ / 16, kMmaTokens = WN_ / 8;
    static constexpr int kBlockK = 128, kSlices = kBlockK / 32;
    static constexpr int kRowCodes = 0, kColumnCodes = BM_ * kBlockK, kRowScales = kColumnCodes + BN_ * kBlockK;
    static constexpr int kColumnScales = kRowScales + BM_ * 2 * kSlices, kStageBytes = kColumnScales + BN_ * 4 * kSlices;
    static_assert(BM_ % WM_ == 0 && BN_ % WN_ == 0 && WM_ % 16 == 0 && WN_ % 8 == 0 && kStageBytes % 16 == 0);
};

// Byte offset of 16-byte chunk c of row r in a stage's [rows][128] codes.
__device__ __forceinline__ int a8_chunk(int r, int c) { return r * 128 + ((c ^ ((r & 3) << 1)) << 4); }

template <class Sched>
constexpr int a8_shared_bytes() {
    return std::max(Sched::kStages * Sched::kStageBytes, staging_bytes<Sched>());
}

template <class Sched, class Out>
__global__ void __launch_bounds__(Sched::kThreads, Sched::kMinBlocksPerSm) projections_a8_kernel(const MmaParams p) {
    namespace a8 = ::infernix::ggml::a8;
    constexpr int BM = Sched::kBlockRows, BN = Sched::kBlockTokens, S = Sched::kStages, THREADS = Sched::kThreads;
    constexpr int WM = Sched::kWarpRows, WN = Sched::kWarpTokens, MT = Sched::kMmaRows, NT = Sched::kMmaTokens;
    constexpr int BK = Sched::kBlockK, SL = Sched::kSlices, CH = BK / 16; // chunks a row
    extern __shared__ __align__(16) unsigned char shared_raw[];

    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int tile = static_cast<int>(blockIdx.x);
    int s          = 0;
    while (s + 1 < p.count && tile >= p.tile_begin[s + 1]) { ++s; }
    const MmaSegment seg = p.seg[s];
    const int K = p.k, M = seg.rows, T = p.columns;
    const int m0 = (tile - p.tile_begin[s]) * BM;
    const int n0 = static_cast<int>(blockIdx.y) * BN;
    const int kTiles = (K + BK - 1) / BK; // K a multiple of 64: the last stage may hold two slices
    const std::size_t column_bytes = static_cast<std::size_t>(p.x8_stride) * 4;
    const auto* x8                 = reinterpret_cast<const std::uint8_t*>(seg.x8);

    // Weights first (immutable: before the PDL wait), then the inputs of the same stages. Rows past the matrix, columns
    // past the call and slices past K read zeros (codes and scales: their terms are +0, which leave acc's bits).
    const auto stage_weights = [&](int stage, int k_tile) {
        unsigned char* base = shared_raw + stage * Sched::kStageBytes;
        const int k0 = k_tile * BK, kv = min(BK, K - k0);
#pragma unroll 1
        for (int item = tid; item < BM * CH; item += THREADS) { // chunk item % CH of row item / CH
            const int row = item / CH, c = item % CH;
            const bool valid    = m0 + row < M;
            const std::size_t r = valid ? static_cast<std::size_t>(m0 + row) : 0;
            ops::cp_async_zfill<16, Cache::cg>(base + Sched::kRowCodes + a8_chunk(row, c),
                                              seg.codes + r * seg.stride + k0 + 16 * c, valid && 16 * c < kv ? 16 : 0);
        }
        for (int row = tid; row < BM; row += THREADS) {
            const bool valid    = m0 + row < M;
            const std::size_t r = valid ? static_cast<std::size_t>(m0 + row) : 0;
            ops::cp_async_zfill<2 * SL>(base + Sched::kRowScales + row * 2 * SL,
                                        seg.scales + r * (seg.stride / 32) + k0 / 32, valid ? kv / 16 : 0);
        }
    };
    const auto stage_inputs = [&](int stage, int k_tile) {
        unsigned char* base = shared_raw + stage * Sched::kStageBytes;
        const int k0 = k_tile * BK, kv = min(BK, K - k0);
#pragma unroll 1
        for (int item = tid; item < BN * CH; item += THREADS) {
            const int col = item / CH, c = item % CH;
            const bool valid           = n0 + col < T;
            const std::uint8_t* column = x8 + (valid ? static_cast<std::size_t>(n0 + col) : 0) * column_bytes;
            ops::cp_async_zfill<16, Cache::cg>(base + Sched::kColumnCodes + a8_chunk(col, c), column + k0 + 16 * c,
                                              valid && 16 * c < kv ? 16 : 0);
        }
        for (int col = tid; col < BN; col += THREADS) { // the stage's slice scales (after the column's K codes)
            const bool valid           = n0 + col < T;
            const std::uint8_t* column = x8 + (valid ? static_cast<std::size_t>(n0 + col) : 0) * column_bytes;
            ops::cp_async_zfill<4 * SL>(base + Sched::kColumnScales + col * 4 * SL, column + K + k0 / 8,
                                        valid ? kv / 8 : 0);
        }
    };

    // The ring as projections_mma_kernel's: S groups of weights, then (after the PDL wait) one group of the first
    // stages' inputs; a step's refill is committed S steps before its stage is read.
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

    const int wm = warp / Sched::kWarpsTokens, wn = warp % Sched::kWarpsTokens;
    const int g = lane >> 2, t = lane & 3;
    float accum[MT][NT][4] = {};
#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile == 0) {
            ops::cp_wait<0>();
        } else {
            ops::cp_wait<S - 1>();
        }
        __syncthreads();
        const unsigned char* base = shared_raw + stage * Sched::kStageBytes;
        // The scales of the stage: rows g and g + 8 of each m16 tile (four FP16 each), columns 2t and 2t + 1 of each n8
        // tile (four FP32 each).
        uint2 ws[MT][2];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int r = wm * WM + 16 * mi + g;
            ws[mi][0]   = *reinterpret_cast<const uint2*>(base + Sched::kRowScales + r * 2 * SL);
            ws[mi][1]   = *reinterpret_cast<const uint2*>(base + Sched::kRowScales + (r + 8) * 2 * SL);
        }
        float4 xs[NT][2];
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int c = wn * WN + 8 * ni + 2 * t;
            xs[ni][0]   = *reinterpret_cast<const float4*>(base + Sched::kColumnScales + c * 4 * SL);
            xs[ni][1]   = *reinterpret_cast<const float4*>(base + Sched::kColumnScales + (c + 1) * 4 * SL);
        }
        const auto lane_bytes = [&](int region, int row, int j) { // bytes 8t .. 8t + 7 of slice j of a row
            return *reinterpret_cast<const int2*>(base + region + a8_chunk(row, 2 * j + (t >> 1)) + 8 * (t & 1));
        };
        const auto half_of = [](const uint2& w, int j) { // the FP16 scale of slice j
            const unsigned word = j < 2 ? w.x : w.y;
            return __half2float(__ushort_as_half(static_cast<unsigned short>((j & 1) == 0 ? word & 0xFFFFU : word >> 16)));
        };
        const auto lane_scale = [](const float4& v, int j) { return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w; };
#pragma unroll
        for (int j = 0; j < SL; ++j) {
            int2 b[NT];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) { b[ni] = lane_bytes(Sched::kColumnCodes, wn * WN + 8 * ni + g, j); }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int r     = wm * WM + 16 * mi + g;
                const int2 lo   = lane_bytes(Sched::kRowCodes, r, j);
                const int2 hi   = lane_bytes(Sched::kRowCodes, r + 8, j);
                const float w0  = half_of(ws[mi][0], j), w1 = half_of(ws[mi][1], j);
                const float mw0 = __fmul_rn(a8::kMinusMagic, w0), mw1 = __fmul_rn(a8::kMinusMagic, w1);
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    int q[4];
                    a8::mma_k32_magic(q, lo.x, hi.x, lo.y, hi.y, b[ni].x, b[ni].y);
                    const float x0 = lane_scale(xs[ni][0], j), x1 = lane_scale(xs[ni][1], j);
                    accum[mi][ni][0] = __fmaf_rn(a8::scaled_term(q[0], w0, mw0), x0, accum[mi][ni][0]);
                    accum[mi][ni][1] = __fmaf_rn(a8::scaled_term(q[1], w0, mw0), x1, accum[mi][ni][1]);
                    accum[mi][ni][2] = __fmaf_rn(a8::scaled_term(q[2], w1, mw1), x0, accum[mi][ni][2]);
                    accum[mi][ni][3] = __fmaf_rn(a8::scaled_term(q[3], w1, mw1), x1, accum[mi][ni][3]);
                }
            }
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
    store_tile<Sched, Out>(shared_raw, accum, seg, m0, n0, T);
}
// A call's distinct 8-bit inputs (up to three) encoded in one launch: a8_encode_columns' codec and layout
// (ops/ggml/ggml_a8.h: per 32-slice the scale of its largest magnitude and the codes; a column's K codes, then its
// K / 32 FP32 scales, `stride` words apart). Four threads a slice, thread c taking elements 8c .. 8c + 7 (one 16-byte
// load); the slice's maximum is the quad's.
struct A8Inputs {
    const std::uint16_t* x[3] = {};
    std::int32_t* out[3]      = {};
    int count                 = 0;
};

constexpr int kEncodeThreads = 256;

__global__ void __launch_bounds__(kEncodeThreads) encode_inputs_kernel(const A8Inputs in, int k, int columns, int stride) {
    namespace a8     = ::infernix::ggml::a8;
    const int slices = k / 32;
    const long long per = static_cast<long long>(columns) * slices * 4; // threads an input (a multiple of 4)
    const long long i   = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int input     = static_cast<int>(i / per);
    const bool live     = input < in.count;
    const long long r   = live ? i - input * per : 0;
    const int item = static_cast<int>(r / 4), c = static_cast<int>(r % 4);
    const int t = item / slices, s = item % slices;
    pdl::wait_for_dependencies();
    uint4 u = make_uint4(0, 0, 0, 0);
    if (live) { u = *reinterpret_cast<const uint4*>(in.x[input] + static_cast<std::size_t>(t) * k + 32 * s + 8 * c); }
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
    std::uint16_t e[8];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        e[2 * m]     = static_cast<std::uint16_t>(w[m] & 0xFFFFU);
        e[2 * m + 1] = static_cast<std::uint16_t>(w[m] >> 16);
    }
    unsigned key = 0;
#pragma unroll
    for (int m = 0; m < 8; ++m) { key = max(key, static_cast<unsigned>(e[m] & 0x7FFFU)); }
    key = max(key, __shfl_xor_sync(0xFFFFFFFFU, key, 1));
    key = max(key, __shfl_xor_sync(0xFFFFFFFFU, key, 2));
    const a8::SliceScale sc = a8::slice_scale(key);
    unsigned packed[2] = {0, 0};
#pragma unroll
    for (int m = 0; m < 8; ++m) { packed[m / 4] |= (static_cast<unsigned>(a8::code(e[m], sc.inv)) & 0xFFU) << (8 * (m % 4)); }
    pdl::trigger_dependents();
    if (live) {
        std::int32_t* column = in.out[input] + static_cast<std::size_t>(t) * stride;
        *reinterpret_cast<uint2*>(column + s * a8::kSliceWords + 2 * c) = make_uint2(packed[0], packed[1]);
        if (c == 0) { reinterpret_cast<float*>(column + a8::scales_offset(k))[s] = sc.d; }
    }
}

void encode_inputs(const A8Inputs& in, int k, int columns, int stride, cudaStream_t stream) {
    for (int i = 0; i < in.count; ++i) {
        if (reinterpret_cast<std::uintptr_t>(in.x[i]) % 16 != 0) {
            throw std::invalid_argument("inkling2 projections_mma: 8-bit inputs need 16-byte aligned BF16 inputs");
        }
    }
    const long long threads = static_cast<long long>(in.count) * columns * (k / 32) * 4;
    const dim3 grid(static_cast<unsigned>((threads + kEncodeThreads - 1) / kEncodeThreads));
    CUDA_CHECK(pdl::launch_consumer({grid, dim3(kEncodeThreads), 0, stream}, encode_inputs_kernel, in, k, columns, stride));
}

// Each segment's first row tile of BM rows (grid x) into q; returns the call's tiles.
std::int32_t segment_tiles(MmaParams& q, const ProjectionArgs& a, int bm) {
    std::int32_t tiles = 0;
    for (int s = 0; s < a.count; ++s) {
        q.tile_begin[s] = tiles;
        tiles += (a.segments[s].rows + bm - 1) / bm;
    }
    q.tile_begin[a.count] = tiles;
    return tiles;
}

template <class Sched, class Out>
void launch_a8(MmaParams q, const ProjectionArgs& a, cudaStream_t stream) {
    constexpr auto kernel = projections_a8_kernel<Sched, Out>;
    const int dynamic     = d::prepare_dynamic_shared<a8_shared_bytes<Sched>(), kernel, true>();
    const std::int32_t tiles = segment_tiles(q, a, Sched::kBlockRows);
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>((q.columns + Sched::kBlockTokens - 1) / Sched::kBlockTokens));
    const pdl::LaunchConfig config{grid, dim3(Sched::kThreads), static_cast<std::size_t>(dynamic), stream};
    CUDA_CHECK(pdl::launch_consumer(config, kernel, q));
}

// A8 tile shapes (the same bits in each): calls of up to 16 columns 32 x 16 (two warps of 16 x 16, five stages, three
// CTAs an SM: the weights' stream), up to 128 columns or few-row matrices 64 x 64 (four warps of 32 x 32, two CTAs an
// SM), wider calls 128 x 128 (eight warps of 64 x 32, two stages).
template <class Out>
void launch_a8_shape(const MmaParams& p, const ProjectionArgs& a, bool small, cudaStream_t stream) {
    if (a.columns <= 16) {
        // 2026-11-23 (decode rounds at context 512, --a8-activations, one / two / three / eight columns): five stages,
        // three CTAs an SM, so more of K2's, W_o's and the head's weights load before the PDL wait: 4.1129 -> 4.0950 ms,
        // -0.44 / -0.56 / -0.21 % wider; six stages and two CTAs -0.37 / -0.54 / -0.74 / -0.10 %; 16-row tiles (six or
        // eight stages, four CTAs) +3 to +9 %.
        launch_a8<A8Schedule<32, 16, 16, 16, 5, 3>, Out>(p, a, stream);
    } else if (small || a.columns <= 128) {
        launch_a8<A8Schedule<64, 64, 32, 32, 2, 2>, Out>(p, a, stream);
    } else {
        // 2026-11-23 (forward test prefill profile, 4,096 columns; K2 / W_o ms): this 24.35 / 10.24; 16 warps of 32 x 32
        // 26.9 / 11.3; 128 x 64 and 64 x 128 in three stages 27.6 / 10.6 and 28.0 / 10.8.
        launch_a8<A8Schedule<128, 128, 64, 32, 2, 1>, Out>(p, a, stream);
    }
}

// ---------------------------------------------------------------------------- decode-width MMA projections

// projections_dmma (calls of up to 16 columns): a CTA per 16-row tile of Splits warps; warp q accumulates the q-th of
// Splits equal runs of K's 32-blocks as a chain of m16n8k16 MMAs from zero, blocks in ascending order, two MMAs a block
// (k-step 0: elements 8t .. 8t + 3 of the block into slots 2t, 2t + 1, 2t + 8, 2t + 9, t = lane % 4; k-step 1: elements
// 8t + 4 .. 8t + 7), so a thread's weights and inputs of a block are one 16-byte load each; then the runs' partials
// added in order. A column's bits depend on its row and its input alone (the same at every width up to 16).
constexpr int kDmmaAhead = 4; // blocks in flight a warp

// Q8: the thread's 8 codes of a 32-block (elements 8t .. 8t + 7) as the BF16 words of the k-slots BF16 storage would
// hold (codes are exact in BF16).
__device__ __forceinline__ uint4 q8_words(uint2 c) {
    const std::uint32_t b[2] = {c.x, c.y};
    std::uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t pair = b[i / 2] >> (16 * (i % 2));
        const auto lo = static_cast<float>(static_cast<std::int8_t>(pair & 0xFFU));
        const auto hi = static_cast<float>(static_cast<std::int8_t>((pair >> 8) & 0xFFU));
        w[i]          = std::uint32_t{__bfloat16_as_ushort(__float2bfloat16_rn(lo))} |
               (std::uint32_t{__bfloat16_as_ushort(__float2bfloat16_rn(hi))} << 16);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// BF16 storage: a block's two MMAs chain into acc. Q8 storage: they run from zero over the codes, and
// acc = fma(the block's sum, the row's group scale, acc) (projections_mma's Q8 form at a block a group).
// Ahead: blocks in flight a warp, the first Ahead fetched before the PDL wait (launch_dmma's choice).
template <int Splits, int NT, class Out, bool Q8, int Ahead>
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
    const uint4* w0 = nullptr;
    const uint4* w1 = nullptr;
    const uint2 *c0 = nullptr, *c1 = nullptr;     // Q8: the rows' codes (8 bytes a thread a block)
    const std::uint16_t *s0 = nullptr, *s1 = nullptr; // Q8: the rows' group scales
    if constexpr (Q8) {
        c0 = reinterpret_cast<const uint2*>(seg.codes + static_cast<std::size_t>(r0) * seg.stride) + t;
        c1 = reinterpret_cast<const uint2*>(seg.codes + static_cast<std::size_t>(r1) * seg.stride) + t;
        s0 = seg.scales + static_cast<std::size_t>(r0) * (seg.stride / 32);
        s1 = seg.scales + static_cast<std::size_t>(r1) * (seg.stride / 32);
    } else {
        w0 = reinterpret_cast<const uint4*>(seg.weight + static_cast<std::size_t>(r0) * K) + t;
        w1 = reinterpret_cast<const uint4*>(seg.weight + static_cast<std::size_t>(r1) * K) + t;
    }
    // Q8 codes and scales stay raw until their block's MMAs, so a fetch never waits on its own load.
    uint4 a0[Ahead], a1[Ahead];
    uint2 q0[Ahead], q1[Ahead];
    unsigned short h0[Ahead], h1[Ahead];
    const auto fetch = [&](int i, int b) {
        if constexpr (Q8) {
            q0[i] = __ldcs(c0 + 4 * b);
            q1[i] = __ldcs(c1 + 4 * b);
            h0[i] = __ldg(s0 + b);
            h1[i] = __ldg(s1 + b);
        } else {
            a0[i] = __ldcs(w0 + 4 * b);
            a1[i] = __ldcs(w1 + 4 * b);
        }
    };
#pragma unroll
    for (int i = 0; i < Ahead; ++i) {
        if (i < blocks) { fetch(i, b0 + i); }
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
    for (int i = 0; i < blocks; i += Ahead) {
#pragma unroll
        for (int j = 0; j < Ahead; ++j) {
            const int b = i + j;
            if (b >= blocks) { break; }
            uint4 xv[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) { xv[n] = live[n] ? x[n][4 * (b0 + b)] : make_uint4(0U, 0U, 0U, 0U); }
            uint4 wa{}, wb{};
            float sa = 0.0F, sb = 0.0F;
            if constexpr (Q8) {
                wa = q8_words(q0[j]);
                wb = q8_words(q1[j]);
                sa = __half2float(__ushort_as_half(h0[j]));
                sb = __half2float(__ushort_as_half(h1[j]));
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                if constexpr (Q8) {
                    float gs[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                    mma_bf16(gs[0], gs[1], gs[2], gs[3], wa.x, wb.x, wa.y, wb.y, xv[n].x, xv[n].y);
                    mma_bf16(gs[0], gs[1], gs[2], gs[3], wa.z, wb.z, wa.w, wb.w, xv[n].z, xv[n].w);
                    acc[n][0] = __fmaf_rn(gs[0], sa, acc[n][0]);
                    acc[n][1] = __fmaf_rn(gs[1], sa, acc[n][1]);
                    acc[n][2] = __fmaf_rn(gs[2], sb, acc[n][2]);
                    acc[n][3] = __fmaf_rn(gs[3], sb, acc[n][3]);
                } else {
                    mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], a0[j].x, a1[j].x, a0[j].y, a1[j].y, xv[n].x, xv[n].y);
                    mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], a0[j].z, a1[j].z, a0[j].w, a1[j].w, xv[n].z, xv[n].w);
                }
            }
            const int next = b + Ahead;
            if (next < blocks) { fetch(j, b0 + next); }
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

// Blocks in flight a warp: kDmmaAhead of BF16 storage, kDmmaQ8Ahead of Q8 (half the bytes a block; 12 against 8:
// 2026-11-23, --q8-weights decode rounds at context 512 -0.79 / -0.37 / -0.59 / -0.28 % at one / two / three / eight
// columns; 16 measured +0.09 to +0.46 %, but at 16 the 16-run instance, W_o's, spills to local memory, so that arm is
// not a fair reading of depth 16 for K2's 4-run one); a BF16 launch of at most kDmmaDeepWarps warps (a few tiles: the
// loop's message-gate downs, 12 tiles of 16 runs) kDmmaDeep, so most of its weights load before the PDL wait, beside the
// kernel before it, on the few SMs it occupies.
constexpr int kDmmaQ8Ahead   = 12;
constexpr int kDmmaDeep      = 8;
constexpr int kDmmaDeepWarps = 512;

template <int Splits, int NT, class Out>
void launch_dmma(const MmaParams& p, std::int32_t tiles, cudaStream_t stream) {
    const pdl::LaunchConfig config{dim3(static_cast<unsigned>(tiles)), dim3(Splits * 32), 0, stream};
    if (p.seg[0].codes != nullptr) {
        CUDA_CHECK(pdl::launch_consumer(config, projections_dmma_kernel<Splits, NT, Out, true, kDmmaQ8Ahead>, p));
    } else if (tiles * Splits <= kDmmaDeepWarps) {
        CUDA_CHECK(pdl::launch_consumer(config, projections_dmma_kernel<Splits, NT, Out, false, kDmmaDeep>, p));
    } else {
        CUDA_CHECK(pdl::launch_consumer(config, projections_dmma_kernel<Splits, NT, Out, false, kDmmaAhead>, p));
    }
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
    // One storage for the call: BF16, or Q8 (codes 8-byte aligned a row, stride a multiple of 128).
    const bool q8 = a.segments[0].q8_scales != nullptr;
    for (int s = 0; s < a.count; ++s) {
        const ProjectionSegment& g = a.segments[s];
        const auto* x              = (g.input >= 0 && g.input <= 2) ? a.x[g.input] : nullptr;
        const std::size_t word     = a.fp32_out ? 4 : 2;
        const bool storage = q8 ? (g.weight == nullptr && g.codes != nullptr && g.q8_scales != nullptr &&
                                   g.nibbles == nullptr && g.q8_stride >= a.k && g.q8_stride % 128 == 0 &&
                                   reinterpret_cast<std::uintptr_t>(g.codes) % 16 == 0)
                                : (g.weight != nullptr && g.codes == nullptr &&
                                   reinterpret_cast<std::uintptr_t>(g.weight) % 16 == 0);
        if (!storage || g.rows <= 0 || x == nullptr || g.out == nullptr || reinterpret_cast<std::uintptr_t>(x) % 16 != 0 ||
            reinterpret_cast<std::uintptr_t>(g.out) % 16 != 0 || (g.rows * word) % 16 != 0) {
            throw std::invalid_argument(std::string("inkling2 ") + what +
                                        ": BF16 or Q8 segments (one storage a call), with 16-byte aligned weights, inputs, "
                                        "outputs and output rows");
        }
        p.seg[s] = {reinterpret_cast<const __nv_bfloat16*>(g.weight), reinterpret_cast<const __nv_bfloat16*>(x), g.out,
                    g.rows};
        if (q8) {
            p.seg[s].codes  = static_cast<const std::uint8_t*>(g.codes);
            p.seg[s].scales = g.q8_scales;
            p.seg[s].stride = g.q8_stride;
        }
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

std::int32_t projection_a8_column_words(std::int32_t k) { return a8_linear_column_words(k, false); }

void projections_mma(const ProjectionArgs& a, cudaStream_t stream) {
    if (a.columns <= 0 || a.k <= 0 || a.k % 64 != 0 || a.count <= 0 || a.count > kMaxProjectionSegments) {
        throw std::invalid_argument("inkling2 projections_mma: 1-8 segments, positive columns, K a multiple of 64");
    }
    MmaParams p{};
    p.count   = a.count;
    p.k       = a.k;
    p.columns = a.columns;
    std::int32_t widest = 0;
    // One storage for the call: BF16, or Q8 (codes and scales, stride a multiple of 128). 8-bit inputs take output rows
    // of any length (the head's FP32 logits rows).
    const bool q8 = a.segments[0].q8_scales != nullptr;
    const bool a8 = a.x8[0] != nullptr || a.x8[1] != nullptr || a.x8[2] != nullptr;
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
        const bool out_aligned = reinterpret_cast<std::uintptr_t>(g.out) % 16 == 0 && (g.rows * word) % 16 == 0;
        if (!storage || g.rows <= 0 || x == nullptr || g.out == nullptr || reinterpret_cast<std::uintptr_t>(x) % 16 != 0 ||
            reinterpret_cast<std::uintptr_t>(g.out) % word != 0 || (!a8 && !out_aligned)) {
            throw std::invalid_argument("inkling2 projections_mma: BF16 or Q8 segments (one storage a call), with 16-byte "
                                        "aligned weights and inputs, and (16-bit inputs) 16-byte aligned outputs and rows");
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
    if (a8) {
        // 8-bit inputs: each input a segment reads encoded once (an input equal to an earlier one shares its codes).
        if (!q8) { throw std::invalid_argument("inkling2 projections_mma: 8-bit inputs (x8) take Q8 segments"); }
        const std::int32_t words = projection_a8_column_words(a.k);
        const std::int32_t* encoded[3] = {};
        A8Inputs in{};
        for (int i = 0; i < 3; ++i) {
            bool used = false;
            for (int s = 0; s < a.count; ++s) { used = used || a.segments[s].input == i; }
            if (!used) { continue; }
            for (int j = 0; j < i && encoded[i] == nullptr; ++j) {
                if (a.x[j] == a.x[i] && encoded[j] != nullptr) { encoded[i] = encoded[j]; }
            }
            if (encoded[i] != nullptr) { continue; }
            if (a.x8[i] == nullptr || reinterpret_cast<std::uintptr_t>(a.x8[i]) % 16 != 0) {
                throw std::invalid_argument("inkling2 projections_mma: every input of an 8-bit call needs 16-byte aligned "
                                            "x8 workspace");
            }
            in.x[in.count]     = a.x[i];
            in.out[in.count++] = a.x8[i];
            encoded[i]         = a.x8[i];
        }
        if (!a.x8_encoded) { encode_inputs(in, a.k, a.columns, words, stream); } // the distinct inputs in one launch
        p.x8_stride = words;
        for (int s = 0; s < a.count; ++s) { p.seg[s].x8 = encoded[a.segments[s].input]; }
        a.fp32_out ? launch_a8_shape<float>(p, a, small, stream) : launch_a8_shape<__nv_bfloat16>(p, a, small, stream);
        return;
    }
    if (q8) {
        a.fp32_out ? launch_shape<float, true>(p, small, stream, tiles, a)
                   : launch_shape<__nv_bfloat16, true>(p, small, stream, tiles, a);
        return;
    }
    a.fp32_out ? launch_shape<float>(p, small, stream, tiles, a) : launch_shape<__nv_bfloat16>(p, small, stream, tiles, a);
}

} // namespace infernix::ops::inkling2
