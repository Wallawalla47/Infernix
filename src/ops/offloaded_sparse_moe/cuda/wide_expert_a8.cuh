#pragma once

// The 8-bit activation wide route of NVFP4 W4A16 experts (WideA16::kA8 with NarrowA16::kA8: `--a8-activations` at
// prefill widths, LOSSY, opt-in; wide_expert.h): wide_expert_a16.cuh's grouped, persistent, warp-specialized
// pipeline with x and h as ops/ggml/ggml_a8.h's 8-bit slices (the narrow 8-bit kernels' codes and slice scales)
// instead of exact A16 limbs.
//
// Arithmetic of one output (weight row r, column t; nvfp4_a8.h's terms in another rounding and order): with P_b
// the exact int32 product of block b's doubled E2M1 codes c2 with the column's codes (|P_b| <= 12 * 16 * 127),
// S_b = canon::e4m3_scaled of its scale word and d_s the scale of the slice s that holds it,
//
//   acc = +0; for b = 0 .. K/16 - 1 in order: acc = fma(P_b * S_b, d_s, acc)   (P_b * S_b exact: < 2^33, 19 bits)
//   y   = bf16_rn(fl32(fl32(acc * 2^-10) * m))                                (nvfp4_a8::row_output)
//
// so a column's output depends on its expert's record and its own codes alone (not on the call, the tile or the
// placement) but is not the narrow kernels' bits (they round fl32(P_b * d_s) before the fma and sum in partials).
// SwiGLU, h = bf16 and its 8-bit codes are the narrow route's boundaries; a pass's wide entries' h rows are rewritten
// in place as their codes (encode_h_a8_kernel, wide_expert.cu) before the down GEMM.
//
//   - A operand, activations: a token's (gate/up) or an entry's (down) 8-bit row, codes then slice scales: the
//     producer warp moves a row's 128 codes and 4 slice scales of a stage with two cp.async.bulk into a 144-byte
//     shared row (consecutive rows start in distinct bank quads, so the ldmatrix reads are conflict-free).
//   - B operand, weights: the record's 144-byte units bulk-copied as in the A16 route, read as the doubled codes.
//   - One m16n8k16 s8 MMA per 16-element block, its int32 accumulators started at 1.5 * 2^23
//     (ops/ggml/ggml_a8_mma.cuh), so P_b * S_b is one FFMA with S_b and -1.5 * 2^23 * S_b from a shared table, and
//     the term one more FFMA with the token's slice scale. Against the A16 route's two k32 MMAs and two int64
//     folds per block, half the tensor-core instructions.
//   - Eight consumer warps of 64 tokens x 16 rows (WarpTile64's shape): each decoded B fragment and scale pair
//     feeds four MMAs; a warp skips the 16-token MMA tiles past its work tile's columns (their outputs are never
//     written). Two stages of 128 K elements, two CTAs of 288 threads per SM (39 KB of shared memory each).

#include "ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh"

#include "ops/ggml/ggml_a8.h"
#include "ops/ggml/ggml_a8_mma.cuh"
#include "ops/offloaded_sparse_moe/nvfp4_a8.h"

namespace infernix::ops::offloaded_moe::wide {

struct ScheduleA8 {
    static constexpr int kBlockTokens     = kTileColumns; // 64
    static constexpr int kBlockRows       = 128;
    static constexpr int kRowGroups       = kBlockRows / 16; // 8
    static constexpr int kBlockK          = 128;
    static constexpr int kUnitsPerKTile   = kBlockK / 16; // 8 record units per row group
    static constexpr int kSlicesPerKTile  = kBlockK / 32; // 4 activation slice scales
    static constexpr int kCodeRowBytes    = kBlockK;      // a row's codes of a stage
    static constexpr int kScaleRowBytes   = 4 * kSlicesPerKTile;
    static constexpr int kRowStride       = kCodeRowBytes + kScaleRowBytes; // 144
    static constexpr int kStages          = 2;
    static constexpr int kMinBlocksPerSm  = 2;
    static constexpr int kWarpsTokens     = 1;
    static constexpr int kWarpsRows       = 8;
    static constexpr int kConsumerWarps   = kWarpsTokens * kWarpsRows;
    static constexpr int kConsumerThreads = kConsumerWarps * 32;
    static constexpr int kProducerThreads = 32;
    static constexpr int kThreads         = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpTokens      = kBlockTokens / kWarpsTokens; // 64
    static constexpr int kWarpRows        = kBlockRows / kWarpsRows;     // 16: one row group
    static constexpr int kMmaTokens       = kWarpTokens / 16;            // 4
    static constexpr int kMmaRows         = kWarpRows / 8;               // 2: lo/hi halves of the row group
    static constexpr int kUnitRunBytes    = kUnitsPerKTile * static_cast<int>(kUnitBytes); // 1,152
    static_assert(kRowStride % 16 == 0 && kUnitRunBytes % 16 == 0 && kWarpRows == 16);
};

struct StorageA8 {
    alignas(128) std::uint8_t a_rows[ScheduleA8::kStages][ScheduleA8::kBlockTokens * ScheduleA8::kRowStride];
    alignas(128) std::uint8_t b_units[ScheduleA8::kStages][ScheduleA8::kRowGroups * ScheduleA8::kUnitRunBytes];
    float2 scaled[256]; // {S, -1.5 * 2^23 * S} of every E4M3 scale word, S = canon::e4m3_scaled (both exact)
    alignas(8) std::uint64_t full[ScheduleA8::kStages];
    alignas(8) std::uint64_t empty[ScheduleA8::kStages];
};

static_assert(ScheduleA8::kMinBlocksPerSm * (sizeof(StorageA8) + 1024) <= 102400, "two A8 CTAs must fit an SM");

// A consumer thread's accumulators: acc[m][r][v] holds token tile_column(m, v) and weight row
// row_block * 128 + 16 warp_n + 8 r + 2 tig + (v & 1) (rows 2i and 2i + 1: gate_i and up_i of the gate/up matrix).
using AccumulatorsA8 = float[ScheduleA8::kMmaTokens][ScheduleA8::kMmaRows][4];

// The matrix a grouped kernel computes: its row groups and K blocks, its offset in a job's record, K, the bytes of
// an activation row and where a tile column's row is: by token (x, encoded once per call; entry_column with
// shared-expert jobs) or by entry (h, encoded per pass). A row holds K codes, then K / 32 binary32 slice scales.
template <class G, Matrix M, bool SharedJobs = false>
struct A8Matrix {
    static constexpr int kRowGroups      = M == Matrix::GateUp ? G::kGateUpRowGroups : G::kDownRowGroups;
    static constexpr int kBlocks         = M == Matrix::GateUp ? G::kGateUpBlocks : G::kDownBlocks;
    static constexpr std::size_t kOffset = M == Matrix::GateUp ? 0 : G::kGateUpBytes;
    static constexpr int kK              = M == Matrix::GateUp ? G::kHidden : G::kIntermediate;
    static constexpr int kRowBytes       = M == Matrix::GateUp ? a8_x_row_bytes(kShapeOf<G>) : A16Planes<G>::kHRowBytes;
    static constexpr bool kTokenRows     = M == Matrix::GateUp;
    static constexpr bool kSharedJobs    = SharedJobs;
    static_assert(kRowGroups % ScheduleA8::kRowGroups == 0 && kBlocks % ScheduleA8::kUnitsPerKTile == 0,
                  "expert matrices are whole 128-row blocks and 128-element stages");
    static_assert(kRowBytes >= kK + kK / 8 && kK % 16 == 0, "a row holds its codes, then its slice scales");
};

template <class Shape, class Epilogue>
__global__ __launch_bounds__(ScheduleA8::kThreads, ScheduleA8::kMinBlocksPerSm) void grouped_kernel_a8(
    const Call call, const std::int32_t pass, const Epilogue epilogue) {
    using S                  = ScheduleA8;
    constexpr int kRowBlocks = Shape::kRowGroups / S::kRowGroups;
    constexpr int kKTiles    = Shape::kBlocks / S::kUnitsPerKTile;
    namespace a8             = ::infernix::ggml::a8;

    extern __shared__ __align__(128) unsigned char a8_shared_bytes[];
    auto& shared = *reinterpret_cast<StorageA8*>(a8_shared_bytes);
    const int tile_begin = call.pass_tiles[pass];
    const int work       = (call.pass_tiles[pass + 1] - tile_begin) * kRowBlocks;
    if (static_cast<int>(blockIdx.x) >= work) { return; }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S::kStages; ++stage) {
            cta_mbarrier_init(&shared.full[stage], 1);
            cta_mbarrier_init(&shared.empty[stage], S::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    for (int w = static_cast<int>(threadIdx.x); w < 256; w += S::kThreads) {
        const float s     = static_cast<float>(canon::e4m3_scaled(static_cast<unsigned>(w)));
        shared.scaled[w] = make_float2(s, __fmul_rn(a8::kMinusMagic, s));
    }
    __syncthreads();

    if (threadIdx.x < S::kProducerThreads) {
        const int lane = static_cast<int>(threadIdx.x);
        int it         = 0;
#pragma unroll 1
        for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
            const int tile = tile_begin + w / kRowBlocks, row_block = w % kRowBlocks;
            const int job = call.tiles[2 * tile], column = call.tiles[2 * tile + 1];
            const int expert  = call.dispatch.jobs[job];
            const int first   = call.dispatch.offsets[expert] + column;
            const int columns = min(S::kBlockTokens, call.dispatch.offsets[expert + 1] - first);
            const std::uint8_t* matrix = call.job_records[job] + Shape::kOffset +
                                         static_cast<std::size_t>(row_block) * S::kRowGroups * Shape::kBlocks * kUnitBytes;
            // The activation row of each of this lane's tile columns (two per lane): x by token, h by entry.
            const std::uint8_t* rows[S::kBlockTokens / 32];
#pragma unroll
            for (int i = 0; i < S::kBlockTokens / 32; ++i) {
                const int c = lane + 32 * i;
                rows[i]     = nullptr;
                if (c < columns) {
                    int row;
                    if constexpr (!Shape::kTokenRows) {
                        row = first + c;
                    } else if constexpr (Shape::kSharedJobs) {
                        row = entry_column(call.dispatch.entries[first + c], call.top_k, call.dispatch.shared_first);
                    } else {
                        row = call.dispatch.entries[first + c] / call.top_k;
                    }
                    rows[i] = (Shape::kTokenRows ? call.x_limbs : call.h_rows) + static_cast<std::size_t>(row) * Shape::kRowBytes;
                }
            }
#pragma unroll 1
            for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
                const int stage = it % S::kStages;
                if (lane == 0) {
                    cta_mbarrier_wait(&shared.empty[stage], 1U ^ ((static_cast<unsigned>(it) / S::kStages) & 1U));
                    cta_mbarrier_arrive_expect_tx(&shared.full[stage], static_cast<std::uint32_t>(
                                                                           columns * S::kRowStride + S::kRowGroups * S::kUnitRunBytes));
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < S::kBlockTokens / 32; ++i) {
                    if (rows[i] != nullptr) {
                        std::uint8_t* dst = shared.a_rows[stage] + (lane + 32 * i) * S::kRowStride;
                        bulk_load(dst, rows[i] + k_tile * S::kCodeRowBytes, S::kCodeRowBytes, &shared.full[stage]);
                        bulk_load(dst + S::kCodeRowBytes, rows[i] + Shape::kK + k_tile * S::kScaleRowBytes, S::kScaleRowBytes,
                                  &shared.full[stage]);
                    }
                }
                if (lane < S::kRowGroups) {
                    bulk_load(shared.b_units[stage] + lane * S::kUnitRunBytes,
                              matrix + (static_cast<std::size_t>(lane) * Shape::kBlocks +
                                        static_cast<std::size_t>(k_tile) * S::kUnitsPerKTile) * kUnitBytes,
                              S::kUnitRunBytes, &shared.full[stage]);
                }
            }
        }
        return;
    }

    const int consumer = static_cast<int>(threadIdx.x) - S::kProducerThreads;
    const int lane = consumer & 31, warp = consumer >> 5;
    TileContext ctx{};
    ctx.warp_m = 0;
    ctx.warp_n = warp;
    ctx.group  = lane >> 2;
    ctx.tig    = lane & 3;
    const int g = ctx.group, t = ctx.tig;
    const int a_row = lane & 15; // ldmatrix.x2 of a 16 x 16-byte A tile: lanes 0-15 address its rows

    int it = 0;
#pragma unroll 1
    for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
        const int tile = tile_begin + w / kRowBlocks;
        ctx.row_block  = w % kRowBlocks;
        const int job = call.tiles[2 * tile], column = call.tiles[2 * tile + 1];
        ctx.expert  = call.dispatch.jobs[job];
        ctx.first   = call.dispatch.offsets[ctx.expert] + column;
        ctx.columns = min(S::kBlockTokens, call.dispatch.offsets[ctx.expert + 1] - ctx.first);
        // The 16-token MMA tiles holding a valid column (warp-uniform); a skipped tile's outputs are never written.
        const int live = min((ctx.columns + 15) / 16, S::kMmaTokens);

        AccumulatorsA8 acc = {};
#pragma unroll 1
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
            const int stage = it % S::kStages;
            cta_mbarrier_wait(&shared.full[stage], (static_cast<unsigned>(it) / S::kStages) & 1U);
            const std::uint8_t* rows  = shared.a_rows[stage];
            const std::uint8_t* units = shared.b_units[stage] + warp * S::kUnitRunBytes;
            float d[S::kMmaTokens][2] = {};
#pragma unroll
            for (int u = 0; u < S::kUnitsPerKTile; ++u) { // one 16-element block per record unit
                if (u % 2 == 0) {                         // the slice scales of tokens g and g + 8 of each tile
#pragma unroll
                    for (int m = 0; m < S::kMmaTokens; ++m) {
                        if (m >= live) { continue; }
                        const std::uint8_t* scale = rows + S::kCodeRowBytes + 4 * (u / 2);
                        d[m][0] = *reinterpret_cast<const float*>(scale + (m * 16 + g) * S::kRowStride);
                        d[m][1] = *reinterpret_cast<const float*>(scale + (m * 16 + g + 8) * S::kRowStride);
                    }
                }
                unsigned a[S::kMmaTokens][2];
#pragma unroll
                for (int m = 0; m < S::kMmaTokens; ++m) {
                    if (m >= live) { continue; }
                    ldmatrix_x2(a[m][0], a[m][1], smem_addr(rows + (m * 16 + a_row) * S::kRowStride + 16 * u));
                }
                const std::uint8_t* unit = units + u * static_cast<int>(kUnitBytes);
                unsigned b[S::kMmaRows];
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) { b[r] = a16_codes(unit, g + 8 * r, t); }
                // Weight rows 8 r + 2t and 8 r + 2t + 1 (the outputs v & 1) of the unit.
                const unsigned lo = *reinterpret_cast<const std::uint16_t*>(unit + 128 + 2 * t);
                const unsigned hi = *reinterpret_cast<const std::uint16_t*>(unit + 136 + 2 * t);
                const float2 sc[S::kMmaRows][2] = {{shared.scaled[lo & 0xFFU], shared.scaled[lo >> 8]},
                                                   {shared.scaled[hi & 0xFFU], shared.scaled[hi >> 8]}};
#pragma unroll
                for (int m = 0; m < S::kMmaTokens; ++m) {
                    if (m >= live) { continue; }
#pragma unroll
                    for (int r = 0; r < S::kMmaRows; ++r) {
                        int p[4];
                        a8::mma_k16_magic(p, a[m][0], a[m][1], b[r]);
#pragma unroll
                        for (int v = 0; v < 4; ++v) {
                            const float term = a8::scaled_term(p[v], sc[r][v & 1].x, sc[r][v & 1].y); // P_b * S_b, exact
                            acc[m][r][v]     = __fmaf_rn(term, d[m][v >> 1], acc[m][r][v]);
                        }
                    }
                }
            }
            fence_reads_before_async_refill(); // the shared reads before the bulk refills
            __syncwarp();
            if (lane == 0) { cta_mbarrier_arrive(&shared.empty[stage]); }
        }
        // Registers and global memory only, so the producer already fills the next tile's stages.
        epilogue(acc, ctx);
    }
}

// Token column of accumulator (m, v): m * 16 + group + 8 (v >> 1) of the work tile.
__device__ __forceinline__ int a8_tile_column(const TileContext& ctx, int m, int half) { return m * 16 + ctx.group + 8 * half; }

// SwiGLU of the gate/up tile into the entry's BF16 h row (A16Planes<G>::kHRowBytes apart): gate and up are
// nvfp4_a8::row_output of rows 2i and 2i + 1 with the expert's multipliers.
template <class G>
struct GateUpEpilogueA8 {
    const ExpertScales* scales;
    std::uint8_t* h_rows;

    __device__ __forceinline__ void operator()(AccumulatorsA8& acc, const TileContext& ctx) const {
        using S               = ScheduleA8;
        const ExpertScales sc = scales[ctx.expert];
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = a8_tile_column(ctx, m, half);
                if (column >= ctx.columns) { continue; }
                auto* h = reinterpret_cast<std::uint16_t*>(h_rows + static_cast<std::size_t>(ctx.first + column) * A16Planes<G>::kHRowBytes);
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint16_t gate = nvfp4_a8::row_output(acc[m][r][2 * half], sc.alpha_gate);
                    const std::uint16_t up   = nvfp4_a8::row_output(acc[m][r][2 * half + 1], sc.alpha_up);
                    h[(ctx.row_block * S::kBlockRows + 16 * ctx.warp_n + 8 * r + 2 * ctx.tig) / 2] = canon::swiglu_bf16(gate, up);
                }
            }
        }
    }
};

// y = nvfp4_a8::row_output(acc, alpha_down) into the entry's output column (G::kHidden rows).
template <class G>
struct DownEpilogueA8 {
    const ExpertScales* scales;
    const std::int32_t* entries;
    std::uint16_t* outputs;

    __device__ __forceinline__ void operator()(AccumulatorsA8& acc, const TileContext& ctx) const {
        using S           = ScheduleA8;
        const float alpha = scales[ctx.expert].alpha_down;
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = a8_tile_column(ctx, m, half);
                if (column >= ctx.columns) { continue; }
                std::uint16_t* out = outputs + static_cast<std::size_t>(entries[ctx.first + column]) * G::kHidden +
                                     ctx.row_block * S::kBlockRows + 16 * ctx.warp_n;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint32_t y0 = nvfp4_a8::row_output(acc[m][r][2 * half], alpha);
                    const std::uint32_t y1 = nvfp4_a8::row_output(acc[m][r][2 * half + 1], alpha);
                    *reinterpret_cast<std::uint32_t*>(out + 8 * r + 2 * ctx.tig) = y0 | (y1 << 16);
                }
            }
        }
    }
};

template <class Shape, class Epilogue>
void launch_a8(const Call& call, std::int32_t pass, const Epilogue& epilogue, cudaStream_t stream) {
    using S             = ScheduleA8;
    constexpr int bytes = static_cast<int>(sizeof(StorageA8));
    // Static: MSVC 2022 rejects the lambda's use of an automatic constexpr local (C2326).
    static constexpr auto kernel = grouped_kernel_a8<Shape, Epilogue>;
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
        return true;
    }();
    (void)configured;
    // One resident wave of this schedule's CTAs (persistent_ctas() counts the A4 schedule's).
    const int ctas = persistent_ctas() / Schedule::kMinBlocksPerSm * S::kMinBlocksPerSm;
    kernel<<<ctas, S::kThreads, bytes, stream>>>(call, pass, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::offloaded_moe::wide
