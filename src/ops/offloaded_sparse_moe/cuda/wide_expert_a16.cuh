#pragma once

// The wide route of W4A16 experts (wide_expert.h, design §16.2.1): the grouped, persistent,
// warp-specialized pipeline of wide_expert.cuh on integer tensor cores, with the narrow route's
// exact arithmetic.
//   - A operand, activations: the canonical X (|X| < 2^21) of the routed entries' x (by token,
//     encoded once per call) or h (by entry, encoded per pass) as X = 2^14 H1 + 2^11 H0 + 8 L1 + L0
//     (H1 signed, L1 a byte, H0 and L0 three bits), 192 bytes per row and 64-element K tile: 64 H1,
//     64 L1, 64 H0 | L0 << 4. The producer warp moves a row's two K tiles of a stage with one
//     cp.async.bulk into a 400-byte shared row (the 16-byte pad keeps ldmatrix conflict-free).
//   - B operand, weights: the record's 144-byte units bulk-copied as in the A4 route, read as the
//     doubled E2M1 codes c (signed bytes) of the narrow route, and 8 c.
//   - Per 16-element block, two mma.m16n8k32 (K = the block twice) with B = [8 c | c]:
//     A = [H1 | H0] (s8) gives c . (X >> 11), A = [L1 | L0] (u8) gives c . (X & 2047), so
//     P = 2048 (c . (X >> 11)) + c . (X & 2047) = c . X exactly in int32; then S += P * e4m3_scaled
//     in int64. S is the narrow route's and the CPU engine's sum, so the epilogues' canonical
//     outputs are their bits. A k32 MMA costs a k16 one's issue time on RTX 5090 (13 cycles per
//     SMSP; a BF16 m16n8k16 26), so a block costs what the BF16 tensor-core GEMM's would.
// Measured choices (RTX 5090, T = 4096 prefill shape): two k32 MMAs per block instead of three
// k16 ones, 128-element stages (the per-row copies, not the tensor pipe, bounded 64-element ones),
// sixteen 32 x 16 consumer warps. Entry-major x planes read through tensor maps made the GEMMs only
// ~1.4 % faster and cost a per-entry x encoding (0.25 ms per 4K call) and 8x the x workspace.

#include "ops/offloaded_sparse_moe/cuda/wide_expert.cuh"

namespace infernix::ops::offloaded_moe::wide {

struct ScheduleA16 {
    static constexpr int kBlockTokens     = kTileColumns; // 64
    static constexpr int kBlockRows       = 128;
    static constexpr int kRowGroups       = kBlockRows / 16; // 8
    static constexpr int kPlaneTiles      = 2;                        // 64-element limb tiles per stage
    static constexpr int kBlockK          = kPlaneTiles * kA16KTile;  // 128
    static constexpr int kUnitsPerKTile   = kBlockK / 16;             // 8 record units per row group
    static constexpr int kStageRowBytes   = kPlaneTiles * kA16LimbTileBytes; // 384
    static constexpr int kStages          = 2;
    static constexpr int kMinBlocksPerSm  = 1;
    // Sixteen consumer warps of 32 columns x 16 rows: a thread's int64 sums take 32 registers.
    static constexpr int kWarpsTokens     = 2;
    static constexpr int kWarpsRows       = 8;
    static constexpr int kConsumerWarps   = kWarpsTokens * kWarpsRows;
    static constexpr int kConsumerThreads = kConsumerWarps * 32;
    static constexpr int kProducerThreads = 32;
    static constexpr int kThreads         = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpTokens      = kBlockTokens / kWarpsTokens; // 32
    static constexpr int kWarpRows        = kBlockRows / kWarpsRows;     // 16
    static constexpr int kMmaTokens       = kWarpTokens / 16;            // 2
    static constexpr int kMmaRows         = kWarpRows / 8;               // 2: lo/hi halves of a row group
    static constexpr int kGroupsPerWarp   = kMmaRows / 2;                // 1
    static constexpr int kRowStride       = kStageRowBytes + 16;         // padded shared row, 400
    static constexpr int kUnitRunBytes    = kUnitsPerKTile * static_cast<int>(kUnitBytes); // 1,152
    static_assert(kUnitRunBytes % 16 == 0 && kRowStride % 16 == 0 && kMmaRows % 2 == 0);
};

struct StorageA16 {
    alignas(128) std::uint8_t a_rows[ScheduleA16::kStages][ScheduleA16::kBlockTokens * ScheduleA16::kRowStride];
    alignas(128) std::uint8_t b_units[ScheduleA16::kStages][ScheduleA16::kRowGroups * ScheduleA16::kUnitRunBytes];
    std::int32_t scaled[256]; // e4m3_scaled of every scale byte
    alignas(8) std::uint64_t full[ScheduleA16::kStages];
    alignas(8) std::uint64_t empty[ScheduleA16::kStages];
};

static_assert(sizeof(StorageA16) + 1024 <= 102400 / ScheduleA16::kMinBlocksPerSm, "the A16 CTAs must fit an SM");

// Sums S[m][r][v] of a consumer thread: column warp_m * 32 + m * 16 + group + (v >= 2 ? 8 : 0) of the
// tile and matrix row row_block * 128 + a16_tile_row(warp_n, r, tig) + (v & 1), so rows 2i and 2i + 1
// (gate_i and up_i of the gate/up matrix) are in the same thread.
using SumsA16 = long long[ScheduleA16::kMmaTokens][ScheduleA16::kMmaRows][4];

__device__ __forceinline__ int a16_tile_row(int warp_n, int r, int tig) {
    return (ScheduleA16::kGroupsPerWarp * warp_n + (r >> 1)) * 16 + 8 * (r & 1) + 2 * tig;
}

// s += p * w in int64 (one IMAD.WIDE on the accumulator's own registers).
__device__ __forceinline__ void fold(long long& s, int p, int w) {
    asm("mad.wide.s32 %0, %1, %2, %0;" : "+l"(s) : "r"(p), "r"(w));
}

// Doubled E2M1 codes of row r (0..15) of a unit at k = 4q..4q+3, as four signed bytes.
__device__ __forceinline__ unsigned a16_codes(const std::uint8_t* unit, int r, int q) {
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(unit + 32 * q + 4 * (r & 7));
    return canon::e2m1_x2_quad((word >> (r < 8 ? 0 : 4)) & 0x0F0F0F0FU);
}

template <Matrix M, class Epilogue>
__global__ __launch_bounds__(ScheduleA16::kThreads, ScheduleA16::kMinBlocksPerSm) void grouped_kernel_a16(
    const Call call, const std::int32_t pass, const Epilogue epilogue) {
    using S                  = ScheduleA16;
    using Shape              = MatrixShape<M>;
    constexpr int kRowBlocks = Shape::kRowGroups / S::kRowGroups;
    constexpr int kKTiles    = Shape::kBlocks / S::kUnitsPerKTile;
    constexpr int kRowBytes  = M == Matrix::GateUp ? kA16XRowBytes : kA16HRowBytes;
    static_assert(Shape::kRowGroups % S::kRowGroups == 0 && Shape::kBlocks % S::kUnitsPerKTile == 0);

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<StorageA16*>(shared_bytes);
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
    for (int w = static_cast<int>(threadIdx.x); w < 256; w += S::kThreads) { shared.scaled[w] = canon::e4m3_scaled(w); }
    __syncthreads();

    if (threadIdx.x < S::kProducerThreads) {
        const int lane = static_cast<int>(threadIdx.x);
        int it = 0;
#pragma unroll 1
        for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
            const int tile = tile_begin + w / kRowBlocks, row_block = w % kRowBlocks;
            const int job = call.tiles[2 * tile], column = call.tiles[2 * tile + 1];
            const int expert  = call.dispatch.jobs[job];
            const int first   = call.dispatch.offsets[expert] + column;
            const int columns = min(S::kBlockTokens, call.dispatch.offsets[expert + 1] - first);
            const std::uint8_t* matrix = call.job_records[job] + Shape::kOffset +
                                         static_cast<std::size_t>(row_block) * S::kRowGroups * Shape::kBlocks * kUnitBytes;
            // The limb row of each of this lane's tile columns (two per lane): x by token, h by entry.
            const std::uint8_t* rows[S::kBlockTokens / 32];
#pragma unroll
            for (int i = 0; i < S::kBlockTokens / 32; ++i) {
                const int c = lane + 32 * i;
                rows[i]     = nullptr;
                if (c < columns) {
                    const int row = M == Matrix::GateUp ? call.dispatch.entries[first + c] / call.top_k : first + c;
                    rows[i] = (M == Matrix::GateUp ? call.x_limbs : call.h_rows) + static_cast<std::size_t>(row) * kRowBytes;
                }
            }
#pragma unroll 1
            for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
                const int stage = it % S::kStages;
                if (lane == 0) {
                    cta_mbarrier_wait(&shared.empty[stage], 1U ^ ((static_cast<unsigned>(it) / S::kStages) & 1U));
                    cta_mbarrier_arrive_expect_tx(&shared.full[stage],
                                                  static_cast<std::uint32_t>(columns * S::kStageRowBytes +
                                                                             S::kRowGroups * S::kUnitRunBytes));
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < S::kBlockTokens / 32; ++i) {
                    if (rows[i] != nullptr) {
                        bulk_load(shared.a_rows[stage] + (lane + 32 * i) * S::kRowStride,
                                  rows[i] + k_tile * S::kStageRowBytes, S::kStageRowBytes, &shared.full[stage]);
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
    ctx.warp_m = warp / S::kWarpsRows;
    ctx.warp_n = warp % S::kWarpsRows;
    ctx.group  = lane >> 2;
    ctx.tig    = lane & 3;
    // ldmatrix.x2 of a 16 x 16-byte A tile: lanes 0-15 address rows 0-15 (lanes 16-31 repeat them).
    const int a_row = lane & 15;

    int it = 0;
#pragma unroll 1
    for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
        const int tile = tile_begin + w / kRowBlocks;
        ctx.row_block  = w % kRowBlocks;
        const int job = call.tiles[2 * tile], column = call.tiles[2 * tile + 1];
        ctx.expert      = call.dispatch.jobs[job];
        ctx.first       = call.dispatch.offsets[ctx.expert] + column;
        ctx.columns     = min(S::kBlockTokens, call.dispatch.offsets[ctx.expert + 1] - ctx.first);

        SumsA16 sums = {};
#pragma unroll 1
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
            const int stage = it % S::kStages;
            cta_mbarrier_wait(&shared.full[stage], (static_cast<unsigned>(it) / S::kStages) & 1U);
#pragma unroll
            for (int u = 0; u < S::kUnitsPerKTile; ++u) { // one 16-element block per record unit
                // [H1 | H0] and [L1 | L0] of rows group and group + 8 (a0, a1 at K 0-15; a2, a3 at 16-31).
                unsigned high[S::kMmaTokens][4], low[S::kMmaTokens][4];
#pragma unroll
                for (int m = 0; m < S::kMmaTokens; ++m) {
                    const std::uint8_t* row = shared.a_rows[stage] +
                                              (ctx.warp_m * S::kWarpTokens + m * 16 + a_row) * S::kRowStride +
                                              (u / 4) * kA16LimbTileBytes + (u % 4) * 16;
                    unsigned packed[2];
                    ldmatrix_x2(high[m][0], high[m][1], smem_addr(row));
                    ldmatrix_x2(low[m][0], low[m][1], smem_addr(row + kA16KTile));
                    ldmatrix_x2(packed[0], packed[1], smem_addr(row + 2 * kA16KTile));
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        high[m][2 + i] = packed[i] & 0x07070707U;
                        low[m][2 + i]  = (packed[i] >> 4) & 0x07070707U;
                    }
                }
                // c (b1, K 16-31) and 8 c (b0, K 0-15) of each n8 tile; |8 c| <= 96 stays a signed byte.
                unsigned b[S::kMmaRows][2];
                int scale[S::kMmaRows][2];
#pragma unroll
                for (int g = 0; g < S::kGroupsPerWarp; ++g) {
                    const std::uint8_t* unit = shared.b_units[stage] + (S::kGroupsPerWarp * ctx.warp_n + g) * S::kUnitRunBytes +
                                               u * static_cast<int>(kUnitBytes);
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const unsigned c   = a16_codes(unit, ctx.group + 8 * half, ctx.tig);
                        b[2 * g + half][0] = (c << 3) & 0xF8F8F8F8U;
                        b[2 * g + half][1] = c;
                    }
                    const unsigned lo_scales = *reinterpret_cast<const std::uint16_t*>(unit + 128 + 2 * ctx.tig);
                    const unsigned hi_scales = *reinterpret_cast<const std::uint16_t*>(unit + 136 + 2 * ctx.tig);
                    scale[2 * g][0]     = shared.scaled[lo_scales & 0xFFU];
                    scale[2 * g][1]     = shared.scaled[lo_scales >> 8];
                    scale[2 * g + 1][0] = shared.scaled[hi_scales & 0xFFU];
                    scale[2 * g + 1][1] = shared.scaled[hi_scales >> 8];
                }
                // P = 2048 (c . (X >> 11)) + c . (X & 2047): the high product of every tile first, so
                // the tiles' chains overlap.
                int p[S::kMmaTokens][S::kMmaRows][4];
#pragma unroll
                for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
                    for (int r = 0; r < S::kMmaRows; ++r) {
                        mma_s8s8_k32_zero(p[m][r][0], p[m][r][1], p[m][r][2], p[m][r][3], high[m][0], high[m][1],
                                          high[m][2], high[m][3], b[r][0], b[r][1]);
                    }
                }
#pragma unroll
                for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
                    for (int r = 0; r < S::kMmaRows; ++r) {
#pragma unroll
                        for (int v = 0; v < 4; ++v) { p[m][r][v] *= 2048; }
                        mma_u8s8_k32(p[m][r][0], p[m][r][1], p[m][r][2], p[m][r][3], low[m][0], low[m][1], low[m][2],
                                     low[m][3], b[r][0], b[r][1]);
#pragma unroll
                        for (int v = 0; v < 4; ++v) { fold(sums[m][r][v], p[m][r][v], scale[r][v & 1]); }
                    }
                }
            }
            __syncwarp();
            if (lane == 0) { cta_mbarrier_arrive(&shared.empty[stage]); }
        }
        // Registers and global memory only, so the producer already fills the next tile's stages.
        epilogue(sums, ctx);
    }
}

// The gate and up outputs of each column (canonical, with its x exponent), SwiGLU into the entry's
// BF16 h row: rows 2i and 2i + 1 of the matrix are gate_i and up_i, in the same thread (a16_tile_row).
struct GateUpEpilogueA16 {
    const ExpertScales* scales;
    const std::int32_t* entries;
    const std::int32_t* x_emax; // per token
    std::uint8_t* h_rows;
    int top_k;

    __device__ __forceinline__ void operator()(SumsA16& sums, const TileContext& ctx) const {
        using S               = ScheduleA16;
        const ExpertScales sc = scales[ctx.expert];
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * S::kWarpTokens + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                const int entry = ctx.first + column;
                const int emax  = x_emax[entries[entry] / top_k];
                auto* h = reinterpret_cast<std::uint16_t*>(h_rows + static_cast<std::size_t>(entry) * kA16HRowBytes);
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint16_t gate = canon::a16_row_output(sums[m][r][2 * half], emax, sc.alpha_gate);
                    const std::uint16_t up   = canon::a16_row_output(sums[m][r][2 * half + 1], emax, sc.alpha_up);
                    h[(ctx.row_block * S::kBlockRows + a16_tile_row(ctx.warp_n, r, ctx.tig)) / 2] = canon::swiglu_bf16(gate, up);
                }
            }
        }
    }
};

// The canonical down output of each column (with its h exponent) into the entry's output column.
struct DownEpilogueA16 {
    const ExpertScales* scales;
    const std::int32_t* entries;
    const std::int32_t* h_emax;
    std::uint16_t* outputs;

    __device__ __forceinline__ void operator()(SumsA16& sums, const TileContext& ctx) const {
        using S           = ScheduleA16;
        const float alpha = scales[ctx.expert].alpha_down;
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * S::kWarpTokens + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                const int emax = h_emax[ctx.first + column];
                std::uint16_t* out = outputs + static_cast<std::size_t>(entries[ctx.first + column]) * kHidden +
                                     ctx.row_block * S::kBlockRows;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint32_t y0 = canon::a16_row_output(sums[m][r][2 * half], emax, alpha);
                    const std::uint32_t y1 = canon::a16_row_output(sums[m][r][2 * half + 1], emax, alpha);
                    *reinterpret_cast<std::uint32_t*>(out + a16_tile_row(ctx.warp_n, r, ctx.tig)) = y0 | (y1 << 16);
                }
            }
        }
    }
};

template <Matrix M, class Epilogue>
void launch_a16(const Call& call, std::int32_t pass, const Epilogue& epilogue, cudaStream_t stream) {
    using S               = ScheduleA16;
    constexpr int bytes   = static_cast<int>(sizeof(StorageA16));
    constexpr auto kernel = grouped_kernel_a16<M, Epilogue>;
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(grouped_kernel_a16<M, Epilogue>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(sizeof(StorageA16))));
        CUDA_CHECK(cudaFuncSetAttribute(grouped_kernel_a16<M, Epilogue>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
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
