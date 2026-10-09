#pragma once

// The wide route of W4A16 experts (wide_expert.h, design §16.2.1): the grouped, persistent,
// warp-specialized pipeline of wide_expert.cuh with BF16 tensor cores (mma.m16n8k16 bf16) in place
// of the block-scaled FP4 ones.
//   - A operand, activations: BF16 rows of the routed entries, x for gate/up (gathered by token,
//     so no x plane is built) and the call's BF16 h plane for down. The producer warp moves each
//     entry's 128-byte K slice with one cp.async.bulk into a 144-byte shared row (the 16-byte pad
//     keeps ldmatrix conflict-free).
//   - B operand, weights: the record's 144-byte units bulk-copied as in the A4 route; each weight
//     c2 / 2 x E4M3 block scale is formed in registers. It has at most six significant bits, so
//     the BF16 fragment is the stored value exactly.
//   - Epilogues: y = bf16_rn(acc * m) with the stored multiplier m (gate, up: then SwiGLU into the
//     entry's h row; down: into its output column). Only the FP32 accumulation order differs from the
//     narrow route's exact sums; the route is qualified against FP64.

#include "ops/offloaded_sparse_moe/cuda/wide_expert.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::offloaded_moe::wide {

struct ScheduleA16 {
    static constexpr int kBlockTokens     = kTileColumns; // 64
    static constexpr int kBlockRows       = 128;
    static constexpr int kRowGroups       = kBlockRows / 16; // 8
    static constexpr int kBlockK          = 64;
    static constexpr int kUnitsPerKTile   = kBlockK / 16; // 4 record units per row group
    static constexpr int kStages          = 3;
    static constexpr int kMinBlocksPerSm  = 2;
    static constexpr int kWarpsTokens     = 2;
    static constexpr int kWarpsRows       = 4;
    static constexpr int kConsumerWarps   = kWarpsTokens * kWarpsRows;
    static constexpr int kConsumerThreads = kConsumerWarps * 32;
    static constexpr int kProducerThreads = 32;
    static constexpr int kThreads         = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpTokens      = kBlockTokens / kWarpsTokens; // 32
    static constexpr int kWarpRows        = kBlockRows / kWarpsRows;     // 32
    static constexpr int kMmaTokens       = kWarpTokens / 16;            // 2
    static constexpr int kMmaRows         = kWarpRows / 8;               // 4: lo/hi halves of 2 row groups
    static constexpr int kSliceBytes      = kBlockK * 2;                 // one entry's BF16 K slice
    static constexpr int kRowStride       = kSliceBytes + 16;            // padded shared row
    static constexpr int kUnitRunBytes    = kUnitsPerKTile * static_cast<int>(kUnitBytes); // 576
    static_assert(kMmaTokens == Schedule::kMmaTokens && kMmaRows == Schedule::kMmaRows &&
                      kBlockRows == Schedule::kBlockRows && kWarpsRows == Schedule::kWarpsRows,
                  "the A16 tiles share the A4 accumulator layout (tile_row, the epilogue mapping)");
    static_assert(kUnitRunBytes % 16 == 0 && kSliceBytes % 16 == 0);
};

struct StorageA16 {
    alignas(128) std::uint8_t a_rows[ScheduleA16::kStages][ScheduleA16::kBlockTokens * ScheduleA16::kRowStride];
    alignas(128) std::uint8_t b_units[ScheduleA16::kStages][ScheduleA16::kRowGroups * ScheduleA16::kUnitRunBytes];
    alignas(8) std::uint64_t full[ScheduleA16::kStages];
    alignas(8) std::uint64_t empty[ScheduleA16::kStages];
};

static_assert(sizeof(StorageA16) + 1024 <= 102400 / ScheduleA16::kMinBlocksPerSm, "two A16 CTAs must fit an SM");

// Two BF16 weights of one row from two consecutive code nibbles (the low or high nibbles of a 16-bit
// load) and the row's half scale (E4M3 value / 2): c2 * half_scale is exact in FP32 and in BF16.
__device__ __forceinline__ unsigned bf16_pair(unsigned codes, int shift, float half_scale) {
    const float v0 = static_cast<float>(canon::e2m1_x2((codes >> shift) & 15U)) * half_scale;
    const float v1 = static_cast<float>(canon::e2m1_x2((codes >> (8 + shift)) & 15U)) * half_scale;
    const __nv_bfloat162 pair = __floats2bfloat162_rn(v0, v1);
    return *reinterpret_cast<const unsigned*>(&pair);
}

// The B fragments of one row group's unit (16 rows x 16 K) for this lane: rows group (lo) and
// group + 8 (hi), K slots 2 tig, 2 tig + 1 (register 0) and 2 tig + 8, 2 tig + 9 (register 1).
// Code (r, k) is nibble (r >= 8) of byte 32 (k / 4) + 4 (r % 8) + k % 4.
__device__ __forceinline__ void a16_fragments(const std::uint8_t* unit, int group, int tig, unsigned (&lo)[2],
                                              unsigned (&hi)[2]) {
    const int offset = 32 * (tig >> 1) + 4 * group + 2 * (tig & 1);
    const unsigned first  = *reinterpret_cast<const std::uint16_t*>(unit + offset);
    const unsigned second = *reinterpret_cast<const std::uint16_t*>(unit + 64 + offset);
    const float s_lo = 0.5F * canon::e4m3_value(unit[128 + group]);
    const float s_hi = 0.5F * canon::e4m3_value(unit[128 + 8 + group]);
    lo[0] = bf16_pair(first, 0, s_lo);
    lo[1] = bf16_pair(second, 0, s_lo);
    hi[0] = bf16_pair(first, 4, s_hi);
    hi[1] = bf16_pair(second, 4, s_hi);
}

template <Matrix M, class Epilogue>
__global__ __launch_bounds__(ScheduleA16::kThreads, ScheduleA16::kMinBlocksPerSm) void grouped_kernel_a16(
    const Call call, const std::int32_t pass, const Epilogue epilogue) {
    using S                  = ScheduleA16;
    using Shape              = MatrixShape<M>;
    constexpr int kRowBlocks = Shape::kRowGroups / S::kRowGroups;
    constexpr int kKTiles    = Shape::kBlocks / S::kUnitsPerKTile;
    constexpr int kWidth     = M == Matrix::GateUp ? kHidden : kIntermediate; // elements per activation row
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
            // The activation row of each of this lane's tile columns (two per lane).
            const std::uint16_t* rows[S::kBlockTokens / 32];
#pragma unroll
            for (int i = 0; i < S::kBlockTokens / 32; ++i) {
                const int c = lane + 32 * i;
                rows[i]     = nullptr;
                if (c < columns) {
                    rows[i] = M == Matrix::GateUp
                                  ? call.x + static_cast<std::size_t>(call.dispatch.entries[first + c] / call.top_k) * kWidth
                                  : call.h16 + static_cast<std::size_t>(first + c) * kWidth;
                }
            }
#pragma unroll 1
            for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
                const int stage = it % S::kStages;
                if (lane == 0) {
                    cta_mbarrier_wait(&shared.empty[stage], 1U ^ ((static_cast<unsigned>(it) / S::kStages) & 1U));
                    cta_mbarrier_arrive_expect_tx(
                        &shared.full[stage], static_cast<std::uint32_t>(columns * S::kSliceBytes + S::kRowGroups * S::kUnitRunBytes));
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < S::kBlockTokens / 32; ++i) {
                    if (rows[i] != nullptr) {
                        bulk_load(shared.a_rows[stage] + (lane + 32 * i) * S::kRowStride, rows[i] + k_tile * S::kBlockK,
                                  S::kSliceBytes, &shared.full[stage]);
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
    // ldmatrix.x4 of a 16 x 16 A tile: lanes 0-15 address rows 0-15 at K 0, lanes 16-31 at K 8.
    const int a_row  = lane & 15;
    const int a_byte = (lane >> 4) * 16;

    int it = 0;
#pragma unroll 1
    for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
        const int tile = tile_begin + w / kRowBlocks;
        ctx.row_block  = w % kRowBlocks;
        const int job = call.tiles[2 * tile], column = call.tiles[2 * tile + 1];
        ctx.expert      = call.dispatch.jobs[job];
        ctx.first       = call.dispatch.offsets[ctx.expert] + column;
        ctx.columns     = min(S::kBlockTokens, call.dispatch.offsets[ctx.expert + 1] - ctx.first);

        Accumulators acc = {};
#pragma unroll 1
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
            const int stage = it % S::kStages;
            cta_mbarrier_wait(&shared.full[stage], (static_cast<unsigned>(it) / S::kStages) & 1U);
#pragma unroll
            for (int u = 0; u < S::kUnitsPerKTile; ++u) { // one 16-wide K step per record unit
                unsigned a[S::kMmaTokens][4];
#pragma unroll
                for (int m = 0; m < S::kMmaTokens; ++m) {
                    const int row = ctx.warp_m * S::kWarpTokens + m * 16 + a_row;
                    ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                smem_addr(shared.a_rows[stage] + row * S::kRowStride + u * 32 + a_byte));
                }
                unsigned b[S::kMmaRows][2];
#pragma unroll
                for (int g = 0; g < S::kMmaRows / 2; ++g) {
                    const std::uint8_t* unit = shared.b_units[stage] + (2 * ctx.warp_n + g) * S::kUnitRunBytes +
                                               u * static_cast<int>(kUnitBytes);
                    a16_fragments(unit, ctx.group, ctx.tig, b[2 * g], b[2 * g + 1]);
                }
#pragma unroll
                for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
                    for (int r = 0; r < S::kMmaRows; ++r) {
                        mma_bf16(acc[m][r][0], acc[m][r][1], acc[m][r][2], acc[m][r][3], a[m][0], a[m][1], a[m][2],
                                 a[m][3], b[r][0], b[r][1]);
                    }
                }
            }
            __syncwarp();
            if (lane == 0) { cta_mbarrier_arrive(&shared.empty[stage]); }
        }
        epilogue(acc, ctx);
    }
}

// SwiGLU of the gate/up tile into the entry's BF16 h row: rows 2i and 2i + 1 of the matrix are
// gate_i and up_i, in the same thread (tile_row).
struct GateUpEpilogueA16 {
    const ExpertScales* scales;
    std::uint16_t* h16;

    __device__ __forceinline__ void operator()(Accumulators& acc, const TileContext& ctx) const {
        using S               = ScheduleA16;
        const ExpertScales sc = scales[ctx.expert];
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * S::kWarpTokens + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                std::uint16_t* h = h16 + static_cast<std::size_t>(ctx.first + column) * kIntermediate;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint16_t gate = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half], sc.alpha_gate));
                    const std::uint16_t up   = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half + 1], sc.alpha_up));
                    h[(ctx.row_block * S::kBlockRows + tile_row(ctx.warp_n, r, ctx.tig)) / 2] = canon::swiglu_bf16(gate, up);
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
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
        return true;
    }();
    (void)configured;
    kernel<<<persistent_ctas(), S::kThreads, bytes, stream>>>(call, pass, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::offloaded_moe::wide
