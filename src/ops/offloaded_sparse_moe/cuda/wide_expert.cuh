#pragma once

// The wide-route grouped GEMM of offloaded_sparse_moe (wide_expert.h). It is the warp-specialized
// TMA pipeline of the dense NVFP4 W4A4 route (ops/linear/nvfp4/nvfp4_a4_tma.cuh) made grouped and
// persistent:
//   - A operand, activations: the same TMA path. A token tile's A4 codes arrive through a
//     64-byte-swizzled tensor map and are read with the same ldmatrix fragments, its scales through
//     a second map; both planes hold one row per routed entry, grouped by expert.
//   - B operand, weights: each record's `nvfp4_expert_rg16_v1` units are bulk-copied unchanged
//     (one cp.async.bulk per row group and K tile, from wherever the job's record lives) and turned
//     into mxf4nvf4 fragments in registers. No tensor-core layout leaks into the record, and no
//     weight is repacked in memory.
//   - Work: a device-built list of (job, 64-column tile) pairs times the matrix's 128-row blocks,
//     walked by a persistent grid, so the pipeline runs on across tiles and experts.
//   - Epilogues receive FP32 accumulators and apply the canonical boundaries of §16.2.

#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"

#include "core/device.h"
#include "core/tma_descriptor_staging.cuh"
#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::offloaded_moe::wide {

// The activation planes of one call: A4(x) for the gate/up GEMM and A4(h) for the down GEMM.
struct alignas(128) Descriptors {
    CUtensorMap x_codes;
    CUtensorMap x_scales;
    CUtensorMap h_codes;
    CUtensorMap h_scales;
};

enum class Matrix { GateUp, Down };

template <Matrix M>
struct MatrixShape;

template <>
struct MatrixShape<Matrix::GateUp> {
    static constexpr int kRowGroups = kGateUpRowGroups; // 80
    static constexpr int kBlocks    = kGateUpBlocks;    // 160 K blocks
    static constexpr std::size_t kOffset = 0;
};

template <>
struct MatrixShape<Matrix::Down> {
    static constexpr int kRowGroups = kDownRowGroups; // 160
    static constexpr int kBlocks    = kDownBlocks;    // 40 K blocks
    static constexpr std::size_t kOffset = kGateUpBytes;
};

// The tile shape follows Nvfp4A4TmaMmaSchedule (128 rows, 128-wide K tiles of two 64-wide MMA
// steps, one producer warp beside the consumers) with a 64-column token tile: an expert sees
// about 20 columns per 1K-token chunk and 80 per 4K chunk, where the dense schedule's 128- or
// 256-token tiles (fixed by its tiled scale layout) would mostly compute padding. Eight consumer
// warps tile it 2 (columns) x 4 (rows); a warp's 32 rows are one 16-intermediate block of h.
struct Schedule {
    static constexpr int kBlockTokens      = kTileColumns; // 64
    static constexpr int kBlockRows        = 128;
    static constexpr int kRowGroups        = kBlockRows / 16; // 8
    static constexpr int kBlockK           = 128;
    static constexpr int kUnitsPerKTile    = kBlockK / 16; // 8 record units per row group
    static constexpr int kStages           = 3;
    static constexpr int kMinBlocksPerSm   = 2;
    static constexpr int kWarpsTokens      = 2;
    static constexpr int kWarpsRows        = 4;
    static constexpr int kConsumerWarps    = kWarpsTokens * kWarpsRows;
    static constexpr int kConsumerThreads  = kConsumerWarps * 32;
    static constexpr int kProducerThreads  = 32;
    static constexpr int kThreads          = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpTokens       = kBlockTokens / kWarpsTokens; // 32
    static constexpr int kWarpRows         = kBlockRows / kWarpsRows;     // 32
    static constexpr int kMmaTokens        = kWarpTokens / 16;            // 2
    static constexpr int kMmaRows          = kWarpRows / 8;               // 4: lo/hi halves of 2 row groups
    static constexpr int kK64PerStage      = kBlockK / 64;                // 2
    static constexpr int kCodeRowBytes     = kBlockK / 2;                 // 64
    static constexpr int kScaleWordsPerRow = 4;                           // a 16-byte scale box: 2 K tiles
    static constexpr int kUnitRunBytes     = kUnitsPerKTile * static_cast<int>(kUnitBytes); // 1,152
    static constexpr std::uint32_t kStageBytes =
        kBlockTokens * kCodeRowBytes + kBlockTokens * kScaleWordsPerRow * 4 + kRowGroups * kUnitRunBytes;
    static_assert(kWarpRows == 32, "a warp's rows are one A4 block of h");
    static_assert(kUnitRunBytes % 16 == 0, "bulk copies move multiples of 16 bytes");
};

template <class S>
struct Storage {
    alignas(1024) std::uint8_t a_codes[S::kStages][S::kBlockTokens * S::kCodeRowBytes];
    alignas(128) std::uint32_t a_scales[S::kStages][S::kBlockTokens * S::kScaleWordsPerRow];
    alignas(128) std::uint8_t b_units[S::kStages][S::kRowGroups * S::kUnitRunBytes];
    alignas(8) std::uint64_t full[S::kStages];
    alignas(8) std::uint64_t empty[S::kStages];
};

// One work tile as the consumer threads see it.
struct TileContext {
    int expert;    // the job's expert
    int first;     // packed row (plane row, dispatch.entries index) of the tile's first column
    int columns;   // valid columns in the tile, 1..64
    int row_block; // 128-row block of the matrix
    int warp_m;    // column half of the tile (32 columns)
    int warp_n;    // 32-row quarter of the block
    int group;     // lane / 4
    int tig;       // lane % 4
};

// Accumulator acc[m][r][v] of a consumer thread holds column
//   warp_m * 32 + m * 16 + group + (v >= 2 ? 8 : 0)
// of the tile and matrix row
//   row_block * 128 + tile_row(warp_n, r, tig) + (v & 1).
// Rows 2i and 2i + 1 are therefore gate_i and up_i of the gate/up matrix in the same thread.
__device__ __forceinline__ int tile_row(int warp_n, int r, int tig) {
    return (2 * warp_n + (r >> 1)) * 16 + 8 * (r & 1) + 2 * tig;
}

using Accumulators = float[Schedule::kMmaTokens][Schedule::kMmaRows][4];

__device__ __forceinline__ void bulk_load(void* destination, const void* source, std::uint32_t bytes,
                                          std::uint64_t* barrier) {
    // The CTA-local destination form: with a shared::cluster destination ptxas guards every copy with
    // a call to a driver routine for remote CTAs, and the driver then reserves ~14 KB of stack per
    // resident thread (3.3 GiB on 170 SMs) at the kernel's first launch.
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :
                 : "r"(smem_addr(destination)), "l"(source), "r"(bytes), "r"(smem_addr(barrier))
                 : "memory");
}

__device__ __forceinline__ std::uint32_t lds32(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

// The B fragments of one row group for one 64-wide K step from its four staged record units
// (unit u at units + 144 u covers K block u of the step). MMA K slots 0-15, 16-31, 32-47, 48-63
// take blocks 0, 2, 1, 3 (the plane_block order of the activations), so the four lanes of a
// group read four distinct 16-byte bank ranges. Within a block, a thread's eight slots pair
// record nibbles k and k + 4 of one half block, which is exactly a masked OR of two code words:
// rows 0-7 of the group are the low nibbles, rows 8-15 the high ones.
__device__ __forceinline__ void rg16_fragments(const std::uint8_t* units, int group, int tig, unsigned (&lo)[2],
                                               unsigned (&hi)[2], unsigned& scale_lo, unsigned& scale_hi) {
    const int quad          = 64 * (tig & 1) + 4 * group;
    const std::uint8_t* b0  = units + 2 * (tig >> 1) * kUnitBytes + quad;
    const std::uint8_t* b1  = units + (1 + 2 * (tig >> 1)) * kUnitBytes + quad;
    const std::uint32_t w0 = lds32(b0), w1 = lds32(b0 + 32), v0 = lds32(b1), v1 = lds32(b1 + 32);
    lo[0] = (w0 & 0x0F0F0F0FU) | ((w1 & 0x0F0F0F0FU) << 4);
    hi[0] = ((w0 >> 4) & 0x0F0F0F0FU) | (w1 & 0xF0F0F0F0U);
    lo[1] = (v0 & 0x0F0F0F0FU) | ((v1 & 0x0F0F0F0FU) << 4);
    hi[1] = ((v0 >> 4) & 0x0F0F0F0FU) | (v1 & 0xF0F0F0F0U);
    // Row r's scale of unit u is byte 128 + r; this lane supplies column (row) `group` and
    // `group` + 8, in slot-block order 0, 2, 1, 3.
    const int word          = 128 + 4 * (group >> 2);
    const unsigned select   = static_cast<unsigned>(group & 3) | (static_cast<unsigned>(4 + (group & 3)) << 4);
    const std::uint32_t s0 = lds32(units + word), s1 = lds32(units + kUnitBytes + word);
    const std::uint32_t s2 = lds32(units + 2 * kUnitBytes + word), s3 = lds32(units + 3 * kUnitBytes + word);
    const std::uint32_t t0 = lds32(units + word + 8), t1 = lds32(units + kUnitBytes + word + 8);
    const std::uint32_t t2 = lds32(units + 2 * kUnitBytes + word + 8), t3 = lds32(units + 3 * kUnitBytes + word + 8);
    scale_lo = __byte_perm(__byte_perm(s0, s2, select), __byte_perm(s1, s3, select), 0x5410);
    scale_hi = __byte_perm(__byte_perm(t0, t2, select), __byte_perm(t1, t3, select), 0x5410);
}

template <Matrix M, class Epilogue>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void grouped_kernel(
#ifdef _WIN32
    // As in nvfp4_a4_tma_kernel: MSVC cannot pass the over-aligned descriptors by value.
    const Descriptors* descriptors_pointer,
#else
    const __grid_constant__ Descriptors descriptors,
#endif
    const Call call, const std::int32_t pass, const Epilogue epilogue) {
#ifdef _WIN32
    const Descriptors& descriptors = *descriptors_pointer;
#endif
    using S                    = Schedule;
    using Shape                = MatrixShape<M>;
    constexpr int kRowBlocks   = Shape::kRowGroups / S::kRowGroups;
    constexpr int kKTiles      = Shape::kBlocks / S::kUnitsPerKTile;
    static_assert(Shape::kRowGroups % S::kRowGroups == 0 && Shape::kBlocks % S::kUnitsPerKTile == 0);
    const CUtensorMap* codes_map  = M == Matrix::GateUp ? &descriptors.x_codes : &descriptors.h_codes;
    const CUtensorMap* scales_map = M == Matrix::GateUp ? &descriptors.x_scales : &descriptors.h_scales;

    extern __shared__ __align__(1024) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<Storage<S>*>(shared_bytes);
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
        if (threadIdx.x == 0) {
#ifdef _WIN32
            acquire_staged_tensor_map(codes_map);
            acquire_staged_tensor_map(scales_map);
#endif
            int it = 0;
#pragma unroll 1
            for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
                const int tile = tile_begin + w / kRowBlocks, row_block = w % kRowBlocks;
                const int job = call.tiles[2 * tile], column = call.tiles[2 * tile + 1];
                const int row = call.dispatch.offsets[call.dispatch.jobs[job]] + column;
                const std::uint8_t* matrix = call.job_records[job] + Shape::kOffset +
                                             static_cast<std::size_t>(row_block) * S::kRowGroups * Shape::kBlocks *
                                                 kUnitBytes;
#pragma unroll 1
                for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
                    const int stage = it % S::kStages;
                    cta_mbarrier_wait(&shared.empty[stage], 1U ^ ((static_cast<unsigned>(it) / S::kStages) & 1U));
                    cta_mbarrier_arrive_expect_tx(&shared.full[stage], S::kStageBytes);
                    detail::nvfp4_tma_load_2d(shared.a_codes[stage], codes_map, k_tile * S::kCodeRowBytes, row,
                                              &shared.full[stage]);
                    // A 16-byte scale box covers two K tiles; each tile loads its own copy.
                    detail::nvfp4_tma_load_2d(shared.a_scales[stage], scales_map, (k_tile / 2) * 16, row,
                                              &shared.full[stage]);
#pragma unroll
                    for (int g = 0; g < S::kRowGroups; ++g) {
                        bulk_load(shared.b_units[stage] + g * S::kUnitRunBytes,
                                  matrix + (static_cast<std::size_t>(g) * Shape::kBlocks +
                                            static_cast<std::size_t>(k_tile) * S::kUnitsPerKTile) *
                                               kUnitBytes,
                                  S::kUnitRunBytes, &shared.full[stage]);
                    }
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
    // The A fragment and scale lanes of nvfp4_a4_tma_kernel.
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);

    int it = 0;
#pragma unroll 1
    for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
        const int tile = tile_begin + w / kRowBlocks;
        ctx.row_block  = w % kRowBlocks;
        const int job = call.tiles[2 * tile], column = call.tiles[2 * tile + 1];
        ctx.expert    = call.dispatch.jobs[job];
        const int begin = call.dispatch.offsets[ctx.expert];
        ctx.first       = begin + column;
        ctx.columns     = min(S::kBlockTokens, call.dispatch.offsets[ctx.expert + 1] - ctx.first);

        Accumulators acc = {};
#pragma unroll 1
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++it) {
            const int stage = it % S::kStages;
            cta_mbarrier_wait(&shared.full[stage], (static_cast<unsigned>(it) / S::kStages) & 1U);
#pragma unroll
            for (int local_k64 = 0; local_k64 < S::kK64PerStage; ++local_k64) {
                unsigned a_fragments[S::kMmaTokens][4];
                unsigned a_scales[S::kMmaTokens];
#pragma unroll
                for (int mma_m = 0; mma_m < S::kMmaTokens; ++mma_m) {
                    const int row          = ctx.warp_m * S::kWarpTokens + mma_m * 16 + a_row_offset;
                    const int logical_byte = local_k64 * 32 + a_column_byte;
                    const int physical_byte = ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
                    ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                                a_fragments[mma_m][3],
                                smem_addr(shared.a_codes[stage] + row * S::kCodeRowBytes + physical_byte));
                    const int scale_row = ctx.warp_m * S::kWarpTokens + mma_m * 16 + sfa_row;
                    a_scales[mma_m] = shared.a_scales[stage][scale_row * S::kScaleWordsPerRow + (k_tile & 1) * 2 +
                                                             local_k64];
                }
                unsigned b_fragments[S::kMmaRows][2];
                unsigned b_scales[S::kMmaRows];
#pragma unroll
                for (int g = 0; g < S::kMmaRows / 2; ++g) {
                    const std::uint8_t* units = shared.b_units[stage] + (2 * ctx.warp_n + g) * S::kUnitRunBytes +
                                                local_k64 * 4 * static_cast<int>(kUnitBytes);
                    rg16_fragments(units, ctx.group, ctx.tig, b_fragments[2 * g], b_fragments[2 * g + 1],
                                   b_scales[2 * g], b_scales[2 * g + 1]);
                }
#pragma unroll
                for (int mma_m = 0; mma_m < S::kMmaTokens; ++mma_m) {
#pragma unroll
                    for (int r = 0; r < S::kMmaRows; ++r) {
                        mma_nvfp4_e4m3(acc[mma_m][r][0], acc[mma_m][r][1], acc[mma_m][r][2], acc[mma_m][r][3],
                                       a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                                       a_fragments[mma_m][3], b_fragments[r][0], b_fragments[r][1],
                                       a_scales[mma_m], b_scales[r]);
                    }
                }
            }
            __syncwarp();
            if (lane == 0) { cta_mbarrier_arrive(&shared.empty[stage]); }
        }
        // Registers and global memory only, so the producer already fills the next tile's stages.
        epilogue(acc, ctx);
    }
}

// ------------------------------------------------------------------------------------- epilogues

// SwiGLU of the gate/up tile and A4 of h with the down input scale, into the entry's h row. The
// four lanes of a group hold the 16 intermediates of one h block for a column (lane tig: block
// elements tig, 4 + tig, 8 + tig, 12 + tig), so the block maximum and the code bytes are formed
// with two shuffles each, and lane tig owns code bytes tig and 4 + tig of the plane layout.
struct GateUpEpilogue {
    const ExpertScales* scales;
    std::uint8_t* h_plane;

    __device__ __forceinline__ void operator()(Accumulators& acc, const TileContext& ctx) const {
        using S               = Schedule;
        const ExpertScales sc = scales[ctx.expert];
        const int block       = ctx.row_block * 4 + ctx.warp_n;
        const int position    = plane_block(block);
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * S::kWarpTokens + m * 16 + ctx.group + 8 * half;
                std::uint16_t h[S::kMmaRows];
                float amax = 0.0F;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint16_t gate = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half], sc.alpha_gate));
                    const std::uint16_t up = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half + 1], sc.alpha_up));
                    h[r]           = canon::swiglu_bf16(gate, up);
                    const float a  = canon::f32_from_bits(canon::f32_bits(canon::bf16_to_f32(h[r])) & 0x7FFFFFFFU);
                    amax           = a > amax ? a : amax;
                }
#pragma unroll
                for (int offset = 1; offset <= 2; offset <<= 1) {
                    const float other = __shfl_xor_sync(0xFFFFFFFFU, amax, offset);
                    amax              = other > amax ? other : amax;
                }
                const std::uint8_t scale = canon::a4_scale_word(amax, sc.input_down);
                unsigned code[S::kMmaRows] = {};
                if (canon::e4m3_scaled(scale) != 0) {
#pragma unroll
                    for (int r = 0; r < S::kMmaRows; ++r) {
                        code[r] = canon::a4_code(canon::bf16_to_f32(h[r]), scale, sc.input_down);
                    }
                }
                unsigned low  = (code[0] | (code[1] << 4)) << (8 * ctx.tig);
                unsigned high = (code[2] | (code[3] << 4)) << (8 * ctx.tig);
#pragma unroll
                for (int offset = 1; offset <= 2; offset <<= 1) {
                    low |= __shfl_xor_sync(0xFFFFFFFFU, low, offset);
                    high |= __shfl_xor_sync(0xFFFFFFFFU, high, offset);
                }
                if (ctx.tig == 0 && column < ctx.columns) {
                    std::uint8_t* row = h_plane + static_cast<std::size_t>(ctx.first + column) * kHRowStride;
                    *reinterpret_cast<uint2*>(row + 8 * position) = make_uint2(low, high);
                    row[kHCodeBytes + position] = scale;
                }
            }
        }
    }
};

// y = bf16_rn(acc * alpha_down) into the entry's output column.
struct DownEpilogue {
    const ExpertScales* scales;
    const std::int32_t* entries;
    std::uint16_t* outputs;

    __device__ __forceinline__ void operator()(Accumulators& acc, const TileContext& ctx) const {
        using S           = Schedule;
        const float alpha = scales[ctx.expert].alpha_down;
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * S::kWarpTokens + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                std::uint16_t* out = outputs + static_cast<std::size_t>(entries[ctx.first + column]) * kHidden +
                                     ctx.row_block * S::kBlockRows;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint32_t y0 = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half], alpha));
                    const std::uint32_t y1 = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half + 1], alpha));
                    *reinterpret_cast<std::uint32_t*>(out + tile_row(ctx.warp_n, r, ctx.tig)) = y0 | (y1 << 16);
                }
            }
        }
    }
};

// ------------------------------------------------------------------------------------ launching

[[nodiscard]] int persistent_ctas();

template <Matrix M, class Epilogue>
void launch(const Call& call, std::int32_t pass, const Epilogue& epilogue, cudaStream_t stream) {
    using S               = Schedule;
    constexpr int bytes   = static_cast<int>(sizeof(Storage<S>));
    constexpr auto kernel = grouped_kernel<M, Epilogue>;
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
        return true;
    }();
    (void)configured;
    const int dynamic = detail::nvfp4_prepare_shared<bytes, kernel, true>();
#ifdef _WIN32
    kernel<<<persistent_ctas(), S::kThreads, dynamic, stream>>>(static_cast<const Descriptors*>(call.descriptors),
                                                                 call, pass, epilogue);
#else
    kernel<<<persistent_ctas(), S::kThreads, dynamic, stream>>>(*static_cast<const Descriptors*>(call.descriptors),
                                                                 call, pass, epilogue);
#endif
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::offloaded_moe::wide
