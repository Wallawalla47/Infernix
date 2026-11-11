#pragma once

// The tensor-core route of bf16_p12 calls wider than the skinny GEMV serves: the BF16 TMA MMA kernel
// (bf16_a16_tma_mma.cuh) with its weight tiles loaded packed and decoded in shared memory. Per K tile, one
// producer thread loads the packed code and nibble tiles (as stored, 12 bits a weight) and the activation tile by
// TMA; decode warps expand the packed tile into the BF16 weight tile, in the 128-byte-swizzled layout the TMA
// gives the BF16 kernel; the consumer warps are that kernel's, unchanged. A decoded weight has its BF16 bits, so a
// call has the bits of the BF16 TMA route on the same tile shape (BlockRows, BlockTokens, BlockK and the warp
// tiles; the stage count and decode warps change no arithmetic).
//
// Pipeline: two rings. The packed ring (PackedStages slots) is filled by one producer thread and released by the
// decode warps as soon as they have decoded a slot, so packed tiles load well ahead of the MMAs. The stage ring
// (Stages slots, each a decoded weight tile and an activation tile) is released by the consumers: a second producer
// thread then loads the next activation tile into the slot and the decode warps write its weights; both count on
// the slot's `full` barrier, which the consumers wait for.

#include "core/tma_descriptor_staging.cuh"
#include "ops/common/math.h"
#include "ops/common/mbarrier.cuh"
#include "ops/common/token_slices.h"
#include "ops/linear/bf16/bf16_a16_tma_mma.cuh"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"
#include "ops/linear/common/output.cuh"

#include <cuda.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace infernix::ops::detail {

struct alignas(128) P12TmaDescriptors {
    CUtensorMap codes;
    CUtensorMap nibbles;
    CUtensorMap activation;
};

#ifdef _WIN32
inline TmaDescriptorStaging<P12TmaDescriptors>& p12_tma_descriptor_staging() {
    static TmaDescriptorStaging<P12TmaDescriptors> staging;
    return staging;
}
#endif

// A Bf16A16TmaMmaSchedule's tile with `Stages` decoded and activation stages, `PackedStages` packed stages and
// `DecodeWarps` decode warps.
template <class Tile, int DecodeWarps, int Stages, int PackedStages, bool Spin = false,
          int MinBlocks = Tile::kMinBlocksPerSm>
struct P12TmaSchedule : Tile {
    static constexpr bool kSpin           = Spin;
    static constexpr int kStages          = Stages;
    static constexpr int kPackedStages    = PackedStages;
    static constexpr int kMinBlocksPerSm  = MinBlocks;
    static constexpr int kDecodeWarps     = DecodeWarps;
    static constexpr int kDecodeThreads   = DecodeWarps * 32;
    static constexpr int kProducerThreads = 64; // warp 0 loads packed tiles, warp 1 activation tiles
    static constexpr int kThreads         = kProducerThreads + kDecodeThreads + Tile::kConsumerWarps * 32;
    static constexpr int kDecodedBytes    = Stages * Tile::kBlockRows * Tile::kBlockK * 2;
    static constexpr int kActivationBytes = Stages * Tile::kBlockTokens * Tile::kBlockK * 2;
    static constexpr int kCodeBytes       = Tile::kBlockRows * Tile::kBlockK;     // a packed stage's
    static constexpr int kNibbleBytes     = Tile::kBlockRows * Tile::kBlockK / 2; // a packed stage's
    static constexpr int kPackedOffset    = kDecodedBytes + kActivationBytes;
    static constexpr int kBarrierOffset   = kPackedOffset + PackedStages * (kCodeBytes + kNibbleBytes);
    static constexpr int kSharedBytes     = kBarrierOffset + 2 * (Stages + PackedStages) * 8;
    // Chunks a decode thread expands per K tile.
    static constexpr int kChunkPasses = Tile::kBlockRows * (Tile::kBlockK / 8) / kDecodeThreads;
    static_assert(!bf16_predicated_rows<Tile> && !bf16_predicated_k<Tile>, "bf16_p12 TMA takes whole tiles");
    static_assert(Tile::kBlockRows * (Tile::kBlockK / 8) % kDecodeThreads == 0);
    static_assert(Stages >= 2 && PackedStages >= 1 && kThreads <= 1024 && kSharedBytes <= 99 * 1024);
};

// A barrier wait that polls instead of suspending: the decode warps and consumers hand every K tile over twice,
// and a suspended warp's wake-up stretches each handoff.
__device__ __forceinline__ void p12_spin_wait(std::uint64_t* barrier, std::uint32_t phase) {
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "spin_%=: \n"
                 "mbarrier.test_wait.parity.shared::cta.b64 done, [%0], %1;\n"
                 "@!done bra spin_%=;\n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase)
                 : "memory");
}

template <bool Spin>
__device__ __forceinline__ void p12_wait(std::uint64_t* barrier, std::uint32_t phase) {
    if constexpr (Spin) {
        p12_spin_wait(barrier, phase);
    } else {
        cta_mbarrier_wait(barrier, phase);
    }
}

inline CUtensorMap p12_tma_byte_map(const void* pointer, int rows, int row_bytes, int box_rows, int box_bytes) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(row_bytes), static_cast<std::uint64_t>(rows)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(row_bytes)};
    const std::uint32_t box[]{static_cast<std::uint32_t>(box_bytes), static_cast<std::uint32_t>(box_rows)};
    const std::uint32_t steps[]{1, 1};
    const auto status = cuTensorMapEncodeTiled(&result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(pointer),
                                               dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                               CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("bf16_p12 TMA descriptor: ") + (name ? name : "CUDA error"));
    }
    return result;
}

template <class S, bool FullTokens, class Output, bool InPlace = false>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void bf16_p12_tma_mma_kernel(
#ifdef _WIN32
    const __grid_constant__ TmaDescriptorParameter<P12TmaDescriptors, InPlace> descriptors_pointer,
#else
    const __grid_constant__ P12TmaDescriptors descriptors,
#endif
    const unsigned* __restrict__ words, const unsigned* __restrict__ escapes, Output output, int rows, int K,
    int token_offset, int count) {
#ifdef _WIN32
    const P12TmaDescriptors& descriptors = *descriptors_pointer;
#endif
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, BK = S::kBlockK, ST = S::kStages, PS = S::kPackedStages;
    constexpr int kPackedBytes = S::kCodeBytes + S::kNibbleBytes;
    const int tiles_r = rows / BR;
    const int tiles_t = (count + BT - 1) / BT;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<S>(blockIdx.x, tiles_r, tiles_t, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    extern __shared__ __align__(1024) unsigned char storage[];
    auto* a             = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b             = reinterpret_cast<__nv_bfloat16*>(storage + S::kDecodedBytes);
    auto* packed        = storage + S::kPackedOffset;
    auto* full          = reinterpret_cast<std::uint64_t*>(storage + S::kBarrierOffset);
    auto* empty         = full + ST;
    auto* packed_full   = empty + ST;
    auto* packed_empty  = packed_full + PS;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < ST; ++stage) {
            cta_mbarrier_init(full + stage, S::kDecodeWarps + 1);
            cta_mbarrier_init(empty + stage, S::kConsumerWarps);
        }
#pragma unroll
        for (int stage = 0; stage < PS; ++stage) {
            cta_mbarrier_init(packed_full + stage, 1);
            cta_mbarrier_init(packed_empty + stage, S::kDecodeWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();
    const int tiles_k = K / BK;
    const int tid     = static_cast<int>(threadIdx.x);
    if (tid < S::kProducerThreads) {
        if (tid == 0) {
#ifdef _WIN32
            acquire_staged_tensor_map<InPlace>(&descriptors.codes);
            acquire_staged_tensor_map<InPlace>(&descriptors.nibbles);
#endif
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % PS;
                cta_mbarrier_wait(packed_empty + stage, 1U ^ ((kt / PS) & 1U));
                unsigned char* codes = packed + stage * kPackedBytes;
                cta_mbarrier_arrive_expect_tx(packed_full + stage, kPackedBytes);
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                             "[%0], [%1, {%2, %3}], [%4];"
                             :
                             : "r"(smem_addr(codes)), "l"(&descriptors.codes), "r"(kt * BK), "r"(row_begin),
                               "r"(smem_addr(packed_full + stage))
                             : "memory");
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                             "[%0], [%1, {%2, %3}], [%4];"
                             :
                             : "r"(smem_addr(codes + S::kCodeBytes)), "l"(&descriptors.nibbles), "r"(kt * BK / 2),
                               "r"(row_begin), "r"(smem_addr(packed_full + stage))
                             : "memory");
            }
        } else if (tid == 32) {
#ifdef _WIN32
            acquire_staged_tensor_map<InPlace>(&descriptors.activation);
#endif
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % ST;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / ST) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, BT * BK * 2);
                bf16_tma_load<S>(b + stage * BT * BK, &descriptors.activation, kt * (BK / 64), token_begin,
                                 full + stage);
            }
        }
        return;
    }
    if (tid < S::kProducerThreads + S::kDecodeThreads) {
        // Decode: thread d expands chunks d, d + kDecodeThreads, ... of the tile (row-major, BK / 8 a row).
        constexpr int kRowChunks = BK / 8;
        const int d = tid - S::kProducerThreads;
        P12Row row_state[S::kChunkPasses];
#pragma unroll
        for (int pass = 0; pass < S::kChunkPasses; ++pass) {
            row_state[pass] = p12_row(words, row_begin + (d + pass * S::kDecodeThreads) / kRowChunks);
        }
        for (int kt = 0; kt < tiles_k; ++kt) {
            const int stage = kt % ST, source = kt % PS;
            p12_wait<S::kSpin>(empty + stage, 1U ^ ((kt / ST) & 1U));
            p12_wait<S::kSpin>(packed_full + source, (kt / PS) & 1U);
            const unsigned char* codes   = packed + source * kPackedBytes;
            const unsigned char* nibbles = codes + S::kCodeBytes;
            __nv_bfloat16* tile          = a + stage * BR * BK;
#pragma unroll
            for (int pass = 0; pass < S::kChunkPasses; ++pass) {
                const int item = d + pass * S::kDecodeThreads;
                const int row = item / kRowChunks, c = item - row * kRowChunks;
                const uint2 code   = *reinterpret_cast<const uint2*>(codes + row * BK + c * 8);
                const unsigned nib = *reinterpret_cast<const unsigned*>(nibbles + row * (BK / 2) + c * 4);
                const uint4 w      = p12_decode_chunk(code, nib, row_state[pass], kt * kRowChunks + c, escapes);
                *reinterpret_cast<uint4*>(&tile[row * BK + bf16_mma_shared_col<S>(row, c * 8)]) = w;
            }
            // The packed reads are ordered before the slot's next TMA refill; the decoded writes are released to
            // the consumers by the arrive.
            fence_reads_before_async_refill();
            __syncwarp();
            if ((tid & 31) == 0) {
                cta_mbarrier_arrive(packed_empty + source);
                cta_mbarrier_arrive(full + stage);
            }
        }
        return;
    }
    const int consumer = tid - S::kProducerThreads - S::kDecodeThreads;
    const int warp = consumer / 32, lane = consumer & 31;
    float accum[S::kMmaRows][S::kMmaTokens][4] = {};
    const auto consume = [&]<bool Live>(std::bool_constant<Live>) {
#pragma unroll S::kConsumerKUnroll
        for (int kt = 0; kt < tiles_k; ++kt) {
            const int stage = kt % ST;
            p12_wait<S::kSpin>(full + stage, (kt / ST) & 1U);
            if constexpr (Live) {
                bf16_mma_compute_stage<S>(a + stage * BR * BK, b + stage * BT * BK, accum, warp, lane);
            }
            fence_reads_before_async_refill();
            __syncwarp();
            if (lane == 0) { cta_mbarrier_arrive(empty + stage); }
        }
    };
    if constexpr (FullTokens) {
        consume(std::true_type{});
    } else {
        // A partly filled last token tile: a warp whose columns all lie past the call does no tensor-core
        // work (its sums are never stored; every stored output keeps its K order). The decode warps and the
        // barrier protocol run unchanged.
        const int first_token = token_begin + warp % S::kWarpsTokens * S::kWarpTokens;
        if (first_token < token_offset + count) {
            consume(std::true_type{});
        } else {
            consume(std::false_type{});
        }
    }
    bf16_finish_mma_tile<S, FullTokens>(output, LinearIdentityEpilogue{}, storage, accum, row_begin, token_begin,
                                        rows, token_offset + count, warp, lane);
}

// Whether `S` takes a [rows, k] packed weight: whole row and K tiles.
template <class S>
[[nodiscard]] constexpr bool p12_tma_fits(std::int32_t rows, std::int32_t k) noexcept {
    return rows % S::kBlockRows == 0 && k % S::kBlockK == 0;
}

template <class S>
void launch_bf16_p12_tma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (!p12_tma_fits<S>(w.n, w.k)) { throw std::invalid_argument("bf16_p12 TMA: rows and K must be whole tiles"); }
    const auto* xs = static_cast<const __nv_bfloat16*>(x.data);
    const P12TmaDescriptors descriptors{
        p12_tma_byte_map(w.qdata, w.n, w.k, S::kBlockRows, S::kBlockK),
        p12_tma_byte_map(w.qhigh, w.n, w.k / 2, S::kBlockRows, S::kBlockK / 2),
        bf16_tma_map<S>(xs, x.ne[1], w.k, S::kBlockTokens, S::kBlockK)};
    const auto* words   = static_cast<const unsigned*>(w.scales);
    const auto* escapes = static_cast<const unsigned*>(w.escapes);
    with_linear_output(out, w.n, [&](auto output) {
        using Output             = decltype(output);
        const auto launch_slices = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
            for_each_token_slice(x.ne[1], S::kBlockTokens, [&](int offset, int count) {
                const auto blocks = static_cast<std::int64_t>(w.n / S::kBlockRows) * div_up(count, S::kBlockTokens);
                if (blocks > 2147483647LL) { throw std::invalid_argument("bf16_p12 TMA grid exceeds grid.x"); }
                const auto launch = [&]<bool Full>() {
                    constexpr auto kernel = bf16_p12_tma_mma_kernel<S, Full, Output, InPlace>;
                    bf16_prepare_shared<S::kSharedBytes, kernel>();
                    kernel<<<static_cast<unsigned>(blocks), S::kThreads, S::kSharedBytes, stream>>>(
                        parameter, words, escapes, output, w.n, w.k, offset, count);
                    CUDA_CHECK(cudaGetLastError());
                };
                if (count % S::kBlockTokens == 0) {
                    launch.template operator()<true>();
                } else {
                    launch.template operator()<false>();
                }
            });
        };
#ifdef _WIN32
        with_tma_descriptor_parameter(p12_tma_descriptor_staging(), descriptors, stream, launch_slices);
#else
        launch_slices(std::false_type{}, descriptors);
#endif
    });
}

} // namespace infernix::ops::detail
