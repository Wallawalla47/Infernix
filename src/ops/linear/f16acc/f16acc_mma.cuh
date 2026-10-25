#pragma once

// f16-accumulate-prefill's GEMM (f16acc_plan.h): E4M3 or int8 weights x FP16-encoded activations on FP16
// Tensor Cores with FP16 accumulation over 32-value K chunks and an FP32 carry.
//
//   out[N,T] = row_scale[N] * inverse[T] * sum_chunks( g[N,chunk] * fp16_sum_k(code[N,k] * x16[k,T]) )
//
// g is 1 for row-scaled FP8 (fp8_e4m3fn_row_bf16 / _f32: the row's word scales in the epilogue), the
// chunk's UE8M0 power of two for MXFP8 (fp8_e4m3fn_g32_e8m0 in g32_scale_rows16_v1: no row word), and the
// group's FP16 scale for Q8 (q8_g32_fp16, row-major codes and scales: no row word; F16AccScales::Fp16,
// cp.async producers only). An int8 code is an FP16 value too, so every weight enters the MMA exactly;
// a Q8 chunk's partial sums are bounded by 32 * 128 * 2 = 8,192.
//
// The pipeline is the exact A16 route's (fp8_a16_ring_mma.cuh, mxfp8_a16_tma_mma.cuh): the same
// schedules (Fp8A16RingMmaSchedule, with Mxfp8ProducerWarps or Mxfp8TmaProducer for the producer), the
// same 128-byte-swizzled stages and mbarrier ring, the encoded activation staged in place of the BF16
// one (both 2-byte values). Consumers widen each code pair exactly to an FP16 pair (cvt.rn.f16x2.e4m3x2)
// and run m16n8k16 f16.f16.f16.f16 MMAs: a chunk's first k16 step starts from zero, its second
// accumulates, and the chunk's half2 partial sums then add into the FP32 carry (FFMA by the group's
// power of two for MXFP8; that group scale comes from the stage's scale bytes, which the producer stages
// as the exact MXFP8 route does).
//
// A TMA-staged stage is released only after each consumer thread's fence_reads_before_async_refill
// (ops/common/mbarrier.cuh; the 2026-10-20 proxy race); cp.async stages need no fence.
//
// Launches are ordinary stream-ordered launches (the route serves prefill widths only), after the
// encoder kernel that writes the operand.

#include "core/device.h"
#include "core/tma_descriptor_staging.cuh"
#include "ops/common/math.cuh"
#include "ops/common/math.h"
#include "ops/common/mbarrier.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/f16acc/f16acc_plan.h"
#include "ops/linear/fp8/fp8_a16_ring_mma.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_shared.cuh"
#include "ops/linear/mxfp8/mxfp8_a16_tma_mma.cuh"
#include "ops/linear/mxfp8/mxfp8_format.h"

#include <cuda.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace infernix::ops::detail {

// The weight's scales: a row word (row-scaled FP8, in the epilogue), a UE8M0 byte per 32-K group (MXFP8)
// or an FP16 word per 32-K group (Q8, whose codes are int8).
enum class F16AccScales : std::uint8_t { Row, Ue8m0, Fp16 };

// The operands of one f16-accumulate projection: row_scales of word type Scale (Row), or the group
// scales (Ue8m0 bytes, Fp16 words).
template <F16AccScales Scales, class Scale = __nv_bfloat16>
struct F16AccOperands {
    const __half* x;       // the encoded activation [K, T]
    const float* inverses; // 2^-s_t per column
    const std::uint8_t* codes;
    const Scale* row_scales;           // Row only
    const std::uint8_t* group_scales;  // Ue8m0 / Fp16 only
    int rows, k, tokens;
};

// Exact E4M3 pair widening into an FP16 pair (every E4M3 value is an FP16 value).
__device__ __forceinline__ unsigned fp8_e4m3x2_to_f16x2_bits(unsigned packed) {
    unsigned result;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(static_cast<std::uint16_t>(packed)));
    return result;
}

// Exact int8 pair widening into an FP16 pair: each byte biased to q + 128 becomes the mantissa of the FP16
// 1024 + (q + 128) (0x64xx), less 1152.
__device__ __forceinline__ unsigned s8x2_to_f16x2_bits(unsigned packed) {
    const unsigned biased = __byte_perm(packed ^ 0x8080U, 0x64646464U, 0x4140U);
    const __half2 value   = __hsub2(half2_from_bits(biased), __floats2half2_rn(1152.0F, 1152.0F));
    return *reinterpret_cast<const unsigned*>(&value);
}

// Shared storage: the schedule's activation and weight stages, the MXFP8 group-scale stages, the
// epilogue's output staging over them, then the barriers. Row-scaled FP8 has no scale stage, and Q8's
// FP16 group scales are read by the consumers from global memory (f16acc_compute_stage): staged, they
// would put the 98,304-byte three-stage T64 ring 48 bytes past the 99 KiB a CTA may use.
template <class Schedule, F16AccScales Scales>
struct F16AccLayout {
    static constexpr int kScaleStageBytes =
        Scales == F16AccScales::Ue8m0 ? Schedule::kBlockRows * Schedule::kBlockK / 32 : 0;
    static constexpr int kScaleStageStride = (kScaleStageBytes + 127) / 128 * 128;
    static constexpr int kScaleOffset      = Schedule::kStages * Schedule::kStageBytes;
    static constexpr int kPipelineBytes    = kScaleOffset + Schedule::kStages * kScaleStageStride;
    static constexpr int kStorageBytes =
        ((kPipelineBytes > Schedule::kOutputStagingBytes ? kPipelineBytes : Schedule::kOutputStagingBytes) + 127) /
        128 * 128;
    static constexpr int kSharedBytes = kStorageBytes + Schedule::kBarrierBytes;
    static_assert(kScaleStageBytes % 16 == 0 && kScaleOffset % 128 == 0);
    static_assert(kSharedBytes <= 99 * 1024, "the f16-accumulate schedule's stages exceed shared memory");
};

// The B fragments come by ldmatrix, one per four n8 tiles, instead of two 16-bit shared loads per tile and
// k16 step (that form ran at ~215 of 419 FP16 TF at 4K-32K tokens; ldmatrix: FP4-Local pp32768 +1.4 % end to
// end, 2026-10-25). Lane (g, t) holds codes 4t .. 4t + 3 of row g of each tile's 16-byte k slice; the encoder
// writes each 16-value activation block permuted (f16acc_encode.cu) so that the A fragment's slots
// (2t, 2t + 1) and (2t + 8, 2t + 9) hold those values. Each k16 step sums the same 16 products.
// scale_stage is the stage's MXFP8 scale bytes, or for Q8 the global FP16 group scales of the CTA's first
// row at the stage's first group (rows of scale_stride words).
template <class Schedule, F16AccScales Scales>
__device__ __forceinline__ void
f16acc_compute_stage(const unsigned char* activation_stage, const std::uint8_t* weight_stage,
                     const std::uint8_t* scale_stage, int scale_stride,
                     float (&carry)[Schedule::kMmaTokens][Schedule::kMmaRows][4], int warp, int lane) {
    constexpr bool GroupScales = Scales != F16AccScales::Row;
    constexpr int kRowBytes = Schedule::kActivationRowBytes;
    static_assert(kRowBytes == 128, "the activation swizzle assumes one 128-byte span per row");
    constexpr int kStepsPerHalf   = 64 / 16;
    constexpr int kChunkSteps     = kF16AccumulateChunk / 16;
    constexpr int kGroupsPerStage = Schedule::kBlockK / 32;
    static_assert(Schedule::kMmaK % kChunkSteps == 0, "a stage holds whole chunks");
    const int warp_token   = warp / Schedule::kWarpsRows;
    const int warp_row     = warp % Schedule::kWarpsRows;
    const int a_matrix     = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_byte       = (a_matrix >> 1) * 16;
    // The accumulator's weight rows (C columns 2t and 2t + 1 of each n8 tile).
    const int c_row = 2 * (lane & 3);
    // Q8: the chunk's FP16 group scales of rows c_row and c_row + 1 of each n8 tile, read-only loads issued
    // at the chunk's first k16 step and used after its last (one half2 register per n8 tile: the T128 ring
    // runs at its 96-register cap).
    __half2 q8_scales[Schedule::kMmaRows];
    const auto* q8_words = reinterpret_cast<const __half*>(scale_stage);
    unsigned partial[Schedule::kMmaTokens][Schedule::kMmaRows][2];
#pragma unroll
    for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
        unsigned a[Schedule::kMmaTokens][4];
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
            const int row     = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
            const int logical = (k_step % kStepsPerHalf) * 32 + a_byte;
            ldmatrix_x4(a[mma_token][0], a[mma_token][1], a[mma_token][2], a[mma_token][3],
                        smem_addr(activation_stage + (k_step / kStepsPerHalf) * Schedule::kActivationHalfBytes +
                                  row * kRowBytes + (logical ^ ((row & 7) << 4))));
        }
        unsigned b[Schedule::kMmaRows][2];
        {
            static_assert(Schedule::kMmaRows <= 2 || Schedule::kMmaRows % 4 == 0, "ldmatrix B takes 1, 2 or 4k tiles");
            // The 16-byte k slice of weight row `row` (lane l addresses row l & 7 of matrix l >> 3).
            const auto slice = [&](int row) {
                return smem_addr(weight_stage + row * Schedule::kBlockK + fp8_mma_shared_byte<Schedule>(row, k_step * 16));
            };
            unsigned codes[Schedule::kMmaRows < 2 ? 2 : Schedule::kMmaRows];
            if constexpr (Schedule::kMmaRows % 4 == 0) {
#pragma unroll
                for (int group = 0; group < Schedule::kMmaRows / 4; ++group) {
                    const int row = warp_row * Schedule::kWarpRows + (group * 4 + (lane >> 3)) * 8 + (lane & 7);
                    ldmatrix_x4(codes[4 * group], codes[4 * group + 1], codes[4 * group + 2], codes[4 * group + 3],
                                slice(row));
                }
            } else {
                // One or two tiles: x2 (with one tile, both matrices are its rows).
                const int tile = Schedule::kMmaRows == 2 ? ((lane >> 3) & 1) : 0;
                ldmatrix_x2(codes[0], codes[1], slice(warp_row * Schedule::kWarpRows + tile * 8 + (lane & 7)));
            }
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                if constexpr (Scales == F16AccScales::Fp16) {
                    b[mma_row][0] = s8x2_to_f16x2_bits(codes[mma_row]);
                    b[mma_row][1] = s8x2_to_f16x2_bits(codes[mma_row] >> 16);
                } else {
                    b[mma_row][0] = fp8_e4m3x2_to_f16x2_bits(codes[mma_row]);
                    b[mma_row][1] = fp8_e4m3x2_to_f16x2_bits(codes[mma_row] >> 16);
                }
            }
        }
        const bool first = k_step % kChunkSteps == 0;
        if constexpr (Scales == F16AccScales::Fp16) {
            if (first) {
                const int group = k_step / kChunkSteps;
#pragma unroll
                for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                    const int row         = warp_row * Schedule::kWarpRows + mma_row * 8 + c_row;
                    const __half* words   = q8_words + static_cast<std::size_t>(row) * scale_stride + group;
                    q8_scales[mma_row]    = __halves2half2(__ldg(words), __ldg(words + scale_stride));
                }
            }
        }
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                unsigned(&d)[2] = partial[mma_token][mma_row];
                if (first) {
                    mma_f16_f16acc_zero(d[0], d[1], a[mma_token][0], a[mma_token][1], a[mma_token][2],
                                        a[mma_token][3], b[mma_row][0], b[mma_row][1]);
                } else {
                    mma_f16_f16acc(d[0], d[1], a[mma_token][0], a[mma_token][1], a[mma_token][2], a[mma_token][3],
                                   b[mma_row][0], b[mma_row][1]);
                }
            }
        }
        if (k_step % kChunkSteps != kChunkSteps - 1) continue;
        // The chunk is complete: its FP16 sums into the FP32 carry, in the C-fragment order.
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            float2 scale = make_float2(1.0F, 1.0F);
            if constexpr (Scales == F16AccScales::Ue8m0) {
                // Rows c_row and c_row + 1 share a 16-row scale tile (c_row is even): one 16-bit load of
                // both UE8M0 words of this chunk's group, each 2^(e - 127) as an FP32 power of two.
                const int row      = warp_row * Schedule::kWarpRows + mma_row * 8 + c_row;
                const int group    = k_step / kChunkSteps;
                const unsigned two = *reinterpret_cast<const std::uint16_t*>(
                    scale_stage + (row >> 4) * kGroupsPerStage * 16 + group * 16 + (row & 15));
                scale = make_float2(__uint_as_float((two & 0xFFU) << 23), __uint_as_float((two >> 8) << 23));
            } else if constexpr (Scales == F16AccScales::Fp16) {
                scale = __half22float2(q8_scales[mma_row]);
            }
#pragma unroll
            for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
                const float2 low  = __half22float2(half2_from_bits(partial[mma_token][mma_row][0]));
                const float2 high = __half22float2(half2_from_bits(partial[mma_token][mma_row][1]));
                float(&c)[4]      = carry[mma_token][mma_row];
                if constexpr (GroupScales) {
                    c[0] = fmaf(low.x, scale.x, c[0]);
                    c[1] = fmaf(low.y, scale.y, c[1]);
                    c[2] = fmaf(high.x, scale.x, c[2]);
                    c[3] = fmaf(high.y, scale.y, c[3]);
                } else {
                    c[0] += low.x;
                    c[1] += low.y;
                    c[2] += high.x;
                    c[3] += high.y;
                }
            }
        }
    }
}

struct alignas(128) F16AccTmaDescriptors {
    CUtensorMap activation;
    CUtensorMap weight;
    CUtensorMap scale; // MXFP8 only
};

#ifdef _WIN32
// The Windows descriptor staging of every f16-accumulate TMA launch (core/tma_descriptor_staging.cuh).
inline TmaDescriptorStaging<F16AccTmaDescriptors>& f16acc_tma_descriptor_staging() {
    static TmaDescriptorStaging<F16AccTmaDescriptors> staging;
    return staging;
}
#endif

// The encoded activation [K, T] as a 2-D FP16 tensor; a box is 64 values of BlockTokens columns (one
// 128-byte swizzle span per column), columns past T zero-filled.
inline CUtensorMap f16acc_activation_map(const __half* pointer, int tokens, int k, int block_tokens) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(tokens)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(k) * sizeof(__half)};
    const std::uint32_t box[]{64, static_cast<std::uint32_t>(block_tokens)};
    const std::uint32_t steps[]{1, 1};
    const auto status = cuTensorMapEncodeTiled(&result, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<__half*>(pointer),
                                               dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("f16 accumulate TMA activation descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

template <class Schedule, bool FullTokens, F16AccScales Scales, class Scale, class Output, class Epilogue,
          bool InPlace = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void f16acc_mma_kernel(
#ifdef _WIN32
    const __grid_constant__ TmaDescriptorParameter<F16AccTmaDescriptors, InPlace> descriptors_pointer,
#else
    const __grid_constant__ F16AccTmaDescriptors descriptors,
#endif
    F16AccOperands<Scales, Scale> operands, Output output, Epilogue epilogue, int token_offset, int count) {
#ifdef _WIN32
    const F16AccTmaDescriptors& descriptors = *descriptors_pointer;
#endif
    using Layout       = F16AccLayout<Schedule, Scales>;
    constexpr bool kTma = kMxfp8TmaProducer<Schedule>;
    constexpr bool GroupScales = Scales != F16AccScales::Row;
    static_assert(!(kTma && Scales == F16AccScales::Fp16), "Q8 group scales are staged by cp.async producers");
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    const int k       = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int tiles_k = k / BK;
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(static_cast<int>(blockIdx.x), operands.rows / BR, div_up(count, BT), row_tile,
                                       token_tile);
    const int row_begin   = row_tile * BR;
    const int token_begin = token_offset + token_tile * BT;

    extern __shared__ __align__(1024) unsigned char f16acc_shared[];
    auto* activation = f16acc_shared;
    auto* weight     = activation + S * Schedule::kActivationStageBytes;
    auto* scales     = f16acc_shared + Layout::kScaleOffset;
    auto* full       = reinterpret_cast<std::uint64_t*>(f16acc_shared + Layout::kStorageBytes);
    auto* empty      = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, kTma ? 1 : Schedule::kProducerThreads);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

#ifdef __CUDACC_RDC__
    // Relocatable device code makes ptxas drop setmaxnreg (src/ops/CMakeLists.txt): a split schedule is
    // built only in f16acc_split_registers.cu, in the non-RDC archive.
    static_assert(!requires { Schedule::kConsumerRegisters; },
                  "f16 accumulate split-register kernels need a non-RDC translation unit");
#endif
    if constexpr (!kTma) {
        if (threadIdx.x < Schedule::kProducerThreads) {
            if constexpr (requires { Schedule::kConsumerRegisters; }) {
                asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(Schedule::kProducerRegisters) : "memory");
            }
            const int lane          = static_cast<int>(threadIdx.x);
            constexpr int kGroups   = BK / 32;
            constexpr int kWeightChunks = BK / 16; // 16-byte chunks per weight row
            // Only the slice's columns: a row past them feeds only its own output row, which the epilogue
            // never stores, so it is left as it is.
            const int rows = FullTokens ? BT : min(BT, token_offset + count - token_begin);
            const auto* x  = reinterpret_cast<const unsigned char*>(operands.x);
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                unsigned char* w          = weight + stage * Schedule::kWeightStageBytes;
                const std::uint8_t* codes = operands.codes + static_cast<std::size_t>(row_begin) * k + kt * BK;
                for (int i = lane; i < BR * kWeightChunks; i += Schedule::kProducerThreads) {
                    const int r = i / kWeightChunks, c = i % kWeightChunks;
                    cp_async<16, Cache::cg>(w + r * BK + fp8_mma_shared_byte<Schedule>(r, c * 16),
                                            codes + static_cast<std::size_t>(r) * k + c * 16);
                }
                if constexpr (Scales == F16AccScales::Ue8m0) {
                    unsigned char* sc = scales + stage * Layout::kScaleStageStride;
                    for (int i = lane; i < (BR / 16) * kGroups; i += Schedule::kProducerThreads) {
                        const int t = i / kGroups, g = i % kGroups;
                        cp_async<16, Cache::cg>(
                            sc + (t * kGroups + g) * 16,
                            operands.group_scales +
                                (static_cast<std::size_t>(row_begin / 16 + t) * (k / 32) + kt * kGroups + g) * 16);
                    }
                }
                // FP16 column rows: each 64-value half a 128-byte span per row under the 128-byte swizzle.
#pragma unroll
                for (int half = 0; half < Schedule::kActivationHalves; ++half) {
                    unsigned char* a =
                        activation + stage * Schedule::kActivationStageBytes + half * Schedule::kActivationHalfBytes;
                    for (int i = lane; i < rows * 8; i += Schedule::kProducerThreads) {
                        const int r = i >> 3, c = i & 7;
                        cp_async<16, Cache::cg>(a + r * 128 + ((c ^ (r & 7)) << 4),
                                                x + (static_cast<std::size_t>(token_begin + r) * k + kt * BK +
                                                     half * 64 + c * 8) * sizeof(__half));
                    }
                }
                fp8_cp_async_arrive(full + stage);
            }
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            return;
        }
    } else if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef _WIN32
            acquire_staged_tensor_map<InPlace>(&descriptors.activation);
            acquire_staged_tensor_map<InPlace>(&descriptors.weight);
            if constexpr (GroupScales) { acquire_staged_tensor_map<InPlace>(&descriptors.scale); }
#endif
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, Schedule::kStageBytes + Layout::kScaleStageBytes);
                fp8_tma_load(weight + stage * Schedule::kWeightStageBytes, &descriptors.weight, kt * BK, row_begin,
                             full + stage);
                if constexpr (GroupScales) {
                    mxfp8_tma_load_scales(scales + stage * Layout::kScaleStageStride, &descriptors.scale,
                                          kt * (BK / 32), row_begin / 16, full + stage);
                }
#pragma unroll
                for (int half = 0; half < Schedule::kActivationHalves; ++half) {
                    fp8_tma_load(activation + stage * Schedule::kActivationStageBytes +
                                     half * Schedule::kActivationHalfBytes,
                                 &descriptors.activation, kt * BK + half * 64, token_begin, full + stage);
                }
            }
        }
        return;
    }

    if constexpr (requires { Schedule::kConsumerRegisters; }) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(Schedule::kConsumerRegisters) : "memory");
    }
    const int tid  = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp = tid >> 5, lane = tid & 31;
    float carry[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        const std::uint8_t* stage_scales =
            Scales == F16AccScales::Fp16
                ? operands.group_scales + (static_cast<std::size_t>(row_begin) * (k / 32) + kt * (BK / 32)) * 2
                : reinterpret_cast<const std::uint8_t*>(scales + stage * Layout::kScaleStageStride);
        f16acc_compute_stage<Schedule, Scales>(
            activation + stage * Schedule::kActivationStageBytes,
            reinterpret_cast<const std::uint8_t*>(weight + stage * Schedule::kWeightStageBytes), stage_scales, k / 32,
            carry, warp, lane);
        // Release only after every lane in this consumer warp has finished its shared reads; a TMA refill
        // is an async-proxy write, so each thread first orders its generic-proxy reads before it.
        if constexpr (kTma) { fence_reads_before_async_refill(); }
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    // All consumers must finish reading staged inputs before the epilogue reuses the storage.
    __syncthreads();
    // Each carry times its column's 2^-s_t (exact) and, for row-scaled FP8, the row's word.
    if constexpr (GroupScales) {
        fp8_finish_mma_tile<Schedule, FullTokens>(output, epilogue, Fp8UnscaledRows{}, f16acc_shared, carry,
                                                  operands.inverses, static_cast<const __nv_bfloat16*>(nullptr),
                                                  row_begin, token_begin, operands.rows, token_offset + count, warp,
                                                  lane);
    } else {
        fp8_finish_mma_tile<Schedule, FullTokens>(output, epilogue, Fp8IdentityRows{}, f16acc_shared, carry,
                                                  operands.inverses, operands.row_scales, row_begin, token_begin,
                                                  operands.rows, token_offset + count, warp, lane);
    }
}

template <class Schedule, F16AccScales Scales, class Scale>
inline void validate_f16acc_operands(const F16AccOperands<Scales, Scale>& p) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    const bool scales = Scales == F16AccScales::Row     ? aligned(p.row_scales, sizeof(Scale) == 4 ? 8 : 4)
                        : Scales == F16AccScales::Ue8m0 ? aligned(p.group_scales, 16)
                                                        : aligned(p.group_scales, Schedule::kBlockK / 16);
    if (p.rows <= 0 || p.k <= 0 || p.tokens <= 0 || p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
        (Scales == F16AccScales::Ue8m0 && Schedule::kBlockRows % 16) || !aligned(p.x, 16) || !aligned(p.inverses, 4) ||
        !aligned(p.codes, 16) || !scales) {
        throw std::invalid_argument("f16 accumulate GEMM requires complete row/K tiles and aligned operands");
    }
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK) throw std::invalid_argument("f16 accumulate static K does not match operands");
    }
}

// MXFP8 (NVIDIA FP4-Local) calls of at least kF16AccRowFastTokens tokens take their schedule with the rows of
// one token tile walked before the next (the weights stay in L2 and each FP16 token tile is read once) instead
// of the tokens of one row tile: at 32,768 tokens the FP16 activation (168 MB at K = 2,560) exceeds L2 and every
// row tile streamed it again from DRAM. FP4-Local pp32768 +3.8 % end to end, pp4096 neutral (2026-10-25). Q8
// keeps its rasters (Dense8's problems 0.79-1.14x row-fast, flat end to end, 2026-10-24), as does row FP8
// (unmeasured). The same bits: each output's sums do not depend on the order of the tiles.
template <class Schedule>
struct F16AccRowFast : Schedule {
    static constexpr Fp8MmaRaster kRaster = Fp8MmaRaster::RowFast;
};

template <F16AccScales Scales>
inline constexpr std::int64_t kF16AccRowFastTokens = Scales == F16AccScales::Ue8m0 ? 8192 : 0;

// Registers per thread a one-CTA-per-SM launch of `threads` gets: the busiest SM sub-partition (warps are
// dealt to the four in turn) holds ceil(warps / 4) warps of its 16,384 registers, in units of 8.
constexpr int f16acc_launch_registers(int threads) {
    const int warps_per_partition = (threads / 32 + 3) / 4;
    const int registers           = 16384 / (warps_per_partition * 32) / 8 * 8;
    return registers < 248 ? registers : 248;
}

// A 16-consumer-warp cp.async schedule (T128 4x4, T256 2x8) staged by one producer warpgroup that hands its
// registers to the consumers (setmaxnreg). With one producer warp the CTA's 17 warps put 5 on an SM
// sub-partition, so every thread was held to 96 registers and the consumers spilled 8-72 B (cuda_stack review,
// 2026-10-24); a warpgroup at 32 registers releases 64 per thread to its sub-partition's 4 consumer warps: 112
// each. The producers' cp.async copies spread over 128 threads; every consumer instruction and so every bit is
// the base schedule's. Chosen per call by f16acc_split_registers.
template <class Base>
struct F16AccSplitRegisters : Base {
    static_assert(Base::kProducerThreads == 32 && Base::kConsumerWarps % 4 == 0 && Base::kMinBlocksPerSm == 1);
    static constexpr int kProducerThreads   = 128;
    static constexpr int kThreads           = kProducerThreads + Base::kConsumerWarps * 32;
    static constexpr int kLaunchRegisters   = f16acc_launch_registers(kThreads);
    static constexpr int kProducerRegisters = 32;
    // Only what the producers release: a consumer warp's increase waits for registers the CTA freed.
    static constexpr int kConsumerRegisters =
        kLaunchRegisters + (kLaunchRegisters - kProducerRegisters) / (Base::kConsumerWarps / 4) / 8 * 8;
    static_assert(kConsumerRegisters > kLaunchRegisters && kConsumerRegisters <= 248);
};

template <class Schedule>
inline constexpr bool kF16AccSplitRegisters = requires { Schedule::kConsumerRegisters; };

template <class Schedule>
inline constexpr bool kF16AccSplitEligible = [] {
    if constexpr (kMxfp8TmaProducer<Schedule> || kF16AccSplitRegisters<Schedule>) {
        return false;
    } else {
        return Schedule::kProducerThreads == 32 && Schedule::kConsumerWarps == 16 && Schedule::kMinBlocksPerSm == 1;
    }
}();

// The split schedule where it measured faster than the 17-warp one (ABBA, Flash-Next's dense shapes,
// 2026-10-24, RTX 5090; split / unsplit time). Q8 (FP16 group scales): the default raster at every width
// (0.91-0.97x at 4,096-32,768 columns); row-fast below 2^28 rows x tokens (0.94-0.97x; 1.01-1.02x from
// 16,384 x 16,384). Row-scaled FP8: row-fast at every width (0.96-0.99x at 16,384 and 32,768); the default
// raster keeps the 17-warp schedule (0.97-1.04x at 4,096, 0.99-1.01x at 16,384, 0.99-1.05x at 32,768: N 13,312
// and 16,384 lose 4-5 %).
template <class Schedule, F16AccScales Scales>
inline constexpr bool kF16AccSplitRaster = Schedule::kRaster == Fp8MmaRaster::RowFast || Scales == F16AccScales::Fp16;

template <class Schedule, F16AccScales Scales>
bool f16acc_split_registers(std::int64_t rows, std::int64_t tokens) {
    if constexpr (Schedule::kRaster == Fp8MmaRaster::RowFast && Scales == F16AccScales::Fp16) {
        return rows * tokens < (std::int64_t{1} << 28);
    } else {
        return true;
    }
}

template <class Schedule, F16AccScales Scales, class Scale, class Output, class Epilogue>
void launch_f16acc_mma(const F16AccOperands<Scales, Scale>& p, Output output, Epilogue epilogue,
                       cudaStream_t stream) {
    // A split schedule is chosen after the raster (it wraps the row-fast one), so it is never re-rastered.
    if constexpr (Schedule::kRaster != Fp8MmaRaster::RowFast && !kF16AccSplitRegisters<Schedule> &&
                  (kF16AccRowFastTokens<Scales> > 0)) {
        if (p.tokens >= kF16AccRowFastTokens<Scales>) {
            launch_f16acc_mma<F16AccRowFast<Schedule>>(p, output, epilogue, stream);
            return;
        }
    }
    if constexpr (kF16AccSplitEligible<Schedule> && kF16AccSplitRaster<Schedule, Scales>) {
        if (f16acc_split_registers<Schedule, Scales>(p.rows, p.tokens)) {
            launch_f16acc_mma<F16AccSplitRegisters<Schedule>>(p, output, epilogue, stream);
            return;
        }
    }
    validate_f16acc_operands<Schedule>(p);
    constexpr bool kTma = kMxfp8TmaProducer<Schedule>;
    F16AccTmaDescriptors descriptors{};
    if constexpr (kTma) {
        descriptors.activation = f16acc_activation_map(p.x, p.tokens, p.k, Schedule::kBlockTokens);
        descriptors.weight     = fp8_tma_map(p.codes, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK);
        if constexpr (Scales == F16AccScales::Ue8m0) {
            descriptors.scale = mxfp8_scale_tma_map(p.group_scales, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK);
        }
    }
    const auto launch_slices = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
        for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
            const auto launch = [&]<bool Full>() {
                constexpr auto kernel = f16acc_mma_kernel<Schedule, Full, Scales, Scale, Output, Epilogue, InPlace>;
                if constexpr (kF16AccSplitRegisters<Schedule>) {
                    // The register hand-over assumes the launch count: compiled at another, a consumer's
                    // setmaxnreg.inc could wait for registers no warp frees.
                    static bool checked = false;
                    if (!checked) {
                        cudaFuncAttributes attributes{};
                        CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
                        if (attributes.numRegs != Schedule::kLaunchRegisters) {
                            throw std::runtime_error("f16 accumulate split-register kernel compiled at " +
                                                     std::to_string(attributes.numRegs) + " registers, not " +
                                                     std::to_string(Schedule::kLaunchRegisters));
                        }
                        checked = true;
                    }
                }
                const int dynamic = fp8_prepare_shared<F16AccLayout<Schedule, Scales>::kSharedBytes, kernel, true>();
                kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(parameter, p, output, epilogue, offset, count);
                CUDA_CHECK(cudaGetLastError());
            };
            count % Schedule::kBlockTokens == 0 ? launch.template operator()<true>()
                                                : launch.template operator()<false>();
        });
    };
#ifdef _WIN32
    if constexpr (kTma) {
        with_tma_descriptor_parameter(f16acc_tma_descriptor_staging(), descriptors, stream, launch_slices);
    } else {
        launch_slices(std::false_type{}, static_cast<const F16AccTmaDescriptors*>(nullptr));
    }
#else
    launch_slices(std::false_type{}, descriptors);
#endif
}

// One schedule of the route into any linear output: the encoder, then the GEMM.
template <class Geometry, class Schedule, class Output>
void fp8_f16acc(const Tensor& x, const Weight& w, Output output, F16AccumulateWorkspace workspace,
                cudaStream_t stream) {
    require_fp8_scale_word<__nv_bfloat16>(w);
    launch_f16_accumulate_encode(x, workspace, stream);
    const F16AccOperands<F16AccScales::Row> p{workspace.x, workspace.inverses, static_cast<const std::uint8_t*>(w.qdata),
                                              static_cast<const __nv_bfloat16*>(w.scales), nullptr, w.n, w.k, x.ne[1]};
    launch_f16acc_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(p, output, LinearIdentityEpilogue{},
                                                                           stream);
}

template <class Geometry, class Schedule, class Output>
void mxfp8_f16acc(const Tensor& x, const Weight& w, Output output, F16AccumulateWorkspace workspace,
                  cudaStream_t stream) {
    launch_f16_accumulate_encode(x, workspace, stream);
    const F16AccOperands<F16AccScales::Ue8m0> p{workspace.x, workspace.inverses, static_cast<const std::uint8_t*>(w.qdata),
                                                static_cast<const __nv_bfloat16*>(nullptr),
                                                static_cast<const std::uint8_t*>(w.scales), w.n, w.k, x.ne[1]};
    launch_f16acc_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(p, output, LinearIdentityEpilogue{},
                                                                           stream);
}

// Q8 (q8_g32_fp16: int8 codes [N, K] and FP16 group scales [N, K / 32], row-major, K a multiple of 128):
// the FP16 group scale folds each chunk into the carry. Its schedules stage by cp.async.
template <class Geometry, class Schedule, class Output>
void q8_f16acc(const Tensor& x, const Weight& w, Output output, F16AccumulateWorkspace workspace, cudaStream_t stream) {
    if (w.padded_shape[1] != w.k) { throw std::invalid_argument("q8 f16-accumulate: K must need no padding"); }
    launch_f16_accumulate_encode(x, workspace, stream);
    const F16AccOperands<F16AccScales::Fp16> p{workspace.x, workspace.inverses, static_cast<const std::uint8_t*>(w.qdata),
                                               static_cast<const __nv_bfloat16*>(nullptr),
                                               static_cast<const std::uint8_t*>(w.scales), w.n, w.k, x.ne[1]};
    launch_f16acc_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(p, output, LinearIdentityEpilogue{},
                                                                           stream);
}

template <class Geometry, class Schedule>
void q8_linear_f16acc(const Tensor& x, const Weight& w, Tensor& out, F16AccumulateWorkspace workspace,
                      cudaStream_t stream) {
    q8_f16acc<Geometry, Schedule>(x, w, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, workspace,
                                  stream);
}

// The plain-linear forms ([N, T] output).
template <class Geometry, class Schedule>
void fp8_linear_f16acc(const Tensor& x, const Weight& w, Tensor& out, F16AccumulateWorkspace workspace,
                       cudaStream_t stream) {
    fp8_f16acc<Geometry, Schedule>(x, w, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, workspace,
                                   stream);
}

template <class Geometry, class Schedule>
void mxfp8_linear_f16acc(const Tensor& x, const Weight& w, Tensor& out, F16AccumulateWorkspace workspace,
                         cudaStream_t stream) {
    mxfp8_f16acc<Geometry, Schedule>(x, w, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, workspace,
                                     stream);
}

// Every split-register launch the routes reach, each built once in f16acc_split_registers.cu (the non-RDC
// archive: relocatable device code drops setmaxnreg) and declared here so that no including translation unit
// instantiates it. X(schedule, scales, output) for the Q8 problems (q8_f16acc.cu: T128 past 3,072 tokens; plain,
// GDN and QSA-group outputs) in the token-fast and row-fast rasters, and the row-scaled FP8 shapes (fp8/shapes:
// T128 and T64 x 256 rows) in the row-fast one (kF16AccSplitRaster). A route that reaches another fails the RDC
// static_assert in f16acc_mma_kernel.
using F16AccRingT128R128 = Fp8A16RingMmaSchedule<128, 128, 4, 4, 2, 1>;
using F16AccRingT64R256  = Fp8A16RingMmaSchedule<64, 256, 2, 8, 2, 1>;
using F16AccT128K2560    = Fp8ScheduleInstance<F16AccRingT128R128, 2560>;
using F16AccT128K6144    = Fp8ScheduleInstance<F16AccRingT128R128, 6144>;
using F16AccT128K640     = Fp8ScheduleInstance<F16AccRingT128R128, 640>;
using F16AccT64R256K2560 = Fp8ScheduleInstance<F16AccRingT64R256, 2560>;
using F16AccT64R256K640  = Fp8ScheduleInstance<F16AccRingT64R256, 640>;
using F16AccQ8GdnOutput  = LinearBf16SegmentedOutput<10240, 6144>;
using F16AccQ8QsaOutput  = LinearBf16SegmentedOutput<6144, 6144, 512, 512, 512, 128>;

#define INFERNIX_F16ACC_SPLIT_RASTERS(X, SCHEDULE, SCALES, OUTPUT) \
    X(SCHEDULE, SCALES, OUTPUT)                                    \
    X(F16AccRowFast<SCHEDULE>, SCALES, OUTPUT)
#define INFERNIX_F16ACC_SPLIT_LAUNCHES(X)                                                           \
    INFERNIX_F16ACC_SPLIT_RASTERS(X, F16AccT128K2560, F16AccScales::Fp16, LinearBf16Output)         \
    INFERNIX_F16ACC_SPLIT_RASTERS(X, F16AccT128K2560, F16AccScales::Fp16, F16AccQ8GdnOutput)        \
    INFERNIX_F16ACC_SPLIT_RASTERS(X, F16AccT128K2560, F16AccScales::Fp16, F16AccQ8QsaOutput)        \
    INFERNIX_F16ACC_SPLIT_RASTERS(X, F16AccT128K6144, F16AccScales::Fp16, LinearBf16Output)         \
    INFERNIX_F16ACC_SPLIT_RASTERS(X, F16AccT128K640, F16AccScales::Fp16, LinearBf16Output)          \
    X(F16AccRowFast<F16AccT128K2560>, F16AccScales::Row, LinearBf16Output)                          \
    X(F16AccRowFast<F16AccT128K6144>, F16AccScales::Row, LinearBf16Output)                          \
    X(F16AccRowFast<F16AccT128K640>, F16AccScales::Row, LinearBf16Output)                           \
    X(F16AccRowFast<F16AccT64R256K2560>, F16AccScales::Row, LinearBf16Output)                       \
    X(F16AccRowFast<F16AccT64R256K640>, F16AccScales::Row, LinearBf16Output)
#define INFERNIX_F16ACC_SPLIT_DECLARE(SCHEDULE, SCALES, OUTPUT)                                                    \
    extern template void launch_f16acc_mma<F16AccSplitRegisters<SCHEDULE>, SCALES, __nv_bfloat16, OUTPUT,         \
                                           LinearIdentityEpilogue>(const F16AccOperands<SCALES, __nv_bfloat16>&, \
                                                                   OUTPUT, LinearIdentityEpilogue, cudaStream_t);
INFERNIX_F16ACC_SPLIT_LAUNCHES(INFERNIX_F16ACC_SPLIT_DECLARE)
#undef INFERNIX_F16ACC_SPLIT_DECLARE

} // namespace infernix::ops::detail
