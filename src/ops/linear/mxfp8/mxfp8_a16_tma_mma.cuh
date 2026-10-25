#pragma once

// MXFP8 (fp8_e4m3fn_g32_e8m0) weight x BF16 activation Tensor Core GEMM: the one route of every
// width.
//
//   out[N,T] = (e4m3_codes[N,K] * 2^(e[N,K/32] - 127)) * x[K,T]
//
// The FP8 A16 TMA GEMM's pipeline (fp8_a16_tma_mma.cuh: producer warps, an mbarrier ring, consumer
// warps widening codes to BF16 B fragments) with a third stream: each stage's UE8M0 scales. A
// consumer multiplies each widened code pair (one row, one 32-element group) by its group's power of
// two in BF16, which is exact for every admitted scale (mxfp8_codec.cuh), so the Tensor Cores see the
// represented weights themselves.
//
// Two producers fill the same 128-byte-swizzled stages for the same consumer, so a schedule's bits do
// not depend on its producer: cp.async 16-byte copies spread over one to four producer warps
// (Mxfp8ProducerWarps), faster at decode widths, and one thread's TMA tensor loads
// (Mxfp8TmaProducer), faster from about 64-256 columns. Each shape's selector picks both per width.
//
// Column invariance: each output accumulates its complete K reduction in one FP32 chain of
// m16n8k16 MMAs in ascending K (activations as A, weights as B, BlockK 128), whatever the tile, the
// stage count or the number of columns in the call. Every MXFP8 shape therefore gives a column the
// same bits at every T, which Qwen3.8-Flash-Next's exact route needs (prefill calls, taps and resumes
// on the 16-token grid, any decode batch). No row multiplier remains for the epilogue.
//
// Programmatic launch: the producer loads the first stages' weights and scales (immutable) before
// the grid dependency resolves and their activations after it; consumers release dependents once
// their main loop is done.
//
// A TMA refill is an async-proxy write into a stage the consumers have read through the generic
// proxy, so each consumer thread orders its reads before its warp releases the stage
// (fence_reads_before_async_refill, ops/common/mbarrier.cuh). Without that fence TMA-staged outputs came out wrong in up to 37 %
// of the calls while offloaded_sparse_moe's routed experts ran beside the kernel
// (tests/ops/linear/test_mxfp8_overlap.cu, 2026-10-20).

#include "core/device.h"
#include "core/pdl.cuh"
#include "core/tma_descriptor_staging.cuh"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_tma_mma.cuh"
#include "ops/linear/fp8/fp8_shared.cuh"
#include "ops/linear/mxfp8/mxfp8_codec.cuh"
#include "ops/linear/mxfp8/mxfp8_format.h"

#include <cuda.h>

#include <stdexcept>
#include <string>

namespace infernix::ops::detail {

struct alignas(128) Mxfp8TmaDescriptors {
    CUtensorMap activation;
    CUtensorMap weight;
    CUtensorMap scale;
};

#ifdef _WIN32
// The Windows descriptor staging of every MXFP8 TMA launch (core/tma_descriptor_staging.cuh).
inline TmaDescriptorStaging<Mxfp8TmaDescriptors>& mxfp8_tma_descriptor_staging() {
    static TmaDescriptorStaging<Mxfp8TmaDescriptors> staging;
    return staging;
}
#endif

// The scale plane [N/16][K/32][16] as a 3-D byte tensor; one box is a stage's BlockRows/16 tiles of
// BlockK/32 groups.
inline CUtensorMap mxfp8_scale_tma_map(const std::uint8_t* pointer, int rows, int k, int block_rows,
                                       int block_k) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{16, static_cast<std::uint64_t>(k / 32), static_cast<std::uint64_t>(rows / 16)};
    const std::uint64_t strides[]{16, static_cast<std::uint64_t>(k / 32) * 16};
    const std::uint32_t box[]{16, static_cast<std::uint32_t>(block_k / 32), static_cast<std::uint32_t>(block_rows / 16)};
    const std::uint32_t steps[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<std::uint8_t*>(pointer), dimensions, strides, box, steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("MXFP8 TMA scale descriptor: ") + (name ? name : "CUDA error"));
    }
    return result;
}

__device__ __forceinline__ void mxfp8_tma_load_scales(void* destination, const CUtensorMap* map, int group, int tile,
                                                      std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(0), "r"(group), "r"(tile), "r"(smem_addr(barrier))
                 : "memory");
}

// Shared storage of the MXFP8 TMA GEMM: the FP8 schedule's activation and weight stages, then one
// scale stage per pipeline stage, then the barriers. A TMA destination must be 128-byte aligned, so
// each scale stage takes a whole number of 128-byte lines (a 16-row tile's stage holds only 64 bytes).
template <class Schedule>
struct Mxfp8TmaLayout {
    static constexpr int kScaleStageBytes  = Schedule::kBlockRows * Schedule::kBlockK / 32;
    static constexpr int kScaleStageStride = (kScaleStageBytes + 127) / 128 * 128;
    static constexpr int kScaleOffset      = Schedule::kStages * Schedule::kStageBytes;
    static constexpr int kPipelineBytes    = kScaleOffset + Schedule::kStages * kScaleStageStride;
    static constexpr int kStorageBytes =
        ((kPipelineBytes > Schedule::kOutputStagingBytes ? kPipelineBytes : Schedule::kOutputStagingBytes) + 127) /
        128 * 128;
    static constexpr int kSharedBytes = kStorageBytes + Schedule::kBarrierBytes;
    static_assert(kScaleStageBytes % 16 == 0 && kScaleOffset % 128 == 0);
    static_assert(kSharedBytes <= 99 * 1024, "the MXFP8 TMA schedule's stages exceed shared memory");
};

// Base with a cp.async producer of `ProducerWarps` warps: a stage's copies spread over them. The
// consumer warps, their tiles and so every bit are Base's. A plain schedule's producer is one
// cp.async warp.
template <class Base, int ProducerWarps>
struct Mxfp8ProducerWarps : Base {
    static constexpr int kProducerThreads = ProducerWarps * 32;
    static constexpr int kThreads         = kProducerThreads + Base::kWarps * 32;
};

// Base staged by the TMA producer: one thread of the producer warp issues the stages' tensor loads.
// The consumer warps, their tiles and so every bit are Base's.
template <class Base>
struct Mxfp8TmaProducer : Base {
    static_assert(Base::kProducerThreads == 32, "the TMA producer is one warp");
    static constexpr bool kTmaProducer = true;
};

template <class Schedule>
inline constexpr bool kMxfp8TmaProducer = requires { Schedule::kTmaProducer; };

template <class Schedule>
__device__ __forceinline__ void
mxfp8_tma_compute_stage(const unsigned char* activation_stage, const std::uint8_t* weight_stage,
                        const std::uint8_t* scale_stage,
                        float (&accumulators)[Schedule::kMmaTokens][Schedule::kMmaRows][4], int warp, int lane) {
    constexpr int kRowBytes = Schedule::kActivationRowBytes;
    static_assert(kRowBytes == 128, "the activation swizzle assumes one 128-byte span per row");
    constexpr int kStepsPerHalf = 64 / 16;
    constexpr int kGroupsPerStage = Schedule::kBlockK / 32;
    const int warp_token = warp / Schedule::kWarpsRows;
    const int warp_row   = warp % Schedule::kWarpsRows;
    const int a_matrix   = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_byte       = (a_matrix >> 1) * 16;
    const int b_row_offset = lane >> 2;
    const int b_byte       = 2 * (lane & 3);
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
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int row        = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
            const int low        = k_step * 16 + b_byte;
            const auto code_pair = [&](int logical) {
                return static_cast<unsigned>(*reinterpret_cast<const std::uint16_t*>(
                    weight_stage + row * Schedule::kBlockK + fp8_mma_shared_byte<Schedule>(row, logical)));
            };
            // Both halves of the fragment (K low and low + 8) lie in group k_step / 2 of this stage.
            const unsigned scale = mxfp8_scale_bf16x2(
                scale_stage[(row >> 4) * kGroupsPerStage * 16 + (k_step >> 1) * 16 + (row & 15)]);
            b[mma_row][0] = mxfp8_scaled_pair(fp8_e4m3x2_to_bf16x2_bits(code_pair(low)), scale);
            b[mma_row][1] = mxfp8_scaled_pair(fp8_e4m3x2_to_bf16x2_bits(code_pair(low + 8)), scale);
        }
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                mma_bf16(accumulators[mma_token][mma_row][0], accumulators[mma_token][mma_row][1],
                         accumulators[mma_token][mma_row][2], accumulators[mma_token][mma_row][3], a[mma_token][0],
                         a[mma_token][1], a[mma_token][2], a[mma_token][3], b[mma_row][0], b[mma_row][1]);
            }
        }
    }
}

// The cp.async producer fills each stage with 16-byte copies in the layout the TMA producer's
// 128-byte-swizzled loads write, and each producer thread's copies arrive on the stage's full
// barrier.
__device__ __forceinline__ void mxfp8_cp_async_16(void* destination, const void* source, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(destination)), "l"(source),
                 "r"(valid ? 16 : 0)
                 : "memory");
}

__device__ __forceinline__ void mxfp8_cp_async_arrive(std::uint64_t* barrier) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_addr(barrier)) : "memory");
}

template <class Schedule, bool FullTokens, class Output, class Epilogue, bool InPlace = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void mxfp8_a16_tma_mma_kernel(
#ifdef _WIN32
    const __grid_constant__ TmaDescriptorParameter<Mxfp8TmaDescriptors, InPlace> descriptors_pointer,
#else
    const __grid_constant__ Mxfp8TmaDescriptors descriptors,
#endif
    Mxfp8Operands operands, Output output, Epilogue epilogue, int token_offset, int count) {
#ifdef _WIN32
    const Mxfp8TmaDescriptors& descriptors = *descriptors_pointer;
#endif
    using Layout = Mxfp8TmaLayout<Schedule>;
    constexpr bool kTma = kMxfp8TmaProducer<Schedule>;
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    const int k       = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int tiles_k = k / BK;
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(static_cast<int>(blockIdx.x), operands.rows / BR, div_up(count, BT), row_tile,
                                       token_tile);
    const int row_begin   = row_tile * BR;
    const int token_begin = token_offset + token_tile * BT;

    extern __shared__ __align__(1024) unsigned char mxfp8_tma_shared[];
    auto* activation = mxfp8_tma_shared;
    auto* weight     = activation + S * Schedule::kActivationStageBytes;
    auto* scales     = mxfp8_tma_shared + Layout::kScaleOffset;
    auto* full       = reinterpret_cast<std::uint64_t*>(mxfp8_tma_shared + Layout::kStorageBytes);
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

    if constexpr (!kTma) {
        if (threadIdx.x < Schedule::kProducerThreads) {
            const int lane = static_cast<int>(threadIdx.x);
            constexpr int kGroups = BK / 32;
            const auto copy_weights = [&](int kt, int stage) {
                std::uint8_t* w = reinterpret_cast<std::uint8_t*>(weight) + stage * Schedule::kWeightStageBytes;
                const std::uint8_t* codes = operands.codes + static_cast<std::size_t>(row_begin) * k + kt * BK;
                for (int i = lane; i < BR * 8; i += Schedule::kProducerThreads) {
                    const int r = i >> 3, c = i & 7;
                    mxfp8_cp_async_16(w + r * 128 + ((c ^ (r & 7)) << 4), codes + static_cast<std::size_t>(r) * k + c * 16,
                                      true);
                }
                std::uint8_t* sc = scales + stage * Layout::kScaleStageStride;
                for (int i = lane; i < (BR / 16) * kGroups; i += Schedule::kProducerThreads) {
                    const int t = i / kGroups, g = i % kGroups;
                    mxfp8_cp_async_16(sc + (t * kGroups + g) * 16,
                                      operands.scales +
                                          (static_cast<std::size_t>(row_begin / 16 + t) * (k / 32) + kt * kGroups + g) * 16,
                                      true);
                }
            };
            // Only the tile's rows that hold tokens: a row past T feeds only its own output row, which
            // the epilogue never stores, so it is not filled (TMA zero-filled it).
            const int rows = FullTokens ? BT : min(BT, operands.tokens - token_begin); // a full tile unrolls
            const auto copy_activations = [&](int kt, int stage) {
#pragma unroll
                for (int half = 0; half < Schedule::kActivationHalves; ++half) {
                    std::uint8_t* a = reinterpret_cast<std::uint8_t*>(activation) + stage * Schedule::kActivationStageBytes +
                                      half * Schedule::kActivationHalfBytes;
                    for (int i = lane; i < rows * 8; i += Schedule::kProducerThreads) {
                        const int r = i >> 3, c = i & 7;
                        mxfp8_cp_async_16(a + r * 128 + ((c ^ (r & 7)) << 4),
                                          operands.x + static_cast<std::size_t>(token_begin + r) * k + kt * BK +
                                              half * 64 + c * 8,
                                          true);
                    }
                }
            };
            const int prologue = tiles_k < S ? tiles_k : S;
            for (int kt = 0; kt < prologue; ++kt) { copy_weights(kt, kt); }
            pdl::wait_for_dependencies();
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                if (kt >= prologue) {
                    cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                    copy_weights(kt, stage);
                }
                copy_activations(kt, stage);
                mxfp8_cp_async_arrive(full + stage);
            }
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            return;
        }
    } else if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef _WIN32
            acquire_staged_tensor_map<InPlace>(&descriptors.activation);
            acquire_staged_tensor_map<InPlace>(&descriptors.weight);
            acquire_staged_tensor_map<InPlace>(&descriptors.scale);
#endif
            const auto load_weights = [&](int kt, int stage) {
                fp8_tma_load(weight + stage * Schedule::kWeightStageBytes, &descriptors.weight, kt * BK, row_begin,
                             full + stage);
                mxfp8_tma_load_scales(scales + stage * Layout::kScaleStageStride, &descriptors.scale, kt * (BK / 32),
                                      row_begin / 16, full + stage);
            };
            // The ring starts empty: the first stages' weights and scales load before the dependency
            // resolves (each barrier expects its stage's activation bytes too).
            const int prologue = tiles_k < S ? tiles_k : S;
            for (int kt = 0; kt < prologue; ++kt) {
                cta_mbarrier_arrive_expect_tx(full + kt, Schedule::kStageBytes + Layout::kScaleStageBytes);
                load_weights(kt, kt);
            }
            pdl::wait_for_dependencies();
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                if (kt >= prologue) {
                    cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                    cta_mbarrier_arrive_expect_tx(full + stage, Schedule::kStageBytes + Layout::kScaleStageBytes);
                    load_weights(kt, stage);
                }
#pragma unroll
                for (int half = 0; half < Schedule::kActivationHalves; ++half) {
                    fp8_tma_load(activation + stage * Schedule::kActivationStageBytes +
                                     half * Schedule::kActivationHalfBytes,
                                 &descriptors.activation, kt * BK + half * 64, token_begin, full + stage);
                }
            }
        } else {
            pdl::wait_for_dependencies();
        }
        return;
    }

    pdl::wait_for_dependencies();
    const int tid  = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp = tid >> 5, lane = tid & 31;
    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        mxfp8_tma_compute_stage<Schedule>(
            activation + stage * Schedule::kActivationStageBytes,
            reinterpret_cast<const std::uint8_t*>(weight + stage * Schedule::kWeightStageBytes),
            reinterpret_cast<const std::uint8_t*>(scales + stage * Layout::kScaleStageStride), accumulators, warp,
            lane);
        // Release only after every lane in this consumer warp has finished its shared reads; a TMA
        // refill is an async-proxy write, so each thread first orders its generic-proxy reads before
        // it.
        if constexpr (kTma) { fence_reads_before_async_refill(); }
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    pdl::trigger_dependents();
    // All consumers must finish reading staged inputs before the epilogue reuses the storage.
    __syncthreads();
    // The represented weights carry their scales: the epilogue multiplies by nothing.
    fp8_finish_mma_tile<Schedule, FullTokens>(output, epilogue, Fp8UnscaledRows{}, mxfp8_tma_shared, accumulators,
                                              nullptr, static_cast<const __nv_bfloat16*>(nullptr), row_begin, token_begin, operands.rows,
                                              token_offset + count, warp, lane);
}

// Decode and verification calls (at most kMxfp8ProgrammaticTokens columns) launch as programmatic
// dependents, so their weight streams start under the kernel before; wider calls launch serialized,
// so nothing launched beside them takes their SMs. The cp.async producer reads only the weights and
// scales before its PDL wait. The TMA producer also reads its tensor maps there, so it is
// programmatic only where they cannot change under it: by value (Linux, or Windows parameter words read
// in place, INFERNIX_TMP_TMA_PARAM_WORDS) or, on Windows, a captured
// launch's own staged copy (an eager Windows launch reads the one staging buffer the staging kernel
// has just stored into, core/tma_descriptor_staging.cuh).
inline constexpr int kMxfp8ProgrammaticTokens = 16;

template <class Schedule, class Output, class Epilogue>
void launch_mxfp8_a16_tma_mma(const Mxfp8Operands& p, Output output, Epilogue epilogue, cudaStream_t stream) {
    validate_mxfp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK || Schedule::kBlockRows % 16)
        throw std::invalid_argument("MXFP8 A16 TMA requires complete row/K tiles");
    if constexpr (!kMxfp8TmaProducer<Schedule>) {
        // No tensor maps: every decode-width launch may be programmatic.
        const pdl::Dependency dependency =
            p.tokens <= kMxfp8ProgrammaticTokens ? pdl::Dependency::Programmatic : pdl::Dependency::Serialized;
        for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
            const auto launch = [&]<bool Full>() {
                constexpr auto kernel = mxfp8_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue>;
                const int dynamic     = fp8_prepare_shared<Mxfp8TmaLayout<Schedule>::kSharedBytes, kernel, true>();
                const pdl::LaunchConfig config{dim3(blocks), dim3(Schedule::kThreads),
                                               static_cast<std::size_t>(dynamic), stream};
#ifdef _WIN32
                CUDA_CHECK(pdl::launch_with(dependency, config, kernel, static_cast<const Mxfp8TmaDescriptors*>(nullptr),
                                            p, output, epilogue, offset, count));
#else
                CUDA_CHECK(pdl::launch_with(dependency, config, kernel, Mxfp8TmaDescriptors{}, p, output, epilogue,
                                            offset, count));
#endif
            };
            count % Schedule::kBlockTokens == 0 ? launch.template operator()<true>()
                                                : launch.template operator()<false>();
        });
    } else {
#ifdef _WIN32
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        const bool captured = capture == cudaStreamCaptureStatusActive;
#else
        constexpr bool captured = true;
#endif
        const Mxfp8TmaDescriptors descriptors{
            fp8_a16_tma_activation_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, 64),
            fp8_tma_map(p.codes, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK),
            mxfp8_scale_tma_map(p.scales, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK)};
        const auto launch_slices = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
            // Maps read in place are by value, as on Linux.
            const bool fixed_maps            = InPlace || captured;
            const pdl::Dependency dependency = fixed_maps && p.tokens <= kMxfp8ProgrammaticTokens
                                                   ? pdl::Dependency::Programmatic
                                                   : pdl::Dependency::Serialized;
            for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
                const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
                const auto launch = [&]<bool Full>() {
                    constexpr auto kernel = mxfp8_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue, InPlace>;
                    const int dynamic = fp8_prepare_shared<Mxfp8TmaLayout<Schedule>::kSharedBytes, kernel, true>();
                    const pdl::LaunchConfig config{dim3(blocks), dim3(Schedule::kThreads),
                                                   static_cast<std::size_t>(dynamic), stream};
                    CUDA_CHECK(
                        pdl::launch_with(dependency, config, kernel, parameter, p, output, epilogue, offset, count));
                };
                count % Schedule::kBlockTokens == 0 ? launch.template operator()<true>()
                                                    : launch.template operator()<false>();
            });
        };
#ifdef _WIN32
        with_tma_descriptor_parameter(mxfp8_tma_descriptor_staging(), descriptors, stream, launch_slices);
#else
        launch_slices(std::false_type{}, descriptors);
#endif
    }
}

} // namespace infernix::ops::detail
