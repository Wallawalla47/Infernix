#pragma once

// Wide-T row-scaled E4M3 weight x BF16 activation Tensor Core GEMM with TMA staging.
//
//   out[N,T] = (scale[N] * e4m3_codes[N,K]) * x[K,T]
//
// One producer warp streams BF16 activation and E4M3 weight tiles through an mbarrier ring, so the
// weight stream keeps several K tiles in flight per CTA. Consumer warps use the activations as the
// MMA A operand and widen each code pair exactly to a BF16 B fragment in registers. Tensor Cores
// accumulate the complete K reduction in FP32; the represented BF16 row scale is applied once
// before the epilogue. The public activation is never quantized.

#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_a8_tma_mma.cuh"

namespace infernix::ops::detail {

inline CUtensorMap fp8_a16_tma_activation_map(const __nv_bfloat16* pointer, int tokens, int k,
                                              int block_tokens, int block_k) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(k),
                                     static_cast<std::uint64_t>(tokens)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(k) * sizeof(__nv_bfloat16)};
    const std::uint32_t box[]{static_cast<std::uint32_t>(block_k),
                              static_cast<std::uint32_t>(block_tokens)};
    const std::uint32_t steps[]{1, 1};
    // A 64-value BF16 row is one 128-byte swizzle span; rows past T are zero-filled.
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<__nv_bfloat16*>(pointer),
        dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("FP8 A16 TMA activation descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

template <class Schedule>
__device__ __forceinline__ void
fp8_a16_tma_compute_stage(const unsigned char* activation_stage, const std::uint8_t* weight_stage,
                          float (&accumulators)[Schedule::kMmaTokens][Schedule::kMmaRows][4],
                          int warp, int lane) {
    constexpr int kRowBytes = Schedule::kActivationRowBytes;
    static_assert(kRowBytes == 128, "the activation swizzle assumes one 128-byte span per row");
    constexpr int kStepsPerHalf = 64 / 16;
    const int warp_token        = warp / Schedule::kWarpsRows;
    const int warp_row          = warp % Schedule::kWarpsRows;
    const int a_matrix          = lane >> 3;
    const int a_row_offset      = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_byte            = (a_matrix >> 1) * 16;
    const int b_row_offset      = lane >> 2;
    const int b_byte            = 2 * (lane & 3);
#pragma unroll
    for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
        unsigned a[Schedule::kMmaTokens][4];
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
            const int row     = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
            const int logical = (k_step % kStepsPerHalf) * 32 + a_byte;
            ldmatrix_x4(a[mma_token][0], a[mma_token][1], a[mma_token][2], a[mma_token][3],
                        smem_addr(activation_stage +
                                  (k_step / kStepsPerHalf) * Schedule::kActivationHalfBytes +
                                  row * kRowBytes + (logical ^ ((row & 7) << 4))));
        }
        unsigned b[Schedule::kMmaRows][2];
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int row        = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
            const int low        = k_step * 16 + b_byte;
            const auto code_pair = [&](int logical) {
                return static_cast<unsigned>(*reinterpret_cast<const std::uint16_t*>(
                    weight_stage + row * Schedule::kBlockK +
                    fp8_mma_shared_byte<Schedule>(row, logical)));
            };
            b[mma_row][0] = fp8_e4m3x2_to_bf16x2_bits(code_pair(low));
            b[mma_row][1] = fp8_e4m3x2_to_bf16x2_bits(code_pair(low + 8));
        }
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                mma_bf16(accumulators[mma_token][mma_row][0], accumulators[mma_token][mma_row][1],
                         accumulators[mma_token][mma_row][2], accumulators[mma_token][mma_row][3],
                         a[mma_token][0], a[mma_token][1], a[mma_token][2], a[mma_token][3],
                         b[mma_row][0], b[mma_row][1]);
            }
        }
    }
}

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_tma_mma_kernel(
#ifdef _WIN32
    const Fp8TmaDescriptors* descriptors_pointer,
#else
    const __grid_constant__ Fp8TmaDescriptors descriptors,
#endif
    Fp8A16Operands operands, Output output, Epilogue epilogue, int token_offset, int count) {
#ifdef _WIN32
    const Fp8TmaDescriptors& descriptors = *descriptors_pointer;
#endif
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    const int k       = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int tiles_k = k / BK;
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(static_cast<int>(blockIdx.x), operands.rows / BR,
                                       div_up(count, BT), row_tile, token_tile);
    const int row_begin   = row_tile * BR;
    const int token_begin = token_offset + token_tile * BT;

    extern __shared__ __align__(1024) unsigned char fp8_a16_tma_shared[];
    auto* activation = fp8_a16_tma_shared;
    auto* weight     = activation + S * Schedule::kActivationStageBytes;
    auto* full       = reinterpret_cast<std::uint64_t*>(fp8_a16_tma_shared +
                                                        fp8_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty      = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef _WIN32
            acquire_staged_tensor_map(&descriptors.activation);
            acquire_staged_tensor_map(&descriptors.weight);
#endif
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, Schedule::kStageBytes);
#pragma unroll
                for (int half = 0; half < Schedule::kActivationHalves; ++half) {
                    fp8_tma_load(activation + stage * Schedule::kActivationStageBytes +
                                     half * Schedule::kActivationHalfBytes,
                                 &descriptors.activation, kt * BK + half * 64, token_begin,
                                 full + stage);
                }
                fp8_tma_load(weight + stage * Schedule::kWeightStageBytes, &descriptors.weight,
                             kt * BK, row_begin, full + stage);
            }
        }
        return;
    }

    const int tid  = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp = tid >> 5, lane = tid & 31;
    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        fp8_a16_tma_compute_stage<Schedule>(
            activation + stage * Schedule::kActivationStageBytes,
            reinterpret_cast<const std::uint8_t*>(weight + stage * Schedule::kWeightStageBytes),
            accumulators, warp, lane);
        // Release only after every lane in this consumer warp has finished its shared reads.
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    // All consumers must finish reading staged inputs before the epilogue reuses the storage.
    __syncthreads();
    fp8_finish_mma_tile<Schedule, FullTokens>(
        output, epilogue, Fp8IdentityRows{}, fp8_a16_tma_shared, accumulators, nullptr,
        operands.scales, row_begin, token_begin, operands.rows, token_offset + count, warp, lane);
}

template <class Schedule, class Output, class Epilogue>
void launch_fp8_a16_tma_mma(const Fp8A16Operands& p, Output output, Epilogue epilogue,
                            cudaStream_t stream) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("FP8 A16 TMA requires complete row/K tiles");
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const Fp8TmaDescriptors descriptors{
        fp8_a16_tma_activation_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, 64),
        fp8_tma_map(p.codes, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK)};
#ifdef _WIN32
    const Fp8TmaDescriptors* staged = fp8_tma_descriptor_staging().stage(descriptors, stream);
#endif
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = fp8_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue>;
            constexpr int bytes =
                fp8_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            const int dynamic = fp8_prepare_shared<bytes, kernel, true>();
#ifdef _WIN32
            kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(staged, p, output, epilogue,
                                                                    offset, count);
#else
            kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(descriptors, p, output,
                                                                    epilogue, offset, count);
#endif
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

} // namespace infernix::ops::detail
