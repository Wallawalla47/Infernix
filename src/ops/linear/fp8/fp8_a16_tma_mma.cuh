#pragma once

// The ring GEMM (fp8_a16_ring_mma.cuh) with a TMA producer: one producer thread loads each stage's
// BF16 activation halves and E4M3 weight tile with TMA (128-byte swizzle, the layout the ring's
// consumers read), and the consumer warps, the arithmetic and so every output bit are the ring's. A
// TMA refill is an async-proxy write, so each consumer thread orders its generic-proxy reads of a stage
// before the warp releases it (fence_reads_before_async_refill, ops/common/mbarrier.cuh; without it a
// refill overwrote stages still being read while other kernels shared the SMs,
// tests/ops/linear/test_fp8_moe_overlap.cu). A shape's selector takes it at the widths where it
// measures faster than the cp.async producer (bench/ops/fp8_flash_next_sweep.cu).

#include "ops/common/mbarrier.cuh"
#include "ops/linear/fp8/fp8_a16_ring_mma.cuh"

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

template <class Schedule, bool FullTokens, class Output, class Epilogue, class Scale = __nv_bfloat16,
          bool InPlace = false>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_tma_mma_kernel(
#ifdef _WIN32
    const __grid_constant__ TmaDescriptorParameter<Fp8TmaDescriptors, InPlace> descriptors_pointer,
#else
    const __grid_constant__ Fp8TmaDescriptors descriptors,
#endif
    Fp8A16OperandsT<Scale> operands, Output output, Epilogue epilogue, int token_offset, int count) {
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
            acquire_staged_tensor_map<InPlace>(&descriptors.activation);
            acquire_staged_tensor_map<InPlace>(&descriptors.weight);
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
        fp8_a16_ring_compute_stage<Schedule>(
            activation + stage * Schedule::kActivationStageBytes,
            reinterpret_cast<const std::uint8_t*>(weight + stage * Schedule::kWeightStageBytes),
            accumulators, warp, lane);
        // Every lane's reads of the stage come before the TMA refill the release admits.
        fence_reads_before_async_refill();
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    // All consumers must finish reading staged inputs before the epilogue reuses the storage.
    __syncthreads();
    fp8_finish_mma_tile<Schedule, FullTokens>(
        output, epilogue, Fp8IdentityRows{}, fp8_a16_tma_shared, accumulators, nullptr,
        operands.scales, row_begin, token_begin, operands.rows, token_offset + count, warp, lane);
}

template <class Schedule, class Output, class Epilogue, class Scale = __nv_bfloat16>
void launch_fp8_a16_tma_mma(const Fp8A16OperandsT<Scale>& p, Output output, Epilogue epilogue,
                            cudaStream_t stream) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("FP8 A16 TMA requires complete row/K tiles");
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const Fp8TmaDescriptors descriptors{
        fp8_a16_tma_activation_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, 64),
        fp8_tma_map(p.codes, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK)};
    const auto launch_slices = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
        for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
            const auto launch = [&]<bool Full>() {
                constexpr auto kernel =
                    fp8_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue, Scale, InPlace>;
                constexpr int bytes =
                    fp8_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
                const int dynamic = fp8_prepare_shared<bytes, kernel, true>();
                kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(parameter, p, output, epilogue,
                                                                        offset, count);
                CUDA_CHECK(cudaGetLastError());
            };
            if (count % Schedule::kBlockTokens == 0)
                launch.template operator()<true>();
            else
                launch.template operator()<false>();
        });
    };
#ifdef _WIN32
    with_tma_descriptor_parameter(fp8_tma_descriptor_staging(), descriptors, stream, launch_slices);
#else
    launch_slices(std::false_type{}, descriptors);
#endif
}

} // namespace infernix::ops::detail
