#pragma once

// Wide-T row-scaled E4M3 weight x BF16 activation Tensor Core GEMM with an asynchronous-copy staging ring.
//
//   out[N,T] = (scale[N] * e4m3_codes[N,K]) * x[K,T]
//
// One producer warp streams BF16 activation and E4M3 weight tiles through an mbarrier ring, so the
// weight stream keeps several K tiles in flight per CTA. Consumer warps use the activations as the
// MMA A operand and widen each code pair exactly to a BF16 B fragment in registers. Tensor Cores
// accumulate the complete K reduction in FP32; the represented BF16 row scale is applied once
// before the epilogue. The public activation is never quantized.
//
// The producer's 32 threads stage each tile with 16-byte cp.async copies into the layout a TMA
// load with the 128-byte (64-byte) swizzle gives (activation rows past the slice's tokens are not
// loaded: they feed only unstored output rows), and each thread's copies arrive on the stage's full
// barrier. Until 2026-10-20 one thread staged the tiles with TMA, and its consumers released a stage
// without ordering their generic-proxy reads (ldmatrix, ld.shared) before the producer's
// async-proxy refill (fence_reads_before_async_refill, ops/common/mbarrier.cuh): while
// offloaded_sparse_moe's routed experts shared the SMs (Qwen3.8-Flash-Next recipe C's shared expert
// on its own stream), a refill overwrote stages still being read
// (tests/ops/linear/test_fp8_moe_overlap.cu). cp.async copies are generic-proxy writes, which the
// stage's barriers order. The consumer, and so every output bit, is unchanged; the fenced TMA
// producer measured level with this one on recipe C (decode and an 8,192-token prefill, within run-to-
// run spread), so the simpler producer stays.

#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_a8_tma_mma.cuh"

namespace infernix::ops::detail {

// Arrives on `barrier` once every cp.async this thread issued earlier has landed; the arrival is one
// of the count the barrier was initialized with (.noinc).
__device__ __forceinline__ void fp8_cp_async_arrive(std::uint64_t* barrier) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" : : "r"(smem_addr(barrier)) : "memory");
}

template <class Schedule>
__device__ __forceinline__ void
fp8_a16_ring_compute_stage(const unsigned char* activation_stage, const std::uint8_t* weight_stage,
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

template <class Schedule, bool FullTokens, class Output, class Epilogue, class Scale = __nv_bfloat16>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_ring_mma_kernel(
    Fp8A16OperandsT<Scale> operands, Output output, Epilogue epilogue, int token_offset, int count) {
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    constexpr int kWeightChunks = BK / 16; // 16-byte chunks per weight row
    const int k       = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int tiles_k = k / BK;
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(static_cast<int>(blockIdx.x), operands.rows / BR,
                                       div_up(count, BT), row_tile, token_tile);
    const int row_begin   = row_tile * BR;
    const int token_begin = token_offset + token_tile * BT;

    extern __shared__ __align__(1024) unsigned char fp8_a16_ring_shared[];
    auto* activation = fp8_a16_ring_shared;
    auto* weight     = activation + S * Schedule::kActivationStageBytes;
    auto* full       = reinterpret_cast<std::uint64_t*>(fp8_a16_ring_shared +
                                                        fp8_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty      = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, Schedule::kProducerThreads);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (threadIdx.x < Schedule::kProducerThreads) {
        const int lane       = static_cast<int>(threadIdx.x);
        const int valid_rows = min(BT, token_offset + count - token_begin);
        for (int kt = 0; kt < tiles_k; ++kt) {
            const int stage = kt % S;
            cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
            // E4M3 rows: each 16-byte chunk where the TMA swizzle put it (the consumer's mapping).
            unsigned char* w = weight + stage * Schedule::kWeightStageBytes;
            const std::uint8_t* codes =
                operands.codes + static_cast<std::size_t>(row_begin) * k + kt * BK;
            for (int i = lane; i < BR * kWeightChunks; i += Schedule::kProducerThreads) {
                const int r = i / kWeightChunks, c = i % kWeightChunks;
                cp_async<16, Cache::cg>(w + r * BK + fp8_mma_shared_byte<Schedule>(r, c * 16),
                                        codes + static_cast<std::size_t>(r) * k + c * 16);
            }
            // BF16 token rows: each 64-value half a 128-byte span per row under the 128-byte
            // swizzle. Only the slice's tokens: a row past them feeds only its own output row,
            // which the epilogue never stores, so it is left as it is.
#pragma unroll
            for (int half = 0; half < Schedule::kActivationHalves; ++half) {
                unsigned char* a = activation + stage * Schedule::kActivationStageBytes +
                                   half * Schedule::kActivationHalfBytes;
                for (int i = lane; i < valid_rows * 8; i += Schedule::kProducerThreads) {
                    const int r = i >> 3, c = i & 7;
                    cp_async<16, Cache::cg>(a + r * 128 + ((c ^ (r & 7)) << 4),
                                            operands.x + static_cast<std::size_t>(token_begin + r) * k +
                                                kt * BK + half * 64 + c * 8);
                }
            }
            fp8_cp_async_arrive(full + stage);
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
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
        // Release only after every lane in this consumer warp has finished its shared reads.
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    // All consumers must finish reading staged inputs before the epilogue reuses the storage.
    __syncthreads();
    fp8_finish_mma_tile<Schedule, FullTokens>(
        output, epilogue, Fp8IdentityRows{}, fp8_a16_ring_shared, accumulators, nullptr,
        operands.scales, row_begin, token_begin, operands.rows, token_offset + count, warp, lane);
}

template <class Schedule, class Output, class Epilogue, class Scale = __nv_bfloat16>
void launch_fp8_a16_ring_mma(const Fp8A16OperandsT<Scale>& p, Output output, Epilogue epilogue,
                            cudaStream_t stream) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("FP8 A16 TMA requires complete row/K tiles");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = fp8_a16_ring_mma_kernel<Schedule, Full, Output, Epilogue, Scale>;
            constexpr int bytes =
                fp8_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            const int dynamic = fp8_prepare_shared<bytes, kernel, true>();
            kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(p, output, epilogue, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

} // namespace infernix::ops::detail
