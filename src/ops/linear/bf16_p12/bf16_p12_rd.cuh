#pragma once

// The register-decode TMA route of bf16_p12 calls of mid widths (a few to a few hundred columns), where a BF16 GEMM is
// bound by its weight bytes and the packed weights' 25 % fewer bytes can make it faster. A producer warp loads, per
// 64-k step, the packed code and nibble tiles of the CTA's rows and the activation tile of its tokens by TMA into a
// ring of stages (mbarriers, no CTA-wide barrier in the K loop); each consumer warp decodes its rows' chunks straight
// into the A fragments of m16n8k16 MMAs (the weights never pass through shared memory decoded) and runs them against
// every token tile of the CTA, so a weight is decoded once per CTA. The stream MMA (bf16_p12_stream.cuh) stages the
// activations with cp.async and a CTA barrier per step; this route's warps only wait for their stage to land.
//
// Fragments. Lane (g = lane / 4, t = lane % 4) reads, per step, the 16 code bytes and two nibble words of chunks 2t and
// 2t + 1 of rows g and g + 8 of each of its m16 row groups. In sub-step j it decodes chunk 2t + j of both rows and
// feeds its eight weights to two MMAs: weights 0-3 where the A fragment has k 2t, 2t + 1, 2t + 8 and 2t + 9, then
// weights 4-7. The B fragments are the same real k of token g: one 16-byte load from the activation tile (TMA's
// 128-byte swizzle: unit u of token r at r * 128 + 16 (u ^ (r & 7)), conflict-free for these loads).
//
// Arithmetic. An output (row, column) is the FP32 MMA accumulation over the steps in order (sub-steps j = 0, 1, each
// its two MMAs in order): it depends on the row's weights and the column's activations only. Every schedule of this
// route gives the same bits, at every width (bits differ from the BF16 routes and the other packed routes).

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"
#include "ops/linear/bf16_p12/bf16_p12_tma.cuh"
#include "ops/linear/common/output.cuh"

#include <cuda.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace infernix::ops::detail {

// RowWarps: consumer warps, each 16 * RowGroups rows of the CTA's row tile and every token of its token tile; Tokens:
// the token tile (a multiple of 8, at most 128); Stages: 64-k steps in the ring; MinBlocks: CTAs an SM.
template <int RowWarps, int RowGroups, int Tokens, int Stages, int MinBlocks = 1>
struct P12RdSchedule {
    static constexpr int kRowWarps        = RowWarps;
    static constexpr int kRowGroups       = RowGroups;
    static constexpr int kTokens          = Tokens;
    static constexpr int kStages          = Stages;
    static constexpr int kMinBlocks       = MinBlocks;
    static constexpr int kStepK           = 64;
    static constexpr int kWarpRows        = 16 * RowGroups;
    static constexpr int kBlockRows       = kWarpRows * RowWarps;
    static constexpr int kTiles           = Tokens / 8;
    static constexpr int kThreads         = 32 * (RowWarps + 1); // the consumers, then the producer warp
    static constexpr int kActivationBytes = Tokens * kStepK * 2;
    static constexpr int kCodeBytes       = kBlockRows * kStepK;
    static constexpr int kNibbleBytes     = kBlockRows * kStepK / 2;
    // A stage: the activation tile (1,024-byte aligned for its 128-byte swizzle), the codes, the nibbles.
    static constexpr int kStageBytes    = (kActivationBytes + kCodeBytes + kNibbleBytes + 1023) / 1024 * 1024;
    static constexpr int kTransferBytes = kActivationBytes + kCodeBytes + kNibbleBytes;
    // The epilogue's [token][row] BF16 tile of each warp reuses the stages once every consumer has left them.
    static constexpr int kEpilogueBytes = RowWarps * kWarpRows * Tokens * 2;
    static constexpr int kBarrierOffset =
        (Stages * kStageBytes > kEpilogueBytes ? Stages * kStageBytes : kEpilogueBytes + 7) / 8 * 8;
    static constexpr int kSharedBytes = kBarrierOffset + 2 * Stages * 8;
    static_assert(RowWarps >= 1 && RowGroups >= 1 && Tokens % 8 == 0 && Tokens >= 8 && Tokens <= 128);
    static_assert(kBlockRows <= 256 && Stages >= 2 && kThreads <= 1024 && kSharedBytes <= 99 * 1024);
};

// The activations x [tokens, k] (BF16, k contiguous) as TMA reads them: 64-k by `box_tokens` boxes, 128-byte swizzle.
inline CUtensorMap p12_rd_activation_map(const __nv_bfloat16* x, int tokens, int k, int box_tokens) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(tokens)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(k) * 2};
    const std::uint32_t box[]{64, static_cast<std::uint32_t>(box_tokens)};
    const std::uint32_t steps[]{1, 1};
    const auto status = cuTensorMapEncodeTiled(&result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                                               const_cast<__nv_bfloat16*>(x), dimensions, strides, box, steps,
                                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("bf16_p12 register-decode TMA descriptor: ") + (name ? name : "CUDA error"));
    }
    return result;
}

__device__ __forceinline__ void p12_rd_tma_2d(void* destination, const CUtensorMap* map, int x, int y,
                                              std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(x), "r"(y), "r"(smem_addr(barrier))
                 : "memory");
}

template <class S, class Output, bool InPlace = false>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocks) void bf16_p12_rd_kernel(
#ifdef _WIN32
    const __grid_constant__ TmaDescriptorParameter<P12TmaDescriptors, InPlace> descriptors_pointer,
#else
    const __grid_constant__ P12TmaDescriptors descriptors,
#endif
    const unsigned* __restrict__ words, const unsigned* __restrict__ escapes, Output output, int rows, int K,
    int tokens) {
#ifdef _WIN32
    const P12TmaDescriptors& descriptors = *descriptors_pointer;
#endif
    constexpr int RG = S::kRowGroups, NT = S::kTiles, ST = S::kStages;
    extern __shared__ __align__(1024) unsigned char storage[];
    auto* full  = reinterpret_cast<std::uint64_t*>(storage + S::kBarrierOffset);
    auto* empty = full + ST;
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int token_tiles = (tokens + S::kTokens - 1) / S::kTokens;
    const int tile_r      = static_cast<int>(blockIdx.x) / token_tiles;
    const int row_begin   = tile_r * S::kBlockRows;
    const int token_begin = (static_cast<int>(blockIdx.x) - tile_r * token_tiles) * S::kTokens;
    if (tid == 0) {
#pragma unroll
        for (int stage = 0; stage < ST; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, S::kRowWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();
    const int steps = K / S::kStepK;

    if (warp == S::kRowWarps) {
        if (lane == 0) {
#ifdef _WIN32
            acquire_staged_tensor_map<InPlace>(&descriptors.codes);
            acquire_staged_tensor_map<InPlace>(&descriptors.nibbles);
            acquire_staged_tensor_map<InPlace>(&descriptors.activation);
#endif
            for (int kt = 0; kt < steps; ++kt) {
                const int stage = kt % ST;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / ST) & 1U));
                unsigned char* base = storage + stage * S::kStageBytes;
                cta_mbarrier_arrive_expect_tx(full + stage, S::kTransferBytes);
                p12_rd_tma_2d(base, &descriptors.activation, kt * S::kStepK, token_begin, full + stage);
                p12_rd_tma_2d(base + S::kActivationBytes, &descriptors.codes, kt * S::kStepK, row_begin, full + stage);
                p12_rd_tma_2d(base + S::kActivationBytes + S::kCodeBytes, &descriptors.nibbles, kt * (S::kStepK / 2),
                              row_begin, full + stage);
            }
        }
        return;
    }

    const int g = lane >> 2, t = lane & 3;
    const int warp_row = warp * S::kWarpRows; // within the tile
    P12Row state[RG][2];
#pragma unroll
    for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
        for (int h = 0; h < 2; ++h) { state[mi][h] = p12_row(words, row_begin + warp_row + mi * 16 + g + 8 * h); }
    }
    float acc[RG][NT][4] = {};
#pragma unroll 1
    for (int kt = 0; kt < steps; ++kt) {
        const int stage = kt % ST;
        cta_mbarrier_wait(full + stage, (kt / ST) & 1U);
        const unsigned char* base    = storage + stage * S::kStageBytes;
        const auto* activations      = reinterpret_cast<const uint4*>(base);
        const unsigned char* codes   = base + S::kActivationBytes;
        const unsigned char* nibbles = codes + S::kCodeBytes;
        uint4 code[RG][2];
        uint2 nib[RG][2];
#pragma unroll
        for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r  = warp_row + mi * 16 + g + 8 * h;
                code[mi][h] = *reinterpret_cast<const uint4*>(codes + r * S::kStepK + 16 * t);
                nib[mi][h]  = *reinterpret_cast<const uint2*>(nibbles + r * (S::kStepK / 2) + 8 * t);
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint4 w[RG][2];
#pragma unroll
            for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4& c = code[mi][h];
                    w[mi][h]       = p12_decode_chunk(j == 0 ? make_uint2(c.x, c.y) : make_uint2(c.z, c.w),
                                                      j == 0 ? nib[mi][h].x : nib[mi][h].y, state[mi][h],
                                                      kt * (S::kStepK / 8) + 2 * t + j, escapes);
                }
            }
            const int unit = (2 * t + j) ^ g; // token nt * 8 + g's swizzled 16-byte unit
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint4 xv = activations[(nt * 8 + g) * 8 + unit];
#pragma unroll
                for (int mi = 0; mi < RG; ++mi) {
                    float(&c)[4] = acc[mi][nt];
                    mma_bf16(c[0], c[1], c[2], c[3], w[mi][0].x, w[mi][1].x, w[mi][0].y, w[mi][1].y, xv.x, xv.y);
                    mma_bf16(c[0], c[1], c[2], c[3], w[mi][0].z, w[mi][1].z, w[mi][0].w, w[mi][1].w, xv.z, xv.w);
                }
            }
        }
        // The stage's reads are ordered before its next TMA refill.
        fence_reads_before_async_refill();
        __syncwarp();
        if (lane == 0) { cta_mbarrier_arrive(empty + stage); }
    }

    if constexpr (std::is_same_v<Output, LinearBf16Output>) {
        // Every consumer has left every stage (all were refilled at most `steps` times and consumed): the stages hold
        // each warp's [token][row] BF16 tile, then 16-byte row runs of each token.
        asm volatile("bar.sync 1, %0;" : : "r"(S::kRowWarps * 32) : "memory");
        auto* tile = reinterpret_cast<__nv_bfloat16*>(storage) + warp * (S::kWarpRows * S::kTokens);
#pragma unroll
        for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int r = mi * 16 + g, c = nt * 8 + 2 * t;
                tile[c * S::kWarpRows + r]           = __float2bfloat16_rn(acc[mi][nt][0]);
                tile[(c + 1) * S::kWarpRows + r]     = __float2bfloat16_rn(acc[mi][nt][1]);
                tile[c * S::kWarpRows + r + 8]       = __float2bfloat16_rn(acc[mi][nt][2]);
                tile[(c + 1) * S::kWarpRows + r + 8] = __float2bfloat16_rn(acc[mi][nt][3]);
            }
        }
        __syncwarp();
        constexpr int kRuns = S::kWarpRows / 8; // 16-byte runs a token
#pragma unroll 1
        for (int i = lane; i < S::kTokens * kRuns; i += 32) {
            const int c = i / kRuns, run = i - c * kRuns;
            if (token_begin + c < tokens) {
                output.store_vector(row_begin + warp_row + run * 8, token_begin + c,
                                    *reinterpret_cast<const uint4*>(tile + c * S::kWarpRows + run * 8));
            }
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int row = row_begin + warp_row + mi * 16 + g, token = token_begin + nt * 8 + 2 * t;
                if (token < tokens) {
                    output.store(row, token, acc[mi][nt][0]);
                    output.store(row + 8, token, acc[mi][nt][2]);
                }
                if (token + 1 < tokens) {
                    output.store(row, token + 1, acc[mi][nt][1]);
                    output.store(row + 8, token + 1, acc[mi][nt][3]);
                }
            }
        }
    }
}

// Whether `S` takes a [rows, k] packed weight: whole row tiles and 64-k steps.
template <class S>
[[nodiscard]] constexpr bool p12_rd_fits(std::int32_t rows, std::int32_t k) noexcept {
    return rows % S::kBlockRows == 0 && k % S::kStepK == 0 && k / 2 % 16 == 0;
}

template <class S>
void launch_bf16_p12_rd(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (!p12_rd_fits<S>(w.n, w.k)) {
        throw std::invalid_argument("bf16_p12 register-decode TMA: rows must be whole row tiles and K whole 64-k steps");
    }
    const auto* xs = static_cast<const __nv_bfloat16*>(x.data);
    const P12TmaDescriptors descriptors{p12_tma_byte_map(w.qdata, w.n, w.k, S::kBlockRows, S::kStepK),
                                        p12_tma_byte_map(w.qhigh, w.n, w.k / 2, S::kBlockRows, S::kStepK / 2),
                                        p12_rd_activation_map(xs, x.ne[1], w.k, S::kTokens)};
    const auto* words      = static_cast<const unsigned*>(w.scales);
    const auto* escapes    = static_cast<const unsigned*>(w.escapes);
    const std::int64_t blocks = static_cast<std::int64_t>(w.n / S::kBlockRows) * div_up(x.ne[1], S::kTokens);
    if (blocks > 2147483647LL) { throw std::invalid_argument("bf16_p12 register-decode TMA grid exceeds grid.x"); }
    with_linear_output(out, w.n, [&](auto output) {
        using Output       = decltype(output);
        const auto launch = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
            constexpr auto kernel = bf16_p12_rd_kernel<S, Output, InPlace>;
            bf16_prepare_shared<S::kSharedBytes, kernel>();
            kernel<<<static_cast<unsigned>(blocks), S::kThreads, S::kSharedBytes, stream>>>(
                parameter, words, escapes, output, w.n, w.k, x.ne[1]);
            CUDA_CHECK(cudaGetLastError());
        };
#ifdef _WIN32
        with_tma_descriptor_parameter(p12_tma_descriptor_staging(), descriptors, stream, launch);
#else
        launch(std::false_type{}, descriptors);
#endif
    });
}

} // namespace infernix::ops::detail
