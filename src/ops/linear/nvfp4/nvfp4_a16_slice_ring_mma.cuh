#pragma once

// The column-invariant NVFP4 A16 arithmetic of the sliced-K tile (nvfp4_a16_sliced_k_mma.cuh) on a
// pipelined tile of many weight rows, bit for bit.
//
// The sliced-K tile of W K-warps gives warp w the 64-value K slice [g*64W + 64w, +64) of every group g
// and reduces a column as ((p0 + p1) + (p2 + p3)) + ... + (p_{W-2} + p_{W-1}), where p_w is warp w's
// FP32 chain of m16n8k16 BF16 MMAs (weights as A, activations as B) from zero over its slices in K
// order. Its CTAs cover 16 weight rows, so each one streams the call's whole activation block from
// L2 (34816 / 16 = 2,176 CTAs read it for the MLP gate/up), and a stage holds 64W values of K, which
// leaves no room for a second stage at 64 columns.
//
// This tile computes the same chains for BlockRows rows. Its consumer warps walk slice w of every
// group together (w = 0, 1, ..., W-1 in turn), so the K stream is slice-major and a stage is one
// 64-value slice: activations, codes and scales of a few kilobytes that one producer warp streams with
// cp.async through a Stages-deep mbarrier ring (cp.async writes are generic-proxy, ordered by the
// barriers; no proxy fence is needed). The consumers widen each code pair with its E4M3 scale in
// registers (nvfp4_scaled_pair_bf16, the sliced tile's own widening) into the A fragment the sliced
// tile loads, and issue the same MMAs into a chain that starts from zero at every slice. After a
// slice's last group, an even slice's chain is kept; an odd slice's is added to it (the sliced tile's
// partner sum) and that pair to the running total in the sliced tile's order. The global divisor and
// the epilogue apply as there. A column's result therefore equals the sliced-K tile's of the same W
// at every width (tests/ops/linear/test_nvfp4_a16_slice_ring.cpp checks it bit for bit).

#include "ops/common/math.h"
#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/token_slices.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {

// Arrives on `barrier` once every cp.async this thread issued earlier has landed; the arrival is one
// of the count the barrier was initialized with (.noinc).
__device__ __forceinline__ void nvfp4_cp_async_arrive(std::uint64_t* barrier) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_addr(barrier))
                 : "memory");
}

// Byte `index` (0..3) of `word`.
__device__ __forceinline__ std::uint8_t nvfp4_ring_byte(unsigned word, int index) {
    return static_cast<std::uint8_t>(word >> (8 * index));
}

// Word `index` (0..3, a compile-time constant after unrolling) of `words`.
__device__ __forceinline__ unsigned nvfp4_ring_word(const uint4& words, int index) {
    return index == 0 ? words.x : index == 1 ? words.y : index == 2 ? words.z : words.w;
}

template <class Schedule>
__device__ __forceinline__ void nvfp4_a16_slice_ring_compute_stage(
    const unsigned char* stage, int warp_row, int warp_token, int lane,
    float (&chain)[Schedule::kMmaRows][Schedule::kMmaTokens][4]) {
    constexpr int MT = Schedule::kMmaRows, NT = Schedule::kMmaTokens;
    const unsigned char* activation = stage;
    const unsigned char* codes      = stage + Schedule::kActivationStageBytes;
    const unsigned char* scales     = codes + Schedule::kCodeStageBytes;
    const int gid = lane >> 2, lid = lane & 3;
    // ldmatrix.x4 of two n8 token tiles: lanes 0-7 tile 0 k 0-7, 8-15 tile 0 k 8-15, 16-23 and 24-31
    // tile 1.
    const int b_row  = (lane & 7) + ((lane >> 4) << 3);
    const int b_high = (lane >> 3) & 1;

    // Each A row's four G16 scales of the slice: one word.
    unsigned scale_words[MT][2];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row = warp_row * Schedule::kWarpRows + mi * 16 + gid;
        scale_words[mi][0] = load_vec<unsigned>(scales + row * 4);
        scale_words[mi][1] = load_vec<unsigned>(scales + (row + 8) * 4);
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        // A row's codes of k16 steps 2*half and 2*half+1: one 16-byte half of its 32 bytes, the halves
        // of rows 4-7 of every 8 swapped (conflict-free).
        uint4 code_words[MT][2];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row = warp_row * Schedule::kWarpRows + mi * 16 + gid;
#pragma unroll
            for (int upper = 0; upper < 2; ++upper) {
                const int r = row + upper * 8;
                code_words[mi][upper] =
                    load_vec<uint4>(codes + r * 32 + ((half ^ ((r >> 2) & 1)) << 4));
            }
        }
#pragma unroll
        for (int step = 2 * half; step < 2 * half + 2; ++step) {
            // k = 16*step + 2*lid (+8): byte 8*step + lid (+4) of the row's 32, i.e. byte lid of word
            // 2*(step & 1) (+1) of its half.
            unsigned a[MT][4];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const uint4& top          = code_words[mi][0];
                const uint4& bottom       = code_words[mi][1];
                const int word            = 2 * (step & 1);
                const std::uint8_t scale0 = nvfp4_ring_byte(scale_words[mi][0], step);
                const std::uint8_t scale1 = nvfp4_ring_byte(scale_words[mi][1], step);
                // a0..a3: rows g and g+8 at k 2*lid, then at k 2*lid + 8 (the sliced tile's order).
                a[mi][0] = nvfp4_scaled_pair_bf16(
                    nvfp4_ring_byte(nvfp4_ring_word(top, word), lid), scale0);
                a[mi][1] = nvfp4_scaled_pair_bf16(
                    nvfp4_ring_byte(nvfp4_ring_word(bottom, word), lid), scale1);
                a[mi][2] = nvfp4_scaled_pair_bf16(
                    nvfp4_ring_byte(nvfp4_ring_word(top, word + 1), lid), scale0);
                a[mi][3] = nvfp4_scaled_pair_bf16(
                    nvfp4_ring_byte(nvfp4_ring_word(bottom, word + 1), lid), scale1);
            }
            unsigned b[NT][2];
#pragma unroll
            for (int pair = 0; pair < NT / 2; ++pair) {
                const int row   = warp_token * Schedule::kWarpTokens + pair * 16 + b_row;
                const int chunk = 2 * step + b_high;
                ldmatrix_x4(b[2 * pair][0], b[2 * pair][1], b[2 * pair + 1][0], b[2 * pair + 1][1],
                            smem_addr(activation + row * 128 + ((chunk ^ (row & 7)) << 4)));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(chain[mi][ni][0], chain[mi][ni][1], chain[mi][ni][2], chain[mi][ni][3],
                             a[mi][0], a[mi][1], a[mi][2], a[mi][3], b[ni][0], b[ni][1]);
                }
            }
        }
    }
}

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
nvfp4_a16_slice_ring_mma_kernel(Nvfp4A16Operands operands, Output output, Epilogue epilogue,
                                RowPolicy row_policy, int token_offset, int count) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, S = Schedule::kStages;
    constexpr int W  = Schedule::kKSlices;
    constexpr int MT = Schedule::kMmaRows, NT = Schedule::kMmaTokens;
    const int k           = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int groups      = k / Schedule::kBlockK;
    const int chunks      = groups * W;
    const int token_tiles = div_up(count, BT);
    // Token tiles of one row tile are neighbours, so its weights leave DRAM once.
    const int token_tile  = static_cast<int>(blockIdx.x) % token_tiles;
    const int row_tile    = static_cast<int>(blockIdx.x) / token_tiles;
    const int row_begin   = row_tile * (BR / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + token_tile * BT;
    const int tokens      = token_offset + count;

    extern __shared__ __align__(128) unsigned char nvfp4_a16_slice_ring_shared[];
    auto* full  = reinterpret_cast<std::uint64_t*>(nvfp4_a16_slice_ring_shared +
                                                   S * Schedule::kStageBytes);
    auto* empty = full + S;
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
        const int lane         = static_cast<int>(threadIdx.x);
        const int valid_tokens = min(BT, tokens - token_begin);
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int stage = chunk % S;
            cta_mbarrier_wait(empty + stage, 1U ^ ((chunk / S) & 1U));
            const int slice = chunk / groups;
            const int k0    = (chunk - slice * groups) * Schedule::kBlockK + slice * Schedule::kSliceK;
            unsigned char* activation = nvfp4_a16_slice_ring_shared + stage * Schedule::kStageBytes;
            unsigned char* codes      = activation + Schedule::kActivationStageBytes;
            unsigned char* scales     = codes + Schedule::kCodeStageBytes;
            // BF16 token rows, 64 values in one 128-byte swizzle span each. Rows past the slice's
            // tokens are not loaded: they feed only output columns the epilogue never stores.
            for (int i = lane; i < valid_tokens * 8; i += Schedule::kProducerThreads) {
                const int r = i >> 3, c = i & 7;
                cp_async<16, Cache::cg>(activation + r * 128 + ((c ^ (r & 7)) << 4),
                                        operands.x + static_cast<std::int64_t>(token_begin + r) * k +
                                            k0 + c * 8);
            }
            for (int i = lane; i < BR * 2; i += Schedule::kProducerThreads) {
                const int r = i >> 1, h = i & 1;
                const int parent = row_policy.weight_row(row_begin, r, operands.rows);
                cp_async<16, Cache::cg>(codes + r * 32 + ((h ^ ((r >> 2) & 1)) << 4),
                                        operands.codes +
                                            static_cast<std::int64_t>(parent) * (k / 2) + k0 / 2 +
                                            h * 16);
            }
            for (int r = lane; r < BR; r += Schedule::kProducerThreads) {
                const int parent = row_policy.weight_row(row_begin, r, operands.rows);
                cp_async<4>(scales + r * 4,
                            operands.scales + nvfp4_scale_byte_offset(parent, k0 / 16, k));
            }
            nvfp4_cp_async_arrive(full + stage);
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }

    const int tid        = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp       = tid >> 5, lane = tid & 31;
    const int warp_row   = warp % Schedule::kWarpsRows;
    const int warp_token = warp / Schedule::kWarpsRows;
    const int gid        = lane >> 2;
    float chain[MT][NT][4] = {};
    float kept[MT][NT][4]  = {};
    float total[MT][NT][4] = {};
    int chunk = 0;
#pragma unroll 1
    for (int slice = 0; slice < W; ++slice) {
#pragma unroll 1
        for (int group = 0; group < groups; ++group, ++chunk) {
            const int stage = chunk % S;
            cta_mbarrier_wait(full + stage, (chunk / S) & 1U);
            nvfp4_a16_slice_ring_compute_stage<Schedule>(
                nvfp4_a16_slice_ring_shared + stage * Schedule::kStageBytes, warp_row, warp_token,
                lane, chain);
            // Release only after every lane of this warp has finished its shared reads.
            __syncwarp();
            if (lane == 0) cta_mbarrier_arrive(empty + stage);
        }
        // The sliced tile's reduction: kept = p_{2j}; total = (p0 + p1), then total += (p_{2j} + p_{2j+1}).
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if ((slice & 1) == 0) {
                        kept[mi][ni][e] = chain[mi][ni][e];
                    } else {
                        const float pair = kept[mi][ni][e] + chain[mi][ni][e];
                        total[mi][ni][e] = slice == 1 ? pair : total[mi][ni][e] + pair;
                    }
                    chain[mi][ni][e] = 0.0f;
                }
            }
        }
    }

    const auto destination =
        linear_output_tile<BR / (RowPolicy::kPaired ? 2 : 1)>(output, row_begin);
    const float alpha = operands.alpha;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int local_row = warp_row * Schedule::kWarpRows + mi * 16 + gid;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int token0 = token_begin + warp_token * Schedule::kWarpTokens + ni * 8 + 2 * (lane & 3);
            const auto& v    = total[mi][ni];
            if constexpr (requires { RowPolicy::kThreadPaired; }) {
                // Rows g (gate) and g+8 (up) of each m16 tile belong to one output row.
                const int out_row = row_policy.output_row(row_begin, local_row);
                if (token0 < tokens)
                    epilogue.apply_pair(destination, out_row, token0, v[0] * alpha, v[2] * alpha);
                if (token0 + 1 < tokens)
                    epilogue.apply_pair(destination, out_row, token0 + 1, v[1] * alpha,
                                        v[3] * alpha);
            } else {
                static_assert(!RowPolicy::kPaired, "paired rows require a thread-paired row policy");
                const int row0 = row_policy.weight_row(row_begin, local_row, operands.rows);
                const int row1 = row_policy.weight_row(row_begin, local_row + 8, operands.rows);
                if (token0 < tokens) {
                    destination.store(row0, token0, epilogue.apply(row0, token0, v[0] * alpha));
                    destination.store(row1, token0, epilogue.apply(row1, token0, v[2] * alpha));
                }
                if (token0 + 1 < tokens) {
                    destination.store(row0, token0 + 1,
                                      epilogue.apply(row0, token0 + 1, v[1] * alpha));
                    destination.store(row1, token0 + 1,
                                      epilogue.apply(row1, token0 + 1, v[3] * alpha));
                }
            }
        }
    }
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a16_slice_ring_mma(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                                     cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
        (Rows::kPaired && (p.rows / 2) % (Schedule::kBlockRows / 2)))
        throw std::invalid_argument("NVFP4 slice ring requires complete row and K-group tiles");
    static constexpr auto kernel = nvfp4_a16_slice_ring_mma_kernel<Schedule, Output, Epilogue, Rows>;
    const int bytes              = nvfp4_prepare_shared<Schedule::kSharedBytes, kernel, true>();
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const int blocks = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
        kernel<<<blocks, Schedule::kThreads, bytes, stream>>>(p, output, epilogue, rows, offset, count);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace infernix::ops::detail
