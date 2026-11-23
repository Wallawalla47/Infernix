#pragma once

// The wide row-split linear GEMM on the staging ring (ops/common/rowsplit_ring.cuh): out[N,T] = W[N,K] * x[K,T]
// for a weight format given by `Format`, into any linear output through any linear epilogue.
//
// Format (q4_ring_codec.cuh, q5_ring_codec.cuh, ...):
//   Weights                                     the packed planes, by value (a kernel parameter)
//   codec(weights, k, padded_k, row_of)         the ring codec of those planes
// Epilogue: a tile epilogue (finish_tile<Schedule, Full>(output, scratch, acc, row_begin, token_begin, rows,
// token_end), the schedule being RowSplitRingSchedule and the scratch one ring stage) or a pointwise one
// (apply(row, token, value)), as the barrier-synchronised kernels take them (q5_a16_mma.cuh). The accumulators hold
// the same FP32 sums as those kernels', so an Op keeps its bits when its route moves here.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_ring.cuh"
#include "ops/common/token_slices.h"
#include "ops/lora/lora_device.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

// The tile every ring linear runs unless its own sweep says otherwise: 128 weight rows x 128 tokens a CTA, sixteen
// MMA warps of 64 rows x 16 tokens, three stages, four producer warps (98.4 KB of shared memory: one CTA an SM).
using RowSplitRingTile128 = GemmCfg<128, 128, 64, 64, 16, 2, 1, false, true, false>;
// Producer warps, RTX 5090, 2026-11-10 (end to end, A B B A): two starve the ring (groupwise 27B prefill +9 ..
// +15 %, the 35B +3 .. +4 %); eight are level on the groupwise 27B and 1.3-2.2 % slower on the 35B.
inline constexpr int kRowSplitRingStages        = 3;
inline constexpr int kRowSplitRingProducerWarps = 4;

// The device's SM count: a ring CTA owns its SM, so a launch runs in waves of this many CTAs.
inline int rowsplit_ring_sms() {
    static const int sms = [] {
        int device = 0, count = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    return sms;
}
// The waves a ring launch of `ctas` CTAs takes.
inline int rowsplit_ring_waves(std::int64_t ctas) {
    return static_cast<int>((ctas + rowsplit_ring_sms() - 1) / rowsplit_ring_sms());
}

template <class Format, class Tile, int Stages, int ProducerWarps, bool Full, class Output, class Epilogue>
__global__ __launch_bounds__(RowSplitRing<Tile, Stages, ProducerWarps>::kThreads, Tile::MIN_BLOCKS)
void rowsplit_ring_linear_kernel(const __nv_bfloat16* __restrict__ x, typename Format::Weights weights,
                                 Output output, Epilogue epilogue, std::int32_t rows, std::int32_t k,
                                 std::int32_t cols, std::int32_t padded_k, std::int32_t token_begin) {
    using Ring = RowSplitRing<Tile, Stages, ProducerWarps>;
    constexpr int BM = Tile::BM, BN = Tile::BN, MT = Tile::MT, NT = Tile::NT;
    Ring::init();

    const int k_tiles = padded_k / Tile::BK;
    const int row0    = static_cast<int>(blockIdx.x) * BM;
    const int col0    = static_cast<int>(blockIdx.y) * BN;

    if (Ring::producer()) {
        const auto row_of = [&](int row) -> std::int64_t { return Full || row0 + row < rows ? row0 + row : -1; };
        Ring::produce(Format::codec(weights, k, padded_k, row_of), x, k, col0, Full ? BN : min(BN, cols - col0),
                      k_tiles);
        return;
    }

    if constexpr (kLoraEpilogue<Epilogue>) {
        // The term's B rows and U columns into L2 while the ring runs (consumer threads).
        lora_prefetch_tile(
            epilogue.lora, BM, [&](int i) { return row0 + i < rows ? row0 + i : rows - 1; }, token_begin + col0,
            Full ? BN : min(BN, cols - col0), static_cast<int>(threadIdx.x), Tile::THREADS);
    }
    float acc[MT][NT][4] = {};
    Ring::consume(acc, k_tiles);
    if constexpr (kLoraEpilogue<Epilogue>) {
        // --lora: the term on each consumer warp's fragments (canonical chains), before any epilogue form below.
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int wm = warp / Tile::WARPS_N, wn = warp % Tile::WARPS_N;
        if constexpr (kLoraStageRank<Epilogue> > kLoraNarrowRank &&
                      (BM + BN) * (2 * kLoraStageRank<Epilogue> + 16) <= Ring::kBarrierOffset) {
            // A wide instance: B's rows and U's columns staged in the free ring (every consumer warp busy each pass).
            // A narrow one reads them from L2: staged, the groupwise 27B's instances took 0.6 % longer at rank 32
            // (2,048-token prefill, RTX 5090, 2026-11-22).
            lora_add_rows_tile_staged<BM, BN, Ring::kBarrierOffset>(
                epilogue.lora, acc, [&](int mm, int i) { return wm * Tile::WM + mm * 16 + i; },
                [&](int mn, int j) { return wn * Tile::WN + mn * 8 + j; },
                [&](int i) { return row0 + i < rows ? row0 + i : rows - 1; }, token_begin + col0, token_begin + cols,
                Ring::storage(), static_cast<int>(threadIdx.x), Tile::THREADS,
                [] { asm volatile("bar.sync 1, %0;" ::"r"(Tile::THREADS) : "memory"); });
        } else {
            lora_add_rows_tile<MT, NT>(
                epilogue.lora, acc,
                [&](int mm, int i) {
                    const int r = row0 + wm * Tile::WM + mm * 16 + i;
                    return r < rows ? r : rows - 1;
                },
                [&](int mn, int j) {
                    const int t = col0 + wn * Tile::WN + mn * 8 + j;
                    return t < cols ? token_begin + t : -1;
                });
        }
    }

    using Schedule = RowSplitRingSchedule<Ring, Tile>;
    if constexpr (requires {
                      epilogue.template finish_tile<Schedule, Full>(output, static_cast<__nv_bfloat16*>(nullptr), acc,
                                                                    row0, token_begin + col0, rows, token_begin + cols);
                  }) {
        // Every consumer is done with the stages before one of them becomes the epilogue's scratch.
        __syncthreads();
        epilogue.template finish_tile<Schedule, Full>(output, reinterpret_cast<__nv_bfloat16*>(Ring::storage()), acc,
                                                      row0, token_begin + col0, rows, token_begin + cols);
    } else if constexpr (requires {
                             epilogue.template finish_tile<Schedule, Full>(output, static_cast<unsigned char*>(nullptr),
                                                                           acc, row0, token_begin + col0, rows,
                                                                           token_begin + cols);
                         }) {
        // A tile epilogue that takes the whole ring as raw scratch (the Q8 residual add's FP32 tile).
        static_assert(Ring::kSharedBytes >= BM * BN * static_cast<int>(sizeof(float)));
        __syncthreads();
        epilogue.template finish_tile<Schedule, Full>(output, Ring::storage(), acc, row0, token_begin + col0, rows,
                                                      token_begin + cols);
    } else {
        const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
        const int wm = warp / Tile::WARPS_N, wn = warp % Tile::WARPS_N;
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row = row0 + wm * Tile::WM + mi * 16 + (lane >> 2);
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int token  = col0 + wn * Tile::WN + ni * 8 + 2 * (lane & 3);
                const auto store = [&](int r, int t, float value) {
                    if (Full || (r < rows && t < cols)) {
                        output.store(r, token_begin + t, epilogue.apply(r, token_begin + t, value));
                    }
                };
                store(row, token, acc[mi][ni][0]);
                store(row, token + 1, acc[mi][ni][1]);
                store(row + 8, token, acc[mi][ni][2]);
                store(row + 8, token + 1, acc[mi][ni][3]);
            }
        }
    }
}

// x: the call's activations [K, tokens]; a full launch reads no bounds (whole row and token tiles, no K padding).
template <class Format, class Tile, int Stages, int ProducerWarps, class Output, class Epilogue>
void launch_rowsplit_ring_linear(const __nv_bfloat16* x, typename Format::Weights weights, std::int32_t rows,
                                 std::int32_t k, std::int32_t padded_k, std::int32_t tokens, Output output,
                                 Epilogue epilogue, cudaStream_t stream) {
    using Ring = RowSplitRing<Tile, Stages, ProducerWarps>;
    for_each_token_slice(tokens, Tile::BN, [&](int offset, int count) {
        const dim3 grid(static_cast<unsigned>(div_up(rows, Tile::BM)), static_cast<unsigned>(div_up(count, Tile::BN)));
        const bool full   = rows % Tile::BM == 0 && count % Tile::BN == 0 && k == padded_k;
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel =
                rowsplit_ring_linear_kernel<Format, Tile, Stages, ProducerWarps, Full, Output, Epilogue>;
            kernel<<<grid, Ring::kThreads, reserve_dynamic_shared<kernel, Ring::kSharedBytes>(), stream>>>(
                x + static_cast<std::int64_t>(offset) * k, weights, output, epilogue, rows, k, count, padded_k, offset);
            CUDA_CHECK(cudaGetLastError());
        };
        if (full) {
            launch.template operator()<true>();
        } else {
            launch.template operator()<false>();
        }
    });
}

} // namespace infernix::ops::detail
