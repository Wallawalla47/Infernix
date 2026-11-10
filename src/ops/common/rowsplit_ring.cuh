#pragma once

// The staging ring of the wide row-split GEMMs (Q4 / Q5 / Q6 / Q8 weights x BF16 activations): the mechanics a
// kernel shares, around its own codec and epilogue.
//
// A CTA is one row/token tile (a GemmCfg: BM weight rows x BN tokens, BK = 64) of consumer warps, the tile's MMA
// warps, followed by producer warps. For every K tile a producer thread issues its share of the BF16 activation
// copies (cp.async), reads its own chunks of the packed weights (a chunk is 32 consecutive K values of one row),
// decodes them to BF16 with the kernel's codec and stores them into the stage's swizzled weight tile, then arrives
// on the stage's full barrier once its copies have landed. A consumer warp waits for the stage, runs its MMAs
// (weights as the A operand, activations as B: ldmatrix from the two swizzled tiles) and releases it. The MMA warps
// therefore never stop for a decode pass or a CTA barrier, which the barrier-synchronised kernels do three times a
// K tile; every output is still one FP32 chain of m16n8k16 MMAs in ascending K over the same decoded BF16 weights,
// so a kernel moved onto the ring keeps its bits.
//
// Threads: consumers are threads [0, Tile::THREADS), producers the rest, so after consume() an epilogue written
// for the consumer tile (threadIdx.x striding over Tile::THREADS) runs unchanged; the producers have returned.
// Storage (dynamic shared memory, kSharedBytes): per stage the weight tile [BM][64] BF16 then the activation tile
// [BN][64] BF16, both under gemm_swz64; then the barriers. Once every consumer has passed consume(), a stage's
// bytes are the epilogue's scratch (BM x BN BF16 when BM == BN's tile bytes allow it: kStageBytes).
//
// Activation rows past the call's tokens are not loaded: activations are the B operand, so a token feeds only its
// own output column, which a bounds-checked epilogue does not store.

#include "core/device.h"
#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops::detail {

// ExtraBytes: per-stage bytes after the two tiles that a codec's publish() fills (the scaled form's FP16 row scales).
template <class Tile, int Stages, int ProducerWarps, int ExtraBytes = 0>
struct RowSplitRing {
    static constexpr int kStages          = Stages;
    static constexpr int kProducerThreads = ProducerWarps * 32;
    static constexpr int kConsumerThreads = Tile::THREADS;
    static constexpr int kThreads         = kConsumerThreads + kProducerThreads;
    static constexpr int kWeightBytes     = Tile::BM * Tile::BK * 2;
    static constexpr int kActivationBytes = Tile::BN * Tile::BK * 2;
    static constexpr int kStageBytes      = kWeightBytes + kActivationBytes + ExtraBytes;
    static constexpr int kBarrierOffset   = Stages * kStageBytes;
    static constexpr int kSharedBytes     = kBarrierOffset + Stages * 2 * static_cast<int>(sizeof(std::uint64_t));
    // Chunks (32 K values of one row) a producer thread decodes a K tile: chunk i is half (i & 1) of row i >> 1.
    static constexpr int kChunks = Tile::BM * 2 / kProducerThreads;
    static_assert(Tile::BK == 64, "a ring stage is one 64-value K tile");
    static_assert(Tile::BM * 2 % kProducerThreads == 0 && kChunks >= 1, "producer threads decode whole chunks");
    static_assert(Stages >= 2 && kThreads <= 1024 && kSharedBytes <= 99 * 1024);

    __device__ static __forceinline__ unsigned char* storage() {
        extern __shared__ __align__(128) unsigned char rowsplit_ring_shared[];
        return rowsplit_ring_shared;
    }
    __device__ static __forceinline__ __nv_bfloat16* weight_tile(int stage) {
        return reinterpret_cast<__nv_bfloat16*>(storage() + stage * kStageBytes);
    }
    __device__ static __forceinline__ __nv_bfloat16* activation_tile(int stage) {
        return reinterpret_cast<__nv_bfloat16*>(storage() + stage * kStageBytes + kWeightBytes);
    }
    __device__ static __forceinline__ unsigned char* extra(int stage) {
        return storage() + stage * kStageBytes + kWeightBytes + kActivationBytes;
    }
    __device__ static __forceinline__ std::uint64_t* full() {
        return reinterpret_cast<std::uint64_t*>(storage() + kBarrierOffset);
    }
    __device__ static __forceinline__ std::uint64_t* empty() { return full() + Stages; }
    __device__ static __forceinline__ bool producer() { return threadIdx.x >= kConsumerThreads; }

    // Every thread, first: the stages' barriers.
    __device__ static __forceinline__ void init() {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int stage = 0; stage < Stages; ++stage) {
                cta_mbarrier_init(full() + stage, kProducerThreads);
                cta_mbarrier_init(empty() + stage, Tile::WARPS);
            }
            cta_mbarrier_fence_init();
        }
        __syncthreads();
    }

    // A producer thread's whole work: `k_tiles` stages of the activation tile x[(token_begin + t) * k + ...] for
    // the tile's first `tokens` tokens (K values at or past k are zero) and of its weight chunks. Codec:
    //   Raw load(int row, int k_tile, int half) const      a chunk's packed bytes and scale (zero for a row or K
    //                                                       range without weights)
    //   void decode(const Raw&, unsigned (&pairs)[16]) const  its 32 BF16 weights as 16 pairs, in K order
    template <class Codec>
    __device__ static __forceinline__ void produce(const Codec& codec, const __nv_bfloat16* __restrict__ x, int k,
                                                   int token_begin, int tokens, int k_tiles) {
        const int p = static_cast<int>(threadIdx.x) - kConsumerThreads;
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int stage = kt % Stages;
            cta_mbarrier_wait(empty() + stage, 1U ^ ((kt / Stages) & 1U));
            __nv_bfloat16* As = weight_tile(stage);
            __nv_bfloat16* Bs = activation_tile(stage);
            const int k0      = kt * Tile::BK;
#pragma unroll 1
            for (int c = p; c < tokens * (Tile::BK / 8); c += kProducerThreads) {
                const int tl = c / (Tile::BK / 8), kl = (c - tl * (Tile::BK / 8)) * 8;
                __nv_bfloat16* dst = &Bs[tl * Tile::BK + gemm_swz64(tl, kl)];
                if (k0 + kl + 8 <= k) {
                    gemm_cp_async<16, Tile>(dst, &x[static_cast<std::int64_t>(token_begin + tl) * k + k0 + kl]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
            // Both chunks' loads are issued before either is decoded.
            typename Codec::Raw raw[kChunks];
#pragma unroll
            for (int j = 0; j < kChunks; ++j) {
                const int i = p + j * kProducerThreads;
                raw[j]      = codec.load(i >> 1, kt, i & 1);
            }
#pragma unroll
            for (int j = 0; j < kChunks; ++j) {
                const int i = p + j * kProducerThreads, row = i >> 1, half = i & 1;
                unsigned pairs[16];
                codec.decode(raw[j], pairs);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    store_vec(&As[row * Tile::BK + gemm_swz64(row, half * 32 + w * 8)],
                              make_uint4(pairs[4 * w], pairs[4 * w + 1], pairs[4 * w + 2], pairs[4 * w + 3]));
                }
            }
            // Arrives once this thread's activation copies have landed (its weight stores precede the arrive).
            asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" : : "r"(smem_addr(full() + stage)) : "memory");
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
    }

    // produce() with the activations from a Source: source(ordinal, token, k_offset) is the address of the eight K
    // values at k_offset of the tile's token, or null for zeros; ordinal counts the thread's copies of a K tile
    // (copy `ordinal` is the same token and K phase in every K tile, so a gathering source resolves its rows
    // once). tile0: the K tiles this CTA's ring has already carried (a kernel that runs several tiles in turn
    // passes the running count to its producers and consumers). A codec with publish(raw, row, half, extra) has
    // it called for every chunk before the stage is published. Kept apart from produce() / consume(): the linear
    // rings routed through these general forms measured +0.4 % on the groupwise 27B's prefill (2026-11-10).
    template <class Codec, class Source>
    __device__ static __forceinline__ void produce_from(const Codec& codec, const Source& source, int tokens,
                                                        int k_tiles, int tile0) {
        const int p = static_cast<int>(threadIdx.x) - kConsumerThreads;
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int ring_tile = tile0 + kt, stage = ring_tile % Stages;
            cta_mbarrier_wait(empty() + stage, 1U ^ ((ring_tile / Stages) & 1U));
            __nv_bfloat16* As = weight_tile(stage);
            __nv_bfloat16* Bs = activation_tile(stage);
            const int k0      = kt * Tile::BK;
            int ordinal       = 0;
#pragma unroll 1
            for (int c = p; c < tokens * (Tile::BK / 8); c += kProducerThreads, ++ordinal) {
                const int tl = c / (Tile::BK / 8), kl = (c - tl * (Tile::BK / 8)) * 8;
                __nv_bfloat16* dst        = &Bs[tl * Tile::BK + gemm_swz64(tl, kl)];
                const __nv_bfloat16* from = source(ordinal, tl, k0 + kl);
                if (from != nullptr) {
                    gemm_cp_async<16, Tile>(dst, from);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
            // Both chunks' loads are issued before either is decoded.
            typename Codec::Raw raw[kChunks];
#pragma unroll
            for (int j = 0; j < kChunks; ++j) {
                const int i = p + j * kProducerThreads;
                raw[j]      = codec.load(i >> 1, kt, i & 1);
            }
#pragma unroll
            for (int j = 0; j < kChunks; ++j) {
                const int i = p + j * kProducerThreads, row = i >> 1, half = i & 1;
                unsigned pairs[16];
                codec.decode(raw[j], pairs);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    store_vec(&As[row * Tile::BK + gemm_swz64(row, half * 32 + w * 8)],
                              make_uint4(pairs[4 * w], pairs[4 * w + 1], pairs[4 * w + 2], pairs[4 * w + 3]));
                }
                if constexpr (requires { codec.publish(raw[j], row, half, extra(stage)); }) {
                    codec.publish(raw[j], row, half, extra(stage));
                }
            }
            // Arrives once this thread's activation copies have landed (its weight stores precede the arrive).
            asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" : : "r"(smem_addr(full() + stage)) : "memory");
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
    }

    // A consumer thread's MMAs over `k_tiles` stages: acc[mi][ni] is row tile mi (16 rows) x token tile ni (8
    // tokens) of its warp's tile, as the plain kernels' accumulators.
    __device__ static __forceinline__ void consume(float (&acc)[Tile::MT][Tile::NT][4], int k_tiles) {
        constexpr int KSUB = Tile::BK / 16;
        const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
        const int wm = warp / Tile::WARPS_N, wn = warp % Tile::WARPS_N;
        const int a_mat    = lane >> 3;
        const int a_rowoff = (lane & 7) + ((a_mat & 1) << 3);
        const int a_coloff = (a_mat >> 1) << 3;
        const int b_rin    = lane & 7;
        const int b_koff   = ((lane >> 3) & 1) << 3;
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int stage = kt % Stages;
            cta_mbarrier_wait(full() + stage, (kt / Stages) & 1U);
            const __nv_bfloat16* As = weight_tile(stage);
            const __nv_bfloat16* Bs = activation_tile(stage);
            unsigned af[Tile::MT][4];
            unsigned bf[Tile::NT][2];
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                const int ks = ki * 16;
#pragma unroll
                for (int mi = 0; mi < Tile::MT; ++mi) {
                    const int arow = wm * Tile::WM + mi * 16 + a_rowoff;
                    ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                smem_addr(&As[arow * Tile::BK + gemm_swz64(arow, ks + a_coloff)]));
                }
#pragma unroll
                for (int ni = 0; ni < Tile::NT; ++ni) {
                    const int brow = wn * Tile::WN + ni * 8 + b_rin;
                    ldmatrix_x2(bf[ni][0], bf[ni][1],
                                smem_addr(&Bs[brow * Tile::BK + gemm_swz64(brow, ks + b_koff)]));
                }
#pragma unroll
                for (int mi = 0; mi < Tile::MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < Tile::NT; ++ni) {
                        mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], af[mi][0],
                                 af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1]);
                    }
                }
            }
            // Release only after every lane of the warp has finished its reads of the stage.
            __syncwarp();
            if (lane == 0) cta_mbarrier_arrive(empty() + stage);
        }
    }

    // consume() for a kernel that runs several tiles in turn (tile0 as produce_from's) and whose warps may be
    // inactive: a warp that is not `active` (its tokens lie past the tile's) passes the stages without their MMAs.
    __device__ static __forceinline__ void consume_run(float (&acc)[Tile::MT][Tile::NT][4], int k_tiles, int tile0,
                                                       bool active) {
        const int lane = static_cast<int>(threadIdx.x) & 31;
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int ring_tile = tile0 + kt, stage = ring_tile % Stages;
            cta_mbarrier_wait(full() + stage, (ring_tile / Stages) & 1U);
            if (active) { stage_mmas(acc, stage); }
            // Release only after every lane of the warp has finished its reads of the stage.
            __syncwarp();
            if (lane == 0) cta_mbarrier_arrive(empty() + stage);
        }
    }

    // consume_run() for weights staged as unscaled codes under one scale per row and K tile: a K tile's MMAs sum
    // into a partial, which is then added times the row's scale, the FP16 the codec published at
    // extra(stage)[row] (one FP32 chain per K tile and one fused multiply-add per K tile, in ascending K).
    __device__ static __forceinline__ void consume_scaled(float (&acc)[Tile::MT][Tile::NT][4], int k_tiles,
                                                          int tile0, bool active) {
        static_assert(ExtraBytes >= Tile::BM * static_cast<int>(sizeof(std::uint16_t)));
        const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
        const int row = (warp / Tile::WARPS_N) * Tile::WM + (lane >> 2);
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int ring_tile = tile0 + kt, stage = ring_tile % Stages;
            cta_mbarrier_wait(full() + stage, (ring_tile / Stages) & 1U);
            if (active) {
                float partial[Tile::MT][Tile::NT][4] = {};
                stage_mmas(partial, stage);
                const std::uint16_t* scale = reinterpret_cast<const std::uint16_t*>(extra(stage));
#pragma unroll
                for (int mi = 0; mi < Tile::MT; ++mi) {
                    const float s0 = __half2float(__ushort_as_half(scale[row + mi * 16]));
                    const float s1 = __half2float(__ushort_as_half(scale[row + mi * 16 + 8]));
#pragma unroll
                    for (int ni = 0; ni < Tile::NT; ++ni) {
                        acc[mi][ni][0] = fmaf(partial[mi][ni][0], s0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(partial[mi][ni][1], s0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(partial[mi][ni][2], s1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(partial[mi][ni][3], s1, acc[mi][ni][3]);
                    }
                }
            }
            __syncwarp();
            if (lane == 0) cta_mbarrier_arrive(empty() + stage);
        }
    }

private:
    // A consumer thread's MMAs of one staged K tile, added to acc in ascending K.
    __device__ static __forceinline__ void stage_mmas(float (&acc)[Tile::MT][Tile::NT][4], int stage) {
        constexpr int KSUB = Tile::BK / 16;
        const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
        const int wm = warp / Tile::WARPS_N, wn = warp % Tile::WARPS_N;
        const int a_mat    = lane >> 3;
        const int a_rowoff = (lane & 7) + ((a_mat & 1) << 3);
        const int a_coloff = (a_mat >> 1) << 3;
        const int b_rin    = lane & 7;
        const int b_koff   = ((lane >> 3) & 1) << 3;
        const __nv_bfloat16* As = weight_tile(stage);
        const __nv_bfloat16* Bs = activation_tile(stage);
        unsigned af[Tile::MT][4];
        unsigned bf[Tile::NT][2];
#pragma unroll
        for (int ki = 0; ki < KSUB; ++ki) {
            const int ks = ki * 16;
#pragma unroll
            for (int mi = 0; mi < Tile::MT; ++mi) {
                const int arow = wm * Tile::WM + mi * 16 + a_rowoff;
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                            smem_addr(&As[arow * Tile::BK + gemm_swz64(arow, ks + a_coloff)]));
            }
#pragma unroll
            for (int ni = 0; ni < Tile::NT; ++ni) {
                const int brow = wn * Tile::WN + ni * 8 + b_rin;
                ldmatrix_x2(bf[ni][0], bf[ni][1], smem_addr(&Bs[brow * Tile::BK + gemm_swz64(brow, ks + b_koff)]));
            }
#pragma unroll
            for (int mi = 0; mi < Tile::MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < Tile::NT; ++ni) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], af[mi][0], af[mi][1],
                             af[mi][2], af[mi][3], bf[ni][0], bf[ni][1]);
                }
            }
        }
    }
};

// The consumer tile under the schedule names the tile epilogues of the row-split linears use.
template <class Ring, class Tile>
struct RowSplitRingSchedule {
    static constexpr int kBlockRows      = Tile::BM;
    static constexpr int kBlockTokens    = Tile::BN;
    static constexpr int kBlockK         = Tile::BK;
    static constexpr int kWarpRows       = Tile::WM;
    static constexpr int kWarpTokens     = Tile::WN;
    static constexpr int kWarpGridRows   = Tile::WARPS_M;
    static constexpr int kWarpGridTokens = Tile::WARPS_N;
    static constexpr int kWarps          = Tile::WARPS;
    static constexpr int kThreads        = Tile::THREADS;
    static constexpr int kMmaRows        = Tile::MT;
    static constexpr int kMmaTokens      = Tile::NT;
    // BF16 elements of the scratch a tile epilogue receives (one stage).
    static constexpr int kScratchElements = Ring::kStageBytes / 2;
};

// One dynamic shared-memory reservation per ring kernel (kernels of one function type need one each).
template <auto Kernel, int Bytes>
int rowsplit_ring_reserve() {
    static const auto status = cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
    CUDA_CHECK(status);
    return Bytes;
}

} // namespace infernix::ops::detail
