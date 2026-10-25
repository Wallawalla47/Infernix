#pragma once

// Pipelined sliced-K MMA for the groupwise RowSplit formats (Q4/Q5/Q6_G64_FP16):
// out[N,T] = W[N,K] . x[K,T] at decode and verification widths.
//
// The arithmetic is the release sliced-K kernels' (q4/q5/q6_a16_sliced_k_mma.cuh) exactly: K warp w
// owns quantization group w of every KWarps x 64 K tile, a group's integer codes are contracted on
// the BF16 tensor cores in the same k16 order, its binary16 scale multiplies the FP32 group sum in
// the same fmaf, and the K warps' partials are reduced in the same linear or pairwise order, so a
// launch returns the release kernel's bits. Only the staging differs:
// - the cp.async ring holds 1-4 K tiles, chosen per launch (sliced_k_route.h), in dynamic shared
//   memory up to the 99 KiB opt-in limit, where the release kernels hold one or two in at most
//   48 KiB of static shared memory;
// - the code and high-bit planes keep their 16-byte chunks XOR-swizzled by row, so a fragment read
//   by eight rows hits eight bank groups, and a lane reads its code bytes of a group as two 8-byte
//   vectors and its high bits as two words, where the release kernels read every byte alone from
//   addresses eight rows share (8-way conflicts);
// - a code pair becomes BF16 integers by a bias: 128 + (q + bias) is exact in BF16, so one lop3
//   places both halves and one hsub2 removes the bias, the bits of __floats2bfloat162_rn(q0, q1);
// - block rows are 16, 32 or 64 (one, two or four m16 row tiles per warp), so a call can trade CTAs
//   for rows per CTA.
// Weights must be full (rows a multiple of the block rows, K == padded K and a multiple of the K
// tile, scales aligned to a K tile's row of scales); the format launchers keep the release kernel
// otherwise.

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/token_slices.h"
#include "ops/linear/common/sliced_k_route.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace infernix::ops::detail {

enum class SlicedKFormat : std::uint8_t { Q4, Q5, Q6 };

// One call's planes: x [K, tokens] BF16 and the RowSplit codes, high bits (Q5, Q6) and FP16 scales.
struct SlicedKOperands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    std::int32_t rows;
    std::int32_t k;
    std::int32_t tokens;
    std::int32_t padded_k;
};

template <SlicedKFormat Format, int BlockRows, int BlockTokens, int TokenCapacity, int KWarps,
          bool Pairwise, int StaticK, Cache WeightCache, Cache ActivationCache>
struct SlicedKPipelinedSchedule {
    static_assert(BlockRows == 16 || BlockRows == 32 || BlockRows == 64);
    static_assert(BlockTokens >= 8 && BlockTokens <= 64 && BlockTokens % 8 == 0);
    static_assert(TokenCapacity > 0 && TokenCapacity <= BlockTokens);
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8);
    static_assert(!Pairwise || KWarps % 2 == 0);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % (KWarps * 64) == 0));
    static constexpr SlicedKFormat kFormat  = Format;
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kTokenCapacity     = TokenCapacity;
    static constexpr int kKWarps            = KWarps;
    static constexpr bool kPairwise         = Pairwise;
    static constexpr int kStaticK           = StaticK;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr Cache kActivationCache = ActivationCache;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kBlockK            = KWarps * 64;
    static constexpr int kMmaRows           = BlockRows / 16;
    static constexpr int kMmaTokens         = BlockTokens / 8;
    static constexpr int kMaxStages         = 4;
    // The launch bound's register budget: 128 registers a thread at 8 and 4 K warps, 255 at 2.
    // Residency is mostly shared-memory bound (the ring); the occupancy query sees the real count.
    static constexpr int kMinBlocksPerSm = KWarps == 8 ? 2 : 4;
    // One ring slot: per row the K tile's codes (32 bytes a group), high bits (Q5 8, Q6 16 bytes a
    // group) and scales (2 bytes a group), then each K warp's activation tile.
    static constexpr int kHighGroupBytes = Format == SlicedKFormat::Q4 ? 0
                                           : Format == SlicedKFormat::Q5 ? 8
                                                                         : 16;
    static constexpr int kCodeRowBytes    = KWarps * 32;
    static constexpr int kHighRowBytes    = KWarps * kHighGroupBytes;
    static constexpr int kScaleRowBytes   = KWarps * 2;
    static constexpr int kCodeBytes       = BlockRows * kCodeRowBytes;
    static constexpr int kHighBytes       = BlockRows * kHighRowBytes;
    static constexpr int kScaleBytes      = (BlockRows * kScaleRowBytes + 15) / 16 * 16;
    static constexpr int kXBytes          = KWarps * BlockTokens * 64 * 2;
    static constexpr int kSlotBytes       = kCodeBytes + kHighBytes + kScaleBytes + kXBytes;
    static constexpr int kWeightSlotBytes = kCodeBytes + kHighBytes + BlockRows * kScaleRowBytes;
    // FP32 partials of the K warps, which reuse the ring after the K loop.
    static constexpr int kPartialBytes = KWarps * kMmaRows * kMmaTokens * 32 * 4 * 4;
    static constexpr int kMaxSharedBytes = 99 * 1024;
    // Whether the accumulators (two of 4 x kMmaRows x kMmaTokens floats) fit the register budget.
    static constexpr bool kFitsRegisters =
        8 * kMmaRows * kMmaTokens + 48 <=
        std::min(255, 65536 / (kThreads * kMinBlocksPerSm));
    static_assert(kSlotBytes <= kMaxSharedBytes && kPartialBytes <= kMaxSharedBytes);

    static constexpr std::size_t shared_bytes(int stages) noexcept {
        return static_cast<std::size_t>(std::max(stages * kSlotBytes, kPartialBytes));
    }
    // The opt-in the kernel takes once: its deepest ring within the limit.
    static constexpr std::size_t kOptInBytes = static_cast<std::size_t>(
        std::min(std::max(kMaxStages * kSlotBytes, kPartialBytes), kMaxSharedBytes));
};

// The schedules of one release sliced-K schedule at other block rows.
template <SlicedKFormat Format, int BaseRows, int BlockTokens, int TokenCapacity, int KWarps,
          bool Pairwise, int StaticK, Cache WeightCache, Cache ActivationCache>
struct SlicedKPipelinedFamily {
    static constexpr int kBaseRows = BaseRows;
    template <int Rows>
    using Schedule = SlicedKPipelinedSchedule<Format, Rows, BlockTokens, TokenCapacity, KWarps,
                                              Pairwise, StaticK, WeightCache, ActivationCache>;
    template <int Rows>
    static constexpr bool fits() {
        if constexpr (Rows == 16 || Rows == 32 || Rows == 64) {
            return Schedule<Rows>::kFitsRegisters;
        } else {
            return false;
        }
    }
};

// Identity rows for formats whose release kernels take no row policy (Q6).
template <int BlockRows>
struct SlicedKIdentityRows {
    static constexpr int kOutputRowsPerCta = BlockRows;
    template <int Rows>
    using Rebind = SlicedKIdentityRows<Rows>;

    __host__ __device__ int output_rows(int rows) const { return rows; }

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

// The output row of a fragment row: a policy may map its CTA's rows (output_row), else identity.
template <class RowPolicy>
__device__ __forceinline__ int sliced_k_output_row(const RowPolicy& policy, int row_begin,
                                                   int local_row) {
    if constexpr (requires { policy.output_row(row_begin, local_row); }) {
        return policy.output_row(row_begin, local_row);
    } else {
        return row_begin + local_row;
    }
}

// A row's 16-byte chunk in its swizzled place: the eight rows a fragment read touches land in eight
// distinct 16-byte bank groups (RowBytes 16-256, a power of two).
template <int RowBytes>
__device__ __forceinline__ int sliced_k_chunk(int row, int chunk) {
    constexpr int kChunks = RowBytes / 16;
    if constexpr (RowBytes >= 128) {
        return chunk ^ (row & 7);
    } else {
        constexpr int kRowsPerLine = 128 / RowBytes;
        return chunk ^ ((row / kRowsPerLine) & (kChunks - 1));
    }
}

__device__ __forceinline__ int sliced_k_x_swizzle(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union SlicedKBf16Bits {
    __nv_bfloat162 pair;
    unsigned bits;
};

// Both halves hold 128 + v (v < 128 in the mantissa); subtracting 128 + bias leaves v - bias exactly.
__device__ __forceinline__ unsigned sliced_k_unbias(unsigned biased, unsigned bias) {
    SlicedKBf16Bits value;
    value.bits = biased;
    SlicedKBf16Bits offset;
    offset.bits = bias;
    SlicedKBf16Bits result;
    result.pair = __hsub2_rn(value.pair, offset.pair);
    return result.bits;
}

// A code pair as two BF16 integers. `code` holds the pair's packed byte (low nibble the even
// value) in bits 0-7, `high` the pair's high bits in its low bits; higher bits are ignored.
template <SlicedKFormat Format>
__device__ __forceinline__ unsigned sliced_k_pair(unsigned code, unsigned high) {
    if constexpr (Format == SlicedKFormat::Q4) {
        // q = (nibble ^ 8) - 8; 136 = 128 + 8.
        constexpr unsigned kBias = 0x43084308u;
        (void)high;
        return sliced_k_unbias(((code & 0x0fu) | ((code & 0xf0u) << 12)) ^ kBias, kBias);
    } else if constexpr (Format == SlicedKFormat::Q5) {
        // q = ((nibble | fifth bit << 4) ^ 16) - 16, the fifth bits in high bits 0 and 1; 144 = 128 + 16.
        constexpr unsigned kBias = 0x43104310u;
        return sliced_k_unbias(((code & 0x0fu) | ((high & 1u) << 4) | ((code & 0xf0u) << 12) |
                                ((high & 2u) << 19)) ^
                                   kBias,
                               kBias);
    } else {
        // q = ((nibble | two high bits << 4) ^ 32) - 32, the high bits in high bits 0-1 and 2-3;
        // 160 = 128 + 32.
        constexpr unsigned kBias = 0x43204320u;
        return sliced_k_unbias(((code & 0x0fu) | ((high & 3u) << 4) | ((code & 0xf0u) << 12) |
                                ((high & 0x0cu) << 18)) ^
                                   kBias,
                               kBias);
    }
}

// cp.async.wait_group with the ring's runtime depth.
__device__ __forceinline__ void sliced_k_wait(int pending) {
    if (pending <= 0) {
        cp_wait<0>();
    } else if (pending == 1) {
        cp_wait<1>();
    } else if (pending == 2) {
        cp_wait<2>();
    } else {
        cp_wait<3>();
    }
}

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
    sliced_k_pipelined_kernel(SlicedKOperands operands, int stages, Output output,
                              Epilogue epilogue, int token_begin, RowPolicy row_policy) {
    constexpr SlicedKFormat F = Schedule::kFormat;
    // Like the release Q4 kernel, wait for the producer first and let dependents launch once the
    // K loop is done; the release Q5/Q6 kernels have no programmatic dependency.
    if constexpr (F == SlicedKFormat::Q4) { pdl::enter_streaming(); }
    constexpr int R  = Schedule::kBlockRows;
    constexpr int T  = Schedule::kBlockTokens;
    constexpr int C  = Schedule::kTokenCapacity;
    constexpr int W  = Schedule::kKWarps;
    constexpr int MT = Schedule::kMmaRows;
    constexpr int NT = Schedule::kMmaTokens;
    constexpr int kCodeRowBytes  = Schedule::kCodeRowBytes;
    constexpr int kHighRowBytes  = Schedule::kHighRowBytes;
    constexpr int kHighGroup     = Schedule::kHighGroupBytes;
    constexpr int kScaleRowBytes = Schedule::kScaleRowBytes;

    extern __shared__ __align__(16) unsigned char sliced_k_shared[];
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int row_begin  = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;
    const int token0     = static_cast<int>(blockIdx.y) * C;
    const int k          = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int row_groups = k / 64;
    const int tiles      = row_groups / W;
    const int tokens     = operands.tokens;

    const auto slot_base = [&](int slot) {
        return sliced_k_shared + static_cast<std::size_t>(slot) * Schedule::kSlotBytes;
    };

    const auto stage_inputs = [&](int slot, int tile) {
        unsigned char* const base  = slot_base(slot);
        unsigned char* const high  = base + Schedule::kCodeBytes;
        unsigned char* const scale = high + Schedule::kHighBytes;
        auto* const x_tile = reinterpret_cast<__nv_bfloat16*>(scale + Schedule::kScaleBytes);
        const int first_group = tile * W;
        constexpr int kCodeChunks = kCodeRowBytes / 16;
        for (int item = tid; item < R * kCodeChunks; item += Schedule::kThreads) {
            const int row   = item / kCodeChunks;
            const int chunk = item % kCodeChunks;
            const std::int64_t group0 =
                std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups + first_group;
            cp_async<16, Schedule::kWeightCache>(
                base + row * kCodeRowBytes + sliced_k_chunk<kCodeRowBytes>(row, chunk) * 16,
                operands.codes + group0 * 32 + chunk * 16);
        }
        if constexpr (kHighGroup > 0) {
            constexpr int kHighChunks = kHighRowBytes / 16;
            for (int item = tid; item < R * kHighChunks; item += Schedule::kThreads) {
                const int row   = item / kHighChunks;
                const int chunk = item % kHighChunks;
                const std::int64_t group0 =
                    std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups + first_group;
                cp_async<16, Schedule::kWeightCache>(
                    high + row * kHighRowBytes + sliced_k_chunk<kHighRowBytes>(row, chunk) * 16,
                    operands.high + group0 * kHighGroup + chunk * 16);
            }
        }
        for (int row = tid; row < R; row += Schedule::kThreads) {
            const std::int64_t group0 =
                std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups + first_group;
            cp_async<kScaleRowBytes>(scale + row * kScaleRowBytes, operands.scales + group0 * 2);
        }
        for (int item = lane; item < C * 8; item += 32) {
            const int token  = item / 8;
            const int k8     = (item % 8) * 8;
            const bool valid = token0 + token < tokens;
            const std::int64_t index =
                valid ? std::int64_t(token0 + token) * k + (first_group + warp) * 64 + k8 : 0;
            cp_async_zfill<16, Schedule::kActivationCache>(
                x_tile + (warp * T + token) * 64 + sliced_k_x_swizzle(token, k8), operands.x + index,
                valid ? 16 : 0);
        }
        cp_commit();
    };

    // Pairs j = 2ks and 2ks + 1 of fragment row `row` (code bytes lid + 4j of this warp's group).
    const auto pairs = [&](const unsigned char* base, int row, int ks, unsigned& first,
                           unsigned& second) {
        const uint2 code = load_vec<uint2>(
            base + row * kCodeRowBytes +
            sliced_k_chunk<kCodeRowBytes>(row, warp * 2 + (ks >> 1)) * 16 + (ks & 1) * 8);
        const int code_shift = 8 * lid;
        unsigned high_first  = 0;
        unsigned high_second = 0;
        if constexpr (F == SlicedKFormat::Q5) {
            // High byte j holds the fifth bits of code byte lid + 4j at bits 2 lid and 2 lid + 1.
            const unsigned char* high = base + Schedule::kCodeBytes;
            const int byte            = warp * 8 + (ks >> 1) * 4;
            const unsigned word       = load_vec<unsigned>(
                high + row * kHighRowBytes + sliced_k_chunk<kHighRowBytes>(row, byte >> 4) * 16 +
                (byte & 15));
            const int shift = 16 * (ks & 1) + 2 * lid;
            high_first      = word >> shift;
            high_second     = word >> (shift + 8);
        } else if constexpr (F == SlicedKFormat::Q6) {
            // High byte b / 2 holds code byte b's four high bits at (b & 1) x 4.
            const unsigned char* high = base + Schedule::kCodeBytes;
            const unsigned word       = load_vec<unsigned>(
                high + row * kHighRowBytes + sliced_k_chunk<kHighRowBytes>(row, warp) * 16 + ks * 4);
            const int shift = 8 * (lid >> 1) + 4 * (lid & 1);
            high_first      = word >> shift;
            high_second     = word >> (shift + 16);
        }
        first  = sliced_k_pair<F>(code.x >> code_shift, high_first);
        second = sliced_k_pair<F>(code.y >> code_shift, high_second);
    };

    float acc[MT][NT][4] = {};
    for (int slot = 0; slot < stages; ++slot) {
        if (slot < tiles) {
            stage_inputs(slot, slot);
        } else {
            cp_commit();
        }
    }
    int slot = 0;
#pragma unroll 1
    for (int tile = 0; tile < tiles; ++tile) {
        sliced_k_wait(stages - 1);
        __syncthreads();
        const unsigned char* const base = slot_base(slot);
        const auto* const scales        = reinterpret_cast<const std::uint16_t*>(
            base + Schedule::kCodeBytes + Schedule::kHighBytes);
        const auto* const x_tile = reinterpret_cast<const __nv_bfloat16*>(
            base + Schedule::kCodeBytes + Schedule::kHighBytes + Schedule::kScaleBytes);
        float group_acc[MT][NT][4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int r0 = mi * 16 + gid;
                unsigned a0, a1, a2, a3;
                pairs(base, r0, ks, a0, a2);
                pairs(base, r0 + 8, ks, a1, a3);
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int br = ni * 8 + (lane & 7);
                    const int bk = ks * 16 + (((lane >> 3) & 1) * 8);
                    unsigned b0, b1;
                    ldmatrix_x2(b0, b1,
                                smem_addr(x_tile + (warp * T + br) * 64 + sliced_k_x_swizzle(br, bk)));
                    mma_bf16(group_acc[mi][ni][0], group_acc[mi][ni][1], group_acc[mi][ni][2],
                             group_acc[mi][ni][3], a0, a1, a2, a3, b0, b1);
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const float top    = __half2float(__ushort_as_half(scales[(mi * 16 + gid) * W + warp]));
            const float bottom = __half2float(__ushort_as_half(scales[(mi * 16 + gid + 8) * W + warp]));
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                acc[mi][ni][0] = fmaf(group_acc[mi][ni][0], top, acc[mi][ni][0]);
                acc[mi][ni][1] = fmaf(group_acc[mi][ni][1], top, acc[mi][ni][1]);
                acc[mi][ni][2] = fmaf(group_acc[mi][ni][2], bottom, acc[mi][ni][2]);
                acc[mi][ni][3] = fmaf(group_acc[mi][ni][3], bottom, acc[mi][ni][3]);
            }
        }
        if (tile + stages < tiles) {
            // Only a reused slot needs a barrier here; the reduction's barrier covers the rest.
            __syncthreads();
            stage_inputs(slot, tile + stages);
        } else if (stages > 1) {
            cp_commit();
        }
        slot = slot + 1 == stages ? 0 : slot + 1;
    }
    if constexpr (F == SlicedKFormat::Q4) { pdl::trigger_dependents(); }
    cp_wait<0>();
    __syncthreads();

    auto* const partial      = reinterpret_cast<float*>(sliced_k_shared);
    const auto partial_at    = [&](int part, int mi, int ni) {
        return partial + (((part * MT + mi) * NT + ni) * 32 + lane) * 4;
    };
    constexpr bool kPairwise = Schedule::kPairwise;
    if (!kPairwise || (warp & 1) != 0) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                store_vec(partial_at(warp, mi, ni),
                          make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]));
            }
        }
    }
    __syncthreads();
    if constexpr (kPairwise) {
        if ((warp & 1) == 0) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const float4 partner = load_vec<float4>(partial_at(warp + 1, mi, ni));
                    acc[mi][ni][0] += partner.x;
                    acc[mi][ni][1] += partner.y;
                    acc[mi][ni][2] += partner.z;
                    acc[mi][ni][3] += partner.w;
                    if (warp != 0) {
                        store_vec(partial_at(warp, mi, ni),
                                  make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2],
                                              acc[mi][ni][3]));
                    }
                }
            }
        }
        __syncthreads();
    }
    if (warp == 0) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                float4 sum;
                if constexpr (kPairwise) {
                    sum = make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
                } else {
                    sum = load_vec<float4>(partial_at(0, mi, ni));
                }
#pragma unroll
                for (int part = kPairwise ? 2 : 1; part < W; part += kPairwise ? 2 : 1) {
                    const float4 value = load_vec<float4>(partial_at(part, mi, ni));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int top_row    = sliced_k_output_row(row_policy, row_begin, mi * 16 + gid);
                const int t0         = token0 + ni * 8 + 2 * lid;
                const int token_end  = token_begin + min(tokens, token0 + C);
                if constexpr (requires {
                                  epilogue.store_fragment(output, top_row, token_begin + t0, sum,
                                                          operands.rows, token_end);
                              }) {
                    // x/y belong to the fragment's top row, z/w to the row 8 below it.
                    epilogue.store_fragment(output, top_row, token_begin + t0, sum, operands.rows,
                                            token_end);
                } else {
                    const int bottom_row =
                        sliced_k_output_row(row_policy, row_begin, mi * 16 + gid + 8);
                    const auto store = [&](int row, int token, float value) {
                        if (token - token0 < C && token < tokens) {
                            const int global_token = token_begin + token;
                            output.store(row, global_token, epilogue.apply(row, global_token, value));
                        }
                    };
                    store(top_row, t0, sum.x);
                    store(top_row, t0 + 1, sum.y);
                    store(bottom_row, t0, sum.z);
                    store(bottom_row, t0 + 1, sum.w);
                }
            }
        }
    }
}

// ---- host side

// Opt-in dynamic shared memory for a kernel's deepest ring (once per kernel).
template <auto Kernel, std::size_t Bytes>
void sliced_k_allow_shared() {
    static const cudaError_t attribute = cudaFuncSetAttribute(
        Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(Bytes));
    CUDA_CHECK(attribute);
}

// The device's SMs (cached per thread and device).
inline int sliced_k_multiprocessors() {
    thread_local int device = -1;
    thread_local int sms    = 0;
    int current             = 0;
    CUDA_CHECK(cudaGetDevice(&current));
    if (current != device) {
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, current));
        device = current;
    }
    return sms;
}

// `kernel`'s resident CTAs per SM at `threads` threads and `shared` dynamic bytes (cached: the
// answer never changes for a kernel).
template <class Kernel>
int sliced_k_resident(Kernel kernel, int threads, std::size_t shared) {
    struct Entry {
        const void* kernel;
        std::size_t shared;
        int resident;
    };
    thread_local std::vector<Entry> cache;
    const void* key = reinterpret_cast<const void*>(kernel);
    for (const Entry& entry : cache) {
        if (entry.kernel == key && entry.shared == shared) { return entry.resident; }
    }
    int resident = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, threads, shared));
    cache.push_back({key, shared, resident});
    return resident;
}

// One launch at a chosen shape (block rows from the schedule, `stages` ring slots).
template <class Schedule, class Output, class Epilogue, class RowPolicy>
void launch_sliced_k_pipelined_shape(const SlicedKOperands& operands, int stages, Output output,
                                     Epilogue epilogue, cudaStream_t stream, RowPolicy row_policy,
                                     pdl::Dependency dependency) {
    constexpr auto kernel = &sliced_k_pipelined_kernel<Schedule, Output, Epilogue, RowPolicy>;
    sliced_k_allow_shared<kernel, Schedule::kOptInBytes>();
    const std::size_t shared = Schedule::shared_bytes(stages);
    if (stages < 1 || stages > Schedule::kMaxStages || shared > Schedule::kOptInBytes) {
        throw std::invalid_argument("sliced-K pipelined: ring depth out of range");
    }
    for_each_token_slice(operands.tokens, Schedule::kTokenCapacity, [&](int offset, int count) {
        const dim3 grid(div_up(row_policy.output_rows(operands.rows), RowPolicy::kOutputRowsPerCta),
                        div_up(count, Schedule::kTokenCapacity));
        SlicedKOperands slice = operands;
        slice.x += static_cast<std::int64_t>(offset) * operands.k;
        slice.tokens = count;
        CUDA_CHECK(pdl::launch_with(dependency,
                                    {grid, dim3(Schedule::kThreads), shared, stream}, kernel,
                                    slice, stages, output, epilogue, offset, row_policy));
    });
}

// Whether a row policy names its form at other block rows (the one-wave route may widen it).
template <class RowPolicy, int Rows>
concept SlicedKRebindable = requires { typename RowPolicy::template Rebind<Rows>; };

// The pipelined route of one release schedule's call: the candidate shapes (the schedule's block
// rows, and under the one-wave route or a forced shape twice as many, each at ring depths 1-4 that
// fit the opt-in limit), the shape choose_sliced_k_shape picks (or INFERNIX_TMP_GW_SHAPE's where
// admissible), and the launch. Callers check that the weights are full first.
template <class Family, class Output, class Epilogue, class RowPolicy>
void launch_sliced_k_pipelined(const SlicedKOperands& operands, Output output, Epilogue epilogue,
                               cudaStream_t stream, RowPolicy row_policy,
                               pdl::Dependency dependency, SlicedKRoute route) {
    constexpr int kBase = Family::kBaseRows;
    constexpr int kWide = 2 * kBase;
    static_assert(Family::template fits<kBase>(), "the base schedule must fit the pipelined kernel");
    const SlicedKShape forced = forced_sliced_k_shape();
    const int sms             = sliced_k_multiprocessors();

    std::array<SlicedKCandidate, 8> candidates{};
    std::size_t count = 0;
    const auto consider = [&]<int Rows, class Policy>(const Policy& policy) {
        using Schedule = typename Family::template Schedule<Rows>;
        if (operands.rows % Rows != 0) { return; }
        constexpr auto kernel = &sliced_k_pipelined_kernel<Schedule, Output, Epilogue, Policy>;
        sliced_k_allow_shared<kernel, Schedule::kOptInBytes>();
        const std::int64_t ctas =
            std::int64_t(div_up(policy.output_rows(operands.rows), Policy::kOutputRowsPerCta)) *
            div_up(std::min<std::int64_t>(operands.tokens,
                                          std::int64_t(Schedule::kTokenCapacity) * kCudaGridYLimit),
                   std::int64_t(Schedule::kTokenCapacity));
        for (int stages = 1; stages <= Schedule::kMaxStages; ++stages) {
            const std::size_t bytes = Schedule::shared_bytes(stages);
            if (bytes > Schedule::kOptInBytes) { continue; }
            candidates[count++] = {{Rows, stages},
                                   ctas,
                                   sliced_k_resident(kernel, Schedule::kThreads, bytes),
                                   Schedule::kWeightSlotBytes};
        }
    };
    consider.template operator()<kBase>(row_policy);
    constexpr bool kWidens = kWide <= 64 && SlicedKRebindable<RowPolicy, kWide> &&
                             Family::template fits<kWide>();
    if constexpr (kWidens) {
        if (route == SlicedKRoute::OneWave || forced.block_rows == kWide) {
            consider.template operator()<kWide>(typename RowPolicy::template Rebind<kWide>{});
        }
    }

    std::size_t chosen = count;
    if (forced.block_rows != 0) {
        for (std::size_t index = 0; index < count; ++index) {
            if (candidates[index].shape == forced && candidates[index].resident > 0) { chosen = index; }
        }
    }
    if (chosen == count) {
        chosen = choose_sliced_k_shape(std::span<const SlicedKCandidate>(candidates.data(), count), sms);
    }
    const SlicedKShape shape = candidates[chosen].shape;
    if constexpr (kWidens) {
        if (shape.block_rows == kWide) {
            launch_sliced_k_pipelined_shape<typename Family::template Schedule<kWide>>(
                operands, shape.stages, output, epilogue, stream,
                typename RowPolicy::template Rebind<kWide>{}, dependency);
            return;
        }
    }
    launch_sliced_k_pipelined_shape<typename Family::template Schedule<kBase>>(
        operands, shape.stages, output, epilogue, stream, row_policy, dependency);
}

} // namespace infernix::ops::detail
