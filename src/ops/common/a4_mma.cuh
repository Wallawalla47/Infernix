#pragma once

// The exact W4A4 tensor-core core shared by the routed-expert narrow routes (offloaded_sparse_moe's Flash-Next banks,
// w4a4_sparse_moe's Inkling 2 banks): one 144-byte nvfp4_expert_rg16_v1 unit (16 rows x one 16-element block) times
// up to eight columns of A4 codes. Per block the s8 m16n8k16 MMA of doubled E2M1 codes gives P_b exactly
// (|P_b| <= 16 * 144); the thread adds P_b times the row's and the column's scaled E4M3 scales (e4m3_scaled, x 2^9 each)
// into int64 sums, so a row sum S = sum_b P_b Sw_b Sa_b is exact and S * 2^-20 is the exact dot product of the decoded
// values (design docs/maintainer/qwen3_8-flash-next-design.md §16.2).

#include "ops/common/canonical_math.h"

#include <cstdint>

namespace infernix::ops {

// D = A B on int8 tensor cores, k16: rows g and g + 8 of A (a0, a1) times column g of B (b0), K = 4 t .. 4 t + 3.
__device__ __forceinline__ void a4_mma_k16(int (&d)[4], int a0, int a1, int b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// The doubled codes of a block's elements 4 t .. 4 t + 3 from its packed bytes 2 t and 2 t + 1 (element 2 j in byte j's
// low nibble, 2 j + 1 in its high one).
__device__ __forceinline__ int a4_x_quad(std::uint32_t pair) {
    const std::uint32_t spread = (pair & 0xFU) | ((pair & 0xF0U) << 4) | ((pair & 0xF00U) << 8) | ((pair & 0xF000U) << 12);
    return static_cast<int>(canon::e2m1_x2_quad(spread));
}

// One unit's contribution to thread (g, t)'s sums: rows g and g + 8 of the unit against column g's codes `b`, the
// rows' weight scales through `scaled` (a4_mma_fill_scaled's table), the columns 2 t and 2 t + 1's scaled activation
// scales sa0 and sa1. s[0..3]: (row g, column 2t), (row g, 2t + 1), (row g + 8, 2t), (row g + 8, 2t + 1).
__device__ __forceinline__ void a4_mma_unit(const std::uint8_t* unit, int b, const std::int32_t* scaled, int sa0, int sa1,
                                            int g, int t, long long (&s)[4]) {
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(unit + 32 * t + 4 * g);
    const int a0 = static_cast<int>(canon::e2m1_x2_quad(word & 0x0F0F0F0FU));        // row g
    const int a1 = static_cast<int>(canon::e2m1_x2_quad((word >> 4) & 0x0F0F0F0FU)); // row g + 8
    int d[4];
    a4_mma_k16(d, a0, a1, b);
    const int w_lo = scaled[unit[128 + g]], w_hi = scaled[unit[136 + g]];
    // |P_b| <= 2,304 and Sw < 2^18: P_b * Sw fits int32 (per block).
    s[0] += static_cast<long long>(d[0] * w_lo) * sa0;
    s[1] += static_cast<long long>(d[1] * w_lo) * sa1;
    s[2] += static_cast<long long>(d[2] * w_hi) * sa0;
    s[3] += static_cast<long long>(d[3] * w_hi) * sa1;
}

// The 256-entry table of e4m3_scaled over every scale word (shared memory, filled by the CTA's threads).
__device__ __forceinline__ void a4_mma_fill_scaled(std::int32_t* scaled, int threads) {
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += threads) { scaled[i] = canon::e4m3_scaled(static_cast<unsigned>(i)); }
}

} // namespace infernix::ops
