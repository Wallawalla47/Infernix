#pragma once

// The native block-scaled W4A4 core of w4a4_sparse_moe's Inkling 2 decode and verify kernels (single_kernel,
// gate_up_a4_kernel, down_a4_kernel): 144-byte nvfp4_expert_rg16_v1 units (16 rows x one 16-element block, the codes
// then a scale word a row) on the m16n8k64 e2m1 MMA with UE4M3 block scales, accumulating in FP32. The exact int64 core
// (a4_mma.cuh) is the other choice (w4a4_moe::Nvfp4Core::Exact). The E4M3 scale words decode natively as
// canon::e4m3_value for every finite word (0x00-0x7E; the op test checks every pair).
//
// One call is one 64-element K step of a 16-row group: four consecutive units (blocks 4 s .. 4 s + 3) times up to eight
// columns, accumulated into c, the D fragment ((row g, column 2 t), (g, 2 t + 1), (g + 8, 2 t), (g + 8, 2 t + 1), as
// a4_mma_unit's sums). The weights are A: the MMA's K slots 0-15, 16-31, 32-47, 48-63 take blocks 0, 2, 1, 3, so a
// thread's words come from units 2 (t / 2) and 2 (t / 2) + 1 at four distinct bank ranges (offloaded_sparse_moe's
// rg16_fragments, there as B), and within a block half a thread's eight slots pair record nibbles k and k + 4: an A word
// is a masked OR of two code words, rows g in the low nibbles, g + 8 in the high ones. A column's codes take the same
// slot order (a4_native_order of each natural code word, once when they are staged). A chain of steps is an FP32 chain:
// its bits are those of its steps in order, the same in every kernel that runs the same chain.

#include "ops/common/mma.cuh"

#include <cstdint>

namespace infernix::ops {

// A natural code word (elements 8 h .. 8 h + 7 of a block, element 2 j in byte j's low nibble) in the MMA's slot
// order: byte i holds element 8 h + i (low nibble) and 8 h + 4 + i (high nibble).
__device__ __forceinline__ std::uint32_t a4_native_order(std::uint32_t w) {
    const auto spread = [](std::uint32_t v) { // nibble k of v's low 16 bits into byte k's low nibble
        return (v & 0xFU) | ((v & 0xF0U) << 4) | ((v & 0xF00U) << 8) | ((v & 0xF000U) << 12);
    };
    return spread(w & 0xFFFFU) | (spread(w >> 16) << 4);
}

// `units`: the step's four 144-byte units; `column`: column g's codes (in slot order) at the step's first block;
// `column_scales`: column g's four E4M3 scale words of the step's blocks, natural order (block 4 s in byte 0).
__device__ __forceinline__ void a4_native_step(const std::uint8_t* units, const std::uint8_t* column,
                                               std::uint32_t column_scales, int g, int t, float (&c)[4]) {
    constexpr int kU = 144;
    const int pair = 2 * (t >> 1), half = t & 1;
    const std::uint8_t* u0 = units + pair * kU + 64 * half + 4 * g;
    const std::uint8_t* u1 = u0 + kU;
    const std::uint32_t w0 = *reinterpret_cast<const std::uint32_t*>(u0);
    const std::uint32_t w1 = *reinterpret_cast<const std::uint32_t*>(u0 + 32);
    const std::uint32_t v0 = *reinterpret_cast<const std::uint32_t*>(u1);
    const std::uint32_t v1 = *reinterpret_cast<const std::uint32_t*>(u1 + 32);
    const unsigned a0 = (w0 & 0x0F0F0F0FU) | ((w1 & 0x0F0F0F0FU) << 4); // row g, slots 8 t .. 8 t + 7
    const unsigned a1 = ((w0 >> 4) & 0x0F0F0F0FU) | (w1 & 0xF0F0F0F0U); // row g + 8
    const unsigned a2 = (v0 & 0x0F0F0F0FU) | ((v1 & 0x0F0F0F0FU) << 4); // row g, slots 32 + 8 t ..
    const unsigned a3 = ((v0 >> 4) & 0x0F0F0F0FU) | (v1 & 0xF0F0F0F0U); // row g + 8
    // The A scales: thread (g, t) supplies row g + 8 (t & 1) (scale thread selector 0), its scale words (byte 128 +
    // row) of units 0, 2, 1, 3 in slot-block order.
    const int r = 128 + g + 8 * half;
    const unsigned sfa = static_cast<unsigned>(units[r]) | (static_cast<unsigned>(units[2 * kU + r]) << 8) |
                         (static_cast<unsigned>(units[kU + r]) << 16) | (static_cast<unsigned>(units[3 * kU + r]) << 24);
    const unsigned b0  = *reinterpret_cast<const std::uint32_t*>(column + 8 * pair + 4 * half);
    const unsigned b1  = *reinterpret_cast<const std::uint32_t*>(column + 8 * (pair + 1) + 4 * half);
    const unsigned sfb = __byte_perm(column_scales, 0U, 0x3120U); // blocks 0, 2, 1, 3
    mma_nvfp4_e4m3(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1, sfa, sfb);
}

} // namespace infernix::ops
