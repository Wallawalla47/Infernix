#pragma once

// The limb planes of W4A16 activations (design §16.2.1), shared by the routes that read them: the wide route's
// grouped GEMM and its encoders (wide_expert_a16.cuh, wide_expert.cu), the dense rg16 projections (rg16_linear.cu)
// and the tensor-core narrow route (narrow_a16_mma.cuh). One 16-element block b of a row is the canonical X
// (canon::a16_value; zero in a non-finite column) as H1 = X >> 14, L1 = (X >> 3) & 255 and H0 | L0 << 4 with
// H0 = (X >> 11) & 7, L0 = X & 7, in K tile b / 4 (kA16LimbTileBytes: 64 H1, 64 L1, 64 H0 | L0 << 4) at block
// offset 16 (b % 4), so X = 2^14 H1 + 2^11 H0 + 8 L1 + L0.

#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_tile.cuh"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::offloaded_moe::wide {

// s += p * w in int64 (one IMAD.WIDE on the accumulator's own registers).
__device__ __forceinline__ void fold(long long& s, int p, int w) {
    asm("mad.wide.s32 %0, %1, %2, %0;" : "+l"(s) : "r"(p), "r"(w));
}

template <class Byte>
__device__ __forceinline__ uint4 pack16(const Byte (&b)[16]) {
    unsigned w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i])) |
               (static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i + 1])) << 8) |
               (static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i + 2])) << 16) |
               (static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i + 3])) << 24);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ void store_a16_block(std::uint8_t* row, int b, const std::uint16_t (&v)[16], int emax) {
    std::uint8_t high[16], low[16], small[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const std::int32_t x = emax == canon::kA16NonFinite ? 0 : canon::a16_value(v[j], emax);
        high[j]  = static_cast<std::uint8_t>(x >> 14);
        low[j]   = static_cast<std::uint8_t>((x >> 3) & 255);
        small[j] = static_cast<std::uint8_t>(((x >> 11) & 7) | ((x & 7) << 4));
    }
    std::uint8_t* at                              = row + (b / 4) * kA16LimbTileBytes + 16 * (b % 4);
    *reinterpret_cast<uint4*>(at)                 = pack16(high);
    *reinterpret_cast<uint4*>(at + kA16KTile)     = pack16(low);
    *reinterpret_cast<uint4*>(at + 2 * kA16KTile) = pack16(small);
}

// Half `half` (elements 8 half .. 8 half + 7) of block b, for producers that hold a block on a thread pair: the same
// bytes store_a16_block writes for them.
__device__ __forceinline__ void store_a16_half(std::uint8_t* row, int b, int half, const std::uint16_t (&v)[8], int emax) {
    std::uint32_t high[2] = {0U, 0U}, low[2] = {0U, 0U}, small[2] = {0U, 0U};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const std::int32_t x = emax == canon::kA16NonFinite ? 0 : canon::a16_value(v[j], emax);
        const int word = j / 4, shift = 8 * (j % 4);
        high[word] |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(x >> 14)) << shift;
        low[word] |= static_cast<std::uint32_t>(static_cast<std::uint8_t>((x >> 3) & 255)) << shift;
        small[word] |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(((x >> 11) & 7) | ((x & 7) << 4))) << shift;
    }
    std::uint8_t* at = row + (b / 4) * kA16LimbTileBytes + 16 * (b % 4) + 8 * half;
    *reinterpret_cast<uint2*>(at)                 = make_uint2(high[0], high[1]);
    *reinterpret_cast<uint2*>(at + kA16KTile)     = make_uint2(low[0], low[1]);
    *reinterpret_cast<uint2*>(at + 2 * kA16KTile) = make_uint2(small[0], small[1]);
}

__device__ __forceinline__ void unpack16(const uint4 v0, const uint4 v1, std::uint16_t (&v)[16]) {
    const unsigned words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        v[2 * w]     = static_cast<std::uint16_t>(words[w] & 0xFFFFU);
        v[2 * w + 1] = static_cast<std::uint16_t>(words[w] >> 16);
    }
}

// The largest magnitude's bits of 16 BF16 values (their column-exponent key, canon::a16_column_exponent).
__device__ __forceinline__ unsigned a16_key(const std::uint16_t (&v)[16]) {
    unsigned key = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) { key = max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    return key;
}

__device__ __forceinline__ int a16_emax(unsigned key) {
    return key >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(key >> 7), 1);
}

// ---------------------------------------------------------------------------- the tensor-core unit core

// How a unit's B operand (a column's limb row) is read: shared memory; the non-coherent path (rows complete before the
// kernel's dependency wait); or L2 (rows another grid of the same call writes, read after an acquire).
enum class LimbLoad { Shared, Nc, Cg };

template <LimbLoad L>
__device__ __forceinline__ std::uint32_t limb_word(const std::uint8_t* p) {
    if constexpr (L == LimbLoad::Nc) {
        return __ldg(reinterpret_cast<const unsigned*>(p));
    } else if constexpr (L == LimbLoad::Cg) {
        return __ldcg(reinterpret_cast<const unsigned*>(p));
    } else {
        return *reinterpret_cast<const std::uint32_t*>(p);
    }
}

// D = A B on int8 tensor cores with signed A and unsigned B (ggml::tile::mma_k32 is signed times signed).
__device__ __forceinline__ void mma_k32_su(int (&d)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

// One NVFP4 rg16 unit's contribution to thread (g, t)'s exact sums (design §16.2.1: P_b with two k32 MMAs, then
// S += P_b * e4m3_scaled(s_b)): its A fragment from the unit's word 32 t + 4 g (rows g and g + 8, elements 4 t ..
// 4 t + 3), B from column g's limb words t of the unit's block (`limbs`: that column's limb row at a16_block_offset),
// the rows' scales through `scaled` (the 256-entry e4m3_scaled table). Sums: rows g, g + 8 x columns 2 t, 2 t + 1.
template <LimbLoad L>
__device__ __forceinline__ void a16_unit(const std::uint8_t* unit, const std::uint8_t* limbs, const std::int32_t* scaled,
                                         int g, int t, long long (&s)[4]) {
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(unit + 32 * t + 4 * g);
    const std::uint32_t c_lo = canon::e2m1_x2_quad(word & 0x0F0F0F0FU);        // row g
    const std::uint32_t c_hi = canon::e2m1_x2_quad((word >> 4) & 0x0F0F0F0FU); // row g + 8
    const int a0 = static_cast<int>((c_lo << 3) & 0xF8F8F8F8U); // 8 c: each byte's own bits
    const int a1 = static_cast<int>((c_hi << 3) & 0xF8F8F8F8U);
    const std::uint32_t h1    = limb_word<L>(limbs);
    const std::uint32_t l1    = limb_word<L>(limbs + kA16KTile);
    const std::uint32_t small = limb_word<L>(limbs + 2 * kA16KTile);
    int hi[4], lo[4];
    ::infernix::ggml::tile::mma_k32(hi, a0, a1, static_cast<int>(c_lo), static_cast<int>(c_hi), static_cast<int>(h1),
                                    static_cast<int>(small & 0x07070707U));
    mma_k32_su(lo, a0, a1, static_cast<int>(c_lo), static_cast<int>(c_hi), static_cast<int>(l1),
               static_cast<int>((small >> 4) & 0x07070707U));
    const int w_lo = scaled[unit[128 + g]], w_hi = scaled[unit[136 + g]];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // P = 2048 hi + lo in wrapping arithmetic: its true value fits int32.
        const int p = static_cast<int>((static_cast<unsigned>(hi[i]) << 11) + static_cast<unsigned>(lo[i]));
        fold(s[i], p, i < 2 ? w_lo : w_hi);
    }
}

// The limb words of block b for thread t within a limb row.
__device__ __forceinline__ int a16_block_offset(int b, int t) { return (b / 4) * kA16LimbTileBytes + 16 * (b % 4) + 4 * t; }

} // namespace infernix::ops::offloaded_moe::wide
