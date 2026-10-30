#pragma once

// The limb planes of W4A16 activations (design §16.2.1), shared by the routes that read them: the wide route's
// grouped GEMM and its encoders (wide_expert_a16.cuh, wide_expert.cu), the dense rg16 projections (rg16_linear.cu)
// and the tensor-core narrow route (narrow_a16_mma.cuh). One 16-element block b of a row is the canonical X
// (canon::a16_value; zero in a non-finite column) as H1 = X >> 14, L1 = (X >> 3) & 255 and H0 | L0 << 4 with
// H0 = (X >> 11) & 7, L0 = X & 7, in K tile b / 4 (kA16LimbTileBytes: 64 H1, 64 L1, 64 H0 | L0 << 4) at block
// offset 16 (b % 4), so X = 2^14 H1 + 2^11 H0 + 8 L1 + L0.

#include "ops/common/canonical_math.h"
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

} // namespace infernix::ops::offloaded_moe::wide
