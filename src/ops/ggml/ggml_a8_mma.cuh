#pragma once

// Integer tensor-core products of the wide 8-bit activation routes (`--a8-activations`: LOSSY, opt-in;
// ops/ggml/ggml_a8_tile.cuh for GGUF weights, ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh for NVFP4 W4A16
// experts), turned into binary32 terms without an integer-to-float conversion instruction.
//
// An int32 MMA whose accumulators start from the bits of M = 1.5 * 2^23 (kMagicBits) returns M + P for the
// exact product sum P. For |P| < 2^22 those bits are the binary32 value M + P exactly (the binade [2^23, 2^24)
// has a spacing of 1), so for a scale w whose product with M is exact (w of at most 21 significant bits),
//
//   scaled_term(M + P, w, -M w) = fma(M + P, w, -M w) = fl32(P * w)
//
// rounds once, like a correctly rounded conversion followed by a correctly rounded product. Every caller bounds
// |P| < 2^19 (32 int8 codes of magnitude <= 127 times 32 activation codes of magnitude <= 127).

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ggml::a8 {

inline constexpr int kMagicBits     = 0x4B400000; // 1.5 * 2^23 as binary32
inline constexpr float kMagic       = 12582912.0F;
inline constexpr float kMinusMagic  = -12582912.0F;

// d = a * b + M over one 16x8x32 int8 tile (s8 x s8, exact in int32).
__device__ __forceinline__ void mma_k32_magic(int (&d)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kMagicBits));
}

// d = a * b + M over one 16x8x16 int8 tile (s8 x s8, exact in int32).
__device__ __forceinline__ void mma_k16_magic(int (&d)[4], unsigned a0, unsigned a1, unsigned b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(kMagicBits));
}

// fl32(P * w) from an accumulator holding M + P (|P| < 2^22) and minus_mw = -M * w (exact).
__device__ __forceinline__ float scaled_term(int magic_p, float w, float minus_mw) {
    return __fmaf_rn(__int_as_float(magic_p), w, minus_mw);
}

} // namespace infernix::ggml::a8
