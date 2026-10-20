#pragma once

// Exact chunk decoders for the reference route's projections (include/infernix/ops/precise_linear.h).
// Each decodes the 8 weights [8 c, 8 c + 8) of one row for precise_dot.cuh, exactly (the format's
// logical weight: tests/ops/quantized_weight.h logical_weight_fp64; GGUF blocks as ggml_exact.h):
// as binary32 weights (Products::Exact: at most 16 significant bits; Products::Split: any binary32
// value), or as integer codes with their group's scale and offset (Products::Grouped).

#include "core/weight.h"
#include "ops/ggml/ggml_exact.h"
#include "ops/precise/precise_dot.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace infernix::ops::precise {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float bf16_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ double bf16_value(std::uint16_t bits) { return bf16_float(bits); }

__device__ __forceinline__ float f16_float(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }

// E4M3FN (no infinities, S.1111.111 is NaN, subnormals m * 2^-9) through the hardware conversion to
// binary16, which holds every E4M3FN value exactly.
__device__ __forceinline__ float e4m3_float(std::uint32_t word) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(word), __NV_E4M3);
    return __half2float(__half(h));
}

// E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives), its binary32 bits built without a branch.
__device__ __forceinline__ float e2m1_float(std::uint32_t nibble) {
    const std::uint32_t m = nibble & 7U, e = m >> 1;
    const std::uint32_t bits = e != 0 ? ((e + 126U) << 23) | ((m & 1U) << 22) : (m != 0 ? 0x3F000000U : 0U);
    return __uint_as_float(bits | ((nibble & 8U) << 28));
}

// The members of a decoder that is not Grouped besides kProducts (decoders stay aggregates).
#define INFERNIX_PRECISE_UNGROUPED                                                                                     \
    static constexpr int kGroupChunks = kDotBlockChunks;                                                               \
    static constexpr bool kOffset     = false;

// Contiguous BF16 rows of up to kMaxSegments matrices (a BF16 Weight [N, K], or projection_fp32's
// [K, N_i] tensors), the rows of segment s at outputs [begin[s], begin[s + 1]).
struct Bf16Rows {
    INFERNIX_PRECISE_UNGROUPED
    static constexpr Products kProducts = Products::Exact;
    static constexpr int kMaxSegments   = 8;
    const bf16* data[kMaxSegments];
    int begin[kMaxSegments + 1];
    int count;
    int k;
    __device__ void decode(int row, int chunk, Weights8& w) const {
        const bf16* base = data[0];
        int first        = 0;
#pragma unroll
        for (int s = 1; s < kMaxSegments; ++s) {
            if (s < count && row >= begin[s]) {
                base  = data[s];
                first = begin[s];
            }
        }
        const uint4 packed = reinterpret_cast<const uint4*>(base + static_cast<std::size_t>(row - first) * k)[chunk];
        const auto* v      = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) { w.w[j] = bf16_float(v[j]); }
    }
};

// Row-split grouped integers (q2/q4/q5/q6 per 64, q8 per 32): code times its group's FP16 scale, exact
// in binary32 (at most 7 + 11 bits; q2-q6 at most 16, so their products are exact, q8's are split).
// Row r's codes at r * code_row bytes (q8: one byte each; q2: four per byte; q4-q6: two nibbles per
// byte, low first), q5/q6 high bits at r * high_row (q5: bit i of byte i / 8 per group of 8 bytes;
// q6: bits 2i, 2i + 1 per group of 16 bytes), FP16 scales at r * groups.
template <QType Q>
struct RowSplit {
    INFERNIX_PRECISE_UNGROUPED
    static constexpr int kBits  = Q == QType::Q2_G64_FP16   ? 2
                                  : Q == QType::Q4_G64_FP16 ? 4
                                  : Q == QType::Q5_G64_FP16 ? 5
                                  : Q == QType::Q6_G64_FP16 ? 6
                                                            : 8;
    static constexpr Products kProducts = kBits == 8 ? Products::Split : Products::Exact;
    static constexpr int kGroup         = kBits == 8 ? 32 : 64;
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const __half* scales;
    int padded_k;
    __device__ void decode(int row, int chunk, Weights8& w) const {
        const int groups = padded_k / kGroup, k0 = 8 * chunk, group = k0 / kGroup, lane = k0 % kGroup;
        const float scale = __half2float(scales[static_cast<std::size_t>(row) * groups + group]);
        int code[8];
        if constexpr (kBits == 8) {
            const uint2 packed = *reinterpret_cast<const uint2*>(codes + static_cast<std::size_t>(row) * padded_k + k0);
            const auto* v      = reinterpret_cast<const std::int8_t*>(&packed);
#pragma unroll
            for (int j = 0; j < 8; ++j) { code[j] = v[j]; }
        } else if constexpr (kBits == 2) {
            const std::uint16_t packed =
                *reinterpret_cast<const std::uint16_t*>(codes + static_cast<std::size_t>(row) * (padded_k / 4) + k0 / 4);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int u = (packed >> (2 * j)) & 3;
                code[j]     = u >= 2 ? u - 4 : u;
            }
        } else {
            const std::uint32_t packed =
                *reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::size_t>(row) * (padded_k / 2) + k0 / 2);
            std::uint32_t hi = 0;
            if constexpr (kBits == 5) {
                hi = high[static_cast<std::size_t>(row) * groups * 8 + group * 8 + lane / 8];
            } else if constexpr (kBits == 6) {
                hi = *reinterpret_cast<const std::uint16_t*>(high + static_cast<std::size_t>(row) * groups * 16 + group * 16 +
                                                             lane / 4);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                std::uint32_t u = (packed >> (4 * j)) & 0xFU;
                if constexpr (kBits == 5) { u |= ((hi >> j) & 1U) << 4; }
                if constexpr (kBits == 6) { u |= ((hi >> (2 * j)) & 3U) << 4; }
                code[j] = (u & (1U << (kBits - 1))) != 0 ? static_cast<int>(u) - (1 << kBits) : static_cast<int>(u);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) { w.w[j] = __fmul_rn(static_cast<float>(code[j]), scale); }
    }
};

// Row-scaled FP8: E4M3FN code times its row's BF16 scale (exact: 4 x 8 significant bits).
struct Fp8Row {
    INFERNIX_PRECISE_UNGROUPED
    static constexpr Products kProducts = Products::Exact;
    const std::uint8_t* codes;
    const std::uint16_t* scales;
    int k;
    __device__ void decode(int row, int chunk, Weights8& w) const {
        const uint2 packed = *reinterpret_cast<const uint2*>(codes + static_cast<std::size_t>(row) * k + 8 * chunk);
        const auto* v      = reinterpret_cast<const std::uint8_t*>(&packed);
        const float scale  = bf16_float(scales[row]);
#pragma unroll
        for (int j = 0; j < 8; ++j) { w.w[j] = __fmul_rn(e4m3_float(v[j]), scale); }
    }
};

// NVFP4 with 16-element E4M3 scales in the K16M128x4 swizzle and a per-matrix divisor: the logical
// weight fl32(e2m1 * scale / divisor), formed in binary64 as the fixture's logical_weight_fp64 does.
struct Nvfp4 {
    INFERNIX_PRECISE_UNGROUPED
    static constexpr Products kProducts = Products::Split;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    double divisor;
    int k;
    __device__ void decode(int row, int chunk, Weights8& w) const {
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::size_t>(row) * (k / 2) + 4 * chunk);
        const int group = chunk / 2, inner = row % 128;
        const std::size_t offset = (static_cast<std::size_t>(row / 128) * (k / 64) + group / 4) * 512 +
                                   static_cast<std::size_t>(inner % 32) * 16 + (inner / 32) * 4 + group % 4;
        const double scale = e4m3_float(scales[offset]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w.w[j] = __double2float_rn(static_cast<double>(e2m1_float(packed >> (4 * j))) * scale / divisor);
        }
    }
};

// A matrix of nvfp4_expert_rg16_v1 row-group units (design §6.2): row r's 16-element block b is in
// the 144-byte unit (r / 16, b) of `blocks` per row group; element k's code is the nibble (low for
// rows 0-7 of the group, high for 8-15) of byte 32 (k / 4) + 4 (r % 8) + k % 4, its E4M3 scale byte
// 128 + r % 16. The weight e2m1 * scale (exact: 2 x 4 significant bits); the matrix's FP32
// multiplier applies to the sum.
struct Rg16 {
    INFERNIX_PRECISE_UNGROUPED
    static constexpr Products kProducts = Products::Exact;
    const std::uint8_t* matrix;
    int blocks;
    __device__ void decode(int row, int chunk, Weights8& w) const {
        const int r = row % 16, i = r % 8, shift = r < 8 ? 0 : 4, k0 = 8 * (chunk % 2);
        const std::uint8_t* unit =
            matrix + (static_cast<std::size_t>(row / 16) * blocks + chunk / 2) * 144;
        const float scale = e4m3_float(unit[128 + r]);
        // Elements k0..k0+3 and k0+4..k0+7: four consecutive bytes each (units are 16-byte aligned).
        const std::uint32_t words[2] = {*reinterpret_cast<const std::uint32_t*>(unit + 32 * (k0 / 4) + 4 * i),
                                        *reinterpret_cast<const std::uint32_t*>(unit + 32 * (k0 / 4 + 1) + 4 * i)};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t nibble = words[j / 4] >> (8 * (j % 4) + shift);
            w.w[j] = __fmul_rn(e2m1_float(nibble), scale);
        }
    }
};

// Block-scaled FP8 rows of length k: E4M3FN codes, one BF16 scale per 128 x 128 tile (row-major
// [rows / 128][k / 128]): code times its tile's scale (exact: 4 x 8 significant bits).
struct Fp8B128 {
    INFERNIX_PRECISE_UNGROUPED
    static constexpr Products kProducts = Products::Exact;
    const std::uint8_t* codes;
    const std::uint16_t* scales;
    int k;
    __device__ void decode(int row, int chunk, Weights8& w) const {
        const uint2 packed = *reinterpret_cast<const uint2*>(codes + static_cast<std::size_t>(row) * k + 8 * chunk);
        const auto* v      = reinterpret_cast<const std::uint8_t*>(&packed);
        const float scale  = bf16_float(scales[static_cast<std::size_t>(row / 128) * (k / 128) + 8 * chunk / 128]);
#pragma unroll
        for (int j = 0; j < 8; ++j) { w.w[j] = __fmul_rn(e4m3_float(v[j]), scale); }
    }
};

// GGUF blocks (ggml_rows_v1 or ggml_tiles16_v1): kappa * (d * sub * code - dmin * min_sub), grouped as
// code (at most 8 bits) times scale = kappa * d * sub minus offset = kappa * dmin * min_sub, both exact in
// binary32 (at most 11 + 7 bits), constant over the format's sub-block of kGroupChunks chunks.
template <QType F>
struct Ggml {
    static constexpr Products kProducts = Products::Grouped;
    static constexpr int kGroupChunks   = F == QType::GGML_Q2_0                                ? 8
                                          : F == QType::GGML_Q6_K || F == QType::GGML_IQ2_S ? 2
                                                                                              : 4;
    static constexpr bool kOffset = F == QType::GGML_Q4_K || F == QType::GGML_Q5_K;
    const std::uint8_t* data;
    std::size_t row_bytes;
    bool tiles16;
    __device__ void decode(int row, int chunk, Weights8& w) const {
        constexpr int kParts      = static_cast<int>(ggml_block_elems(F)) / 8;
        constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
        static_assert(kParts % kGroupChunks == 0, "a group lies inside one block");
        const int block = chunk / kParts;
        const std::uint8_t* b =
            tiles16 ? data + static_cast<std::size_t>(row / 16) * 16 * row_bytes +
                          static_cast<std::size_t>(16 * block + row % 16) * kBlockBytes
                    : data + static_cast<std::size_t>(row) * row_bytes + static_cast<std::size_t>(block) * kBlockBytes;
        const ggml::Chunk c = ggml::chunk<F>(b, chunk % kParts);
        const float kappa   = ldexpf(1.0F, ggml::kappa_log2(F));
        w.scale             = __fmul_rn(kappa, __fmul_rn(f16_float(c.d), static_cast<float>(c.sub)));
        w.offset            = __fmul_rn(kappa, __fmul_rn(f16_float(c.dmin), static_cast<float>(c.min_sub)));
#pragma unroll
        for (int j = 0; j < 8; ++j) { w.w[j] = static_cast<float>(c.code[j]); }
    }
};

} // namespace infernix::ops::precise
