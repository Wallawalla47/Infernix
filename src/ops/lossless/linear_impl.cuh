#pragma once

#include "infernix/ops/lossless_linear.h"

#include "core/device.h"
#include "infernix/ops/ggml_linear.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/ggml/ggml_exact.h"
#include "ops/lossless/wide.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

// The --lossless route's projections (include/infernix/ops/lossless_linear.h): exact sums of exactly
// decoded weights times BF16 activations, rounded once. Shared by lossless_linear.cu (the API and the
// non-GGUF formats) and lossless_linear_ggml_{k,i}.cu (the GGUF formats' instantiations).
//
// Encoding (lossless_encode_kernel). A BF16 value is M 2^(e - 134) with M = 128 | mantissa and e its
// exponent field (subnormals: M = mantissa, e = 1). For each column and 64-element block b with
// largest exponent e_b, element k becomes the integer X_k = M_k 2^(e_k - e_b + 21) (|X| < 2^29), so
// x_k = X_k 2^(e_b - 155). An element 22 to 43 binades below e_b goes on a second ("lo") grid 22 binades
// lower instead, X'_k = M_k 2^(e_k - e_b + 43) (x_k = X'_k 2^(e_b - 177)); one further below marks the
// column inexact. Each grid is stored as four byte limbs, X = L0 + 2^8 L1 + 2^16 L2 + 2^24 L3 with L0-L2
// unsigned and L3 = X >> 24 signed, per column and block [4][64] bytes; a (64-column tile, block) flag
// says whether any of the tile's columns has lo elements in the block.
//
// Weights (the formats below). Each weight is an integer W times its scale group's m_s 2^(e_s - bias)
// (and, for some formats, a per-row factor applied to the finished sum). W is held as P signed-byte
// digit planes, W = sum_p 2^(7 p) D_p (P = 1 when W fits a byte).
//  - Row-split Q2/Q4/Q5/Q6 (64 per group), Q8 (32 per group): W = code, FP16 scale m_s 2^(e_s - 25).
//  - NVFP4 (16 per group): W = 2 e2m1 (|W| <= 12), E4M3 scale m_s 2^(e_s - 10) (so the weight is
//    W m_s 2^(e_s - 11)), and the finished sum is divided by the matrix's FP32 divisor (exactly: the
//    quotient carries a sticky bit).
//  - FP8 row (E4M3 times the row's BF16 scale): W = E4M3 value times 2^9 (|W| < 2^18, three planes),
//    one group per 64-element block without a scale, and the finished sum multiplied by the row scale.
//  - MXFP8 (g32_scale_rows16_v1): W = E4M3 value times 2^9, scale 2^(e8 - 136) per 32.
//  - Block FP8 (fp8_expert_b128_v1 records): W = E4M3 value times 2^9, the 128 x 128 tile's BF16 scale
//    per 64-element group.
//  - NVFP4 expert records (nvfp4_expert_rg16_v1): as NVFP4 in 144-byte row-group units, the finished
//    sum multiplied by the matrix's FP32 multiplier (gate rows even, up rows odd).
//  - GGUF blocks (ggml_rows_v1, ggml_tiles16_v1, expert records): kappa (d sub code - dmin min_sub)
//    (ops/ggml/ggml_exact.h): W = code, scale mant(d) sub 2^(e(d) - 25) kappa per 16, 32 or 64; for
//    Q4_K/Q5_K the min term -mant(dmin) min_sub 2^(e(dmin) - 25) kappa times the group's activation sum
//    (exact per column and grid, from the encoding) is a second term of the group.
//
// The tile mode (lossless_linear_tiles, routed experts in offloaded records): each 64-column tile
// takes its weights from its own record, its columns read through an input map and are written
// through an output map.
//
// Products (lossless_gemm_kernel). Per block and scale group, mma (m16n8k32, or m16n8k16 for 16-element
// groups) multiplies each limb plane by each weight plane, exact in int32; the group's sum
// S = sum_l sum_p 2^(8 l + 7 p) P_lp is exact in int64 (|S| < 2^53 for every format). The group's term
// S m_s 2^(e_b + e_s - c) is added into an int128 whose unit is 2^L, L = E_col + E_row - c - 64 (E_col
// the column's largest block exponent, E_row the row's largest scale exponent), so its shift is
// 64 - (E_col - e_b) - (E_row - e_s) <= 64 (22 less for the lo grid's) and the sum stays below 2^127. A term shifted below the
// unit (more than 64 binades under the largest possible) marks the output inexact.
//
// Rounding. The int128 times 2^L (times or divided by the row factor) is rounded to BF16 or FP32 (round
// to nearest, ties to even, with subnormals, overflow to infinity). An inexact output (an inexact
// column, a non-finite column or a dropped term) is flagged in a bitmap and recomputed by
// lossless_fallback_kernel: one warp per output, every product W m_s M added at its exponent into a
// 352-bit fixed-point accumulator (bit 0 = 2^-160) that holds every finite sum exactly, then rounded
// the same way. Non-finite columns follow IEEE: NaN in, or infinities of both signs, or an infinity
// times a zero weight give NaN.
//
// BF16 weights take the same route: per (row, 64-block) on a 29-bit grid below the block's largest
// exponent (and a lo grid 22 binades lower) as five balanced 7-bit digit planes, groups of 32 (their
// sums below 2^63); the exact fallback's accumulator is 576 bits wide for them.

namespace infernix::ops::lossless_linear_impl {

using lossless::add_shifted;
using lossless::I128;
using lossless::round_magnitude;
using lossless::Wide;

constexpr int kBlockK        = 64;
constexpr int kLimbs         = 4;
constexpr int kGridBits      = 21;
constexpr int kLoShift       = 22; // the lo grid's unit below the hi grid's
constexpr int kChunkColumns  = 512;
constexpr int kTileTokens    = 64;
constexpr int kTileRows      = 32;  // two CTAs per SM (a thread's 8 int128 accumulators fit 128 registers)
constexpr int kThreads       = 256; // 8 warps: 4 along tokens (16 each) x 2 along rows (16 each)
constexpr int kWarpRows      = kTileRows / 2;        // rows per warp
constexpr int kRowTiles      = kWarpRows / 8;        // a warp's 8-row mma tiles
constexpr int kRowParts      = kThreads / kTileRows; // threads per row in the prologue
static_assert(kRowTiles % 2 == 0 && kRowParts <= 32, "B fragments load two 8-row tiles per ldmatrix_x4");
constexpr int kBlockBytes    = kLimbs * kBlockK;     // 256 bytes of limbs per column and block
constexpr int kAStride       = kBlockBytes + 16;     // padded shared row (ldmatrix without conflicts)
constexpr int kBStride       = kBlockK + 16;         // 80
constexpr int kFlagInexact   = 1;
constexpr int kFlagNonFinite = 2;
constexpr int kWideWords     = 11; // 352 bits
constexpr int kWideBias      = 160;
constexpr int kBf16Words     = 18; // 576 bits
constexpr int kBf16Bias      = 266;
constexpr int kGgmlWords     = 13;  // 416 bits: GGUF products up to 2^33, split in two
constexpr int kFp8Words      = 19;  // 608 bits: E8M0 and BF16 tile scales span 254 binades
constexpr int kMxfp8Bias     = 272;
constexpr int kFp8B128Bias   = 280;
constexpr int kSpecialExp    = 1 << 20; // element(): a non-finite weight (W = 0: NaN, else its sign)

static_assert(kChunkColumns % kTileTokens == 0, "column chunks hold whole tiles");

inline void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless_linear: ") + message); }
}

inline std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// The workspace of one column chunk (at most kChunkColumns columns), and BF16 weights' scan of the rows.
struct Layout {
    std::size_t limbs, lo_limbs, tile_lo, exps, col_emax, col_flags, bitmap, xsum, block_exp, tile_lo_w, row_inexact,
        total;
};

inline Layout layout_of(std::int32_t k, std::int32_t rows, std::int32_t columns, bool bf16_weight) {
    const std::size_t blocks = static_cast<std::size_t>(k) / kBlockK;
    const std::size_t c      = static_cast<std::size_t>(std::min(columns, kChunkColumns));
    const std::size_t r      = bf16_weight ? static_cast<std::size_t>(rows) : 0;
    Layout l{};
    l.limbs     = 0;
    l.lo_limbs  = l.limbs + align256(c * blocks * kBlockBytes);
    l.tile_lo   = l.lo_limbs + align256(c * blocks * kBlockBytes);
    l.exps      = l.tile_lo + align256((c + kTileTokens - 1) / kTileTokens * blocks);
    l.col_emax  = l.exps + align256(c * blocks * sizeof(std::int16_t));
    l.col_flags = l.col_emax + align256(c * sizeof(std::int32_t));
    l.bitmap    = l.col_flags + align256(c * sizeof(std::int32_t));
    l.xsum        = l.bitmap + align256((static_cast<std::size_t>(rows) * c + 31) / 32 * sizeof(std::uint32_t));
    l.block_exp   = l.xsum + align256(c * blocks * 4 * sizeof(long long));
    l.tile_lo_w   = l.block_exp + align256(r * blocks * sizeof(std::int16_t));
    l.row_inexact = l.tile_lo_w + align256((r + kTileRows - 1) / kTileRows * blocks);
    l.total       = l.row_inexact + align256(r * sizeof(std::int32_t));
    return l;
}

// --------------------------------------------------------------------------------- encoding

// One warp per (column, block): lane i holds elements 2i and 2i + 1.
// Column c reads x's column in_columns[c] (identity when null; a negative entry reads zeros). With xsum,
// each half block's sums of X and X' (int64 [column][block][half][grid]).
static __global__ void lossless_encode_kernel(const std::uint16_t* __restrict__ x, const std::int32_t* __restrict__ in_columns,
                                       int k, int columns, std::uint8_t* __restrict__ limbs,
                                       std::uint8_t* __restrict__ lo_limbs, std::uint8_t* __restrict__ tile_lo,
                                       std::int16_t* __restrict__ exps, std::int32_t* __restrict__ col_emax,
                                       std::int32_t* __restrict__ col_flags, long long* __restrict__ xsum) {
    const int warp_global = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int blocks      = k / kBlockK;
    if (warp_global >= blocks * columns) { return; }
    const int column = warp_global / blocks, block = warp_global % blocks;
    const int source = in_columns != nullptr ? in_columns[column] : column;
    const std::uint32_t pair =
        source < 0 ? 0U
                   : *reinterpret_cast<const std::uint32_t*>(x + static_cast<std::size_t>(source) * k + block * kBlockK +
                                                             2 * lane);
    int e[2], m[2], s[2], flags = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const std::uint32_t bits = (pair >> (16 * i)) & 0xFFFFU;
        const int field          = static_cast<int>((bits >> 7) & 0xFF);
        const int mantissa       = static_cast<int>(bits & 0x7F);
        s[i]                     = static_cast<int>(bits >> 15);
        if (field == 255) {
            flags |= kFlagNonFinite;
            m[i] = 0;
            e[i] = 0;
        } else if (field == 0) {
            m[i] = mantissa;
            e[i] = mantissa != 0 ? 1 : 0;
        } else {
            m[i] = 128 | mantissa;
            e[i] = field;
        }
    }
    int eb = max(e[0], e[1]);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { eb = max(eb, __shfl_xor_sync(0xFFFFFFFFU, eb, offset)); }
    if (eb == 0) { eb = 1; } // an all-zero (or non-finite) block
    std::uint32_t out[2], lo[2];
    bool has_lo = false;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int shift = e[i] - eb + kGridBits;
        int X = 0, Y = 0;
        if (m[i] != 0) {
            if (shift >= 0) {
                X = m[i] << shift;
            } else if (shift + kLoShift >= 0) {
                Y      = m[i] << (shift + kLoShift);
                has_lo = true;
            } else {
                flags |= kFlagInexact; // more than 43 binades below the block's largest: the exact fallback
            }
        }
        if (s[i] != 0) {
            X = -X;
            Y = -Y;
        }
        out[i] = static_cast<std::uint32_t>(X);
        lo[i]  = static_cast<std::uint32_t>(Y);
    }
    // Limb plane l of this (column, block): bytes [l * 64, l * 64 + 64), for each grid.
    const std::size_t at = (static_cast<std::size_t>(column) * blocks + block) * kBlockBytes;
#pragma unroll
    for (int l = 0; l < kLimbs; ++l) {
        const std::uint32_t b0 = (out[0] >> (8 * l)) & 0xFFU, b1 = (out[1] >> (8 * l)) & 0xFFU;
        *reinterpret_cast<std::uint16_t*>(limbs + at + l * kBlockK + 2 * lane) = static_cast<std::uint16_t>(b0 | (b1 << 8));
        const std::uint32_t c0 = (lo[0] >> (8 * l)) & 0xFFU, c1 = (lo[1] >> (8 * l)) & 0xFFU;
        *reinterpret_cast<std::uint16_t*>(lo_limbs + at + l * kBlockK + 2 * lane) = static_cast<std::uint16_t>(c0 | (c1 << 8));
    }
    if (__any_sync(0xFFFFFFFFU, has_lo) && lane == 0) { tile_lo[(column / kTileTokens) * blocks + block] = 1; }
    if (xsum != nullptr) {
        // Lanes 0-15 hold the block's first 32 elements, lanes 16-31 the rest.
        long long hs = static_cast<long long>(static_cast<int>(out[0])) + static_cast<int>(out[1]);
        long long ls = static_cast<long long>(static_cast<int>(lo[0])) + static_cast<int>(lo[1]);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            hs += __shfl_xor_sync(0xFFFFFFFFU, hs, offset);
            ls += __shfl_xor_sync(0xFFFFFFFFU, ls, offset);
        }
        if ((lane & 15) == 0) {
            long long* at = xsum + ((static_cast<std::size_t>(column) * blocks + block) * 2 + lane / 16) * 2;
            at[0] = hs;
            at[1] = ls;
        }
    }
    // (L3 = X >> 24 is the arithmetic shift: its byte is the two's complement's fourth byte.)
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { flags |= __shfl_xor_sync(0xFFFFFFFFU, flags, offset); }
    if (lane == 0) {
        exps[static_cast<std::size_t>(column) * blocks + block] = static_cast<std::int16_t>(eb);
        atomicMax(col_emax + column, eb);
        if (flags != 0) { atomicOr(col_flags + column, flags); }
    }
}

// --------------------------------------------------------------------------------- weight formats

// The matrix as the kernels read it (the fields a format does not use are null or zero).
struct WeightView {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const void* scales;
    int row_bytes;          // bytes of one row's codes
    int groups;             // scale groups per (padded) row
    int k;                  // the weight's K (NVFP4's scale swizzle, BF16 rows)
    std::uint32_t divisor;  // NVFP4: the FP32 divisor's bits
    // BF16 (bf16_weight_scan_kernel, per call; rows relative to row_base): each (row, block)'s largest
    // exponent field, the (64-row tile, block)s holding lo-grid weights, the rows the fallback computes.
    const std::int16_t* block_exp;
    const std::uint8_t* tile_lo_w;
    const std::int32_t* row_inexact;
    int blocks;
    int row_base;
    int tiles16;            // GGUF: ggml_tiles16_v1
    std::uint32_t alpha[2]; // NVFP4 expert records: the FP32 multipliers of even and odd rows (bits)
    // The design reference's activation scales (LosslessActivationScale, LosslessTile::act): FP32 bits,
    // 0 none: a divisor of the call, multipliers of even and odd rows.
    std::uint32_t act_div;
    std::uint32_t act_mul[2];
};

// The tile mode: a view whose codes and scales pointers are offsets into each tile's record.
__device__ __forceinline__ WeightView tile_view(const WeightView& w, const LosslessTile& t) {
    WeightView v = w;
    v.codes      = t.record + reinterpret_cast<std::uintptr_t>(w.codes);
    v.scales     = t.record + reinterpret_cast<std::uintptr_t>(w.scales);
    v.alpha[0]   = t.alpha[0];
    v.alpha[1]   = t.alpha[1];
    v.act_mul[0] = t.act[0];
    v.act_mul[1] = t.act[1];
    return v;
}

enum class Factor { None, RowScale, RowScaleF32, Divisor, Multiplier };

// An FP16 scale as m_s 2^(e_s - 25): signed integer significand and exponent field (subnormals e_s = 1).
__device__ __forceinline__ void fp16_parts(std::uint16_t bits, int& m, int& e) {
    const int field = (bits >> 10) & 0x1F, mantissa = bits & 0x3FF;
    m               = field == 0 ? mantissa : (1024 | mantissa);
    e               = field == 0 ? 1 : field;
    if ((bits & 0x8000U) != 0) { m = -m; }
}

// A (nonnegative) E4M3FN word as m 2^(e - 10) (subnormals e = 1).
__device__ __forceinline__ void e4m3_parts(std::uint32_t bits, int& m, int& e) {
    const int field = static_cast<int>((bits >> 3) & 0xF), mantissa = static_cast<int>(bits & 7U);
    m               = field == 0 ? mantissa : (8 | mantissa);
    e               = field == 0 ? 1 : field;
}

// E4M3FN value times 2^9 as a signed integer (|W| <= 15 2^14).
__device__ __forceinline__ int e4m3_integer(std::uint32_t bits) {
    int m = 0, e = 1;
    e4m3_parts(bits, m, e);
    const int w = m << (e - 1);
    return (bits & 0x80U) != 0 ? -w : w;
}

// Two times an E2M1 value (|W| <= 12).
__device__ __forceinline__ int e2m1_twice(std::uint32_t nibble) {
    constexpr std::uint32_t kTable = 0xC8643210U; // 0, 1, 2, 3, 4, 6, 8, 12 as nibbles
    const int v = static_cast<int>((kTable >> (4 * (nibble & 7U))) & 0xFU);
    return (nibble & 8U) != 0 ? -v : v;
}

// Row-split grouped integers: code times its group's FP16 scale.
template <int Bits>
struct RowSplitFormat {
    static constexpr int kPlanes   = 1;
    static constexpr int kGroup    = Bits == 8 ? 32 : 64;
    static constexpr int kUnitBias = 244;
    static constexpr int kShiftTop = 64;
    static constexpr bool kScaled  = true;
    static constexpr bool kWeightLo = false;
    static constexpr Factor kFactor = Factor::None;
    static constexpr int kWide      = kWideWords;
    static constexpr int kWideBase  = kWideBias;
    static constexpr bool kOffset   = false;
    static constexpr bool kSplit    = false;

    __device__ static int code(const WeightView& w, int row, int k) {
        if constexpr (Bits == 8) {
            return static_cast<std::int8_t>(w.codes[static_cast<std::size_t>(row) * w.row_bytes + k]);
        } else if constexpr (Bits == 2) {
            const int u = (w.codes[static_cast<std::size_t>(row) * w.row_bytes + k / 4] >> (2 * (k & 3))) & 3;
            return u >= 2 ? u - 4 : u;
        } else {
            const std::uint8_t byte = w.codes[static_cast<std::size_t>(row) * w.row_bytes + k / 2];
            std::uint32_t u         = (k & 1) ? (byte >> 4) : (byte & 0xFU);
            const int group = k / kGroup, i = k % kGroup;
            if constexpr (Bits == 5) {
                u |= ((w.high[static_cast<std::size_t>(row) * w.groups * 8 + group * 8 + i / 8] >> (i & 7)) & 1U) << 4;
            } else if constexpr (Bits == 6) {
                const int bit = 2 * i;
                u |= ((w.high[static_cast<std::size_t>(row) * w.groups * 16 + group * 16 + bit / 8] >> (bit & 7)) & 3U) << 4;
            }
            return (u & (1U << (Bits - 1))) != 0 ? static_cast<int>(u) - (1 << Bits) : static_cast<int>(u);
        }
    }

    // Codes [16 q, 16 q + 16) of block b of the row.
    __device__ static void decode16(const WeightView& w, int row, int b, int q, std::int8_t* dst, int plane, bool) {
        const int k0 = b * kBlockK + 16 * q;
        std::int8_t* out = dst;
        if constexpr (Bits == 8) {
            *reinterpret_cast<uint4*>(out) =
                *reinterpret_cast<const uint4*>(w.codes + static_cast<std::size_t>(row) * w.row_bytes + k0);
        } else if constexpr (Bits == 2) {
            const std::uint32_t packed =
                *reinterpret_cast<const std::uint32_t*>(w.codes + static_cast<std::size_t>(row) * w.row_bytes + k0 / 4);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int u = static_cast<int>((packed >> (2 * j)) & 3U);
                out[j]      = static_cast<std::int8_t>(u >= 2 ? u - 4 : u);
            }
        } else {
            const uint2 packed =
                *reinterpret_cast<const uint2*>(w.codes + static_cast<std::size_t>(row) * w.row_bytes + k0 / 2);
            std::uint32_t hi = 0;
            const int group = k0 / kGroup, lane = k0 % kGroup;
            if constexpr (Bits == 5) {
                hi = *reinterpret_cast<const std::uint16_t*>(w.high + static_cast<std::size_t>(row) * w.groups * 8 +
                                                            group * 8 + lane / 8);
            } else if constexpr (Bits == 6) {
                hi = *reinterpret_cast<const std::uint32_t*>(w.high + static_cast<std::size_t>(row) * w.groups * 16 +
                                                            group * 16 + lane / 4);
            }
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const std::uint32_t word = j < 8 ? packed.x : packed.y;
                std::uint32_t u          = (word >> (4 * (j & 7))) & 0xFU;
                if constexpr (Bits == 5) { u |= ((hi >> j) & 1U) << 4; }
                if constexpr (Bits == 6) { u |= ((hi >> (2 * j)) & 3U) << 4; }
                out[j] = static_cast<std::int8_t>((u & (1U << (Bits - 1))) != 0 ? static_cast<int>(u) - (1 << Bits)
                                                                                 : static_cast<int>(u));
            }
        }
    }

    __device__ static void scale(const WeightView& w, int row, int group, int& m, int& e) {
        fp16_parts(static_cast<const std::uint16_t*>(w.scales)[static_cast<std::size_t>(row) * w.groups + group], m, e);
    }

    // The fallback's weight: W 2^wexp.
    __device__ static void element(const WeightView& w, int row, int k, int& W, int& wexp) {
        int m = 0, e = 1;
        scale(w, row, k / kGroup, m, e);
        W    = code(w, row, k) * m;
        wexp = e - 25;
    }
};

// NVFP4 (block_scale_k16_m128x4_v1): E2M1 codes two per byte (low nibble first), the 16-element groups'
// E4M3 scales in the K16M128x4 swizzle, the sum divided by the FP32 divisor.
struct Nvfp4Format {
    static constexpr int kPlanes   = 1;
    static constexpr int kGroup    = 16;
    static constexpr int kUnitBias = 230;
    static constexpr int kShiftTop = 64;
    static constexpr bool kScaled  = true;
    static constexpr bool kWeightLo = false;
    static constexpr Factor kFactor = Factor::Divisor;
    static constexpr int kWide      = kWideWords;
    static constexpr int kWideBase  = kWideBias;
    static constexpr bool kOffset   = false;
    static constexpr bool kSplit    = false;

    __device__ static std::size_t scale_offset(const WeightView& w, int row, int group) {
        const int inner = row % 128;
        return (static_cast<std::size_t>(row / 128) * (w.k / 64) + group / 4) * 512 +
               static_cast<std::size_t>(inner % 32) * 16 + (inner / 32) * 4 + group % 4;
    }

    __device__ static void decode16(const WeightView& w, int row, int b, int q, std::int8_t* dst, int plane, bool) {
        const int k0 = b * kBlockK + 16 * q;
        const uint2 packed = *reinterpret_cast<const uint2*>(w.codes + static_cast<std::size_t>(row) * w.row_bytes + k0 / 2);
        std::int8_t* out   = dst;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const std::uint32_t word = j < 8 ? packed.x : packed.y;
            out[j] = static_cast<std::int8_t>(e2m1_twice((word >> (4 * (j & 7))) & 0xFU));
        }
    }

    __device__ static void scale(const WeightView& w, int row, int group, int& m, int& e) {
        e4m3_parts(static_cast<const std::uint8_t*>(w.scales)[scale_offset(w, row, group)], m, e);
    }

    __device__ static void element(const WeightView& w, int row, int k, int& W, int& wexp) {
        int m = 0, e = 1;
        scale(w, row, k / kGroup, m, e);
        const std::uint8_t byte = w.codes[static_cast<std::size_t>(row) * w.row_bytes + k / 2];
        W    = e2m1_twice((k & 1) ? (byte >> 4) : (byte & 0xFU)) * m;
        wexp = e - 11;
    }
};

// Sixteen E4M3 codes as the three balanced base-128 digit planes of W = value 2^9.
__device__ __forceinline__ void fp8_digits16(const std::uint8_t* src, std::int8_t* dst, int plane) {
    const uint4 packed = *reinterpret_cast<const uint4*>(src);
    const auto* bytes  = reinterpret_cast<const std::uint8_t*>(&packed);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int W        = e4m3_integer(bytes[j]);
        const int d0 = ((W + 64) & 127) - 64;
        W            = (W - d0) >> 7;
        const int d1 = ((W + 64) & 127) - 64;
        W            = (W - d1) >> 7;
        dst[j]             = static_cast<std::int8_t>(d0);
        dst[plane + j]     = static_cast<std::int8_t>(d1);
        dst[2 * plane + j] = static_cast<std::int8_t>(W);
    }
}

// Row-scaled FP8: the E4M3 value times 2^9 as three balanced base-128 digits, no group scale, the sum
// multiplied by the row's BF16 scale (fp8_e4m3fn_row_bf16) or, in Fp8RowF32Format, its FP32 multiplier
// (fp8_e4m3fn_row_f32: ModelOpt's per-tensor weight_scale in every row).
struct Fp8RowFormat {
    static constexpr int kPlanes   = 3;
    static constexpr int kGroup    = 64;
    static constexpr int kUnitBias = 228;
    static constexpr int kShiftTop = 64;
    static constexpr bool kScaled  = false;
    static constexpr bool kWeightLo = false;
    static constexpr Factor kFactor = Factor::RowScale;
    static constexpr int kWide      = kWideWords;
    static constexpr int kWideBase  = kWideBias;
    static constexpr bool kOffset   = false;
    static constexpr bool kSplit    = false;

    __device__ static void decode16(const WeightView& w, int row, int b, int q, std::int8_t* dst, int plane, bool) {
        fp8_digits16(w.codes + static_cast<std::size_t>(row) * w.row_bytes + b * kBlockK + 16 * q, dst, plane);
    }

    __device__ static void scale(const WeightView&, int, int, int& m, int& e) {
        m = 1;
        e = 0;
    }

    __device__ static void element(const WeightView& w, int row, int k, int& W, int& wexp) {
        W    = e4m3_integer(w.codes[static_cast<std::size_t>(row) * w.row_bytes + k]);
        wexp = -9;
    }
};

struct Fp8RowF32Format : Fp8RowFormat {
    static constexpr Factor kFactor = Factor::RowScaleF32;
};

// MXFP8 (g32_scale_rows16_v1): W = E4M3 value 2^9, the 32-element group's UE8M0 scale 2^(e8 - 127), so
// the weight is W 2^(e8 - 136). Codes [N, K]; the scale of (row, group) at ((row / 16) (K / 32) + group)
// 16 + row % 16.
struct Mxfp8Format {
    static constexpr int kPlanes    = 3;
    static constexpr int kGroup     = 32;
    static constexpr int kShiftTop  = 64;
    static constexpr int kUnitBias  = 155 + 136 + kShiftTop;
    static constexpr bool kScaled   = true;
    static constexpr bool kWeightLo = false;
    static constexpr Factor kFactor = Factor::None;
    static constexpr int kWide      = kFp8Words;
    static constexpr int kWideBase  = kMxfp8Bias;
    static constexpr bool kOffset   = false;
    static constexpr bool kSplit    = false;

    __device__ static void decode16(const WeightView& w, int row, int b, int q, std::int8_t* dst, int plane, bool) {
        fp8_digits16(w.codes + static_cast<std::size_t>(row) * w.k + b * kBlockK + 16 * q, dst, plane);
    }

    __device__ static void scale(const WeightView& w, int row, int group, int& m, int& e) {
        m = 1;
        e = static_cast<const std::uint8_t*>(w.scales)[(static_cast<std::size_t>(row / 16) * (w.k / 32) + group) * 16 +
                                                       row % 16];
    }

    __device__ static void element(const WeightView& w, int row, int k, int& W, int& wexp) {
        int m = 1, e = 0;
        scale(w, row, k / kGroup, m, e);
        W    = e4m3_integer(w.codes[static_cast<std::size_t>(row) * w.k + k]);
        wexp = e - 136;
    }
};

// Block FP8 rows of length K (fp8_expert_b128_v1's matrices): W = E4M3 value 2^9, the 128 x 128 tile's
// BF16 scale m 2^(e - 134) (row-major [rows / 128][K / 128]) per 64-element group, so the weight is
// W m 2^(e - 143).
struct Fp8B128Format {
    static constexpr int kPlanes    = 3;
    static constexpr int kGroup     = 64;
    static constexpr int kShiftTop  = 56;
    static constexpr int kUnitBias  = 155 + 143 + kShiftTop;
    static constexpr bool kScaled   = true;
    static constexpr bool kWeightLo = false;
    static constexpr Factor kFactor = Factor::None;
    static constexpr int kWide      = kFp8Words;
    static constexpr int kWideBase  = kFp8B128Bias;
    static constexpr bool kOffset   = false;
    static constexpr bool kSplit    = true;

    __device__ static void decode16(const WeightView& w, int row, int b, int q, std::int8_t* dst, int plane, bool) {
        fp8_digits16(w.codes + static_cast<std::size_t>(row) * w.k + b * kBlockK + 16 * q, dst, plane);
    }

    __device__ static void scale(const WeightView& w, int row, int group, int& m, int& e) {
        const std::uint16_t bits = static_cast<const std::uint16_t*>(
            w.scales)[static_cast<std::size_t>(row / 128) * (w.k / 128) + (group * kGroup) / 128];
        const int field = (bits >> 7) & 0xFF, mantissa = bits & 0x7F;
        m = field == 0 ? mantissa : (128 | mantissa);
        e = field == 0 ? 1 : field;
        if ((bits & 0x8000U) != 0) { m = -m; }
    }

    __device__ static void element(const WeightView& w, int row, int k, int& W, int& wexp) {
        int m = 0, e = 1;
        scale(w, row, k / kGroup, m, e);
        W    = e4m3_integer(w.codes[static_cast<std::size_t>(row) * w.k + k]) * m;
        wexp = e - 143;
    }
};

// NVFP4 expert records (nvfp4_expert_rg16_v1): row r's 16-element block b in the 144-byte unit
// (r / 16, b) of `groups` (K / 16) per row group; element j's code the nibble (low for rows 0-7 of the
// group, high for 8-15) of byte 32 (j / 4) + 4 (r % 8) + j % 4, its E4M3 scale byte 128 + r % 16. The
// finished sum is multiplied by the matrix's FP32 multiplier (alpha[row & 1]).
struct Rg16Format {
    static constexpr int kPlanes    = 1;
    static constexpr int kGroup     = 16;
    static constexpr int kShiftTop  = 64;
    static constexpr int kUnitBias  = 230;
    static constexpr bool kScaled   = true;
    static constexpr bool kWeightLo = false;
    static constexpr Factor kFactor = Factor::Multiplier;
    static constexpr int kWide      = kWideWords;
    static constexpr int kWideBase  = kWideBias;
    static constexpr bool kOffset   = false;
    static constexpr bool kSplit    = false;

    __device__ static const std::uint8_t* unit(const WeightView& w, int row, int block16) {
        return w.codes + (static_cast<std::size_t>(row / 16) * w.groups + block16) * 144;
    }

    __device__ static void decode16(const WeightView& w, int row, int b, int q, std::int8_t* dst, int, bool) {
        const std::uint8_t* u = unit(w, row, 4 * b + q);
        const int i = row % 8, shift = (row % 16) < 8 ? 0 : 4;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(u + 32 * t + 4 * i);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                dst[4 * t + j] = static_cast<std::int8_t>(e2m1_twice((word >> (8 * j + shift)) & 0xFU));
            }
        }
    }

    __device__ static void scale(const WeightView& w, int row, int group, int& m, int& e) {
        e4m3_parts(unit(w, row, group)[128 + row % 16], m, e);
    }

    __device__ static void element(const WeightView& w, int row, int k, int& W, int& wexp) {
        int m = 0, e = 1;
        scale(w, row, k / kGroup, m, e);
        const int j = k % 16, shift = (row % 16) < 8 ? 0 : 4;
        const std::uint8_t byte = unit(w, row, k / 16)[32 * (j / 4) + 4 * (row % 8) + j % 4];
        W    = e2m1_twice((byte >> shift) & 0xFU) * m;
        wexp = e - 11;
    }
};

// GGUF blocks (ggml_rows_v1 rows of row_bytes, or ggml_tiles16_v1): the codes of ggml::chunk, the group
// scale mant(d) sub 2^(e(d) - 25) kappa and, for Q4_K/Q5_K, the min term's -mant(dmin) min_sub.
template <QType Q>
struct GgmlFormat {
    static constexpr int kChunksGroup = Q == QType::GGML_Q2_0                                 ? 8
                                        : Q == QType::GGML_Q6_K || Q == QType::GGML_IQ2_S ? 2
                                                                                            : 4;
    static constexpr int kPlanes    = 1;
    static constexpr int kGroup     = 8 * kChunksGroup;
    static constexpr int kKappa     = Q == QType::GGML_IQ2_S ? -3 : Q == QType::GGML_IQ3_XXS ? -2 : 0;
    static constexpr int kShiftTop  = 56;
    static constexpr int kUnitBias  = 180 - kKappa + kShiftTop;
    static constexpr bool kScaled   = true;
    static constexpr bool kWeightLo = false;
    static constexpr Factor kFactor = Factor::None;
    static constexpr int kWide      = kGgmlWords;
    static constexpr int kWideBase  = kWideBias;
    static constexpr bool kOffset   = Q == QType::GGML_Q4_K || Q == QType::GGML_Q5_K;
    static constexpr bool kSplit    = true;
    static constexpr int kElems     = static_cast<int>(ggml_block_elems(Q));
    static constexpr int kBytes     = static_cast<int>(ggml_block_bytes(Q));
    static_assert(!kOffset || kGroup == 32, "the min term's activation sums are per 32 elements");

    __device__ static ggml::Chunk chunk_at(const WeightView& w, int row, int k) { // k a multiple of 8
        const int blk = k / kElems;
        const std::uint8_t* b =
            w.tiles16 != 0 ? w.codes + static_cast<std::size_t>(row / 16) * 16 * w.row_bytes +
                                 static_cast<std::size_t>(16 * blk + row % 16) * kBytes
                           : w.codes + static_cast<std::size_t>(row) * w.row_bytes + static_cast<std::size_t>(blk) * kBytes;
        return ggml::chunk<Q>(b, (k % kElems) / 8);
    }

    __device__ static void decode16(const WeightView& w, int row, int b, int q, std::int8_t* dst, int, bool) {
        const int k0 = b * kBlockK + 16 * q;
        const ggml::Chunk c0 = chunk_at(w, row, k0), c1 = chunk_at(w, row, k0 + 8);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            dst[j]     = static_cast<std::int8_t>(c0.code[j]);
            dst[8 + j] = static_cast<std::int8_t>(c1.code[j]);
        }
    }

    __device__ static void scale(const WeightView& w, int row, int group, int& m, int& e) {
        const ggml::Chunk c = chunk_at(w, row, group * kGroup);
        fp16_parts(c.d, m, e);
        m *= c.sub;
    }

    __device__ static void offset(const WeightView& w, int row, int group, int& m, int& e) {
        const ggml::Chunk c = chunk_at(w, row, group * kGroup);
        fp16_parts(c.dmin, m, e);
        m *= -c.min_sub;
    }

    __device__ static void element(const WeightView& w, int row, int k, int& W, int& wexp) {
        const ggml::Chunk c = chunk_at(w, row, k & ~7);
        int m = 0, e = 1;
        fp16_parts(c.d, m, e);
        W    = c.code[k & 7] * m * c.sub;
        wexp = e - 25 + kKappa;
    }

    __device__ static void element_offset(const WeightView& w, int row, int k, int& W, int& wexp) {
        const ggml::Chunk c = chunk_at(w, row, k & ~7);
        int m = 0, e = 1;
        fp16_parts(c.dmin, m, e);
        W    = -m * c.min_sub;
        wexp = e - 25 + kKappa;
    }
};

// BF16 weights: per (row, 64-block), each weight on a 29-bit grid below the block's largest exponent
// e_wb (W = M 2^(e - e_wb + 21); the weight is W 2^(e_wb - 155)) as five balanced 7-bit digit planes;
// weights 22-43 binades below e_wb on the lo grid 22 binades lower (decoded with lo, multiplied only
// for the row tiles that hold them); further below, the row's outputs take the exact fallback (finite
// weights only). Groups of 32 keep each group's sum below 2^63, and the int128 keeps 53 binades of room.
struct Bf16GridFormat {
    static constexpr int kPlanes    = 5;
    static constexpr int kGroup     = 32;
    static constexpr int kShiftTop  = 53;
    static constexpr int kUnitBias  = 310 + kShiftTop;
    static constexpr bool kScaled   = true;
    static constexpr bool kWeightLo = true;
    static constexpr Factor kFactor = Factor::None;
    static constexpr int kWide      = kBf16Words;
    static constexpr int kWideBase  = kBf16Bias;
    static constexpr bool kOffset   = false;
    static constexpr bool kSplit    = false;

    __device__ static int block_exp(const WeightView& w, int row, int b) {
        return w.block_exp[static_cast<std::size_t>(row - w.row_base) * w.blocks + b];
    }

    __device__ static void decode16(const WeightView& w, int row, int b, int q, std::int8_t* dst, int plane, bool lo) {
        const int k0     = b * kBlockK + 16 * q;
        const auto* src  = reinterpret_cast<const uint4*>(w.codes + (static_cast<std::size_t>(row) * w.k + k0) * 2);
        const uint4 v0 = src[0], v1 = src[1];
        const std::uint32_t words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        const int eb = block_exp(w, row, b);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const std::uint32_t bits = (words[j / 2] >> (16 * (j & 1))) & 0xFFFFU;
            const int field = static_cast<int>((bits >> 7) & 0xFF), mant = static_cast<int>(bits & 0x7F);
            const int M     = field == 0 ? mant : (128 | mant);
            const int sh    = (field == 0 ? 1 : field) - eb + kGridBits;
            int W           = 0;
            if (M != 0 && field != 255) {
                if (!lo && sh >= 0) {
                    W = M << sh;
                } else if (lo && sh < 0 && sh + kLoShift >= 0) {
                    W = M << (sh + kLoShift);
                }
            }
            if ((bits & 0x8000U) != 0) { W = -W; }
            const int d0 = ((W + 64) & 127) - 64;
            W            = (W - d0) >> 7;
            const int d1 = ((W + 64) & 127) - 64;
            W            = (W - d1) >> 7;
            const int d2 = ((W + 64) & 127) - 64;
            W            = (W - d2) >> 7;
            const int d3 = ((W + 64) & 127) - 64;
            W            = (W - d3) >> 7;
            dst[j]             = static_cast<std::int8_t>(d0);
            dst[plane + j]     = static_cast<std::int8_t>(d1);
            dst[2 * plane + j] = static_cast<std::int8_t>(d2);
            dst[3 * plane + j] = static_cast<std::int8_t>(d3);
            dst[4 * plane + j] = static_cast<std::int8_t>(W);
        }
    }

    __device__ static void scale(const WeightView& w, int row, int group, int& m, int& e) {
        m = 1;
        e = block_exp(w, row, group / 2);
    }

    __device__ static void element(const WeightView& w, int row, int k, int& W, int& wexp) {
        const std::uint16_t bits = reinterpret_cast<const std::uint16_t*>(w.codes)[static_cast<std::size_t>(row) * w.k + k];
        const int field = (bits >> 7) & 0xFF, mant = bits & 0x7F;
        if (field == 255) {
            W    = mant != 0 ? 0 : ((bits & 0x8000U) != 0 ? -1 : 1);
            wexp = kSpecialExp;
            return;
        }
        const int M = field == 0 ? mant : (128 | mant);
        W           = (bits & 0x8000U) != 0 ? -M : M;
        wexp        = (field == 0 ? 1 : field) - 134;
    }
};

// BF16 weights of a call (rows [row_begin, row_begin + rows)): one warp per (row, block), its largest
// exponent field, whether the block holds lo-grid weights (flagged per 64-row tile), and the rows holding
// a weight further below (or a non-finite one).
static __global__ void bf16_weight_scan_kernel(const std::uint16_t* w, int k, int row_begin, int rows, std::int16_t* block_exp,
                                        std::uint8_t* tile_lo, std::int32_t* row_inexact) {
    const int warp_global = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int blocks      = k / kBlockK;
    if (warp_global >= rows * blocks) { return; }
    const int r = warp_global / blocks, b = warp_global % blocks;
    const std::uint32_t pair = *reinterpret_cast<const std::uint32_t*>(
        w + static_cast<std::size_t>(row_begin + r) * k + b * kBlockK + 2 * lane);
    int e[2];
    bool bad = false;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const std::uint32_t bits = (pair >> (16 * i)) & 0xFFFFU;
        const int field = static_cast<int>((bits >> 7) & 0xFF), mant = static_cast<int>(bits & 0x7F);
        e[i] = (field == 0 && mant == 0) ? 0 : (field == 0 ? 1 : field);
        if (field == 255) {
            bad  = true;
            e[i] = 0;
        }
    }
    int eb = max(e[0], e[1]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { eb = max(eb, __shfl_xor_sync(0xFFFFFFFFU, eb, o)); }
    if (eb == 0) { eb = 1; }
    bool lo = false;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (e[i] == 0) { continue; }
        const int sh = e[i] - eb + kGridBits;
        if (sh < 0 && sh + kLoShift >= 0) { lo = true; }
        if (sh + kLoShift < 0) { bad = true; }
    }
    lo  = __any_sync(0xFFFFFFFFU, lo);
    bad = __any_sync(0xFFFFFFFFU, bad);
    if (lane == 0) {
        block_exp[static_cast<std::size_t>(r) * blocks + b] = static_cast<std::int16_t>(eb);
        if (lo) { tile_lo[static_cast<std::size_t>(r / kTileRows) * blocks + b] = 1; }
        if (bad) { row_inexact[r] = 1; }
    }
}

// The per-row factor as an integer and a power of two (RowScale: the BF16 scale m 2^(e - 134);
// RowScaleF32: the row's FP32 multiplier m 2^(e - 150), its sign in negative; Divisor: the FP32 divisor
// m 2^(e - 150); Multiplier: the FP32 multiplier of the row's parity, its sign in negative).
template <class F>
__device__ __forceinline__ void row_factor(const WeightView& w, int row, std::uint32_t& m, int& e, bool& negative) {
    negative = false;
    if constexpr (F::kFactor == Factor::Multiplier) {
        const std::uint32_t bits = w.alpha[row & 1];
        const int field = static_cast<int>((bits >> 23) & 0xFF), mantissa = static_cast<int>(bits & 0x7FFFFF);
        m        = static_cast<std::uint32_t>(field == 0 ? mantissa : ((1 << 23) | mantissa));
        e        = (field == 0 ? 1 : field) - 150;
        negative = (bits >> 31) != 0;
    } else if constexpr (F::kFactor == Factor::RowScaleF32) {
        const std::uint32_t bits = static_cast<const std::uint32_t*>(w.scales)[row];
        const int field = static_cast<int>((bits >> 23) & 0xFF), mantissa = static_cast<int>(bits & 0x7FFFFF);
        m        = static_cast<std::uint32_t>(field == 0 ? mantissa : ((1 << 23) | mantissa));
        e        = (field == 0 ? 1 : field) - 150;
        negative = (bits >> 31) != 0;
    } else if constexpr (F::kFactor == Factor::RowScale) {
        const std::uint16_t bits = static_cast<const std::uint16_t*>(w.scales)[row];
        const int field = (bits >> 7) & 0xFF, mantissa = bits & 0x7F;
        m = static_cast<std::uint32_t>(field == 0 ? mantissa : (128 | mantissa));
        e = (field == 0 ? 1 : field) - 134;
    } else if constexpr (F::kFactor == Factor::Divisor) {
        const int field = static_cast<int>((w.divisor >> 23) & 0xFF), mantissa = static_cast<int>(w.divisor & 0x7FFFFF);
        m = static_cast<std::uint32_t>(field == 0 ? mantissa : ((1 << 23) | mantissa));
        e = (field == 0 ? 1 : field) - 150;
    } else {
        m = 1;
        e = 0;
    }
}

// --------------------------------------------------------------------------------- rounding

// sign, mag (Words words) times 2^unit, with the format's row factor applied exactly, rounded.
template <class F, int Words>
__device__ std::uint32_t round_with_factor(const std::uint32_t (&mag)[Words], int unit, bool negative, bool bf16,
                                           const WeightView& w, int row) {
    if constexpr (F::kFactor == Factor::None) {
        return round_magnitude<Words>(mag, unit, negative, bf16);
    } else {
        std::uint32_t fm = 0;
        int fe           = 0;
        bool fneg        = false;
        row_factor<F>(w, row, fm, fe, fneg);
        if constexpr (F::kFactor == Factor::RowScale || F::kFactor == Factor::RowScaleF32 ||
                      F::kFactor == Factor::Multiplier) {
            std::uint32_t prod[Words + 1];
            unsigned long long carry = 0;
            bool zero                = true;
#pragma unroll
            for (int i = 0; i < Words; ++i) {
                const unsigned long long v = static_cast<unsigned long long>(mag[i]) * fm + carry;
                prod[i]                    = static_cast<std::uint32_t>(v);
                carry                      = v >> 32;
                zero                       = zero && prod[i] == 0;
            }
            prod[Words] = static_cast<std::uint32_t>(carry);
            zero        = zero && carry == 0;
            return round_magnitude<Words + 1>(prod, unit + fe, (negative != fneg) && !zero, bf16);
        } else {
            // x / (m 2^fe) = (x 2^64 / m) 2^(-64 - fe): 40 or more quotient bits beyond any nonzero x.
            std::uint32_t q[Words + 2];
            q[0] = 0;
            q[1] = 0;
#pragma unroll
            for (int i = 0; i < Words; ++i) { q[i + 2] = mag[i]; }
            const bool sticky = lossless::divide_small<Words + 2>(q, fm);
            return round_magnitude<Words + 2>(q, unit - 64 - fe, negative, bf16, sticky);
        }
    }
}

// An FP32 scale as m 2^e (subnormals: field 1).
__device__ __forceinline__ void fp32_parts(std::uint32_t bits, std::uint32_t& m, int& e) {
    const int field = static_cast<int>((bits >> 23) & 0xFF), mantissa = static_cast<int>(bits & 0x7FFFFF);
    m = static_cast<std::uint32_t>(field == 0 ? mantissa : ((1 << 23) | mantissa));
    e = (field == 0 ? 1 : field) - 150;
}

template <int Words>
__device__ __forceinline__ void multiply_small(std::uint32_t (&mag)[Words], std::uint32_t factor) {
    unsigned long long carry = 0;
#pragma unroll
    for (int i = 0; i < Words; ++i) {
        const unsigned long long v = static_cast<unsigned long long>(mag[i]) * factor + carry;
        mag[i]                     = static_cast<std::uint32_t>(v);
        carry                      = v >> 32;
    }
}

// round_with_factor with the design reference's activation scales (lossless-route.md §3b): mag 2^unit
// times the row factor and the activation multipliers (act: the row's; column: the column's), divided by
// the row divisor and the activation divisor, all exact, then rounded once. Below mag lie four padding
// words, so each division by a significand under 2^24 keeps 80 or more quotient bits beyond any nonzero
// magnitude, and a remainder in either leaves the sticky bit (floor(floor(x / a) / b) = floor(x / ab),
// exact exactly when both are); above it, three words take the multipliers' growth.
template <class F, int Words>
__device__ std::uint32_t round_scaled(const std::uint32_t (&mag)[Words], int unit, bool negative, bool bf16,
                                      const WeightView& w, int row, std::uint32_t act, std::uint32_t column) {
    constexpr int N = Words + 7;
    std::uint32_t q[N];
#pragma unroll
    for (int i = 0; i < N; ++i) { q[i] = i >= 4 && i < 4 + Words ? mag[i - 4] : 0U; }
    int e    = unit - 128;
    bool neg = negative;
    std::uint32_t fm = 0;
    int fe           = 0;
    bool fneg        = false;
    row_factor<F>(w, row, fm, fe, fneg);
    if constexpr (F::kFactor == Factor::RowScale || F::kFactor == Factor::RowScaleF32 ||
                  F::kFactor == Factor::Multiplier) {
        multiply_small(q, fm);
        e += fe;
        neg = neg != fneg;
    }
    const std::uint32_t factors[2] = {act, column};
#pragma unroll
    for (int f = 0; f < 2; ++f) {
        const std::uint32_t bits = factors[f];
        if (bits == 0) { continue; }
        std::uint32_t m = 0;
        int me          = 0;
        fp32_parts(bits, m, me);
        multiply_small(q, m);
        e += me;
        neg = neg != ((bits >> 31) != 0);
    }
    bool sticky = false;
    if constexpr (F::kFactor == Factor::Divisor) {
        sticky = lossless::divide_small<N>(q, fm);
        e -= fe;
    }
    if (w.act_div != 0) {
        std::uint32_t m = 0;
        int me          = 0;
        fp32_parts(w.act_div, m, me);
        const bool rest = lossless::divide_small<N>(q, m);
        sticky          = sticky || rest;
        e -= me;
    }
    bool zero = !sticky;
#pragma unroll
    for (int i = 0; i < N; ++i) { zero = zero && q[i] == 0; }
    return round_magnitude<N>(q, e, neg && !zero, bf16, sticky);
}

// Whether a design-reference activation scale applies to this output.
__device__ __forceinline__ bool scaled_output(const WeightView& w, std::uint32_t act, std::uint32_t column) {
    return (w.act_div | act | column) != 0;
}

__device__ __forceinline__ void store_output(void* out, int ld, int out_row, int row, int column, bool bf16,
                                             std::uint32_t bits) {
    const std::size_t i = static_cast<std::size_t>(column) * ld + out_row + row;
    if (bf16) {
        static_cast<std::uint16_t*>(out)[i] = static_cast<std::uint16_t>(bits);
    } else {
        static_cast<std::uint32_t*>(out)[i] = bits;
    }
}

// m16n8k16 integer products (each half of an m16n8k32 step's fragments), exact in int32.
__device__ __forceinline__ void mma_u8s8_k16(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1, unsigned b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_s8s8_k16(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1, unsigned b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0));
}

// --------------------------------------------------------------------------------- GEMM

struct GemmArgs {
    WeightView w;
    int row_begin;
    const std::int32_t* tile_experts; // grouped: each 64-column tile's expert (-1: no columns), its rows
    int expert_stride;                // at expert * expert_stride + row_begin
    int tile_base;                    // the chunk's first tile
    int rows;
    int k;
    int columns;
    const std::uint8_t* limbs;
    const std::uint8_t* lo_limbs;
    const std::uint8_t* tile_lo; // [column tile][block]: the tile has lo elements in the block
    const std::int16_t* exps;
    const std::int32_t* col_emax;
    const std::int32_t* col_flags;
    std::uint32_t* bitmap; // rows x columns, row-major bits (row * columns + column)
    void* out;
    int ld;
    int out_row;
    bool bf16;
    const LosslessTile* tiles;       // the tile mode: each tile's record (tile_base + blockIdx.y)
    const std::int32_t* out_columns; // the chunk's output columns (null: the column itself; -1: none)
    const long long* xsum;           // the min term's activation sums (Q4_K/Q5_K)
    const float* act_column;         // the chunk's columns' activation scales (null: none)
};

template <bool Offset>
struct OffsetShared {};

template <>
struct OffsetShared<true> {
    long long xsum[kTileTokens][2][2]; // [column][half block][grid]
    std::int32_t off_m[kTileRows][2];
    std::int32_t off_e[kTileRows][2];
};

template <int Planes, bool Offset>
struct GemmShared {
    alignas(128) std::uint8_t a[kTileTokens * kAStride];
    alignas(128) std::int8_t b[Planes][kTileRows][kBStride];
    std::int16_t exps[kTileTokens];
    std::int32_t scale_m[kTileRows][4];
    std::int32_t scale_e[kTileRows][4];
    std::int32_t row_emax[kTileRows];
    std::int32_t col_emax[kTileTokens];
    std::int32_t col_flags[kTileTokens];
    OffsetShared<Offset> off;
};

template <class F>
__global__ __launch_bounds__(kThreads, 2) void lossless_gemm_kernel(const GemmArgs args) {
    constexpr int kPlanes      = F::kPlanes;
    constexpr int kGroup       = F::kGroup;
    constexpr int kGroupsBlock = kBlockK / kGroup; // scale groups per 64-element block (1, 2 or 4)
    __shared__ GemmShared<kPlanes, F::kOffset> sh;
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp & 3, warp_n = warp >> 2;
    const int group = lane >> 2, tig = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * kTileRows, tok0 = static_cast<int>(blockIdx.y) * kTileTokens;
    int row_begin = args.row_begin;
    if (args.tile_experts != nullptr) {
        const int expert = args.tile_experts[args.tile_base + static_cast<int>(blockIdx.y)];
        if (expert < 0) { return; }
        row_begin += expert * args.expert_stride;
    }
    WeightView w = args.w;
    if (args.tiles != nullptr) {
        const LosslessTile t = args.tiles[args.tile_base + static_cast<int>(blockIdx.y)];
        if (t.record == nullptr) { return; }
        w = tile_view(args.w, t);
    }
    const int blocks = args.k / kBlockK;

    // Prologue: each row's largest scale exponent, each column's largest block exponent and flags.
    if (tid < kTileTokens) {
        const int c       = tok0 + tid;
        sh.col_emax[tid]  = c < args.columns ? args.col_emax[c] : 1;
        sh.col_flags[tid] = c < args.columns ? args.col_flags[c] : 0;
    }
    {
        const int r = tid / kRowParts, part = tid % kRowParts;
        int emax = 0;
        if constexpr (F::kScaled) {
            if (row0 + r < args.rows) {
                const int used_groups = args.k / kGroup;
                for (int g = part; g < used_groups; g += kRowParts) {
                    int m = 0, e = 1;
                    F::scale(w, row_begin + row0 + r, g, m, e);
                    if (m != 0) { emax = max(emax, e); }
                    if constexpr (F::kOffset) {
                        F::offset(w, row_begin + row0 + r, g, m, e);
                        if (m != 0) { emax = max(emax, e); }
                    }
                }
            }
#pragma unroll
            for (int o = 1; o < kRowParts; o <<= 1) { emax = max(emax, __shfl_xor_sync(0xFFFFFFFFU, emax, o)); }
        }
        if (part == 0) { sh.row_emax[r] = emax; }
    }

    // Accumulators: tile j (8 rows), value v (d0..d3: tokens group / group + 8, rows 2 tig / 2 tig + 1).
    I128 acc[kRowTiles][4];
    bool dropped[kRowTiles][4];
#pragma unroll
    for (int j = 0; j < kRowTiles; ++j) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            acc[j][v]     = I128{0ULL, 0LL};
            dropped[j][v] = false;
        }
    }

    for (int b = 0; b < blocks; ++b) {
        __syncthreads(); // the previous block's shared reads are done
        // A: 64 columns x 256 bytes of limbs, 16 bytes per copy.
        for (int i = tid; i < kTileTokens * (kBlockBytes / 16); i += kThreads) {
            const int t = i / (kBlockBytes / 16), part = i % (kBlockBytes / 16);
            std::uint8_t* dst = sh.a + t * kAStride + part * 16;
            if (tok0 + t < args.columns) {
                cp_async<16>(dst, args.limbs + (static_cast<std::size_t>(tok0 + t) * blocks + b) * kBlockBytes + part * 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
        cp_commit();
        if (tid < kTileTokens) {
            sh.exps[tid] = tok0 + tid < args.columns ? args.exps[static_cast<std::size_t>(tok0 + tid) * blocks + b] : 1;
        }
        // B: 64 rows x 64 codes per plane (four 16-code parts per row), and each row's scales of this block.
        if (tid < kTileRows * 4) {
            const int r = tid / 4, q = tid % 4;
            if (row0 + r < args.rows) {
                // Plane p of the part at &sh.b[0][r][16 q] + p kTileRows kBStride.
                F::decode16(w, row_begin + row0 + r, b, q, &sh.b[0][r][16 * q], kTileRows * kBStride, false);
            } else {
#pragma unroll
                for (int p = 0; p < kPlanes; ++p) {
                    *reinterpret_cast<uint4*>(&sh.b[p][r][16 * q]) = make_uint4(0, 0, 0, 0);
                }
            }
            if (q < kGroupsBlock) {
                int m = 0, e = 1;
                if (row0 + r < args.rows) { F::scale(w, row_begin + row0 + r, b * kGroupsBlock + q, m, e); }
                sh.scale_m[r][q] = m;
                sh.scale_e[r][q] = e;
                if constexpr (F::kOffset) {
                    int om = 0, oe = 1;
                    if (row0 + r < args.rows) { F::offset(w, row_begin + row0 + r, b * kGroupsBlock + q, om, oe); }
                    sh.off.off_m[r][q] = om;
                    sh.off.off_e[r][q] = oe;
                }
            }
        }
        if constexpr (F::kOffset) {
            if (tid < kTileTokens) {
                const bool in = tok0 + tid < args.columns;
                const long long* at = args.xsum + (static_cast<std::size_t>(tok0 + tid) * blocks + b) * 4;
#pragma unroll
                for (int i = 0; i < 4; ++i) { sh.off.xsum[tid][i / 2][i % 2] = in ? at[i] : 0; }
            }
        }
        cp_wait<0>();
        __syncthreads();

        // The block's scale groups from the limbs in sh.a, their terms `down` binades below the hi grid's.
        const auto accumulate = [&](int down) {
#pragma unroll
        for (int g = 0; g < kGroupsBlock; ++g) {
            long long S[kRowTiles][4];
#pragma unroll
            for (int j = 0; j < kRowTiles; ++j) {
#pragma unroll
                for (int v = 0; v < 4; ++v) { S[j][v] = 0; }
            }
            // (BF16's five planes stay rolled: unrolled, they spill.)
            constexpr int kPlaneUnroll = kPlanes > 3 ? 1 : kPlanes;
#pragma unroll kPlaneUnroll
            for (int p = 0; p < kPlanes; ++p) {
                // B fragments of the group's K steps for the warp's four 8-row tiles.
                constexpr int kSteps = kGroup >= 32 ? kGroup / 32 : 1;
                unsigned bfrag[kRowTiles][kSteps][2];
#pragma unroll
                for (int si = 0; si < kSteps; ++si) {
                    const int s = kGroup >= 32 ? g * kSteps + si : g / 2;
#pragma unroll
                    for (int jj = 0; jj < kRowTiles / 2; ++jj) {
                        const int row  = warp_n * kWarpRows + (2 * jj + (lane >> 4)) * 8 + (lane & 7);
                        const int kofs = s * 32 + ((lane >> 3) & 1) * 16;
                        ldmatrix_x4(bfrag[2 * jj][si][0], bfrag[2 * jj][si][1], bfrag[2 * jj + 1][si][0],
                                    bfrag[2 * jj + 1][si][1], smem_addr(&sh.b[p][row][kofs]));
                    }
                }
#pragma unroll
                for (int l = 0; l < kLimbs; ++l) {
                    int P[kRowTiles][4];
#pragma unroll
                    for (int j = 0; j < kRowTiles; ++j) {
#pragma unroll
                        for (int v = 0; v < 4; ++v) { P[j][v] = 0; }
                    }
#pragma unroll
                    for (int si = 0; si < kSteps; ++si) {
                        const int s     = kGroup >= 32 ? g * kSteps + si : g / 2;
                        const int token = warp_m * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                        const int kofs  = s * 32 + (lane >> 4) * 16;
                        unsigned a[4];
                        ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(sh.a + token * kAStride + l * kBlockK + kofs));
#pragma unroll
                        for (int j = 0; j < kRowTiles; ++j) {
                            if constexpr (kGroup >= 32) {
                                if (l == kLimbs - 1) {
                                    mma_s8(P[j][0], P[j][1], P[j][2], P[j][3], a[0], a[1], a[2], a[3], bfrag[j][si][0],
                                           bfrag[j][si][1]);
                                } else {
                                    mma_u8s8_k32(P[j][0], P[j][1], P[j][2], P[j][3], a[0], a[1], a[2], a[3],
                                                 bfrag[j][si][0], bfrag[j][si][1]);
                                }
                            } else {
                                // 16-element group: one half of the k32 step's fragments.
                                const int h = g & 1;
                                if (l == kLimbs - 1) {
                                    mma_s8s8_k16(P[j][0], P[j][1], P[j][2], P[j][3], a[2 * h], a[2 * h + 1],
                                                 bfrag[j][si][h]);
                                } else {
                                    mma_u8s8_k16(P[j][0], P[j][1], P[j][2], P[j][3], a[2 * h], a[2 * h + 1],
                                                 bfrag[j][si][h]);
                                }
                            }
                        }
                    }
#pragma unroll
                    for (int j = 0; j < kRowTiles; ++j) {
#pragma unroll
                        for (int v = 0; v < 4; ++v) { S[j][v] += static_cast<long long>(P[j][v]) << (8 * l + 7 * p); }
                    }
                }
            }
            // Terms into the accumulators. A term's shift is kShiftTop - down - (its column's exponent gap) - (its
            // row's), and usually lies in [0, 63], where the add is branch-free; a warp with any other shift
            // (bits dropped below the unit, or the 2^64 placement) takes add_shifted. The check needs only the
            // gaps' bounds over the thread's 2 columns and 8 rows.
            int col_lo = INT_MAX, col_hi = INT_MIN, row_lo = INT_MAX, row_hi = INT_MIN;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int t   = warp_m * 16 + group + half * 8;
                const int gap = sh.col_emax[t] - sh.exps[t];
                col_lo        = min(col_lo, gap);
                col_hi        = max(col_hi, gap);
            }
#pragma unroll
            for (int j = 0; j < kRowTiles; ++j) {
#pragma unroll
                for (int odd = 0; odd < 2; ++odd) {
                    const int r   = warp_n * kWarpRows + j * 8 + 2 * tig + odd;
                    const int gap = sh.row_emax[r] - sh.scale_e[r][g];
                    row_lo        = min(row_lo, gap);
                    row_hi        = max(row_hi, gap);
                }
            }
            const bool in_range = F::kShiftTop - down - col_hi - row_hi >= 0 && F::kShiftTop - down - col_lo - row_lo <= 63;
            const auto add_terms = [&](auto add) {
#pragma unroll
                for (int j = 0; j < kRowTiles; ++j) {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const int t = warp_m * 16 + group + (v >> 1) * 8;
                        const int r = warp_n * kWarpRows + j * 8 + 2 * tig + (v & 1);
                        add(j, v, S[j][v] * static_cast<long long>(sh.scale_m[r][g]),
                            F::kShiftTop - (sh.col_emax[t] - sh.exps[t]) - (sh.row_emax[r] - sh.scale_e[r][g]) - down);
                    }
                }
            };
            if (__all_sync(0xFFFFFFFFU, in_range)) {
                add_terms([&](int j, int v, long long term, int shift) {
                    lossless::add_shifted_in_range(acc[j][v], term, shift);
                });
            } else {
                add_terms([&](int j, int v, long long term, int shift) {
                    add_shifted(acc[j][v], term, shift, dropped[j][v]);
                });
            }
            if constexpr (F::kOffset) {
#pragma unroll
                for (int j = 0; j < kRowTiles; ++j) {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const int t = warp_m * 16 + group + (v >> 1) * 8;
                        const int r = warp_n * kWarpRows + j * 8 + 2 * tig + (v & 1);
                        // The min term: the group's activation sum on this pass's grid times -mant(dmin) min_sub.
                        const long long o = sh.off.xsum[t][g][down != 0 ? 1 : 0] *
                                            static_cast<long long>(sh.off.off_m[r][g]);
                        add_shifted(acc[j][v], o,
                                    F::kShiftTop - (sh.col_emax[t] - sh.exps[t]) - (sh.row_emax[r] - sh.off.off_e[r][g]) -
                                        down,
                                    dropped[j][v]);
                    }
                }
            }
        }
        };
        // Another grid's limbs of this block into sh.a (the B tile and scales stay).
        const auto load_a = [&](const std::uint8_t* limbs) {
            __syncthreads();
            for (int i = tid; i < kTileTokens * (kBlockBytes / 16); i += kThreads) {
                const int t = i / (kBlockBytes / 16), part = i % (kBlockBytes / 16);
                std::uint8_t* dst = sh.a + t * kAStride + part * 16;
                if (tok0 + t < args.columns) {
                    cp_async<16>(dst, limbs + (static_cast<std::size_t>(tok0 + t) * blocks + b) * kBlockBytes + part * 16);
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                }
            }
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        };
        // Passes: hi x hi; with the tile's lo activations, lo x hi; with the rows' lo-grid weights, the
        // activations sh.a holds x lo, then (with lo activations) hi x lo. One accumulate site.
        const bool a_lo = args.tile_lo[static_cast<std::size_t>(blockIdx.y) * blocks + b] != 0;
        bool w_lo       = false;
        if constexpr (F::kWeightLo) { w_lo = w.tile_lo_w[static_cast<std::size_t>(row0 / kTileRows) * blocks + b] != 0; }
        const int passes = 1 + (a_lo ? 1 : 0) + (w_lo ? (a_lo ? 2 : 1) : 0);
#pragma unroll 1
        for (int pass = 0; pass < passes; ++pass) {
            int down = 0;
            if (pass == 1 && a_lo) {
                load_a(args.lo_limbs);
                down = kLoShift;
            } else if (pass == (a_lo ? 2 : 1)) {
                if constexpr (F::kWeightLo) {
                    // The rows' lo-grid weights of this block.
                    __syncthreads();
                    const int r = tid / 4, q = tid % 4;
                    if (tid < kTileRows * 4 && row0 + r < args.rows) {
                        F::decode16(w, row_begin + row0 + r, b, q, &sh.b[0][r][16 * q], kTileRows * kBStride, true);
                    }
                    __syncthreads();
                }
                down = a_lo ? 2 * kLoShift : kLoShift;
            } else if (pass == 3) {
                load_a(args.limbs);
                down = kLoShift;
            }
            accumulate(down);
        }
    }

    // Round, or flag for the exact fallback.
#pragma unroll
    for (int j = 0; j < kRowTiles; ++j) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int t = warp_m * 16 + group + (v >> 1) * 8;
            const int r = warp_n * kWarpRows + j * 8 + 2 * tig + (v & 1);
            const int column = tok0 + t, row = row0 + r;
            if (column >= args.columns || row >= args.rows) { continue; }
            const int out_column = args.out_columns != nullptr ? args.out_columns[column] : column;
            if (out_column < 0) { continue; }
            bool inexact_row = false;
            if constexpr (F::kWeightLo) { inexact_row = w.row_inexact[row_begin - w.row_base + row] != 0; }
            if (dropped[j][v] || sh.col_flags[t] != 0 || inexact_row) {
                const std::size_t bit = static_cast<std::size_t>(row) * args.columns + column;
                atomicOr(args.bitmap + bit / 32, 1U << (bit % 32));
                continue;
            }
            const int unit = sh.col_emax[t] + sh.row_emax[r] - F::kUnitBias;
            std::uint32_t mag[4];
            const bool negative = lossless::magnitude_i128(acc[j][v], mag);
            const std::uint32_t act = w.act_mul[(row_begin + row) & 1];
            const std::uint32_t col = args.act_column != nullptr ? __float_as_uint(args.act_column[column]) : 0U;
            store_output(args.out, args.ld, args.out_row, row, out_column, args.bf16,
                         scaled_output(w, act, col)
                             ? round_scaled<F, 4>(mag, unit, negative, args.bf16, w, row_begin + row, act, col)
                             : round_with_factor<F, 4>(mag, unit, negative, args.bf16, w, row_begin + row));
        }
    }
}

// --------------------------------------------------------------------------------- exact fallback

struct FallbackArgs {
    WeightView w;
    int row_begin;
    const std::int32_t* tile_experts;
    int expert_stride;
    int tile_base;
    int rows;
    int k;
    int columns;
    const std::uint16_t* x; // BF16 [K, T]: this chunk's columns, or (in_columns) the whole input
    const std::int32_t* in_columns;
    const std::uint32_t* bitmap;
    void* out;
    int ld;
    int out_row;
    bool bf16;
    const LosslessTile* tiles;
    const std::int32_t* out_columns;
    const float* act_column;
};

__device__ __forceinline__ std::uint32_t non_finite_result(bool nan, bool pos_inf, bool neg_inf, bool bf16) {
    if (nan || (pos_inf && neg_inf)) { return bf16 ? 0x7FC0U : 0x7FC00000U; }
    return (neg_inf ? (bf16 ? 0x8000U : 0x80000000U) : 0U) | (bf16 ? 0x7F80U : 0x7F800000U);
}

// One warp per flagged output: a 352-bit fixed-point accumulator (int64 words of 32-bit chunks, carries
// deferred) per lane, then summed across the warp.
template <class F>
__global__ void lossless_fallback_kernel(const FallbackArgs args) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warps = static_cast<int>(gridDim.x * blockDim.x) >> 5;
    const std::size_t total = static_cast<std::size_t>(args.rows) * args.columns;
    const std::size_t words = (total + 31) / 32;
    for (std::size_t wi = static_cast<std::size_t>((blockIdx.x * blockDim.x + threadIdx.x) >> 5); wi < words;
         wi += static_cast<std::size_t>(warps)) {
        std::uint32_t pending = args.bitmap[wi];
        while (pending != 0) {
            const int bit = __ffs(pending) - 1;
            pending &= pending - 1;
            const std::size_t index = wi * 32 + bit;
            if (index >= total) { break; }
            const int row = static_cast<int>(index / args.columns), column = static_cast<int>(index % args.columns);
            const int row_begin =
                args.row_begin + (args.tile_experts != nullptr
                                      ? args.tile_experts[args.tile_base + column / kTileTokens] * args.expert_stride
                                      : 0);
            const int wrow = row_begin + row;
            WeightView w   = args.w;
            if (args.tiles != nullptr) { w = tile_view(args.w, args.tiles[args.tile_base + column / kTileTokens]); }
            const int source        = args.in_columns != nullptr ? args.in_columns[column] : column;
            const int out_column    = args.out_columns != nullptr ? args.out_columns[column] : column;
            const std::uint16_t* xc = args.x + static_cast<std::size_t>(source < 0 ? 0 : source) * args.k;
            Wide<F::kWide> acc;
            acc.clear();
            int nan = 0, pos_inf = 0, neg_inf = 0;
            for (int k = lane; k < args.k; k += 32) {
                const std::uint16_t xb = source < 0 ? 0 : xc[k];
                const int field = (xb >> 7) & 0xFF, mant = xb & 0x7F;
                int W = 0, wexp = 0;
                F::element(w, wrow, k, W, wexp);
                const bool w_special = wexp == kSpecialExp;
                if (field == 255 || w_special) {
                    // NaN, or infinity times zero, is NaN; otherwise the infinity's sign times the other's.
                    const bool x_zero = field == 0 && mant == 0;
                    int Wv            = W; // the weight's sign (zero: a zero weight)
                    if constexpr (F::kOffset) {
                        int W2 = 0, e2 = 0;
                        F::element_offset(w, wrow, k, W2, e2);
                        const double v = ldexp(static_cast<double>(W), wexp) + ldexp(static_cast<double>(W2), e2);
                        Wv             = v > 0.0 ? 1 : (v < 0.0 ? -1 : 0);
                    }
                    if ((field == 255 && mant != 0) || Wv == 0 || x_zero) {
                        nan = 1;
                    } else if (((xb >> 15) != 0) != (Wv < 0)) {
                        neg_inf = 1;
                    } else {
                        pos_inf = 1;
                    }
                    continue;
                }
                const int M = field == 0 ? mant : (128 | mant);
                if (M == 0) { continue; }
                const int e = field == 0 ? 1 : field;
                const auto add = [&](int Wt, int wexpt) {
                    if (Wt == 0) { return; }
                    long long p = static_cast<long long>(Wt) * M;
                    if ((xb >> 15) != 0) { p = -p; }
                    if constexpr (F::kSplit) { // |p| < 2^34: two terms below 2^24 and 2^10
                        const long long hi = p >> 24;
                        acc.add(p - hi * (1LL << 24), e - 134 + wexpt + F::kWideBase);
                        acc.add(hi, e - 110 + wexpt + F::kWideBase);
                    } else { // |p| < 2^31
                        acc.add(p, e - 134 + wexpt + F::kWideBase);
                    }
                };
                add(W, wexp);
                if constexpr (F::kOffset) {
                    F::element_offset(w, wrow, k, W, wexp);
                    add(W, wexp);
                }
            }
            acc.warp_sum();
            nan     = __reduce_or_sync(0xFFFFFFFFU, nan);
            pos_inf = __reduce_or_sync(0xFFFFFFFFU, pos_inf);
            neg_inf = __reduce_or_sync(0xFFFFFFFFU, neg_inf);
            if (lane != 0 || out_column < 0) { continue; }
            std::uint32_t result;
            if (nan || pos_inf || neg_inf) {
                result = non_finite_result(nan, pos_inf, neg_inf, args.bf16);
            } else {
                std::uint32_t mag[F::kWide];
                const bool negative     = acc.finish(mag);
                const std::uint32_t act = w.act_mul[wrow & 1];
                const std::uint32_t col = args.act_column != nullptr ? __float_as_uint(args.act_column[column]) : 0U;
                result = scaled_output(w, act, col)
                             ? round_scaled<F, F::kWide>(mag, -F::kWideBase, negative, args.bf16, w, wrow, act, col)
                             : round_with_factor<F, F::kWide>(mag, -F::kWideBase, negative, args.bf16, w, wrow);
            }
            store_output(args.out, args.ld, args.out_row, row, out_column, args.bf16, result);
        }
    }
}

// --------------------------------------------------------------------------------- launch

struct Call {
    const Tensor* x;
    WeightView w;
    std::int32_t row_begin;
    const std::int32_t* tile_experts;
    int expert_stride;
    std::int32_t rows;
    Tensor* out;
    std::int32_t out_row;
    std::int32_t columns; // the product's columns (x's own, unless in_columns maps them)
    QType qtype;          // GGUF: the block format
    const LosslessTile* tiles       = nullptr; // the tile mode
    const std::int32_t* in_columns  = nullptr;
    const std::int32_t* out_columns = nullptr;
    const float* act_column         = nullptr; // the activation scales of x's columns (w.act_div: the divisor)
};

// The parts of a chunk a launch runs: `encode` encodes the chunk's activation columns (after zeroing the
// buffers the encode accumulates into); `xsum` makes that encode also form the half-block sums a K-quant
// offset format reads. Projections of one input share one encode: the first encodes (with xsum when any
// of them reads it), the others reuse the workspace's encoding.
struct ChunkStage {
    bool encode = true;
    bool xsum   = false;
};

template <class F>
void launch_chunk(const Call& c, std::int32_t first, std::int32_t columns, std::uint8_t* base, const Layout& l,
                  cudaStream_t stream, ChunkStage stage = ChunkStage{true, F::kOffset}) {
    const int k      = c.x->ne[0];
    const int blocks = k / kBlockK;
    auto* limbs      = base + l.limbs;
    auto* lo_limbs   = base + l.lo_limbs;
    auto* tile_lo    = base + l.tile_lo;
    auto* exps       = reinterpret_cast<std::int16_t*>(base + l.exps);
    auto* col_emax   = reinterpret_cast<std::int32_t*>(base + l.col_emax);
    auto* col_flags  = reinterpret_cast<std::int32_t*>(base + l.col_flags);
    auto* bitmap     = reinterpret_cast<std::uint32_t*>(base + l.bitmap);
    const auto* xbase           = static_cast<const std::uint16_t*>(c.x->data);
    const std::uint16_t* xs     = c.in_columns != nullptr ? xbase : xbase + static_cast<std::size_t>(first) * k;
    const std::int32_t* in_map  = c.in_columns != nullptr ? c.in_columns + first : nullptr;
    const std::int32_t* out_map = c.out_columns != nullptr ? c.out_columns + first : nullptr;
    long long* xsum             = F::kOffset ? reinterpret_cast<long long*>(base + l.xsum) : nullptr;
    if (stage.encode) {
        // tile_lo, exps, col_emax and col_flags lie together: one memset zeroes them.
        CUDA_CHECK(cudaMemsetAsync(tile_lo, 0, l.bitmap - l.tile_lo, stream));
        const int warps_total = blocks * columns;
        const int threads     = 256;
        const int grid        = (warps_total * 32 + threads - 1) / threads;
        lossless_encode_kernel<<<grid, threads, 0, stream>>>(
            xs, in_map, k, columns, limbs, lo_limbs, tile_lo, exps, col_emax, col_flags,
            stage.xsum ? reinterpret_cast<long long*>(base + l.xsum) : nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemsetAsync(bitmap, 0, (static_cast<std::size_t>(c.rows) * columns + 31) / 32 * sizeof(std::uint32_t),
                               stream));
    const bool bf16 = c.out->dtype == DType::BF16;
    void* out_data  = out_map != nullptr ? c.out->data
                                         : static_cast<std::uint8_t*>(c.out->data) +
                                               static_cast<std::size_t>(first) * c.out->ne[0] * dtype_size(c.out->dtype);
    {
        GemmArgs a{};
        a.w         = c.w;
        a.row_begin = c.row_begin;
        a.tile_experts  = c.tile_experts;
        a.expert_stride = c.expert_stride;
        a.tile_base = first / kTileTokens;
        a.rows      = c.rows;
        a.k         = k;
        a.columns   = columns;
        a.limbs     = limbs;
        a.lo_limbs  = lo_limbs;
        a.tile_lo   = tile_lo;
        a.exps      = exps;
        a.col_emax  = col_emax;
        a.col_flags = col_flags;
        a.bitmap    = bitmap;
        a.out       = out_data;
        a.ld        = c.out->ne[0];
        a.out_row   = c.out_row;
        a.bf16      = bf16;
        a.tiles       = c.tiles;
        a.out_columns = out_map;
        a.xsum        = xsum;
        a.act_column  = c.act_column != nullptr ? c.act_column + first : nullptr;
        const dim3 grid((c.rows + kTileRows - 1) / kTileRows, (columns + kTileTokens - 1) / kTileTokens);
        lossless_gemm_kernel<F><<<grid, kThreads, 0, stream>>>(a);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        FallbackArgs f{};
        f.w         = c.w;
        f.row_begin = c.row_begin;
        f.tile_experts  = c.tile_experts;
        f.expert_stride = c.expert_stride;
        f.tile_base = first / kTileTokens;
        f.rows      = c.rows;
        f.k         = k;
        f.columns   = columns;
        f.x           = xs;
        f.in_columns  = in_map;
        f.tiles       = c.tiles;
        f.out_columns = out_map;
        f.act_column  = c.act_column != nullptr ? c.act_column + first : nullptr;
        f.bitmap      = bitmap;
        f.out       = out_data;
        f.ld        = c.out->ne[0];
        f.out_row   = c.out_row;
        f.bf16      = bf16;
        const std::size_t words = (static_cast<std::size_t>(c.rows) * columns + 31) / 32;
        const int threads = 256;
        const int grid    = static_cast<int>(std::min<std::size_t>((words * 32 + threads - 1) / threads, 4096));
        lossless_fallback_kernel<F><<<grid, threads, 0, stream>>>(f);
        CUDA_CHECK(cudaGetLastError());
    }
}

enum class Route { None, Q2, Q4, Q5, Q6, Q8, Nvfp4, Fp8Row, Fp8RowF32, Bf16, Mxfp8, Ggml, Rg16, Fp8B128 };

// The GGUF formats' launches (lossless_linear_ggml_k.cu: the K-quants and IQ4; lossless_linear_ggml_i.cu:
// IQ2/IQ3 and Q2_0).
// `stage` null: the format's own (encode, with xsum when it reads it).
bool launch_ggml_k(QType q, const Call& c, std::int32_t first, std::int32_t count, std::uint8_t* base,
                   const Layout& l, cudaStream_t stream, const ChunkStage* stage = nullptr);
bool launch_ggml_i(QType q, const Call& c, std::int32_t first, std::int32_t count, std::uint8_t* base,
                   const Layout& l, cudaStream_t stream, const ChunkStage* stage = nullptr);

template <class F>
void launch_chunk_stage(const Call& c, std::int32_t first, std::int32_t count, std::uint8_t* base, const Layout& l,
                        cudaStream_t stream, const ChunkStage* stage) {
    launch_chunk<F>(c, first, count, base, l, stream, stage != nullptr ? *stage : ChunkStage{true, F::kOffset});
}

// Whether a GGUF block format's projection reads the half-block activation sums (GgmlFormat::kOffset).
inline bool ggml_reads_xsum(QType q) noexcept { return q == QType::GGML_Q4_K || q == QType::GGML_Q5_K; }

inline bool is_ggml_format(QType q) noexcept {
    return q == QType::GGML_Q2_0 || q == QType::GGML_Q4_K || q == QType::GGML_Q5_K || q == QType::GGML_Q6_K ||
           q == QType::GGML_IQ2_S || q == QType::GGML_IQ3_XXS || q == QType::GGML_IQ3_S || q == QType::GGML_IQ4_XS ||
           q == QType::GGML_IQ4_NL;
}

inline Route route_of(const Weight& w) noexcept {
    switch (w.qtype) {
    case QType::Q2_G64_FP16: return w.layout == QuantLayout::RowSplit ? Route::Q2 : Route::None;
    case QType::Q4_G64_FP16: return w.layout == QuantLayout::RowSplit ? Route::Q4 : Route::None;
    case QType::Q5_G64_FP16: return w.layout == QuantLayout::RowSplit ? Route::Q5 : Route::None;
    case QType::Q6_G64_FP16: return w.layout == QuantLayout::RowSplit ? Route::Q6 : Route::None;
    case QType::Q8_G32_FP16: return w.layout == QuantLayout::RowSplit ? Route::Q8 : Route::None;
    case QType::NVFP4:
        return w.layout == QuantLayout::BlockScaleK16M128x4 && w.n % 128 == 0 && w.k % 64 == 0 &&
                       w.weight_scale_divisor > 0.0F
                   ? Route::Nvfp4
                   : Route::None;
    case QType::FP8_E4M3FN_ROW_BF16: return w.layout == QuantLayout::RowScale ? Route::Fp8Row : Route::None;
    case QType::FP8_E4M3FN_ROW_F32: return w.layout == QuantLayout::RowScale ? Route::Fp8RowF32 : Route::None;
    // A dense nvfp4_rg16_v1 matrix (the Qwen3.5 NVFP4 head): the expert record's matrix layout, one
    // FP32 multiplier for every row.
    case QType::NVFP4_MUL:
        return w.layout == QuantLayout::Rg16 && w.n % 16 == 0 && w.k % 64 == 0 && std::isfinite(w.weight_multiplier)
                   ? Route::Rg16
                   : Route::None;
    case QType::BF16: return w.layout == QuantLayout::Contiguous ? Route::Bf16 : Route::None;
    case QType::FP8_E4M3FN_G32_E8M0:
        return w.layout == QuantLayout::G32ScaleRows16 && w.k % 32 == 0 ? Route::Mxfp8 : Route::None;
    default: return is_ggml_format(w.qtype) && is_ggml_matrix_layout(w.layout) ? Route::Ggml : Route::None;
    }
}

inline WeightView view_of(const Weight& weight, Route route) {
    WeightView v{};
    v.codes  = static_cast<const std::uint8_t*>(weight.qdata);
    v.high   = static_cast<const std::uint8_t*>(weight.qhigh);
    v.scales = weight.scales;
    v.k      = weight.k;
    switch (route) {
    case Route::Q2: v.row_bytes = weight.padded_shape[1] / 4; v.groups = weight.padded_shape[1] / 64; break;
    case Route::Q4:
    case Route::Q5:
    case Route::Q6: v.row_bytes = weight.padded_shape[1] / 2; v.groups = weight.padded_shape[1] / 64; break;
    case Route::Q8: v.row_bytes = weight.padded_shape[1]; v.groups = weight.padded_shape[1] / 32; break;
    case Route::Nvfp4: {
        v.row_bytes = weight.k / 2;
        v.groups    = weight.k / 16;
        std::memcpy(&v.divisor, &weight.weight_scale_divisor, sizeof(v.divisor));
        break;
    }
    case Route::Fp8Row:
    case Route::Fp8RowF32: v.row_bytes = weight.k; v.groups = 1; break;
    case Route::Rg16: {
        v.groups = weight.k / 16;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &weight.weight_multiplier, sizeof(bits));
        v.alpha[0] = v.alpha[1] = bits;
        break;
    }
    case Route::Mxfp8: v.row_bytes = weight.k; break;
    case Route::Ggml:
        v.row_bytes = static_cast<int>(ggml_row_bytes(weight.qtype, weight.k));
        v.tiles16   = weight.layout == QuantLayout::GgmlTiles16 ? 1 : 0;
        break;
    default: break;
    }
    return v;
}

inline void check_x(const Tensor& x, const Weight& weight) {
    require(x.dtype == DType::BF16 && x.ne[1] > 0 && x.ne[2] == 1 && x.ne[3] == 1 && x.nb[0] == 2 &&
                x.nb[1] == static_cast<std::int64_t>(x.ne[0]) * 2,
            "x must be contiguous BF16 [K, T]");
    require(x.ne[0] % kBlockK == 0 && weight.k == x.ne[0], "K must match the weight and be a multiple of 64");
}

inline void check_out(const Tensor& out, int columns, int out_row, int rows) {
    require((out.dtype == DType::BF16 || out.dtype == DType::FP32) && out.ne[1] == columns && out_row >= 0 &&
                out_row + rows <= out.ne[0] &&
                out.nb[1] == static_cast<std::int64_t>(out.ne[0] * dtype_size(out.dtype)),
            "out must be contiguous BF16 or FP32 [N_out, T] with the rows inside");
}

} // namespace infernix::ops::lossless_linear_impl
