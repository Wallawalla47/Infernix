#pragma once

// The reference route's dot products (include/infernix/ops/precise_linear.h). Each thread of a CTA owns
// R weight rows and the CTA's up to kDotColumns activation columns and walks k in ascending order:
//  - every product w * x is formed exactly in binary32: BF16 x times a weight of at most 16 significant
//    bits (Products::Exact), a binary32 weight whose product's rounding error an FMA recovers
//    (Products::Split), or an integer code whose group scale and offset multiply the group's sums
//    (Products::Grouped);
//  - the products of each block (kDotBlockChunks chunks of 8, or a Grouped decoder's group) are summed
//    as an unevaluated binary32 pair (two-sum, Ogita-Rump-Oishi Sum2), and the blocks join the total
//    in double-binary32 arithmetic (Joldes-Muller-Popescu AccurateDWPlusDW and DWTimesFP3);
//  - the result is the total's two parts added in binary64 (exactly), which the caller rounds once.
// The result differs from the exact sum by at most (4096 + 2 K) u^2 times the sum of the terms'
// magnitudes (u = 2^-24; a Grouped term is scale * code * x or offset * x), unless a product or a
// partial sum leaves binary32's normal range. An output's bits therefore depend only on its row's
// weights, its column and K, whatever the launch shape. The CTA stages its columns as binary32 in K
// slices of kDotSlice elements; every thread of the CTA must call dot().

#include <cstdint>

namespace infernix::ops::precise {

constexpr int kDotColumns     = 8;
constexpr int kDotSlice       = 256;
constexpr int kDotBlockChunks = 4; // Exact and Split: chunks of 8 summed as a pair before joining the total

// How a decoder's weights meet x (each decoder in precise_decode.cuh declares kProducts).
enum class Products {
    Exact,   // w has at most 16 significant bits: w * x is exact for every BF16 x
    Split,   // w is any binary32 value: w * x may round, and fma(w, x, -w * x) is its error
    Grouped, // w = scale * code - offset over kGroupChunks chunks; integer codes of at most 8 bits
};

// One chunk of 8 consecutive weights of a row: Exact and Split decoders fill w with the weights,
// Grouped decoders with the codes and their group's scale and offset (kOffset decoders only).
struct Weights8 {
    float w[8];
    float scale;
    float offset;
};

// An unevaluated sum hi + lo with |lo| <= ulp(hi) / 2.
struct Df {
    float hi;
    float lo;
};

// s + e == a + b exactly, s = fl(a + b) (Knuth).
__device__ __forceinline__ void two_sum(float a, float b, float& s, float& e) {
    s              = __fadd_rn(a, b);
    const float bb = __fsub_rn(s, a);
    e              = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb));
}

// The same when a == 0 or exponent(a) >= exponent(b) (Dekker).
__device__ __forceinline__ void fast_two_sum(float a, float b, float& s, float& e) {
    s = __fadd_rn(a, b);
    e = __fsub_rn(b, __fsub_rn(s, a));
}

// AccurateDWPlusDW: relative error at most 3 u^2 / (1 - 4 u).
__device__ __forceinline__ Df df_add(Df x, Df y) {
    float sh, sl, th, tl, vh, vl;
    two_sum(x.hi, y.hi, sh, sl);
    two_sum(x.lo, y.lo, th, tl);
    fast_two_sum(sh, __fadd_rn(sl, th), vh, vl);
    Df z;
    fast_two_sum(vh, __fadd_rn(tl, vl), z.hi, z.lo);
    return z;
}

// DWTimesFP3: relative error at most 2 u^2.
__device__ __forceinline__ Df df_mul(Df x, float y) {
    const float ch = __fmul_rn(x.hi, y);
    const float cl = __fmaf_rn(x.lo, y, __fmaf_rn(x.hi, y, -ch));
    Df z;
    fast_two_sum(ch, cl, z.hi, z.lo);
    return z;
}

// A block's pair (hi, lo), normalized.
__device__ __forceinline__ Df df_of(float hi, float lo) {
    Df z;
    two_sum(hi, lo, z.hi, z.lo);
    return z;
}

struct DotShared {
    alignas(16) float x[kDotColumns][kDotSlice];
    Df xsum[kDotColumns][kDotSlice / 8]; // each 8-element part's sum (Grouped decoders with offsets)
};

// column(c, chunk, v) loads elements [8 chunk, 8 chunk + 8) of column c < count as binary32 (exactly);
// rows(q, chunk, w) decodes those of this thread's row q < R (called only when `active`). Decoder
// supplies kProducts, kGroupChunks (the chunks of a block: kDotBlockChunks, or a Grouped decoder's
// group, at most 8) and kOffset (Grouped decoders with offsets). k is a multiple of 8.
template <int R, class Decoder, class Column, class Rows>
__device__ __forceinline__ void dot(int k, int count, const Column& column, const Rows& rows, bool active,
                                    DotShared& shared, double (&acc)[R][kDotColumns]) {
    constexpr Products kMode   = Decoder::kProducts;
    constexpr bool kOffset     = Decoder::kOffset;
    constexpr int kFlushChunks = Decoder::kGroupChunks;
    static_assert(kFlushChunks >= 1 && kFlushChunks <= 8 && (kMode == Products::Grouped || !kOffset));
    Df total[R][kDotColumns];
    float hi[R][kDotColumns], lo[R][kDotColumns];
#pragma unroll
    for (int q = 0; q < R; ++q) {
#pragma unroll
        for (int c = 0; c < kDotColumns; ++c) {
            total[q][c] = {0.0F, 0.0F};
            hi[q][c]    = 0.0F;
            lo[q][c]    = 0.0F;
        }
    }
    constexpr int kParts = kDotSlice / 8;
    const int chunks     = k / 8;
    for (int s = 0; s < chunks; s += kParts) {
        for (int i = static_cast<int>(threadIdx.x); i < kDotColumns * kParts; i += static_cast<int>(blockDim.x)) {
            const int c = i / kParts, part = i % kParts, chunk = s + part;
            float v[8];
            if (c < count && chunk < chunks) {
                column(c, chunk, v);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = 0.0F; }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) { shared.x[c][8 * part + e] = v[e]; }
            if constexpr (kOffset) {
                float h = 0.0F, l = 0.0F;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float err;
                    two_sum(h, v[e], h, err);
                    l = __fadd_rn(l, err);
                }
                shared.xsum[c][part] = df_of(h, l);
            }
        }
        __syncthreads();
        if (active) {
            const int parts = min(kParts, chunks - s);
            for (int part = 0; part < parts; ++part) {
                const int chunk  = s + part;
                const bool flush = (chunk + 1) % kFlushChunks == 0 || chunk + 1 == chunks;
#pragma unroll
                for (int q = 0; q < R; ++q) {
                    Weights8 w;
                    rows(q, chunk, w);
                    // Every column, used or not: the staging zeroed columns from count on, and their
                    // sums are never read (no per-column branch in the loop).
#pragma unroll
                    for (int c = 0; c < kDotColumns; ++c) {
                        const float4* x4 = reinterpret_cast<const float4*>(&shared.x[c][8 * part]);
                        const float4 xa = x4[0], xb = x4[1];
                        const float x[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            const float p = __fmul_rn(w.w[e], x[e]);
                            float err;
                            two_sum(hi[q][c], p, hi[q][c], err);
                            lo[q][c] = __fadd_rn(lo[q][c], err);
                            if constexpr (kMode == Products::Split) {
                                lo[q][c] = __fadd_rn(lo[q][c], __fmaf_rn(w.w[e], x[e], -p));
                            }
                        }
                        if constexpr (kOffset) {
                            total[q][c] = df_add(total[q][c], df_mul(shared.xsum[c][part], -w.offset));
                        }
                        if (flush) {
                            const Df block = df_of(hi[q][c], lo[q][c]);
                            if constexpr (kMode == Products::Grouped) {
                                total[q][c] = df_add(total[q][c], df_mul(block, w.scale));
                            } else {
                                total[q][c] = df_add(total[q][c], block);
                            }
                            hi[q][c] = 0.0F;
                            lo[q][c] = 0.0F;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
#pragma unroll
        for (int c = 0; c < kDotColumns; ++c) {
            acc[q][c] = static_cast<double>(total[q][c].hi) + static_cast<double>(total[q][c].lo);
        }
    }
}

} // namespace infernix::ops::precise
