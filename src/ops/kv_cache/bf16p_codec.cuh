#pragma once

// The packed BF16 KV row (KvCacheStorage::Bf16Packed; docs/maintainer/paged-kv-cache.md §4.3): UZ-40's bf16_p12 form
// (include/infernix/ops/bf16_p12.h), lossless, chosen per row when the row is appended. A row (one K or V head vector,
// 256 values) takes 384 bytes, three whole 128-byte lines, against BF16's 512:
//
//   [0, 256)    code[d] = sign << 7 | the seven fraction bits of value d
//   [256, 384)  nibble[d] (d even: the low nibble of byte 256 + d / 2):
//                 0..13  exponent base + nibble
//                 14     exponent 0 (a zero or a subnormal: the code byte is the value)
//                 15     an escape: its exponent is in the side record
//
// and an 8-byte side record in the storage's scale plane:
//
//   0           base (the window of normal exponents is [base, base + 13])
//   1           bits 0-2: n, the escapes (0..kBf16pEscapes); bit 7: some value has exponent 0
//   [2, 8)      the escapes' exponents in column order; unused bytes zero
//
// An escape is a value outside the window: an infinity or NaN, or a normal value too large or too small for it. A V row
// holds the BF16 source, which readers convert to FP16 exactly as the bf16 append does, so both storages give
// attention the same K and V bits (readers: softmax_attention/dense/causal_cache/bf16/packed_tile_io.cuh). An all-zero
// record with zero bytes (a page never written, as the graph capture's zeroed pages) decodes to zeros.
//
// The window: a row whose normal exponents span at most 14 binades takes base = clamp(lowest, 1, 240); a wider row
// takes the lowest base of the windows holding most of them (base <= 240: a decoder's placeholder exponent, base + 15
// for an escape, stays inside the 8-bit field). A row with more than kBf16pEscapes escapes has no packed
// form: the append stores it with its first kBf16pEscapes escapes and reports it (bf16p_encode_row returns false), and
// the request that appended it fails, as its output would not be the bf16 storage's.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kBf16pRowValues      = 256;
inline constexpr int kBf16pRowBytes       = 384; // a stored row
inline constexpr int kBf16pRowUnits       = kBf16pRowBytes / 16;
inline constexpr int kBf16pNibbles        = 256; // the nibbles' offset in a row
inline constexpr int kBf16pSideBytes      = 8;
inline constexpr int kBf16pEscapes        = 6;
inline constexpr unsigned kBf16pWindow    = 14; // binades of normal exponents
inline constexpr unsigned kBf16pZeroCode  = 14;
inline constexpr unsigned kBf16pEscapeCode = 15;

namespace detail {

// prmt.b32 in its default mode: a selector nibble with bit 3 set gives the selected byte's sign in all eight bits.
__device__ __forceinline__ unsigned bf16p_prmt(unsigned a, unsigned b, unsigned selector) {
    unsigned r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(selector));
    return r;
}

// Value pair j of a lane's chunk (values 2 j, 2 j + 1, the low and high halves of output word j): its code bytes as
// [code, sign, code, sign] (bytes 0, 1 or 2, 3 of code word j / 2), and its exponent nibbles as [nibble, 0, nibble,
// 0] from lo (the even values' nibbles a byte) and hi (the odd values').
template <int J> inline constexpr unsigned kBf16pCodeSelector = J % 2 == 0 ? 0x9180U : 0xB3A2U;
template <int J> inline constexpr unsigned kBf16pNibbleSelector = (J | (J | 8) << 4 | (J + 4) << 8 | (J + 12) << 12);

// The values of a lane's chunk its base does not give: a value of exponent 0 (nibble 14) keeps its code byte alone,
// and an escape (nibble 15) takes the side's next exponent in column order; in the BF16 words (`F16` false: bits 7..14
// of the half) or the FP16 words a V row gives attention (`F16` true: from the decoded sign and fraction, exactly as
// bf16x2_bits_to_f16x2_bits converts; exponent 0 is below FP16's subnormals, so a signed zero). The whole warp calls it
// for one row (the escapes' order is a warp scan).
template <bool F16>
__device__ __forceinline__ void bf16p_patch(unsigned (&w)[4], unsigned nib, uint2 side, int lane) {
    // Bit 4 i: value i is an escape (nibble 15) or of exponent 0 (nibble 14).
    const unsigned high    = (nib >> 1) & (nib >> 2) & (nib >> 3) & 0x11111111U;
    const unsigned escaped = nib & high, zero = ~nib & high;
    if (zero != 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((zero >> (4 * i)) & 1U) {
                w[i >> 1] &= F16 ? ~(0x7FFFU << (16 * (i & 1))) : ~(0xFFU << (7 + 16 * (i & 1)));
            }
        }
    }
    if (((side.x >> 8) & 7U) == 0) { return; } // uniform: the row's record
    const unsigned mine = __popc(escaped);
    unsigned index      = mine;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const unsigned n = __shfl_up_sync(0xFFFFFFFFU, index, d);
        if (lane >= d) { index += n; }
    }
    index -= mine;
    const unsigned long long exponents = (static_cast<unsigned long long>(side.y) << 32 | side.x) >> 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((escaped >> (4 * i)) & 1U) {
            // Past the record's escapes only in a row that has no packed form (its request fails).
            const unsigned e = index < kBf16pEscapes ? static_cast<unsigned>(exponents >> (8 * index)) & 0xFFU : 0U;
            ++index;
            if constexpr (F16) {
                const unsigned shift = 16 * (i & 1);
                const unsigned half  = (w[i >> 1] >> shift) & 0xFFFFU;
                const unsigned sign = half & 0x8000U, fraction = (half >> 3) & 0x7FU;
                const unsigned f16  = e >= 113 && e <= 142
                                          ? sign | ((e - 112U) << 10) | (fraction << 3)
                                          : bf16x2_bits_to_f16x2_bits(sign | (e << 7) | fraction) & 0xFFFFU;
                w[i >> 1]           = (w[i >> 1] & ~(0xFFFFU << shift)) | (f16 << shift);
            } else {
                const unsigned shift = 7 + 16 * (i & 1);
                w[i >> 1]            = (w[i >> 1] & ~(0xFFU << shift)) | (e << shift);
            }
        }
    }
}

} // namespace detail

// Lane L's chunk (values 8 L .. 8 L + 7) of a packed row - its code bytes `code` (row bytes 8 L ..) and exponent
// nibbles `nib` (row bytes 256 + 4 L ..), the row's side record `side` - as eight BF16 words. The whole warp calls it
// for one row. A word: (code bytes with their signs) & 0x807F807F | (nibbles + base) << 7 - nibbles 14 and 15 give
// base + 14 or 15 <= 255, which bf16p_patch replaces.
__device__ __forceinline__ uint4 bf16p_decode_regs(uint2 code, unsigned nib, uint2 side, int lane) {
    const unsigned lo = nib & 0x0F0F0F0FU, hi = (nib >> 4) & 0x0F0F0F0FU;
    const unsigned base7 = (side.x & 0xFFU) * 0x00800080U;
    const auto pair      = [&](unsigned word, unsigned code_selector, unsigned nibble_selector) {
        return (detail::bf16p_prmt(word, 0, code_selector) & 0x807F807FU) |
               ((detail::bf16p_prmt(lo, hi, nibble_selector) << 7) + base7);
    };
    unsigned w[4] = {pair(code.x, detail::kBf16pCodeSelector<0>, detail::kBf16pNibbleSelector<0>),
                     pair(code.x, detail::kBf16pCodeSelector<1>, detail::kBf16pNibbleSelector<1>),
                     pair(code.y, detail::kBf16pCodeSelector<2>, detail::kBf16pNibbleSelector<2>),
                     pair(code.y, detail::kBf16pCodeSelector<3>, detail::kBf16pNibbleSelector<3>)};
    if (((side.x >> 8) & 0xFFU) != 0) { detail::bf16p_patch<false>(w, nib, side, lane); }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ uint4 bf16p_f16_from_bf16(uint4 bits) {
    const int4 converted = bf16x8_bits_to_f16x8_bits(make_int4(static_cast<int>(bits.x), static_cast<int>(bits.y),
                                                               static_cast<int>(bits.z), static_cast<int>(bits.w)));
    return make_uint4(static_cast<unsigned>(converted.x), static_cast<unsigned>(converted.y),
                      static_cast<unsigned>(converted.z), static_cast<unsigned>(converted.w));
}

// Lane L's chunk of a packed V row as the eight FP16 words the bf16 storage holds (FP16_RNE of each BF16 value). A
// BF16 value with exponent 113..142 is an FP16 normal: sign, exponent - 112, the seven fraction bits shifted up by 3,
// exactly. A row whose window lies there (base 113..128: nibbles 14 and 15 then give at most 31, inside the field)
// decodes straight to those bits; its other values, and every other row, take the round-to-nearest conversion the
// append uses. The whole warp calls it for one row (inputs as for bf16p_decode_regs).
__device__ __forceinline__ uint4 bf16p_decode_regs_f16(uint2 code, unsigned nib, uint2 side, int lane) {
    const unsigned base = side.x & 0xFFU;
    if (base < 113 || base > 128) { return bf16p_f16_from_bf16(bf16p_decode_regs(code, nib, side, lane)); }
    const unsigned lo = nib & 0x0F0F0F0FU, hi = (nib >> 4) & 0x0F0F0F0FU;
    const unsigned base10 = (base - 112U) * 0x04000400U;
    const auto pair       = [&](unsigned word, unsigned code_selector, unsigned nibble_selector) {
        const unsigned t        = detail::bf16p_prmt(word, 0, code_selector);
        const unsigned exponent = (detail::bf16p_prmt(lo, hi, nibble_selector) << 10) + base10;
        return ((t << 3) & 0x03F803F8U) | (t & 0x80008000U) | exponent;
    };
    unsigned w[4] = {pair(code.x, detail::kBf16pCodeSelector<0>, detail::kBf16pNibbleSelector<0>),
                     pair(code.x, detail::kBf16pCodeSelector<1>, detail::kBf16pNibbleSelector<1>),
                     pair(code.y, detail::kBf16pCodeSelector<2>, detail::kBf16pNibbleSelector<2>),
                     pair(code.y, detail::kBf16pCodeSelector<3>, detail::kBf16pNibbleSelector<3>)};
    if (((side.x >> 8) & 0xFFU) != 0) { detail::bf16p_patch<true>(w, nib, side, lane); }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// Packs one row: lane L holds values 8 L .. 8 L + 7 (`bits`, BF16); `scratch` is the warp's 256 words of shared
// memory. The whole warp calls it; it writes the row's 384 bytes and its 8-byte side record, and returns whether the
// row has a packed form (false: more than kBf16pEscapes escapes, of which the record keeps the first; uniform).
__device__ __forceinline__ bool bf16p_encode_row(uint4 bits, std::uint8_t* __restrict__ row,
                                                 std::uint8_t* __restrict__ side, unsigned* scratch, int lane) {
    constexpr unsigned kFull = 0xFFFFFFFFU;
    const unsigned word[4]   = {bits.x, bits.y, bits.z, bits.w};
    unsigned e[8];
    unsigned lowest = 255, highest = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        e[i] = (word[i >> 1] >> (7 + 16 * (i & 1))) & 0xFFU;
        if (e[i] != 0 && e[i] != 255) {
            lowest  = min(lowest, e[i]);
            highest = max(highest, e[i]);
        }
    }
    lowest  = __reduce_min_sync(kFull, lowest);
    highest = __reduce_max_sync(kFull, highest);
    unsigned base;
    if (highest < lowest || highest - lowest < kBf16pWindow) {
        // Every normal exponent fits one window (or there is none).
        base = highest < lowest ? 1U : min(max(lowest, 1U), 240U);
    } else {
        // The lowest base of the windows holding most normal exponents: count[j] = exponents <= j, then
        // window(b) = count[b + 13] - count[b - 1] for b in 1..240 (lane L takes b = 8 L + 1 .. 8 L + 8).
        *reinterpret_cast<uint4*>(scratch + 8 * lane)     = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(scratch + 8 * lane + 4) = make_uint4(0, 0, 0, 0);
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (e[i] != 0 && e[i] != 255) { atomicAdd(scratch + e[i], 1U); }
        }
        __syncwarp();
        unsigned bins[8], run = 0;
        {
            const uint4 lo      = *reinterpret_cast<const uint4*>(scratch + 8 * lane);
            const uint4 hi      = *reinterpret_cast<const uint4*>(scratch + 8 * lane + 4);
            const unsigned h[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                run += h[i];
                bins[i] = run;
            }
        }
        unsigned offset = run;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const unsigned n = __shfl_up_sync(kFull, offset, d);
            if (lane >= d) { offset += n; }
        }
        offset -= run;
        __syncwarp();
        *reinterpret_cast<uint4*>(scratch + 8 * lane) =
            make_uint4(bins[0] + offset, bins[1] + offset, bins[2] + offset, bins[3] + offset);
        *reinterpret_cast<uint4*>(scratch + 8 * lane + 4) =
            make_uint4(bins[4] + offset, bins[5] + offset, bins[6] + offset, bins[7] + offset);
        __syncwarp();
        unsigned best = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int b = 8 * lane + 1 + i;
            if (b <= 240) {
                const unsigned count = scratch[b + kBf16pWindow - 1] - scratch[b - 1];
                best                 = max(best, count << 8 | static_cast<unsigned>(255 - b));
            }
        }
        best = __reduce_max_sync(kFull, best);
        base = 255U - (best & 0xFFU);
        __syncwarp(); // the scratch is free for the warp's next row
    }

    unsigned nibbles = 0, escaped = 0, zero = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned offset = e[i] - base; // wraps for e < base
        const bool inside     = e[i] != 0 && e[i] != 255 && offset < kBf16pWindow;
        const unsigned nibble = e[i] == 0 ? kBf16pZeroCode : inside ? offset : kBf16pEscapeCode;
        nibbles |= nibble << (4 * i);
        escaped |= (nibble == kBf16pEscapeCode ? 1U : 0U) << i;
        zero |= e[i] == 0 ? 1U : 0U;
    }
    const unsigned mine  = __popc(escaped);
    const unsigned total = __reduce_add_sync(kFull, mine);
    const bool zeros     = __any_sync(kFull, zero != 0);
    unsigned code[2]     = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned h = (word[i >> 1] >> (16 * (i & 1))) & 0xFFFFU;
        code[i >> 2] |= (((h >> 8) & 0x80U) | (h & 0x7FU)) << (8 * (i & 3));
    }
    *reinterpret_cast<uint2*>(row + 8 * lane)                    = make_uint2(code[0], code[1]);
    *reinterpret_cast<unsigned*>(row + kBf16pNibbles + 4 * lane) = nibbles;
    // The side record: this lane's escape exponents after the lower lanes' (bytes 2 + index), OR-combined.
    unsigned before = mine;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const unsigned n = __shfl_up_sync(kFull, before, d);
        if (lane >= d) { before += n; }
    }
    before -= mine;
    unsigned long long record = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((escaped >> i) & 1U) {
            if (before < static_cast<unsigned>(kBf16pEscapes)) {
                record |= static_cast<unsigned long long>(e[i]) << (8 * (2 + before));
            }
            ++before;
        }
    }
    const unsigned lo    = __reduce_or_sync(kFull, static_cast<unsigned>(record));
    const unsigned hi    = __reduce_or_sync(kFull, static_cast<unsigned>(record >> 32));
    const unsigned count = min(total, static_cast<unsigned>(kBf16pEscapes)) | (zeros ? 0x80U : 0U);
    if (lane == 0) { *reinterpret_cast<uint2*>(side) = make_uint2(lo | base | count << 8, hi); }
    return total <= static_cast<unsigned>(kBf16pEscapes);
}

} // namespace infernix::ops
