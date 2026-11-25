// Inkling 2's bf16p row encoder and piece decoder: the attention instances' encode of a new pool or window-ring row and
// decode of a pool's or a window ring's rows (kv_codec.cuh includes it; it depends on no device symbol, so code without
// relocatable device code can test it).
#pragma once

#include "ops/kv_cache/bf16p_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::inkling2::detail {

// bf16p at 128 (kv_codec.cuh kv_encode_bf16p): a row's 192 code bytes - [0, 128) sign << 7 | the seven fraction bits of
// value d, [128, 192) value d's exponent nibble (byte 128 + d / 2, d even the low nibble: 0..13 exponent base + nibble,
// 14 exponent 0, 15 an escape) - and a 16-byte side record: base, min(escapes, 14) | 0x80 when some exponent is 0, the
// escapes' exponents in column order. Inkling's rows need more escapes than the D256 form's 6 (its position-0 key rows
// reach 7).
inline constexpr unsigned kBf16p128Escapes = 14;

// A row (a warp; `bits` lane l's four BF16 words, values 4 l .. 4 l + 3: a V row's BF16 source, which its FP16 storage
// rounds) encoded at `row` and `side`. The base: a row whose normal exponents span at most 14 binades takes
// clamp(lowest, 1, 240) (1 with none); a wider one the lowest b in 1..240 whose window [b, b + 13] holds the most of
// them. That b is 1 or e - 13 for a normal exponent e of the row (a b > 1 whose window misses b + 13 holds no more than
// the window at b - 1), so only those are counted, one distinct candidate a step (the warp's lowest untried). A row
// without escapes writes its record without the escapes' scan. Returns whether the row has a packed form (at most 14
// escapes, kBf16p128Escapes; otherwise the record keeps the first 14 and the request fails). Uniform over the warp.
__device__ __forceinline__ bool kv_encode_bf16p(const std::uint16_t (&bits)[4], int lane, std::uint8_t* row,
                                                std::uint8_t* side) {
    constexpr unsigned kAll  = 0xFFFFFFFFU;
    constexpr unsigned kNone = 0xFFFFU;
    unsigned e[4];
    unsigned lowest = 255, highest = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        e[i] = (bits[i] >> 7) & 0xFFU;
        if (e[i] != 0 && e[i] != 255) {
            lowest  = min(lowest, e[i]);
            highest = max(highest, e[i]);
        }
    }
    lowest  = __reduce_min_sync(kAll, lowest);
    highest = __reduce_max_sync(kAll, highest);
    unsigned base;
    if (highest < lowest || highest - lowest < ops::kBf16pWindow) {
        base = highest < lowest ? 1U : min(max(lowest, 1U), 240U);
    } else {
        // window(b) = the normal exponents in [b, b + 13] (e - b wraps for e < b); the most, ties to the lowest b.
        const auto window = [&](unsigned b) {
            unsigned n = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) { n += e[i] != 0 && e[i] != 255 && e[i] - b < ops::kBf16pWindow ? 1U : 0U; }
            return __reduce_add_sync(kAll, n);
        };
        unsigned pending[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { pending[i] = e[i] >= 14 && e[i] <= 253 ? e[i] - 13 : kNone; } // (e 14..253: normal)
        unsigned best = window(1) << 8 | 254U; // b = 1: 255 - b
        for (;;) {
            const unsigned b = __reduce_min_sync(kAll, min(min(pending[0], pending[1]), min(pending[2], pending[3])));
            if (b == kNone) { break; }
            best = max(best, window(b) << 8 | (255U - b));
#pragma unroll
            for (int i = 0; i < 4; ++i) { pending[i] = pending[i] == b ? kNone : pending[i]; }
        }
        base = 255U - (best & 0xFFU);
    }
    unsigned nibbles = 0, escaped = 0, zero = 0, code = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned offset = e[i] - base; // wraps for e < base
        const bool inside     = e[i] != 0 && e[i] != 255 && offset < ops::kBf16pWindow;
        const unsigned nibble = e[i] == 0 ? ops::kBf16pZeroCode : inside ? offset : ops::kBf16pEscapeCode;
        nibbles |= nibble << (4 * i);
        escaped |= (nibble == ops::kBf16pEscapeCode ? 1U : 0U) << i;
        zero |= e[i] == 0 ? 1U : 0U;
        code |= (((bits[i] >> 8) & 0x80U) | (bits[i] & 0x7FU)) << (8 * i);
    }
    *reinterpret_cast<std::uint32_t*>(row + 4 * lane)       = code;
    *reinterpret_cast<std::uint16_t*>(row + 128 + 2 * lane) = static_cast<std::uint16_t>(nibbles);
    const bool zeros = __any_sync(kAll, zero != 0);
    if (__all_sync(kAll, escaped == 0)) { // no escapes: base and the zero flag
        if (lane == 0) { *reinterpret_cast<uint4*>(side) = make_uint4(base | (zeros ? 0x8000U : 0U), 0U, 0U, 0U); }
        return true;
    }
    const unsigned mine  = __popc(escaped);
    const unsigned total = __reduce_add_sync(kAll, mine);
    unsigned before      = mine;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const unsigned n = __shfl_up_sync(kAll, before, d);
        if (lane >= d) { before += n; }
    }
    before -= mine;
    // The record's bytes 2 + k hold escape k (k < 14), gathered as its low and high 8 bytes (shifts, not an array a run
    // time index would put in local memory).
    std::uint64_t low = 0, high = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if ((escaped >> i) & 1U) {
            if (before < kBf16p128Escapes) {
                const unsigned byte    = 2 + before;
                const std::uint64_t at = std::uint64_t{e[i]} << (8 * (byte & 7U));
                if (byte < 8) {
                    low |= at;
                } else {
                    high |= at;
                }
            }
            ++before;
        }
    }
    const unsigned count = min(total, kBf16p128Escapes) | (zeros ? 0x80U : 0U);
    uint4 record;
    record.x = __reduce_or_sync(kAll, static_cast<unsigned>(low)) | base | count << 8;
    record.y = __reduce_or_sync(kAll, static_cast<unsigned>(low >> 32));
    record.z = __reduce_or_sync(kAll, static_cast<unsigned>(high));
    record.w = __reduce_or_sync(kAll, static_cast<unsigned>(high >> 32));
    if (lane == 0) { *reinterpret_cast<uint4*>(side) = record; }
    return total <= kBf16p128Escapes;
}

// Piece p (values 8 p .. 8 p + 7) of a bf16p row (`row` its code bytes, `side` its record) as 8 represented words: a
// key's BF16 word (Value false), a value's FP16 word, fp16_rn of the BF16 value (Value true). The BF16 words as
// bf16p_decode_regs forms them - (code bytes with their signs) & 0x807F807F | (nibbles + base) << 7, base <= 240 keeping
// nibbles 14 and 15 inside the field - then a piece holding a nibble 14 or 15 replaced: exponent 0, or its escape's
// exponent (escapes counted over the row's earlier pieces; past the record's 14 only in a row with no packed form, whose
// request fails: exponent 0 as kv_decode16's). A value's words by one packed FP32 -> FP16 conversion of a pair.
template <bool Value>
__device__ __forceinline__ uint4 kv_decode8_bf16p(const std::uint8_t* row, const std::uint8_t* side, int piece) {
    const uint2 code     = *reinterpret_cast<const uint2*>(row + 8 * piece);
    const unsigned nib   = *reinterpret_cast<const std::uint32_t*>(row + 128 + 4 * piece);
    const unsigned head  = *reinterpret_cast<const std::uint32_t*>(side); // base, count | zero flag
    const unsigned lo    = nib & 0x0F0F0F0FU, hi = (nib >> 4) & 0x0F0F0F0FU;
    const unsigned base7 = (head & 0xFFU) * 0x00800080U;
    const auto pair      = [&](unsigned word, unsigned code_selector, unsigned nibble_selector) {
        return (ops::detail::bf16p_prmt(word, 0, code_selector) & 0x807F807FU) |
               ((ops::detail::bf16p_prmt(lo, hi, nibble_selector) << 7) + base7);
    };
    unsigned w[4] = {pair(code.x, ops::detail::kBf16pCodeSelector<0>, ops::detail::kBf16pNibbleSelector<0>),
                     pair(code.x, ops::detail::kBf16pCodeSelector<1>, ops::detail::kBf16pNibbleSelector<1>),
                     pair(code.y, ops::detail::kBf16pCodeSelector<2>, ops::detail::kBf16pNibbleSelector<2>),
                     pair(code.y, ops::detail::kBf16pCodeSelector<3>, ops::detail::kBf16pNibbleSelector<3>)};
    // Bit 4 i: value i is an escape (nibble 15) or of exponent 0 (nibble 14).
    const unsigned high = (nib >> 1) & (nib >> 2) & (nib >> 3) & 0x11111111U;
    if (high != 0) [[unlikely]] {
        const unsigned escaped = nib & high, zero = ~nib & high;
        unsigned index = 0; // the row's escapes before this piece
        if (escaped != 0) {
            for (int k = 0; k < piece; ++k) {
                const unsigned n = *reinterpret_cast<const std::uint32_t*>(row + 128 + 4 * k);
                index += __popc(n & (n >> 1) & (n >> 2) & (n >> 3) & 0x11111111U);
            }
        }
        const uint4 record = *reinterpret_cast<const uint4*>(side);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned shift = 7 + 16 * (i & 1);
            if ((zero >> (4 * i)) & 1U) { w[i >> 1] &= ~(0xFFU << shift); }
            if ((escaped >> (4 * i)) & 1U) {
                // Escape k's exponent: record byte 2 + k, its word chosen by selects (a run-time indexed array would
                // land in local memory).
                unsigned e = 0;
                if (index < kBf16p128Escapes) {
                    const unsigned byte = 2 + index, q = byte >> 2;
                    const unsigned low = (q & 1U) != 0 ? record.y : record.x, upper = (q & 1U) != 0 ? record.w : record.z;
                    e = (((q & 2U) != 0 ? upper : low) >> (8 * (byte & 3U))) & 0xFFU;
                }
                ++index;
                w[i >> 1] = (w[i >> 1] & ~(0xFFU << shift)) | (e << shift);
            }
        }
    }
    if constexpr (Value) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __half2 h = __floats2half2_rn(__uint_as_float(w[j] << 16), __uint_as_float(w[j] & 0xFFFF0000U));
            w[j]            = *reinterpret_cast<const std::uint32_t*>(&h);
        }
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

} // namespace infernix::ops::inkling2::detail
