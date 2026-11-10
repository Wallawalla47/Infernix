#pragma once

// The CPU walks' scale decode of CSF units (csf_record.h): a unit's 16 scale bytes from its body's 8 offset bytes,
// its row group's 16 bases and the group's escape cursor, in an SSE register, branch-free, for every vector ISA's
// unit walk. Included by the CPU expert sources only (x86).

#include "ops/offloaded_sparse_moe/csf_record.h"

#include <array>
#include <cstdint>
#include <immintrin.h>

#if defined(_MSC_VER) && !defined(__clang__)
#    define INFERNIX_CSF_TARGET(x)
#else
#    define INFERNIX_CSF_TARGET(x) __attribute__((target(x)))
#endif

namespace infernix::ops::offloaded_moe::csf {

// The pshufb control that moves a unit's escapes (in row order) to their rows' bytes, 0x80 (zero) for the other
// rows, as the sum of two 16-byte entries: low[m] for the escaped rows 0-7 (mask m: bytes 0-7 the escape index of
// each escaped row, the k-th set bit of m taking k; bytes 8-15 the count of m's escapes, which rows 8-15's indices
// follow) and high[m] for rows 8-15 (bytes 0-7 zero; bytes 8-15 each escaped row's index among m's set bits, 0x80 for
// the others: 0x80 plus the low count still has bit 7 set, so pshufb still zeroes those rows).
struct ExpandTables {
    alignas(16) std::array<std::array<std::uint8_t, 16>, 256> low{};
    alignas(16) std::array<std::array<std::uint8_t, 16>, 256> high{};
};

inline constexpr ExpandTables kExpandTables = [] {
    ExpandTables t;
    for (unsigned m = 0; m < 256; ++m) {
        unsigned k = 0;
        for (unsigned i = 0; i < 8; ++i) {
            const bool escaped = ((m >> i) & 1U) != 0;
            t.low[m][i]        = escaped ? static_cast<std::uint8_t>(k) : std::uint8_t{0x80};
            t.high[m][8 + i]   = escaped ? static_cast<std::uint8_t>(k) : std::uint8_t{0x80};
            k += escaped ? 1U : 0U;
        }
        for (unsigned i = 8; i < 16; ++i) { t.low[m][i] = static_cast<std::uint8_t>(k); }
    }
    return t;
}();

// A unit's 16 scale bytes (lane r: row r): its offsets plus the group's bases, its escaped rows from `escape` (the
// group's cursor), and how many escapes it took, by which the caller advances the cursor. Returned by value: a cursor
// passed by reference stayed in memory (MSVC), a store-to-load round trip on every unit's path. Reads 16 bytes at
// `escape` (within the record: the group's bodies follow its escapes). The cursor's step (popcount of the unit's own
// mask) depends on nothing earlier, so successive units' decodes overlap; ~18 instructions a unit.
struct UnitScales {
    __m128i scales;
    std::uint32_t escapes;
};
INFERNIX_CSF_TARGET("avx2,popcnt")
inline UnitScales unit_scales_sse(const std::uint8_t* body, __m128i bases, const std::uint8_t* escape) {
    const __m128i nibble  = _mm_set1_epi8(0x0F);
    const __m128i packed  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(body + kCodeBytes));
    // Byte 2k: packed byte k's low nibble (row 2k); byte 2k + 1: its high nibble (row 2k + 1).
    const __m128i offsets = _mm_and_si128(_mm_unpacklo_epi8(packed, _mm_srli_epi16(packed, 4)), nibble);
    const __m128i escaped = _mm_cmpeq_epi8(offsets, nibble);
    const auto mask       = static_cast<unsigned>(_mm_movemask_epi8(escaped));
    const __m128i control =
        _mm_add_epi8(_mm_load_si128(reinterpret_cast<const __m128i*>(kExpandTables.low[mask & 0xFFU].data())),
                     _mm_load_si128(reinterpret_cast<const __m128i*>(kExpandTables.high[mask >> 8].data())));
    const __m128i fill = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(escape)), control);
    return {_mm_or_si128(_mm_andnot_si128(escaped, _mm_add_epi8(offsets, bases)), fill),
            static_cast<std::uint32_t>(_mm_popcnt_u32(mask))};
}

} // namespace infernix::ops::offloaded_moe::csf
