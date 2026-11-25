#pragma once

// The BF16 P12 blockdir row's packed reading, shared by the device kernel and its host tests. A 4 KiB
// block holds 17 rows of P12's 240 data bytes (160 sign-and-mantissa, 80 exponent nibbles) and one
// 16-byte directory: a 3-bit base code per row into the volume header's 8-entry base table, a 3-bit
// escape count (7: the block's escapes are in the file's side stream) and four 18-bit escape slots
// (12-bit position in the block, 6-bit signed delta from the row's base). The bytes and directory
// layout are the host reader's (program/ngram_volume.cpp) and the packer's (tools/flash_next/
// ngram_p12_ext.py); this header holds only the decode both sides must agree on.

#include <cstdint>

#if defined(__CUDACC__)
#define INFERNIX_NGRAM_HD __host__ __device__
#else
#define INFERNIX_NGRAM_HD
#endif

namespace infernix::ngram {

constexpr unsigned kBlockdirRows           = 17;
constexpr unsigned kBlockdirWidth          = 160;
constexpr unsigned kBlockdirPlaneBytes     = 240;  // 160 sign-and-mantissa + 80 exponent nibbles
constexpr unsigned kBlockdirDirectoryBytes = 16;
constexpr unsigned kBlockdirBaseTableSize  = 8;
constexpr unsigned kBlockdirCodeBits       = 3;
constexpr unsigned kBlockdirSlots          = 4;
constexpr unsigned kBlockdirSideSentinel   = 7;

// A resident-mode row as the kernel reads it: the 240 B data planes, the block's 16 B directory, then
// the row's index in its block at byte 256 (the rest zero). A row whose block put its escapes in the
// side stream (or whose directory is invalid) is staged as the canonical P12 row instead: its 16 B
// trailer sits at 240 and byte 256 is kBlockdirCanonicalMarker.
constexpr unsigned kBlockdirDirectoryOffset = kBlockdirPlaneBytes;                                    // 240
constexpr unsigned kBlockdirHeaderOffset    = kBlockdirPlaneBytes + kBlockdirDirectoryBytes;          // 256
constexpr unsigned kBlockdirStagingRowBytes = kBlockdirHeaderOffset + 16;                             // 272
constexpr std::uint8_t kBlockdirCanonicalMarker = 0xFF;

// Whether a row may be staged as a canonical P12 row rather than the packed reading: a directory
// whose count is not an inline slot count (7 is the side stream; 5 and 6 are invalid).
INFERNIX_NGRAM_HD inline bool blockdir_row_is_canonical(unsigned escape_count) {
    return escape_count > kBlockdirSlots;
}

INFERNIX_NGRAM_HD inline unsigned blockdir_bit(const std::uint8_t* directory, unsigned at) {
    return (static_cast<unsigned>(directory[at >> 3]) >> (at & 7)) & 1U;
}

// The row's 3-bit code into the volume's base table.
INFERNIX_NGRAM_HD inline unsigned blockdir_code(const std::uint8_t* directory, unsigned row_in_block) {
    unsigned value = 0;
    for (unsigned k = 0; k < kBlockdirCodeBits; ++k) {
        value |= blockdir_bit(directory, kBlockdirCodeBits * row_in_block + k) << k;
    }
    return value;
}

// The block's inline escape count (its slots' entries apply to whichever rows their positions name).
INFERNIX_NGRAM_HD inline unsigned blockdir_escape_count(const std::uint8_t* directory) {
    const unsigned at = kBlockdirCodeBits * kBlockdirRows;
    return blockdir_bit(directory, at) | (blockdir_bit(directory, at + 1) << 1U) |
           (blockdir_bit(directory, at + 2) << 2U);
}

// Slot e as packed: bits 0..11 the position in the block (row * width + column), bits 12..17 the
// signed delta from that row's base.
INFERNIX_NGRAM_HD inline std::uint32_t blockdir_slot(const std::uint8_t* directory, unsigned e) {
    std::uint32_t slot = 0;
    const unsigned start = kBlockdirCodeBits * kBlockdirRows + 3 + 18 * e;
    for (unsigned k = 0; k < 18; ++k) { slot |= blockdir_bit(directory, start + k) << k; }
    return slot;
}

INFERNIX_NGRAM_HD inline int blockdir_slot_delta(std::uint32_t slot) {
    int delta = static_cast<int>(slot >> 12U);
    if ((delta & 32) != 0) { delta -= 64; }
    return delta;
}

// A value's BF16 bits from its stored sign-and-mantissa byte, its row's base and its offset q < 15
// (the codec BF16 P12 and blockdir share).
INFERNIX_NGRAM_HD inline std::uint16_t p12_value_bits(std::uint8_t code, unsigned base, unsigned q) {
    return static_cast<std::uint16_t>(((code & 0x80U) << 8U) | ((base + q) << 7U) | (code & 0x7FU));
}

// A value from a canonical P12 row's 16 B trailer (base, count, then 3-byte escapes).
INFERNIX_NGRAM_HD inline std::uint16_t p12_trailer_value_bits(const std::uint8_t* trailer, int j,
                                                             std::uint8_t code, unsigned q) {
    if (q < 15U) { return p12_value_bits(code, trailer[0], q); }
    const unsigned escapes   = trailer[1];
    const std::uint8_t* list = trailer + 2;
    for (unsigned e = 0; e < escapes; ++e) {
        if (list[3 * e] == j) {
            return static_cast<std::uint16_t>(static_cast<unsigned>(list[3 * e + 1]) |
                                              (static_cast<unsigned>(list[3 * e + 2]) << 8U));
        }
    }
    return 0;
}

// The value at column j of a packed blockdir row: the row's base comes from the volume's table through
// the directory's 3-bit code, and an escaped value from the slot whose position names this row and
// column (the four slots apply to the whole block). A row that no slot names decodes its escapes as 0,
// as a P12 row with no matching escape entry does.
INFERNIX_NGRAM_HD inline std::uint16_t blockdir_value_bits(const std::uint8_t* directory,
                                                           const std::uint8_t* bases, unsigned row_in_block,
                                                           unsigned width, int j, std::uint8_t code,
                                                           unsigned q) {
    const unsigned base = bases[blockdir_code(directory, row_in_block)];
    if (q < 15U) { return p12_value_bits(code, base, q); }
    const unsigned count   = blockdir_escape_count(directory);
    const std::uint32_t at = row_in_block * width + static_cast<unsigned>(j);
    for (unsigned e = 0; e < count && e < kBlockdirSlots; ++e) {
        const std::uint32_t slot = blockdir_slot(directory, e);
        if ((slot & 0xFFFU) != at) { continue; }
        const int exponent = static_cast<int>(base) + blockdir_slot_delta(slot);
        if (exponent < 0 || exponent > 255) { return 0; }
        return static_cast<std::uint16_t>((static_cast<unsigned>(code & 0x80U) << 8U) |
                                          (static_cast<unsigned>(exponent) << 7U) | (code & 0x7FU));
    }
    return 0;
}

} // namespace infernix::ngram

#undef INFERNIX_NGRAM_HD
