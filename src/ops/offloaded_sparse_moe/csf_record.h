#pragma once

// CSF host records: an NVFP4 expert record (`nvfp4_expert_rg16_v1`: 144-byte units of 16 rows x 16 elements, 128
// code bytes then the 16 rows' E4M3 block scales, row-group-major, the gate/up matrix's row groups then the down
// matrix's) with each block scale stored as a 4-bit offset from its row's base, so every host-to-device copy and every
// CPU walk of the record reads ~5 % fewer bytes. The codes are unchanged and every scale byte is restored exactly (a
// stored byte, or a stored base plus a stored offset), so an expanded record is the artifact's record bit for bit and
// every kernel reads the same weights. A full-mode bank's records are compressed once at load, in place at the bank's
// stride (docs/maintainer/qwen3_8-flash-next-design.md §19.3.54).
//
// A compressed record is its row groups' regions in order; row group r occupies [table[r], table[r + 1]) of it, the
// (row groups + 1) u32 offsets of the record's table (held beside the bank, not in the record), table[0] = 0 and the
// last entry the record's compressed bytes. A region is one of:
//   compressed  [16 row bases][escapes][units' bodies]
//               - the 16 rows' bases (each the first code of a 15-code window: offsets 0..14 from it);
//               - every scale outside its row's window, in unit order and, within a unit, row order, zero-padded to 16;
//               - per unit (blocks of the group, kBodyBytes each, ending the region): its 128 code bytes, then 8 bytes
//                 of 4-bit offsets (row r in byte r / 2, the low nibble for even r), kEscape for an escaped scale;
//   raw         the group's 144-byte units as stored, when compressing would not make the region smaller.
// A region's size tells which it is (raw: exactly blocks x 144 bytes; compressed regions are smaller), so a row group
// is self-contained: a reader starts at any group with an escape cursor at its escapes. Every region starts 16-byte
// aligned and no region starts after its expanded offset, so a record never exceeds its expanded size and expanding
// a record in place in descending row-group order never overwrites a region not yet read.

#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#    define INFERNIX_CSF_HD __host__ __device__
#else
#    define INFERNIX_CSF_HD
#endif

namespace infernix::ops::offloaded_moe::csf {

inline constexpr int kCodeBytes = 128;
inline constexpr int kBodyBytes = 136; // a unit's code bytes and its 8 offset bytes
inline constexpr int kBaseBytes = 16;  // a compressed group's row bases
inline constexpr int kEscape    = 15;  // the offset that marks an escaped scale
inline constexpr int kWindow    = 15;  // offsets 0..14

// A record's row groups: the gate/up matrix's (gate_up_rgs of gate_up_blocks units), then the down matrix's
// (down_rgs of down_blocks units). The GPU copies a record in chunks of gate_up_blocks units: one gate/up row group,
// or down_per_chunk() down row groups (H / I of them: a whole number at every registered geometry).
struct Geometry {
    int gate_up_rgs = 0, down_rgs = 0, gate_up_blocks = 0, down_blocks = 0;

    static constexpr Geometry of(const ExpertShape& shape) { // host: kernels take a Geometry
        return {shape.gate_up_row_groups(), shape.down_row_groups(), shape.gate_up_blocks(), shape.down_blocks()};
    }
    [[nodiscard]] INFERNIX_CSF_HD constexpr int rgs() const { return gate_up_rgs + down_rgs; }
    [[nodiscard]] INFERNIX_CSF_HD constexpr int table_words() const { return rgs() + 1; }
    [[nodiscard]] INFERNIX_CSF_HD constexpr int blocks(int rg) const { return rg < gate_up_rgs ? gate_up_blocks : down_blocks; }
    // Row group rg's expanded offset in the record.
    [[nodiscard]] INFERNIX_CSF_HD constexpr std::size_t expanded_offset(int rg) const {
        return rg < gate_up_rgs
                   ? static_cast<std::size_t>(rg) * gate_up_blocks * kUnitBytes
                   : (static_cast<std::size_t>(gate_up_rgs) * gate_up_blocks + static_cast<std::size_t>(rg - gate_up_rgs) * down_blocks) *
                         kUnitBytes;
    }
    [[nodiscard]] INFERNIX_CSF_HD constexpr std::size_t expanded_bytes() const { return expanded_offset(rgs()); }
    // GPU chunks: each chunk_units() units.
    [[nodiscard]] INFERNIX_CSF_HD constexpr int chunk_units() const { return gate_up_blocks; }
    [[nodiscard]] INFERNIX_CSF_HD constexpr int down_per_chunk() const { return gate_up_blocks / down_blocks; }
    [[nodiscard]] INFERNIX_CSF_HD constexpr int gate_up_chunks() const { return gate_up_rgs; }
    [[nodiscard]] INFERNIX_CSF_HD constexpr int chunks() const { return gate_up_rgs + down_rgs / down_per_chunk(); }
    [[nodiscard]] INFERNIX_CSF_HD constexpr int chunk_first_rg(int c) const {
        return c < gate_up_rgs ? c : gate_up_rgs + (c - gate_up_rgs) * down_per_chunk();
    }
    [[nodiscard]] INFERNIX_CSF_HD constexpr int chunk_rgs(int c) const { return c < gate_up_rgs ? 1 : down_per_chunk(); }
};

// The most units and row groups of a GPU chunk at any registered geometry (the 122B's 192-unit gate/up row groups,
// Flash-Next's four 40-unit down row groups), and its most bytes (raw regions).
inline constexpr int kMaxChunkUnits = 192;
inline constexpr int kMaxChunkRgs   = 4;
inline constexpr int kMaxChunkBytes = kMaxChunkUnits * static_cast<int>(kUnitBytes);
// The most table words at any registered geometry (the 122B's: 2 * 1024 / 16 gate/up and 3072 / 16 down row groups).
inline constexpr int kMaxTableWords = 2 * 1024 / 16 + 3072 / 16 + 1;

// One row group as a reader sees it: unit b at units + b * (bases ? kBodyBytes : kUnitBytes); a compressed group's
// 16 row bases and the first of its escapes, or (raw) bases == nullptr and the scales in the units.
struct Group {
    const std::uint8_t* units   = nullptr;
    const std::uint8_t* bases   = nullptr;
    const std::uint8_t* escapes = nullptr;
};

// Row group rg of a compressed record (`data`, its `table`) of `blocks` units a row.
INFERNIX_CSF_HD inline Group group(const std::uint8_t* data, const std::uint32_t* table, int rg, int blocks) {
    const std::uint32_t begin = table[rg], end = table[rg + 1];
    if (end - begin == static_cast<std::uint32_t>(blocks) * kUnitBytes) { return {data + begin, nullptr, nullptr}; }
    return {data + end - static_cast<std::size_t>(blocks) * kBodyBytes, data + begin, data + begin + kBaseBytes};
}

// Row group rg of a CPU matrix view (w4a4_expert.h Rg16Matrix: expanded, or a CSF record's matrix).
inline Group group(const Rg16Matrix& m, int rg) {
    if (m.csf == nullptr) { return {m.data + static_cast<std::size_t>(rg) * m.blocks * kUnitBytes, nullptr, nullptr}; }
    return group(m.data, m.csf, m.first_rg + rg, m.blocks);
}

// The offset of row r in a body's 8 offset bytes.
INFERNIX_CSF_HD inline int offset_of(const std::uint8_t* body, int r) {
    const std::uint8_t b = body[kCodeBytes + r / 2];
    return (r & 1) == 0 ? (b & 15) : (b >> 4);
}

// The 16 scale bytes of a compressed unit (its body, its group's bases and its first escape); returns the unit's
// escapes' end (the next unit's first escape).
INFERNIX_CSF_HD inline const std::uint8_t* unit_scales(const std::uint8_t* body, const std::uint8_t* bases,
                                                       const std::uint8_t* escape, std::uint8_t (&scales)[16]) {
    for (int r = 0; r < 16; ++r) {
        const int o = offset_of(body, r);
        scales[r]   = o == kEscape ? *escape++ : static_cast<std::uint8_t>(bases[r] + o);
    }
    return escape;
}

// Host: a matrix row group's units and their 16 scale bytes each (the scalar walks' view: a compressed group's
// scales decoded once for every column).
struct GroupScales {
    const std::uint8_t* units = nullptr;
    int pitch                 = 0;
    std::uint8_t scales[kMaxChunkUnits][16];
};
inline void group_scales(const Rg16Matrix& m, int rg, GroupScales& out) {
    const Group g = group(m, rg);
    out.units     = g.units;
    out.pitch     = g.bases != nullptr ? kBodyBytes : static_cast<int>(kUnitBytes);
    const std::uint8_t* escape = g.escapes;
    for (int b = 0; b < m.blocks; ++b) {
        const std::uint8_t* unit = g.units + static_cast<std::size_t>(b) * out.pitch;
        if (g.bases != nullptr) {
            escape = unit_scales(unit, g.bases, escape, out.scales[b]);
        } else {
            for (int r = 0; r < 16; ++r) { out.scales[b][r] = unit[kCodeBytes + r]; }
        }
    }
}

// A bank's tables where the host and the device read them (record e's at host + e * words, device + e * words;
// mapped host memory), or none (null: the bank's records are expanded).
struct BankTables {
    const std::uint32_t* host   = nullptr;
    const std::uint32_t* device = nullptr;
    [[nodiscard]] explicit operator bool() const noexcept { return host != nullptr; }
};

// ---- CSF records in RAM slots (the SSD tier, runtime/expert_offload/host_expert_tier.h): a slot of `pitch` bytes holds
// a compressed record with its table in the slot's tail (the tier knows which slots do). Wherever such a slot's
// address is published to a reader (the device's record table, a CPU ticket), its bit 0 is set (kTag): the reader
// finds the table at untagged + slot_table_offset. An untagged address is an expanded record.
inline constexpr std::uint64_t kTag = 1;
// The table's bytes (16-aligned) and offset in a slot of `pitch` bytes.
INFERNIX_CSF_HD inline constexpr std::uint64_t slot_table_bytes(int table_words) {
    return (static_cast<std::uint64_t>(table_words) * 4 + 15) / 16 * 16;
}
INFERNIX_CSF_HD inline constexpr std::uint64_t slot_table_offset(int table_words, std::uint64_t pitch) {
    return pitch - slot_table_bytes(table_words);
}
INFERNIX_CSF_HD inline bool tagged(const void* p) { return (reinterpret_cast<std::uint64_t>(p) & kTag) != 0; }
template <class T>
INFERNIX_CSF_HD inline T* untag(T* p) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uint64_t>(p) & ~kTag);
}
template <class T>
INFERNIX_CSF_HD inline T* tag(T* p) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uint64_t>(p) | kTag);
}
// A tagged slot address's table (pitch: the slot's bytes).
INFERNIX_CSF_HD inline const std::uint32_t* slot_table(const std::uint8_t* tagged_slot, int table_words, std::uint64_t pitch) {
    return reinterpret_cast<const std::uint32_t*>(untag(tagged_slot) + slot_table_offset(table_words, pitch));
}

// Host: compresses the expanded record in a RAM slot of `pitch` bytes in place, with its table in the slot's tail;
// returns false, leaving the slot's expanded record as it was, when the compressed record would not leave room for
// the table (a record whose groups are nearly all raw).
bool compress_slot(const ExpertShape& shape, std::uint8_t* slot, std::uint64_t pitch);
// Host: expands a compressed record in place (its row groups in descending order: no region starts after its expanded
// offset, so each is read before any later one's units overwrite it).
void expand_in_place_host(const ExpertShape& shape, std::uint8_t* record, const std::uint32_t* table);

// Host: compresses an expanded record of `shape` into `out` and its table (Geometry::table_words() entries); returns
// the compressed bytes (table's last entry). `out` may be `expanded` (in place: row groups are compressed in order,
// each read before its region is written, and no region starts after its expanded offset).
std::size_t compress_record(const ExpertShape& shape, const std::uint8_t* expanded, std::uint8_t* out, std::uint32_t* table);
// Host: expands a compressed record into `out` (shape.record_bytes()); the reference for every other expansion.
void expand_record(const ExpertShape& shape, const std::uint8_t* data, const std::uint32_t* table, std::uint8_t* out);
// Host: whether `table` is a well-formed table of a compressed record of `shape` that fits `stride` bytes (offsets
// 16-byte aligned and increasing, each region raw-sized or smaller with room for its bodies and bases).
[[nodiscard]] bool valid_table(const ExpertShape& shape, const std::uint32_t* table, std::uint64_t stride);

} // namespace infernix::ops::offloaded_moe::csf
