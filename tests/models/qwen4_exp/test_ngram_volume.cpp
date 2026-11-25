// NgramVolume::read_rows on a synthetic volume (n-gram step S1, design §19.3.4). The oracle is the
// closed-form row contents f(r, k), never the volume: random rows, duplicates and rows sharing a block
// read each distinct block once (counters), more than 64 distinct blocks wrap the ring, a second
// call hits the host cache, two rows of one cache slot both return their own bytes, a row past the
// table throws, and a volume of the wrong size is refused at open. A call of thousands of distinct
// blocks (a prefill chunk) takes the multi-threaded route: the same oracle, counters and cache.
// resolve_ngram_volume accepts a volume of any format in place of the artifact's table and refuses mismatches.

#include "test_temp_path.h"
#include "infernix/ngram_blockdir.h"
#include "models/qwen4_exp/program/ngram_volume.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace q4 = infernix::models::qwen4_exp;

namespace {

constexpr std::uint32_t kRowBytes = 16;
constexpr std::uint32_t kBlockBytes = 4096;
constexpr std::uint32_t kRowsPerBlock = kBlockBytes / kRowBytes;
constexpr std::uint64_t kRows = 1ULL << 21; // twice the cache's slots: slot collisions exist

std::uint8_t f(std::uint64_t row, std::size_t k) { return static_cast<std::uint8_t>((row * 29U + k * 13U + 7U) & 0xFFU); }

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

template <class T>
void put(std::vector<std::byte>& header, std::size_t at, T value) {
    std::memcpy(header.data() + at, &value, sizeof(T));
}

q4::NgramTableConfig make_volume(const std::filesystem::path& path) {
    q4::NgramTableConfig table;
    table.rows           = kRows;
    table.row_bytes      = kRowBytes;
    table.rows_per_block = kRowsPerBlock;
    table.block_bytes    = kBlockBytes;
    table.header_bytes   = 4096;
    table.blocks         = kRows / kRowsPerBlock;
    table.file_bytes     = table.header_bytes + table.blocks * kBlockBytes;
    for (std::size_t i = 0; i < table.volume_id.size(); ++i) { table.volume_id[i] = static_cast<std::uint8_t>(i * 17); }
    std::vector<std::byte> header(4096);
    std::memcpy(header.data(), "NINFERNG", 8);
    put<std::uint32_t>(header, 8, 1);
    put<std::uint32_t>(header, 12, table.header_bytes);
    put<std::uint64_t>(header, 16, table.rows);
    put<std::uint32_t>(header, 24, table.row_bytes);
    put<std::uint32_t>(header, 28, table.rows_per_block);
    put<std::uint32_t>(header, 32, table.block_bytes);
    put<std::uint64_t>(header, 36, table.blocks);
    std::memcpy(header.data() + 44, table.volume_id.data(), 16);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
    std::vector<std::uint8_t> block(kBlockBytes);
    for (std::uint64_t b = 0; b < table.blocks; ++b) {
        for (std::uint32_t r = 0; r < kRowsPerBlock; ++r) {
            for (std::uint32_t k = 0; k < kRowBytes; ++k) { block[r * kRowBytes + k] = f(b * kRowsPerBlock + r, k); }
        }
        out.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(block.size()));
    }
    return table;
}

bool rows_match(std::span<const std::uint32_t> rows, const std::vector<std::byte>& out) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (std::uint32_t k = 0; k < kRowBytes; ++k) {
            if (static_cast<std::uint8_t>(out[i * kRowBytes + k]) != f(rows[i], k)) { return false; }
        }
    }
    return true;
}

std::vector<std::byte> read(const q4::NgramVolume& volume, const std::vector<std::uint32_t>& rows) {
    std::vector<std::byte> out(rows.size() * kRowBytes);
    volume.read_rows(rows, out);
    return out;
}

std::uint64_t distinct_blocks(const std::vector<std::uint32_t>& rows) {
    std::set<std::uint32_t> blocks;
    for (const auto r : rows) { blocks.insert(r / kRowsPerBlock); }
    return blocks.size();
}

// A volume header alone (resolve_ngram_volume reads nothing else): rows of `row_bytes`, as the
// converter lays them out, with `id`, an FP8 scale word (0: none recorded), the row format tag (0: a
// volume written before the tag) and an NVFP4 multiplier (FP32 bits).
void write_header(const std::filesystem::path& path, std::uint64_t rows, std::uint32_t row_bytes, std::uint8_t id,
                  std::uint16_t scale = 0, std::uint16_t tag = 0, std::uint32_t multiplier = 0) {
    std::vector<std::byte> header(4096);
    const std::uint32_t per_block = row_bytes == 0 ? 1 : kBlockBytes / row_bytes; // 0: a corrupt header
    std::memcpy(header.data(), "NINFERNG", 8);
    put<std::uint32_t>(header, 8, 1);
    put<std::uint32_t>(header, 12, 4096);
    put<std::uint64_t>(header, 16, rows);
    put<std::uint32_t>(header, 24, row_bytes);
    put<std::uint32_t>(header, 28, per_block);
    put<std::uint32_t>(header, 32, kBlockBytes);
    put<std::uint64_t>(header, 36, (rows + per_block - 1) / per_block);
    std::memset(header.data() + 44, id, 16);
    put<std::uint16_t>(header, 60, scale);
    put<std::uint16_t>(header, 62, tag);
    put<std::uint32_t>(header, 64, multiplier);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
}

// A P12 blockdir volume header: 256-byte decoded rows, 17 a block, W 160, 240-byte planes, the eight
// bases (after the side stream's fields) and the side stream's geometry.
void write_blockdir_header(const std::filesystem::path& path, std::uint64_t rows, const std::uint8_t bases[8],
                           std::uint64_t side_bytes = 0) {
    const std::uint64_t blocks = (rows + 16) / 17;
    std::vector<std::byte> header(4096);
    std::memcpy(header.data(), "NINFERNG", 8);
    put<std::uint32_t>(header, 8, 1);
    put<std::uint32_t>(header, 12, 4096);
    put<std::uint64_t>(header, 16, rows);
    put<std::uint32_t>(header, 24, 256);
    put<std::uint32_t>(header, 28, 17);
    put<std::uint32_t>(header, 32, 4096);
    put<std::uint64_t>(header, 36, blocks);
    std::memset(header.data() + 44, 0x55, 16);
    put<std::uint16_t>(header, 62, 7);
    put<std::uint16_t>(header, 68, 160);
    put<std::uint16_t>(header, 70, 240);
    std::memcpy(header.data() + 116, bases, 8);
    put<std::uint64_t>(header, 92, side_bytes == 0 ? 0 : 4096 + blocks * 4096);
    put<std::uint64_t>(header, 100, side_bytes);
    put<std::uint64_t>(header, 108, side_bytes == 0 ? 0 : 1);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
}

// A row's 240 P12 data bytes for a base and one optional escape (column, exponent): every value's
// exponent is base + j % 15, its code 0x11, and the escaped value's q nibble is 15.
void put_blockdir_plane(std::uint8_t* block, std::uint32_t row_in_block, std::uint32_t base, int escape_column,
                        int escape_exponent) {
    std::uint8_t* plane = block + row_in_block * 240;
    for (std::uint32_t j = 0; j < 160; ++j) {
        plane[j] = 0x11;
        const unsigned q = static_cast<int>(j) == escape_column ? 15U : (j % 15U);
        plane[160 + j / 2] |= static_cast<std::uint8_t>(q << (4 * (j & 1)));
    }
    (void)base;
    (void)escape_exponent;
}

void put_directory_bits(std::uint8_t* directory, unsigned at, unsigned count, std::uint32_t value) {
    for (unsigned k = 0; k < count; ++k) {
        if ((value >> k) & 1U) { directory[(at + k) / 8] |= static_cast<std::uint8_t>(1U << ((at + k) % 8)); }
    }
}

// resolve_ngram_volume: the artifact's own volume (its format and id) keeps the artifact's table; a
// volume of the same table geometry in any of the five formats (FP8 rows of the head width, IQ4_NL
// blocks of 18 bytes per 32, BF16 rows of twice the width, NVFP4 codes and block scales, BF16 P12 rows
// of W + W/2 + 16 bytes) replaces any artifact's table with its own geometry and id, an FP8 one only
// with a scale and an NVFP4 one only with a multiplier in its header; another row count or row size is
// refused. The header's format tag decides the format (NVFP4 rows have IQ4_NL's 90 bytes); a volume
// written before the tag is read by row size. BF16 P12 is a run-time volume only, never an artifact's own.
void test_resolve(const std::filesystem::path& path) {
    constexpr std::uint64_t rows        = 100'003;
    constexpr std::uint16_t kScale      = 0x3A83;     // a positive normal BF16 value
    constexpr std::uint32_t kMultiplier = 0x380B6B3CU; // a positive normal FP32 value
    const q4::NgramRowFormat formats[]  = {q4::NgramRowFormat::Fp8E4m3, q4::NgramRowFormat::Iq4Nl,
                                           q4::NgramRowFormat::Bf16, q4::NgramRowFormat::Nvfp4,
                                           q4::NgramRowFormat::Bf16P12};
    const std::uint32_t row_bytes[]     = {160, 90, 320, 90, 256};
    const std::uint16_t tags[]          = {1, 2, 3, 4, 6};
    constexpr int kFormats              = 5;
    constexpr int kArtifactFormats      = 4; // the formats an artifact binds as its own (not BF16 P12)
    const auto scale_of = [&](int format) { return formats[format] == q4::NgramRowFormat::Fp8E4m3 ? kScale : std::uint16_t{0}; };
    const auto multiplier_of = [&](int format) {
        return formats[format] == q4::NgramRowFormat::Nvfp4 ? kMultiplier : std::uint32_t{0};
    };
    for (int own = 0; own < kArtifactFormats; ++own) {
        q4::PleConfig ple;
        ple.embed_dim             = 2560;
        ple.ngram.ngram_size      = 3;
        ple.ngram.heads_per_ngram = 8; // 16 heads of 160
        ple.table.format          = formats[own];
        ple.table.rows            = rows;
        ple.table.row_bytes       = row_bytes[own];
        ple.table.volume_id.fill(0xAA);
        const std::string name = "artifact table " + std::to_string(own) + ": ";
        const auto refused = [&] {
            try {
                (void)q4::resolve_ngram_volume(path, ple);
            } catch (const std::runtime_error&) { return true; }
            return false;
        };

        write_header(path, rows, row_bytes[own], 0xAA, 0, tags[own], multiplier_of(own));
        const q4::NgramTableConfig mine = q4::resolve_ngram_volume(path, ple);
        check(mine.format == formats[own] && mine.volume_id == ple.table.volume_id && !mine.fp8_scale &&
                  !mine.nvfp4_scale,
              name + "the artifact's own volume keeps its table (and its own scale)");
        for (int other = 0; other < kFormats; ++other) {
            // Another volume of any format, the artifact's own format included (another id), tagged as
            // volumes are written now.
            write_header(path, rows, row_bytes[other], 0x55, scale_of(other), tags[other], multiplier_of(other));
            const q4::NgramTableConfig t = q4::resolve_ngram_volume(path, ple);
            const std::uint32_t per_block = kBlockBytes / row_bytes[other];
            const bool fp8 = formats[other] == q4::NgramRowFormat::Fp8E4m3, nvfp4 = formats[other] == q4::NgramRowFormat::Nvfp4;
            check(t.format == formats[other] && t.row_bytes == row_bytes[other] && t.rows_per_block == per_block &&
                      t.blocks == (rows + per_block - 1) / per_block && t.file_bytes == 4096 + t.blocks * 4096 &&
                      t.volume_id[0] == 0x55 &&
                      t.fp8_scale == (fp8 ? std::optional<std::uint16_t>(kScale) : std::nullopt) &&
                      t.nvfp4_scale == (nvfp4 ? std::optional<std::uint32_t>(kMultiplier) : std::nullopt),
                  name + "another volume of format " + std::to_string(other) + " replaces the table");
            // As published before the tag: the row size tells the format (BF16 P12 never predates it).
            if (!nvfp4 && formats[other] != q4::NgramRowFormat::Bf16P12) {
                write_header(path, rows, row_bytes[other], 0x55, scale_of(other));
                check(q4::resolve_ngram_volume(path, ple).format == formats[other],
                      name + "an untagged volume of format " + std::to_string(other) + " is read by its row size");
            }
            write_header(path, rows + 1, row_bytes[other], 0x55, kScale, tags[other], multiplier_of(other));
            check(refused(), name + "another volume of another row count is refused");
        }
        write_header(path, rows, 90, 0x55, 0, 4, 0);
        check(refused(), name + "an NVFP4 volume without a multiplier is refused");
        write_header(path, rows, 90, 0x55, 0, 4, 0x80000001U);
        check(refused(), name + "an NVFP4 volume with a negative multiplier is refused");
        write_header(path, rows, 160, 0x55, kScale, 4, kMultiplier);
        check(refused(), name + "a tag that disagrees with the row size is refused");
        write_header(path, rows, 90, 0x55, 0, 9, kMultiplier);
        check(refused(), name + "an unknown format tag is refused");
        write_header(path, rows, 160, 0x55);
        check(refused(), name + "another FP8 volume without a recorded scale is refused");
        // NVIDIA's published FP8 volume (written before volumes recorded a scale): its id names its scale.
        write_header(path, rows, 160, 0x55);
        {
            constexpr std::uint8_t kNvidia[16] = {0x59, 0xE5, 0xD7, 0x7E, 0x80, 0x2A, 0xC9, 0x5D,
                                                  0x8F, 0xAF, 0x0A, 0x80, 0x0B, 0x8F, 0x23, 0x01};
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(44);
            file.write(reinterpret_cast<const char*>(kNvidia), 16);
        }
        if (formats[own] != q4::NgramRowFormat::Fp8E4m3) {
            const q4::NgramTableConfig nvidia = q4::resolve_ngram_volume(path, ple);
            check(nvidia.format == q4::NgramRowFormat::Fp8E4m3 && nvidia.fp8_scale == std::uint16_t{0x3951},
                  name + "NVIDIA's published FP8 volume brings its known scale");
        }
        write_header(path, rows, 160, 0x55, 0xBA83);
        check(refused(), name + "a negative recorded scale is refused");
        write_header(path, rows, 160, 0x55, 0x7F80);
        check(refused(), name + "an infinite recorded scale is refused");
        write_header(path, rows, 200, 0x55);
        check(refused(), name + "a row size of no known format is refused");
    }
    // BF16 P12 rows: W + W/2 + 16 bytes (160 values -> 256 bytes, 16 rows per 4 KiB block), replacing an
    // FP8 artifact's table. The tag names the format; a row size that is not its formula is refused.
    {
        q4::PleConfig ple;
        ple.embed_dim             = 2560;
        ple.ngram.ngram_size      = 3;
        ple.ngram.heads_per_ngram = 8; // 16 heads of 160
        ple.table.format          = q4::NgramRowFormat::Fp8E4m3;
        ple.table.rows            = rows;
        ple.table.row_bytes       = 160;
        ple.table.volume_id.fill(0xAA);
        const auto refused = [&] {
            try {
                (void)q4::resolve_ngram_volume(path, ple);
            } catch (const std::runtime_error&) { return true; }
            return false;
        };
        write_header(path, rows, 256, 0x55, 0, 6, 0);
        const q4::NgramTableConfig p12 = q4::resolve_ngram_volume(path, ple);
        check(p12.format == q4::NgramRowFormat::Bf16P12 && p12.row_bytes == 256 && p12.rows_per_block == 16 &&
                  p12.blocks == (rows + 15) / 16 && p12.file_bytes == 4096 + p12.blocks * 4096 && !p12.fp8_scale &&
                  !p12.nvfp4_scale && p12.volume_id[0] == 0x55,
              "a BF16 P12 volume of 256-byte rows replaces the table");
        write_header(path, rows, 320, 0x55, 0, 6, 0);
        check(refused(), "a BF16 P12 tag with BF16 row bytes is refused");
        write_header(path, rows, 256, 0x55, 0, 9, 0);
        check(refused(), "a format tag past the known formats is refused");
        const std::uint8_t bases[8] = {96, 120, 200, 220, 0, 0, 0, 0};
        write_blockdir_header(path, rows, bases);
        const q4::NgramTableConfig blockdir = q4::resolve_ngram_volume(path, ple);
        check(blockdir.format == q4::NgramRowFormat::Bf16P12Blockdir && blockdir.row_bytes == 256 &&
                  blockdir.rows_per_block == 17 && blockdir.blocks == (rows + 16) / 17 &&
                  blockdir.file_bytes == 4096 + blockdir.blocks * 4096 && blockdir.block_width == 160 &&
                  blockdir.slot_bytes == 240 && blockdir.blockdir_bases ==
                      std::array<std::uint8_t, 8>{96, 120, 200, 220, 0, 0, 0, 0} && blockdir.side_bytes == 0,
              "a BF16 P12 blockdir volume replaces the table");
        write_blockdir_header(path, rows, bases, 4096);
        const q4::NgramTableConfig spilled = q4::resolve_ngram_volume(path, ple);
        check(spilled.side_offset == 4096 + spilled.blocks * 4096 && spilled.side_bytes == 4096 &&
                  spilled.side_count == 1 && spilled.file_bytes == spilled.side_offset + 4096,
              "a blockdir volume's side stream extends its file");
        const std::uint8_t bad_bases[8] = {96, 120, 200, 241, 0, 0, 0, 0};
        write_blockdir_header(path, rows, bad_bases);
        check(refused(), "a blockdir base outside [0, 240] is refused");
        write_blockdir_header(path, rows, bases); // then a wrong W
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        const std::uint16_t width = 128;
        file.seekp(68);
        file.write(reinterpret_cast<const char*>(&width), 2);
        file.close();
        check(refused(), "a blockdir W other than 160 is refused");
        write_header(path, rows, 320, 0x55, 0, 7, 0);
        check(refused(), "a blockdir tag with P12 row bytes is refused");
    }
}

// A corrupt header recording rows of 0 bytes, for a head width (150) no block format divides, where
// IQ4_NL's and NVFP4's row size is 0 too: refused, tagged or not, for another volume and for the artifact's
// own id, instead of dividing the block by its row size.
void test_zero_row_bytes(const std::filesystem::path& path) {
    q4::PleConfig ple;
    ple.embed_dim             = 2400;
    ple.ngram.ngram_size      = 3;
    ple.ngram.heads_per_ngram = 8; // 16 heads of 150
    ple.table.format          = q4::NgramRowFormat::Fp8E4m3;
    ple.table.rows            = 100'003;
    ple.table.row_bytes       = 150;
    ple.table.volume_id.fill(0xAA);
    check(ple.head_width() == 150, "the corrupt-header table has heads of 150");
    for (const std::uint16_t tag : {std::uint16_t{0}, std::uint16_t{2}, std::uint16_t{4}}) {
        for (const std::uint8_t id : {std::uint8_t{0x55}, std::uint8_t{0xAA}}) {
            write_header(path, ple.table.rows, 0, id, 0, tag, 0x380B6B3CU);
            bool refused = false;
            try {
                (void)q4::resolve_ngram_volume(path, ple);
            } catch (const std::runtime_error&) { refused = true; }
            check(refused, "a header of 0-byte rows (tag " + std::to_string(tag) + ", id " + std::to_string(id) +
                               ") is refused");
        }
    }
}


// A blockdir volume read: each row resolves to the canonical P12 row (its base from the directory's
// 3-bit code, its escaped values from the four slots or, for a block past them, the side stream), and
// the row cache holds that self-contained row, so a second call returns it without the directory.
void test_blockdir(const std::filesystem::path& path) {
    constexpr std::uint64_t rows = 40; // three blocks: 17, 17 and 6 rows
    const std::uint8_t bases[8] = {96, 120, 200, 220, 0, 0, 0, 0};
    write_blockdir_header(path, rows, bases, 4096);
    std::vector<std::uint8_t> blocks(3 * 4096, 0);
    // Block 0: four rows with one escape each in one of the four slots (delta +20 from the row's
    // base), the rows' bases from the directory's 3-bit codes 0..3.
    for (std::uint32_t r = 0; r < 17; ++r) {
        const bool escaped = r < 4;
        put_blockdir_plane(blocks.data(), r, escaped ? bases[r] : 96, escaped ? static_cast<int>(5 + r) : -1,
                           escaped ? bases[r] + 20 : 0);
    }
    for (std::uint32_t r = 0; r < 4; ++r) {
        put_directory_bits(blocks.data() + 17 * 240, 3 * r, 3, r);
        put_directory_bits(blocks.data() + 17 * 240, 3 * 17 + 3 + 18 * r, 18,
                           (r * 160U + 5U + r) | (20U << 12U));
    }
    put_directory_bits(blocks.data() + 17 * 240, 3 * 17, 3, 4);
    // Block 1: six rows each with one escape at exponent 40 (delta -56, outside six signed bits):
    // the whole block's escapes go to the side stream (count 7).
    for (std::uint32_t r = 0; r < 17; ++r) { put_blockdir_plane(blocks.data() + 4096, r, 96, r < 6 ? 3 : -1, 40); }
    put_directory_bits(blocks.data() + 4096 + 17 * 240, 3 * 17, 3, 7);
    // Block 2: no escapes.
    for (std::uint32_t r = 0; r < 6; ++r) { put_blockdir_plane(blocks.data() + 2 * 4096, r, 96, -1, 0); }
    std::vector<std::uint8_t> side(4096, 0);
    const std::uint32_t side_block = 1;
    std::memcpy(side.data(), &side_block, 4);
    const std::uint16_t side_count = 6;
    std::memcpy(side.data() + 4, &side_count, 2);
    for (std::uint32_t e = 0; e < 6; ++e) {
        const std::uint16_t position = static_cast<std::uint16_t>(e * 160 + 3);
        const std::uint16_t bits     = static_cast<std::uint16_t>((40 << 7) | 0x11);
        std::memcpy(side.data() + 6 + 4 * e, &position, 2);
        std::memcpy(side.data() + 6 + 4 * e + 2, &bits, 2);
    }
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out.write(reinterpret_cast<const char*>(blocks.data()), static_cast<std::streamsize>(blocks.size()));
        out.write(reinterpret_cast<const char*>(side.data()), static_cast<std::streamsize>(side.size()));
    }
    q4::PleConfig ple;
    ple.embed_dim             = 2560;
    ple.ngram.ngram_size      = 3;
    ple.ngram.heads_per_ngram = 8;
    ple.table.format          = q4::NgramRowFormat::Fp8E4m3;
    ple.table.rows            = rows;
    ple.table.row_bytes       = 160;
    ple.table.volume_id.fill(0xAA);
    const q4::NgramTableConfig table = q4::resolve_ngram_volume(path, ple);
    check(table.format == q4::NgramRowFormat::Bf16P12Blockdir && table.file_bytes == 3 * 4096 + 2 * 4096,
          "the blockdir test volume resolves");
    const q4::NgramVolume volume(path, table);
    std::vector<std::uint8_t> want(rows * 256, 0);
    for (std::uint64_t r = 0; r < rows; ++r) {
        const std::uint64_t block = r / 17, slot = r % 17;
        std::memcpy(want.data() + r * 256, blocks.data() + block * 4096 + slot * 240, 240);
        std::uint8_t* row = want.data() + r * 256;
        const bool inline_escape = block == 0 && slot < 4;
        const bool spilled       = block == 1 && slot < 6;
        row[240]                 = bases[inline_escape ? slot : 0];
        row[241]                 = static_cast<std::uint8_t>(inline_escape || spilled ? 1 : 0);
        if (inline_escape) {
            const std::uint16_t value = static_cast<std::uint16_t>(((bases[slot] + 20) << 7) | 0x11);
            row[242] = static_cast<std::uint8_t>(5 + slot);
            row[243] = static_cast<std::uint8_t>(value & 0xFF);
            row[244] = static_cast<std::uint8_t>(value >> 8);
        } else if (spilled) {
            row[242] = 3;
            row[243] = static_cast<std::uint8_t>(((40 << 7) | 0x11) & 0xFF);
            row[244] = static_cast<std::uint8_t>(((40 << 7) | 0x11) >> 8);
        }
    }
    std::vector<std::uint32_t> all(static_cast<std::size_t>(rows));
    for (std::uint32_t r = 0; r < rows; ++r) { all[r] = r; }
    std::vector<std::byte> got(rows * 256);
    volume.read_rows(all, got);
    bool same = std::memcmp(got.data(), want.data(), want.size()) == 0;
    check(same, "every blockdir row resolves to its canonical P12 row");
    if (!same) {
        for (std::uint64_t r = 0; r < rows; ++r) {
            if (std::memcmp(got.data() + r * 256, want.data() + r * 256, 256) != 0) {
                std::fprintf(stderr, "  row %llu differs\n", static_cast<unsigned long long>(r));
                break;
            }
        }
    }
    // The cache holds the resolved rows: the second call reads nothing and returns the same bytes.
    const auto before     = volume.counters();
    std::vector<std::uint32_t> again{0, 1, 17, 33, 34, 0, 1};
    std::vector<std::byte> cached(again.size() * 256);
    volume.read_rows(again, cached);
    check(volume.counters().reads == before.reads && volume.counters().hits - before.hits == again.size(),
          "the blockdir row cache serves the second call");
    bool matched = true;
    for (std::size_t i = 0; i < again.size(); ++i) {
        matched = matched && std::memcmp(cached.data() + i * 256, want.data() + again[i] * 256, 256) == 0;
    }
    check(matched, "a cached blockdir row is its self-contained P12 row");
}


// A blockdir plane: every value's exponent base + j % 15 with code 0x11; each escaped column takes
// q = 15 and its exact exponent (the escape's value is (exponent << 7) | 0x11).
void put_blockdir_row(std::uint8_t* block, std::uint32_t row_in_block, unsigned base,
                      const std::vector<std::pair<int, int>>& escapes) {
    std::uint8_t* plane = block + row_in_block * 240;
    for (int j = 0; j < 160; ++j) {
        unsigned q = static_cast<unsigned>(j % 15);
        for (const auto& [column, exponent] : escapes) {
            if (column == j) { q = 15; }
        }
        (void)base;
        (void)escapes;
        plane[j] = 0x11;
        plane[160 + j / 2] |= static_cast<std::uint8_t>(q << (4 * (j & 1)));
    }
}

// The resident-mode staging the blockdir kernel reads must decode, row for row, to the same words as
// the host canonicalizer: the packed rows of a resident table through the volume's base table and the
// block directory, the canonical rows of a demand read through the P12 trailer. The fixture exercises
// all eight base-table entries, four inline slots (one row with two escapes), a block whose scan must
// skip empty slots, and a block resolved from the side stream.
void test_blockdir_packed(const std::filesystem::path& path) {
    constexpr std::uint64_t rows = 51; // three blocks
    const std::uint8_t bases[8] = {53, 54, 105, 106, 107, 108, 109, 200};
    write_blockdir_header(path, rows, bases, 4096);
    std::vector<std::uint8_t> blocks(3 * 4096, 0);
    struct Escape { int row, column, exponent; };
    // Block 0: one escape in four rows, each row's base a different table entry.
    const std::vector<Escape> block0 = {{1, 7, 73}, {2, 9, 129}, {3, 11, 135}, {4, 13, 134}};
    // Block 1: one row with two escapes and two empty slots (the scan must skip them).
    const std::vector<Escape> block1 = {{5, 3, 108}, {5, 8, 109}};
    for (std::uint32_t r = 0; r < 17; ++r) {
        std::vector<std::pair<int, int>> collapsed0, collapsed1;
        for (const auto& e : block0) {
            if (e.row == static_cast<int>(r)) { collapsed0.push_back({e.column, e.exponent}); }
        }
        for (const auto& e : block1) {
            if (e.row == static_cast<int>(r)) { collapsed1.push_back({e.column, e.exponent}); }
        }
        put_blockdir_row(blocks.data(), r, bases[r % 8], collapsed0);
        put_blockdir_row(blocks.data() + 4096, r, bases[r % 8], collapsed1);
        std::vector<std::pair<int, int>> collapsed2;
        const int side_rows_at[3]    = {2, 7, 7};
        const int side_columns_at[3] = {4, 9, 12};
        const int side_exponents_at[3] = {77, 80, 83};
        for (unsigned e = 0; e < 3; ++e) {
            if (side_rows_at[e] == static_cast<int>(r)) {
                collapsed2.push_back({side_columns_at[e], side_exponents_at[e]});
            }
        }
        put_blockdir_row(blocks.data() + 2 * 4096, r, bases[r % 8], collapsed2);
    }
    std::uint8_t* directory0 = blocks.data() + 17 * 240;
    std::uint8_t* directory1 = blocks.data() + 4096 + 17 * 240;
    std::uint8_t* directory2 = blocks.data() + 2 * 4096 + 17 * 240;
    for (std::uint32_t r = 0; r < 17; ++r) {
        put_directory_bits(directory0, 3 * r, 3, r % 8);
        put_directory_bits(directory1, 3 * r, 3, r % 8);
        put_directory_bits(directory2, 3 * r, 3, r % 8);
    }
    const auto put_slot = [](std::uint8_t* directory, unsigned slot, int row, int column, int exponent,
                             unsigned base) {
        const unsigned position = static_cast<unsigned>(row) * 160U + static_cast<unsigned>(column);
        const unsigned delta    = static_cast<unsigned>((exponent - static_cast<int>(base)) & 0x3F);
        put_directory_bits(directory, 3 * 17 + 3 + 18 * slot, 18, position | (delta << 12));
    };
    put_directory_bits(directory0, 3 * 17, 3, 4);
    for (unsigned e = 0; e < block0.size(); ++e) {
        put_slot(directory0, e, block0[e].row, block0[e].column, block0[e].exponent, bases[block0[e].row % 8]);
    }
    put_directory_bits(directory1, 3 * 17, 3, 4); // two slots used, two left zero
    for (unsigned e = 0; e < block1.size(); ++e) {
        put_slot(directory1, e, block1[e].row, block1[e].column, block1[e].exponent, bases[block1[e].row % 8]);
    }
    put_directory_bits(directory2, 3 * 17, 3, 7); // the side stream
    std::vector<std::uint8_t> side(4096, 0);
    const std::uint32_t side_block = 2;
    std::memcpy(side.data(), &side_block, 4);
    const std::uint16_t side_count = 3;
    std::memcpy(side.data() + 4, &side_count, 2);
    const int side_rows[3]    = {2, 7, 7};
    const int side_columns[3] = {4, 9, 12};
    const int side_exponents[3] = {77, 80, 83};
    for (unsigned e = 0; e < 3; ++e) {
        const auto position = static_cast<std::uint16_t>(side_rows[e] * 160 + side_columns[e]);
        const auto word = static_cast<std::uint16_t>((side_exponents[e] << 7) | 0x11);
        std::memcpy(side.data() + 6 + 4 * e, &position, 2);
        std::memcpy(side.data() + 6 + 4 * e + 2, &word, 2);
    }
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out.write(reinterpret_cast<const char*>(blocks.data()), static_cast<std::streamsize>(blocks.size()));
        out.write(reinterpret_cast<const char*>(side.data()), static_cast<std::streamsize>(side.size()));
    }
    q4::PleConfig ple;
    ple.embed_dim             = 2560;
    ple.ngram.ngram_size      = 3;
    ple.ngram.heads_per_ngram = 8;
    ple.table.format          = q4::NgramRowFormat::Fp8E4m3;
    ple.table.rows            = rows;
    ple.table.row_bytes       = 160;
    ple.table.volume_id.fill(0xAA);
    const q4::NgramTableConfig table = q4::resolve_ngram_volume(path, ple);
    check(table.format == q4::NgramRowFormat::Bf16P12Blockdir,
          "the blockdir packed-test volume resolves");
    check(q4::NgramVolume::consumer_row_bytes(table, true) == 272 &&
              q4::NgramVolume::consumer_row_bytes(table, false) == 256,
          "a blockdir row is packed in resident mode and canonical on demand");

    std::vector<std::uint32_t> all(static_cast<std::size_t>(rows));
    for (std::uint32_t r = 0; r < rows; ++r) { all[r] = r; }
    const std::size_t staging = q4::NgramVolume::consumer_row_bytes(table, true);
    std::vector<std::byte> packed(rows * staging);
    const q4::NgramVolume resident(path, table, true);
    resident.read_rows(all, packed);
    std::vector<std::byte> canonical(rows * 256);
    const q4::NgramVolume demand(path, table, false);
    demand.read_rows(all, canonical);

    // The fixture's exact words: base + j % 15, or the escape's exponent, sign 0 and mantissa 0x11.
    std::vector<std::uint16_t> want(rows * 160);
    for (std::uint64_t r = 0; r < rows; ++r) {
        const std::uint64_t block = r / 17, slot = r % 17;
        const unsigned base = bases[slot % 8];
        int escaped_column = -1, escaped_exponent = 0, escaped_column2 = -1, escaped_exponent2 = 0;
        if (block == 0) {
            for (const auto& e : block0) {
                if (static_cast<std::uint64_t>(e.row) == slot) { escaped_column = e.column; escaped_exponent = e.exponent; }
            }
        } else if (block == 1) {
            bool first = true;
            for (const auto& e : block1) {
                if (static_cast<std::uint64_t>(e.row) == slot && first) {
                    escaped_column = e.column;
                    escaped_exponent = e.exponent;
                    first = false;
                } else if (static_cast<std::uint64_t>(e.row) == slot) {
                    escaped_column2 = e.column;
                    escaped_exponent2 = e.exponent;
                }
            }
        } else {
            for (unsigned e = 0; e < 3; ++e) {
                if (static_cast<std::uint64_t>(side_rows[e]) == slot && side_columns[e] != escaped_column) {
                    if (escaped_column < 0) { escaped_column = side_columns[e]; escaped_exponent = side_exponents[e]; }
                    else { escaped_column2 = side_columns[e]; escaped_exponent2 = side_exponents[e]; }
                }
            }
        }
        for (int j = 0; j < 160; ++j) {
            unsigned exponent = base + static_cast<unsigned>(j % 15);
            if (j == escaped_column) { exponent = static_cast<unsigned>(escaped_exponent); }
            if (j == escaped_column2) { exponent = static_cast<unsigned>(escaped_exponent2); }
            want[r * 160 + j] = static_cast<std::uint16_t>((exponent << 7) | 0x11);
        }
    }

    // The staging layout the kernel reads: planes, the block directory, the row's index; a side-stream
    // block's rows are canonical rows marked at the header byte.
    bool layout = true;
    for (std::uint64_t r = 0; r < rows; ++r) {
        const std::uint64_t block = r / 17, slot = r % 17;
        const std::uint8_t* row = reinterpret_cast<const std::uint8_t*>(packed.data() + r * staging);
        layout = layout && std::memcmp(row, blocks.data() + block * 4096 + slot * 240, 240) == 0;
        if (block == 2) {
            layout = layout && row[256] == 0xFF;
        } else {
            layout = layout && std::memcmp(row + 240, blocks.data() + block * 4096 + 17 * 240, 16) == 0 &&
                     row[256] == slot;
        }
        for (std::size_t b = 257; b < staging; ++b) { layout = layout && row[b] == 0; }
    }
    check(layout, "a resident blockdir row stages its planes, its block directory and its index");

    // The kernel's decode (the same header functions the device kernel calls) over the staged rows.
    bool packed_ok = true;
    for (std::uint64_t r = 0; r < rows; ++r) {
        const std::uint8_t* row = reinterpret_cast<const std::uint8_t*>(packed.data() + r * staging);
        for (int j = 0; j < 160; ++j) {
            const std::uint8_t code = row[j];
            const unsigned q = (row[160 + (j >> 1)] >> (4 * (j & 1))) & 15U;
            std::uint16_t bits;
            if (row[256] == 0xFF) {
                bits = infernix::ngram::p12_trailer_value_bits(row + 240, j, code, q);
            } else {
                bits = infernix::ngram::blockdir_value_bits(row + 240, bases, row[256], 160, j, code, q);
            }
            if (bits != want[r * 160 + j]) {
                std::printf("  packed row %llu value %d: got 0x%04x want 0x%04x\n",
                            static_cast<unsigned long long>(r), j, bits, want[r * 160 + j]);
                packed_ok = false;
                break;
            }
        }
        if (!packed_ok) { break; }
    }
    check(packed_ok, "the packed row decode matches the fixture's words");

    // The host canonicalizer's demand rows decode to the same words through the P12 trailer.
    bool demand_ok = true;
    for (std::uint64_t r = 0; r < rows; ++r) {
        const std::uint8_t* row = reinterpret_cast<const std::uint8_t*>(canonical.data() + r * 256);
        for (int j = 0; j < 160; ++j) {
            const std::uint8_t code = row[j];
            const unsigned q = (row[160 + (j >> 1)] >> (4 * (j & 1))) & 15U;
            const std::uint16_t bits = infernix::ngram::p12_trailer_value_bits(row + 240, j, code, q);
            if (bits != want[r * 160 + j]) { demand_ok = false; break; }
        }
        if (!demand_ok) { break; }
    }
    check(demand_ok, "the host canonicalizer's rows match the fixture's words");
}

} // namespace

int main() {
    const auto path = infernix::test::unique_temp_path("infernix_ngram_volume_test.ngram");
    try {
        const q4::NgramTableConfig table = make_volume(path);
        const q4::NgramVolume volume(path, table);
        std::mt19937 rng(20261004);

        // Random rows (cold): one read per distinct block.
        std::vector<std::uint32_t> rows(500);
        for (auto& r : rows) { r = static_cast<std::uint32_t>(rng() % kRows); }
        auto before = volume.counters();
        check(rows_match(rows, read(volume, rows)), "random rows equal the oracle");
        check(volume.counters().reads - before.reads == distinct_blocks(rows), "one read per distinct block");

        // Duplicates and rows sharing a block, in one call: one read per distinct block.
        const std::uint32_t base = 1'000'000 / kRowsPerBlock * kRowsPerBlock; // a block's first row
        std::vector<std::uint32_t> shared{base, base + 1, base + 5, base, base + 1, base + kRowsPerBlock * 3};
        before = volume.counters();
        check(rows_match(shared, read(volume, shared)), "duplicates and shared blocks equal the oracle");
        check(volume.counters().reads - before.reads == 2, "two distinct blocks, two reads");

        // More than 64 distinct blocks: the ring wraps.
        std::vector<std::uint32_t> many;
        for (std::uint32_t b = 0; b < 200; ++b) { many.push_back((b * 37 + 11) * kRowsPerBlock + b % kRowsPerBlock); }
        before = volume.counters();
        check(rows_match(many, read(volume, many)), "200 distinct blocks equal the oracle");
        check(volume.counters().reads - before.reads == 200, "200 reads through a 64-block ring");

        // The same rows again: host-cache hits, no reads.
        before = volume.counters();
        check(rows_match(many, read(volume, many)), "cached rows equal the oracle");
        check(volume.counters().reads == before.reads && volume.counters().hits - before.hits == many.size(),
              "a second call hits the cache");
        check(volume.cached(many), "cached() reports rows already read");
        std::vector<std::uint32_t> with_new = many;
        std::uint32_t unread = 7;
        const auto seen = [&](std::uint32_t r) {
            return std::find(rows.begin(), rows.end(), r) != rows.end() ||
                   std::find(shared.begin(), shared.end(), r) != shared.end() ||
                   std::find(many.begin(), many.end(), r) != many.end();
        };
        while (seen(unread)) { unread += 9973; }
        with_new.push_back(unread); // never read
        check(!volume.cached(with_new), "cached() reports a row not yet read");
        check(!volume.cached(std::vector<std::uint32_t>{static_cast<std::uint32_t>(kRows)}), "cached() rejects a row past the table");

        // The startup read-rate probe times real reads on the large-call route and leaves the row cache
        // and counters alone; a zero time limit stops it after its first round.
        before                  = volume.counters();
        const auto probe        = volume.probe_reads(64, 60.0);
        const auto rounds       = volume.probe_reads(1000, 60.0);
        const auto limited      = volume.probe_reads(1'000'000, 0.0);
        check(probe.reads == 64 && probe.seconds > 0.0 && probe.reads_per_second() > 0.0,
              "the read probe completes its reads and times them");
        check(rounds.reads == 1000, "the read probe completes a count that is not a whole number of rounds");
        check(limited.reads == q4::NgramVolume::kReadThreads * q4::NgramVolume::kParallelInFlight,
              "the read probe stops at its time limit after its first round");
        check(volume.counters().rows == before.rows && volume.counters().reads == before.reads &&
                  volume.counters().hits == before.hits && volume.cached(many) && !volume.cached(with_new),
              "the read probe leaves the counters and the row cache unchanged");

        // Two rows of one cache slot (the slot is a hash of the row id; brute-force a collision).
        const auto slot_of = [](std::uint32_t row) { return (row * 2654435761U) >> 12U; };
        std::uint32_t a = 0, b = 0;
        for (std::uint32_t r = 1; r < kRows && b == 0; ++r) {
            if (slot_of(r) == slot_of(0)) { b = r; }
        }
        check(b != 0, "the test found two rows sharing a cache slot");
        const std::vector<std::uint32_t> pair{a, b, a, b};
        check(rows_match(pair, read(volume, pair)), "rows sharing a cache slot return their own bytes");
        check(rows_match(pair, read(volume, pair)), "and again after evicting each other");

        // A prefill-sized call (thousands of distinct blocks, duplicates and cache-slot collisions):
        // the multi-threaded route, then every row from the cache.
        std::vector<std::uint32_t> large(20000);
        for (auto& r : large) { r = static_cast<std::uint32_t>(rng() % kRows); }
        for (std::size_t i = 0; i < 2000; ++i) { large[rng() % large.size()] = large[rng() % large.size()]; }
        const std::uint64_t large_blocks = [&] {
            std::set<std::uint32_t> missing;
            for (const auto r : large) {
                if (!volume.cached(std::vector<std::uint32_t>{r})) { missing.insert(r / kRowsPerBlock); }
            }
            return static_cast<std::uint64_t>(missing.size());
        }();
        check(large_blocks >= 4000, "the large call has thousands of uncached blocks");
        before = volume.counters();
        check(rows_match(large, read(volume, large)), "a prefill-sized call equals the oracle");
        check(volume.counters().reads - before.reads == large_blocks, "one read per distinct uncached block");
        check(rows_match(large, read(volume, large)), "the large call's rows again equal the oracle");

        bool threw = false;
        try {
            (void)read(volume, {static_cast<std::uint32_t>(kRows)});
        } catch (const std::out_of_range&) { threw = true; }
        check(threw, "a row past the table throws out_of_range");

        // RAM residency: the whole table read at open into memory locked for the CPU (not page-locked for
        // the device); every call is served from it with the volume's bytes and no further reads.
        {
            const q4::NgramVolume ram(path, table, true);
            check(ram.resident(), "a resident table reports its residency");
            const auto opened = ram.counters();
            check(rows_match(rows, read(ram, rows)) && rows_match(shared, read(ram, shared)) &&
                      rows_match(many, read(ram, many)) && rows_match(large, read(ram, large)) &&
                      rows_match(pair, read(ram, pair)),
                  "resident rows equal the oracle");
            check(ram.counters().reads == opened.reads, "a resident table reads nothing after its open");
            if (const std::string why = ram.resident_lock_failure(); !why.empty()) {
                std::printf("note: the resident table was not locked in RAM (%s)\n", why.c_str());
            }
            threw = false;
            try {
                (void)read(ram, {static_cast<std::uint32_t>(kRows)});
            } catch (const std::out_of_range&) { threw = true; }
            check(threw, "a resident row past the table throws out_of_range");
        }

        q4::NgramTableConfig wrong = table;
        wrong.file_bytes += 4096;
        threw = false;
        try {
            const q4::NgramVolume bad(path, wrong);
        } catch (const std::runtime_error&) { threw = true; }
        check(threw, "a volume of the wrong size is refused at open");
        const auto headers = infernix::test::unique_temp_path("infernix_ngram_resolve_test.ngram");
        test_resolve(headers); // its own file: the volume above holds `path` without write sharing
        test_zero_row_bytes(headers);
        const auto blockdir_path = infernix::test::unique_temp_path("infernix_ngram_blockdir_test.ngram");
        test_blockdir(blockdir_path);
        const auto blockdir_packed_path = infernix::test::unique_temp_path("infernix_ngram_blockdir_packed.ngram");
        test_blockdir_packed(blockdir_packed_path);
        std::error_code ignored_packed;
        std::filesystem::remove(blockdir_packed_path, ignored_packed);
        std::error_code ignored_blockdir;
        std::filesystem::remove(blockdir_path, ignored_blockdir);
        std::error_code removed;
        std::filesystem::remove(headers, removed);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::printf(failures == 0 ? "ngram volume checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
