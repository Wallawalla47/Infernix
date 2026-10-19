// NgramVolume::read_rows on a synthetic volume (n-gram step S1, design §19.3.4). The oracle is the
// closed-form row contents f(r, k), never the volume: random rows, duplicates and rows sharing a block
// read each distinct block once (counters), more than 64 distinct blocks wrap the ring, a second
// call hits the host cache, two rows of one cache slot both return their own bytes, a row past the
// table throws, and a volume of the wrong size is refused at open. A call of thousands of distinct
// blocks (a prefill chunk) takes the multi-threaded route: the same oracle, counters and cache.
// resolve_ngram_volume accepts a volume of any format in place of the artifact's table and refuses mismatches.

#include "test_temp_path.h"
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
// converter lays them out, with `id` and an FP8 scale word (0: none recorded).
void write_header(const std::filesystem::path& path, std::uint64_t rows, std::uint32_t row_bytes, std::uint8_t id,
                  std::uint16_t scale = 0) {
    std::vector<std::byte> header(4096);
    const std::uint32_t per_block = kBlockBytes / row_bytes;
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
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
}

// resolve_ngram_volume: the artifact's own volume (its format and id) keeps the artifact's table; a
// volume of the same table geometry in any of the three formats (FP8 rows of the head width, IQ4_NL
// blocks of 18 bytes per 32, BF16 rows of twice the width) replaces any artifact's table with its own
// geometry and id, an FP8 one only with a scale in its header; another row count or row size is refused.
void test_resolve(const std::filesystem::path& path) {
    constexpr std::uint64_t rows        = 100'003;
    constexpr std::uint16_t kScale      = 0x3A83; // a positive normal BF16 value
    const q4::NgramRowFormat formats[]  = {q4::NgramRowFormat::Fp8E4m3, q4::NgramRowFormat::Iq4Nl,
                                           q4::NgramRowFormat::Bf16};
    const std::uint32_t row_bytes[]     = {160, 90, 320};
    for (int own = 0; own < 3; ++own) {
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

        write_header(path, rows, row_bytes[own], 0xAA);
        const q4::NgramTableConfig mine = q4::resolve_ngram_volume(path, ple);
        check(mine.format == formats[own] && mine.volume_id == ple.table.volume_id && !mine.fp8_scale,
              name + "the artifact's own volume keeps its table (and its own scale)");
        for (int other = 0; other < 3; ++other) {
            // Another volume of any format, the artifact's own format included (another id).
            write_header(path, rows, row_bytes[other], 0x55, formats[other] == q4::NgramRowFormat::Fp8E4m3 ? kScale : 0);
            const q4::NgramTableConfig t = q4::resolve_ngram_volume(path, ple);
            const std::uint32_t per_block = kBlockBytes / row_bytes[other];
            check(t.format == formats[other] && t.row_bytes == row_bytes[other] && t.rows_per_block == per_block &&
                      t.blocks == (rows + per_block - 1) / per_block && t.file_bytes == 4096 + t.blocks * 4096 &&
                      t.volume_id[0] == 0x55 &&
                      t.fp8_scale == (formats[other] == q4::NgramRowFormat::Fp8E4m3 ? std::optional<std::uint16_t>(kScale)
                                                                                    : std::nullopt),
                  name + "another volume of format " + std::to_string(other) + " replaces the table");
            write_header(path, rows + 1, row_bytes[other], 0x55, kScale);
            check(refused(), name + "another volume of another row count is refused");
        }
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

        q4::NgramTableConfig wrong = table;
        wrong.file_bytes += 4096;
        threw = false;
        try {
            const q4::NgramVolume bad(path, wrong);
        } catch (const std::runtime_error&) { threw = true; }
        check(threw, "a volume of the wrong size is refused at open");
        const auto headers = infernix::test::unique_temp_path("infernix_ngram_resolve_test.ngram");
        test_resolve(headers); // its own file: the volume above holds `path` without write sharing
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
