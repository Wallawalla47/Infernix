// NgramVolume::read_rows on a synthetic volume (n-gram step S1, design §19.3.4). The oracle is the
// closed-form row contents f(r, k), never the volume: random rows, duplicates and rows sharing a block
// read each distinct block once (counters), more than 64 distinct blocks wrap the ring, a second
// call hits the host cache, two rows of one cache slot both return their own bytes, a row past the
// table throws, and a volume of the wrong size is refused at open. A call of thousands of distinct
// blocks (a prefill chunk) takes the multi-threaded route: the same oracle, counters and cache.

#include "models/qwen4_exp/program/ngram_volume.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace q4 = ninfer::models::qwen4_exp;

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

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "ninfer_ngram_volume_test.ngram";
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
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::printf(failures == 0 ? "ngram volume checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
