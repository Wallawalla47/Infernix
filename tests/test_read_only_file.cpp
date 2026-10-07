// ReadOnlyFile::read_direct_blocks (n-gram step S1, design §19.3.4): unbuffered 4 KiB block reads
// through a ring of 1, 4 or 64 blocks, offsets in random order with repeats; every index is
// consumed exactly once with its own block's bytes, and a read past the end throws after the reads
// in flight have drained. FileMapping::None maps nothing and reads the same bytes.

#include "core/read_only_file.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBlock  = 4096;
constexpr std::size_t kBlocks = 300;

std::byte pattern(std::uint64_t block, std::size_t byte) {
    return static_cast<std::byte>((block * 131U + byte * 7U + 3U) & 0xFFU);
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// A 4 KiB-aligned span of `blocks` blocks inside `storage`.
std::span<std::byte> aligned_ring(std::vector<std::byte>& storage, std::size_t blocks) {
    storage.assign((blocks + 1) * kBlock, std::byte{0});
    const auto base = reinterpret_cast<std::uintptr_t>(storage.data());
    return {storage.data() + (kBlock - base % kBlock) % kBlock, blocks * kBlock};
}

void run(const infernix::ReadOnlyFile& file, std::size_t count, std::size_t ring_blocks, std::mt19937& rng) {
    std::vector<std::uint64_t> blocks(count);
    for (auto& b : blocks) { b = rng() % kBlocks; } // random order, repeats likely
    std::vector<std::uint64_t> offsets;
    for (const auto b : blocks) { offsets.push_back(b * kBlock); }
    std::vector<std::byte> storage;
    const auto ring = aligned_ring(storage, ring_blocks);
    std::vector<int> seen(count, 0);
    bool bytes_ok = true;
    file.read_direct_blocks(offsets, kBlock, ring, [&](std::size_t i, std::span<const std::byte> data) {
        ++seen[i];
        for (std::size_t k = 0; k < kBlock; k += 97) { bytes_ok &= data[k] == pattern(blocks[i], k); }
    });
    bool once = true;
    for (const int s : seen) { once &= s == 1; }
    const std::string where = std::to_string(count) + " reads, ring of " + std::to_string(ring_blocks);
    check(once, "every index consumed exactly once (" + where + ")");
    check(bytes_ok, "every block carries its own bytes (" + where + ")");
}

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "infernix_read_only_file_test.bin";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        std::vector<std::byte> block(kBlock);
        for (std::uint64_t b = 0; b < kBlocks; ++b) {
            for (std::size_t k = 0; k < kBlock; ++k) { block[k] = pattern(b, k); }
            out.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(kBlock));
        }
    }
    try {
        const infernix::ReadOnlyFile file(path);
        std::mt19937 rng(20261004);
        for (const std::size_t ring : {1U, 4U, 64U}) {
            for (const std::size_t count : {1U, 63U, 64U, 65U, 1000U}) { run(file, count, ring, rng); }
        }
        // A read past the end: the reads before it still complete, then the call throws.
        std::vector<std::uint64_t> offsets{0, kBlock, kBlocks * kBlock, 2 * kBlock};
        std::vector<std::byte> storage;
        const auto ring = aligned_ring(storage, 4);
        bool threw = false;
        try {
            file.read_direct_blocks(offsets, kBlock, ring, [](std::size_t, std::span<const std::byte>) {});
        } catch (const std::exception&) { threw = true; }
        check(threw, "a read past the end throws");
        // The file stays usable afterwards.
        std::mt19937 again(7);
        run(file, 65, 64, again);
        check(file.mapped_bytes().size() == kBlocks * kBlock, "FileMapping::Whole maps the whole file");

        // FileMapping::None (the n-gram volume): nothing mapped, the same direct reads and live size.
        const infernix::ReadOnlyFile direct(path, infernix::FileMapping::None);
        check(direct.mapped_bytes().empty(), "FileMapping::None maps nothing");
        check(direct.current_bytes() == kBlocks * kBlock, "FileMapping::None reports the file's size");
        std::mt19937 unmapped(11);
        for (const std::size_t count : {1U, 65U, 1000U}) { run(direct, count, 64, unmapped); }
        std::vector<std::byte> one_storage;
        const auto one = aligned_ring(one_storage, 1);
        check(direct.read_direct(5 * kBlock, one) == kBlock && one[0] == pattern(5, 0) && one[4095] == pattern(5, 4095),
              "FileMapping::None read_direct returns the block's bytes");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::printf(failures == 0 ? "read_only_file checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
