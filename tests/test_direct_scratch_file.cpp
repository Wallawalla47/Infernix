// Core DirectScratchFile (hybrid-prefix-cache-spec §5.7, the prefix cache's SSD tier): unbuffered
// writes and reads of a reserved scratch file against a closed-form byte oracle, many batches in
// flight at once, reads submitted behind a backlog of writes, argument checks, the waker, and the
// file's removal when it closes. CPU and disk only.
#include "test_temp_path.h"
#include "core/direct_scratch_file.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using infernix::DirectScratchFile;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kSlab   = 256 * 1024; // a small slab: 64 pages
constexpr std::size_t kSlabs  = 192;
constexpr std::uint64_t kFile = static_cast<std::uint64_t>(kSlab) * kSlabs;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::uint8_t oracle(std::uint64_t slab, std::size_t at, std::uint32_t round) {
    return static_cast<std::uint8_t>(((slab + 1) * 2654435761ULL >> 7) ^ (at * 40503U >> 3) ^ (round * 151U) ^ at);
}

struct Aligned {
    std::byte* p      = nullptr;
    std::size_t bytes = 0;
    explicit Aligned(std::size_t n)
        : p(static_cast<std::byte*>(::operator new(n, std::align_val_t{DirectScratchFile::kAlignment}))), bytes(n) {}
    ~Aligned() { ::operator delete(p, std::align_val_t{DirectScratchFile::kAlignment}); }
    Aligned(const Aligned&)            = delete;
    Aligned& operator=(const Aligned&) = delete;
};

std::vector<DirectScratchFile::Completion> collect(DirectScratchFile& file, std::size_t count) {
    std::vector<DirectScratchFile::Completion> out;
    const auto stop = Clock::now() + std::chrono::seconds(30);
    while (out.size() < count && Clock::now() < stop) {
        if (file.poll(out) == 0) { std::this_thread::yield(); }
    }
    return out;
}

void fill(std::byte* memory, std::uint64_t slab, std::uint32_t round) {
    for (std::size_t i = 0; i < kSlab; ++i) { memory[i] = static_cast<std::byte>(oracle(slab, i, round)); }
}

bool holds(const std::byte* memory, std::uint64_t slab, std::uint32_t round) {
    for (std::size_t i = 0; i < kSlab; ++i) {
        if (static_cast<std::uint8_t>(memory[i]) != oracle(slab, i, round)) { return false; }
    }
    return true;
}

} // namespace

int main() {
    const std::filesystem::path directory = infernix::test::unique_temp_path("scratch_dir");
    std::filesystem::create_directories(directory);
    std::filesystem::path created;
    try {
        DirectScratchFile file(directory, kFile, DirectScratchFile::Options{.read_depth = 4, .write_depth = 2});
        created = file.path();
        check(file.bytes() == kFile, "the reserved size");
        check(created.parent_path() == std::filesystem::absolute(directory), "the file lies in its directory");
        std::atomic<std::uint32_t> woken{0};
        file.set_waker([&] { woken.fetch_add(1); });

        // Every slab written by its own batch (most in flight together), out of order across the file;
        // then the whole file read back in coalesced batches of 3 slabs.
        Aligned memory(kSlab * kSlabs);
        for (std::uint32_t round = 0; round < 2; ++round) {
            std::vector<std::uint64_t> order(kSlabs);
            for (std::uint64_t i = 0; i < kSlabs; ++i) { order[i] = (i * 37 + round * 11) % kSlabs; }
            for (const std::uint64_t slab : order) {
                std::byte* at = memory.p + slab * kSlab;
                fill(at, slab, round);
                file.submit(DirectScratchFile::Batch{.tag = slab,
                                                     .op  = DirectScratchFile::Op::Write,
                                                     .ios = {{.offset = slab * kSlab, .bytes = kSlab, .memory = at}}});
            }
            const auto writes = collect(file, kSlabs);
            check(writes.size() == kSlabs, "every write completes once");
            check(std::all_of(writes.begin(), writes.end(), [](const auto& c) { return c.ok && c.bytes == kSlab; }),
                  "the writes succeed");
            std::fill(memory.p, memory.p + memory.bytes, std::byte{0});
            std::uint64_t tag = 1000;
            for (std::uint64_t slab = 0; slab < kSlabs; slab += 3) {
                DirectScratchFile::Batch batch{.tag = tag++, .op = DirectScratchFile::Op::Read, .ios = {}};
                // Two I/Os: one of two consecutive slabs, one of a single slab.
                batch.ios.push_back({.offset = slab * kSlab, .bytes = 2 * kSlab, .memory = memory.p + slab * kSlab});
                batch.ios.push_back(
                    {.offset = (slab + 2) * kSlab, .bytes = kSlab, .memory = memory.p + (slab + 2) * kSlab});
                file.submit(std::move(batch));
            }
            const auto reads = collect(file, kSlabs / 3);
            check(reads.size() == kSlabs / 3, "every read batch completes once");
            check(std::all_of(reads.begin(), reads.end(), [](const auto& c) { return c.ok && c.bytes == 3 * kSlab; }),
                  "the reads succeed");
            bool exact = true;
            for (std::uint64_t slab = 0; slab < kSlabs; ++slab) { exact = exact && holds(memory.p + slab * kSlab, slab, round); }
            check(exact, "the file returns what was written, round " + std::to_string(round));
        }
        check(woken.load() == 2 * (kSlabs + kSlabs / 3), "the waker runs once per finished batch");

        // Reads submitted behind a backlog of writes finish before the backlog does: no write is
        // issued while a read waits.
        Aligned scratch(kSlab * kSlabs);
        for (std::uint64_t slab = 0; slab < kSlabs; ++slab) {
            file.submit(DirectScratchFile::Batch{
                .tag = 1u << 20 | slab,
                .op  = DirectScratchFile::Op::Write,
                .ios = {{.offset = slab * kSlab, .bytes = kSlab, .memory = memory.p + slab * kSlab}}});
        }
        file.submit(DirectScratchFile::Batch{.tag = 7,
                                             .op  = DirectScratchFile::Op::Read,
                                             .ios = {{.offset = 0, .bytes = kSlab, .memory = scratch.p}}});
        const auto mixed = collect(file, kSlabs + 1);
        const auto read_at = std::find_if(mixed.begin(), mixed.end(), [](const auto& c) { return c.tag == 7; });
        check(read_at != mixed.end() && read_at->ok, "the read behind the writes completes");
        check(read_at != mixed.end() && static_cast<std::size_t>(read_at - mixed.begin()) < kSlabs / 2,
              "the read overtakes most of the queued writes");
        file.wait_idle();
        check(file.outstanding() == 0, "nothing outstanding once every batch was returned");

        // Argument checks.
        const auto refused = [&](DirectScratchFile::Io io) {
            try {
                file.submit(DirectScratchFile::Batch{.tag = 9, .op = DirectScratchFile::Op::Read, .ios = {io}});
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        check(refused({.offset = 512, .bytes = kSlab, .memory = scratch.p}), "a misaligned offset");
        check(refused({.offset = 0, .bytes = 1000, .memory = scratch.p}), "a misaligned size");
        check(refused({.offset = 0, .bytes = kSlab, .memory = scratch.p + 16}), "a misaligned address");
        check(refused({.offset = kFile - 4096, .bytes = 8192, .memory = scratch.p}), "a range past the end");
        check(refused({.offset = 0, .bytes = 0, .memory = scratch.p}), "an empty transfer");
        try {
            file.submit(DirectScratchFile::Batch{.tag = 9, .op = DirectScratchFile::Op::Read, .ios = {}});
            check(false, "an empty batch is refused");
        } catch (const std::invalid_argument&) {}
#ifdef _WIN32
        check(std::filesystem::exists(created), "the file exists while open");
#else
        check(!std::filesystem::exists(created), "the file is unlinked at once (its space returns when it closes)");
#endif
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    }
    check(!created.empty() && !std::filesystem::exists(created), "the file is removed when closed");
    try {
        DirectScratchFile missing(directory / "missing", kFile);
        check(false, "a missing directory is refused");
    } catch (const std::system_error&) {}
    try {
        DirectScratchFile odd(directory, 1000);
        check(false, "a size not a multiple of 4 KiB is refused");
    } catch (const std::invalid_argument&) {}
    // A kept file: reopened with another size it starts over at that size on the drive (a smaller one is
    // cut, on every platform), and a finished one opens for reading only, shared and unchanged.
    try {
        const std::filesystem::path kept = directory / "kept.bin";
        Aligned memory(4 * kSlab);
        std::uint64_t hashes[4] = {};
        {
            DirectScratchFile file(directory, 8 * kSlab, DirectScratchFile::Options{.kept_file = kept});
            check(!file.kept_content(), "a new kept file holds nothing");
        }
        check(std::filesystem::file_size(kept) == 8 * kSlab, "a kept file stays, at its size");
        {
            DirectScratchFile file(directory, 4 * kSlab, DirectScratchFile::Options{.kept_file = kept});
            check(!file.kept_content(), "a kept file of another size is not the previous run's");
            for (std::uint64_t slab = 0; slab < 4; ++slab) { fill(memory.p + slab * kSlab, slab, 5); }
            file.submit(DirectScratchFile::Batch{
                .tag = 1,
                .op  = DirectScratchFile::Op::Write,
                .ios = {{.offset = 0, .bytes = 4 * kSlab, .memory = memory.p, .hashes = hashes, .hash_stride = kSlab}}});
            const auto written = collect(file, 1);
            check(written.size() == 1 && written.front().ok, "the kept file is written");
            file.flush();
        }
        check(std::filesystem::file_size(kept) == 4 * kSlab, "a kept file reopened smaller is cut to the new size");
        std::fill(memory.p, memory.p + memory.bytes, std::byte{0});
        {
            DirectScratchFile reader(directory, 4 * kSlab, DirectScratchFile::Options{.kept_file = kept, .read_only = true});
            DirectScratchFile second(directory, 4 * kSlab, DirectScratchFile::Options{.kept_file = kept, .read_only = true});
            check(reader.kept_content() && second.kept_content(), "a finished file opens for reading, twice at once");
            std::uint64_t read_hashes[4] = {};
            reader.submit(DirectScratchFile::Batch{
                .tag = 2,
                .op  = DirectScratchFile::Op::Read,
                .ios = {{.offset = 0, .bytes = 4 * kSlab, .memory = memory.p, .hashes = read_hashes, .hash_stride = kSlab}}});
            const auto got = collect(reader, 1);
            bool exact     = got.size() == 1 && got.front().ok;
            for (std::uint64_t slab = 0; slab < 4; ++slab) {
                exact = exact && holds(memory.p + slab * kSlab, slab, 5) && read_hashes[slab] == hashes[slab] && hashes[slab] != 0;
            }
            check(exact, "it returns what was written, each slab with the hash taken when it was written");
            try {
                reader.submit(DirectScratchFile::Batch{.tag = 3,
                                                       .op  = DirectScratchFile::Op::Write,
                                                       .ios = {{.offset = 0, .bytes = kSlab, .memory = memory.p}}});
                check(false, "a write to a read-only file is refused");
            } catch (const std::invalid_argument&) {}
        }
        check(std::filesystem::file_size(kept) == 4 * kSlab, "reading leaves the file as it was");
        try {
            DirectScratchFile wrong(directory, 8 * kSlab, DirectScratchFile::Options{.kept_file = kept, .read_only = true});
            check(false, "a read-only file of another size is refused");
        } catch (const std::system_error&) {}
        check(std::filesystem::file_size(kept) == 4 * kSlab, "and is not resized");
        try {
            DirectScratchFile absent(directory, 4 * kSlab,
                                     DirectScratchFile::Options{.kept_file = directory / "absent.bin", .read_only = true});
            check(false, "a missing read-only file is refused");
        } catch (const std::system_error&) {}
        check(!std::filesystem::exists(directory / "absent.bin"), "and is not created");
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    }
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "direct scratch file: ok\n";
    return 0;
}
