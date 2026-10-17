#include "models/qwen4_exp/program/ngram_volume.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace infernix::models::qwen4_exp {
namespace {

constexpr char kMagic[8]          = {'N', 'I', 'N', 'F', 'E', 'R', 'N', 'G'};
constexpr std::uint32_t kVersion  = 1;
constexpr std::size_t kHeaderSize = 8 + 4 + 4 + 8 + 4 + 4 + 4 + 8 + 16;

template <class T>
T read_le(const std::byte* p) {
    T value{};
    std::memcpy(&value, p, sizeof(T));
    return value;
}

} // namespace

NgramVolume::NgramVolume(const std::filesystem::path& path, const NgramTableConfig& table, bool resident)
    : file_(path, FileMapping::None), table_(table) {
    if (!resident) {
        tags_.assign(std::size_t{1} << kCacheBits, 0xFFFFFFFFU);
        cache_.assign((std::size_t{1} << kCacheBits) * table.row_bytes, std::byte{0});
    }
    if (table.block_bytes < 4096 || table.block_bytes % 4096 != 0 || table.header_bytes % 4096 != 0) {
        throw std::runtime_error(path.string() + ": n-gram blocks and header must be whole 4 KiB pages");
    }
    ring_storage_.assign(kInFlight * table.block_bytes + 4096, std::byte{0});
    const auto base = reinterpret_cast<std::uintptr_t>(ring_storage_.data());
    ring_           = ring_storage_.data() + ((4096 - base % 4096) % 4096);
    if (file_.current_bytes() != table.file_bytes) {
        throw std::runtime_error(path.string() + ": n-gram volume size differs from the artifact's table");
    }
    std::span<std::byte> header(ring_, 4096);
    if (file_.read_direct(0, header) != 4096) {
        throw std::runtime_error(path.string() + ": short n-gram volume header");
    }
    const std::byte* p = header.data();
    if (std::memcmp(p, kMagic, 8) != 0 || read_le<std::uint32_t>(p + 8) != kVersion ||
        read_le<std::uint32_t>(p + 12) != table.header_bytes || read_le<std::uint64_t>(p + 16) != table.rows ||
        read_le<std::uint32_t>(p + 24) != table.row_bytes ||
        read_le<std::uint32_t>(p + 28) != table.rows_per_block ||
        read_le<std::uint32_t>(p + 32) != table.block_bytes || read_le<std::uint64_t>(p + 36) != table.blocks ||
        std::memcmp(p + 44, table.volume_id.data(), 16) != 0) {
        throw std::runtime_error(path.string() + ": n-gram volume does not belong to this artifact");
    }
    static_assert(kHeaderSize == 60);
    if (resident) {
        // The whole table in RAM: 1 GiB unbuffered reads (each keeps 16 reads of 8 MiB in flight on
        // Windows) straight into the block, which is then page-locked and mapped in one call.
        const auto start = std::chrono::steady_clock::now();
        resident_ = std::make_unique<RegisteredHostBuffer>(resident_bytes(table));
        constexpr std::size_t kPiece = std::size_t{1} << 30;
        for (std::size_t at = 0; at < resident_->size(); at += kPiece) {
            const std::size_t bytes = std::min(kPiece, resident_->size() - at);
            if (file_.read_direct(table.header_bytes + at, std::span<std::byte>(resident_->data() + at, bytes)) != bytes) {
                throw std::runtime_error(path.string() + ": short n-gram volume read");
            }
        }
        resident_->pin();
        counters_.reads   = table.blocks;
        counters_.read_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    }
}

void NgramVolume::read_rows(std::span<const std::uint32_t> rows, std::span<std::byte> out) const {
    if (out.size() != rows.size() * table_.row_bytes) {
        throw std::invalid_argument("n-gram rows: output size differs from the row count");
    }
    const std::size_t row_bytes = table_.row_bytes;
    if (resident_) {
        const std::byte* blocks = resident_->data();
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] >= table_.rows) { throw std::out_of_range("n-gram row id outside the table"); }
            const std::uint64_t at = static_cast<std::uint64_t>(rows[i] / table_.rows_per_block) * table_.block_bytes +
                                     static_cast<std::uint64_t>(rows[i] % table_.rows_per_block) * row_bytes;
            std::memcpy(out.data() + i * row_bytes, blocks + at, row_bytes);
        }
        counters_.rows += rows.size();
        counters_.hits += rows.size();
        return;
    }
    misses_.clear();
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] >= table_.rows) { throw std::out_of_range("n-gram row id outside the table"); }
        const std::size_t slot = slot_of(rows[i]);
        if (tags_[slot] == rows[i]) {
            std::memcpy(out.data() + i * row_bytes, cache_.data() + slot * row_bytes, row_bytes);
            ++counters_.hits;
        } else {
            misses_.push_back((static_cast<std::uint64_t>(rows[i] / table_.rows_per_block) << 32U) | i);
        }
    }
    counters_.rows += rows.size();
    if (misses_.empty()) { return; }
    const auto start = std::chrono::steady_clock::now();
    // One read per distinct block: a block serves every missing row it holds, however often each
    // appears in the call.
    std::sort(misses_.begin(), misses_.end());
    offsets_.clear();
    first_miss_.clear();
    for (std::size_t m = 0; m < misses_.size(); ++m) {
        const std::uint64_t block = misses_[m] >> 32U;
        if (m == 0 || block != (misses_[m - 1] >> 32U)) {
            offsets_.push_back(table_.header_bytes + block * table_.block_bytes);
            first_miss_.push_back(m);
        }
    }
    first_miss_.push_back(misses_.size());
    const std::size_t blocks = offsets_.size();
    if (blocks >= kParallelBlocks) {
        // Each thread reads a contiguous range of the blocks into its own ring and copies their rows
        // to `out` (distinct indices); the cache is filled afterwards on this thread, from `out`.
        const std::size_t ring_bytes = kParallelInFlight * table_.block_bytes;
        if (parallel_ring_ == nullptr) {
            parallel_storage_.assign(kReadThreads * ring_bytes + 4096, std::byte{0});
            const auto base = reinterpret_cast<std::uintptr_t>(parallel_storage_.data());
            parallel_ring_  = parallel_storage_.data() + ((4096 - base % 4096) % 4096);
        }
        std::vector<std::exception_ptr> errors(kReadThreads);
        std::vector<std::thread> threads;
        threads.reserve(kReadThreads);
        for (std::size_t t = 0; t < kReadThreads; ++t) {
            const std::size_t lo = blocks * t / kReadThreads, hi = blocks * (t + 1) / kReadThreads;
            threads.emplace_back([&, t, lo, hi] {
                try {
                    file_.read_direct_blocks(
                        std::span<const std::uint64_t>(offsets_.data() + lo, hi - lo), table_.block_bytes,
                        std::span<std::byte>(parallel_ring_ + t * ring_bytes, ring_bytes),
                        [&](std::size_t b, std::span<const std::byte> block) {
                            for (std::size_t m = first_miss_[lo + b]; m < first_miss_[lo + b + 1]; ++m) {
                                const auto i = static_cast<std::size_t>(misses_[m] & 0xFFFFFFFFULL);
                                std::memcpy(out.data() + i * row_bytes,
                                            block.data() + (rows[i] % table_.rows_per_block) * row_bytes, row_bytes);
                            }
                        });
                } catch (...) { errors[t] = std::current_exception(); }
            });
        }
        for (auto& thread : threads) { thread.join(); }
        for (const auto& error : errors) {
            if (error) { std::rethrow_exception(error); }
        }
        for (const std::uint64_t miss : misses_) {
            const auto i           = static_cast<std::size_t>(miss & 0xFFFFFFFFULL);
            const std::size_t slot = slot_of(rows[i]);
            tags_[slot]            = rows[i];
            std::memcpy(cache_.data() + slot * row_bytes, out.data() + i * row_bytes, row_bytes);
        }
        counters_.reads += blocks;
        counters_.read_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        return;
    }
    file_.read_direct_blocks(
        offsets_, table_.block_bytes, std::span<std::byte>(ring_, kInFlight * table_.block_bytes),
        [&](std::size_t b, std::span<const std::byte> block) {
            for (std::size_t m = first_miss_[b]; m < first_miss_[b + 1]; ++m) {
                const auto i         = static_cast<std::size_t>(misses_[m] & 0xFFFFFFFFULL);
                const std::byte* row = block.data() + (rows[i] % table_.rows_per_block) * row_bytes;
                std::memcpy(out.data() + i * row_bytes, row, row_bytes);
                const std::size_t slot = slot_of(rows[i]);
                tags_[slot]            = rows[i];
                std::memcpy(cache_.data() + slot * row_bytes, row, row_bytes);
            }
        });
    counters_.reads += offsets_.size();
    counters_.read_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

bool NgramVolume::cached(std::span<const std::uint32_t> rows) const noexcept {
    if (resident_) {
        for (const std::uint32_t row : rows) {
            if (row >= table_.rows) { return false; }
        }
        return true;
    }
    for (const std::uint32_t row : rows) {
        if (row >= table_.rows || tags_[slot_of(row)] != row) { return false; }
    }
    return true;
}

std::filesystem::path default_ngram_volume(const std::filesystem::path& artifact) {
    return std::filesystem::path(artifact.string() + ".ngram");
}

NgramTableConfig resolve_ngram_volume(const std::filesystem::path& path, const PleConfig& ple) {
    const auto fail = [&](const std::string& why) { throw std::runtime_error(path.string() + ": " + why); };
    std::array<std::byte, kHeaderSize> header{};
    {
        std::ifstream file(path, std::ios::binary);
        if (!file) { fail("cannot open the n-gram volume"); }
        if (!file.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()))) {
            fail("short n-gram volume header");
        }
    }
    const std::byte* p = header.data();
    if (std::memcmp(p, kMagic, 8) != 0 || read_le<std::uint32_t>(p + 8) != kVersion) {
        fail("not an n-gram volume (v1)");
    }
    NgramTableConfig out;
    out.header_bytes   = read_le<std::uint32_t>(p + 12);
    out.rows           = read_le<std::uint64_t>(p + 16);
    out.row_bytes      = read_le<std::uint32_t>(p + 24);
    out.rows_per_block = read_le<std::uint32_t>(p + 28);
    out.block_bytes    = read_le<std::uint32_t>(p + 32);
    out.blocks         = read_le<std::uint64_t>(p + 36);
    std::memcpy(out.volume_id.data(), p + 44, 16);
    // The row format follows from the row size: head_width FP8 codes, IQ4_NL blocks of 32 (18 bytes
    // each) or head_width BF16 values.
    const std::uint32_t width = ple.head_width();
    if (out.row_bytes == 2 * width) {
        out.format = NgramRowFormat::Bf16;
    } else if (out.row_bytes == width) {
        out.format = NgramRowFormat::Fp8E4m3;
    } else if (width % 32 == 0 && out.row_bytes == width / 32 * 18) {
        out.format = NgramRowFormat::Iq4Nl;
    } else {
        fail("n-gram rows of " + std::to_string(out.row_bytes) + " bytes fit no table of this model");
    }
    if (out.format == ple.table.format) { return ple.table; } // its own volume: NgramVolume checks the id
    if (ple.table.format == NgramRowFormat::Bf16) {
        fail("this artifact's n-gram table is BF16; a quantized volume cannot replace it (use its BF16 volume)");
    }
    if (out.format != NgramRowFormat::Bf16) {
        fail("an n-gram volume of another quantized format than the artifact's cannot replace its table; "
             "use the artifact's volume or a BF16 volume (tools/flash_next/ngram_bf16.py)");
    }
    out.file_bytes = static_cast<std::uint64_t>(out.header_bytes) + out.blocks * out.block_bytes;
    if (out.rows != ple.table.rows || out.block_bytes != 4096 || out.header_bytes != 4096 ||
        out.rows_per_block != out.block_bytes / out.row_bytes ||
        out.blocks != (out.rows + out.rows_per_block - 1) / out.rows_per_block) {
        fail("the BF16 n-gram volume's geometry differs from this model's table");
    }
    return out;
}

} // namespace infernix::models::qwen4_exp
