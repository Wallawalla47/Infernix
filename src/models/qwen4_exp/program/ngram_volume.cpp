#include "models/qwen4_exp/program/ngram_volume.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
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

NgramVolume::NgramVolume(const std::filesystem::path& path, const NgramTableConfig& table)
    : file_(path, FileMapping::None), table_(table), tags_(std::size_t{1} << kCacheBits, 0xFFFFFFFFU),
      cache_((std::size_t{1} << kCacheBits) * table.row_bytes) {
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
}

void NgramVolume::read_rows(std::span<const std::uint32_t> rows, std::span<std::byte> out) const {
    if (out.size() != rows.size() * table_.row_bytes) {
        throw std::invalid_argument("n-gram rows: output size differs from the row count");
    }
    const std::size_t row_bytes = table_.row_bytes;
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
    for (const std::uint32_t row : rows) {
        if (row >= table_.rows || tags_[slot_of(row)] != row) { return false; }
    }
    return true;
}

std::filesystem::path default_ngram_volume(const std::filesystem::path& artifact) {
    return std::filesystem::path(artifact.string() + ".ngram");
}

} // namespace infernix::models::qwen4_exp
