#pragma once

// The PLE n-gram table volume written by the converter (docs/maintainer/qwen3_8-flash-next-design.md
// §12.2): a 4 KiB header, then rows of `row_bytes` FP8 codes, `rows_per_block` per 4 KiB block,
// never straddling a block. Rows are read with unbuffered I/O, one block per row, so the 52 GB
// table never fills the page cache. A direct-mapped host cache keeps recently read rows, and the
// blocks of one call's missing rows are read together so their NVMe latency overlaps.

#include "core/read_only_file.h"
#include "models/qwen4_exp/config.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

class NgramVolume {
public:
    // Validates the header against the artifact's table geometry and volume id.
    NgramVolume(const std::filesystem::path& path, const NgramTableConfig& table);

    [[nodiscard]] const NgramTableConfig& table() const noexcept { return table_; }

    // Copies each row's codes to out[i * row_bytes, (i + 1) * row_bytes).
    void read_rows(std::span<const std::uint32_t> rows, std::span<std::byte> out) const;

    [[nodiscard]] std::uint64_t cache_hits() const noexcept { return hits_; }
    [[nodiscard]] std::uint64_t cache_misses() const noexcept { return misses_; }

private:
    static constexpr unsigned kCacheBits = 20; // 2^20 rows, ~170 MB with 160-byte rows

    ReadOnlyFile file_;
    NgramTableConfig table_;
    mutable std::vector<std::byte> block_; // 4 KiB-aligned bounce blocks for one call's misses
    mutable std::byte* aligned_ = nullptr;
    mutable std::vector<std::uint32_t> tags_; // row id per cache slot, UINT32_MAX when empty
    mutable std::vector<std::byte> cache_;    // row codes per slot
    mutable std::vector<std::size_t> missing_;
    mutable std::vector<ReadOnlyFile::DirectRead> reads_;
    mutable std::uint64_t hits_ = 0, misses_ = 0;
};

// The volume beside the artifact: `<artifact>.ngram`.
[[nodiscard]] std::filesystem::path default_ngram_volume(const std::filesystem::path& artifact);

} // namespace ninfer::models::qwen4_exp
