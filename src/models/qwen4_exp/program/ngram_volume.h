#pragma once

// The PLE n-gram table volume written by the converter (docs/maintainer/qwen3_8-flash-next-design.md
// §12.2): a 4 KiB header, then rows of `row_bytes` FP8 codes, `rows_per_block` per 4 KiB block,
// never straddling a block. Rows are read with unbuffered I/O so the 52 GB table never fills the
// page cache. A direct-mapped host cache keeps recently read rows; one call's missing rows are read
// once per distinct block, up to 64 blocks in flight through a fixed ring, so their NVMe latency
// overlaps and the memory stays bounded.

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

    // Row traffic since the volume was opened: rows requested, rows served by the host cache, 4 KiB
    // blocks read from the volume, and the wall time of the calls' volume reads.
    struct Counters {
        std::uint64_t rows = 0, hits = 0, reads = 0, read_ns = 0;
    };
    [[nodiscard]] const Counters& counters() const noexcept { return counters_; }

    // Bytes of the host row cache a volume of this geometry allocates.
    [[nodiscard]] static std::uint64_t cache_bytes(const NgramTableConfig& table) noexcept {
        return (std::uint64_t{1} << kCacheBits) * table.row_bytes;
    }

private:
    static constexpr unsigned kCacheBits  = 20; // 2^20 rows, ~170 MB with 160-byte rows
    static constexpr std::size_t kInFlight = 64; // blocks read at once

    ReadOnlyFile file_;
    NgramTableConfig table_;
    mutable std::vector<std::byte> ring_storage_; // kInFlight blocks, 4 KiB-aligned at ring_
    mutable std::byte* ring_ = nullptr;
    mutable std::vector<std::uint32_t> tags_; // row id per cache slot, UINT32_MAX when empty
    mutable std::vector<std::byte> cache_;    // row codes per slot
    // One call's misses as (block << 32 | output index), sorted so each block's rows are adjacent;
    // the distinct blocks' offsets and where each block's misses begin.
    mutable std::vector<std::uint64_t> misses_;
    mutable std::vector<std::uint64_t> offsets_;
    mutable std::vector<std::size_t> first_miss_;
    mutable Counters counters_;
};

// The volume beside the artifact: `<artifact>.ngram`.
[[nodiscard]] std::filesystem::path default_ngram_volume(const std::filesystem::path& artifact);

} // namespace ninfer::models::qwen4_exp
