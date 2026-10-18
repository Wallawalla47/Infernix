#pragma once

// The PLE n-gram table volume written by the converter (docs/maintainer/qwen3_8-flash-next-design.md
// §12.2): a 4 KiB header, then rows of `row_bytes` FP8 codes, `rows_per_block` per 4 KiB block,
// never straddling a block. Rows are read with unbuffered I/O so the 52 GB table never fills the
// page cache. A direct-mapped host cache keeps recently read rows; one call's missing rows are read
// once per distinct block, up to 64 blocks in flight through a fixed ring, so their NVMe latency
// overlaps and the memory stays bounded. A call with many distinct blocks (a prefill chunk: tens of
// thousands) splits them over kReadThreads threads with a kParallelInFlight-block ring each: one
// thread submitting overlapped reads reaches ~220K reads/s on an Optane P5800X, eight threads at
// depth 16 ~880K (fn/probes/ioring_bench.cpp; the volume must not be mapped, design §19.3.8 F9).

#include "core/host_memory.h"
#include "core/read_only_file.h"
#include "models/qwen4_exp/config.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp {

class NgramVolume {
public:
    // Validates the header against the artifact's table geometry and volume id. With `resident`,
    // the whole table is read into pinned host memory here (large unbuffered reads, then one
    // registration) and every read_rows call is served from RAM; the row cache is not allocated.
    NgramVolume(const std::filesystem::path& path, const NgramTableConfig& table, bool resident = false);

    // Host bytes a resident table pins: its blocks (the 4 KiB padding included).
    [[nodiscard]] static std::uint64_t resident_bytes(const NgramTableConfig& table) noexcept {
        return table.blocks * table.block_bytes;
    }
    [[nodiscard]] bool resident() const noexcept { return resident_ != nullptr; }

    [[nodiscard]] const NgramTableConfig& table() const noexcept { return table_; }

    // Copies each row's codes to out[i * row_bytes, (i + 1) * row_bytes).
    void read_rows(std::span<const std::uint32_t> rows, std::span<std::byte> out) const;

    // Whether every row is in the host cache, so read_rows would read nothing from the volume.
    [[nodiscard]] bool cached(std::span<const std::uint32_t> rows) const noexcept;

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
    static constexpr std::size_t kInFlight         = 64;  // blocks read at once (one thread)
    static constexpr std::size_t kReadThreads      = 8;   // threads of a large call
    static constexpr std::size_t kParallelInFlight = 16;  // blocks in flight per thread of a large call
    static constexpr std::size_t kParallelBlocks = 512; // distinct blocks that make a call large

    // The direct-mapped cache slot of a row.
    static std::size_t slot_of(std::uint32_t row) noexcept {
        return static_cast<std::size_t>((row * 2654435761U) >> (32U - kCacheBits));
    }

    ReadOnlyFile file_;
    NgramTableConfig table_;
    std::unique_ptr<RegisteredHostBuffer> resident_; // the whole table's blocks (RAM mode)
    mutable std::vector<std::byte> ring_storage_; // kInFlight blocks, 4 KiB-aligned at ring_
    mutable std::byte* ring_ = nullptr;
    mutable std::vector<std::byte> parallel_storage_; // kReadThreads rings of kParallelInFlight blocks
    mutable std::byte* parallel_ring_ = nullptr;
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

// The table a volume holds for this model, from its header: the artifact's own table when the volume
// is its own (its format and volume id), else the volume's table, in any of the three formats (the
// format follows from the row size), which any Qwen3.8-Flash-Next artifact of this geometry accepts
// in place of its own. Such an FP8 volume needs its per-tensor scale (NgramTableConfig::fp8_scale):
// the BF16 word after the header's fixed fields, which volumes written since record, or for a volume
// published before, the scale known for its volume id. Throws std::runtime_error for another
// geometry, an unknown row size or an FP8 volume whose scale is unknown.
[[nodiscard]] NgramTableConfig resolve_ngram_volume(const std::filesystem::path& path, const PleConfig& ple);

} // namespace infernix::models::qwen4_exp
