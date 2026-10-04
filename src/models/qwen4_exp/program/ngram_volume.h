#pragma once

// The PLE n-gram table volume written by the converter (docs/maintainer/qwen3_8-flash-next-design.md
// §12.2): a 4 KiB header, then rows of `row_bytes` FP8 codes, `rows_per_block` per 4 KiB block,
// never straddling a block. Rows are read with unbuffered I/O, one block per row, so the 52 GB
// table never fills the page cache.

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

private:
    ReadOnlyFile file_;
    NgramTableConfig table_;
    mutable std::vector<std::byte> block_; // 4 KiB-aligned bounce block
    std::byte* aligned_ = nullptr;
};

// The volume beside the artifact: `<artifact>.ngram`.
[[nodiscard]] std::filesystem::path default_ngram_volume(const std::filesystem::path& artifact);

} // namespace ninfer::models::qwen4_exp
