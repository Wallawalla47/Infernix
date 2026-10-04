#include "models/qwen4_exp/program/ngram_volume.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp {
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
    : file_(path), table_(table), block_(2 * 4096) {
    const auto base = reinterpret_cast<std::uintptr_t>(block_.data());
    aligned_        = block_.data() + ((4096 - base % 4096) % 4096);
    if (file_.current_bytes() != table.file_bytes) {
        throw std::runtime_error(path.string() + ": n-gram volume size differs from the artifact's table");
    }
    std::span<std::byte> header(aligned_, 4096);
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
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] >= table_.rows) { throw std::out_of_range("n-gram row id outside the table"); }
        const std::uint64_t block = rows[i] / table_.rows_per_block;
        const std::uint64_t slot  = rows[i] % table_.rows_per_block;
        std::span<std::byte> bounce(aligned_, table_.block_bytes);
        if (file_.read_direct(table_.header_bytes + block * table_.block_bytes, bounce) != table_.block_bytes) {
            throw std::runtime_error("n-gram volume: short block read");
        }
        std::memcpy(out.data() + i * table_.row_bytes, aligned_ + slot * table_.row_bytes, table_.row_bytes);
    }
}

std::filesystem::path default_ngram_volume(const std::filesystem::path& artifact) {
    return std::filesystem::path(artifact.string() + ".ngram");
}

} // namespace ninfer::models::qwen4_exp
