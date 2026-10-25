#include "models/qwen4_exp/program/ngram_volume.h"
#include "core/temporary_switches.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>

namespace infernix::models::qwen4_exp {
namespace {

constexpr char kMagic[8]          = {'N', 'I', 'N', 'F', 'E', 'R', 'N', 'G'};
constexpr std::uint32_t kVersion  = 1;
constexpr std::size_t kHeaderSize = 8 + 4 + 4 + 8 + 4 + 4 + 4 + 8 + 16;
// An FP8 table's per-tensor scale (BF16 bits) after the fixed fields; 0 when not recorded.
constexpr std::size_t kScaleOffset = kHeaderSize;
// The row format (1 FP8, 2 IQ4_NL, 3 BF16, 4 NVFP4; 0 in volumes written before 2026-10-19, whose
// row size tells their format) and an NVFP4 table's FP32 multiplier (bits).
constexpr std::size_t kFormatOffset      = kScaleOffset + 2;
constexpr std::size_t kNvfp4ScaleOffset  = kFormatOffset + 2;
constexpr std::size_t kHeaderFieldsBytes = kNvfp4ScaleOffset + 4;
static_assert(kHeaderFieldsBytes <= 4096);

template <class T>
T read_le(const std::byte* p) {
    T value{};
    std::memcpy(&value, p, sizeof(T));
    return value;
}

// A recorded scale is a positive, finite, normal BF16 value.
bool valid_fp8_scale(std::uint16_t bits) noexcept {
    const unsigned exponent = (bits >> 7U) & 0xFFU;
    return (bits & 0x8000U) == 0 && exponent != 0 && exponent != 0xFFU;
}

// A recorded NVFP4 multiplier is a positive, finite, normal FP32 value.
bool valid_nvfp4_scale(std::uint32_t bits) noexcept {
    const unsigned exponent = (bits >> 23U) & 0xFFU;
    return (bits & 0x80000000U) == 0 && exponent != 0 && exponent != 0xFFU;
}

// FP8 volumes published before volumes recorded their scale, by volume id (16 random bytes the
// converter drew, so an id names one file): their scale is the artifacts' ngram_scale. Published
// files stay unchanged.
struct KnownFp8Volume {
    std::array<std::uint8_t, 16> id;
    std::uint16_t scale;
};
constexpr KnownFp8Volume kKnownFp8Volumes[] = {
    // NVIDIA's table (Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram, 52,429,058,048 bytes, in the
    // NVIDIA NVFP4 and Dense8 repositories): 0x3951 = 1.9931793e-4.
    {{0x59, 0xE5, 0xD7, 0x7E, 0x80, 0x2A, 0xC9, 0x5D, 0x8F, 0xAF, 0x0A, 0x80, 0x0B, 0x8F, 0x23, 0x01}, 0x3951},
};

} // namespace

NgramVolume::NgramVolume(const std::filesystem::path& path, const NgramTableConfig& table, bool resident)
    : file_(path, FileMapping::None), table_(table) {
    parallel_gather_ = temporary_switches::flag("INFERNIX_TMP_NGRAM_PARALLEL_GATHER");
    scatter_         = temporary_switches::flag("INFERNIX_TMP_NGRAM_SCATTER");
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
        }
        const auto gather = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i) {
                const std::uint64_t at = static_cast<std::uint64_t>(rows[i] / table_.rows_per_block) * table_.block_bytes +
                                         static_cast<std::uint64_t>(rows[i] % table_.rows_per_block) * row_bytes;
                std::memcpy(out.data() + i * row_bytes, blocks + at, row_bytes);
            }
        };
        if (parallel_gather_ && rows.size() >= kParallelGatherRows) {
            // Contiguous index ranges per thread: every row lands where the serial gather puts it.
            std::vector<std::thread> threads;
            threads.reserve(kReadThreads - 1);
            for (std::size_t t = 1; t < kReadThreads; ++t) {
                threads.emplace_back(gather, rows.size() * t / kReadThreads, rows.size() * (t + 1) / kReadThreads);
            }
            gather(0, rows.size() / kReadThreads);
            for (auto& thread : threads) { thread.join(); }
        } else {
            gather(0, rows.size());
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
        // The reading order: sorted, or (INFERNIX_TMP_NGRAM_SCATTER) position j reads sorted block
        // j * stride mod blocks, with the stride near blocks / golden ratio and coprime to blocks, so each
        // thread's range spreads over the whole table.
        const std::uint64_t* order = offsets_.data();
        const auto sorted_block    = [&](std::size_t j) { return scatter_ ? scattered_blocks_[j] : j; };
        if (scatter_) {
            std::size_t stride = std::max<std::size_t>(1, static_cast<std::size_t>(static_cast<double>(blocks) * 0.6180339887));
            while (std::gcd(stride, blocks) != 1) { ++stride; }
            scattered_offsets_.resize(blocks);
            scattered_blocks_.resize(blocks);
            for (std::size_t j = 0; j < blocks; ++j) {
                const std::size_t b      = static_cast<std::size_t>((static_cast<std::uint64_t>(j) * stride) % blocks);
                scattered_blocks_[j]     = b;
                scattered_offsets_[j]    = offsets_[b];
            }
            order = scattered_offsets_.data();
        }
        std::vector<std::exception_ptr> errors(kReadThreads);
        std::vector<std::thread> threads;
        threads.reserve(kReadThreads);
        for (std::size_t t = 0; t < kReadThreads; ++t) {
            const std::size_t lo = blocks * t / kReadThreads, hi = blocks * (t + 1) / kReadThreads;
            threads.emplace_back([&, t, lo, hi] {
                try {
                    file_.read_direct_blocks(
                        std::span<const std::uint64_t>(order + lo, hi - lo), table_.block_bytes,
                        std::span<std::byte>(parallel_ring_ + t * ring_bytes, ring_bytes),
                        [&](std::size_t b, std::span<const std::byte> block) {
                            const std::size_t sorted = sorted_block(lo + b);
                            for (std::size_t m = first_miss_[sorted]; m < first_miss_[sorted + 1]; ++m) {
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
    std::array<std::byte, kHeaderFieldsBytes> header{};
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
    // The row format: the header's tag, or for a volume written before the tag, the row size: head_width
    // FP8 codes, IQ4_NL blocks of 32 (18 bytes each) or head_width BF16 values. NVFP4 rows (head_width / 2
    // code bytes and head_width / 16 scales) have IQ4_NL's size, so they always carry the tag.
    const std::uint32_t width = ple.head_width();
    const auto row_bytes_of = [&](NgramRowFormat format) -> std::uint32_t {
        switch (format) {
        case NgramRowFormat::Fp8E4m3: return width;
        case NgramRowFormat::Iq4Nl: return width % 32 == 0 ? width / 32 * 18 : 0;
        case NgramRowFormat::Bf16: return 2 * width;
        case NgramRowFormat::Nvfp4: return width % 16 == 0 ? width / 2 + width / 16 : 0;
        }
        return 0;
    };
    const std::uint16_t tag = read_le<std::uint16_t>(p + kFormatOffset);
    if (tag != 0) {
        if (tag > 4) { fail("unknown n-gram row format tag " + std::to_string(tag)); }
        constexpr NgramRowFormat kTagged[] = {NgramRowFormat::Fp8E4m3, NgramRowFormat::Iq4Nl, NgramRowFormat::Bf16,
                                              NgramRowFormat::Nvfp4};
        out.format = kTagged[tag - 1];
        if (out.row_bytes != row_bytes_of(out.format)) {
            fail("n-gram rows of " + std::to_string(out.row_bytes) + " bytes differ from its format's");
        }
    } else if (out.row_bytes == row_bytes_of(NgramRowFormat::Bf16)) {
        out.format = NgramRowFormat::Bf16;
    } else if (out.row_bytes == row_bytes_of(NgramRowFormat::Fp8E4m3)) {
        out.format = NgramRowFormat::Fp8E4m3;
    } else if (out.row_bytes == row_bytes_of(NgramRowFormat::Iq4Nl)) {
        out.format = NgramRowFormat::Iq4Nl;
    } else {
        fail("n-gram rows of " + std::to_string(out.row_bytes) + " bytes fit no table of this model");
    }
    // Its own volume (the artifact's format and volume id): the artifact's table and, for FP8 and
    // NVFP4, its scale.
    if (out.format == ple.table.format && out.volume_id == ple.table.volume_id) { return ple.table; }
    // Another table of this model, in any of the four formats: its own geometry and id, and an FP8
    // table's scale or an NVFP4 table's multiplier from its header (a quantization's scale is a
    // property of the volume, not of the artifact).
    out.file_bytes = static_cast<std::uint64_t>(out.header_bytes) + out.blocks * out.block_bytes;
    if (out.rows != ple.table.rows || out.block_bytes != 4096 || out.header_bytes != 4096 ||
        out.rows_per_block != out.block_bytes / out.row_bytes ||
        out.blocks != (out.rows + out.rows_per_block - 1) / out.rows_per_block) {
        fail("the n-gram volume's geometry differs from this model's table");
    }
    if (out.format == NgramRowFormat::Fp8E4m3) {
        // Its header's scale, or a published volume's known one.
        std::uint16_t scale = read_le<std::uint16_t>(p + kScaleOffset);
        for (const KnownFp8Volume& known : kKnownFp8Volumes) {
            if (scale == 0 && known.id == out.volume_id) { scale = known.scale; }
        }
        if (!valid_fp8_scale(scale)) {
            fail("this FP8 n-gram volume records no scale, so it can only serve the artifact it was written for; "
                 "use the BF16 or IQ4_NL volume, or NVIDIA's published FP8 volume");
        }
        out.fp8_scale = scale;
    }
    if (out.format == NgramRowFormat::Nvfp4) {
        const std::uint32_t scale = read_le<std::uint32_t>(p + kNvfp4ScaleOffset);
        if (!valid_nvfp4_scale(scale)) { fail("this NVFP4 n-gram volume records no valid multiplier"); }
        out.nvfp4_scale = scale;
    }
    return out;
}

} // namespace infernix::models::qwen4_exp
