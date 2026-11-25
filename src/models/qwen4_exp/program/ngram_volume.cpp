#include "models/qwen4_exp/program/ngram_volume.h"

#include "infernix/ngram_blockdir.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace infernix::models::qwen4_exp {
namespace {

constexpr char kMagic[8]          = {'N', 'I', 'N', 'F', 'E', 'R', 'N', 'G'};
constexpr std::uint32_t kVersion  = 1;
constexpr std::size_t kHeaderSize = 8 + 4 + 4 + 8 + 4 + 4 + 4 + 8 + 16;
// An FP8 table's per-tensor scale (BF16 bits) after the fixed fields; 0 when not recorded.
constexpr std::size_t kScaleOffset = kHeaderSize;
// The row format (1 FP8, 2 IQ4_NL, 3 BF16, 4 NVFP4, 5 Q8_0, 6 BF16 P12; 0 in volumes written before
// 2026-10-19, whose row size tells their format) and an NVFP4 table's FP32 multiplier (bits).
constexpr std::size_t kFormatOffset      = kScaleOffset + 2;
constexpr std::size_t kNvfp4ScaleOffset  = kFormatOffset + 2;
// The P12 blockdir extension after the fixed fields: W, the on-disk row size, the side stream's
// offset, length and record count, and the file's eight-entry base table in the eight bytes after
// those fields.
constexpr std::size_t kWidthOffset         = kNvfp4ScaleOffset + 4; // 68
constexpr std::size_t kSlotBytesOffset     = kWidthOffset + 2;      // 70
constexpr std::size_t kCodeLengthsOffset   = kSlotBytesOffset + 6;  // 76
constexpr std::size_t kSideOffsetOffset    = kCodeLengthsOffset + 16; // 92
constexpr std::size_t kSideBytesOffset     = kSideOffsetOffset + 8;   // 100
constexpr std::size_t kSideCountOffset     = kSideBytesOffset + 8;    // 108
constexpr std::size_t kBasesOffset         = kSideCountOffset + 8;    // 116
constexpr std::size_t kHeaderFieldsBytes   = kBasesOffset + 8;        // 124
static_assert(kHeaderFieldsBytes <= 4096);
// BF16 P12 blockdir: W values a row, 17 rows and one 16-byte directory a 4 KiB block, decoded into
// canonical P12 rows (W + W/2 + 16 bytes). The directory holds a 3-bit base code per row (an index
// into the file's eight-entry base table), a 3-bit escape count and four 18-bit escape slots; a
// block that needs a fifth slot puts all of its escapes in the side stream (count 7).
constexpr std::uint32_t kBlockdirWidth       = 160;
constexpr std::uint32_t kBlockdirRows        = 17;
constexpr std::uint32_t kBlockdirPlaneBytes  = kBlockdirWidth + kBlockdirWidth / 2;
constexpr std::uint32_t kBlockdirRowBytes    = kBlockdirPlaneBytes + 16;
constexpr unsigned kBlockdirMaxRowEscapes    = 4;
constexpr unsigned kBlockdirCodeBits         = 3;
constexpr unsigned kBlockdirBlockSlots       = 4;   // 17 x 3 + 3 + 4 x 18 = 126 of the 128 bits
constexpr unsigned kBlockdirBaseTableSize    = 8;   // bytes the header's base table occupies
static_assert(kBlockdirBaseTableSize == std::tuple_size_v<decltype(NgramTableConfig::blockdir_bases)>);
// The device kernel's reading of the same directory (include/infernix/ngram_blockdir.h) must agree.
static_assert(kBlockdirRows == ngram::kBlockdirRows && kBlockdirWidth == ngram::kBlockdirWidth &&
              kBlockdirPlaneBytes == ngram::kBlockdirPlaneBytes && kBlockdirCodeBits == ngram::kBlockdirCodeBits &&
              kBlockdirBlockSlots == ngram::kBlockdirSlots && kBlockdirBaseTableSize == ngram::kBlockdirBaseTableSize);

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
    if (table_.format == NgramRowFormat::Bf16P12Blockdir) {
        // The header's extension must equal the table resolve_ngram_volume read from it: W, the
        // on-disk row size, the base table and the side stream.
        if (read_le<std::uint16_t>(p + kWidthOffset) != table_.block_width ||
            read_le<std::uint16_t>(p + kSlotBytesOffset) != table_.slot_bytes ||
            std::memcmp(p + kBasesOffset, table_.blockdir_bases.data(), kBlockdirBaseTableSize) != 0 ||
            read_le<std::uint64_t>(p + kSideOffsetOffset) != table_.side_offset ||
            read_le<std::uint64_t>(p + kSideBytesOffset) != table_.side_bytes ||
            read_le<std::uint64_t>(p + kSideCountOffset) != table_.side_count) {
            throw std::runtime_error(path.string() + ": n-gram volume does not belong to this artifact");
        }
        if (table_.side_bytes != 0) {
            side_storage_.assign(table_.side_bytes + 4096, std::byte{0});
            const auto side_base = reinterpret_cast<std::uintptr_t>(side_storage_.data());
            side_                = side_storage_.data() + ((4096 - side_base % 4096) % 4096);
            if (file_.read_direct(table_.side_offset, std::span<std::byte>(side_, table_.side_bytes)) !=
                table_.side_bytes) {
                throw std::runtime_error(path.string() + ": short n-gram volume side stream");
            }
            std::uint64_t at = 0;
            for (std::uint64_t record = 0; record < table_.side_count; ++record) {
                if (at + 6 > table_.side_bytes) {
                    throw std::runtime_error(path.string() + ": truncated n-gram volume side stream");
                }
                const std::uint32_t block = read_le<std::uint32_t>(side_ + at);
                const std::uint32_t count = read_le<std::uint16_t>(side_ + at + 4);
                at += 6;
                if (at + 4ULL * count > table_.side_bytes) {
                    throw std::runtime_error(path.string() + ": truncated n-gram volume side stream");
                }
                if (!side_index_.emplace(block, SideRef{.offset = at, .count = count}).second) {
                    throw std::runtime_error(path.string() + ": a block appears twice in the n-gram side stream");
                }
                at += 4ULL * count;
            }
        }
    }
    if (resident) {
        // The whole table in RAM: a block locked in the working set (only the CPU reads it), then 1 GiB
        // unbuffered reads (each keeps 16 reads of 8 MiB in flight on Windows) straight into it.
        const auto start = std::chrono::steady_clock::now();
        resident_ = std::make_unique<ResidentHostBuffer>(resident_bytes(table));
        constexpr std::size_t kPiece = std::size_t{1} << 30;
        for (std::size_t at = 0; at < resident_->size(); at += kPiece) {
            const std::size_t bytes = std::min(kPiece, resident_->size() - at);
            if (file_.read_direct(table.header_bytes + at, std::span<std::byte>(resident_->data() + at, bytes)) != bytes) {
                throw std::runtime_error(path.string() + ": short n-gram volume read");
            }
        }
        counters_.reads   = table.blocks;
        counters_.read_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    }
}

// A blockdir block's row as the canonical P12 row the decoder reads: the 240 data bytes verbatim,
// then the base its 3-bit code selects and that row's escaped values as P12's 3-byte entries (the
// exponent from the directory's 6-bit signed delta, or the full BF16 word from the side stream).
void NgramVolume::decode_blockdir_row(const std::byte* block, std::uint32_t row, std::uint32_t row_in_block,
                                      std::byte* out) const {
    const std::byte* directory = block + kBlockdirRows * kBlockdirPlaneBytes;
    const auto bit = [&](unsigned at) {
        return (static_cast<unsigned>(directory[at / 8]) >> (at % 8)) & 1U;
    };
    const auto code_at = [&](unsigned at) {
        unsigned value = 0;
        for (unsigned k = 0; k < kBlockdirCodeBits; ++k) { value |= bit(at + k) << k; }
        return value;
    };
    const unsigned code  = code_at(kBlockdirCodeBits * row_in_block);
    const unsigned base  = table_.blockdir_bases[code];
    const unsigned count = bit(kBlockdirCodeBits * kBlockdirRows) |
                           (bit(kBlockdirCodeBits * kBlockdirRows + 1) << 1U) |
                           (bit(kBlockdirCodeBits * kBlockdirRows + 2) << 2U);
    std::memcpy(out, block + static_cast<std::size_t>(row_in_block) * kBlockdirPlaneBytes, kBlockdirPlaneBytes);
    std::uint8_t columns[kBlockdirMaxRowEscapes];
    std::uint16_t values[kBlockdirMaxRowEscapes];
    unsigned escapes = 0;
    const auto add = [&](std::uint32_t position, std::uint32_t value) {
        if (position / kBlockdirWidth != row_in_block) { return; }
        if (escapes == kBlockdirMaxRowEscapes) {
            throw std::runtime_error("a blockdir row holds more than four escapes");
        }
        columns[escapes] = static_cast<std::uint8_t>(position % kBlockdirWidth);
        values[escapes]  = static_cast<std::uint16_t>(value);
        ++escapes;
    };
    if (count == 7) {
        const auto it = side_index_.find(row / kBlockdirRows);
        if (it == side_index_.end()) {
            throw std::runtime_error("a blockdir block names the side stream without a record");
        }
        const std::byte* entries = side_ + it->second.offset;
        for (std::uint32_t e = 0; e < it->second.count; ++e) {
            add(read_le<std::uint16_t>(entries + 4ULL * e), read_le<std::uint16_t>(entries + 4ULL * e + 2));
        }
    } else if (count <= kBlockdirBlockSlots) {
        for (unsigned e = 0; e < count; ++e) {
            std::uint32_t slot = 0;
            const unsigned start = kBlockdirCodeBits * kBlockdirRows + 3 + 18 * e;
            for (unsigned k = 0; k < 18; ++k) { slot |= bit(start + k) << k; }
            const std::uint32_t position = slot & 0xFFFU;
            int delta = static_cast<int>(slot >> 12U);
            if ((delta & 32) != 0) { delta -= 64; }
            const int exponent = static_cast<int>(base) + delta;
            if (exponent < 0 || exponent > 255) {
                throw std::runtime_error("a blockdir escape's exponent is outside BF16");
            }
            const auto plane = static_cast<std::uint8_t>(out[position % kBlockdirWidth]);
            add(position, (static_cast<std::uint32_t>(plane & 0x80U) << 8U) |
                              (static_cast<std::uint32_t>(exponent) << 7U) | (plane & 0x7FU));
        }
    } else {
        throw std::runtime_error("a blockdir block records an invalid escape count");
    }
    out[kBlockdirPlaneBytes]       = static_cast<std::byte>(base);
    out[kBlockdirPlaneBytes + 1]   = static_cast<std::byte>(escapes);
    for (unsigned e = 0; e < escapes; ++e) {
        std::byte* entry = out + kBlockdirPlaneBytes + 2 + 3 * e;
        entry[0]         = static_cast<std::byte>(columns[e]);
        entry[1]         = static_cast<std::byte>(values[e] & 0xFFU);
        entry[2]         = static_cast<std::byte>(values[e] >> 8U);
    }
}

// A resident blockdir row as the packed-mode kernel reads it (include/infernix/ngram_blockdir.h):
// the 240 data planes verbatim, the block's 16-byte directory and the row's index in the block. A
// block whose escapes went to the side stream (count 7) or whose count is invalid is the canonical
// P12 row instead, marked at the header byte, so the kernel decodes it through the P12 trailer that
// decode_blockdir_row already produced.
void NgramVolume::stage_blockdir_row(const std::byte* block, std::uint32_t row, std::uint32_t row_in_block,
                                     std::byte* out) const {
    const std::byte* directory = block + kBlockdirRows * kBlockdirPlaneBytes;
    const auto bit = [&](unsigned at) {
        return (static_cast<unsigned>(directory[at / 8]) >> (at % 8)) & 1U;
    };
    const unsigned count = bit(kBlockdirCodeBits * kBlockdirRows) |
                           (bit(kBlockdirCodeBits * kBlockdirRows + 1) << 1U) |
                           (bit(kBlockdirCodeBits * kBlockdirRows + 2) << 2U);
    if (ngram::blockdir_row_is_canonical(count)) {
        decode_blockdir_row(block, row, row_in_block, out); // requests the side stream, or throws
        out[ngram::kBlockdirHeaderOffset] = std::byte{ngram::kBlockdirCanonicalMarker};
        std::memset(out + ngram::kBlockdirHeaderOffset + 1, 0,
                    ngram::kBlockdirStagingRowBytes - (ngram::kBlockdirHeaderOffset + 1));
        return;
    }
    std::memcpy(out, block + static_cast<std::size_t>(row_in_block) * kBlockdirPlaneBytes,
                kBlockdirPlaneBytes);
    std::memcpy(out + ngram::kBlockdirDirectoryOffset, directory, ngram::kBlockdirDirectoryBytes);
    out[ngram::kBlockdirHeaderOffset] = static_cast<std::byte>(row_in_block);
    std::memset(out + ngram::kBlockdirHeaderOffset + 1, 0,
                ngram::kBlockdirStagingRowBytes - (ngram::kBlockdirHeaderOffset + 1));
}

std::uint32_t NgramVolume::consumer_row_bytes(const NgramTableConfig& table, bool resident) noexcept {
    // A demand read resolves a blockdir row into the canonical P12 row at cache fill; a resident table
    // is consumed packed, as its 240 data bytes, the block directory and the row's index.
    if (table.format == NgramRowFormat::Bf16P12Blockdir) {
        return resident ? ngram::kBlockdirStagingRowBytes : kBlockdirRowBytes;
    }
    return table.row_bytes;
}
void NgramVolume::copy_row(const std::byte* block, std::uint32_t row, std::uint32_t row_in_block,
                           std::byte* out) const {
    if (table_.format == NgramRowFormat::Bf16P12Blockdir) {
        if (resident_) {
            stage_blockdir_row(block, row, row_in_block, out);
        } else {
            decode_blockdir_row(block, row, row_in_block, out);
        }
        return;
    }
    std::memcpy(out, block + static_cast<std::size_t>(row_in_block) * table_.row_bytes, table_.row_bytes);
}

void NgramVolume::read_rows(std::span<const std::uint32_t> rows, std::span<std::byte> out) const {
    const std::size_t row_bytes = consumer_row_bytes();
    if (out.size() != rows.size() * row_bytes) {
        throw std::invalid_argument("n-gram rows: output size differs from the row count");
    }
    if (resident_) {
        const std::byte* blocks = resident_->data();
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] >= table_.rows) { throw std::out_of_range("n-gram row id outside the table"); }
        }
        // One thread: gathering a prefill chunk's rows on kReadThreads threads (candidates §8.3 GX-08) measured
        // flat (prefill is compute-bound; removed 2026-10-28).
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const std::uint64_t block = static_cast<std::uint64_t>(rows[i] / table_.rows_per_block);
            copy_row(blocks + block * table_.block_bytes, rows[i], rows[i] % table_.rows_per_block,
                     out.data() + i * row_bytes);
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
        // Each thread copies its blocks' rows to `out` (distinct indices); the cache is filled
        // afterwards on this thread, from `out`.
        read_parallel(offsets_, [&](std::size_t b, std::span<const std::byte> block) {
            for (std::size_t m = first_miss_[b]; m < first_miss_[b + 1]; ++m) {
                const auto i = static_cast<std::size_t>(misses_[m] & 0xFFFFFFFFULL);
                copy_row(block.data(), rows[i], rows[i] % table_.rows_per_block, out.data() + i * row_bytes);
            }
        });
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
                const auto i = static_cast<std::size_t>(misses_[m] & 0xFFFFFFFFULL);
                std::byte* row = out.data() + i * row_bytes;
                copy_row(block.data(), rows[i], rows[i] % table_.rows_per_block, row);
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

std::byte* NgramVolume::parallel_rings() const {
    if (parallel_ring_ == nullptr) {
        parallel_storage_.assign(kReadThreads * kParallelInFlight * table_.block_bytes + 4096, std::byte{0});
        const auto base = reinterpret_cast<std::uintptr_t>(parallel_storage_.data());
        parallel_ring_  = parallel_storage_.data() + ((4096 - base % 4096) % 4096);
    }
    return parallel_ring_;
}

void NgramVolume::read_parallel(std::span<const std::uint64_t> offsets,
                                const std::function<void(std::size_t, std::span<const std::byte>)>& consume) const {
    const std::size_t ring_bytes = kParallelInFlight * table_.block_bytes;
    std::byte* const rings       = parallel_rings();
    const std::size_t blocks     = offsets.size();
    // In the given (sorted) order, each thread one contiguous range: a scattered order (candidates BX-04)
    // measured flat (removed 2026-10-28).
    std::vector<std::exception_ptr> errors(kReadThreads);
    std::vector<std::thread> threads;
    threads.reserve(kReadThreads);
    for (std::size_t t = 0; t < kReadThreads; ++t) {
        const std::size_t lo = blocks * t / kReadThreads, hi = blocks * (t + 1) / kReadThreads;
        if (lo == hi) { continue; }
        threads.emplace_back([&, t, lo, hi] {
            try {
                file_.read_direct_blocks(offsets.subspan(lo, hi - lo), table_.block_bytes,
                                         std::span<std::byte>(rings + t * ring_bytes, ring_bytes),
                                         [&](std::size_t b, std::span<const std::byte> block) { consume(lo + b, block); });
            } catch (...) { errors[t] = std::current_exception(); }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    for (const auto& error : errors) {
        if (error) { std::rethrow_exception(error); }
    }
}

NgramVolume::ReadProbe NgramVolume::probe_reads(std::size_t reads, double max_seconds) const {
    ReadProbe probe;
    if (resident_ || table_.blocks == 0) { return probe; }
    // A different set of blocks each start, so the drive's own cache cannot answer the same ones.
    std::uint64_t state = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 1U;
    std::vector<std::uint64_t> offsets;
    std::size_t round   = kReadThreads * kParallelInFlight;
    const auto start    = std::chrono::steady_clock::now();
    while (probe.reads < reads) {
        offsets.resize(std::min(round, reads - probe.reads));
        for (std::uint64_t& at : offsets) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            at    = table_.header_bytes + (state >> 33U) % table_.blocks * table_.block_bytes;
        }
        std::sort(offsets.begin(), offsets.end()); // as read_rows reads a call's blocks
        read_parallel(offsets, [](std::size_t, std::span<const std::byte>) {});
        probe.reads += offsets.size();
        probe.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (probe.seconds >= max_seconds) { break; }
        round *= 2;
    }
    return probe;
}

std::filesystem::path default_ngram_volume(const std::filesystem::path& artifact) {
    return std::filesystem::path(artifact.string() + ".ngram");
}

NgramTableConfig resolve_ngram_volume(const std::filesystem::path& path, const PleConfig& ple) {
    const auto fail = [&](const std::string& why) { throw std::runtime_error(path.string() + ": " + why); };
    // The header's first page, read unbuffered like every later read of the volume: once a process has
    // read a file through the cache, Windows completes its unbuffered reads about one at a time (~12K
    // reads/s; F9, Basalt #57).
    std::array<std::byte, kHeaderFieldsBytes> header{};
    {
        std::unique_ptr<ReadOnlyFile> file;
        try {
            file = std::make_unique<ReadOnlyFile>(path, FileMapping::None);
        } catch (const std::exception&) { fail("cannot open the n-gram volume"); }
        std::vector<std::byte> page(2 * 4096);
        const auto base = reinterpret_cast<std::uintptr_t>(page.data());
        const std::span<std::byte> aligned(page.data() + (4096 - base % 4096) % 4096, 4096);
        if (file->read_direct(0, aligned) < header.size()) { fail("short n-gram volume header"); }
        std::memcpy(header.data(), aligned.data(), header.size());
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
    // A corrupt header's empty rows would match a format this head width cannot hold (row_bytes_of's 0).
    if (out.row_bytes == 0) { fail("the n-gram volume header records rows of 0 bytes"); }
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
        case NgramRowFormat::Q8_0: return width % 32 == 0 ? width / 32 * 34 : 0;
        case NgramRowFormat::Bf16P12: return width % 2 == 0 && width <= 255 ? width + width / 2 + 16 : 0;
        case NgramRowFormat::Bf16P12Blockdir: return width == kBlockdirWidth ? kBlockdirRowBytes : 0;
        }
        return 0;
    };
    const std::uint16_t tag = read_le<std::uint16_t>(p + kFormatOffset);
    if (tag != 0) {
        if (tag > 7) { fail("unknown n-gram row format tag " + std::to_string(tag)); }
        constexpr NgramRowFormat kTagged[] = {NgramRowFormat::Fp8E4m3, NgramRowFormat::Iq4Nl, NgramRowFormat::Bf16,
                                              NgramRowFormat::Nvfp4, NgramRowFormat::Q8_0, NgramRowFormat::Bf16P12,
                                              NgramRowFormat::Bf16P12Blockdir};
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
    // Another table of this model, in any of the formats: its own geometry and id, and an FP8 table's
    // scale or an NVFP4 table's multiplier from its header (a quantization's scale is a property of
    // the volume, not of the artifact).
    out.file_bytes = static_cast<std::uint64_t>(out.header_bytes) + out.blocks * out.block_bytes;
    if (out.format == NgramRowFormat::Bf16P12Blockdir) {
        // The header's W, its on-disk row size, base table and side stream; 17 rows a 4 KiB block and
        // the decoded P12 row width are the format's, not the row size's, geometry.
        out.block_width = read_le<std::uint16_t>(p + kWidthOffset);
        out.slot_bytes  = read_le<std::uint16_t>(p + kSlotBytesOffset);
        std::memcpy(out.blockdir_bases.data(), p + kBasesOffset, kBlockdirBaseTableSize);
        out.side_offset = read_le<std::uint64_t>(p + kSideOffsetOffset);
        out.side_bytes  = read_le<std::uint64_t>(p + kSideBytesOffset);
        out.side_count  = read_le<std::uint64_t>(p + kSideCountOffset);
        if (out.block_width != kBlockdirWidth || out.slot_bytes != kBlockdirPlaneBytes ||
            out.rows_per_block != kBlockdirRows || out.row_bytes != row_bytes_of(out.format)) {
            fail("a blockdir volume must hold 17 rows of 160 values a 4 KiB block");
        }
        for (const std::uint8_t base : out.blockdir_bases) {
            if (base > 240) { fail("a blockdir base table entry is outside [0, 240]"); }
        }
        if (out.side_bytes % 4096 != 0 ||
            (out.side_bytes == 0 ? (out.side_offset != 0 || out.side_count != 0)
                                 : (out.side_offset != out.file_bytes || out.side_count == 0))) {
            fail("the blockdir side stream's header fields disagree");
        }
        out.file_bytes += out.side_bytes;
    }
    if (out.format != NgramRowFormat::Bf16P12Blockdir &&
        out.rows_per_block != out.block_bytes / out.row_bytes) {
        fail("the n-gram volume's geometry differs from this model's table");
    }
    if (out.rows != ple.table.rows || out.block_bytes != 4096 || out.header_bytes != 4096 ||
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
