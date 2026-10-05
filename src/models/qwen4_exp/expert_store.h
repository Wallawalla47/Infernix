#pragma once

#include "artifact/materializer.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

// The routed experts read in place from the artifact (design §19.3.7, the SSD tier; plan
// memory-tiers.md §4.4): for every (layer, expert) record, its file segments (one, or two for a
// record that straddles part files), each 4 KiB-aligned in file offset and size so it is one
// unbuffered read; and each layer's expert multipliers (the bank's scale tail), read at load.
// Immutable Model data; it opens no file (the tier's reader opens its own deny-write handles).
class ExpertStore {
public:
    static constexpr std::uint64_t kAlignment = 4096;

    struct Segment {
        std::uint32_t file   = 0; // index into files()
        std::uint64_t offset = 0; // byte offset in the file
        std::uint32_t at     = 0; // byte offset inside the record
        std::uint32_t bytes  = 0;
    };

    // `banks[l]` is layer l's streamed expert-bank object, `multipliers[l]` its scale tail
    // ([experts][3] FP32 words). Throws ArtifactError on a record that is not 4 KiB-aligned.
    ExpertStore(const artifact::StreamSource& source, std::span<const artifact::ObjectHandle> banks,
                std::uint64_t record_stride, std::uint32_t experts, std::vector<std::vector<float>> multipliers);

    [[nodiscard]] std::span<const std::filesystem::path> files() const noexcept { return files_; }
    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }
    [[nodiscard]] std::uint32_t experts() const noexcept { return experts_; }
    // Bytes read per record: the 4 KiB-aligned record pitch.
    [[nodiscard]] std::uint64_t record_bytes() const noexcept { return record_stride_; }
    [[nodiscard]] std::span<const Segment> segments(std::uint32_t layer, std::uint32_t expert) const;
    [[nodiscard]] const float* multipliers(std::uint32_t layer) const { return multipliers_.at(layer).data(); }
    // Records whose bytes span two files.
    [[nodiscard]] std::uint32_t straddling() const noexcept { return straddling_; }

private:
    struct Record {
        std::array<Segment, 2> segments{};
        std::uint8_t count = 0;
    };
    std::vector<std::filesystem::path> files_;
    std::uint32_t layers_ = 0, experts_ = 0, straddling_ = 0;
    std::uint64_t record_stride_ = 0;
    std::vector<Record> records_; // [layer][expert]
    std::vector<std::vector<float>> multipliers_;
};

} // namespace ninfer::models::qwen4_exp
