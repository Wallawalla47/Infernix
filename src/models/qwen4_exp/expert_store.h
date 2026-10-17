#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp {

// The routed experts read in place from the artifact (design §19.3.7, the SSD tier; plan
// memory-tiers.md §4.4): for every (layer, expert) record, its file segments (one, or two for a
// record that straddles part files), each 4 KiB-aligned in file offset and size so it is one
// unbuffered read; and each layer's expert multipliers (the bank's scale tail), read at load. A
// GGUF import's banks (ggml_expert_*_v1) have a record stride per layer and no scale tail.
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

    // `banks[l]` is layer l's streamed expert-bank object with records `record_strides[l]` bytes
    // apart, `multipliers[l]` its scale tail ([experts][3] FP32 words; empty for a bank without one).
    // Throws ArtifactError on a record that is not 4 KiB-aligned.
    ExpertStore(const artifact::StreamSource& source, std::span<const artifact::ObjectHandle> banks,
                std::vector<std::uint64_t> record_strides, std::uint32_t experts,
                std::vector<std::vector<float>> multipliers);
    // From explicit locations (tests and tools): records[layer * experts + e] lists a record's
    // segments, which cover its layer's stride; the same alignment rules apply.
    ExpertStore(std::vector<std::filesystem::path> files, std::vector<std::uint64_t> record_strides,
                std::uint32_t experts, const std::vector<std::vector<Segment>>& records,
                std::vector<std::vector<float>> multipliers);

    [[nodiscard]] std::span<const std::filesystem::path> files() const noexcept { return files_; }
    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }
    [[nodiscard]] std::uint32_t experts() const noexcept { return experts_; }
    // The largest record pitch: the size of a slot that holds any record.
    [[nodiscard]] std::uint64_t record_bytes() const noexcept { return max_stride_; }
    // Bytes read per record of `layer`: its 4 KiB-aligned record pitch.
    [[nodiscard]] std::uint64_t record_bytes(std::uint32_t layer) const { return strides_.at(layer); }
    [[nodiscard]] std::span<const Segment> segments(std::uint32_t layer, std::uint32_t expert) const;
    // Layer `layer`'s scale tail, or null for a bank without one.
    [[nodiscard]] const float* multipliers(std::uint32_t layer) const {
        const auto& words = multipliers_.at(layer);
        return words.empty() ? nullptr : words.data();
    }
    // Records whose bytes span two files.
    [[nodiscard]] std::uint32_t straddling() const noexcept { return straddling_; }

    // The most segments of one record: two in the artifact (a record that straddles part files)
    // plus its tail in an expert stripe file.
    static constexpr std::size_t kMaxSegments = 3;

private:
    struct Record {
        std::array<Segment, kMaxSegments> segments{};
        std::uint8_t count = 0;
    };
    std::vector<std::filesystem::path> files_;
    std::uint32_t layers_ = 0, experts_ = 0, straddling_ = 0;
    std::vector<std::uint64_t> strides_; // per layer
    std::uint64_t max_stride_ = 0;
    std::vector<Record> records_; // [layer][expert]
    std::vector<std::vector<float>> multipliers_;
};

// An expert stripe file (tools/flash_next/expert_stripe.py): the tail of every routed expert record
// of one artifact, on another drive, so the SSD tier reads each record from two drives at once (the
// head from the artifact, the tail from here) without a second copy of the experts.
//
// Format v1, little-endian: "INFXSTR1", u32 version 1, u32 layers, u32 experts, u32 header bytes
// (a 4 KiB multiple), the artifact's 16-byte id, then per layer u64 record stride, u64 head bytes
// (the part read from the artifact; a 4 KiB multiple below the stride) and u64 file offset of the
// layer's first tail; zero padding to the header bytes. Each layer's tails follow at its offset,
// expert by expert, each stride - head bytes.
struct ExpertStripe {
    static constexpr std::uint32_t kVersion = 1;
    struct Layer {
        std::uint64_t stride = 0, head = 0, offset = 0;
    };
    std::filesystem::path path;
    artifact::ArtifactId artifact_id{};
    std::uint32_t experts = 0;
    std::vector<Layer> layers;
};

// Reads and checks a stripe file's header (its alignment, and that the file holds every tail);
// throws ArtifactError.
[[nodiscard]] ExpertStripe read_expert_stripe(const std::filesystem::path& path);

// `base` with each record's tail read from `stripe`: its head from the artifact's segments, cut at
// the stripe's head bytes, then one segment in the stripe file. Throws ArtifactError when the
// stripe is of another artifact (`artifact_id`) or another geometry.
[[nodiscard]] ExpertStore with_stripe(const ExpertStore& base, const ExpertStripe& stripe,
                                      const artifact::ArtifactId& artifact_id);

} // namespace infernix::models::qwen4_exp
