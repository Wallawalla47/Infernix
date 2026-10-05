#include "models/qwen4_exp/expert_store.h"

#include "artifact/framing.h"

#include <string>

namespace ninfer::models::qwen4_exp {

ExpertStore::ExpertStore(const artifact::StreamSource& source, std::span<const artifact::ObjectHandle> banks,
                         std::uint64_t record_stride, std::uint32_t experts, std::vector<std::vector<float>> multipliers)
    : files_(source.files().begin(), source.files().end()), layers_(static_cast<std::uint32_t>(banks.size())),
      experts_(experts), record_stride_(record_stride), multipliers_(std::move(multipliers)) {
    if (record_stride == 0 || record_stride % kAlignment != 0 || record_stride > 0xFFFFFFFFULL) {
        throw artifact::ArtifactError("streamed expert records need a 4 KiB-aligned pitch");
    }
    if (multipliers_.size() != banks.size()) { throw artifact::ArtifactError("streamed expert multipliers per layer"); }
    records_.resize(static_cast<std::size_t>(layers_) * experts_);
    for (std::uint32_t l = 0; l < layers_; ++l) {
        if (multipliers_[l].size() != 3ULL * experts_) {
            throw artifact::ArtifactError("streamed expert multipliers differ from the expert count");
        }
        if (source.object_bytes(banks[l]) < record_stride * experts_) {
            throw artifact::ArtifactError("streamed expert bank is smaller than its records");
        }
        for (std::uint32_t e = 0; e < experts_; ++e) {
            const auto pieces = source.segments(banks[l], record_stride * e, record_stride);
            Record& record    = records_[static_cast<std::size_t>(l) * experts_ + e];
            if (pieces.empty() || pieces.size() > record.segments.size()) {
                throw artifact::ArtifactError("streamed expert record spans more than two files");
            }
            for (const auto& piece : pieces) {
                if (piece.offset % kAlignment != 0 || piece.bytes % kAlignment != 0) {
                    throw artifact::ArtifactError("streamed expert record L" + std::to_string(l) + " E" +
                                                  std::to_string(e) + " is not 4 KiB-aligned in its file");
                }
                record.segments[record.count++] = {piece.file, piece.offset, static_cast<std::uint32_t>(piece.at),
                                                   static_cast<std::uint32_t>(piece.bytes)};
            }
            straddling_ += record.count > 1 ? 1U : 0U;
        }
    }
}

std::span<const ExpertStore::Segment> ExpertStore::segments(std::uint32_t layer, std::uint32_t expert) const {
    const Record& record = records_.at(static_cast<std::size_t>(layer) * experts_ + expert);
    return {record.segments.data(), record.count};
}

} // namespace ninfer::models::qwen4_exp
