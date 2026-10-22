#include "runtime/expert_offload/expert_store.h"

#include "artifact/framing.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

namespace infernix::runtime::expert_offload {
namespace {

// Every layer's pitch is a 4 KiB multiple that fits a Segment; returns the largest.
std::uint64_t check_strides(const std::vector<std::uint64_t>& strides, std::size_t layers, const char* what) {
    if (strides.size() != layers ||
        std::any_of(strides.begin(), strides.end(), [](std::uint64_t s) {
            return s == 0 || s % ExpertStore::kAlignment != 0 || s > 0xFFFFFFFFULL;
        })) {
        throw artifact::ArtifactError(what);
    }
    return *std::max_element(strides.begin(), strides.end());
}

// A layer's scale tail has three words per expert, or none (a GGUF import's bank).
void check_multipliers(const std::vector<std::vector<float>>& multipliers, std::size_t layers, std::uint32_t experts) {
    if (multipliers.size() != layers) { throw artifact::ArtifactError("streamed expert multipliers per layer"); }
    for (const auto& words : multipliers) {
        if (!words.empty() && words.size() != 3ULL * experts) {
            throw artifact::ArtifactError("streamed expert multipliers differ from the expert count");
        }
    }
}

} // namespace

ExpertStore::ExpertStore(const artifact::StreamSource& source, std::span<const artifact::ObjectHandle> banks,
                         std::vector<std::uint64_t> record_strides, std::uint32_t experts,
                         std::vector<std::vector<float>> multipliers)
    : files_(source.files().begin(), source.files().end()), layers_(static_cast<std::uint32_t>(banks.size())),
      experts_(experts), strides_(std::move(record_strides)), multipliers_(std::move(multipliers)) {
    max_stride_ = check_strides(strides_, banks.size(), "streamed expert records need a 4 KiB-aligned pitch per layer");
    check_multipliers(multipliers_, banks.size(), experts_);
    records_.resize(static_cast<std::size_t>(layers_) * experts_);
    for (std::uint32_t l = 0; l < layers_; ++l) {
        const std::uint64_t stride = strides_[l];
        if (source.object_bytes(banks[l]) < stride * experts_) {
            throw artifact::ArtifactError("streamed expert bank is smaller than its records");
        }
        for (std::uint32_t e = 0; e < experts_; ++e) {
            const auto pieces = source.segments(banks[l], stride * e, stride);
            Record& record    = records_[static_cast<std::size_t>(l) * experts_ + e];
            if (pieces.empty() || pieces.size() > 2) {
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

ExpertStore::ExpertStore(std::vector<std::filesystem::path> files, std::vector<std::uint64_t> record_strides,
                         std::uint32_t experts, const std::vector<std::vector<Segment>>& records,
                         std::vector<std::vector<float>> multipliers)
    : files_(std::move(files)), layers_(static_cast<std::uint32_t>(record_strides.size())), experts_(experts),
      strides_(std::move(record_strides)), multipliers_(std::move(multipliers)) {
    max_stride_ = check_strides(strides_, strides_.size(), "expert store: inconsistent explicit records");
    check_multipliers(multipliers_, layers_, experts_);
    if (records.size() != static_cast<std::size_t>(layers_) * experts) {
        throw artifact::ArtifactError("expert store: inconsistent explicit records");
    }
    records_.resize(records.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
        Record& record = records_[i];
        if (records[i].empty() || records[i].size() > record.segments.size()) {
            throw artifact::ArtifactError("expert store: a record needs one to three segments");
        }
        std::uint64_t covered = 0;
        for (const Segment& s : records[i]) {
            if (s.file >= files_.size() || s.offset % kAlignment != 0 || s.bytes % kAlignment != 0 || s.at != covered) {
                throw artifact::ArtifactError("expert store: a segment is misaligned or out of order");
            }
            covered += s.bytes;
            record.segments[record.count++] = s;
        }
        if (covered != strides_[i / experts_]) {
            throw artifact::ArtifactError("expert store: segments do not cover the record");
        }
        straddling_ += record.count > 1 ? 1U : 0U;
    }
}

std::span<const ExpertStore::Segment> ExpertStore::segments(std::uint32_t layer, std::uint32_t expert) const {
    const Record& record = records_.at(static_cast<std::size_t>(layer) * experts_ + expert);
    return {record.segments.data(), record.count};
}

namespace {

constexpr char kStripeMagic[8] = {'I', 'N', 'F', 'X', 'S', 'T', 'R', '1'};
constexpr std::size_t kStripeFixed = 40; // magic, version, layers, experts, header bytes, artifact id
constexpr std::size_t kStripeLayer = 24; // stride, head, offset

std::uint64_t le(const std::uint8_t* p, int bytes) {
    std::uint64_t v = 0;
    for (int i = bytes - 1; i >= 0; --i) { v = (v << 8U) | p[i]; }
    return v;
}

} // namespace

ExpertStripe read_expert_stripe(const std::filesystem::path& path) {
    const auto fail = [&](const std::string& why) {
        throw artifact::ArtifactError("expert stripe " + path.string() + ": " + why);
    };
    std::ifstream file(path, std::ios::binary);
    if (!file) { fail("cannot open"); }
    std::vector<std::uint8_t> header(ExpertStore::kAlignment);
    if (!file.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()))) {
        fail("shorter than its header");
    }
    if (!std::equal(std::begin(kStripeMagic), std::end(kStripeMagic), header.begin())) {
        fail("not an expert stripe (or not completely written)");
    }
    ExpertStripe out;
    out.path = path;
    if (le(&header[8], 4) != ExpertStripe::kVersion) { fail("version " + std::to_string(le(&header[8], 4)) + " is not 1"); }
    const auto layers         = static_cast<std::uint32_t>(le(&header[12], 4));
    out.experts               = static_cast<std::uint32_t>(le(&header[16], 4));
    const std::uint64_t bytes = le(&header[20], 4);
    std::memcpy(out.artifact_id.data(), &header[24], out.artifact_id.size());
    if (layers == 0 || out.experts == 0 || bytes % ExpertStore::kAlignment != 0 ||
        bytes < kStripeFixed + static_cast<std::uint64_t>(layers) * kStripeLayer) {
        fail("inconsistent header");
    }
    header.resize(bytes);
    if (bytes > ExpertStore::kAlignment &&
        !file.read(reinterpret_cast<char*>(header.data() + ExpertStore::kAlignment),
                   static_cast<std::streamsize>(bytes - ExpertStore::kAlignment))) {
        fail("shorter than its header");
    }
    std::error_code error;
    const std::uint64_t size = std::filesystem::file_size(path, error);
    if (error) { fail("cannot read its size"); }
    std::uint64_t end = bytes;
    for (std::uint32_t l = 0; l < layers; ++l) {
        const std::uint8_t* p = &header[kStripeFixed + static_cast<std::size_t>(l) * kStripeLayer];
        ExpertStripe::Layer layer{.stride = le(p, 8), .head = le(p + 8, 8), .offset = le(p + 16, 8)};
        if (layer.stride == 0 || layer.stride % ExpertStore::kAlignment != 0 || layer.head % ExpertStore::kAlignment != 0 ||
            layer.head >= layer.stride || layer.offset % ExpertStore::kAlignment != 0 || layer.offset < end) {
            fail("layer " + std::to_string(l) + " is misaligned or overlaps another");
        }
        end = layer.offset + (layer.stride - layer.head) * out.experts;
        out.layers.push_back(layer);
    }
    if (size < end) { fail("holds " + std::to_string(size) + " bytes of " + std::to_string(end)); }
    return out;
}

ExpertStore with_stripe(const ExpertStore& base, const ExpertStripe& stripe, const artifact::ArtifactId& artifact_id) {
    const auto fail = [&](const std::string& why) {
        throw artifact::ArtifactError("expert stripe " + stripe.path.string() + ": " + why);
    };
    if (stripe.artifact_id != artifact_id) { fail("belongs to another artifact"); }
    if (stripe.layers.size() != base.layers() || stripe.experts != base.experts()) {
        fail("has another layer or expert count than the artifact");
    }
    std::vector<std::filesystem::path> files(base.files().begin(), base.files().end());
    files.push_back(stripe.path);
    const auto tail_file = static_cast<std::uint32_t>(files.size() - 1);
    std::vector<std::uint64_t> strides;
    std::vector<std::vector<float>> multipliers;
    std::vector<std::vector<ExpertStore::Segment>> records;
    records.reserve(static_cast<std::size_t>(base.layers()) * base.experts());
    for (std::uint32_t l = 0; l < base.layers(); ++l) {
        const ExpertStripe::Layer& layer = stripe.layers[l];
        if (layer.stride != base.record_bytes(l)) {
            fail("layer " + std::to_string(l) + " has another record stride than the artifact");
        }
        strides.push_back(layer.stride);
        const float* words = base.multipliers(l);
        multipliers.emplace_back(words, words == nullptr ? words : words + 3ULL * base.experts());
        const std::uint64_t tail = layer.stride - layer.head;
        for (std::uint32_t e = 0; e < base.experts(); ++e) {
            std::vector<ExpertStore::Segment> record;
            for (ExpertStore::Segment s : base.segments(l, e)) { // the head: the artifact's bytes below layer.head
                if (s.at >= layer.head) { break; }
                s.bytes = static_cast<std::uint32_t>(std::min<std::uint64_t>(s.bytes, layer.head - s.at));
                record.push_back(s);
            }
            record.push_back({tail_file, layer.offset + tail * e, static_cast<std::uint32_t>(layer.head),
                              static_cast<std::uint32_t>(tail)});
            records.push_back(std::move(record));
        }
    }
    return ExpertStore(std::move(files), std::move(strides), base.experts(), records, std::move(multipliers));
}

} // namespace infernix::runtime::expert_offload
