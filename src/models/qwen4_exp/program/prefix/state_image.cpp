#include "models/qwen4_exp/program/prefix/state_image.h"

#include "core/device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp::prefix {
namespace {

constexpr std::uint64_t kHeaderMagic   = 0x31474D4934515751ULL; // "QWQ4IMG1"
constexpr std::uint32_t kHeaderVersion = 1;

// Header bytes: magic, version, fingerprint, frontier, mtp_next, flags, mtp_accept.
struct PackedHeader {
    std::uint64_t magic       = 0;
    std::uint32_t version     = 0;
    std::uint32_t flags       = 0; // bit 0 mtp_written, bit 1 lineage_echo
    std::uint64_t fingerprint = 0;
    std::uint64_t image_bytes = 0;
    std::uint32_t frontier    = 0;
    std::int32_t mtp_next     = -1;
    std::array<double, 8> mtp_accept{};
};
static_assert(sizeof(PackedHeader) <= kStateImageHeaderBytes);

std::size_t align_up(std::size_t bytes) {
    return (bytes + kStateImageAlignment - 1U) / kStateImageAlignment * kStateImageAlignment;
}

std::uint64_t fnv1a(std::uint64_t hash, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        hash ^= (value >> (8 * i)) & 0xFFU;
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

// One copy per contiguous run between a Device range and the segmented Host image.
template <class Copy>
void for_each_segment_run(std::size_t offset, std::size_t bytes, std::size_t segment_bytes, Copy&& copy) {
    std::size_t done = 0;
    while (done < bytes) {
        const std::size_t at      = offset + done;
        const std::size_t segment = at / segment_bytes;
        const std::size_t within  = at % segment_bytes;
        const std::size_t run     = std::min(bytes - done, segment_bytes - within);
        copy(segment, within, done, run);
        done += run;
    }
}

} // namespace

StateImageSpec state_image_spec(const TextConfig& c, bool mtp) {
    StateImageSpec spec;
    spec.layer_types         = c.layer_types;
    spec.ple_layer           = c.ple.layer;
    spec.gdn_recurrent_bytes = static_cast<std::size_t>(c.gdn.key_head_dim) * c.gdn.value_head_dim * c.gdn.value_heads *
                               sizeof(float);
    spec.gdn_conv_bytes = static_cast<std::size_t>(c.gdn.conv_channels()) * (c.gdn.conv_kernel - 1U) * 2U;
    spec.ple_bytes      = static_cast<std::size_t>(c.residual_width()) * c.ple.conv_span() * 2U;
    spec.qsa_tail_bytes = static_cast<std::size_t>(c.qsa.index_head_dim) * (c.qsa.compress_ratio - 1U) * 2U;
    spec.mtp            = mtp;
    if (mtp) {
        spec.mtp_saved_bytes = static_cast<std::size_t>(c.residual_width()) * 2U;
        spec.mtp_tail_bytes  = spec.qsa_tail_bytes;
    }
    return spec;
}

StateImageLayout plan_state_image(const StateImageSpec& spec) {
    if (spec.layer_types.empty() || spec.ple_layer >= spec.layer_types.size()) {
        throw std::invalid_argument("Qwen4Exp state image: the PLE layer is outside the decoder");
    }
    StateImageLayout layout;
    layout.spec           = spec;
    std::size_t offset    = 0;
    const auto add_part   = [&](StateImagePart::Kind kind, std::uint32_t index, std::size_t bytes) {
        if (bytes == 0) { throw std::invalid_argument("Qwen4Exp state image: a part has no bytes"); }
        layout.parts.push_back(StateImagePart{kind, index, offset, bytes});
        offset = align_up(offset + bytes);
    };
    add_part(StateImagePart::Kind::Header, 0, kStateImageHeaderBytes);
    std::uint32_t gdn = 0, attention = 0;
    for (std::uint32_t layer = 0; layer < spec.layer_types.size(); ++layer) {
        layout.group_begin.push_back(static_cast<std::uint32_t>(layout.parts.size()));
        if (layer == spec.ple_layer) { add_part(StateImagePart::Kind::Ple, 0, spec.ple_bytes); }
        if (spec.layer_types[layer] == MixerKind::Gdn) {
            add_part(StateImagePart::Kind::GdnRecurrent, gdn, spec.gdn_recurrent_bytes);
            add_part(StateImagePart::Kind::GdnConv, gdn, spec.gdn_conv_bytes);
            ++gdn;
        } else {
            add_part(StateImagePart::Kind::QsaTail, attention, spec.qsa_tail_bytes);
            ++attention;
        }
    }
    layout.group_begin.push_back(static_cast<std::uint32_t>(layout.parts.size()));
    if (spec.mtp) {
        add_part(StateImagePart::Kind::MtpSaved, 0, spec.mtp_saved_bytes);
        add_part(StateImagePart::Kind::MtpTail, 0, spec.mtp_tail_bytes);
    }
    layout.group_begin.push_back(static_cast<std::uint32_t>(layout.parts.size()));
    layout.image_bytes = offset;
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    for (const StateImagePart& part : layout.parts) {
        hash = fnv1a(hash, static_cast<std::uint64_t>(part.kind));
        hash = fnv1a(hash, part.index);
        hash = fnv1a(hash, part.offset);
        hash = fnv1a(hash, part.bytes);
    }
    layout.fingerprint = hash;
    return layout;
}

void write_state_image_header(const StateImageLayout& layout, const StateImageHeader& header,
                              std::byte* image_begin) {
    PackedHeader packed;
    packed.magic       = kHeaderMagic;
    packed.version     = kHeaderVersion;
    packed.flags       = (header.mtp_written ? 1U : 0U) | (header.lineage_echo ? 2U : 0U);
    packed.fingerprint = layout.fingerprint;
    packed.image_bytes = layout.image_bytes;
    packed.frontier    = header.frontier;
    packed.mtp_next    = header.mtp_next;
    packed.mtp_accept  = header.mtp_accept;
    std::memset(image_begin, 0, kStateImageHeaderBytes);
    std::memcpy(image_begin, &packed, sizeof(packed));
}

std::optional<StateImageHeader> read_state_image_header(const StateImageLayout& layout,
                                                        const std::byte* image_begin) {
    PackedHeader packed;
    std::memcpy(&packed, image_begin, sizeof(packed));
    if (packed.magic != kHeaderMagic || packed.version != kHeaderVersion ||
        packed.fingerprint != layout.fingerprint || packed.image_bytes != layout.image_bytes ||
        (packed.flags & ~3U) != 0) {
        return std::nullopt;
    }
    StateImageHeader header;
    header.frontier     = packed.frontier;
    header.mtp_written  = (packed.flags & 1U) != 0;
    header.lineage_echo = (packed.flags & 2U) != 0;
    header.mtp_next     = packed.mtp_next;
    header.mtp_accept   = packed.mtp_accept;
    return header;
}

LaneStateImage::LaneStateImage(const StateImageLayout& layout, const LaneStateBuffers& buffers)
    : layout_(&layout), buffers_(buffers) {
    const StateImageSpec& spec = layout.spec;
    std::uint32_t gdn_layers = 0, attention_layers = 0;
    for (const MixerKind kind : spec.layer_types) { ++(kind == MixerKind::Gdn ? gdn_layers : attention_layers); }
    if (buffers.lanes == 0 || buffers.ple == nullptr || (attention_layers + (spec.mtp ? 1U : 0U) > 0 && buffers.tails == nullptr) ||
        (spec.mtp && buffers.mtp_saved == nullptr)) {
        throw std::invalid_argument("Qwen4Exp state image: lane buffers are missing");
    }
    if (gdn_layers > 0) {
        if (buffers.gdn == nullptr || buffers.gdn->layer_count() != gdn_layers ||
            buffers.gdn->slot_count() < static_cast<std::int32_t>(buffers.lanes) ||
            buffers.gdn->recurrent_slot(0, 0).bytes() != spec.gdn_recurrent_bytes ||
            buffers.gdn->conv_slot(0, 0).bytes() != spec.gdn_conv_bytes) {
            throw std::invalid_argument("Qwen4Exp state image: the GDN state pool does not match the layout");
        }
    }
}

std::byte* LaneStateImage::device_part(const StateImagePart& part, std::uint32_t lane) const {
    check_lane(lane);
    const StateImageSpec& spec = layout_->spec;
    std::uint32_t attention_layers = 0;
    for (const MixerKind kind : spec.layer_types) { attention_layers += kind == MixerKind::Attention ? 1U : 0U; }
    const auto slot = static_cast<std::int32_t>(lane);
    switch (part.kind) {
    case StateImagePart::Kind::Header: break;
    case StateImagePart::Kind::Ple: return buffers_.ple + static_cast<std::size_t>(lane) * spec.ple_bytes;
    case StateImagePart::Kind::GdnRecurrent:
        return static_cast<std::byte*>(buffers_.gdn->recurrent_slot(part.index, slot).data);
    case StateImagePart::Kind::GdnConv: return static_cast<std::byte*>(buffers_.gdn->conv_slot(part.index, slot).data);
    case StateImagePart::Kind::QsaTail:
        return buffers_.tails + (static_cast<std::size_t>(part.index) * buffers_.lanes + lane) * spec.qsa_tail_bytes;
    case StateImagePart::Kind::MtpSaved:
        return buffers_.mtp_saved + static_cast<std::size_t>(lane) * spec.mtp_saved_bytes;
    case StateImagePart::Kind::MtpTail:
        return buffers_.tails + (static_cast<std::size_t>(attention_layers) * buffers_.lanes + lane) * spec.qsa_tail_bytes;
    }
    throw std::logic_error("Qwen4Exp state image: the header has no Device bytes");
}

void LaneStateImage::check_lane(std::uint32_t lane) const {
    if (lane >= buffers_.lanes) { throw std::out_of_range("Qwen4Exp state image: lane out of range"); }
}

void LaneStateImage::check_image(std::size_t segments, std::size_t segment_bytes) const {
    if (segment_bytes == 0 || segments * segment_bytes < layout_->image_bytes) {
        throw std::invalid_argument("Qwen4Exp state image: the Host segments do not hold an image");
    }
}

void LaneStateImage::copy_lane_to_host(std::uint32_t lane, const HostImage& image, cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    for (std::size_t p = 1; p < layout_->parts.size(); ++p) {
        const StateImagePart& part = layout_->parts[p];
        const std::byte* source    = device_part(part, lane);
        for_each_segment_run(part.offset, part.bytes, image.segment_bytes,
                             [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                                 CUDA_CHECK(cudaMemcpyAsync(image.segments[segment] + within, source + done, run,
                                                            cudaMemcpyDeviceToHost, stream));
                             });
    }
}

void LaneStateImage::copy_group_to_host(std::uint32_t group, std::uint32_t lane, const HostImage& image,
                                        cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    if (group >= layout_->groups()) { throw std::out_of_range("Qwen4Exp state image: group out of range"); }
    for (std::uint32_t p = layout_->group_begin[group]; p < layout_->group_begin[group + 1U]; ++p) {
        const StateImagePart& part = layout_->parts[p];
        const std::byte* source    = device_part(part, lane);
        for_each_segment_run(part.offset, part.bytes, image.segment_bytes,
                             [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                                 CUDA_CHECK(cudaMemcpyAsync(image.segments[segment] + within, source + done, run,
                                                            cudaMemcpyDeviceToHost, stream));
                             });
    }
}

void LaneStateImage::copy_lane_to_packed(std::uint32_t lane, std::byte* packed, cudaStream_t stream) const {
    for (std::size_t p = 1; p < layout_->parts.size(); ++p) {
        const StateImagePart& part = layout_->parts[p];
        CUDA_CHECK(cudaMemcpyAsync(packed + part.offset, device_part(part, lane), part.bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void LaneStateImage::copy_packed_to_host(const std::byte* packed, const HostImage& image, cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    for_each_segment_run(0, layout_->image_bytes, image.segment_bytes,
                         [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                             CUDA_CHECK(cudaMemcpyAsync(image.segments[segment] + within, packed + done, run,
                                                        cudaMemcpyDeviceToHost, stream));
                         });
}

void LaneStateImage::copy_group_from_host(std::uint32_t group, const HostImageConst& image, std::uint32_t lane,
                                          cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    if (group >= layout_->groups()) { throw std::out_of_range("Qwen4Exp state image: group out of range"); }
    for (std::uint32_t p = layout_->group_begin[group]; p < layout_->group_begin[group + 1U]; ++p) {
        const StateImagePart& part = layout_->parts[p];
        std::byte* destination     = device_part(part, lane);
        for_each_segment_run(part.offset, part.bytes, image.segment_bytes,
                             [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                                 CUDA_CHECK(cudaMemcpyAsync(destination + done, image.segments[segment] + within, run,
                                                            cudaMemcpyHostToDevice, stream));
                             });
    }
}

void LaneStateImage::copy_group_from_packed(std::uint32_t group, const std::byte* packed, std::uint32_t lane,
                                            cudaStream_t stream) const {
    if (group >= layout_->groups()) { throw std::out_of_range("Qwen4Exp state image: group out of range"); }
    for (std::uint32_t p = layout_->group_begin[group]; p < layout_->group_begin[group + 1U]; ++p) {
        const StateImagePart& part = layout_->parts[p];
        CUDA_CHECK(cudaMemcpyAsync(device_part(part, lane), packed + part.offset, part.bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

KvPageGeometry kv_page_geometry(const TextConfig& c, KvCacheStorage storage, bool mtp) {
    const auto layout   = paged_kv_storage_layout(storage, static_cast<std::int32_t>(c.attention.head_dim));
    const auto kv_heads = static_cast<std::int32_t>(c.attention.kv_heads);
    KvPageGeometry out;
    out.kv_layers = c.attention_layers + (mtp ? 1U : 0U);
    for (std::uint32_t l = 0; l < out.kv_layers; ++l) {
        out.geometry.planes.push_back({layout.key.data_dtype, layout.key.data_leading_extent, kv_heads});
        if (layout.key.has_scale()) {
            out.geometry.planes.push_back({layout.key.scale_dtype, layout.key.scale_leading_extent, kv_heads});
        }
        out.geometry.planes.push_back({layout.value.data_dtype, layout.value.data_leading_extent, kv_heads});
        if (layout.value.has_scale()) {
            out.geometry.planes.push_back({layout.value.scale_dtype, layout.value.scale_leading_extent, kv_heads});
        }
        out.geometry.planes.push_back(
            {DType::BF16, static_cast<std::int32_t>(c.qsa.index_head_dim / c.qsa.compress_ratio), 1});
    }
    out.planes_per_layer = out.kv_layers == 0 ? 0U : static_cast<std::uint32_t>(out.geometry.planes.size()) / out.kv_layers;
    return out;
}

PlaneRange kv_layer_planes(const KvPageGeometry& geometry, std::uint32_t kv_layer) {
    if (kv_layer >= geometry.kv_layers) { throw std::out_of_range("Qwen4Exp KV layer out of range"); }
    const std::size_t begin = static_cast<std::size_t>(kv_layer) * geometry.planes_per_layer;
    return PlaneRange{begin, begin + geometry.planes_per_layer};
}

} // namespace ninfer::models::qwen4_exp::prefix
