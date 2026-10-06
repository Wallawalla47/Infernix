#pragma once

// The packed state image of one Qwen4Exp lane at a frontier (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.1): every byte of the lane's GDN recurrent and convolution state, PLE convolution history,
// QSA raw-key tails and MTP state that its KV pages do not hold. Parts follow the forward pass, so a
// restore lands each decoder layer's state as one group the forward pass can wait for on its own:
//
//   header | layer 0 .. L-1: GDN (recurrent, conv) or attention (QSA tail, and with a vector-quantized
//          KV storage its exact window's five planes); the PLE history joins the group of the layer
//          it precedes | MTP group: saved residual, MTP tail (and window)
//
// Every part starts on a 256-byte boundary. All copies are byte copies; nothing is requantized. On
// the Host the packed byte o lives at segments[o / segment_bytes] + o % segment_bytes (the slabs of
// the prefix cache's pinned pool, not contiguous); a packed Device slot holds the same layout.

#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "models/qwen4_exp/config.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp::prefix {

inline constexpr std::size_t kStateImageHeaderBytes = 256;
inline constexpr std::size_t kStateImageAlignment   = 256;

// The exact recent-key window of the vector-quantized KV storages (core/paged_kv_storage.h,
// kv_window.cuh) per KV layer (the attention layers, then MTP's) and lane: five planes, each
// [leading, kKVWindowSlots, kv_heads, lanes]: K codes and V codes (I8, 256), K and V group scales
// (FP16, 4), tags (I32, 2). A KV layer's planes follow each other; layers follow each other.
struct KvWindowGeometry {
    static constexpr int kPlanes = 5;
    std::int32_t kv_heads   = 0; // 0: the storage has no window
    std::uint32_t kv_layers = 0;
    std::uint32_t lanes     = 0;

    [[nodiscard]] bool present() const noexcept { return kv_heads > 0; }
    // One lane's bytes of plane p of a KV layer.
    [[nodiscard]] std::size_t lane_bytes(int plane) const;
    [[nodiscard]] std::size_t layer_bytes() const;
    [[nodiscard]] std::size_t bytes() const { return layer_bytes() * kv_layers; }
    // Offset of plane p of a KV layer from the window's base.
    [[nodiscard]] std::size_t plane_offset(std::uint32_t kv_layer, int plane) const;
};

[[nodiscard]] KvWindowGeometry kv_window_geometry(const TextConfig& config, KvCacheStorage storage, bool mtp,
                                                  std::uint32_t lanes);
// KV layer k's window planes at `base` (slots unset: the forward pass sets each call's).
[[nodiscard]] PagedKVWindowView kv_window_view(const KvWindowGeometry& geometry, void* base, std::uint32_t kv_layer);

// Byte sizes of one lane's state, per layer where it is per layer.
struct StateImageSpec {
    std::vector<MixerKind> layer_types; // decoder layers in order
    std::uint32_t ple_layer          = 0; // the PLE history joins this layer's group
    std::size_t gdn_recurrent_bytes  = 0;
    std::size_t gdn_conv_bytes       = 0;
    std::size_t ple_bytes            = 0;
    std::size_t qsa_tail_bytes       = 0;
    bool mtp                         = false;
    std::size_t mtp_saved_bytes      = 0;
    std::size_t mtp_tail_bytes       = 0;
    // One lane's bytes of each exact-window plane per KV layer (zero without a window).
    std::array<std::size_t, KvWindowGeometry::kPlanes> window_bytes{};
};

[[nodiscard]] StateImageSpec state_image_spec(const TextConfig& config, bool mtp, KvCacheStorage storage);

struct StateImagePart {
    enum class Kind : std::uint8_t { Header, Ple, GdnRecurrent, GdnConv, QsaTail, MtpSaved, MtpTail, QsaWindow };

    Kind kind           = Kind::Header;
    // The layer's index among GDN or attention layers; for a window plane, KV layer x 5 + plane;
    // 0 otherwise.
    std::uint32_t index = 0;
    std::size_t offset  = 0; // in the packed image
    std::size_t bytes   = 0;
};

// Group g < layers is decoder layer g; group `layers` is the MTP group (empty without MTP). The
// header (part 0) belongs to no group.
struct StateImageLayout {
    StateImageSpec spec;
    std::vector<StateImagePart> parts;
    std::vector<std::uint32_t> group_begin; // group g's parts are [group_begin[g], group_begin[g + 1])
    std::size_t image_bytes  = 0;
    std::uint64_t fingerprint = 0; // of every part's kind, index, offset and size

    [[nodiscard]] std::uint32_t groups() const noexcept {
        return static_cast<std::uint32_t>(group_begin.size()) - 1U;
    }
    [[nodiscard]] std::uint32_t mtp_group() const noexcept { return groups() - 1U; }
};

[[nodiscard]] StateImageLayout plan_state_image(const StateImageSpec& spec);

// Snapshot meta kept in the header part (written and read on the Host).
struct StateImageHeader {
    std::uint32_t frontier = 0;
    // MTP cell frontier - 1 is final, and the token it encodes (history[frontier]); -1 when none.
    bool mtp_written      = false;
    std::int32_t mtp_next = -1;
    // The capturing request resumed from an endpoint.
    bool lineage_echo = false;
    // The snapshot is the generation-opener tap (an exact split before the assistant turn).
    bool opener = false;
    std::array<double, 8> mtp_accept{};
};

void write_state_image_header(const StateImageLayout& layout, const StateImageHeader& header,
                              std::byte* image_begin);
// Absent when the bytes are not a header of this layout.
[[nodiscard]] std::optional<StateImageHeader> read_state_image_header(const StateImageLayout& layout,
                                                                      const std::byte* image_begin);

// Where the Program keeps every lane's state (program_impl.h).
struct LaneStateBuffers {
    const LinearAttentionStatePool* gdn = nullptr;
    std::uint32_t lanes                 = 0;
    std::byte* ple                      = nullptr; // lane l at ple + l * spec.ple_bytes
    // Tail slab of KV layer k (attention layers, then MTP) and lane l at
    // tails + (k * lanes + l) * spec.qsa_tail_bytes.
    std::byte* tails     = nullptr;
    std::byte* mtp_saved = nullptr; // lane l at mtp_saved + l * spec.mtp_saved_bytes
    // The exact window (vector-quantized storages): KvWindowGeometry's planes at `window`.
    std::byte* window = nullptr;
    KvWindowGeometry window_geometry;
};

// Host placement of one packed image.
struct HostImage {
    std::span<std::byte* const> segments;
    std::size_t segment_bytes = 0;
};

struct HostImageConst {
    std::span<const std::byte* const> segments;
    std::size_t segment_bytes = 0;
};

// Copies between a lane's state, packed Device slots and Host images. Every copy is enqueued on the
// caller's stream; the caller orders it against the compute stream and records any events.
class LaneStateImage {
public:
    // Checks the buffers against the layout's sizes.
    LaneStateImage(const StateImageLayout& layout, const LaneStateBuffers& buffers);

    [[nodiscard]] const StateImageLayout& layout() const noexcept { return *layout_; }
    // The lane's Device bytes of a part (none for the header).
    [[nodiscard]] std::byte* device_part(const StateImagePart& part, std::uint32_t lane) const;

    // Capture: every part but the header, or one group's parts.
    void copy_lane_to_host(std::uint32_t lane, const HostImage& image, cudaStream_t stream) const;
    void copy_group_to_host(std::uint32_t group, std::uint32_t lane, const HostImage& image, cudaStream_t stream) const;
    void copy_lane_to_packed(std::uint32_t lane, std::byte* packed, cudaStream_t stream) const;
    // Write-through of a packed Device slot (header included, as the slot holds it).
    void copy_packed_to_host(const std::byte* packed, const HostImage& image, cudaStream_t stream) const;
    // Restore of one group's parts.
    void copy_group_from_host(std::uint32_t group, const HostImageConst& image, std::uint32_t lane,
                              cudaStream_t stream) const;
    void copy_group_from_packed(std::uint32_t group, const std::byte* packed, std::uint32_t lane,
                                cudaStream_t stream) const;

private:
    void check_lane(std::uint32_t lane) const;
    void check_image(std::size_t segments, std::size_t segment_bytes) const;

    const StateImageLayout* layout_ = nullptr;
    LaneStateBuffers buffers_;
};

// The Program's KV page group: for every KV layer (the attention layers, then MTP's), K data, K
// scales, V data, V scales (scales only for formats that have them), then the pooled index keys.
// A cached block's page records move per KV layer, so a restore lands the planes each attention
// layer reads in forward order.
struct KvPageGeometry {
    KVPageGeometry geometry;
    std::uint32_t kv_layers           = 0;
    std::uint32_t planes_per_layer    = 0;
};

[[nodiscard]] KvPageGeometry kv_page_geometry(const TextConfig& config, KvCacheStorage storage, bool mtp);

struct PlaneRange {
    std::size_t begin = 0;
    std::size_t end   = 0;
};

// The planes of KV layer `kv_layer` (attention layer index, or attention_layers for MTP).
[[nodiscard]] PlaneRange kv_layer_planes(const KvPageGeometry& geometry, std::uint32_t kv_layer);

} // namespace ninfer::models::qwen4_exp::prefix
