#pragma once

// The Vision encode window of a Qwen4Exp lane (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.2, "The encode window and its memory"): the pure planning the Program and its host tests
// share. Items to encode are those ending past the reused prefix; one layer-major pass encodes
// them all into one token-ordered handoff, BF16 [out, V_encoded], which lives in lent expert
// frames until the lane's prefill ends. Placement W runs the whole pass from the prefill
// workspace in one admission step; placement L lends one run of frames for handoff and pass and
// runs the pass in steps of several layers, so other lanes keep decoding between them.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp {

// Design estimates until the tower is measured (VM2): tensor-core GEMM and attention rates, and the
// longest step one admission call may take (other lanes wait for it; VM7 tunes it).
inline constexpr double kVisionGemmFlops       = 1.0e14;
inline constexpr double kVisionAttentionFlops  = 6.0e13;
inline constexpr double kVisionStepSeconds     = 0.25;
inline constexpr double kVisionGemmFlopsPerPatch = 842.0e6; // 27 layers, patch embedding, merger

struct VisionWindowItem {
    std::uint32_t token_begin = 0, token_end = 0; // the item's visual tokens [begin, end) of the prompt
    std::uint32_t patches     = 0;                // P_i
    std::uint32_t merged      = 0;                // V_i = P_i / merge^2
};

struct VisionWindowPlan {
    std::vector<std::uint32_t> items; // indices of the encoded items (token_end > reused)
    std::uint32_t encoded_tokens = 0; // V_encoded
    std::size_t handoff_bytes    = 0; // 2 * out * V_encoded
    std::size_t window_bytes     = 0; // weight staging (offload) + pass arena
    double seconds               = 0; // estimated tower time
    bool lease                   = false; // placement L
    std::uint32_t frames         = 0;     // frames lent: the handoff (W) or handoff + window (L)
    std::uint32_t steps          = 0;     // admission steps (W: 1)
    std::uint32_t layers_per_step = 0;    // L: encoder layers per step
};

// Plans the window over `items` (prompt order) past `reused` tokens. `pass_bytes` is the pass arena
// of the encoded items, `staging_bytes` the weight staging (0 when resident), `work_capacity` the
// prefill workspace, `frame_bytes` the expert frame stride, `out` the tower's output width,
// `hidden` its width and `depth` its layers.
[[nodiscard]] VisionWindowPlan plan_vision_window(std::span<const VisionWindowItem> items, std::uint32_t reused,
                                                  std::size_t pass_bytes, std::size_t staging_bytes,
                                                  std::size_t work_capacity, std::uint64_t frame_bytes,
                                                  std::uint32_t out, std::uint32_t hidden, std::uint32_t depth);

// The estimated tower time of items with these patch counts.
[[nodiscard]] double vision_seconds(std::span<const std::uint32_t> patches, std::uint32_t hidden);

// The visual tokens of a lane: every encoded item's visual token positions, ascending; handoff
// column c holds the embedding of token visual[c].
struct VisualSlice {
    std::uint32_t first = 0, count = 0; // handoff columns [first, first + count)
};

// The handoff columns of the visual tokens in [a, b), and their columns relative to `base` written
// to `columns` (count entries).
VisualSlice visual_slice(std::span<const std::uint32_t> visual, std::uint32_t a, std::uint32_t b,
                         std::uint32_t base, std::span<std::int32_t> columns);

} // namespace infernix::models::qwen4_exp
