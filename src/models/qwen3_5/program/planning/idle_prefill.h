#pragma once

// idle-wide-prefill (FastPath::IdleWidePrefill): the width of a staged prefill step taken while no
// other lane waits to decode. The rules here are pure so that startup planning, the Program and the
// tests share them.

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace infernix::models::qwen3_5::detail {

// Idle widths startup plans, widest first. Each is taken as a prefill step takes the chunk
// (clamped to the context and rounded down to whole prompt-attention waves); startup keeps the
// widest whose plan fits memory (runtime/engine/model_instance.cpp). On the RTX 5090 a 4096-token
// step lifts prefill 2-7 % on the dense NVFP4 27B and 19-21 % on the 35B MoE over 1024 (NInfer
// 5bcef9e2c, int8 KV), while a decode round waiting behind one step would stall about 3.4x longer.
inline constexpr std::array<std::uint32_t, 3> kIdlePrefillRequestedWidths{4096U, 2048U, 1024U};

// The candidate idle widths, widest first, from `step_widths` (the requested widths as a prefill
// step takes them). Each is rounded down to a multiple of `chunk`, so a wide step from a cursor
// ends on a boundary a chunk-wide prefill from the same cursor also reaches; only widths wider
// than the chunk remain, each once.
[[nodiscard]] inline std::vector<std::uint32_t>
idle_prefill_width_ladder(std::uint32_t chunk, std::span<const std::uint32_t> step_widths) {
    std::vector<std::uint32_t> out;
    if (chunk == 0) { return out; }
    for (const std::uint32_t width : step_widths) {
        const std::uint32_t aligned = width / chunk * chunk;
        if (aligned > chunk && std::find(out.begin(), out.end(), aligned) == out.end()) {
            out.push_back(aligned);
        }
    }
    std::sort(out.begin(), out.end(), std::greater<>());
    return out;
}

// The tokens a prefill step may take from `cursor` (at most `tokens`) so that it ends no later
// than the chunk boundary where a chunk-wide prefill realizes a flexible prefix-cache tap at
// `tap_position`. A chunk-wide prefill visits the boundaries cursor + k * chunk (k >= 1) below the
// prompt end and realizes the tap at the first one that reaches it, or at the start of the prompt's
// final chunk (hybrid_after_prefill_chunk). When no such boundary lies below the prompt end the
// step is not limited.
[[nodiscard]] constexpr std::uint32_t flexible_tap_step_limit(std::uint32_t cursor,
                                                              std::uint32_t prompt_tokens,
                                                              std::uint32_t chunk,
                                                              std::uint32_t tap_position,
                                                              std::uint32_t tokens) noexcept {
    if (chunk == 0 || cursor >= prompt_tokens) { return tokens; }
    const std::uint64_t remaining = prompt_tokens - cursor;
    // First boundary at or past the tap.
    const std::uint64_t reach =
        tap_position > cursor ? (static_cast<std::uint64_t>(tap_position - cursor) + chunk - 1U) /
                                    chunk
                              : 1U;
    // First boundary from which the rest of the prompt is one chunk or less:
    // ceil((remaining - chunk) / chunk).
    const std::uint64_t final_chunk = remaining > chunk ? (remaining - 1U) / chunk : 1U;
    const std::uint64_t boundaries = std::max<std::uint64_t>(1U, std::min(reach, final_chunk));
    const std::uint64_t limit      = boundaries * chunk;
    if (limit >= remaining) { return tokens; }
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(tokens, limit));
}

} // namespace infernix::models::qwen3_5::detail
