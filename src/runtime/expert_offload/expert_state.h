#pragma once

// The expert cache's saved state (design §19.3.5 S4b, warm start): every expert's LFRU count and the
// resident experts, highest score first, written at stop and between requests and read at the next
// start to fill the VRAM frames before the first request. A file is used only when its identity
// (the artifact and the key count) matches; anything else is ignored with a reason.

#include "runtime/expert_offload/expert_cache.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <cstdint>
#include <string_view>

namespace infernix::runtime::expert_offload::expert_cache {

struct ExpertStateLoad {
    std::optional<SavedState> state;
    std::string message; // why nothing was loaded
};

// Writes `state` to `path` through `path.tmp` and a rename; returns an error message, empty on success.
[[nodiscard]] std::string save_expert_state(const std::filesystem::path& path, std::string_view identity,
                                            const SavedState& state);

// Reads a state saved for `identity` with `keys` keys.
[[nodiscard]] ExpertStateLoad load_expert_state(const std::filesystem::path& path, std::string_view identity,
                                                std::uint32_t keys);

// A model's calibrated expert order as a saved state (S4b's seed without a saved state; Qwen3.8-Flash-Next's
// program/calibrated_order.h, the Qwen3.5-122B-A10B's models/qwen3_5/program/calibrated_order.h):
// `expert_at(layer, rank)` is the expert of `layer` at `rank` (0: the highest cache priority), each of the `layers`
// rows a permutation of [0, experts). Key = layer * experts + expert. Counts are the ranks reversed (experts - rank,
// a weak prior that real use soon outweighs), and the ranked list takes every layer's first expert, then every
// layer's second, and so on. Throws std::invalid_argument when a row is not a permutation.
[[nodiscard]] SavedState calibrated_order_state(std::uint32_t layers, std::uint32_t experts,
                                               const std::function<std::uint32_t(std::uint32_t, std::uint32_t)>& expert_at);

// Where a warm start's seed came from.
enum class WarmStartSource : std::uint8_t { None, Saved, Calibrated };
struct WarmStartSeed {
    WarmStartSource source = WarmStartSource::None;
    std::optional<SavedState> state; // the seed (Saved, Calibrated)
    std::string message;             // why no saved state was used (empty: one was, or no file was given)
};
// The warm start's seed: the state saved at `path` (empty: no file) whenever it is usable; else, unless seeding is
// off (`seed` false: --expert-state off), the model's calibrated order (`calibrated`; null when the model has none);
// else none.
[[nodiscard]] WarmStartSeed choose_warm_start(const std::filesystem::path& path, std::string_view identity,
                                              std::uint32_t keys, bool seed, const SavedState* calibrated);

} // namespace infernix::runtime::expert_offload::expert_cache
