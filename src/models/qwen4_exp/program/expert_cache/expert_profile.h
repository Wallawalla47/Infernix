#pragma once

// The artifact's shipped expert routing profile (design §9.3 item 4): a converter- or
// calibration-produced frequency ranking stored as one resource object in the pack (role
// `expert_profile` of the text component, artifact-container.md §9.3) that seeds the same warm-start
// pipeline (design §19.3.5 S4b) the saved-state sidecar feeds, so a cold first run places experts
// like a warmed one. The bytes are the saved state's semantics without identity: the pack carries
// them, so the artifact itself binds them. A profile the model cannot use is refused with a reason;
// the cache then starts as it does today.

#include "models/qwen4_exp/program/expert_cache/expert_state.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace infernix::models::qwen4_exp::expert_cache {

// The text component's resource role that carries the profile (artifact-container.md §9.3).
inline constexpr std::string_view kExpertProfileRole = "expert_profile";

struct ExpertProfileLoad {
    std::optional<SavedState> profile;
    std::string message; // why nothing was parsed (empty: profile parsed, or none shipped)
};

// Parses profile bytes against `keys` (layers * experts): the shape must match exactly, an unknown
// version is refused, and a damaged body (length, ranking bounds or duplicates) is refused.
[[nodiscard]] ExpertProfileLoad parse_expert_profile(std::span<const std::byte> bytes, std::uint32_t keys);

// Warm-start source choice: the runtime-learned sidecar state wins; the shipped profile seeds only
// when `loaded` carries no state (missing or refused file), and then records its source, keeping the
// sidecar's refusal reason. Returns whether the profile became the seed. Callers gate this on the
// warm start being enabled at all (--expert-state off seeds nothing).
bool apply_shipped_profile(ExpertStateLoad& loaded, const std::optional<SavedState>& profile);

// The S4b warm start's source decision, exactly as Program::warm_start_experts applies it: with the
// warm start enabled (a state file and the expert cache), load the sidecar and fall back to the
// artifact's shipped profile; with it disabled, no source at all. Kept out of the Program so the
// precedence and the gating are testable without a device.
[[nodiscard]] ExpertStateLoad warm_start_load(const std::filesystem::path& state_path, bool expert_cache,
                                              std::string_view identity, std::uint32_t keys,
                                              const std::optional<SavedState>& profile);

} // namespace infernix::models::qwen4_exp::expert_cache
