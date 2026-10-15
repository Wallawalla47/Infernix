#include "models/qwen4_exp/program/expert_cache/expert_profile.h"

#include <array>
#include <cstring>
#include <vector>

namespace infernix::models::qwen4_exp::expert_cache {
namespace {

// Layout (version 1), little-endian: magic, version, key count, counts[keys], ranked count,
// ranked keys. Same body as the saved-state file (expert_state.cpp) without the identity: the
// profile rides inside the artifact, which is its own binding.
constexpr std::array<char, 8> kMagic = {'N', 'I', 'N', 'F', 'Q', '4', 'E', 'P'};
constexpr std::uint32_t kVersion     = 1;
constexpr std::size_t kHeaderBytes   = kMagic.size() + 2 * sizeof(std::uint32_t);

template <class T>
bool get(const std::byte*& at, const std::byte* end, T& value) {
    if (static_cast<std::size_t>(end - at) < sizeof(T)) { return false; }
    std::memcpy(&value, at, sizeof(T));
    at += sizeof(T);
    return true;
}

} // namespace

ExpertProfileLoad parse_expert_profile(std::span<const std::byte> bytes, std::uint32_t keys) {
    ExpertProfileLoad out;
    if (bytes.size() < kHeaderBytes) {
        out.message = "the shipped expert profile is too short";
        return out;
    }
    const std::byte* at  = bytes.data();
    const std::byte* end = at + bytes.size();
    std::array<char, 8> magic{};
    std::memcpy(magic.data(), at, magic.size());
    at += magic.size();
    std::uint32_t version = 0, profile_keys = 0;
    if (magic != kMagic || !get(at, end, version) || !get(at, end, profile_keys)) {
        out.message = "not a shipped expert profile of this version";
        return out;
    }
    if (version != kVersion) {
        out.message = "unsupported shipped expert profile version " + std::to_string(version);
        return out;
    }
    if (profile_keys != keys) {
        out.message = "the shipped expert profile has another expert count";
        return out;
    }
    SavedState state;
    state.counts.resize(keys);
    if (static_cast<std::size_t>(end - at) < state.counts.size() * sizeof(std::uint32_t)) {
        out.message = "the shipped expert profile is truncated";
        return out;
    }
    std::memcpy(state.counts.data(), at, state.counts.size() * sizeof(std::uint32_t));
    at += state.counts.size() * sizeof(std::uint32_t);
    std::uint32_t ranked = 0;
    if (!get(at, end, ranked) || ranked > keys ||
        static_cast<std::size_t>(end - at) < static_cast<std::size_t>(ranked) * sizeof(std::uint32_t)) {
        out.message = "the shipped expert profile is damaged";
        return out;
    }
    state.ranked.resize(ranked);
    std::memcpy(state.ranked.data(), at, static_cast<std::size_t>(ranked) * sizeof(std::uint32_t));
    at += static_cast<std::size_t>(ranked) * sizeof(std::uint32_t);
    if (at != end) {
        out.message = "the shipped expert profile has trailing bytes";
        return out;
    }
    std::vector<std::uint8_t> listed(keys, 0);
    for (const std::uint32_t key : state.ranked) {
        if (key >= keys || listed[key] != 0) {
            out.message = "the shipped expert profile is damaged";
            return out;
        }
        listed[key] = 1;
    }
    out.profile = std::move(state);
    return out;
}

bool apply_shipped_profile(ExpertStateLoad& loaded, const std::optional<SavedState>& profile) {
    if (loaded.state || !profile) { return false; }
    loaded.state   = *profile;
    loaded.shipped = true;
    // The source is the profile, but the reason the sidecar gave way stays readable.
    const std::string reason = std::move(loaded.message);
    loaded.message = reason.empty()
                         ? "the artifact's shipped expert profile"
                         : "the artifact's shipped expert profile, after: " + reason;
    return true;
}

ExpertStateLoad warm_start_load(const std::filesystem::path& state_path, bool expert_cache,
                                std::string_view identity, std::uint32_t keys,
                                const std::optional<SavedState>& profile) {
    ExpertStateLoad load;
    if (!state_path.empty() && expert_cache) {
        load = load_expert_state(state_path, identity, keys);
        // The runtime-learned sidecar wins; the shipped profile seeds only a cold first run
        // (missing or refused file). apply_shipped_profile records the source for the logs.
        apply_shipped_profile(load, profile);
    }
    return load;
}

} // namespace infernix::models::qwen4_exp::expert_cache
