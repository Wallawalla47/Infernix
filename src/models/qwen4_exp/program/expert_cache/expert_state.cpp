#include "models/qwen4_exp/program/expert_cache/expert_state.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace ninfer::models::qwen4_exp::expert_cache {
namespace {

// Layout (version 1): magic, version, identity length and bytes, key count, counts[keys], ranked
// count, ranked keys, then the magic again as a footer.
constexpr std::array<char, 8> kMagic = {'N', 'I', 'N', 'F', 'Q', '4', 'E', 'S'};
constexpr std::uint32_t kVersion     = 1;

template <class T>
void put(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <class T>
bool get(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

} // namespace

std::string save_expert_state(const std::filesystem::path& path, std::string_view identity,
                              const SavedState& state) {
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) { return "cannot create " + temporary.string(); }
        out.write(kMagic.data(), kMagic.size());
        put(out, kVersion);
        put(out, static_cast<std::uint32_t>(identity.size()));
        out.write(identity.data(), static_cast<std::streamsize>(identity.size()));
        put(out, static_cast<std::uint32_t>(state.counts.size()));
        out.write(reinterpret_cast<const char*>(state.counts.data()),
                  static_cast<std::streamsize>(state.counts.size() * sizeof(std::uint32_t)));
        put(out, static_cast<std::uint32_t>(state.ranked.size()));
        out.write(reinterpret_cast<const char*>(state.ranked.data()),
                  static_cast<std::streamsize>(state.ranked.size() * sizeof(std::uint32_t)));
        out.write(kMagic.data(), kMagic.size());
        out.flush();
        if (!out) {
            out.close();
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return "writing " + temporary.string() + " failed";
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return "cannot replace " + path.string();
    }
    return {};
}

ExpertStateLoad load_expert_state(const std::filesystem::path& path, std::string_view identity, std::uint32_t keys) {
    ExpertStateLoad out;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        out.message = "no saved expert state at " + path.string() + " yet";
        return out;
    }
    std::array<char, 8> magic{};
    std::uint32_t version = 0, identity_bytes = 0, count_keys = 0, ranked = 0;
    in.read(magic.data(), magic.size());
    if (!in || magic != kMagic || !get(in, version) || version != kVersion) {
        out.message = "not an expert state file of this version";
        return out;
    }
    if (!get(in, identity_bytes) || identity_bytes > (1U << 16)) {
        out.message = "the expert state file is damaged";
        return out;
    }
    std::string saved(identity_bytes, '\0');
    in.read(saved.data(), identity_bytes);
    if (!in || saved != identity) {
        out.message = "the saved expert state belongs to another artifact";
        return out;
    }
    if (!get(in, count_keys) || count_keys != keys) {
        out.message = "the saved expert state has another expert count";
        return out;
    }
    SavedState state;
    state.counts.resize(keys);
    in.read(reinterpret_cast<char*>(state.counts.data()), static_cast<std::streamsize>(keys * sizeof(std::uint32_t)));
    if (!in || !get(in, ranked) || ranked > keys) {
        out.message = "the expert state file is damaged";
        return out;
    }
    state.ranked.resize(ranked);
    in.read(reinterpret_cast<char*>(state.ranked.data()), static_cast<std::streamsize>(ranked * sizeof(std::uint32_t)));
    in.read(magic.data(), magic.size());
    if (!in || magic != kMagic) {
        out.message = "the expert state file is truncated";
        return out;
    }
    for (const std::uint32_t key : state.ranked) {
        if (key >= keys) {
            out.message = "the expert state file is damaged";
            return out;
        }
    }
    out.state = std::move(state);
    return out;
}

} // namespace ninfer::models::qwen4_exp::expert_cache
