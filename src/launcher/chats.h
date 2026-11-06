#pragma once

// The playground's chats: one JSON file per chat in `<data folder>/chats/`, so every browser sees
// them and they survive restarts. The launcher stores what the page sends (the web UI owns the
// shape, see api-types.ts Chat) after checking it is a chat object with a valid id.

#include "launcher/util.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>

namespace infernix::launcher {

class ChatStore {
public:
    static constexpr std::size_t kMaxChatBytes = 256u << 20; // images included

    explicit ChatStore(std::filesystem::path dir);

    // 1-64 characters of [A-Za-z0-9_-].
    [[nodiscard]] static bool valid_id(std::string_view id);

    // Summaries, newest first.
    [[nodiscard]] Json list() const;
    [[nodiscard]] std::optional<Json> get(const std::string& id) const;
    // Stores the chat (throws std::invalid_argument when it is not one); returns its summary.
    Json put(const std::string& id, Json chat);
    bool remove(const std::string& id);

private:
    // A file's size and write time: a summary is parsed again only when they change.
    struct Stamp {
        std::uintmax_t size = 0;
        std::filesystem::file_time_type written;
        bool operator==(const Stamp&) const = default;
    };
    struct Cached {
        Stamp stamp;
        Json summary;
    };

    [[nodiscard]] std::filesystem::path path_of(const std::string& id) const;
    [[nodiscard]] static Json summary(const Json& chat);
    [[nodiscard]] static std::optional<Stamp> stamp_of(const std::filesystem::path& path);

    std::filesystem::path dir_;
    mutable std::mutex mutex_;
    mutable std::map<std::filesystem::path, Cached> summaries_; // the last list's, and saves since
};

} // namespace infernix::launcher
