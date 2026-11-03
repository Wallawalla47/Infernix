#include "launcher/chats.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace infernix::launcher {

ChatStore::ChatStore(std::filesystem::path dir) : dir_(std::move(dir)) {}

bool ChatStore::valid_id(std::string_view id) {
    return !id.empty() && id.size() <= 64 && std::all_of(id.begin(), id.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-';
    });
}

std::filesystem::path ChatStore::path_of(const std::string& id) const { return dir_ / (id + ".json"); }

Json ChatStore::summary(const Json& chat) {
    const Json messages = chat.value("messages", Json::array());
    return {{"id", chat.value("id", "")},
            {"title", chat.value("title", "")},
            {"model", chat.contains("model") ? chat["model"] : Json(nullptr)},
            {"messages", messages.is_array() ? messages.size() : 0},
            {"created_at", json_or<std::int64_t>(chat, "created_at", 0)},
            {"updated_at", json_or<std::int64_t>(chat, "updated_at", 0)}};
}

std::optional<ChatStore::Stamp> ChatStore::stamp_of(const std::filesystem::path& path) {
    std::error_code error;
    Stamp stamp;
    stamp.size = std::filesystem::file_size(path, error);
    if (error) { return std::nullopt; }
    stamp.written = std::filesystem::last_write_time(path, error);
    if (error) { return std::nullopt; }
    return stamp;
}

Json ChatStore::list() const {
    // Chats are written whole (renamed into place), so they are read without the lock, and only
    // the files changed since the last list are parsed: a slow list holds up no save.
    std::map<std::filesystem::path, Cached> current;
    std::vector<std::filesystem::path> changed;
    {
        std::lock_guard lock(mutex_);
        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(dir_, error)) {
            if (entry.path().extension() != ".json") { continue; }
            const std::optional<Stamp> stamp = stamp_of(entry.path());
            if (!stamp) { continue; }
            if (const auto it = summaries_.find(entry.path()); it != summaries_.end() && it->second.stamp == *stamp) {
                current.emplace(entry.path(), it->second);
            } else {
                changed.push_back(entry.path());
            }
        }
    }
    for (const std::filesystem::path& path : changed) {
        const std::optional<Stamp> stamp       = stamp_of(path);
        const std::optional<std::string> text = read_file(path);
        if (!stamp || !text) { continue; }
        try {
            const Json chat = Json::parse(*text);
            if (chat.is_object()) { current.emplace(path, Cached{*stamp, summary(chat)}); }
        } catch (const Json::exception&) {}
    }
    std::vector<Json> chats;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [path, cached] : current) {
            summaries_.insert_or_assign(path, cached);
            chats.push_back(cached.summary);
        }
        for (auto it = summaries_.begin(); it != summaries_.end();) {
            it = current.contains(it->first) ? std::next(it) : summaries_.erase(it);
        }
    }
    std::sort(chats.begin(), chats.end(), [](const Json& a, const Json& b) {
        return a.value("updated_at", std::int64_t{0}) > b.value("updated_at", std::int64_t{0});
    });
    return Json(chats);
}

std::optional<Json> ChatStore::get(const std::string& id) const {
    if (!valid_id(id)) { return std::nullopt; }
    const std::optional<std::string> text = read_file(path_of(id)); // written whole: no lock

    if (!text) { return std::nullopt; }
    try {
        return Json::parse(*text);
    } catch (const Json::exception&) { return std::nullopt; }
}

Json ChatStore::put(const std::string& id, Json chat) {
    if (!valid_id(id)) { throw std::invalid_argument("a chat id is 1-64 letters, digits, - or _"); }
    if (!chat.is_object() || !chat.contains("messages") || !chat["messages"].is_array()) {
        throw std::invalid_argument("a chat is an object with a messages array");
    }
    chat["id"] = id;
    if (!chat.contains("created_at")) { chat["created_at"] = now_ms(); }
    const std::string text = chat.dump(-1, ' ', false, Json::error_handler_t::replace);
    if (text.size() > kMaxChatBytes) { throw std::invalid_argument("the chat is larger than 256 MB"); }
    std::lock_guard lock(mutex_);
    std::error_code error;
    std::filesystem::create_directories(dir_, error);
    const std::filesystem::path path = path_of(id);
    write_file_atomic(path, text);
    Json brief = summary(chat);
    if (const std::optional<Stamp> stamp = stamp_of(path)) { summaries_.insert_or_assign(path, Cached{*stamp, brief}); }
    return brief;
}

bool ChatStore::remove(const std::string& id) {
    if (!valid_id(id)) { return false; }
    std::lock_guard lock(mutex_);
    std::error_code error;
    summaries_.erase(path_of(id));
    return std::filesystem::remove(path_of(id), error);
}

} // namespace infernix::launcher
