#include "launcher/chats.h"

#include <algorithm>
#include <cctype>
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

Json ChatStore::list() const {
    std::lock_guard lock(mutex_);
    std::vector<Json> chats;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(dir_, error)) {
        if (entry.path().extension() != ".json") { continue; }
        const std::optional<std::string> text = read_file(entry.path());
        if (!text) { continue; }
        try {
            const Json chat = Json::parse(*text);
            if (chat.is_object()) { chats.push_back(summary(chat)); }
        } catch (const Json::exception&) {}
    }
    std::sort(chats.begin(), chats.end(), [](const Json& a, const Json& b) {
        return a.value("updated_at", std::int64_t{0}) > b.value("updated_at", std::int64_t{0});
    });
    return Json(chats);
}

std::optional<Json> ChatStore::get(const std::string& id) const {
    if (!valid_id(id)) { return std::nullopt; }
    std::lock_guard lock(mutex_);
    const std::optional<std::string> text = read_file(path_of(id));
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
    write_file_atomic(path_of(id), text);
    return summary(chat);
}

bool ChatStore::remove(const std::string& id) {
    if (!valid_id(id)) { return false; }
    std::lock_guard lock(mutex_);
    std::error_code error;
    return std::filesystem::remove(path_of(id), error);
}

} // namespace infernix::launcher
