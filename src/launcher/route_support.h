#pragma once

// HTTP helpers the launcher's route files share (launcher.cpp, launch_plan.cpp, public_api.cpp,
// management_routes.cpp): JSON answers and error bodies, Claude Code's model ids, the session cookie,
// tokens and the folder containment check.

#include "launcher/util.h"

#include <httplib.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <string_view>

namespace infernix::launcher {

using Request  = httplib::Request;
using Response = httplib::Response;

inline void send_json(Response& response, const Json& body, int status = 200) {
    response.status = status;
    response.set_header("Cache-Control", "no-store");
    response.set_content(body.dump(-1, ' ', false, Json::error_handler_t::replace), "application/json");
}

inline void send_error(Response& response, int status, const std::string& code, const std::string& message) {
    send_json(response, {{"error", {{"message", message}, {"code", code}}}}, status);
}

inline void openai_error(Response& response, int status, const std::string& type, const std::string& code,
                  const std::string& message) {
    send_json(response, {{"error", {{"message", message}, {"type", type}, {"param", nullptr}, {"code", code}}}}, status);
}

inline void anthropic_error(Response& response, int status, const std::string& type, const std::string& message) {
    send_json(response, {{"type", "error"}, {"error", {{"type", type}, {"message", message}}}}, status);
}

// llama-server's error body: the HTTP status as `code`.
inline void llamacpp_error(Response& response, int status, const std::string& type, const std::string& message) {
    send_json(response, {{"error", {{"code", status}, {"message", message}, {"type", type}}}}, status);
}

// Claude Code's gateway model discovery lists only ids that contain "claude" or "anthropic", so it
// is offered each model as "<model id>:claude-code", which requests may name too. A suffix, unlike
// a "claude-" prefix, keeps Claude Code's CLAUDE_CODE_MAX_CONTEXT_TOKENS in effect for the id.
inline constexpr std::string_view kClaudeCodeSuffix = ":claude-code";

inline bool is_claude_code_client(const httplib::Request& request) {
    const std::string agent = lower(request.get_header_value("User-Agent"));
    return agent.starts_with("claude-cli/") || agent.starts_with("claude-code/");
}

// The model id inside Claude Code's spelling of it; nullopt for any other name.
inline std::optional<std::string> strip_claude_code(std::string_view name) {
    if (name.size() <= kClaudeCodeSuffix.size() ||
        !iequals(name.substr(name.size() - kClaudeCodeSuffix.size()), kClaudeCodeSuffix)) {
        return std::nullopt;
    }
    return std::string(name.substr(0, name.size() - kClaudeCodeSuffix.size()));
}

// The session cookie's value; "" for none.
inline std::string session_cookie(const httplib::Request& request) {
    const std::string cookies = request.get_header_value("Cookie");
    std::size_t at            = 0;
    while (at < cookies.size()) {
        std::size_t end = cookies.find(';', at);
        if (end == std::string::npos) { end = cookies.size(); }
        const std::string pair = trim(std::string_view(cookies).substr(at, end - at));
        if (pair.starts_with("infernix_session=")) { return pair.substr(17); }
        at = end + 1;
    }
    return {};
}

// Random letters and digits from the system's random source.
inline std::string random_token(std::size_t length) {
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    std::random_device device;
    std::uniform_int_distribution<int> pick(0, static_cast<int>(sizeof(kAlphabet)) - 2);
    std::string token;
    for (std::size_t i = 0; i < length; ++i) { token.push_back(kAlphabet[pick(device)]); }
    return token;
}

inline bool same_or_inside(const std::filesystem::path& inner, const std::filesystem::path& outer, int depth) {
    const std::string a = lower(path_utf8(inner.lexically_normal()));
    std::string b       = lower(path_utf8(outer.lexically_normal()));
    while (!b.empty() && (b.back() == '\\' || b.back() == '/')) { b.pop_back(); }
    if (a == b) { return true; }
    if (!a.starts_with(b) || a.size() <= b.size() || (a[b.size()] != '\\' && a[b.size()] != '/')) { return false; }
    const std::string rest = a.substr(b.size() + 1);
    const auto levels      = std::count_if(rest.begin(), rest.end(), [](char c) { return c == '\\' || c == '/'; });
    return levels < depth;
}

} // namespace infernix::launcher
