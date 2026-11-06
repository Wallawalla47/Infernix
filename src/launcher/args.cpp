#include "launcher/args.h"

#include <algorithm>
#include <charconv>
#include <cstdint>

namespace infernix::launcher {
namespace {

bool is_flag_token(std::string_view token) {
    return token.size() > 2 && token.starts_with("--");
}

bool matches(std::string_view token, std::string_view flag) {
    return token == flag || (token.size() > flag.size() && token.starts_with(flag) &&
                             token[flag.size()] == '=');
}

bool next_is_value(const Args& args, std::size_t index, std::string_view flag,
                   const TakesValue& takes_value) {
    if (index + 1 >= args.size()) { return false; }
    if (takes_value) {
        if (const std::optional<bool> known = takes_value(flag)) { return *known; }
    }
    return !is_flag_token(args[index + 1]);
}

} // namespace

std::size_t find_flag(const Args& args, std::string_view flag) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (matches(args[i], flag)) { return i; }
    }
    return std::string::npos;
}

bool has_flag(const Args& args, std::string_view flag) {
    return find_flag(args, flag) != std::string::npos;
}

std::optional<std::string> flag_value(const Args& args, std::string_view flag) {
    const std::size_t index = find_flag(args, flag);
    if (index == std::string::npos) { return std::nullopt; }
    const std::string& token = args[index];
    if (token.size() > flag.size()) { return token.substr(flag.size() + 1); }
    if (index + 1 < args.size() && !is_flag_token(args[index + 1])) { return args[index + 1]; }
    return std::nullopt;
}

void set_flag(Args& args, std::string_view flag, std::string_view value) {
    const std::size_t index = find_flag(args, flag);
    if (index == std::string::npos) {
        args.emplace_back(flag);
        args.emplace_back(value);
        return;
    }
    // Keep the first occurrence's position (with its old value token, if separate); drop any later
    // duplicates. The flag takes a value, so a non-flag token right after a bare `--flag` is it.
    std::size_t after = index + 1;
    if (args[index] == flag && after < args.size() && !is_flag_token(args[after])) { ++after; }
    Args tail(args.begin() + static_cast<std::ptrdiff_t>(after), args.end());
    remove_flag(tail, flag, [](std::string_view) -> std::optional<bool> { return true; });
    args.resize(index);
    args.emplace_back(flag);
    args.emplace_back(value);
    args.insert(args.end(), tail.begin(), tail.end());
}

void set_switch(Args& args, std::string_view flag) {
    if (!has_flag(args, flag)) { args.emplace_back(flag); }
}

void remove_flag(Args& args, std::string_view flag, const TakesValue& takes_value) {
    Args out;
    out.reserve(args.size());
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (!matches(args[i], flag)) {
            out.push_back(args[i]);
            continue;
        }
        if (args[i] == flag && next_is_value(args, i, flag, takes_value)) { ++i; }
    }
    args = std::move(out);
}

const std::vector<std::string_view>& managed_flags() {
    static const std::vector<std::string_view> kFlags = {
        "--host",          "--port",          "--model-id",    "--any-model-name",
        "--log-stats-panel", "--log-colours", "--config",      "--api-key",
        "--cors",          "--help",
    };
    return kFlags;
}

std::vector<std::string> strip_managed_flags(Args& args, const TakesValue& takes_value) {
    static const std::vector<std::string_view> kSwitches = {"--any-model-name", "--cors", "--help"};
    std::vector<std::string> removed;
    for (std::string_view flag : managed_flags()) {
        if (!has_flag(args, flag)) { continue; }
        removed.emplace_back(flag);
        const bool is_switch = std::find(kSwitches.begin(), kSwitches.end(), flag) != kSwitches.end();
        remove_flag(args, flag, [&](std::string_view f) -> std::optional<bool> {
            if (f == flag) { return !is_switch; }
            return takes_value ? takes_value(f) : std::nullopt;
        });
    }
    return removed;
}

std::uint64_t context_length(const Args& args) {
    constexpr std::uint64_t kServeDefault = 8192;
    const std::optional<std::string> value = flag_value(args, "--max-context");
    if (!value) { return kServeDefault; }
    std::uint64_t parsed = 0;
    const auto [end, error] = std::from_chars(value->data(), value->data() + value->size(), parsed);
    if (error != std::errc{} || end != value->data() + value->size() || parsed == 0) {
        return kServeDefault;
    }
    return parsed;
}

std::string windows_quote(std::string_view arg) {
    if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string_view::npos) {
        return std::string(arg);
    }
    return windows_quoted(arg);
}

std::string batch_command_line(const std::vector<std::string>& argv) {
    constexpr std::string_view kMetacharacters = "&|<>^()";
    std::string out;
    bool quoted = false; // cmd's own quote state, which every " on the line toggles (an escaped one too)
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i != 0) { out.push_back(' '); }
        const bool special = argv[i].find_first_of(kMetacharacters) != std::string::npos;
        for (const char c : special ? windows_quoted(argv[i]) : windows_quote(argv[i])) {
            if (c == '"') {
                quoted = !quoted;
            } else if (!quoted && kMetacharacters.find(c) != std::string_view::npos) {
                out.push_back('^');
            }
            out.push_back(c);
        }
    }
    return out;
}

std::string windows_quoted(std::string_view arg) {
    std::string out = "\"";
    std::size_t backslashes = 0;
    for (const char c : arg) {
        if (c == '\\') {
            ++backslashes;
            continue;
        }
        if (c == '"') {
            out.append(backslashes * 2 + 1, '\\');
        } else {
            out.append(backslashes, '\\');
        }
        backslashes = 0;
        out.push_back(c);
    }
    out.append(backslashes * 2, '\\');
    out.push_back('"');
    return out;
}

std::string shell_quote(std::string_view arg) {
#ifdef _WIN32
    return windows_quote(arg);
#else
    if (!arg.empty() &&
        arg.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./=:,+@%") ==
            std::string_view::npos) {
        return std::string(arg);
    }
    std::string out = "'";
    for (const char c : arg) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
#endif
}

std::string command_line_for_display(const std::vector<std::string>& argv) {
    std::string out;
    for (const std::string& arg : argv) {
        if (!out.empty()) { out.push_back(' '); }
        out += shell_quote(arg);
    }
    return out;
}

} // namespace infernix::launcher
