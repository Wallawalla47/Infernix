#include "launcher/serve_help.h"

#include "launcher/process.h"

#include <cctype>
#include <exception>
#include <mutex>
#include <sstream>

namespace infernix::launcher {
namespace {

bool is_section(std::string_view line) {
    if (line.empty() || line.front() == ' ') { return false; }
    bool letters = false;
    for (const char c : line) {
        if (std::islower(static_cast<unsigned char>(c))) { return false; }
        if (std::isupper(static_cast<unsigned char>(c))) { letters = true; }
    }
    return letters;
}

std::size_t indent(std::string_view line) {
    std::size_t n = 0;
    while (n < line.size() && line[n] == ' ') { ++n; }
    return n;
}

} // namespace

std::vector<ServeOption> parse_serve_help(std::string_view help) {
    std::vector<ServeOption> options;
    std::string section = "GENERAL";
    std::istringstream lines{std::string(help)};
    std::string raw;
    ServeOption* current = nullptr;
    while (std::getline(lines, raw)) {
        if (!raw.empty() && raw.back() == '\r') { raw.pop_back(); }
        const std::string_view line(raw);
        if (trim(line).empty()) {
            current = nullptr;
            continue;
        }
        if (is_section(line)) {
            section = trim(line);
            current = nullptr;
            continue;
        }
        const std::size_t lead = indent(line);
        if (lead >= 2 && lead <= 4 && line.substr(lead).starts_with("--")) {
            // "--flag[, -x] [ARG]   description": the flag part ends at two spaces.
            const std::string_view rest = line.substr(lead);
            const std::size_t gap       = rest.find("  ");
            const std::string head(rest.substr(0, gap));
            const std::string description = gap == std::string_view::npos ? std::string{} : trim(rest.substr(gap));
            ServeOption option;
            option.section = section;
            const std::size_t space = head.find(' ');
            std::string flag        = head.substr(0, space);
            if (!flag.empty() && flag.back() == ',') { flag.pop_back(); }
            option.flag = flag;
            if (space != std::string::npos) {
                std::string arg = trim(head.substr(space));
                // "-h" aliases after a comma are not arguments.
                if (arg.starts_with("-")) { arg.clear(); }
                if (!arg.empty()) { option.arg = arg; }
            }
            option.description = description;
            options.push_back(std::move(option));
            current = &options.back();
            continue;
        }
        if (current != nullptr && lead > 4) {
            if (!current->description.empty()) { current->description += ' '; }
            current->description += trim(line);
        }
    }
    return options;
}

std::optional<bool> ServeOptions::takes_value(std::string_view flag) const {
    for (const ServeOption& option : options) {
        if (option.flag == flag) { return option.arg.has_value(); }
    }
    return std::nullopt;
}

bool ServeOptions::known(std::string_view flag) const {
    for (const ServeOption& option : options) {
        if (option.flag == flag) { return true; }
    }
    return false;
}

bool ServeOptions::lists_value(std::string_view flag, std::string_view value) const {
    for (const ServeOption& option : options) {
        if (option.flag != flag || !option.arg) { continue; }
        std::string_view choices = *option.arg;
        while (!choices.empty()) {
            const std::size_t bar = choices.find('|');
            if (choices.substr(0, bar) == value) { return true; }
            if (bar == std::string_view::npos) { break; }
            choices.remove_prefix(bar + 1);
        }
    }
    return false;
}

ServeHelpCache::ServeHelpCache(Capture capture) : capture_(std::move(capture)) {}

ServeOptions ServeHelpCache::capture_options(const std::filesystem::path& executable) const {
    ServeOptions result;
    result.executable = executable;
    std::error_code error;
    result.found = std::filesystem::is_regular_file(executable, error);
    if (!result.found) {
        result.error = "infernix-serve was not found at " + path_utf8(executable);
    } else if (const std::optional<std::string> help = capture_(executable)) {
        result.options = parse_serve_help(*help);
        if (result.options.empty()) { result.error = "infernix-serve --help listed no options"; }
    } else {
        result.error = "infernix-serve --help did not run";
    }
    return result;
}

ServeOptions ServeHelpCache::get(const std::filesystem::path& executable) {
    const std::string key = path_utf8(executable) + "|" + std::to_string(modified_ms(executable));
    std::optional<std::promise<ServeOptions>> promise;
    std::shared_future<ServeOptions> options;
    std::uint64_t number = 0;
    {
        std::lock_guard lock(mutex_);
        if (key != key_ || !options_.valid()) {
            promise.emplace();
            key_     = key;
            options_ = promise->get_future().share();
            number   = ++capture_number_;
        }
        options = options_;
    }
    if (!promise) { return options.get(); }
    // This call captures. A failure is forgotten (unless a later capture replaced it) before the
    // callers waiting on it see it, so the next call tries again.
    const auto forget = [&] {
        std::lock_guard lock(mutex_);
        if (capture_number_ == number) { options_ = std::shared_future<ServeOptions>(); }
    };
    ServeOptions result;
    try {
        result = capture_options(executable);
    } catch (...) {
        forget();
        promise->set_exception(std::current_exception());
        throw;
    }
    if (result.options.empty()) { forget(); }
    promise->set_value(result);
    return result;
}

ServeOptions load_serve_options(const std::filesystem::path& executable) {
    static ServeHelpCache cache([](const std::filesystem::path& path) -> std::optional<std::string> {
        const std::optional<CaptureResult> help = run_capture(path, {"--help"}, std::chrono::seconds(20));
        if (!help) { return std::nullopt; }
        return help->output;
    });
    return cache.get(executable);
}

Json to_json(const ServeOptions& options, const std::optional<std::string>& build) {
    Json list = Json::array();
    for (const ServeOption& option : options.options) {
        list.push_back({{"flag", option.flag},
                        {"arg", option.arg ? Json(*option.arg) : Json(nullptr)},
                        {"section", option.section},
                        {"description", option.description}});
    }
    return {{"executable", path_utf8(options.executable)},
            {"found", options.found},
            {"error", options.error ? Json(*options.error) : Json(nullptr)},
            {"build", build ? Json(*build) : Json(nullptr)},
            {"options", list}};
}

} // namespace infernix::launcher
