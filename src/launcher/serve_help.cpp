#include "launcher/serve_help.h"

#include "launcher/process.h"

#include <cctype>
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

ServeOptions load_serve_options(const std::filesystem::path& executable) {
    static std::mutex mutex;
    static std::string cached_key;
    static ServeOptions cached;
    const std::string key = path_utf8(executable) + "|" + std::to_string(modified_ms(executable));
    std::lock_guard lock(mutex);
    if (key == cached_key) { return cached; }
    ServeOptions result;
    result.executable = executable;
    std::error_code error;
    result.found = std::filesystem::is_regular_file(executable, error);
    if (!result.found) {
        result.error = "infernix-serve was not found at " + path_utf8(executable);
    } else if (const std::optional<CaptureResult> help =
                   run_capture(executable, {"--help"}, std::chrono::seconds(20))) {
        result.options = parse_serve_help(help->output);
        if (result.options.empty()) { result.error = "infernix-serve --help listed no options"; }
    } else {
        result.error = "infernix-serve --help did not run";
    }
    cached_key = key;
    cached     = result;
    return result;
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
