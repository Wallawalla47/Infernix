#pragma once

// infernix-serve's own `--help`, parsed into options (flag, argument, section, description): the
// configuration page's option reference, and which flags take a value.

#include "launcher/util.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::launcher {

struct ServeOption {
    std::string flag;
    std::optional<std::string> arg;
    std::string section;
    std::string description;
};

[[nodiscard]] std::vector<ServeOption> parse_serve_help(std::string_view help);

struct ServeOptions {
    std::filesystem::path executable;
    bool found = false;
    std::optional<std::string> error;
    std::vector<ServeOption> options;

    // Whether `flag` takes a value; nullopt when the help does not list it.
    [[nodiscard]] std::optional<bool> takes_value(std::string_view flag) const;
    [[nodiscard]] bool known(std::string_view flag) const;
    // Whether `flag`'s argument names `value` among its "a|b|c" choices.
    [[nodiscard]] bool lists_value(std::string_view flag, std::string_view value) const;
};

// The parsed `--help` per executable path and modification time. The capture runs outside the
// cache's lock, and callers asking for the same executable meanwhile share its result. Only a help
// that lists options is kept: a run that did not start, timed out or listed nothing is tried again
// by the next call.
class ServeHelpCache {
public:
    // The executable's `--help` output; nullopt when it did not run.
    using Capture = std::function<std::optional<std::string>(const std::filesystem::path& executable)>;

    explicit ServeHelpCache(Capture capture);
    [[nodiscard]] ServeOptions get(const std::filesystem::path& executable);

private:
    [[nodiscard]] ServeOptions capture_options(const std::filesystem::path& executable) const;

    Capture capture_;
    std::mutex mutex_;
    std::string key_;
    std::shared_future<ServeOptions> options_; // valid while key_'s capture runs or after it listed options
    std::uint64_t capture_number_ = 0;         // of the capture options_ belongs to
};

// Runs `executable --help` through the launcher's one ServeHelpCache. A caller may wait for the
// capture (up to 20 s), so none calls it holding a lock other requests need.
[[nodiscard]] ServeOptions load_serve_options(const std::filesystem::path& executable);

[[nodiscard]] Json to_json(const ServeOptions& options, const std::optional<std::string>& build);

} // namespace infernix::launcher
