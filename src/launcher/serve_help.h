#pragma once

// infernix-serve's own `--help`, parsed into options (flag, argument, section, description): the
// configuration page's option reference, and which flags take a value.

#include "launcher/util.h"

#include <filesystem>
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

// Runs `executable --help` (cached by path and modification time).
[[nodiscard]] ServeOptions load_serve_options(const std::filesystem::path& executable);

[[nodiscard]] Json to_json(const ServeOptions& options, const std::optional<std::string>& build);

} // namespace infernix::launcher
