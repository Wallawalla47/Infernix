#include "product/config/config_file.h"

#include <toml.hpp>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace infernix::product {
namespace {

// One key that applies to the program, with the node that gives its value.
struct Setting {
    std::string key;
    const toml::node* value = nullptr;
    toml::source_position at{};
};

class FileReader {
public:
    FileReader(const std::string& path, const ConfigProgram& program)
        : path_(path), program_(program) {}

    ConfigFile read() {
        std::ifstream stream(std::filesystem::path(path_), std::ios::binary);
        if (!stream) { throw std::invalid_argument("--config " + path_ + ": cannot be read"); }
        const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        if (stream.bad()) { throw std::invalid_argument("--config " + path_ + ": cannot be read"); }
        toml::table root;
        try {
            root = toml::parse(text, path_);
        } catch (const toml::parse_error& error) {
            fail(error.source().begin, "invalid TOML: " + std::string(error.description()));
        }

        // The top level first, then the program's table, whose keys replace the top level's.
        for (auto&& [key, node] : root) {
            const std::string name(key.str());
            if (name == program_.table || name == program_.other_table) {
                if (!node.is_table()) { fail(key.source().begin, "'" + name + "' must be a table of options"); }
                continue;
            }
            if (node.is_table()) {
                fail(key.source().begin, "unknown table '" + name + "' (the tables are [" +
                                             std::string(program_.table) + "] and [" +
                                             std::string(program_.other_table) + "])");
            }
            put(Setting{name, &node, key.source().begin});
        }
        if (const toml::table* own = root[program_.table].as_table()) {
            for (auto&& [key, node] : *own) {
                const std::string name(key.str());
                if (node.is_table()) {
                    fail(key.source().begin, "'" + std::string(program_.table) + "." + name +
                                                 "': a table holds options, not other tables");
                }
                put(Setting{name, &node, key.source().begin});
            }
        }
        std::stable_sort(settings_.begin(), settings_.end(), [](const Setting& a, const Setting& b) {
            return std::tie(a.at.line, a.at.column) < std::tie(b.at.line, b.at.column);
        });

        ConfigFile file;
        for (const Setting& setting : settings_) { apply(setting, file); }
        return file;
    }

private:
    [[noreturn]] void fail(const toml::source_position& at, const std::string& message) const {
        throw std::invalid_argument("--config " + path_ + ":" + std::to_string(at.line) + ":" +
                                    std::to_string(at.column) + ": " + message);
    }

    void put(Setting setting) {
        const auto same = std::find_if(settings_.begin(), settings_.end(),
                                       [&](const Setting& other) { return other.key == setting.key; });
        if (same == settings_.end()) {
            settings_.push_back(std::move(setting));
        } else {
            *same = std::move(setting);
        }
    }

    // A string, or a number as the text that parses back to it; empty for any other node.
    static std::optional<std::string> scalar_text(const toml::node& node) {
        if (const auto* text = node.as_string()) { return text->get(); }
        if (const auto* integer = node.as_integer()) { return std::to_string(integer->get()); }
        if (const auto* real = node.as_floating_point()) {
            char buffer[64];
            const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), real->get());
            return std::string(buffer, converted.ptr);
        }
        return std::nullopt;
    }

    void apply(const Setting& setting, ConfigFile& file) const {
        const std::string& key  = setting.key;
        const toml::node& value = *setting.value;
        if (key == "config") { fail(setting.at, "a config file cannot name another one"); }
        if (key == "model") {
            const auto* path = value.as_string();
            if (path == nullptr) { fail(setting.at, "'model' takes the artifact path as a string"); }
            file.model = path->get();
            return;
        }
        const std::string name     = "--" + key;
        const ConfigOption* option = program_.find_option(name);
        if (option == nullptr) {
            fail(setting.at, "unknown option '" + key + "'" +
                                 (key.find('_') != std::string::npos
                                      ? " (option names use dashes, as on the command line)"
                                      : ""));
        }
        if (!option->takes_value) {
            const auto* flag = value.as_boolean();
            if (flag == nullptr) { fail(setting.at, "'" + key + "' is a flag and takes true or false"); }
            if (flag->get()) { file.arguments.push_back(name); }
            return;
        }
        if (const auto* values = value.as_array()) {
            if (!option->repeatable) { fail(setting.at, "'" + key + "' takes one value, not an array"); }
            for (const toml::node& item : *values) {
                std::optional<std::string> text = scalar_text(item);
                if (!text) { fail(setting.at, "'" + key + "' takes strings or numbers"); }
                file.arguments.push_back(name);
                file.arguments.push_back(std::move(*text));
            }
            return;
        }
        std::optional<std::string> text = scalar_text(value);
        if (!text) { fail(setting.at, "'" + key + "' takes a string or a number"); }
        file.arguments.push_back(name);
        file.arguments.push_back(std::move(*text));
    }

    const std::string& path_;
    const ConfigProgram& program_;
    std::vector<Setting> settings_;
};

} // namespace

ConfigFile read_config_file(const std::string& path, const ConfigProgram& program) {
    return FileReader(path, program).read();
}

ProgramArguments read_program_arguments(int argc, char** argv, const ConfigProgram& program) {
    ProgramArguments result;
    int i = 1;
    if (argc > 1 && !std::string_view(argv[1]).starts_with("--")) {
        result.model = argv[1];
        i            = 2;
    }
    std::vector<std::string> command_line;
    for (; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config") {
            if (++i >= argc) { throw std::invalid_argument("--config needs a value"); }
            if (result.config_path) { throw std::invalid_argument("--config may be given only once"); }
            result.config_path = argv[i];
            continue;
        }
        const ConfigOption* option = program.find_option(arg);
        command_line.push_back(std::move(arg));
        if (option != nullptr && option->takes_value && i + 1 < argc) { command_line.emplace_back(argv[++i]); }
    }
    if (result.config_path) {
        ConfigFile file = read_config_file(*result.config_path, program);
        if (result.model.empty() && file.model) { result.model = std::move(*file.model); }
        result.config_arguments = file.arguments;
        result.arguments        = std::move(file.arguments);
    }
    result.arguments.insert(result.arguments.end(), std::make_move_iterator(command_line.begin()),
                            std::make_move_iterator(command_line.end()));
    return result;
}

std::string describe_config(const std::string& path, const std::vector<std::string>& arguments) {
    std::string line = "config " + path + ":";
    if (arguments.empty()) { return line + " (no options)"; }
    for (const std::string& argument : arguments) {
        line += ' ';
        if (argument.find(' ') == std::string::npos && !argument.empty()) {
            line += argument;
        } else {
            line += '"' + argument + '"';
        }
    }
    return line;
}

std::vector<std::string> redact_option_values(std::vector<std::string> arguments,
                                              std::initializer_list<std::string_view> secrets) {
    for (std::size_t i = 0; i + 1 < arguments.size(); ++i) {
        if (std::find(secrets.begin(), secrets.end(), arguments[i]) != secrets.end()) {
            arguments[++i] = "<redacted>";
        }
    }
    return arguments;
}

} // namespace infernix::product
