#pragma once

// --config FILE for infernix and infernix-serve. The file is TOML; its keys are the program's long
// option names without the dashes, so it accepts exactly the program's command-line options and each
// value reaches the option's own parser as its command-line text. Only config_file.cpp sees TOML.

#include <functional>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::product {

// One command-line option of a program.
struct ConfigOption {
    std::string_view name;    // with the dashes: "--max-context"
    bool takes_value = true;  // false: a flag, true or false in a file
    bool repeatable  = false; // may be given more than once; a file gives each value of an array
};

// How a program reads a config file. Top-level keys and the keys of the program's own table
// ([serve] or [cli]) apply, the table's overriding the top level's; the other program's table is
// skipped, so one file can configure both programs.
struct ConfigProgram {
    std::string_view table;       // this program's table: "serve" or "cli"
    std::string_view other_table; // the other program's table
    // The option of that name ("--port"), or nullptr when the program has none.
    std::function<const ConfigOption*(std::string_view)> find_option;
};

// A file's options as command-line arguments, ordered by their place in the file.
struct ConfigFile {
    std::optional<std::string> model; // the `model` key: the artifact path
    std::vector<std::string> arguments;
};

// Reads `path`. Throws std::invalid_argument naming the file, line and key for a file that cannot
// be read or parsed, an unknown key or table, a value of the wrong type, an array for an option
// that cannot repeat, or a `config` key.
[[nodiscard]] ConfigFile read_config_file(const std::string& path, const ConfigProgram& program);

// A program's command line with its --config file merged in.
struct ProgramArguments {
    // The artifact path: the command line's first argument when it is not an option, else the
    // file's `model` key; empty when neither gives one.
    std::string model;
    // The file's options, then the command line's: a later option overrides an earlier one, so the
    // command line overrides the file; values of a repeatable option accumulate.
    std::vector<std::string> arguments;
    std::optional<std::string> config_path;
    std::vector<std::string> config_arguments; // the file's options alone
};

// argv[0] is the program. --config FILE may be given once; an option's value is never read as an
// option (`--model-id --config` names a model id).
[[nodiscard]] ProgramArguments read_program_arguments(int argc, char** argv,
                                                      const ConfigProgram& program);

// The startup log's line for a config file: `config FILE: --max-context 4096 --greedy`, with an
// argument that holds a space quoted; "(no options)" when the file gave none.
[[nodiscard]] std::string describe_config(const std::string& path, const std::vector<std::string>& arguments);

// `arguments` with the value of each option named in `secrets` replaced by "<redacted>".
[[nodiscard]] std::vector<std::string>
redact_option_values(std::vector<std::string> arguments,
                     std::initializer_list<std::string_view> secrets);

} // namespace infernix::product
