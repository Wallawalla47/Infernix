#pragma once

// infernix-serve launch parameters as an argument vector: lookups and edits by flag, the flags the
// launcher sets itself, and quoting for display and for CreateProcess.

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::launcher {

using Args = std::vector<std::string>;

// Whether a flag takes a value. The parsed `--help` answers it when available; without it a flag
// takes a value when the next token is not itself a flag.
using TakesValue = std::function<std::optional<bool>(std::string_view flag)>;

// Index of the first occurrence of `flag` (as `--flag` or `--flag=value`), or npos.
[[nodiscard]] std::size_t find_flag(const Args& args, std::string_view flag);
[[nodiscard]] bool has_flag(const Args& args, std::string_view flag);
// The value after `flag` (or after `=`); nullopt when absent or valueless.
[[nodiscard]] std::optional<std::string> flag_value(const Args& args, std::string_view flag);
// Replaces every occurrence of `flag` with one `flag value` (appended when absent).
void set_flag(Args& args, std::string_view flag, std::string_view value);
// Adds a valueless flag when absent.
void set_switch(Args& args, std::string_view flag);
// Removes every occurrence of `flag` and the value that follows it when it takes one.
void remove_flag(Args& args, std::string_view flag, const TakesValue& takes_value = {});

// Flags infernix-serve gets from the launcher: the listen address, the public model id, console
// rendering and the config file. User launch parameters never carry them.
[[nodiscard]] const std::vector<std::string_view>& managed_flags();
// Removes the managed flags from `args`; returns the names removed.
std::vector<std::string> strip_managed_flags(Args& args, const TakesValue& takes_value = {});

// Effective --max-context (infernix-serve's default 8192 when absent or unreadable).
[[nodiscard]] std::uint64_t context_length(const Args& args);

// One argument quoted for the host shell (cmd/PowerShell on Windows, POSIX sh elsewhere), for
// display.
[[nodiscard]] std::string shell_quote(std::string_view arg);
[[nodiscard]] std::string command_line_for_display(const std::vector<std::string>& argv);
// One argument quoted by the rules CommandLineToArgvW and the MSVC runtime undo: bare when it can
// be, windows_quoted always quoted.
[[nodiscard]] std::string windows_quote(std::string_view arg);
[[nodiscard]] std::string windows_quoted(std::string_view arg);
// A command line cmd.exe reads before the program does (a batch file): each argument
// windows_quote'd, and quoted whenever it holds one of cmd's metacharacters; those the line's
// inner quotes leave outside cmd's own quote state are caret-escaped. A batch file doubles % itself.
[[nodiscard]] std::string batch_command_line(const std::vector<std::string>& argv);

} // namespace infernix::launcher
