#pragma once

// Launch scripts in and out. Import reads the infernix-serve (or ninfer-serve) commands of a
// Windows batch file (`set VAR=value`, `^` line continuations, %VAR% expansion) or a POSIX shell
// script (`export VAR=value`, `VAR=value command` prefixes, `\` continuations, $VAR / ${VAR}
// expansion): each command's executable, artifact (the first positional argument), the arguments
// after it and the environment set before it. Export writes a configuration as such a script.

#include "launcher/args.h"
#include "launcher/store.h"

#include <string>
#include <string_view>
#include <vector>

namespace infernix::launcher {

struct ScriptCommand {
    std::string executable;
    std::string artifact; // empty when the command names none
    Args args;            // after the artifact, in order
    EnvVars env;          // set before the command (script-wide and per-command)
};

enum class ScriptShell { Cmd, Sh };

// The shell a script is written for: by its file name's extension, else by its contents.
[[nodiscard]] ScriptShell script_shell(std::string_view text, std::string_view file_name);

// Every infernix-serve / ninfer-serve command of the script, and every command of `server_name` (the
// launcher's own server, whatever its file is called). `takes_value` tells which flags take a value
// (so a flag's value is never taken for the artifact).
[[nodiscard]] std::vector<ScriptCommand> parse_launch_script(std::string_view text, ScriptShell shell,
                                                             const TakesValue& takes_value,
                                                             std::string_view server_name = {});

// A runnable script for one configuration: the environment, then infernix-serve with the artifact,
// the launch parameters and a listen address and model id of its own.
[[nodiscard]] std::string write_launch_script(ScriptShell shell, const std::string& executable,
                                              const std::string& artifact, const Args& args, const EnvVars& env,
                                              const std::string& model_id);

} // namespace infernix::launcher
