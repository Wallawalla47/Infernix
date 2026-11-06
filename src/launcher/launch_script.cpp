#include "launcher/launch_script.h"

#include "launcher/util.h"

#include <algorithm>
#include <cctype>
#include <map>

namespace infernix::launcher {
namespace {

// The file name of a command token or path, lower case.
std::string file_name_of(std::string_view token) {
    std::string name = lower(token);
    const std::size_t slash = name.find_last_of("/\\");
    if (slash != std::string::npos) { name.erase(0, slash + 1); }
    return name;
}

bool server_executable(std::string_view token, std::string_view server_name) {
    const std::string name = file_name_of(token);
    return name == "infernix-serve" || name == "infernix-serve.exe" || name == "ninfer-serve" ||
           name == "ninfer-serve.exe" || (!server_name.empty() && name == file_name_of(server_name));
}

// Tokens that end a command (redirections, pipes, command separators).
bool command_end(std::string_view token) {
    return token == "|" || token == "||" || token == "&" || token == "&&" || token == ";" || token.starts_with(">") ||
           token.starts_with("1>") || token.starts_with("2>") || token.starts_with("<");
}

std::vector<std::string> split_lines(std::string_view text) {
    std::vector<std::string> lines;
    std::string line;
    for (const char c : text) {
        if (c == '\n') {
            lines.push_back(line);
            line.clear();
        } else if (c != '\r') {
            line.push_back(c);
        }
    }
    lines.push_back(line);
    return lines;
}

// --- Windows batch ---------------------------------------------------------------------------

// %NAME% from the variables the script set; %~dp0 (the script's folder) dropped, leaving a path
// relative to it; anything else left as written.
std::string expand_cmd(std::string_view line, const std::map<std::string, std::string>& vars) {
    std::string out;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] != '%') {
            out.push_back(line[i]);
            continue;
        }
        if (line.substr(i, 5) == "%~dp0") {
            i += 4;
            continue;
        }
        const std::size_t end = line.find('%', i + 1);
        if (end == std::string_view::npos) {
            out.push_back('%');
            continue;
        }
        const std::string name = lower(line.substr(i + 1, end - i - 1));
        if (const auto it = vars.find(name); it != vars.end()) {
            out += it->second;
            i = end;
        } else {
            out.push_back('%');
        }
    }
    return out;
}

// cmd's pass first: a ^ outside cmd's quotes, which every " toggles, escapes the next character;
// then CommandLineToArgvW's rules on what the program receives.
std::vector<std::string> tokenize_cmd(std::string_view batch_line) {
    std::string line;
    bool cmd_quoted = false;
    for (std::size_t i = 0; i < batch_line.size(); ++i) {
        const char c = batch_line[i];
        if (c == '"') {
            cmd_quoted = !cmd_quoted;
        } else if (!cmd_quoted && c == '^' && i + 1 < batch_line.size()) {
            line.push_back(batch_line[++i]);
            continue;
        }
        line.push_back(c);
    }
    std::vector<std::string> tokens;
    std::string token;
    bool in_token = false, quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\\') {
            std::size_t slashes = 0;
            while (i < line.size() && line[i] == '\\') {
                ++slashes;
                ++i;
            }
            if (i < line.size() && line[i] == '"') {
                token.append(slashes / 2, '\\');
                if (slashes % 2 == 1) {
                    token.push_back('"');
                } else {
                    quoted = !quoted;
                }
            } else {
                token.append(slashes, '\\');
                --i;
            }
            in_token = true;
            continue;
        }
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                token.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
            in_token = true;
            continue;
        }
        if (!quoted && (c == ' ' || c == '\t')) {
            if (in_token) { tokens.push_back(std::move(token)); }
            token.clear();
            in_token = false;
            continue;
        }
        token.push_back(c);
        in_token = true;
    }
    if (in_token) { tokens.push_back(std::move(token)); }
    return tokens;
}

// --- POSIX shell -----------------------------------------------------------------------------

std::string expand_sh_name(std::string_view text, std::size_t& i, const std::map<std::string, std::string>& vars) {
    // At a '$': ${NAME} or $NAME; unknown names stay as written.
    std::size_t start = i + 1, end = start;
    const bool braced = start < text.size() && text[start] == '{';
    if (braced) {
        end = text.find('}', start);
        if (end == std::string_view::npos) { return "$"; }
        const std::string name(text.substr(start + 1, end - start - 1));
        if (const auto it = vars.find(name); it != vars.end()) {
            i = end;
            return it->second;
        }
        return "$";
    }
    while (end < text.size() && (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_')) { ++end; }
    if (end == start) { return "$"; }
    const std::string name(text.substr(start, end - start));
    if (const auto it = vars.find(name); it != vars.end()) {
        i = end - 1;
        return it->second;
    }
    return "$";
}

std::vector<std::string> tokenize_sh(std::string_view line, const std::map<std::string, std::string>& vars) {
    std::vector<std::string> tokens;
    std::string token;
    bool in_token = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '#' && !in_token) { break; }
        if (c == ' ' || c == '\t') {
            if (in_token) { tokens.push_back(std::move(token)); }
            token.clear();
            in_token = false;
            continue;
        }
        in_token = true;
        if (c == '\'') {
            const std::size_t end = line.find('\'', i + 1);
            token += line.substr(i + 1, end == std::string_view::npos ? std::string_view::npos : end - i - 1);
            i = end == std::string_view::npos ? line.size() : end;
        } else if (c == '"') {
            for (++i; i < line.size() && line[i] != '"'; ++i) {
                if (line[i] == '\\' && i + 1 < line.size() && std::string_view("\"\\$`").find(line[i + 1]) != std::string_view::npos) {
                    token.push_back(line[++i]);
                } else if (line[i] == '$') {
                    token += expand_sh_name(line, i, vars);
                } else {
                    token.push_back(line[i]);
                }
            }
        } else if (c == '\\' && i + 1 < line.size()) {
            token.push_back(line[++i]);
        } else if (c == '$') {
            token += expand_sh_name(line, i, vars);
        } else {
            token.push_back(c);
        }
    }
    if (in_token) { tokens.push_back(std::move(token)); }
    return tokens;
}

bool assignment(std::string_view token, std::string& name, std::string& value) {
    const std::size_t equals = token.find('=');
    if (equals == std::string_view::npos || equals == 0) { return false; }
    for (std::size_t i = 0; i < equals; ++i) {
        const char c = token[i];
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_') || (i == 0 && std::isdigit(static_cast<unsigned char>(c)))) {
            return false;
        }
    }
    name  = std::string(token.substr(0, equals));
    value = std::string(token.substr(equals + 1));
    return true;
}

void set_var(EnvVars& env, std::map<std::string, std::string>& vars, const std::string& name, const std::string& value,
             bool fold_case) {
    vars[fold_case ? lower(name) : name] = value;
    const auto it = std::find_if(env.begin(), env.end(), [&](const auto& entry) {
        return fold_case ? iequals(entry.first, name) : entry.first == name;
    });
    if (it != env.end()) {
        it->second = value;
    } else {
        env.emplace_back(name, value);
    }
}

// The command's arguments after the executable: the artifact (the first positional) and the rest.
ScriptCommand make_command(std::string executable, std::vector<std::string> tokens, EnvVars env,
                           const TakesValue& takes_value) {
    ScriptCommand command;
    command.executable = std::move(executable);
    command.env        = std::move(env);
    std::size_t artifact = std::string::npos;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];
        if (!token.starts_with("-")) {
            artifact = i;
            break;
        }
        if (token.find('=') != std::string::npos) { continue; }
        const std::optional<bool> takes = takes_value ? takes_value(token) : std::nullopt;
        if (takes.value_or(i + 1 < tokens.size() && !tokens[i + 1].starts_with("-") && i != 0)) { ++i; }
    }
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i == artifact) {
            command.artifact = tokens[i];
        } else {
            command.args.push_back(tokens[i]);
        }
    }
    return command;
}

std::string posix_quote(std::string_view arg) {
    if (!arg.empty() && std::all_of(arg.begin(), arg.end(), [](char c) {
            return std::isalnum(static_cast<unsigned char>(c)) || std::string_view("-_./:=,+@%").find(c) != std::string_view::npos;
        })) {
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
    return out + "'";
}

} // namespace

ScriptShell script_shell(std::string_view text, std::string_view file_name) {
    const std::string name = lower(file_name);
    if (name.ends_with(".bat") || name.ends_with(".cmd")) { return ScriptShell::Cmd; }
    if (name.ends_with(".sh") || name.ends_with(".bash")) { return ScriptShell::Sh; }
    const std::string lowered = lower(text);
    if (lowered.starts_with("#!") || lowered.find("\nexport ") != std::string::npos || lowered.starts_with("export ")) {
        return ScriptShell::Sh;
    }
    if (lowered.find("@echo off") != std::string::npos || lowered.find("\nset ") != std::string::npos ||
        lowered.starts_with("set ") || lowered.find(".exe") != std::string::npos) {
        return ScriptShell::Cmd;
    }
    return ScriptShell::Sh;
}

std::vector<ScriptCommand> parse_launch_script(std::string_view text, ScriptShell shell, const TakesValue& takes_value,
                                               std::string_view server_name) {
    // Logical lines: continuations joined (batch: a trailing ^; shell: a trailing \).
    std::vector<std::string> lines;
    std::string pending;
    const char continuation = shell == ScriptShell::Cmd ? '^' : '\\';
    for (std::string line : split_lines(text)) {
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) { line.pop_back(); }
        if (!line.empty() && line.back() == continuation) {
            line.pop_back();
            pending += line + " ";
            continue;
        }
        lines.push_back(pending + line);
        pending.clear();
    }
    if (!pending.empty()) { lines.push_back(pending); }

    std::vector<ScriptCommand> commands;
    EnvVars env;
    std::map<std::string, std::string> vars;
    const bool cmd = shell == ScriptShell::Cmd;
    for (const std::string& raw : lines) {
        std::string line = trim(raw);
        if (cmd) {
            while (line.starts_with("@")) { line.erase(0, 1); }
            const std::string lowered = lower(line);
            if (line.empty() || lowered.starts_with("rem ") || lowered == "rem" || line.starts_with("::") ||
                line.starts_with(":")) {
                continue;
            }
            if (lowered.starts_with("set ") && !lowered.starts_with("set /")) {
                std::string body = trim(std::string_view(line).substr(4));
                if (body.size() >= 2 && body.front() == '"' && body.back() == '"') { body = body.substr(1, body.size() - 2); }
                const std::size_t equals = body.find('=');
                if (equals != std::string::npos && equals > 0) {
                    set_var(env, vars, body.substr(0, equals), expand_cmd(body.substr(equals + 1), vars), true);
                }
                continue;
            }
            line = expand_cmd(line, vars);
            const std::vector<std::string> tokens = tokenize_cmd(line);
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                if (!server_executable(tokens[i], server_name)) { continue; }
                std::vector<std::string> rest;
                for (std::size_t j = i + 1; j < tokens.size() && !command_end(tokens[j]); ++j) { rest.push_back(tokens[j]); }
                commands.push_back(make_command(tokens[i], std::move(rest), env, takes_value));
                break;
            }
            continue;
        }
        const std::vector<std::string> tokens = tokenize_sh(line, vars);
        if (tokens.empty()) { continue; }
        std::size_t at = 0;
        if (tokens[0] == "export" || tokens[0] == "declare" || tokens[0] == "local") { at = 1; }
        EnvVars command_env = env;
        std::map<std::string, std::string> command_vars = vars;
        std::string name, value;
        bool only_assignments = true;
        for (; at < tokens.size(); ++at) {
            if (tokens[at] == "-x" && tokens[0] == "declare") { continue; }
            if (!assignment(tokens[at], name, value)) {
                only_assignments = false;
                break;
            }
            set_var(command_env, command_vars, name, value, false);
        }
        if (only_assignments) {
            // export VAR=value: the environment from here on; a plain VAR=value is a shell variable
            // (expanded in later lines, not passed to the server).
            if (tokens[0] == "export" || tokens[0] == "declare") { env = std::move(command_env); }
            vars = std::move(command_vars);
            continue;
        }
        for (std::size_t i = at; i < tokens.size(); ++i) {
            if (command_end(tokens[i])) { break; }
            if (!server_executable(tokens[i], server_name)) { continue; }
            std::vector<std::string> rest;
            for (std::size_t j = i + 1; j < tokens.size() && !command_end(tokens[j]); ++j) { rest.push_back(tokens[j]); }
            commands.push_back(make_command(tokens[i], std::move(rest), std::move(command_env), takes_value));
            break;
        }
    }
    return commands;
}

std::string write_launch_script(ScriptShell shell, const std::string& executable, const std::string& artifact,
                                const Args& args, const EnvVars& env, const std::string& model_id) {
    Args argv = {artifact};
    argv.insert(argv.end(), args.begin(), args.end());
    for (const char* flag : {"--host", "127.0.0.1", "--port", "8080", "--model-id"}) { argv.emplace_back(flag); }
    argv.push_back(model_id);
    std::string out;
    if (shell == ScriptShell::Cmd) {
        // In a batch file % must be doubled.
        const auto batch = [](std::string text) {
            std::string escaped;
            for (const char c : text) {
                escaped.push_back(c);
                if (c == '%') { escaped.push_back('%'); }
            }
            return escaped;
        };
        // The file is UTF-8, which cmd reads only in code page 65001.
        out = "@echo off\r\nchcp 65001>nul\r\nrem " + model_id + ": written by infernix-launcher\r\n";
        for (const auto& [name, value] : env) { out += batch("set \"" + name + "=" + value + "\"") + "\r\n"; }
        argv.insert(argv.begin(), executable);
        out += batch(batch_command_line(argv)) + "\r\npause\r\n";
        return out;
    }
    out = "#!/usr/bin/env bash\n# " + model_id + ": written by infernix-launcher\n";
    for (const auto& [name, value] : env) { out += "export " + name + "=" + posix_quote(value) + "\n"; }
    out += "exec " + posix_quote(executable);
    for (const std::string& arg : argv) { out += " " + posix_quote(arg); }
    return out + "\n";
}

} // namespace infernix::launcher
