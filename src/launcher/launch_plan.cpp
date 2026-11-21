// The launcher's launch planning (Launcher members, launcher.h): launch scripts matched to the library,
// infernix-serve --plan fits and the configuration preview.

#include "launcher/launcher.h"

#include "launcher/launch_script.h"
#include "launcher/load_advice.h"
#include "launcher/process.h"
#include "launcher/route_support.h"

#include <cstdio>

namespace infernix::launcher {

Json Launcher::import_script_json(const std::string& text, const std::string& file_name) const {
    const ServeOptions serve = serve_options();
    const TakesValue takes   = [&serve](std::string_view flag) { return serve.takes_value(flag); };
    Json commands            = Json::array();
    for (ScriptCommand& command :
         parse_launch_script(text, script_shell(text, file_name), takes, path_utf8(serve_executable().filename()))) {
        std::vector<std::string> warnings;
        const std::optional<std::string> alias = flag_value(command.args, "--model-id");
        const std::vector<std::string> removed = strip_managed_flags(command.args, takes);
        if (!removed.empty()) {
            std::string list;
            for (const std::string& flag : removed) { list += (list.empty() ? "" : ", ") + flag; }
            warnings.push_back("set by the launcher, so dropped: " + list);
        }
        if (lower(path_utf8(utf8_path(command.executable).filename())).starts_with("ninfer")) {
            warnings.push_back("a command for NInfer's server (ninfer-serve): options infernix-serve lacks are named below");
        }
        if (serve.found && !serve.options.empty()) {
            for (const std::string& arg : command.args) {
                if (!arg.starts_with("--")) { continue; }
                const std::string flag = arg.substr(0, arg.find('='));
                if (!serve.known(flag)) { warnings.push_back(flag + " is not an option of this infernix-serve"); }
            }
        }
        // The artifact: by its path, else by its file name.
        Json model = nullptr;
        if (!command.artifact.empty()) {
            const std::filesystem::path wanted = utf8_path(command.artifact);
            const std::string wanted_path      = lower(path_utf8(wanted.lexically_normal()));
            const std::string wanted_name      = lower(path_utf8(wanted.filename()));
            std::lock_guard lock(library_mutex_);
            const ScannedModel* by_name = nullptr;
            for (const ScannedModel& candidate : library_.models) {
                for (const std::filesystem::path& location : candidate.locations) {
                    if (lower(path_utf8(location.lexically_normal())) == wanted_path) {
                        model = {{"key", candidate.key}, {"name", candidate.name}, {"file_name", candidate.file_name}};
                    }
                }
                if (lower(candidate.file_name) == wanted_name && by_name == nullptr) { by_name = &candidate; }
            }
            if (model.is_null() && by_name != nullptr) {
                model = {{"key", by_name->key}, {"name", by_name->name}, {"file_name", by_name->file_name}};
            }
        } else {
            warnings.push_back("the command names no artifact");
        }
        Json env = Json::array();
        for (const auto& [name, value] : command.env) { env.push_back({{"name", name}, {"value", value}}); }
        commands.push_back({{"artifact", command.artifact},
                            {"model", model},
                            {"alias", alias ? Json(*alias) : Json(nullptr)},
                            {"args", command.args},
                            {"env", env},
                            {"warnings", warnings}});
    }
    return {{"commands", commands}};
}

Json Launcher::fit_plan_json(const std::string& config_id, const ScannedModel& model, const Args& args,
                             const EnvVars& env, std::chrono::milliseconds timeout) const {
    Json out = {{"supported", true},    {"fits", nullptr},       {"summary", ""},      {"vram", nullptr},
                {"ram", nullptr},       {"max_context", nullptr}, {"kv_tokens", nullptr}, {"error", nullptr},
                {"suggestions", Json::array()}, {"report", nullptr}};
    const ServeOptions serve = serve_options();
    if (!serve.options.empty() && !serve.known("--plan")) {
        out["supported"] = false;
        out["summary"]   = "this infernix-serve cannot plan (no --plan); update it";
        return out;
    }
    const TakesValue takes = [&serve](std::string_view flag) { return serve.takes_value(flag); };
    Args argv              = {path_utf8(model.path)};
    Args own               = args;
    strip_managed_flags(own, takes);
    argv.insert(argv.end(), own.begin(), own.end());
    std::error_code ignored;
    std::filesystem::create_directories(options_.data_dir / "plans", ignored);
    const std::filesystem::path report = options_.data_dir / "plans" / ("plan-" + random_token(12) + ".json");
    argv.push_back("--plan");
    argv.push_back(path_utf8(report));
    const std::optional<CaptureResult> run = run_capture(serve_executable(), argv, timeout, env);
    const std::optional<std::string> text  = read_file(report);
    std::filesystem::remove(report, ignored);
    Json plan;
    try {
        if (text) { plan = Json::parse(*text); }
    } catch (const Json::exception&) {}
    if (!plan.is_object()) {
        out["error"] = !run ? "infernix-serve --plan did not finish" : "infernix-serve --plan wrote no report (exit status " +
                                                                           std::to_string(run->exit_code) + ")";
        out["summary"] = out["error"];
        return out;
    }
    out["report"] = plan;
    if (plan.value("error", Json(nullptr)).is_string()) { out["error"] = plan["error"]; }
    if (plan.contains("fits") && plan["fits"].is_boolean()) { out["fits"] = plan["fits"]; }
    const auto bytes = [](const Json& object, const char* key) { return json_or<std::uint64_t>(object, key, 0); };
    const Json gpu   = plan.value("gpu", Json::object());
    const Json vram  = plan.value("vram", Json::object());
    const Json ram   = plan.value("ram", Json::object());
    const auto parts = [&](const Json& source, std::initializer_list<std::pair<const char*, const char*>> names) {
        Json list = Json::array();
        for (const auto& [key, label] : names) {
            if (const std::uint64_t value = bytes(source, key); value > 0) { list.push_back({{"label", label}, {"bytes", value}}); }
        }
        return list;
    };
    if (vram.is_object() && !vram.empty()) {
        out["vram"] = {{"total", bytes(gpu, "total_bytes")},
                       {"available", bytes(gpu, "free_bytes")},
                       {"planned", bytes(vram, "planned_bytes")},
                       {"parts", parts(vram, {{"weights_bytes", "Weights"},
                                              {"draft_screen_bytes", "Draft screen"},
                                              {"adapter_bytes", "LoRA adapter"},
                                              {"kv_bytes", "KV cache"},
                                              {"workspace_bytes", "Workspace"},
                                              {"graph_allowance_bytes", "CUDA graphs"},
                                              {"expert_cache_bytes", "Expert cache"},
                                              {"headroom_bytes", "Headroom"}})}};
    }
    if (ram.is_object() && !ram.empty()) {
        out["ram"] = {{"total", bytes(ram, "total_bytes")},
                      {"available", bytes(ram, "available_bytes")},
                      {"planned", bytes(ram, "planned_bytes")},
                      {"parts", parts(ram, {{"experts_bytes", "Experts"},
                                            {"pinned_bytes", "Pinned"},
                                            {"host_context_bytes", "Host prefix cache"},
                                            {"ngram_bytes", "N-gram table"},
                                            {"ram_free_bytes", "Left free (--ram-free-mib)"}})}};
    }
    if (plan.contains("max_context") && plan["max_context"].is_number()) { out["max_context"] = plan["max_context"]; }
    if (vram.contains("kv_tokens") && vram["kv_tokens"].is_number()) { out["kv_tokens"] = vram["kv_tokens"]; }

    const auto gib = [](std::uint64_t value) {
        char text_buffer[32];
        std::snprintf(text_buffer, sizeof(text_buffer), "%.1f", static_cast<double>(value) / (1ull << 30));
        return std::string(text_buffer);
    };
    std::string summary;
    if (out["fits"].is_boolean()) {
        summary = out["fits"].get<bool>() ? "Fits: " : "Does not fit: ";
        if (!out["vram"].is_null()) {
            summary += gib(bytes(vram, "planned_bytes")) + " of " + gib(bytes(gpu, "total_bytes")) + " GiB VRAM";
        }
        if (!out["ram"].is_null()) {
            summary += (out["vram"].is_null() ? "" : ", ") + gib(bytes(ram, "planned_bytes")) + " of " +
                       gib(bytes(ram, "total_bytes")) + " GiB RAM";
        }
        if (!out["fits"].get<bool>() && out["error"].is_string()) { summary += " (" + out["error"].get<std::string>() + ")"; }
    } else {
        summary = out["error"].is_string() ? out["error"].get<std::string>() : "no answer";
    }
    // The running model holds memory the plan counts as used.
    const InstanceSnapshot instance = supervisor_->snapshot();
    if ((instance.state == InstanceState::Ready || instance.state == InstanceState::Starting) && !instance.alias.empty()) {
        summary += instance.model_key == config_id
                       ? " (measured with this model loaded: its own memory counts as used; unload it for an exact answer)"
                       : " (measured with " + instance.alias + " loaded: its memory counts as used)";
    }
    out["summary"] = summary;

    if (out["fits"] == false) {
        const std::string error = out["error"].is_string() ? out["error"].get<std::string>() : std::string();
        std::string kind        = classify_load_failure(error).value_or("");
        if (kind.empty()) { kind = bytes(vram, "left_bytes") == 0 && !vram.empty() ? "vram" : "ram"; }
        ResourceFacts facts;
        facts.ram_total     = bytes(ram, "total_bytes");
        facts.ram_available = bytes(ram, "available_bytes");
        facts.vram_total    = bytes(gpu, "total_bytes");
        Json suggestions    = Json::array();
        for (const LoadSuggestion& suggestion : suggest_load_fixes(kind, model, args, error, facts)) {
            suggestions.push_back(to_json(suggestion));
        }
        // The plan's own largest context comes first when it is smaller than asked, in place of the
        // estimate from the refusal's figures.
        if (!context_auto(args) && out["max_context"].is_number_unsigned() && out["max_context"].get<std::uint64_t>() < context_length(args) &&
            out["max_context"].get<std::uint64_t>() > 0) {
            const std::uint64_t fit_tokens = out["max_context"].get<std::uint64_t>();
            Args fitted = args;
            set_flag(fitted, "--max-context", std::to_string(fit_tokens));
            Json kept = Json::array();
            for (const Json& suggestion : suggestions) {
                if (!suggestion.value("title", "").starts_with("Lower the context")) { kept.push_back(suggestion); }
            }
            std::string tokens = std::to_string(fit_tokens);
            for (int at = static_cast<int>(tokens.size()) - 3; at > 0; at -= 3) { tokens.insert(static_cast<std::size_t>(at), ","); }
            const Json fit = {{"title", "Lower the context to " + tokens + " tokens"},
                              {"detail", "The largest --max-context that fits with these settings, by the server's own sizing."},
                              {"args", fitted}};
            kept.insert(kept.begin(), fit);
            suggestions = std::move(kept);
        }
        out["suggestions"] = suggestions;
    }
    return out;
}

Json Launcher::preview_json(const std::string& config_id, const ScannedModel& model, const std::string& alias,
                            Args args, const EnvVars& env) const {
    const ServeOptions serve = serve_options();
    const TakesValue takes   = [&serve](std::string_view flag) { return serve.takes_value(flag); };
    std::vector<std::string> warnings;
    const std::vector<std::string> removed = strip_managed_flags(args, takes);
    if (!removed.empty()) {
        std::string list;
        for (const std::string& flag : removed) { list += (list.empty() ? "" : ", ") + flag; }
        warnings.push_back("set by the launcher, so removed: " + list);
    }
    if (serve.found && !serve.options.empty()) {
        for (const std::string& arg : args) {
            if (!arg.starts_with("--")) { continue; }
            const std::string flag = arg.substr(0, arg.find('='));
            if (!serve.known(flag)) { warnings.push_back(flag + " is not an option of this infernix-serve"); }
        }
    } else if (serve.error) {
        warnings.push_back(*serve.error);
    }
    for (const char* flag : {"--draft-model", "--ngram-volume", "--chat-template", "--expert-stripe", "--lora",
                             "--context-cost-presets"}) {
        if (const std::optional<std::string> value = flag_value(args, flag)) {
            std::error_code error;
            if (!std::filesystem::exists(utf8_path(*value), error)) {
                warnings.push_back(std::string(flag) + ": file not found: " + *value);
            }
        }
    }
    // Options infernix-serve refuses without another (the NOTES of its --help): named here, before
    // a load fails on them.
    const auto set = [&args](std::string_view flag) {
        const std::optional<std::string> value = flag_value(args, flag);
        return value ? *value != "0" : has_flag(args, flag);
    };
    const auto needs = [&](std::string_view flag, std::string_view other, bool other_set) {
        if (set(flag) && !other_set) { warnings.push_back(std::string(flag) + " needs " + std::string(other)); }
    };
    needs("--ssd-context-gib", "--ssd-context-dir", set("--ssd-context-dir"));
    needs("--ssd-context-dir", "--ssd-context-gib", set("--ssd-context-gib"));
    needs("--ssd-context-gib-per-hour", "--ssd-context-gib and --ssd-context-dir",
          set("--ssd-context-gib") && set("--ssd-context-dir"));
    if (const std::optional<std::string> folder = flag_value(args, "--ssd-context-dir")) {
        std::error_code error;
        if (!std::filesystem::is_directory(utf8_path(*folder), error)) {
            warnings.push_back("--ssd-context-dir: folder not found: " + *folder);
        }
    }
    needs("--prefix-cache-save-mins",
          "a prefix cache file (Settings, Keep each model's prefix cache, or --prefix-cache-file)",
          has_flag(args, "--prefix-cache-file") || prefix_cache_file(config_id, args).has_value());
    needs("--ngram-native-sessions", "--ngram-archive-mib", set("--ngram-archive-mib"));
    needs("--request-log-max-mib", "--request-log-jsonl", set("--request-log-jsonl"));
    needs("--request-log-keep", "--request-log-max-mib", set("--request-log-max-mib"));
    if (flag_value(args, "--vision-offload").value_or("") == "on" && !has_flag(args, "--vision")) {
        warnings.push_back("--vision-offload on needs --vision");
    }
    if (model.needs_ngram_volume && !model.own_ngram_volume && !has_flag(args, "--ngram-volume")) {
        warnings.push_back(model.family_label +
                           " needs an n-gram volume (--ngram-volume): none was found in the model folders; download one "
                           "in Discover");
    }
    if (has_flag(args, "--lora") && (model.needs_ngram_volume || model.num_experts.has_value())) {
        warnings.push_back("--lora adapts dense Qwen3.5 models (the 27B) only; this model's MLPs are experts");
    }
    if (has_flag(args, "--draft-model") && !model.accepts_draft_model) {
        warnings.push_back("--draft-model applies to Qwen3.8-Flash-Next only; this artifact bundles its drafters");
    }
    if (!model.missing_files.empty()) {
        warnings.push_back("missing or incomplete parts: " + std::to_string(model.missing_files.size()));
    }
    if (model.native_context != 0 && context_length(args) > model.native_context &&
        !has_flag(args, "--rope-yarn-factor")) {
        warnings.push_back("--max-context is beyond the model's native " + std::to_string(model.native_context) +
                           " tokens: add --rope-yarn-factor");
    }
    for (const auto& [name, value] : env) {
        if (name.empty() || name.find('=') != std::string::npos) { warnings.push_back("not a variable name: " + name); }
    }
    const std::vector<std::string> managed = {"--host", "127.0.0.1", "--port", "<auto>", "--model-id", alias,
                                              "--any-model-name", "--log-stats-panel", "off", "--log-colours", "off"};
    std::vector<std::string> argv = {path_utf8(serve_executable()), path_utf8(model.path)};
    argv.insert(argv.end(), args.begin(), args.end());
    if (const std::optional<std::filesystem::path> cache = prefix_cache_file(config_id, args)) {
        argv.push_back("--prefix-cache-file");
        argv.push_back(path_utf8(*cache));
    }
    argv.insert(argv.end(), managed.begin(), managed.end());
    std::string command_line;
    for (const auto& [name, value] : env) {
#ifdef _WIN32
        command_line += "set \"" + name + "=" + value + "\" && ";
#else
        command_line += name + "=" + shell_quote(value) + " ";
#endif
    }
    command_line += command_line_for_display(argv);
    return {{"executable", path_utf8(serve_executable())},
            {"argv", argv},
            {"command_line", command_line},
            {"warnings", warnings},
            {"managed", Json::array({"--host", "--port", "--model-id", "--any-model-name", "--log-stats-panel",
                                     "--log-colours"})}};
}

} // namespace infernix::launcher
