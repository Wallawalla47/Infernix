// The launcher's management API (/launcher/api; Launcher members, launcher.h): its routes by area,
// registered in the order register_management_routes calls them (httplib answers a request with the first
// route that matches it), each behind admin_allowed unless it says otherwise.

#include "launcher/launcher.h"

#include "launcher/http_access.h"
#include "launcher/launch_script.h"
#include "launcher/process.h"
#include "launcher/route_support.h"
#include "launcher/sha256.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <regex>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace infernix::launcher {
namespace {

// A query parameter's unsigned value, `fallback` without it; std::invalid_argument (a 400) when it
// is not one in range.
std::uint64_t query_u64(const Request& request, const char* name, std::uint64_t fallback) {
    if (!request.has_param(name)) { return fallback; }
    const std::string text = request.get_param_value(name);
    std::uint64_t value    = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string(name) + " is not an unsigned integer in range");
    }
    return value;
}

// Folder listing for the folder picker.
Json list_folder(const std::string& path_text) {
    Json entries = Json::array();
    Json out     = {{"path", path_text}, {"parent", nullptr}, {"free_bytes", nullptr}, {"total_bytes", nullptr},
                    {"error", nullptr}};
    if (path_text.empty()) {
#ifdef _WIN32
        const DWORD drives = GetLogicalDrives();
        for (int i = 0; i < 26; ++i) {
            if ((drives & (1u << i)) == 0) { continue; }
            const std::string root = std::string(1, static_cast<char>('A' + i)) + ":\\";
            const UINT type        = GetDriveTypeA(root.c_str());
            if (type == DRIVE_NO_ROOT_DIR || type == DRIVE_CDROM) { continue; }
            entries.push_back({{"name", root}, {"path", root}, {"kind", "drive"}});
        }
#else
        entries.push_back({{"name", "/"}, {"path", "/"}, {"kind", "drive"}});
        if (const char* home = std::getenv("HOME")) {
            entries.push_back({{"name", std::string("~ (") + home + ")"}, {"path", home}, {"kind", "dir"}});
        }
#endif
        out["entries"] = entries;
        return out;
    }
    const std::filesystem::path path = utf8_path(path_text);
    std::error_code error;
    std::vector<std::pair<std::string, std::filesystem::path>> dirs;
    for (const auto& entry : std::filesystem::directory_iterator(
             path, std::filesystem::directory_options::skip_permission_denied, error)) {
        std::error_code status_error;
        if (!entry.is_directory(status_error)) { continue; }
        const std::string name = path_utf8(entry.path().filename());
        if (name.empty() || name.front() == '.' || name == "$RECYCLE.BIN" || name == "System Volume Information") {
            continue;
        }
        dirs.emplace_back(name, entry.path());
    }
    if (error) { out["error"] = error_text(error); }
    std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) { return lower(a.first) < lower(b.first); });
    for (const auto& [name, dir] : dirs) { entries.push_back({{"name", name}, {"path", path_utf8(dir)}, {"kind", "dir"}}); }
    out["entries"] = entries;
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty() && parent != path) {
        out["parent"] = path_utf8(parent);
    } else {
        out["parent"] = "";
    }
    const std::filesystem::space_info space = std::filesystem::space(path, error);
    if (!error) {
        out["free_bytes"]  = space.available;
        out["total_bytes"] = space.capacity;
    }
    return out;
}

// A management request's body: a JSON object, {} when empty; std::invalid_argument (a 400) for anything else.
Json body_json(const Request& request) {
    if (request.body.empty()) { return Json::object(); }
    Json value = parse_client_json(request.body);
    if (!value.is_object()) { throw std::invalid_argument("the request body must be a JSON object"); }
    return value;
}

} // namespace

httplib::Server::Handler Launcher::admin(httplib::Server::Handler handler) {
    return [this, handler = std::move(handler)](const Request& request, Response& response) {
        if (!admin_allowed(request, response)) { return; }
        try {
            handler(request, response);
        } catch (const std::invalid_argument& error) {
            send_error(response, 400, "invalid_request", error.what());
        } catch (const Json::exception& error) {
            send_error(response, 400, "invalid_json", error.what());
        } catch (const std::exception& error) { send_error(response, 502, "failed", error.what()); }
    };
}

void Launcher::register_management_routes() {
    const std::string api = "/launcher/api";
    register_settings_routes(api);
    register_config_routes(api);
    register_status_routes(api);
    register_download_routes(api);
    register_update_routes(api);
    register_disk_routes(api);
    register_key_routes(api);
    register_chat_routes(api);
    register_usage_routes(api);
    register_fs_routes(api);
    register_event_routes(api);
}

void Launcher::register_settings_routes(const std::string& api) {
    server_.Get(api + "/state", admin([this](const Request&, Response& s) { send_json(s, state_json()); }));
    server_.Get(api + "/settings",
                admin([this](const Request&, Response& s) { send_json(s, settings_to_json(store_.settings())); }));
    server_.Put(api + "/settings", admin([this](const Request& q, Response& s) {
                    const Json patch           = body_json(q);
                    const Settings before      = store_.settings();
                    if (store_.patch_settings(patch)) { restart_required_ = true; }
                    router_->clear_holds(); // a load that failed may succeed now (the server port, say)
                    const Settings after = store_.settings();
                    if (before.model_dirs != after.model_dirs || before.scan_depth != after.scan_depth) { request_scan(); }
                    // The event log names what changed (secrets only as set or cleared).
                    const Json old_values = settings_to_json(before), new_values = settings_to_json(after);
                    std::string changed;
                    for (const auto& [name, value] : new_values.items()) {
                        if (old_values.value(name, Json()) == value) { continue; }
                        changed += (changed.empty() ? "" : ", ") + name;
                        if (name != "api_key_set" && name != "hf_token_set") { changed += " = " + value.dump(); }
                    }
                    if (!changed.empty()) { event("info", "settings", "settings changed: " + changed); }
                    apply_log_dir();
                    const Json settings = settings_to_json(after);
                    publish("settings", settings);
                    send_json(s, settings);
                }));
    server_.Get(api + "/library", admin([this](const Request&, Response& s) { send_json(s, library_json()); }));
    server_.Post(api + "/library/rescan", admin([this](const Request&, Response& s) {
                     request_scan();
                     send_json(s, {{"ok", true}});
                 }));
    server_.Post(api + "/dirs", admin([this](const Request& q, Response& s) {
                     const std::string path = trim(json_or<std::string>(body_json(q), "path", ""));
                     std::error_code error;
                     if (path.empty() || !std::filesystem::is_directory(utf8_path(path), error)) {
                         throw std::invalid_argument("not a folder: " + path);
                     }
                     store_.update_settings([&](Settings& settings) {
                         if (std::find(settings.model_dirs.begin(), settings.model_dirs.end(), path) ==
                             settings.model_dirs.end()) {
                             settings.model_dirs.push_back(path);
                         }
                     });
                     event("info", "settings", "model folder added: " + path);
                     request_scan();
                     const Json settings = settings_to_json(store_.settings());
                     publish("settings", settings);
                     send_json(s, settings);
                 }));
    server_.Delete(api + "/dirs", admin([this](const Request& q, Response& s) {
                       const std::string path = q.get_param_value("path");
                       store_.update_settings([&](Settings& settings) {
                           settings.model_dirs.erase(
                               std::remove(settings.model_dirs.begin(), settings.model_dirs.end(), path),
                               settings.model_dirs.end());
                       });
                       event("info", "settings", "model folder removed: " + path);
                       request_scan();
                       const Json settings = settings_to_json(store_.settings());
                       publish("settings", settings);
                       send_json(s, settings);
                   }));
}

void Launcher::register_config_routes(const std::string& api) {
    // Model configurations: {id} is a configuration id (an artifact key, or "<key>@<8 hex>").
    const std::string config_path = R"(/models/([0-9a-f]{32}(?:@[0-9a-f]{8})?))";
    const auto with_config = [this](const Request& q, const std::function<void(const ConfigRef&, const LibraryScan&)>& f) {
        std::lock_guard lock(library_mutex_);
        const std::optional<ConfigRef> ref = find_config_locked(q.matches[1].str());
        if (!ref) { throw std::invalid_argument("no configuration " + q.matches[1].str()); }
        f(*ref, library_);
    };
    const auto parse_args = [this](const Json& value) {
        if (!value.is_array()) { throw std::invalid_argument("args must be an array"); }
        Args args;
        for (const Json& arg : value) {
            if (!arg.is_string()) { throw std::invalid_argument("args must be strings"); }
            args.push_back(arg.get<std::string>());
        }
        const ServeOptions serve = serve_options();
        strip_managed_flags(args, [&serve](std::string_view f) { return serve.takes_value(f); });
        return args;
    };
    const auto parse_env = [](const Json& value) {
        if (!value.is_array()) { throw std::invalid_argument("env must be an array of {name, value}"); }
        EnvVars env;
        for (const Json& variable : value) {
            const std::string name = trim(json_or<std::string>(variable, "name", ""));
            if (name.empty() || name.find('=') != std::string::npos || name.find_first_of(" \t\r\n") != std::string::npos) {
                throw std::invalid_argument("not an environment variable name: \"" + name + "\"");
            }
            env.emplace_back(name, json_or<std::string>(variable, "value", ""));
        }
        return env;
    };
    // A model id no other configuration uses (the library lock held).
    const auto check_alias = [this](const std::string& alias, const std::string& self, const LibraryScan& scan) {
        if (alias.find_first_of(" \t\r\n") != std::string::npos) {
            throw std::invalid_argument("the model id cannot contain spaces");
        }
        for (const ScannedModel& other : scan.models) {
            for (const std::string& id : config_ids(other)) {
                if (id != self && iequals(effective_alias(id, other, scan), alias)) {
                    throw std::invalid_argument("another configuration already uses the model id " + alias);
                }
            }
        }
    };
    const auto config_changed = [this](const Json& config, const std::string& what) {
        event("info", "settings", "configuration " + config.value("alias", "") + " " + what,
              {{"model_key", config.value("id", "")}, {"args", config.value("args", Json::array())}});
        publish("config", config);
        publish_library();
        publish_model_event("*", "models_reload"); // a model id or the listing may have changed
    };
    server_.Get(api + config_path + "/config", admin([this, with_config](const Request& q, Response& s) {
                    with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                        send_json(s, model_config_json(ref.id, *ref.model, scan));
                    });
                }));
    server_.Put(api + config_path + "/config",
                admin([this, with_config, parse_args, parse_env, check_alias, config_changed](const Request& q,
                                                                                                       Response& s) {
                    const Json patch = body_json(q);
                    const std::optional<Args> patch_args =
                        patch.contains("args") ? std::optional<Args>(parse_args(patch["args"])) : std::nullopt;
                    Json config;
                    with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                        const ScannedModel& m    = *ref.model;
                        const bool named         = is_named_config(ref.id);
                        StoredModelConfig stored = store_.model_config(ref.id).value_or(StoredModelConfig{});
                        if (patch.contains("name") && named) {
                            const std::string name = trim(json_or<std::string>(patch, "name", ""));
                            if (name.empty()) { throw std::invalid_argument("a configuration needs a name"); }
                            stored.name = name;
                        }
                        if (patch.contains("alias")) {
                            const std::string alias     = trim(json_or<std::string>(patch, "alias", ""));
                            const std::string def_alias = config_default_alias(ref.id, m, scan);
                            if (alias.empty() || alias == def_alias) {
                                stored.alias.reset();
                            } else {
                                check_alias(alias, ref.id, scan);
                                stored.alias = alias;
                            }
                        }
                        if (patch.contains("listed")) { stored.listed = json_or<bool>(patch, "listed", true); }
                        if (patch_args) {
                            // The default configuration follows the defaults until it differs; a named
                            // one keeps its own.
                            if (!named && *patch_args == model_default_args(m, scan)) {
                                stored.args.reset();
                            } else {
                                stored.args = *patch_args;
                            }
                        }
                        if (patch.contains("env")) { stored.env = parse_env(patch["env"]); }
                        if (patch.contains("idle_unload_min")) {
                            const Json& value = patch["idle_unload_min"];
                            if (value.is_null()) {
                                stored.idle_unload_min.reset();
                            } else if (value.is_number_integer() && value.get<int>() >= 0) {
                                stored.idle_unload_min = value.get<int>();
                            } else {
                                throw std::invalid_argument("idle_unload_min is minutes (0: never) or null");
                            }
                        }
                        stored.updated_at = now_ms();
                        store_.put_model_config(ref.id, stored);
                        router_->clear_hold(ref.id); // a changed configuration may load now
                        config = model_config_json(ref.id, m, scan);
                    });
                    config_changed(config, "saved");
                    send_json(s, config);
                }));
    server_.Post(api + R"(/models/([0-9a-f]{32})/configs)",
                 admin([this, with_config, parse_args, parse_env, check_alias, config_changed](const Request& q,
                                                                                                        Response& s) {
                     const Json body        = body_json(q);
                     const std::string name = trim(json_or<std::string>(body, "name", ""));
                     if (name.empty()) { throw std::invalid_argument("a configuration needs a name"); }
                     const std::optional<Args> body_args =
                         body.contains("args") ? std::optional<Args>(parse_args(body["args"])) : std::nullopt;
                     Json config;
                     with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                         // A copy of `from` (one of this artifact's configurations), else of the defaults.
                         std::string from = json_or<std::string>(body, "from", ref.id);
                         if (artifact_key_of(from) != ref.model->key || !find_config_locked(from)) { from = ref.id; }
                         StoredModelConfig stored;
                         stored.name = name;
                         stored.args = body_args ? *body_args : effective_args(from, *ref.model, scan);
                         stored.env  = body.contains("env") ? parse_env(body["env"]) : effective_env(from);
                         if (const std::optional<StoredModelConfig> source = store_.model_config(from)) {
                             stored.idle_unload_min = source->idle_unload_min;
                         }
                         const std::string id = new_config_id(ref.model->key);
                         if (const std::string alias = trim(json_or<std::string>(body, "alias", "")); !alias.empty()) {
                             check_alias(alias, id, scan);
                             stored.alias = alias;
                         }
                         stored.updated_at = now_ms();
                         store_.put_model_config(id, stored);
                         // Its derived model id must not clash with another configuration's.
                         if (!stored.alias) {
                             const std::string alias = config_default_alias(id, *ref.model, scan);
                             for (const ScannedModel& other : scan.models) {
                                 for (const std::string& other_id : config_ids(other)) {
                                     if (other_id != id && iequals(effective_alias(other_id, other, scan), alias)) {
                                         stored.alias = alias + "-" + id.substr(id.find('@') + 1, 4);
                                         store_.put_model_config(id, stored);
                                     }
                                 }
                             }
                         }
                         config = model_config_json(id, *ref.model, scan);
                     });
                     config_changed(config, "created");
                     send_json(s, config);
                 }));
    server_.Delete(api + config_path + "/config", admin([this, with_config](const Request& q, Response& s) {
                       std::string id, alias;
                       with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                           if (!is_named_config(ref.id)) {
                               throw std::invalid_argument("the default configuration cannot be deleted (restore it instead)");
                           }
                           id    = ref.id;
                           alias = effective_alias(ref.id, *ref.model, scan);
                       });
                       if (const InstanceSnapshot instance = supervisor_->snapshot();
                           instance.model_key == id && instance.state != InstanceState::Idle &&
                           instance.state != InstanceState::Failed) {
                           throw std::invalid_argument(alias + " is loaded: unload it first");
                       }
                       store_.erase_model_config(id);
                       event("info", "settings", "configuration " + alias + " deleted");
                       publish_library();
                       publish_model_event("*", "models_reload");
                       send_json(s, {{"ok", true}});
                   }));
    server_.Post(api + config_path + "/config/reset", admin([this, with_config, config_changed](const Request& q, Response& s) {
                     Json config;
                     with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                         const std::optional<StoredModelConfig> stored = store_.model_config(ref.id);
                         StoredModelConfig reset;
                         reset.listed     = stored ? stored->listed : true;
                         reset.updated_at = now_ms();
                         if (is_named_config(ref.id)) {
                             // A named configuration keeps its name and model id, with the defaults.
                             reset.name  = stored ? stored->name : std::nullopt;
                             reset.alias = stored ? stored->alias : std::nullopt;
                             reset.args  = model_default_args(*ref.model, scan);
                         }
                         store_.put_model_config(ref.id, reset);
                         router_->clear_hold(ref.id);
                         config = model_config_json(ref.id, *ref.model, scan);
                     });
                     config_changed(config, "restored to defaults");
                     send_json(s, config);
                 }));
    server_.Post(api + config_path + "/preview", admin([this, with_config, parse_env](const Request& q, Response& s) {
                     const Json body = body_json(q);
                     std::string id, alias;
                     ScannedModel model;
                     Args args;
                     EnvVars env;
                     with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                         id    = ref.id;
                         model = *ref.model;
                         alias = effective_alias(ref.id, *ref.model, scan);
                         args  = effective_args(ref.id, *ref.model, scan);
                         if (body.contains("args") && body["args"].is_array()) {
                             args.clear();
                             for (const Json& arg : body["args"]) {
                                 if (arg.is_string()) { args.push_back(arg.get<std::string>()); }
                             }
                         }
                         env = body.contains("env") ? parse_env(body["env"]) : effective_env(ref.id);
                     });
                     // Outside the library lock: it runs --help when not cached and looks for the files named.
                     send_json(s, preview_json(id, model, alias, std::move(args), env));
                 }));
    server_.Get(api + config_path + "/script", admin([this, with_config](const Request& q, Response& s) {
                    const ScriptShell shell = q.get_param_value("shell") == "sh" ? ScriptShell::Sh : ScriptShell::Cmd;
                    with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                        const std::string alias = effective_alias(ref.id, *ref.model, scan);
                        Args args               = effective_args(ref.id, *ref.model, scan);
                        const std::string text  = write_launch_script(shell, path_utf8(serve_executable()),
                                                                      path_utf8(ref.model->path), args,
                                                                      effective_env(ref.id), alias);
                        std::string file = "Launch-" + alias;
                        std::replace_if(file.begin(), file.end(), [](char c) { return std::string_view("\\/:*?\"<>|").find(c) != std::string_view::npos; }, '_');
                        send_json(s, {{"file_name", file + (shell == ScriptShell::Sh ? ".sh" : ".bat")}, {"text", text}});
                    });
                }));
    server_.Post(api + "/scripts/import", admin([this](const Request& q, Response& s) {
                     const Json body = body_json(q);
                     send_json(s, import_script_json(json_or<std::string>(body, "text", ""),
                                                     json_or<std::string>(body, "file_name", "")));
                 }));
    server_.Post(api + config_path + "/plan", admin([this, with_config, parse_args, parse_env](const Request& q, Response& s) {
                     const Json body = body_json(q);
                     const std::optional<Args> body_args =
                         body.contains("args") ? std::optional<Args>(parse_args(body["args"])) : std::nullopt;
                     std::string id;
                     ScannedModel model;
                     Args args;
                     EnvVars env;
                     with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                         id    = ref.id;
                         model = *ref.model;
                         args  = body_args ? *body_args : effective_args(ref.id, *ref.model, scan);
                         env   = body.contains("env") ? parse_env(body["env"]) : effective_env(ref.id);
                     });
                     send_json(s, fit_plan_json(id, model, args, env));
                 }));
    server_.Post(api + config_path + "/load", admin([this, with_config](const Request& q, Response& s) {
                     const bool force = json_or<bool>(body_json(q), "force", false);
                     std::string key, alias;
                     with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                         key   = ref.id;
                         alias = effective_alias(ref.id, *ref.model, scan);
                     });
                     event("info", "model", "load of " + alias + " requested from the launcher" + (force ? " (now)" : ""));
                     router_->request_load(key, force);
                     send_json(s, instance_json());
                 }));
    server_.Post(api + "/unload", admin([this](const Request&, Response& s) {
                     event("info", "model", "unload requested from the launcher");
                     router_->request_unload();
                     send_json(s, instance_json());
                 }));
    server_.Post(api + "/restart", admin([this](const Request&, Response& s) {
                     const InstanceSnapshot instance = supervisor_->snapshot();
                     if (instance.model_key.empty()) { throw std::invalid_argument("no model is loaded"); }
                     event("info", "model", "restart of " + instance.alias + " requested from the launcher");
                     router_->request_unload();
                     router_->request_load(instance.model_key, true);
                     send_json(s, instance_json());
                 }));
}

void Launcher::register_status_routes(const std::string& api) {
    // Live state.
    server_.Get(api + "/instance", admin([this](const Request&, Response& s) { send_json(s, instance_json()); }));
    server_.Get(api + "/logs", admin([this](const Request& q, Response& s) {
                    const std::uint64_t after = query_u64(q, "after", 0);
                    const auto limit = static_cast<std::size_t>(std::clamp<std::uint64_t>(query_u64(q, "limit", 2000), 1, 20000));
                    send_json(s, logs_.page(after, limit));
                }));
    server_.Get(api + "/eventlog", admin([this](const Request& q, Response& s) {
                    const std::uint64_t after = query_u64(q, "after", 0);
                    const auto limit = static_cast<std::size_t>(std::clamp<std::uint64_t>(query_u64(q, "limit", 2000), 1, 5000));
                    send_json(s, event_log_.page(after, limit));
                }));
    server_.Post(api + "/logs/open", admin([this](const Request&, Response& s) {
                     // Only the log folder: this opens a file manager on this computer.
                     open_in_browser(path_utf8(event_log_.dir()));
                     send_json(s, {{"ok", true}});
                 }));
    server_.Get(api + "/metrics", admin([this](const Request&, Response& s) { send_json(s, supervisor_->metrics()); }));
    server_.Get(api + "/telemetry", admin([this](const Request&, Response& s) { send_json(s, telemetry_->history()); }));
    server_.Get(api + "/activity", admin([this](const Request&, Response& s) { send_json(s, activity_json()); }));
    server_.Get(api + "/serve/options", admin([this](const Request&, Response& s) {
                    send_json(s, to_json(serve_options(), supervisor_->serve_build()));
                }));
}

void Launcher::register_download_routes(const std::string& api) {
    // Catalog and downloads.
    server_.Get(api + "/catalog", admin([this](const Request&, Response& s) { send_json(s, catalog_); }));
    server_.Get(api + "/hf/files", admin([this](const Request& q, Response& s) {
                    send_json(s, hf_files_json(trim(q.get_param_value("repo"))));
                }));
    server_.Get(api + "/hf/adapters", admin([this](const Request& q, Response& s) {
                    send_json(s, hf_adapters_json(trim(q.get_param_value("base")), trim(q.get_param_value("search"))));
                }));
    server_.Get(api + "/downloads", admin([this](const Request&, Response& s) { send_json(s, downloads_->list()); }));
    server_.Post(api + "/downloads", admin([this](const Request& q, Response& s) {
                     const Json request = body_json(q);
                     const Json job     = downloads_->create(request);
                     event("info", "download",
                           "download of " + job.value("repo", "") + " queued: " + std::to_string(job["files"].size()) +
                               " file(s), " + format_bytes(job.value("total", std::uint64_t{0})) + " into " +
                               job.value("target_dir", ""));
                     store_.update_settings([&](Settings& settings) {
                         settings.download_dir = json_or<std::string>(request, "dest_dir", settings.download_dir);
                     });
                     send_json(s, job);
                 }));
    server_.Post(api + R"(/downloads/([0-9a-f]+)/(pause|resume|cancel))", admin([this](const Request& q, Response& s) {
                     const std::string id = q.matches[1].str(), action = q.matches[2].str();
                     event("info", "download", "download " + id + ": " + action + " requested from the launcher");
                     if (action == "pause") { downloads_->pause(id); }
                     if (action == "resume") { downloads_->resume(id); }
                     if (action == "cancel") { downloads_->cancel(id); }
                     send_json(s, downloads_->list());
                 }));
    server_.Delete(api + R"(/downloads/([0-9a-f]+))", admin([this](const Request& q, Response& s) {
                       downloads_->remove(q.matches[1].str());
                       send_json(s, downloads_->list());
                   }));
}

void Launcher::register_update_routes(const std::string& api) {
    // Updates: models downloaded through the launcher, and Infernix itself.
    server_.Get(api + "/updates", admin([this](const Request&, Response& s) { send_json(s, model_updates_->state()); }));
    server_.Post(api + "/updates/check", admin([this](const Request&, Response& s) {
                     model_updates_->check();
                     send_json(s, model_updates_->state());
                 }));
    server_.Post(api + R"(/updates/([0-9a-f]+)/apply)", admin([this](const Request& q, Response& s) {
                     const std::optional<Json> update = model_updates_->update_for(q.matches[1].str());
                     if (!update) { throw std::invalid_argument("no update for that download"); }
                     // Files a running server has open cannot be replaced.
                     const std::string target = update->value("target_dir", "");
                     const InstanceSnapshot instance = supervisor_->snapshot();
                     if (instance.state != InstanceState::Idle && instance.state != InstanceState::Failed) {
                         std::lock_guard lock(library_mutex_);
                         if (const ScannedModel* loaded = library_.model(instance.artifact_key);
                             loaded != nullptr && same_or_inside(loaded->path.parent_path(), utf8_path(target), 0)) {
                             throw std::invalid_argument(instance.alias + " is loaded from that folder: unload it first");
                         }
                     }
                     Json files = Json::array();
                     for (const Json& file : update->value("files", Json::array())) { files.push_back(file.value("path", "")); }
                     const Json job = downloads_->create({{"repo", update->value("repo", "")},
                                                          {"dest_dir", target},
                                                          {"files", files},
                                                          {"subfolder", false},
                                                          {"replace", true}});
                     event("info", "download", "update of " + update->value("repo", "") + " queued: " +
                                                   std::to_string(files.size()) + " changed file(s)");
                     send_json(s, job);
                 }));
    server_.Get(api + "/infernix/release", admin([this](const Request&, Response& s) { send_json(s, infernix_update_->state()); }));
    server_.Post(api + "/infernix/update", admin([this](const Request&, Response& s) {
                     infernix_update_->download();
                     event("info", "launcher", "Infernix update download started");
                     send_json(s, infernix_update_->state());
                 }));
    server_.Post(api + "/infernix/install", admin([this](const Request&, Response& s) {
                     infernix_update_->install();
                     event("info", "launcher", "installing the Infernix update: the launcher stops and starts again");
                     send_json(s, infernix_update_->state());
                     // The answer goes out first; then an orderly stop (the model saves its caches) ends
                     // the process and the update script takes over. A member thread, joined by stop(), so a
                     // stop that finishes first never leaves it to use a destroyed launcher.
                     std::lock_guard lock(install_stopper_mutex_);
                     if (!install_stopper_.joinable() && !stopping_) {
                         install_stopper_ = std::thread([this] {
                             std::this_thread::sleep_for(std::chrono::milliseconds(500));
                             server_.stop();
                         });
                     }
                 }));
}

void Launcher::register_disk_routes(const std::string& api) {
    // Disk: deleting an artifact's files, a drafter, an n-gram volume or a LoRA adapter.
    const auto check_confirm = [](const Request& q, const std::string& file_name) {
        if (json_or<std::string>(body_json(q), "confirm", "") != file_name) {
            throw std::invalid_argument("type the file name (" + file_name + ") to confirm");
        }
    };
    const auto remove_files = [this](const std::vector<std::filesystem::path>& files, const std::string& what) {
        std::uint64_t freed = 0;
        std::vector<std::string> failed;
        for (const std::filesystem::path& file : files) {
            std::error_code error;
            const std::uint64_t size = std::filesystem::file_size(file, error);
            if (error) { continue; }
            if (std::filesystem::remove(file, error)) {
                freed += size;
            } else {
                failed.push_back(path_utf8(file) + (error ? " (" + error_text(error) + ")" : ""));
            }
        }
        event(failed.empty() ? "info" : "warn", "settings",
              "deleted " + what + " (" + format_bytes(freed) + " freed)" +
                  (failed.empty() ? "" : "; could not delete: " + failed.front()));
        if (!failed.empty()) { throw std::runtime_error("could not delete " + failed.front()); }
    };
    server_.Delete(api + R"(/models/([0-9a-f]{32})/files)", admin([this, check_confirm, remove_files](const Request& q, Response& s) {
                       std::vector<std::filesystem::path> files;
                       std::string file_name, key;
                       {
                           std::lock_guard lock(library_mutex_);
                           const ScannedModel* model = library_.model(q.matches[1].str());
                           if (model == nullptr) { throw std::invalid_argument("no model " + q.matches[1].str()); }
                           key       = model->key;
                           file_name = model->file_name;
                           files     = {model->path};
                           files.insert(files.end(), model->part_files.begin(), model->part_files.end());
                           for (const char* suffix : {".conversion.json", ".expert-state"}) {
                               std::filesystem::path sidecar = model->path;
                               sidecar += suffix;
                               files.push_back(sidecar);
                           }
                       }
                       // One snapshot, and the router's queued or starting load too (it acts up to 250 ms later):
                       // deleting the parts of a model that is starting leaves a half-deleted artifact.
                       const InstanceSnapshot instance = supervisor_->snapshot();
                       const auto names = [&key](const std::optional<std::string>& config) {
                           return config && artifact_key_of(*config) == key;
                       };
                       if ((instance.artifact_key == key && instance.state != InstanceState::Idle &&
                            instance.state != InstanceState::Failed) ||
                           (router_ && (names(router_->pending_model()) || names(router_->loading_model())))) {
                           throw std::invalid_argument(file_name + " is loaded or about to load: unload it first");
                       }
                       check_confirm(q, file_name);
                       remove_files(files, file_name);
                       for (const auto& [id, config] : store_.named_configs(key)) { store_.erase_model_config(id); }
                       store_.erase_model_config(key);
                       request_scan();
                       send_json(s, library_json());
                   }));
    server_.Delete(api + R"(/(drafters|ngram|adapters)/([0-9a-f]{32}))", admin([this, check_confirm, remove_files](const Request& q, Response& s) {
                       // The files to delete; what a running server would name (an adapter: its folder or its
                       // weights file); the name the user types to confirm.
                       std::vector<std::filesystem::path> files;
                       std::vector<std::filesystem::path> named_as;
                       std::filesystem::path folder;
                       std::string file_name;
                       {
                           std::lock_guard lock(library_mutex_);
                           const std::string kind = q.matches[1].str(), key = q.matches[2].str();
                           if (kind == "drafters") {
                               for (const ScannedDrafter& drafter : library_.drafters) {
                                   if (drafter.key == key) { files = named_as = {drafter.path}, file_name = drafter.file_name; }
                               }
                           } else if (kind == "ngram") {
                               for (const ScannedNgramVolume& volume : library_.ngram_volumes) {
                                   if (volume.key == key) { files = named_as = {volume.path}, file_name = volume.file_name; }
                               }
                           } else {
                               for (const ScannedAdapter& adapter : library_.adapters) {
                                   if (adapter.key != key) { continue; }
                                   folder    = adapter.path;
                                   files     = {adapter.path / "adapter_model.safetensors", adapter.path / "adapter_config.json"};
                                   named_as  = {adapter.path, files.front()};
                                   file_name = adapter.name;
                               }
                           }
                       }
                       if (files.empty()) { throw std::invalid_argument("no such file in the library"); }
                       // A running server may have it open.
                       const InstanceSnapshot instance = supervisor_->snapshot();
                       if (instance.state != InstanceState::Idle && instance.state != InstanceState::Failed) {
                           for (const std::string& arg : instance.argv) {
                               for (const std::filesystem::path& path : named_as) {
                                   if (iequals(arg, path_utf8(path))) {
                                       throw std::invalid_argument(file_name + " is in use by " + instance.alias + ": unload it first");
                                   }
                               }
                           }
                       }
                       check_confirm(q, file_name);
                       remove_files(files, file_name);
                       if (!folder.empty()) {
                           // The adapter's folder goes too when nothing else is left in it.
                           std::error_code ignored;
                           if (std::filesystem::is_empty(folder, ignored)) { std::filesystem::remove(folder, ignored); }
                       }
                       request_scan();
                       send_json(s, library_json());
                   }));
}

void Launcher::register_key_routes(const std::string& api) {
    // API keys and sign-in.
    server_.Post(api + "/keys", admin([this](const Request& q, Response& s) {
                     const Json body        = body_json(q);
                     const std::string name = trim(json_or<std::string>(body, "name", ""));
                     if (name.empty()) { throw std::invalid_argument("a key needs a name"); }
                     ApiKey key;
                     const std::string secret = "inx-" + random_token(32);
                     key.id                   = lower(random_token(8));
                     key.name                 = name;
                     key.hash                 = sha256_hex(secret);
                     key.prefix               = secret.substr(0, 8);
                     key.scope                = json_or<std::string>(body, "scope", "api") == "admin" ? "admin" : "api";
                     key.created_at           = now_ms();
                     store_.add_api_key(key);
                     event("info", "settings", "API key \"" + name + "\" created (" + key.scope + ")");
                     publish("settings", settings_to_json(store_.settings()));
                     send_json(s, {{"key",
                                    {{"id", key.id},
                                     {"name", key.name},
                                     {"prefix", key.prefix},
                                     {"scope", key.scope},
                                     {"created_at", key.created_at},
                                     {"last_used_at", nullptr}}},
                                   {"secret", secret}});
                 }));
    server_.Delete(api + R"(/keys/([0-9a-z]+))", admin([this](const Request& q, Response& s) {
                       const std::string id = q.matches[1].str();
                       std::string name     = id;
                       for (const ApiKey& key : store_.settings().api_keys) {
                           if (key.id == id) { name = key.name; }
                       }
                       if (!store_.remove_api_key(id)) { throw std::invalid_argument("no key " + id); }
                       {
                           // Its sign-ins end with it.
                           std::lock_guard lock(sessions_mutex_);
                           std::erase_if(sessions_, [&](const auto& entry) { return entry.second == id; });
                       }
                       event("info", "settings", "API key \"" + name + "\" revoked");
                       publish("settings", settings_to_json(store_.settings()));
                       send_json(s, {{"ok", true}});
                   }));
    // The session routes answer everyone (signing in is how a remote browser gets in), after the
    // checks against other web pages.
    server_.Get(api + "/session", [this](const Request& q, Response& s) {
        if (!page_checks(q, &s, false)) { return; }
        send_json(s, session_json(q));
    });
    server_.Post(api + "/session", [this](const Request& q, Response& s) {
        if (!page_checks(q, &s, false)) { return; }
        std::string secret;
        try {
            secret = trim(json_or<std::string>(Json::parse(q.body), "key", ""));
        } catch (const Json::exception&) {}
        const std::optional<ApiKey> key = store_.use_api_key(secret);
        if (!key || key->scope != "admin") {
            event("warn", "settings", "a sign-in from " + q.remote_addr + " was refused");
            send_error(s, 401, "sign_in", "that is not an admin key of this launcher");
            return;
        }
        const std::string token = random_token(40);
        {
            std::lock_guard lock(sessions_mutex_);
            sessions_[token] = key->id;
        }
        event("info", "settings", "signed in from " + q.remote_addr + " with the key \"" + key->name + "\"");
        s.set_header("Set-Cookie", "infernix_session=" + token + "; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000");
        send_json(s, {{"required", !is_loopback(q.remote_addr)}, {"signed_in", true}, {"key_name", key->name}});
    });
    server_.Delete(api + "/session", [this](const Request& q, Response& s) {
        if (!page_checks(q, &s, false)) { return; }
        {
            std::lock_guard lock(sessions_mutex_);
            sessions_.erase(session_cookie(q));
        }
        s.set_header("Set-Cookie", "infernix_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0");
        Json session         = session_json(q);
        session["signed_in"] = false;
        session["key_name"]  = nullptr;
        send_json(s, session);
    });
}

void Launcher::register_chat_routes(const std::string& api) {
    // Chats (the playground).
    server_.Get(api + "/chats", admin([this](const Request&, Response& s) { send_json(s, chats_.list()); }));
    server_.Get(api + R"(/chats/([A-Za-z0-9_-]{1,64}))", admin([this](const Request& q, Response& s) {
                    const std::optional<Json> chat = chats_.get(q.matches[1].str());
                    if (!chat) {
                        send_error(s, 404, "not_found", "no chat " + q.matches[1].str());
                        return;
                    }
                    send_json(s, *chat);
                }));
    server_.Put(api + R"(/chats/([A-Za-z0-9_-]{1,64}))", admin([this](const Request& q, Response& s) {
                    send_json(s, chats_.put(q.matches[1].str(), parse_client_json(q.body)));
                }));
    server_.Delete(api + R"(/chats/([A-Za-z0-9_-]{1,64}))", admin([this](const Request& q, Response& s) {
                       chats_.remove(q.matches[1].str());
                       send_json(s, {{"ok", true}});
                   }));
}

void Launcher::register_usage_routes(const std::string& api) {
    // Usage.
    server_.Get(api + "/usage", admin([this](const Request& q, Response& s) {
                    static const std::regex day(R"(\d{4}-\d{2}-\d{2})");
                    const std::string to   = q.has_param("to") ? q.get_param_value("to") : local_day(now_ms());
                    const std::string from = q.has_param("from") ? q.get_param_value("from")
                                                                 : local_day(now_ms() - 29ll * 86400000);
                    if (!std::regex_match(from, day) || !std::regex_match(to, day)) {
                        throw std::invalid_argument("from and to are YYYY-MM-DD");
                    }
                    send_json(s, {{"from", from}, {"to", to}, {"rows", usage_.rows(from, to)}});
                }));
}

void Launcher::register_fs_routes(const std::string& api) {
    // Folder picker.
    server_.Get(api + "/fs", admin([](const Request& q, Response& s) { send_json(s, list_folder(q.get_param_value("path"))); }));
    server_.Post(api + "/fs/mkdir", admin([](const Request& q, Response& s) {
                     const std::string path = trim(json_or<std::string>(body_json(q), "path", ""));
                     if (path.empty()) { throw std::invalid_argument("no folder given"); }
                     std::error_code error;
                     std::filesystem::create_directories(utf8_path(path), error);
                     if (error) { throw std::invalid_argument("cannot create " + path + ": " + error_text(error)); }
                     send_json(s, list_folder(path));
                 }));
}

void Launcher::register_event_routes(const std::string& api) {
    // The event stream.
    server_.Get(api + "/events", admin([this](const Request&, Response& s) {
                    auto subscription = events_.subscribe();
                    subscription->push(EventHub::format("hello", state_json()));
                    subscription->push(EventHub::format("library", library_json()));
                    subscription->push(EventHub::format("downloads", downloads_->list()));
                    subscription->push(EventHub::format("activity", activity_json()));
                    subscription->push(EventHub::format("metrics", supervisor_->metrics()));
                    subscription->push(EventHub::format("infernix_release", infernix_update_->state()));
                    subscription->push(EventHub::format("model_updates", model_updates_->state()));
                    s.set_header("Cache-Control", "no-cache");
                    s.set_header("X-Accel-Buffering", "no");
                    s.set_chunked_content_provider(
                        "text/event-stream",
                        [subscription](size_t, httplib::DataSink& sink) {
                            const std::optional<std::string> event = subscription->next(std::chrono::seconds(15));
                            if (!event) {
                                if (subscription->closed()) {
                                    sink.done();
                                    return true;
                                }
                                static constexpr std::string_view kPing = ": ping\n\n";
                                return sink.write(kPing.data(), kPing.size());
                            }
                            return sink.write(event->data(), event->size());
                        },
                        [subscription](bool) { subscription->close(); });
                }));
}

} // namespace infernix::launcher
