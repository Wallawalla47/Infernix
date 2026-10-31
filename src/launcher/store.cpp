#include "launcher/store.h"

#include "launcher/sha256.h"

#include <algorithm>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#endif

namespace infernix::launcher {
namespace {

constexpr int kSchema = 1;

std::filesystem::path home_dir() {
#ifdef _WIN32
    if (const wchar_t* profile = _wgetenv(L"USERPROFILE")) { return profile; }
#else
    if (const char* home = std::getenv("HOME")) { return home; }
#endif
    return std::filesystem::current_path();
}

Json read_json(const std::filesystem::path& path) {
    const std::optional<std::string> text = read_file(path);
    if (!text) { return Json::object(); }
    try {
        Json value = Json::parse(*text);
        return value.is_object() ? value : Json::object();
    } catch (const Json::exception&) {
        // Keep the unreadable file for the user and start from defaults.
        std::error_code ignored;
        std::filesystem::path aside = path;
        aside += ".unreadable";
        std::filesystem::rename(path, aside, ignored);
        std::fprintf(stderr, "launcher: %s was not valid JSON; moved aside, defaults used\n",
                     path_utf8(path).c_str());
        return Json::object();
    }
}

int clamp_int(const Json& value, int low, int high, int fallback) {
    if (!value.is_number()) { return fallback; }
    const double number = value.get<double>();
    if (!(number >= low && number <= high)) { return fallback; }
    return static_cast<int>(number);
}

void apply_settings(Settings& s, const Json& j) {
    const auto text = [&](const char* key, std::string& target) {
        if (j.contains(key) && j[key].is_string()) { target = j[key].get<std::string>(); }
    };
    const auto flag = [&](const char* key, bool& target) {
        if (j.contains(key) && j[key].is_boolean()) { target = j[key].get<bool>(); }
    };
    const auto number = [&](const char* key, int& target, int low, int high) {
        if (j.contains(key)) { target = clamp_int(j[key], low, high, target); }
    };
    text("host", s.host);
    number("port", s.port, 1, 65535);
    flag("cors", s.cors);
    flag("allow_remote_admin", s.allow_remote_admin);
    text("serve_executable", s.serve_executable);
    if (j.contains("model_dirs") && j["model_dirs"].is_array()) {
        std::vector<std::string> dirs;
        for (const Json& dir : j["model_dirs"]) {
            if (!dir.is_string()) { continue; }
            const std::string path = trim(dir.get<std::string>());
            if (!path.empty() && std::find(dirs.begin(), dirs.end(), path) == dirs.end()) { dirs.push_back(path); }
        }
        s.model_dirs = std::move(dirs);
    }
    number("scan_depth", s.scan_depth, 0, 6);
    text("download_dir", s.download_dir);
    number("child_port", s.child_port, 0, 65535);
    number("drain_timeout_s", s.drain_timeout_s, 0, 86400);
    number("load_timeout_s", s.load_timeout_s, 10, 86400);
    number("stop_timeout_s", s.stop_timeout_s, 5, 3600);
    number("idle_unload_min", s.idle_unload_min, 0, 100000);
    if (j.contains("model_routes") && j["model_routes"].is_array()) {
        std::vector<ModelRoute> routes;
        for (const Json& route : j["model_routes"]) {
            const std::string pattern = trim(json_or<std::string>(route, "pattern", ""));
            const std::string target  = trim(json_or<std::string>(route, "target", ""));
            if (!pattern.empty() && !target.empty()) { routes.push_back({pattern, target}); }
        }
        s.model_routes = std::move(routes);
    }
    flag("pin_loaded", s.pin_loaded);
    flag("llamacpp_api", s.llamacpp_api);
    if (j.contains("unknown_model") && j["unknown_model"].is_string()) {
        const std::string policy = j["unknown_model"].get<std::string>();
        if (policy == "loaded" || policy == "reject") { s.unknown_model = policy; }
    }
    if (j.contains("default_model")) {
        const Json& value = j["default_model"];
        if (value.is_null() || (value.is_string() && value.get<std::string>().empty())) {
            s.default_model.reset();
        } else if (value.is_string()) {
            s.default_model = value.get<std::string>();
        }
    }
    flag("autoload", s.autoload);
    number("download_connections", s.download_connections, 1, 32);
    flag("verify_downloads", s.verify_downloads);
    flag("open_browser", s.open_browser);
    flag("persist_prefix_cache", s.persist_prefix_cache);
    if (j.contains("log_dir") && j["log_dir"].is_string()) { s.log_dir = trim(j["log_dir"].get<std::string>()); }
    flag("log_bodies", s.log_bodies);
    flag("watchdog", s.watchdog);
    flag("restart_on_failure", s.restart_on_failure);
    number("hang_timeout_s", s.hang_timeout_s, 10, 3600);
    number("stall_timeout_s", s.stall_timeout_s, 0, 86400);
    number("load_ram_floor_mib", s.load_ram_floor_mib, 0, 262144);
    // The token: the file stores it as given; the API takes it write-only ("" clears).
    text("hf_token", s.hf_token);
    s.hf_token = trim(s.hf_token);
}

Json api_key_json(const ApiKey& key, bool with_hash) {
    Json j = {{"id", key.id},
              {"name", key.name},
              {"prefix", key.prefix},
              {"scope", key.scope},
              {"created_at", key.created_at},
              {"last_used_at", key.last_used_at ? Json(*key.last_used_at) : Json(nullptr)}};
    if (with_hash) { j["hash"] = key.hash; }
    return j;
}

Json settings_file_json(const Settings& s) {
    Json j = settings_to_json(s);
    j.erase("hf_token_set");
    j["hf_token"] = s.hf_token;
    Json keys     = Json::array();
    for (const ApiKey& key : s.api_keys) { keys.push_back(api_key_json(key, true)); }
    j["api_keys"] = keys;
    return j;
}

// The keys as the memory file holds them (with their hashes).
std::vector<ApiKey> read_api_keys(const Json& settings) {
    std::vector<ApiKey> keys;
    for (const Json& value : settings.value("api_keys", Json::array())) {
        ApiKey key;
        key.id         = json_or<std::string>(value, "id", "");
        key.name       = json_or<std::string>(value, "name", "");
        key.hash       = json_or<std::string>(value, "hash", "");
        key.prefix     = json_or<std::string>(value, "prefix", "");
        key.scope      = json_or<std::string>(value, "scope", "api") == "admin" ? "admin" : "api";
        key.created_at = json_or<std::int64_t>(value, "created_at", 0);
        if (value.contains("last_used_at") && value["last_used_at"].is_number_integer()) {
            key.last_used_at = value["last_used_at"].get<std::int64_t>();
        }
        if (!key.id.empty() && key.hash.size() == 64) { keys.push_back(std::move(key)); }
    }
    return keys;
}

} // namespace

Json settings_to_json(const Settings& s) {
    return {
        {"host", s.host},
        {"port", s.port},
        {"api_keys",
         [&] {
             Json keys = Json::array();
             for (const ApiKey& key : s.api_keys) { keys.push_back(api_key_json(key, false)); }
             return keys;
         }()},
        {"cors", s.cors},
        {"allow_remote_admin", s.allow_remote_admin},
        {"serve_executable", s.serve_executable},
        {"model_dirs", s.model_dirs},
        {"scan_depth", s.scan_depth},
        {"download_dir", s.download_dir},
        {"child_port", s.child_port},
        {"drain_timeout_s", s.drain_timeout_s},
        {"load_timeout_s", s.load_timeout_s},
        {"stop_timeout_s", s.stop_timeout_s},
        {"idle_unload_min", s.idle_unload_min},
        {"model_routes",
         [&] {
             Json routes = Json::array();
             for (const ModelRoute& route : s.model_routes) {
                 routes.push_back({{"pattern", route.pattern}, {"target", route.target}});
             }
             return routes;
         }()},
        {"pin_loaded", s.pin_loaded},
        {"llamacpp_api", s.llamacpp_api},
        {"unknown_model", s.unknown_model},
        {"default_model", s.default_model ? Json(*s.default_model) : Json(nullptr)},
        {"autoload", s.autoload},
        {"download_connections", s.download_connections},
        {"verify_downloads", s.verify_downloads},
        {"hf_token_set", !s.hf_token.empty()},
        {"open_browser", s.open_browser},
        {"persist_prefix_cache", s.persist_prefix_cache},
        {"log_dir", s.log_dir},
        {"log_bodies", s.log_bodies},
        {"watchdog", s.watchdog},
        {"restart_on_failure", s.restart_on_failure},
        {"hang_timeout_s", s.hang_timeout_s},
        {"stall_timeout_s", s.stall_timeout_s},
        {"load_ram_floor_mib", s.load_ram_floor_mib},
    };
}

std::filesystem::path user_data_dir() {
#ifdef _WIN32
    PWSTR local = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &local))) {
        std::filesystem::path base(local);
        CoTaskMemFree(local);
        return base / "Infernix" / "launcher";
    }
    if (local != nullptr) { CoTaskMemFree(local); }
    return home_dir() / "AppData" / "Local" / "Infernix" / "launcher";
#else
    if (const char* config = std::getenv("XDG_CONFIG_HOME"); config != nullptr && *config != '\0') {
        return std::filesystem::path(config) / "infernix" / "launcher";
    }
    return home_dir() / ".config" / "infernix" / "launcher";
#endif
}

bool writable_dir(const std::filesystem::path& dir) {
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    const std::filesystem::path probe = dir / ".infernix-launcher-write-test";
    {
        std::ofstream out(probe, std::ios::binary | std::ios::trunc);
        if (!out) { return false; }
        out << "ok";
        if (!out) { return false; }
    }
    std::filesystem::remove(probe, error);
    return true;
}

DataDirChoice choose_data_dir() {
    const std::filesystem::path beside = executable_dir();
    if (writable_dir(beside)) { return {beside, false}; }
    return {user_data_dir(), true};
}

Store::Store(std::filesystem::path data_dir) : data_dir_(std::move(data_dir)) {
    std::error_code ignored;
    std::filesystem::create_directories(data_dir_, ignored);
    load();
}

std::filesystem::path Store::memory_file() const { return data_dir_ / kMemoryFile; }

void Store::load() {
    std::lock_guard lock(mutex_);
    const Json memory   = read_json(memory_file());
    const Json settings = memory.value("settings", Json::object());
    apply_settings(settings_, settings);
    if (settings.is_object()) { settings_.api_keys = read_api_keys(settings); }
    const Json models = memory.value("models", Json::object());
    if (models.is_object()) {
        for (const auto& [key, value] : models.items()) {
            if (!value.is_object()) { continue; }
            StoredModelConfig config;
            if (value.contains("name") && value["name"].is_string()) { config.name = value["name"].get<std::string>(); }
            if (value.contains("alias") && value["alias"].is_string()) {
                config.alias = value["alias"].get<std::string>();
            }
            config.listed = json_or<bool>(value, "listed", true);
            if (value.contains("args") && value["args"].is_array()) {
                Args args;
                for (const Json& arg : value["args"]) {
                    if (arg.is_string()) { args.push_back(arg.get<std::string>()); }
                }
                config.args = std::move(args);
            }
            for (const Json& variable : value.value("env", Json::array())) {
                const std::string name = json_or<std::string>(variable, "name", "");
                if (!name.empty()) { config.env.emplace_back(name, json_or<std::string>(variable, "value", "")); }
            }
            if (value.contains("idle_unload_min") && value["idle_unload_min"].is_number_integer()) {
                config.idle_unload_min = value["idle_unload_min"].get<int>();
            }
            if (value.contains("updated_at") && value["updated_at"].is_number_integer()) {
                config.updated_at = value["updated_at"].get<std::int64_t>();
            }
            // A named configuration without its name or args is unusable.
            if (is_named_config(key) && (!config.name || !config.args)) { continue; }
            models_[key] = std::move(config);
        }
    }
    if (memory.contains("last_model") && memory["last_model"].is_string()) {
        last_model_ = memory["last_model"].get<std::string>();
    }
    downloads_ = memory.value("downloads", Json::array());
    if (!downloads_.is_array()) { downloads_ = Json::array(); }
    // First run: write the file so the user can find and back it up.
    if (!std::filesystem::exists(memory_file())) { save_locked(); }
}

Settings Store::settings() const {
    std::lock_guard lock(mutex_);
    return settings_;
}

bool Store::patch_settings(const Json& patch) {
    std::lock_guard lock(mutex_);
    const Settings before = settings_;
    apply_settings(settings_, patch);
    save_locked();
    return before.host != settings_.host || before.port != settings_.port || before.cors != settings_.cors ||
           before.allow_remote_admin != settings_.allow_remote_admin;
}

void Store::update_settings(const std::function<void(Settings&)>& change) {
    std::lock_guard lock(mutex_);
    change(settings_);
    save_locked();
}

std::optional<StoredModelConfig> Store::model_config(const std::string& key) const {
    std::lock_guard lock(mutex_);
    const auto it = models_.find(key);
    if (it == models_.end()) { return std::nullopt; }
    return it->second;
}

void Store::put_model_config(const std::string& key, const StoredModelConfig& config) {
    std::lock_guard lock(mutex_);
    models_[key] = config;
    save_locked();
}

void Store::erase_model_config(const std::string& key) {
    std::lock_guard lock(mutex_);
    models_.erase(key);
    save_locked();
}

std::vector<std::pair<std::string, StoredModelConfig>> Store::named_configs(const std::string& artifact_key) const {
    std::lock_guard lock(mutex_);
    std::vector<std::pair<std::string, StoredModelConfig>> out;
    for (const auto& [id, config] : models_) {
        if (is_named_config(id) && artifact_key_of(id) == artifact_key) { out.emplace_back(id, config); }
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return a.second.updated_at.value_or(0) < b.second.updated_at.value_or(0);
    });
    return out;
}

void Store::add_api_key(const ApiKey& key) {
    std::lock_guard lock(mutex_);
    settings_.api_keys.push_back(key);
    save_locked();
}

bool Store::remove_api_key(const std::string& id) {
    std::lock_guard lock(mutex_);
    const auto before = settings_.api_keys.size();
    std::erase_if(settings_.api_keys, [&](const ApiKey& key) { return key.id == id; });
    if (settings_.api_keys.size() == before) { return false; }
    save_locked();
    return true;
}

std::optional<ApiKey> Store::use_api_key(std::string_view secret) {
    if (secret.empty()) { return std::nullopt; }
    const std::string hash = sha256_hex(secret);
    std::lock_guard lock(mutex_);
    for (ApiKey& key : settings_.api_keys) {
        if (key.hash == hash) {
            key.last_used_at = now_ms();
            return key;
        }
    }
    return std::nullopt;
}

std::string artifact_key_of(std::string_view config_id) {
    return std::string(config_id.substr(0, config_id.find('@')));
}

bool is_named_config(std::string_view config_id) { return config_id.find('@') != std::string_view::npos; }

std::string new_config_id(const std::string& artifact_key) {
    std::random_device device;
    const std::uint32_t value = device();
    return artifact_key + "@" + hex_string(reinterpret_cast<const std::uint8_t*>(&value), sizeof(value));
}

std::optional<std::string> Store::last_model() const {
    std::lock_guard lock(mutex_);
    return last_model_;
}

void Store::set_last_model(const std::optional<std::string>& key) {
    std::lock_guard lock(mutex_);
    if (last_model_ == key) { return; }
    last_model_ = key;
    save_locked();
}

std::string Store::hf_token() const {
    {
        std::lock_guard lock(mutex_);
        if (!settings_.hf_token.empty()) { return settings_.hf_token; }
    }
    for (const char* name : {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}) {
        if (const char* value = std::getenv(name); value != nullptr && *value != '\0') { return trim(value); }
    }
    std::filesystem::path file;
    if (const char* hf_home = std::getenv("HF_HOME"); hf_home != nullptr && *hf_home != '\0') {
        file = std::filesystem::path(hf_home) / "token";
    } else {
        file = home_dir() / ".cache" / "huggingface" / "token";
    }
    const std::optional<std::string> token = read_file(file);
    return token ? trim(*token) : std::string{};
}

Json Store::downloads() const {
    std::lock_guard lock(mutex_);
    return downloads_;
}

void Store::set_downloads(const Json& jobs) {
    std::lock_guard lock(mutex_);
    if (downloads_ == jobs) { return; }
    downloads_ = jobs;
    save_locked();
}

void Store::save_locked() const {
    Json models = Json::object();
    for (const auto& [key, config] : models_) {
        Json value = {{"listed", config.listed}};
        if (config.name) { value["name"] = *config.name; }
        if (config.alias) { value["alias"] = *config.alias; }
        if (config.args) { value["args"] = *config.args; }
        if (!config.env.empty()) {
            Json env = Json::array();
            for (const auto& [name, setting] : config.env) { env.push_back({{"name", name}, {"value", setting}}); }
            value["env"] = env;
        }
        if (config.idle_unload_min) { value["idle_unload_min"] = *config.idle_unload_min; }
        if (config.updated_at) { value["updated_at"] = *config.updated_at; }
        models[key] = std::move(value);
    }
    const Json memory = {{"schema", kSchema},
                         {"settings", settings_file_json(settings_)},
                         {"models", models},
                         {"last_model", last_model_ ? Json(*last_model_) : Json(nullptr)},
                         {"downloads", downloads_}};
    try {
        write_file_atomic(memory_file(), memory.dump(2) + "\n");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "launcher: %s\n", error.what());
    }
}

} // namespace infernix::launcher
