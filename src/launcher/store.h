#pragma once

// Everything the launcher remembers between runs, in one file beside the launcher,
// infernix-launcher.json: its settings, each artifact's model id, listing and launch parameters
// (keyed by artifact id, so moving or renaming the files keeps them), the last model loaded and the
// download queue. The file is replaced atomically on every change.

#include "launcher/args.h"
#include "launcher/util.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace infernix::launcher {

// A named API key: the launcher keeps only the secret's SHA-256 and its first characters.
struct ApiKey {
    std::string id;     // 8 hex digits
    std::string name;
    std::string hash;   // SHA-256 of the secret, hex
    std::string prefix; // the secret's first 8 characters
    std::string scope = "api"; // api | admin
    std::int64_t created_at = 0;
    std::optional<std::int64_t> last_used_at;
};

// Requests naming a model that matches `pattern` (case-insensitive, `*` any run) go to `target`:
// a configuration id, or "loaded".
struct ModelRoute {
    std::string pattern;
    std::string target;
    bool operator==(const ModelRoute&) const = default;
};

struct Settings {
    std::string host                 = "127.0.0.1";
    int port                         = 1234;
    std::vector<ApiKey> api_keys;
    bool cors                        = true;
    bool allow_remote_admin          = false;
    std::string serve_executable;
    std::vector<std::string> model_dirs;
    int scan_depth                   = 2;
    std::string download_dir;
    int child_port                   = 0;
    int drain_timeout_s              = 120;
    int load_timeout_s               = 1200;
    int stop_timeout_s               = 180;
    int idle_unload_min              = 0;
    std::vector<ModelRoute> model_routes;
    bool pin_loaded                  = false;
    // Answer llama-server's router API (/props, /models/load, /models/unload, /models/sse) for
    // tools with a llama.cpp provider. Off: tools that tell servers apart by these paths (Hermes)
    // do not take the launcher for llama.cpp.
    bool llamacpp_api                = false;
    std::string unknown_model        = "loaded";
    std::optional<std::string> default_model;
    bool autoload                    = false;
    int download_connections         = 8;
    bool verify_downloads            = true;
    std::string hf_token;
    bool open_browser                = true;
    bool persist_prefix_cache        = false;
    std::string log_dir;             // "" = the launcher's data folder
    bool log_bodies                  = false;
    bool watchdog                    = true;
    bool restart_on_failure          = true;
    int hang_timeout_s               = 60;
    int stall_timeout_s              = 300;
    int load_ram_floor_mib           = 2048; // 0: off
};

// A configuration id: an artifact key (the artifact's default configuration) or
// "<artifact key>@<8 hex>" (a named configuration).
[[nodiscard]] std::string artifact_key_of(std::string_view config_id);
[[nodiscard]] bool is_named_config(std::string_view config_id);
[[nodiscard]] std::string new_config_id(const std::string& artifact_key);

using EnvVars = std::vector<std::pair<std::string, std::string>>;

// The persisted part of one configuration; absent members follow the computed defaults. A named
// configuration always has its name and its own args.
struct StoredModelConfig {
    std::optional<std::string> name;
    std::optional<std::string> alias;
    bool listed = true;
    std::optional<Args> args;
    EnvVars env;
    std::optional<int> idle_unload_min; // none: the global setting; 0: never
    std::optional<std::int64_t> updated_at;
};

class Store {
public:
    static constexpr const char* kMemoryFile = "infernix-launcher.json";

    explicit Store(std::filesystem::path data_dir);

    [[nodiscard]] const std::filesystem::path& data_dir() const noexcept { return data_dir_; }
    [[nodiscard]] std::filesystem::path memory_file() const;

    [[nodiscard]] Settings settings() const;
    // Applies a SettingsPatch-shaped JSON object; returns true when a restart-only member changed
    // (host, port, CORS or remote admin).
    bool patch_settings(const Json& patch);
    void update_settings(const std::function<void(Settings&)>& change);

    [[nodiscard]] std::optional<StoredModelConfig> model_config(const std::string& key) const;
    void put_model_config(const std::string& key, const StoredModelConfig& config);
    void erase_model_config(const std::string& key);
    // The named configurations of an artifact, oldest first.
    [[nodiscard]] std::vector<std::pair<std::string, StoredModelConfig>> named_configs(const std::string& artifact_key) const;

    // Named API keys (the secrets themselves are never stored).
    void add_api_key(const ApiKey& key);
    bool remove_api_key(const std::string& id);
    // The key whose secret this is; records its use (saved with the next change).
    [[nodiscard]] std::optional<ApiKey> use_api_key(std::string_view secret);

    [[nodiscard]] std::optional<std::string> last_model() const;
    void set_last_model(const std::optional<std::string>& key);

    // The download queue (owned by Downloads, kept here so one file holds everything).
    [[nodiscard]] Json downloads() const;
    void set_downloads(const Json& jobs);

    // The Hugging Face token: settings, else HF_TOKEN, else the huggingface_hub token file.
    [[nodiscard]] std::string hf_token() const;

private:
    void load();
    void save_locked() const;

    std::filesystem::path data_dir_;
    mutable std::mutex mutex_;
    Settings settings_;
    std::map<std::string, StoredModelConfig> models_;
    std::optional<std::string> last_model_;
    Json downloads_ = Json::array();
};

[[nodiscard]] Json settings_to_json(const Settings& settings);

// The folder for infernix-launcher.json (and, unless Settings::log_dir names another, the logs): the
// launcher's own folder, so the
// launcher is portable; when that folder cannot be written (Program Files, a read-only share), the
// user's %LOCALAPPDATA%\Infernix\launcher (Linux $XDG_CONFIG_HOME/infernix/launcher or
// ~/.config/infernix/launcher) with `fallback` set.
struct DataDirChoice {
    std::filesystem::path dir;
    bool fallback = false;
};
[[nodiscard]] DataDirChoice choose_data_dir();
[[nodiscard]] std::filesystem::path user_data_dir();

} // namespace infernix::launcher
