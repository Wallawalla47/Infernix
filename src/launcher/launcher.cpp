#include "launcher/launcher.h"

#include "launcher/assets.h"
#include "launcher/hf_hub.h"
#include "launcher/launch_script.h"
#include "launcher/load_advice.h"
#include "launcher/model_defaults.h"
#include "launcher/process.h"
#include "launcher/proxy.h"
#include "launcher/routing.h"
#include "launcher/sha256.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <regex>
#include <set>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace infernix::launcher {
namespace {

using Request  = httplib::Request;
using Response = httplib::Response;

constexpr std::size_t kActivityLimit = 100;
constexpr const char* kServeLogDir    = "serve-logs";

// Settings::log_dir, else the launcher's data folder.
std::filesystem::path resolve_log_dir(const Settings& settings, const std::filesystem::path& data_dir) {
    return settings.log_dir.empty() ? data_dir : utf8_path(settings.log_dir);
}

// "07:14:56" (local time).
std::string local_clock(std::int64_t ms) {
    const std::time_t seconds = static_cast<std::time_t>(ms / 1000);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &seconds);
#else
    localtime_r(&seconds, &local);
#endif
    char stamp[16];
    std::strftime(stamp, sizeof(stamp), "%H:%M:%S", &local);
    return stamp;
}

// "2.1 s", "340 ms".
std::string format_duration(std::int64_t ms) {
    char text[32];
    if (ms < 1000) {
        std::snprintf(text, sizeof(text), "%lld ms", static_cast<long long>(ms));
    } else if (ms < 600000) {
        std::snprintf(text, sizeof(text), "%.1f s", static_cast<double>(ms) / 1000.0);
    } else {
        std::snprintf(text, sizeof(text), "%.1f min", static_cast<double>(ms) / 60000.0);
    }
    return text;
}

void send_json(Response& response, const Json& body, int status = 200) {
    response.status = status;
    response.set_header("Cache-Control", "no-store");
    response.set_content(body.dump(-1, ' ', false, Json::error_handler_t::replace), "application/json");
}

void send_error(Response& response, int status, const std::string& code, const std::string& message) {
    send_json(response, {{"error", {{"message", message}, {"code", code}}}}, status);
}

void openai_error(Response& response, int status, const std::string& type, const std::string& code,
                  const std::string& message) {
    send_json(response, {{"error", {{"message", message}, {"type", type}, {"param", nullptr}, {"code", code}}}}, status);
}

void anthropic_error(Response& response, int status, const std::string& type, const std::string& message) {
    send_json(response, {{"type", "error"}, {"error", {{"type", type}, {"message", message}}}}, status);
}

// llama-server's error body: the HTTP status as `code`.
void llamacpp_error(Response& response, int status, const std::string& type, const std::string& message) {
    send_json(response, {{"error", {{"code", status}, {"message", message}, {"type", type}}}}, status);
}

// The default output limit: --default-max-tokens, else infernix-serve's 8,192; at most the context.
std::uint64_t default_output_tokens(const Args& args, std::uint64_t context) {
    std::uint64_t max_output = 8192;
    if (const std::optional<std::string> value = flag_value(args, "--default-max-tokens")) {
        try {
            max_output = std::stoull(*value);
        } catch (const std::exception&) {}
    }
    return std::min(max_output, context);
}

bool is_loopback(const std::string& address) {
    return address == "::1" || address.starts_with("127.") || address.starts_with("::ffff:127.") ||
           address == "localhost";
}

// The host of a Host header or an origin's authority: lower case, without the port or brackets.
std::string authority_host(std::string_view authority) {
    if (authority.starts_with('[')) {
        const std::size_t close = authority.find(']');
        return lower(authority.substr(1, close == std::string_view::npos ? std::string_view::npos : close - 1));
    }
    return lower(authority.substr(0, authority.find(':')));
}

bool ip_literal(const std::string& host) {
    if (host.find(':') != std::string::npos) { return true; } // IPv6
    int dots = 0;
    for (const char c : host) {
        if (c == '.') {
            ++dots;
        } else if (c < '0' || c > '9') {
            return false;
        }
    }
    return dots == 3;
}

bool loopback_host(const std::string& host) {
    return host == "localhost" || host.ends_with(".localhost") || host == "::1" || host.starts_with("127.");
}

// Claude Code's gateway model discovery lists only ids that contain "claude" or "anthropic", so it
// is offered each model as "<model id>:claude-code", which requests may name too. A suffix, unlike
// a "claude-" prefix, keeps Claude Code's CLAUDE_CODE_MAX_CONTEXT_TOKENS in effect for the id.
constexpr std::string_view kClaudeCodeSuffix = ":claude-code";

bool is_claude_code_client(const httplib::Request& request) {
    const std::string agent = lower(request.get_header_value("User-Agent"));
    return agent.starts_with("claude-cli/") || agent.starts_with("claude-code/");
}

// The model id inside Claude Code's spelling of it; nullopt for any other name.
std::optional<std::string> strip_claude_code(std::string_view name) {
    if (name.size() <= kClaudeCodeSuffix.size() ||
        !iequals(name.substr(name.size() - kClaudeCodeSuffix.size()), kClaudeCodeSuffix)) {
        return std::nullopt;
    }
    return std::string(name.substr(0, name.size() - kClaudeCodeSuffix.size()));
}

// The API key a request sends, as `Authorization: Bearer`, `x-api-key` or `api-key`; "" for none.
std::string presented_key(const httplib::Request& request) {
    std::string bearer = request.get_header_value("Authorization");
    if (bearer.size() > 7 && iequals(bearer.substr(0, 7), "Bearer ")) { return trim(bearer.substr(7)); }
    for (const char* header : {"x-api-key", "api-key"}) {
        if (std::string value = trim(request.get_header_value(header)); !value.empty()) { return value; }
    }
    return {};
}

// The session cookie's value; "" for none.
std::string session_cookie(const httplib::Request& request) {
    const std::string cookies = request.get_header_value("Cookie");
    std::size_t at            = 0;
    while (at < cookies.size()) {
        std::size_t end = cookies.find(';', at);
        if (end == std::string::npos) { end = cookies.size(); }
        const std::string pair = trim(std::string_view(cookies).substr(at, end - at));
        if (pair.starts_with("infernix_session=")) { return pair.substr(17); }
        at = end + 1;
    }
    return {};
}

// Random letters and digits from the system's random source.
std::string random_token(std::size_t length) {
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    std::random_device device;
    std::uniform_int_distribution<int> pick(0, static_cast<int>(sizeof(kAlphabet)) - 2);
    std::string token;
    for (std::size_t i = 0; i < length; ++i) { token.push_back(kAlphabet[pick(device)]); }
    return token;
}

std::string rfc3339(std::int64_t ms) {
    const std::time_t seconds = static_cast<std::time_t>(ms / 1000);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

std::string json_body_model(const std::string& body, bool& stream, bool& parsed) {
    parsed = false;
    stream = false;
    try {
        const Json value = Json::parse(body);
        parsed           = value.is_object();
        if (!parsed) { return {}; }
        stream = json_or<bool>(value, "stream", false);
        return json_or<std::string>(value, "model", "");
    } catch (const Json::exception&) { return {}; }
}

int pick_free_port() {
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    probe.stop();
    if (port <= 0) { throw std::runtime_error("no free loopback port for infernix-serve"); }
    return port;
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
    if (error) { out["error"] = error.message(); }
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

bool same_or_inside(const std::filesystem::path& inner, const std::filesystem::path& outer, int depth) {
    const std::string a = lower(path_utf8(inner.lexically_normal()));
    std::string b       = lower(path_utf8(outer.lexically_normal()));
    while (!b.empty() && (b.back() == '\\' || b.back() == '/')) { b.pop_back(); }
    if (a == b) { return true; }
    if (!a.starts_with(b) || a.size() <= b.size() || (a[b.size()] != '\\' && a[b.size()] != '/')) { return false; }
    const std::string rest = a.substr(b.size() + 1);
    const auto levels      = std::count_if(rest.begin(), rest.end(), [](char c) { return c == '\\' || c == '/'; });
    return levels < depth;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Construction

Launcher::Launcher(LauncherOptions options)
    : options_(std::move(options)), store_(options_.data_dir),
      event_log_(resolve_log_dir(store_.settings(), options_.data_dir)),
      logs_(resolve_log_dir(store_.settings(), options_.data_dir) / kServeLogDir),
      applied_log_dir_(resolve_log_dir(store_.settings(), options_.data_dir)), usage_(options_.data_dir),
      chats_(options_.data_dir / "chats") {
    try {
        catalog_ = Json::parse(std::string(reinterpret_cast<const char*>(assets::kCatalogJson), assets::kCatalogJsonSize));
    } catch (const Json::exception&) { catalog_ = {{"entries", Json::array()}}; }

    // The console shows the event log (but its debug records) and the server's errors; the web
    // UI shows everything.
    event_log_.set_listener([this](const EventRecord& record) {
        if (record.level != "debug") {
            std::fprintf(stdout, "%s\n", format_event_line(record).c_str() + 11); // without the date
            std::fflush(stdout);
        }
        std::lock_guard lock(pending_logs_mutex_);
        pending_events_.push_back(record);
    });
    logs_.set_listener([this](const std::vector<LogRecord>& records) {
        if (supervisor_) {
            for (const LogRecord& record : records) { supervisor_->observe(record); }
        }
        for (const LogRecord& record : records) {
            if (record.source == "server" && (record.level == "error" || record.level == "critical")) {
                std::fprintf(stdout, "%s  %-5s  serve | %s\n", local_clock(record.t).c_str(), record.level.c_str(),
                             record.text.c_str());
                std::fflush(stdout);
            }
        }
        std::lock_guard lock(pending_logs_mutex_);
        pending_logs_.insert(pending_logs_.end(), records.begin(), records.end());
    });

    supervisor_ = std::make_unique<Supervisor>(
        logs_, Supervisor::Callbacks{
                   .changed =
                       [this] {
                           const InstanceSnapshot snapshot = supervisor_->snapshot();
                           if ((snapshot.state == InstanceState::Idle || snapshot.state == InstanceState::Failed ||
                                snapshot.state == InstanceState::Stopping) &&
                               router_) {
                               router_->instance_ended(snapshot.instance);
                           }
                           // A start load_model retries on another port is not a failure to report.
                           if (snapshot.state == InstanceState::Failed && port_retry_due(snapshot)) { return; }
                           on_instance_changed(snapshot);
                           publish_instance();
                       },
                   .metrics =
                       [this](const Json& metrics) {
                           publish("metrics", metrics);
                           absorb_metrics(metrics);
                       },
                   .watchdog = [this] {
                       const Settings settings = store_.settings();
                       return WatchdogConfig{settings.watchdog, settings.hang_timeout_s, settings.stall_timeout_s,
                                             router_ && router_->in_flight() > 0};
                   },
                   .notice = [this](const std::string& message) { event("warn", "watchdog", message); },
               });

    router_ = std::make_unique<Router>(Router::Hooks{
        .load = [this](const std::string& key) { return load_model(key); },
        .stop = [this] { supervisor_->stop(std::chrono::seconds(store_.settings().stop_timeout_s)); },
        .changed =
            [this] {
                publish_instance();
                publish("activity", activity_json());
            },
        .drain_timeout_s = [this] { return store_.settings().drain_timeout_s; },
        .load_timeout_s  = [this] { return store_.settings().load_timeout_s; },
        .idle_unload_min =
            [this](const std::string& key) {
                // The configuration's own limit, else the global one.
                const std::optional<StoredModelConfig> stored = store_.model_config(key);
                if (stored && stored->idle_unload_min) { return *stored->idle_unload_min; }
                return store_.settings().idle_unload_min;
            },
    });

    telemetry_ = std::make_unique<Telemetry>(
        [this]() -> std::optional<std::pair<ServerReading, double>> {
            const InstanceSnapshot snapshot = supervisor_->snapshot();
            if (!snapshot.usage) { return std::nullopt; }
            ServerReading reading;
            reading.working_set   = snapshot.usage->working_set;
            reading.private_bytes = snapshot.usage->private_bytes;
            return std::make_pair(reading, snapshot.usage->cpu_seconds);
        },
        [this](const SystemReading& reading) {
            publish("system", to_json(reading));
            guard_load_memory(reading);
        });

    downloads_ = std::make_unique<Downloads>(
        Downloads::Hooks{
            .token       = [this] { return store_.hf_token(); },
            .connections = [this] { return store_.settings().download_connections; },
            .verify      = [this] { return store_.settings().verify_downloads; },
            .finished =
                [this](const std::filesystem::path& dest, const std::filesystem::path& target, const std::string& repo) {
                    event("info", "download", "downloaded " + repo + " into " + path_utf8(target));
                    download_finished(dest, target);
                    toast("success", "Downloaded " + repo + " into " + path_utf8(target));
                },
            .failed =
                [this](const std::string& repo, const std::string& error) {
                    event("error", "download", "download of " + repo + " failed: " + error);
                },
            .changed    = [this](const Json& jobs) { publish("downloads", jobs); },
            .load_state = [this] { return store_.downloads(); },
            .save_state = [this](const Json& jobs) { store_.set_downloads(jobs); },
        });

    model_updates_ = std::make_unique<ModelUpdates>(ModelUpdates::Hooks{
        .jobs    = [this] { return downloads_->list(); },
        .token   = [this] { return store_.hf_token(); },
        .changed = [this](const Json& state) { publish("model_updates", state); },
    });
    infernix_update_ = std::make_unique<InfernixUpdate>(InfernixUpdate::Options{
        .build_id     = options_.version,
        .install_dir  = executable_dir(),
        .work_dir     = options_.data_dir / "updates",
        .restart_args = options_.restart_args,
        .changed      = [this](const Json& state) { publish("infernix_release", state); },
    });

    scanner_     = std::thread([this] { scan_loop(); });
    log_flusher_ = std::thread([this] { flush_logs_loop(); });
    register_routes();
}

Launcher::~Launcher() { stop(); }

bool Launcher::bind(std::string& error) {
    const Settings settings = store_.settings();
    bound_host_             = options_.host.value_or(settings.host);
    bound_port_             = options_.port.value_or(settings.port);
    if (!server_.bind_to_port(bound_host_, bound_port_)) {
        error = "cannot listen on " + bound_host_ + ":" + std::to_string(bound_port_) +
                " (another program, such as LM Studio, may use it; choose another with --port)";
        return false;
    }
    return true;
}

std::string Launcher::event_log_file() const { return path_utf8(event_log_.events_file()); }

std::string Launcher::ui_url() const {
    const std::string host = bound_host_ == "0.0.0.0" || bound_host_ == "::" || bound_host_.empty() ? "127.0.0.1" : bound_host_;
    return "http://" + (host.find(':') != std::string::npos ? "[" + host + "]" : host) + ":" + std::to_string(bound_port_);
}

void Launcher::serve() {
    const Settings settings = store_.settings();
    event("info", "launcher",
          "infernix-launcher " + options_.version + " started on " + bound_host_ + ":" + std::to_string(bound_port_) +
              " (memory " + path_utf8(store_.memory_file()) + ")");
    if (!is_loopback(bound_host_) && settings.api_keys.empty()) {
        event("warn", "launcher",
              "listening on " + bound_host_ + " without an API key: anyone who can reach the port can use the models");
    }
    if (settings.autoload) {
        std::optional<std::string> key = settings.default_model ? settings.default_model : store_.last_model();
        if (key) {
            std::thread([this, key = *key] {
                // Let the first scan finish so the model is known.
                for (int i = 0; i < 100; ++i) {
                    {
                        std::lock_guard lock(library_mutex_);
                        if (library_.scanned_at != 0) { break; }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                router_->request_load(key, false);
            }).detach();
        }
    }
    if (options_.open_browser && settings.open_browser) { open_in_browser(ui_url()); }
    server_.listen_after_bind();
}

void Launcher::stop() {
    // A second caller (Ctrl+C beside the end of serve()) waits for the first to finish.
    std::lock_guard guard(stop_mutex_);
    if (stopped_) { return; }
    event("info", "launcher", "shutting down");
    stopping_ = true;
    events_.close_all();
    model_events_.close_all();
    server_.stop();
    // A model still loading is abandoned (and stopped by the router's thread) instead of awaited.
    if (supervisor_) { supervisor_->shutdown(); }
    if (router_) { router_->shutdown(); }
    if (supervisor_) { supervisor_->stop(std::chrono::seconds(store_.settings().stop_timeout_s)); }
    if (downloads_) { downloads_->shutdown(); }
    if (model_updates_) { model_updates_->shutdown(); }
    if (infernix_update_) { infernix_update_->shutdown(); }
    if (telemetry_) { telemetry_->stop(); }
    {
        std::lock_guard lock(library_mutex_);
        scan_requested_ = true;
    }
    scan_cv_.notify_all();
    if (scanner_.joinable()) { scanner_.join(); }
    if (log_flusher_.joinable()) { log_flusher_.join(); }
    usage_.flush();
    // The API keys' last use is kept in memory between other changes.
    store_.update_settings([](Settings&) {});
    stopped_ = true;
}

// ---------------------------------------------------------------------------------------------
// Events

void Launcher::publish(const std::string& name, const Json& data) {
    if (stopping_) { return; }
    events_.publish(name, data);
}

void Launcher::toast(const std::string& level, const std::string& message) {
    publish("toast", {{"level", level}, {"message", message}});
}

void Launcher::flush_logs_loop() {
    while (!stopping_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        std::vector<LogRecord> batch;
        std::vector<EventRecord> events;
        {
            std::lock_guard lock(pending_logs_mutex_);
            batch.swap(pending_logs_);
            events.swap(pending_events_);
        }
        if (!events.empty()) {
            std::sort(events.begin(), events.end(), [](const EventRecord& a, const EventRecord& b) { return a.seq < b.seq; });
            Json lines = Json::array();
            for (const EventRecord& record : events) { lines.push_back(to_json(record)); }
            publish("eventlog", {{"lines", lines}});
        }
        if (batch.empty()) { continue; }
        // Listeners run outside the log lock, so records from two threads can arrive swapped.
        std::sort(batch.begin(), batch.end(), [](const LogRecord& a, const LogRecord& b) { return a.seq < b.seq; });
        Json lines = Json::array();
        for (const LogRecord& record : batch) { lines.push_back(to_json(record)); }
        publish("log", {{"lines", lines}});
    }
}

Json Launcher::state_json() const {
    const Settings settings = store_.settings();
    const std::string base  = ui_url();
    return {{"version", options_.version},
            {"data_dir", path_utf8(options_.data_dir)},
            {"memory_file", path_utf8(store_.memory_file())},
            {"logs", log_paths_json()},
            {"endpoints", {{"openai", base + "/v1"}, {"anthropic", base}}},
            {"listen", {{"host", bound_host_}, {"port", bound_port_}}},
            {"restart_required", restart_required_.load()},
            {"settings", settings_to_json(settings)},
            {"instance", instance_json()},
            {"system", to_json(telemetry_->facts())}};
}

// ---------------------------------------------------------------------------------------------
// Library

void Launcher::request_scan() {
    {
        std::lock_guard lock(library_mutex_);
        scan_requested_ = true;
    }
    scan_cv_.notify_all();
}

void Launcher::scan_loop() {
    // Runs `infernix-serve --help` once now, so no API call waits for it under the library lock.
    (void)serve_options();
    std::unique_lock lock(library_mutex_);
    while (!stopping_) {
        // Rescan on request, and every minute to notice files copied in by hand.
        scan_cv_.wait_for(lock, std::chrono::seconds(60), [this] { return scan_requested_; });
        if (stopping_) { break; }
        const bool requested = scan_requested_;
        scan_requested_      = false;
        scanning_            = true;
        lock.unlock();
        if (requested) { publish("library", library_json()); }
        const Settings settings = store_.settings();
        std::vector<std::filesystem::path> dirs;
        for (const std::string& dir : settings.model_dirs) { dirs.push_back(utf8_path(dir)); }
        LibraryScan scan = scan_library(dirs, settings.scan_depth);
        lock.lock();
        library_  = std::move(scan);
        scanning_ = false;
        lock.unlock();
        Json library = library_json();
        Json compare = library;
        compare.erase("scanned_at");
        const std::string digest = compare.dump();
        lock.lock();
        const bool changed = digest != library_digest_;
        library_digest_    = digest;
        lock.unlock();
        if (changed || requested) {
            publish("library", library);
            publish("activity", activity_json());
        }
        if (changed) { publish_model_event("*", "models_reload"); }
        lock.lock();
    }
}

Args Launcher::effective_args(const std::string& config_id, const ScannedModel& model, const LibraryScan& scan) const {
    const std::optional<StoredModelConfig> stored = store_.model_config(config_id);
    if (stored && stored->args) { return *stored->args; }
    return model_default_args(model, scan);
}

EnvVars Launcher::effective_env(const std::string& config_id) const {
    const std::optional<StoredModelConfig> stored = store_.model_config(config_id);
    return stored ? stored->env : EnvVars{};
}

std::string Launcher::config_name(const std::string& config_id) const {
    if (!is_named_config(config_id)) { return "Default"; }
    const std::optional<StoredModelConfig> stored = store_.model_config(config_id);
    return stored && stored->name ? *stored->name : config_id;
}

std::vector<std::string> Launcher::config_ids(const ScannedModel& model) const {
    std::vector<std::string> ids = {model.key};
    for (const auto& [id, config] : store_.named_configs(model.key)) { ids.push_back(id); }
    return ids;
}

std::string Launcher::config_default_alias(const std::string& config_id, const ScannedModel& model,
                                           const LibraryScan& scan) const {
    const std::string base = default_alias(scan, model);
    if (!is_named_config(config_id)) { return base; }
    // "<artifact id>-<name>": the name lowered, its spaces and symbols as single dashes.
    std::string slug;
    for (const char c : lower(config_name(config_id))) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_') {
            slug.push_back(c);
        } else if (!slug.empty() && slug.back() != '-') {
            slug.push_back('-');
        }
    }
    while (!slug.empty() && slug.back() == '-') { slug.pop_back(); }
    return base + "-" + (slug.empty() ? config_id.substr(config_id.find('@') + 1) : slug);
}

Args Launcher::model_default_args(const ScannedModel& model, const LibraryScan& scan) const {
    const SystemFacts facts = telemetry_->facts();
    HostFacts host;
    host.ram_total_bytes = facts.ram_total;
    if (!facts.gpus.empty()) { host.vram_total_bytes = facts.gpus.front().second.second; }
    return default_args(scan, model, host);
}

std::string Launcher::effective_alias(const std::string& config_id, const ScannedModel& model,
                                      const LibraryScan& scan) const {
    const std::optional<StoredModelConfig> stored = store_.model_config(config_id);
    if (stored && stored->alias && !stored->alias->empty()) { return *stored->alias; }
    return config_default_alias(config_id, model, scan);
}

Json Launcher::model_config_json(const std::string& config_id, const ScannedModel& model, const LibraryScan& scan) const {
    const std::optional<StoredModelConfig> stored = store_.model_config(config_id);
    const bool named            = is_named_config(config_id);
    const Args defaults         = model_default_args(model, scan);
    const Args args             = stored && stored->args ? *stored->args : defaults;
    const std::string def_alias = config_default_alias(config_id, model, scan);
    const std::string alias     = stored && stored->alias && !stored->alias->empty() ? *stored->alias : def_alias;
    Json env                    = Json::array();
    if (stored) {
        for (const auto& [name, value] : stored->env) { env.push_back({{"name", name}, {"value", value}}); }
    }
    return {{"id", config_id},
            {"key", model.key},
            {"name", config_name(config_id)},
            {"is_default", !named},
            {"alias", alias},
            {"default_alias", def_alias},
            {"listed", stored ? stored->listed : true},
            {"args", args},
            {"default_args", defaults},
            {"env", env},
            {"idle_unload_min", stored && stored->idle_unload_min ? Json(*stored->idle_unload_min) : Json(nullptr)},
            {"customized", named || args != defaults || alias != def_alias || !env.empty()},
            {"context_length", context_length(args)},
            {"vision", model.has_vision && has_flag(args, "--vision")},
            {"updated_at", stored && stored->updated_at ? Json(*stored->updated_at) : Json(nullptr)}};
}

Json Launcher::library_json() const {
    std::lock_guard lock(library_mutex_);
    Json library        = to_json(library_);
    library["scanning"] = scanning_;
    for (Json& model : library["models"]) {
        if (const ScannedModel* scanned = library_.model(model["key"].get<std::string>())) {
            Json configs = Json::array();
            for (const std::string& id : config_ids(*scanned)) { configs.push_back(model_config_json(id, *scanned, library_)); }
            model["config"]  = configs.front();
            model["configs"] = configs;
        }
    }
    // Each folder's share of the library and its drive's room.
    for (Json& dir : library["dirs"]) {
        const std::filesystem::path path = utf8_path(dir["path"].get<std::string>());
        std::uint64_t bytes              = 0;
        const auto inside = [&](const std::filesystem::path& file) { return same_or_inside(file.parent_path(), path, 64); };
        for (const ScannedModel& model : library_.models) {
            if (inside(model.path)) { bytes += model.bytes; }
        }
        for (const ScannedDrafter& drafter : library_.drafters) {
            if (inside(drafter.path)) { bytes += drafter.bytes; }
        }
        for (const ScannedNgramVolume& volume : library_.ngram_volumes) {
            if (inside(volume.path)) { bytes += volume.bytes; }
        }
        dir["library_bytes"] = bytes;
        std::error_code error;
        const std::filesystem::space_info space = std::filesystem::space(path, error);
        dir["free_bytes"]  = error ? Json(nullptr) : Json(space.available);
        dir["total_bytes"] = error ? Json(nullptr) : Json(space.capacity);
    }
    return library;
}

std::optional<Launcher::ConfigRef> Launcher::find_config_locked(const std::string& config_id) const {
    const ScannedModel* model = library_.model(artifact_key_of(config_id));
    if (model == nullptr) { return std::nullopt; }
    if (is_named_config(config_id) && !store_.model_config(config_id)) { return std::nullopt; }
    return ConfigRef{config_id, model};
}

std::filesystem::path Launcher::serve_executable() const {
    if (options_.serve_executable) { return *options_.serve_executable; }
    const Settings settings = store_.settings();
    if (!settings.serve_executable.empty()) {
        // A relative setting is beside the launcher (the folder travels with its memory file).
        const std::filesystem::path configured = utf8_path(settings.serve_executable);
        return configured.is_relative() ? executable_dir() / configured : configured;
    }
#ifdef _WIN32
    return executable_dir() / "infernix-serve.exe";
#else
    return executable_dir() / "infernix-serve";
#endif
}

ServeOptions Launcher::serve_options() const { return load_serve_options(serve_executable()); }

std::optional<std::string> Launcher::resolve_model(const std::string& name) const {
    if (name.empty()) { return std::nullopt; }
    std::lock_guard lock(library_mutex_);
    // Every configuration's model id first, then the artifact's other names (its default).
    for (const ScannedModel& model : library_.models) {
        for (const std::string& id : config_ids(model)) {
            if (iequals(effective_alias(id, model, library_), name) || iequals(id, name)) { return id; }
        }
    }
    for (const ScannedModel& model : library_.models) {
        std::string stem = path_utf8(model.path.stem());
        if (iequals(model.name, name) || iequals(model.file_name, name) || iequals(stem, name) ||
            iequals(default_alias(library_, model), name)) {
            return model.key;
        }
    }
    // Claude Code's spelling of a model id (its model picker lists them so).
    if (const std::optional<std::string> base = strip_claude_code(name)) {
        for (const ScannedModel& model : library_.models) {
            for (const std::string& id : config_ids(model)) {
                if (iequals(effective_alias(id, model, library_), *base)) { return id; }
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> Launcher::alias_owner(const std::string& alias) const {
    std::lock_guard lock(library_mutex_);
    for (const ScannedModel& model : library_.models) {
        for (const std::string& id : config_ids(model)) {
            if (iequals(effective_alias(id, model, library_), alias)) { return id; }
        }
    }
    return std::nullopt;
}

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
                             const EnvVars& env) const {
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
    const std::optional<CaptureResult> run = run_capture(serve_executable(), argv, std::chrono::minutes(3), env);
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
        if (out["max_context"].is_number_unsigned() && out["max_context"].get<std::uint64_t>() < context_length(args) &&
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

Json Launcher::preview_json(const std::string& config_id, const ScannedModel& model, const LibraryScan& scan,
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
    for (const char* flag : {"--draft-model", "--ngram-volume", "--chat-template", "--expert-stripe",
                             "--context-cost-presets"}) {
        if (const std::optional<std::string> value = flag_value(args, flag)) {
            std::error_code error;
            if (!std::filesystem::exists(utf8_path(*value), error)) {
                warnings.push_back(std::string(flag) + ": file not found: " + *value);
            }
        }
    }
    if (model.needs_ngram_volume && !model.own_ngram_volume && !has_flag(args, "--ngram-volume")) {
        warnings.push_back(model.family_label +
                           " needs an n-gram volume (--ngram-volume): none was found in the model folders; download one "
                           "in Discover");
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
    const std::vector<std::string> managed = {"--host", "127.0.0.1", "--port", "<auto>", "--model-id",
                                              effective_alias(config_id, model, scan), "--any-model-name",
                                              "--log-stats-panel", "off", "--log-colours", "off"};
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

// ---------------------------------------------------------------------------------------------
// Model lifecycle

std::optional<std::filesystem::path> Launcher::prefix_cache_file(const std::string& config_id, const Args& args) const {
    if (!store_.settings().persist_prefix_cache || has_flag(args, "--prefix-cache-file") ||
        has_flag(args, "--no-prefix-reuse")) {
        return std::nullopt;
    }
    std::string name = config_id;
    std::replace(name.begin(), name.end(), '@', '-');
    return options_.data_dir / "launcher-prefix-cache" / (name + ".cache");
}

Router::Loaded Launcher::load_model(const std::string& key) {
    Router::Loaded result;
    LibraryScan scan;
    {
        std::lock_guard lock(library_mutex_);
        scan = library_;
    }
    const ScannedModel* model = scan.model(artifact_key_of(key));
    if (model == nullptr || (is_named_config(key) && !store_.model_config(key))) {
        result.error = "the model is not in the library (was its folder removed?)";
        return result;
    }
    if (!model->missing_files.empty()) {
        result.error = model->file_name + " has missing or incomplete parts";
        return result;
    }
    if (model->family == "unknown") {
        result.error = model->file_name + " is a " + model->model_type + " model, which Infernix does not run";
        return result;
    }
    const std::filesystem::path executable = serve_executable();
    std::error_code error;
    if (!std::filesystem::is_regular_file(executable, error)) {
        result.error = "infernix-serve was not found at " + path_utf8(executable) + " (Settings, Server)";
        return result;
    }
    const Settings settings = store_.settings();
    supervisor_->stop(std::chrono::seconds(settings.stop_timeout_s));

    const ServeOptions serve = serve_options();
    const TakesValue takes   = [&serve](std::string_view flag) { return serve.takes_value(flag); };
    Args args                = effective_args(key, *model, scan);
    strip_managed_flags(args, takes);
    const std::string alias = effective_alias(key, *model, scan);

    LaunchSpec spec;
    spec.model_key    = key;
    spec.artifact_key = model->key;
    spec.config_name  = is_named_config(key) ? config_name(key) : std::string();
    spec.alias        = alias;
    spec.name         = model->name;
    spec.executable   = executable;
    spec.working_dir  = options_.data_dir;
    spec.env          = effective_env(key);
    // Absolute, as the server starts in the data folder (a model folder may be named relatively).
    spec.args = {path_utf8(std::filesystem::absolute(model->path, error))};
    spec.args.insert(spec.args.end(), args.begin(), args.end());
    if (const std::optional<std::filesystem::path> cache = prefix_cache_file(key, args)) {
        std::filesystem::create_directories(cache->parent_path(), error);
        spec.args.push_back("--prefix-cache-file");
        spec.args.push_back(path_utf8(*cache));
    }
    const std::size_t launch_args = spec.args.size(); // the managed flags, the port among them, follow
    // Managed flags; options an older infernix-serve lacks are left out.
    const auto supported = [&serve](std::string_view flag) { return serve.options.empty() || serve.known(flag); };

    // infernix-serve binds its port exclusively and exits before loading when another program holds
    // it. An automatic port is free when picked, but another program can take it in the moment
    // before the server binds it: that start is tried again on another port, unreported.
    constexpr int kPortAttempts = 3;
    const auto started          = std::chrono::steady_clock::now();
    int port                    = 0;
    bool ready                  = false;
    InstanceSnapshot snapshot;
    for (int attempt = 1;; ++attempt) {
        try {
            port = settings.child_port != 0 ? settings.child_port : pick_free_port();
        } catch (const std::exception& failure) {
            result.error = failure.what();
            return result;
        }
        spec.port = port;
        spec.args.resize(launch_args);
        spec.args.insert(spec.args.end(), {"--host", "127.0.0.1", "--port", std::to_string(port), "--model-id", alias});
        if (supported("--any-model-name")) { spec.args.push_back("--any-model-name"); }
        if (supported("--log-stats-panel")) { spec.args.insert(spec.args.end(), {"--log-stats-panel", "off"}); }
        if (supported("--log-colours")) { spec.args.insert(spec.args.end(), {"--log-colours", "off"}); }
        port_retry_allowed_ = settings.child_port == 0 && attempt < kPortAttempts;
        try {
            supervisor_->start(spec);
        } catch (const std::exception& failure) {
            port_retry_allowed_ = false;
            result.error        = failure.what();
            return result;
        }
        if (attempt == 1) { store_.set_last_model(key); }
        ready    = supervisor_->wait_ready(started + std::chrono::seconds(settings.load_timeout_s));
        snapshot = supervisor_->snapshot();
        if (ready || snapshot.state != InstanceState::Failed || !port_retry_due(snapshot)) { break; }
        event("info", "model",
              alias + ": port " + std::to_string(port) +
                  " was taken by another program before infernix-serve bound it; starting it on another port",
              {{"model_key", key}, {"port", port}});
    }
    port_retry_allowed_ = false;
    if (!ready) {
        if (snapshot.state == InstanceState::Starting) {
            supervisor_->stop(std::chrono::seconds(settings.stop_timeout_s));
            result.error = alias + " did not become ready within " + std::to_string(settings.load_timeout_s) + " s";
        } else {
            result.error = snapshot.error.value_or(alias + " stopped while loading");
        }
        return result;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    char text[64];
    std::snprintf(text, sizeof(text), " is ready (%.0f s)", seconds);
    toast("success", alias + text);
    result.port     = port;
    result.instance = snapshot.instance;
    return result;
}

Json Launcher::instance_json() const {
    const InstanceSnapshot s = supervisor_->snapshot();
    Json memory              = nullptr;
    if (s.usage) { memory = {{"working_set", s.usage->working_set}, {"private_bytes", s.usage->private_bytes}}; }
    const std::optional<std::string> pending = router_ ? router_->pending_model() : std::nullopt;
    const auto opt_i64 = [](const std::optional<std::int64_t>& v) { return v ? Json(*v) : Json(nullptr); };
    Json incident = nullptr;
    {
        std::lock_guard lock(incident_mutex_);
        incident = last_incident_;
    }
    // When an idle model is unloaded: its configuration's limit (else the global one) after the
    // last request, while nothing runs.
    const Settings settings         = store_.settings();
    const std::int64_t last_request = router_ ? router_->last_activity_ms() : 0;
    Json idle_unload_at             = nullptr;
    if (s.state == InstanceState::Ready && router_ && router_->in_flight() == 0 && last_request > 0) {
        const std::optional<StoredModelConfig> stored = store_.model_config(s.model_key);
        const int minutes = stored && stored->idle_unload_min ? *stored->idle_unload_min : settings.idle_unload_min;
        if (minutes > 0) { idle_unload_at = last_request + static_cast<std::int64_t>(minutes) * 60000; }
    }
    return {{"state", state_name(s.state)},
            {"instance", s.instance},
            {"model_key", s.model_key.empty() ? Json(nullptr) : Json(s.model_key)},
            {"alias", s.alias.empty() ? Json(nullptr) : Json(s.alias)},
            {"name", s.name.empty() ? Json(nullptr) : Json(s.name)},
            {"pid", s.pid ? Json(*s.pid) : Json(nullptr)},
            {"port", s.port == 0 ? Json(nullptr) : Json(s.port)},
            {"started_at", opt_i64(s.started_at)},
            {"ready_at", opt_i64(s.ready_at)},
            {"phase", s.phase},
            {"argv", s.argv},
            {"exit_code", s.exit_code ? Json(*s.exit_code) : Json(nullptr)},
            {"error", s.error ? Json(*s.error) : Json(nullptr)},
            {"in_flight", router_ ? router_->in_flight() : 0},
            {"waiting", router_ ? router_->waiting() : 0},
            {"pending_model", pending ? Json(*pending) : Json(nullptr)},
            {"artifact_key", s.artifact_key.empty() ? Json(nullptr) : Json(s.artifact_key)},
            {"config_name", s.config_name.empty() ? Json("Default") : Json(s.config_name)},
            {"last_request_at", last_request > 0 ? Json(last_request) : Json(nullptr)},
            {"idle_unload_at", idle_unload_at},
            {"pinned", s.state == InstanceState::Ready && settings.pin_loaded},
            {"memory", memory},
            {"incident", incident}};
}

void Launcher::publish_instance() { publish("instance", instance_json()); }

// ---------------------------------------------------------------------------------------------
// Event log and the watchdog's incidents

void Launcher::event(const std::string& level, const std::string& category, const std::string& message, Json fields) {
    event_log_.add(level, category, message, std::move(fields));
}

std::filesystem::path Launcher::log_dir() const { return resolve_log_dir(store_.settings(), options_.data_dir); }

Json Launcher::log_paths_json() const {
    const std::filesystem::path dir = event_log_.dir();
    return {{"dir", path_utf8(dir)},
            {"events_file", path_utf8(event_log_.events_file())},
            {"requests_file", path_utf8(event_log_.requests_file())},
            {"serve_dir", path_utf8(dir / kServeLogDir)}};
}

void Launcher::apply_log_dir() {
    const std::filesystem::path dir = log_dir();
    if (dir == applied_log_dir_) { return; }
    applied_log_dir_ = dir;
    std::string error;
    if (event_log_.set_dir(dir, error)) {
        event("info", "settings", "the logs are now written to " + path_utf8(dir));
    } else {
        event("error", "settings", "the log folder cannot be written (" + error + "); events stay in memory only");
        toast("error", "The log folder cannot be written: " + error);
    }
    logs_.set_dir(dir / kServeLogDir);
    publish("logs", log_paths_json());
}

void Launcher::on_instance_changed(const InstanceSnapshot& snapshot) {
    {
        // Several notifications per state (phases while it loads): report each state once.
        std::lock_guard lock(incident_mutex_);
        if (snapshot.instance == seen_instance_ && snapshot.state == seen_state_) { return; }
        seen_instance_ = snapshot.instance;
        seen_state_    = snapshot.state;
    }
    const std::string& alias = snapshot.alias;
    // llama.cpp's model_status events (/models/sse); a failed load is `unloaded` with a non-zero
    // exit code, which llama.cpp clients report as a load failure.
    switch (snapshot.state) {
    case InstanceState::Starting: publish_model_event(alias, "model_status", {{"status", "loading"}}); break;
    case InstanceState::Ready: publish_model_event(alias, "model_status", {{"status", "loaded"}}); break;
    case InstanceState::Idle:
        if (snapshot.instance != 0) { publish_model_event(alias, "model_status", {{"status", "unloaded"}}); }
        break;
    case InstanceState::Failed: {
        const int code = snapshot.exit_code && *snapshot.exit_code != 0 ? *snapshot.exit_code : 1;
        publish_model_event(alias, "model_status", {{"status", "unloaded"}, {"exit_code", code}});
        break;
    }
    case InstanceState::Stopping: break;
    }
    switch (snapshot.state) {
    case InstanceState::Starting:
        event("info", "model", "loading " + alias + " (port " + std::to_string(snapshot.port) + ")",
              {{"model_key", snapshot.model_key}, {"argv", snapshot.argv}});
        break;
    case InstanceState::Ready: {
        const std::int64_t took = snapshot.ready_at && snapshot.started_at ? *snapshot.ready_at - *snapshot.started_at : 0;
        event("info", "model", alias + " is ready (" + format_duration(took) + ")", {{"model_key", snapshot.model_key}});
        // A load that did not fit is answered by one that does.
        std::lock_guard lock(incident_mutex_);
        if (last_incident_.is_object() && last_incident_.value("phase", "") == "load") { last_incident_ = nullptr; }
        break;
    }
    case InstanceState::Stopping: event("info", "model", "stopping " + alias); break;
    case InstanceState::Idle:
        if (snapshot.instance != 0) { event("info", "model", alias + " stopped"); }
        break;
    case InstanceState::Failed: {
        const std::string error = snapshot.error.value_or("it stopped while loading");
        if (snapshot.incident == "ram") {
            handle_load_failure(snapshot, "ram", error); // the launcher's RAM floor ended the load
        } else if (snapshot.incident) {
            handle_incident(snapshot);
        } else {
            // The server's own words: its last warnings and errors while it loaded.
            const std::string detail = failure_detail(snapshot);
            if (const std::optional<std::string> kind = classify_load_failure(detail)) {
                handle_load_failure(snapshot, *kind, detail);
            } else {
                event("error", "model", alias + " failed to load: " + error, {{"model_key", snapshot.model_key}});
                toast("error", alias + " failed to load: " + error);
            }
        }
        break;
    }
    }
}

std::string Launcher::failure_detail(const InstanceSnapshot& snapshot) const {
    std::string detail = snapshot.error.value_or("it stopped while loading");
    for (const LogRecord& record : logs_.recent(snapshot.instance, 8, true)) { detail += "\n" + record.text; }
    return detail;
}

bool Launcher::port_retry_due(const InstanceSnapshot& snapshot) const {
    return port_retry_allowed_ && !snapshot.incident && classify_load_failure(failure_detail(snapshot)) == "port";
}

void Launcher::handle_load_failure(const InstanceSnapshot& snapshot, const std::string& kind, const std::string& detail) {
    std::optional<ScannedModel> model;
    Args args;
    EnvVars env;
    {
        std::lock_guard lock(library_mutex_);
        if (const std::optional<ConfigRef> ref = find_config_locked(snapshot.model_key)) {
            model = *ref->model;
            args  = effective_args(ref->id, *ref->model, library_);
            env   = effective_env(ref->id);
        }
    }
    ResourceFacts facts;
    if (const std::optional<SystemReading> reading = telemetry_->latest()) {
        facts.ram_total     = reading->ram_total;
        facts.ram_available = reading->ram_available;
        if (!reading->gpus.empty()) {
            facts.vram_total = reading->gpus.front().vram_total;
            facts.vram_used  = reading->gpus.front().vram_used;
        }
    }
    Json suggestions = Json::array();
    if (model && kind == "vram") {
        // The server's own plan (about a second; the failed server's memory is free again) names the
        // largest context that fits exactly, where the refusal's figures give an estimate.
        const Json plan = fit_plan_json(snapshot.model_key, *model, args, env);
        if (plan.value("fits", Json(nullptr)) == false) { suggestions = plan.value("suggestions", Json::array()); }
    }
    if (kind == "port") {
        for (const LoadSuggestion& suggestion : suggest_port_fixes(snapshot.port, store_.settings().child_port != 0)) {
            suggestions.push_back(to_json(suggestion));
        }
    } else if (model && suggestions.empty()) {
        for (const LoadSuggestion& suggestion : suggest_load_fixes(kind, *model, args, detail, facts)) {
            suggestions.push_back(to_json(suggestion));
        }
    }
    const std::string message = snapshot.error.value_or("infernix-serve stopped while loading");
    const std::string action  = snapshot.incident ? "the load was stopped" : "the model was not loaded";
    const std::string what    = kind == "vram"   ? "did not fit in GPU memory"
                                : kind == "port" ? "could not use its port"
                                                 : "ran out of system RAM";
    {
        std::lock_guard lock(incident_mutex_);
        last_incident_ = {{"t", now_ms()},          {"kind", kind},       {"phase", "load"},
                          {"model_key", snapshot.model_key}, {"alias", snapshot.alias}, {"message", message},
                          {"action", action},       {"restarted", false}, {"suggestions", suggestions}};
    }
    std::string titles;
    for (const Json& suggestion : suggestions) {
        titles += (titles.empty() ? "" : "; ") + suggestion["title"].get<std::string>();
    }
    event("error", "model", snapshot.alias + " " + what + ": " + message + (titles.empty() ? "" : " (try: " + titles + ")"),
          {{"kind", kind}, {"model_key", snapshot.model_key}, {"suggestions", suggestions}});
    toast("error", snapshot.alias + " " + what + (suggestions.empty() ? "" : "; the dashboard suggests changes"));
    publish_instance();
}

void Launcher::guard_load_memory(const SystemReading& reading) {
    const Settings settings = store_.settings();
    const InstanceSnapshot snapshot = supervisor_->snapshot();
    const std::uint64_t floor = static_cast<std::uint64_t>(settings.load_ram_floor_mib) << 20;
    if (!settings.watchdog || floor == 0 || snapshot.state != InstanceState::Starting || reading.ram_total == 0 ||
        reading.ram_available >= floor) {
        low_ram_samples_ = 0;
        return;
    }
    // Three readings in a row (a second apart), so a brief dip does not end a load.
    if (++low_ram_samples_ < 3) { return; }
    low_ram_samples_ = 0;
    char text[256];
    std::snprintf(text, sizeof(text),
                  "free RAM fell to %.1f GiB of %.1f GiB while it loaded (the floor is %.1f GiB; below it the computer "
                  "pages heavily)",
                  static_cast<double>(reading.ram_available) / (1 << 30), static_cast<double>(reading.ram_total) / (1 << 30),
                  static_cast<double>(floor) / (1 << 30));
    supervisor_->end_load(snapshot.instance, "ram", text);
}

void Launcher::handle_incident(const InstanceSnapshot& snapshot) {
    const std::string kind    = *snapshot.incident;
    const std::string message = snapshot.error.value_or("infernix-serve stopped");
    const Settings settings   = store_.settings();
    const std::optional<std::string> next = router_ ? router_->pending_model() : std::nullopt;
    std::string action;
    bool restart = false;
    if (!settings.watchdog || !settings.restart_on_failure) {
        action = "not restarted (automatic restarts are off); the next request loads it again";
    } else if (next && *next != snapshot.model_key) {
        action = "not restarted: another model is loading next";
    } else {
        std::lock_guard lock(incident_mutex_);
        const auto now = std::chrono::steady_clock::now();
        while (!restarts_.empty() && now - restarts_.front() > kRestartWindow) { restarts_.pop_front(); }
        if (restarts_.size() >= kRestartLimit) {
            action = "not restarted: it failed " + std::to_string(kRestartLimit) + " times in " +
                     std::to_string(kRestartWindow.count()) + " minutes; the next request loads it again";
        } else {
            restarts_.push_back(now);
            action = "restarting it (" + std::to_string(restarts_.size()) + " of " + std::to_string(kRestartLimit) +
                     " in " + std::to_string(kRestartWindow.count()) + " minutes)";
            restart = true;
        }
    }
    {
        std::lock_guard lock(incident_mutex_);
        last_incident_ = {{"t", now_ms()},
                          {"kind", kind},
                          {"phase", "running"},
                          {"model_key", snapshot.model_key},
                          {"alias", snapshot.alias},
                          {"message", message},
                          {"action", action},
                          {"restarted", restart},
                          {"suggestions", Json::array()}};
    }
    event("error", "watchdog", snapshot.alias + ": " + message + "; " + action,
          {{"kind", kind},
           {"model_key", snapshot.model_key},
           {"exit_code", snapshot.exit_code ? Json(*snapshot.exit_code) : Json(nullptr)}});
    const char* title = kind == "crash" ? "infernix-serve crashed" : kind == "hang" ? "infernix-serve stopped responding"
                                                                                      : "Requests stopped progressing";
    toast("error", std::string(title) + " (" + snapshot.alias + "); " + action);
    if (restart && router_) { router_->request_load(snapshot.model_key, false); }
}

void Launcher::log_api_request(const ApiCall& call, int status, std::optional<std::int64_t> first_byte_ms,
                               std::uint64_t bytes_out) {
    const std::int64_t duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - call.arrived).count();
    std::string served;
    if (call.key) {
        std::lock_guard lock(library_mutex_);
        if (const std::optional<ConfigRef> ref = find_config_locked(*call.key)) {
            served = effective_alias(ref->id, *ref->model, library_);
        }
    }
    std::string message = call.method + " " + call.path + " " + std::to_string(status) + " in " + format_duration(duration_ms);
    if (call.stream && first_byte_ms) { message += " (first event " + format_duration(*first_byte_ms) + ")"; }
    if (!served.empty()) { message += " · " + served; }
    if (call.model && *call.model != served) { message += " (asked for " + *call.model + ")"; }
    if (call.stream) { message += " · stream"; }
    if (call.routed_by && *call.routed_by != "name") { message += " · " + *call.routed_by; }
    message += " · " + (call.key_name ? *call.key_name + " (" + call.client + ")" : call.client);
    if (call.error) { message += " · " + *call.error; }
    const std::string level = call.quiet && status < 400 ? "debug"
                              : status >= 500 || status == 0 ? "error"
                              : status >= 400                ? "warn"
                                                             : "info";
    event(level, "api", message,
          {{"id", call.id},
           {"client", call.client},
           {"user_agent", call.user_agent.empty() ? Json(nullptr) : Json(call.user_agent)},
           {"method", call.method},
           {"path", call.path},
           {"model", call.model ? Json(*call.model) : Json(nullptr)},
           {"model_key", call.key ? Json(*call.key) : Json(nullptr)},
           {"stream", call.stream},
           {"status", status},
           {"duration_ms", duration_ms},
           {"first_byte_ms", first_byte_ms ? Json(*first_byte_ms) : Json(nullptr)},
           {"bytes_in", call.bytes_in},
           {"bytes_out", bytes_out},
           {"error", call.error ? Json(*call.error) : Json(nullptr)},
           {"key_name", call.key_name ? Json(*call.key_name) : Json(nullptr)},
           {"routed_by", call.routed_by ? Json(*call.routed_by) : Json(nullptr)}});
}

void Launcher::log_request_body(const ApiCall& call, const std::string& request_body, int status,
                                const std::string& response_body, bool truncated) {
    // JSON bodies are kept as JSON; anything else (and a stream's events) as text.
    const auto value = [](const std::string& text) {
        try {
            return Json::parse(text);
        } catch (const Json::exception&) { return Json(text); }
    };
    const bool request_cut = request_body.size() > kBodyCaptureLimit;
    event_log_.add_request_body(
        {{"t", now_ms()},
         {"id", call.id},
         {"client", call.client},
         {"method", call.method},
         {"path", call.path},
         {"model", call.model ? Json(*call.model) : Json(nullptr)},
         {"model_key", call.key ? Json(*call.key) : Json(nullptr)},
         {"status", status},
         {"duration_ms",
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - call.arrived).count()},
         {"stream", call.stream},
         {"request", request_cut ? Json(request_body.substr(0, kBodyCaptureLimit)) : value(request_body)},
         {"request_truncated", request_cut},
         {"response", call.stream || truncated ? Json(response_body) : value(response_body)},
         {"response_truncated", truncated}});
}

// ---------------------------------------------------------------------------------------------
// Proxy activity

std::uint64_t Launcher::activity_begin(const Request& request, const ApiCall& call) {
    std::uint64_t id = 0;
    {
        std::lock_guard lock(activity_mutex_);
        ProxyRecord record;
        record.id        = next_activity_++;
        record.t         = now_ms();
        record.method    = request.method;
        record.path      = request.path;
        record.model     = call.model;
        record.model_key = call.key;
        record.stream    = call.stream;
        record.client    = request.remote_addr;
        record.key_name  = call.key_name;
        record.routed_by = call.routed_by;
        id               = record.id;
        activity_.push_back(std::move(record));
        while (activity_.size() > kActivityLimit) { activity_.pop_front(); }
    }
    publish("activity", activity_json());
    return id;
}

void Launcher::activity_update(std::uint64_t id, const std::string& state, int status,
                               const std::optional<std::string>& error) {
    {
        std::lock_guard lock(activity_mutex_);
        for (ProxyRecord& record : activity_) {
            if (record.id != id) { continue; }
            record.state  = state;
            record.status = status;
            if (error) { record.error = error; }
            if (state == "done" || state == "failed") {
                record.duration_ms = now_ms() - record.t;
                record.progress    = nullptr;
            }
        }
    }
    publish("activity", activity_json());
}

void Launcher::absorb_metrics(const Json& metrics) {
    if (!metrics.is_object()) { return; }
    const auto activity_of = [](const Json& label) -> std::optional<std::uint64_t> {
        if (!label.is_string() || !label.get<std::string>().starts_with("launcher-")) { return std::nullopt; }
        try {
            return std::stoull(label.get<std::string>().substr(9));
        } catch (const std::exception&) { return std::nullopt; }
    };
    bool changed = false;
    // Running requests' progress.
    const Json live = metrics.value("live", Json::object());
    {
        std::lock_guard lock(activity_mutex_);
        std::set<std::uint64_t> active;
        for (const Json& item : live.value("active", Json::array())) {
            const std::optional<std::uint64_t> id = activity_of(item.value("label", Json(nullptr)));
            if (!id) { continue; }
            active.insert(*id);
            for (ProxyRecord& record : activity_) {
                if (record.id != *id || (record.state != "running" && record.state != "waiting")) { continue; }
                const Json progress = {{"phase", item.value("phase", "queued")},
                                       {"prompt_tokens", item.value("prompt_tokens", 0)},
                                       {"cached_tokens", item.value("cached_tokens", 0)},
                                       {"processed_tokens", item.value("processed_tokens", 0)},
                                       {"output_tokens", item.value("output_tokens", 0)},
                                       {"elapsed_ms", item.value("elapsed_ms", 0)}};
                if (record.progress != progress) {
                    record.progress = progress;
                    changed         = true;
                }
            }
        }
        for (ProxyRecord& record : activity_) {
            if (!record.progress.is_null() && !active.contains(record.id)) {
                record.progress = nullptr;
                changed         = true;
            }
        }
    }
    // Finished requests' usage, once each (the server keeps its last rows; each poll repeats them).
    const InstanceSnapshot snapshot = supervisor_->snapshot();
    // Row ids restart with each server; an older server without instance_id is told apart by the
    // launcher's own instance number.
    std::string instance = live.value("instance_id", "");
    if (instance.empty()) { instance = "launcher-" + std::to_string(snapshot.instance); }
    const std::string serving = snapshot.model_key;
    std::vector<UsageSample> samples;
    {
        std::lock_guard lock(activity_mutex_);
        for (const Json& row : metrics.value("requests", Json::array())) {
            const std::string seen = instance + ":" + std::to_string(row.value("id", std::uint64_t{0}));
            if (!counted_rows_.insert(seen).second) { continue; }
            counted_order_.push_back(seen);
            UsageSample sample;
            sample.t = static_cast<std::int64_t>(row.value("time", 0.0) * 1000.0);
            if (sample.t == 0) { sample.t = now_ms(); }
            const std::string finish = row.value("finish", "");
            sample.failed            = finish == "failed" || finish == "cancelled";
            sample.prompt_tokens     = row.value("prompt_tokens", std::uint64_t{0});
            sample.cached_tokens     = row.value("reused", std::uint64_t{0});
            sample.prefill_tokens    = sample.prompt_tokens - std::min(sample.cached_tokens, sample.prompt_tokens);
            sample.output_tokens     = row.value("output_tokens", std::uint64_t{0});
            sample.prefill_s         = row.value("prompt_ms", 0.0) / 1000.0;
            sample.decode_s          = row.value("decode_ms", 0.0) / 1000.0;
            sample.config_id         = serving;
            sample.client            = "unknown";
            const Json label         = row.value("label", Json(nullptr));
            if (const std::optional<std::uint64_t> id = activity_of(label)) {
                for (ProxyRecord& record : activity_) {
                    if (record.id != *id) { continue; }
                    if (record.model_key) { sample.config_id = *record.model_key; }
                    sample.client = record.key_name ? *record.key_name : record.client;
                }
            } else if (label.is_string()) {
                sample.client = "label " + label.get<std::string>();
            }
            samples.push_back(std::move(sample));
        }
        // Bounded: the server repeats at most its last 128 rows, so the oldest marks can go.
        while (counted_order_.size() > 4096) {
            counted_rows_.erase(counted_order_.front());
            counted_order_.pop_front();
        }
    }
    if (!samples.empty()) {
        std::lock_guard lock(library_mutex_);
        for (UsageSample& sample : samples) {
            if (const std::optional<ConfigRef> ref = find_config_locked(sample.config_id)) {
                sample.model = effective_alias(ref->id, *ref->model, library_);
            } else {
                sample.model = sample.config_id;
            }
        }
    }
    for (const UsageSample& sample : samples) { usage_.add(sample); }
    if (changed) { publish("activity", activity_json()); }
}

Json Launcher::activity_json() const {
    Json recent = Json::array();
    {
        // The newest 50, oldest first.
        std::lock_guard lock(activity_mutex_);
        const std::size_t skip = activity_.size() > 50 ? activity_.size() - 50 : 0;
        for (auto it = activity_.begin() + static_cast<std::ptrdiff_t>(skip); it != activity_.end(); ++it) {
            const ProxyRecord& r = *it;
            recent.push_back({{"id", r.id},
                              {"t", r.t},
                              {"method", r.method},
                              {"path", r.path},
                              {"model", r.model ? Json(*r.model) : Json(nullptr)},
                              {"model_key", r.model_key ? Json(*r.model_key) : Json(nullptr)},
                              {"stream", r.stream},
                              {"status", r.status},
                              {"state", r.state},
                              {"duration_ms", r.duration_ms ? Json(*r.duration_ms) : Json(nullptr)},
                              {"error", r.error ? Json(*r.error) : Json(nullptr)},
                              {"routed_by", r.routed_by ? Json(*r.routed_by) : Json(nullptr)},
                              {"client", r.client},
                              {"key_name", r.key_name ? Json(*r.key_name) : Json(nullptr)},
                              {"progress", r.progress}});
        }
    }
    return {{"in_flight", router_ ? router_->in_flight() : 0}, {"waiting", router_ ? router_->waiting() : 0},
            {"recent", recent}};
}

// ---------------------------------------------------------------------------------------------
// Downloads

Json Launcher::hf_files_json(const std::string& repo) const {
    const HfListing listing = hf_list_files(repo, store_.hf_token());
    std::string kind        = "model";
    for (const Json& entry : catalog_.value("entries", Json::array())) {
        if (iequals(json_or<std::string>(entry, "repo", ""), repo)) { kind = json_or<std::string>(entry, "kind", "model"); }
    }
    const Settings settings = store_.settings();
    const auto present = [&](const HfFile& file) -> Json {
        const std::filesystem::path name = utf8_path(file.path).filename();
        for (const std::string& dir : settings.model_dirs) {
            const std::filesystem::path root = utf8_path(dir);
            std::vector<std::filesystem::path> candidates = {root / name};
            std::error_code error;
            for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
                std::error_code status;
                if (entry.is_directory(status)) { candidates.push_back(entry.path() / name); }
            }
            for (const auto& candidate : candidates) {
                std::error_code status;
                if (std::filesystem::is_regular_file(candidate, status) &&
                    std::filesystem::file_size(candidate, status) == file.size) {
                    return path_utf8(candidate);
                }
            }
        }
        return nullptr;
    };
    bool has_model = false;
    for (const HfFile& file : listing.files) {
        const std::string lowered = lower(file.path);
        if (lowered.ends_with(".infernix") || lowered.ends_with(".ninfer") || lowered.find(".ninfer.part-") != std::string::npos) {
            has_model = true;
        }
    }
    Json files = Json::array();
    for (const HfFile& file : listing.files) {
        const std::string lowered = lower(file.path);
        std::string role          = "other";
        if (lowered.ends_with(".ngram")) {
            role = "ngram";
        } else if (lowered.ends_with(".expert-state")) {
            role = "expert-state";
        } else if (lowered.ends_with(".conversion.json")) {
            role = "report";
        } else if (lowered.find(".ninfer.part-") != std::string::npos) {
            role = "model-part";
        } else if (lowered.ends_with(".infernix") || lowered.ends_with(".ninfer")) {
            const std::size_t of = lowered.rfind("-of-");
            const bool later_part = of != std::string::npos && of >= 6 && lowered.substr(of - 5, 5) != "00001";
            role = kind == "drafter" ? "drafter" : later_part ? "model-part" : "model";
        }
        const bool selected = role == "model" || role == "model-part" || role == "drafter" ||
                              role == "expert-state" || (role == "ngram" && !has_model);
        files.push_back({{"path", file.path},
                         {"bytes", file.size},
                         {"sha256", file.sha256 ? Json(*file.sha256) : Json(nullptr)},
                         {"role", role},
                         {"default_selected", selected},
                         {"present_locally", present(file)}});
    }
    return {{"repo", listing.repo}, {"revision", listing.revision}, {"files", files}};
}

void Launcher::download_finished(const std::filesystem::path& dest_dir, const std::filesystem::path& target_dir) {
    store_.update_settings([&](Settings& settings) {
        for (const std::string& dir : settings.model_dirs) {
            if (same_or_inside(target_dir, utf8_path(dir), settings.scan_depth)) { return; }
        }
        settings.model_dirs.push_back(path_utf8(dest_dir));
    });
    publish("settings", settings_to_json(store_.settings()));
    request_scan();
}

// ---------------------------------------------------------------------------------------------
// HTTP: access control

bool Launcher::page_checks(const Request& request, Response* response, bool has_key) const {
    // Web pages the user opens elsewhere must not drive this API (the public routes send
    // `Access-Control-Allow-Origin: *`). A page's requests carry its Origin, refused unless it is this
    // page's own or a loopback one; a DNS-rebinding page names its own domain in Host, refused unless
    // Host is an IP address or localhost (or the request carries an admin key).
    const std::string host      = request.get_header_value("Host");
    const std::string host_name = authority_host(host);
    if (!has_key && !host_name.empty() && !ip_literal(host_name) && !loopback_host(host_name)) {
        if (response) {
            send_error(*response, 403, "forbidden",
                       "open the launcher by IP address or localhost (not " + host_name + "), or send an admin key");
        }
        return false;
    }
    const std::string origin = request.get_header_value("Origin");
    if (!origin.empty()) {
        const std::size_t scheme    = origin.find("://");
        const std::string authority = scheme == std::string::npos ? std::string() : origin.substr(scheme + 3);
        if (authority.empty() || (!iequals(authority, host) && !loopback_host(authority_host(authority)))) {
            if (response) {
                send_error(*response, 403, "forbidden", "the launcher's management API does not answer other web pages");
            }
            return false;
        }
    }
    return true;
}

std::optional<ApiKey> Launcher::admin_key(const Request& request) {
    // An admin key in a header, or the sign-in cookie of one.
    if (const std::optional<ApiKey> key = store_.use_api_key(presented_key(request)); key && key->scope == "admin") {
        return key;
    }
    const std::string token = session_cookie(request);
    if (token.empty()) { return std::nullopt; }
    std::string key_id;
    {
        std::lock_guard lock(sessions_mutex_);
        const auto it = sessions_.find(token);
        if (it == sessions_.end()) { return std::nullopt; }
        key_id = it->second;
    }
    for (const ApiKey& key : store_.settings().api_keys) {
        if (key.id == key_id && key.scope == "admin") { return key; }
    }
    return std::nullopt;
}

bool Launcher::admin_allowed(const Request& request, Response& response) {
    const Settings settings           = store_.settings();
    const std::optional<ApiKey> admin = admin_key(request);
    if (!page_checks(request, &response, admin.has_value())) { return false; }
    if (is_loopback(request.remote_addr)) { return true; }
    if (!settings.allow_remote_admin) {
        send_error(response, 403, "forbidden", "the launcher's management API answers this computer only (Settings, Server)");
        return false;
    }
    if (settings.api_keys.empty() || admin) { return true; }
    send_error(response, 401, "sign_in", "sign in with an admin API key to manage the launcher from this computer");
    return false;
}

Json Launcher::session_json(const Request& request) {
    const Settings settings           = store_.settings();
    const std::optional<ApiKey> admin = admin_key(request);
    const bool required = !is_loopback(request.remote_addr) && !settings.api_keys.empty();
    return {{"required", required},
            {"signed_in", admin.has_value()},
            {"key_name", admin ? Json(admin->name) : Json(nullptr)}};
}

bool Launcher::from_own_page(const Request& request) {
    // The launcher's own page: same origin by the browser's word (or its Origin), a Host the
    // management API accepts, and this computer or a signed-in browser.
    const bool same_origin = iequals(request.get_header_value("Sec-Fetch-Site"), "same-origin") ||
                             [&] {
                                 const std::string origin = request.get_header_value("Origin");
                                 const std::size_t scheme = origin.find("://");
                                 return scheme != std::string::npos &&
                                        iequals(origin.substr(scheme + 3), request.get_header_value("Host"));
                             }();
    if (!same_origin || !page_checks(request, nullptr, false)) { return false; }
    return is_loopback(request.remote_addr) || admin_key(request).has_value();
}

bool Launcher::client_authorized(const Request& request, Response& response, bool anthropic, ApiCall& call) {
    if (from_own_page(request)) {
        call.from_ui  = true;
        call.key_name = "launcher chat";
        return true;
    }
    const std::string secret = presented_key(request);
    if (const std::optional<ApiKey> key = store_.use_api_key(secret)) {
        call.key_name = key->name;
        return true;
    }
    if (store_.settings().api_keys.empty()) { return true; }
    if (anthropic) {
        anthropic_error(response, 401, "authentication_error", "invalid x-api-key");
    } else {
        openai_error(response, 401, "invalid_request_error", "invalid_api_key", "Incorrect API key provided");
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// HTTP: public OpenAI / Anthropic routes

// One entry of /v1/models, readable by OpenAI clients, Anthropic clients (type, display_name,
// created_at, max_input_tokens) and the tools that read other servers' fields: llama.cpp and
// llama-swap (status.value, meta.n_ctx, architecture's modalities), vLLM (max_model_len), LM Studio
// (state, max_context_length), OpenRouter-style catalogues (name, description, context_length,
// supported_parameters).
Json Launcher::model_entry(const std::string& config_id, const ScannedModel& model, const LibraryScan& scan,
                           bool claude_code) const {
    const Args args                 = effective_args(config_id, model, scan);
    const std::uint64_t context     = context_length(args);
    const std::uint64_t max_output  = default_output_tokens(args, context);
    const InstanceSnapshot instance = supervisor_->snapshot();
    std::string state               = "not-loaded";
    std::string status              = "unloaded";
    std::string state_words         = "loads on first request";
    if (instance.model_key == config_id && instance.state == InstanceState::Ready) {
        state = status = state_words = "loaded";
    } else if (instance.model_key == config_id && instance.state == InstanceState::Starting) {
        state = status = state_words = "loading";
    }
    const std::string alias    = effective_alias(config_id, model, scan);
    const std::int64_t created = model.modified / 1000;
    const bool vision          = has_flag(args, "--vision");
    const std::string id = claude_code && !icontains(alias, "claude") && !icontains(alias, "anthropic")
                               ? alias + std::string(kClaudeCodeSuffix)
                               : alias;
    std::string description = model.family_label;
    if (model.quantization) { description += ", " + *model.quantization; }
    description += ", " + std::to_string((context + 500) / 1000) + "K context";
    if (vision) { description += ", vision"; }
    description += ", " + state_words;
    const Json input = vision ? Json::array({"text", "image"}) : Json::array({"text"});
    return {{"id", id},
            {"object", "model"},
            {"type", "model"},
            {"created", created},
            {"created_at", rfc3339(model.modified)},
            {"owned_by", "infernix"},
            {"name", alias},
            {"display_name", alias},
            {"description", description},
            {"context_length", context},
            {"context_window", context},
            {"max_model_len", context},
            {"max_context_length", context},
            {"max_input_tokens", context},
            {"max_tokens", max_output},
            {"state", state},
            {"status", {{"value", status}}},
            {"architecture",
             {{"modality", vision ? "text+image->text" : "text->text"},
              {"input_modalities", input},
              {"output_modalities", Json::array({"text"})}}},
            {"family", model.family_label},
            {"quantization", model.quantization ? Json(*model.quantization) : Json(nullptr)},
            {"capabilities", {{"vision", vision}, {"tools", true}, {"reasoning", true}}},
            // The request fields infernix-serve honours (docs/serving.md), in OpenRouter's names.
            {"supported_parameters",
             {"max_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "seed",
              "stop", "tools", "tool_choice", "parallel_tool_calls", "reasoning", "include_reasoning",
              "response_format", "structured_outputs"}},
            {"meta",
             {{"architecture", model.architecture},
              {"n_ctx", context},
              {"n_ctx_train", model.native_context},
              {"n_vocab", model.vocab_size},
              {"n_embd", model.hidden_size},
              {"size", model.bytes}}}};
}

void Launcher::handle_models(const Request& request, Response& response, ApiCall& call) {
    call.quiet           = true;
    const bool anthropic = request.has_header("anthropic-version");
    if (!client_authorized(request, response, anthropic, call)) { return; }
    const bool claude_code = is_claude_code_client(request);
    std::lock_guard lock(library_mutex_);
    if (request.matches.size() > 1 && request.matches[1].matched) {
        const std::string requested = request.matches[1].str();
        const std::optional<std::string> base = strip_claude_code(requested);
        for (const std::string& id : base ? std::vector{requested, *base} : std::vector{requested}) {
            for (const ScannedModel& model : library_.models) {
                for (const std::string& config_id : config_ids(model)) {
                    if (iequals(effective_alias(config_id, model, library_), id) || iequals(config_id, id)) {
                        // Asked by Claude Code's spelling, the entry carries it.
                        Json entry = model_entry(config_id, model, library_);
                        if (base) { entry["id"] = requested; }
                        send_json(response, entry);
                        return;
                    }
                }
            }
        }
        if (anthropic) {
            anthropic_error(response, 404, "not_found_error", "model: " + requested);
        } else {
            openai_error(response, 404, "invalid_request_error", "model_not_found",
                         "The model '" + requested + "' does not exist");
        }
        return;
    }
    Json data = Json::array();
    for (const ScannedModel& model : library_.models) {
        // Incomplete and unsupported artifacts (no Infernix model family) are not offered, nor
        // hidden configurations; each other configuration is a model of its own.
        if (!model.missing_files.empty() || model.family == "unknown") { continue; }
        for (const std::string& config_id : config_ids(model)) {
            const std::optional<StoredModelConfig> stored = store_.model_config(config_id);
            if (stored && !stored->listed) { continue; }
            data.push_back(model_entry(config_id, model, library_, claude_code));
        }
    }
    // Anthropic's list is paged (limit 1-1000, default 20; after_id or before_id); OpenAI's is whole.
    std::size_t begin = 0, end = data.size();
    if (anthropic) {
        std::size_t limit = 20;
        if (request.has_param("limit")) {
            try {
                limit = std::stoul(request.get_param_value("limit"));
            } catch (const std::exception&) {}
            limit = std::clamp<std::size_t>(limit, 1, 1000);
        }
        const auto index_of = [&data](const std::string& id) -> std::optional<std::size_t> {
            for (std::size_t i = 0; i < data.size(); ++i) {
                if (data[i]["id"] == id) { return i; }
            }
            return std::nullopt;
        };
        if (request.has_param("before_id")) {
            end   = index_of(request.get_param_value("before_id")).value_or(0);
            begin = end > limit ? end - limit : 0;
        } else {
            if (request.has_param("after_id")) {
                const std::optional<std::size_t> after = index_of(request.get_param_value("after_id"));
                begin = after ? *after + 1 : data.size();
            }
            end = std::min(data.size(), begin + limit);
        }
    }
    Json page = Json::array();
    for (std::size_t i = begin; i < end; ++i) { page.push_back(std::move(data[i])); }
    const bool more = request.has_param("before_id") ? begin > 0 : end < data.size();
    Json body = {{"object", "list"}, {"data", page}, {"has_more", anthropic && more}};
    body["first_id"] = page.empty() ? Json(nullptr) : page.front()["id"];
    body["last_id"]  = page.empty() ? Json(nullptr) : page.back()["id"];
    send_json(response, body);
}

// ---------------------------------------------------------------------------------------------
// HTTP: llama-server's router API (Settings.llamacpp_api)

Json Launcher::props_json(const std::string& config_id, const ScannedModel& model, const LibraryScan& scan) const {
    const Args args                 = effective_args(config_id, model, scan);
    const std::uint64_t context     = context_length(args);
    const InstanceSnapshot instance = supervisor_->snapshot();
    const bool loaded               = instance.model_key == config_id && instance.state == InstanceState::Ready;
    int slots                       = 1; // infernix-serve's --max-concurrency default
    if (const std::optional<std::string> value = flag_value(args, "--max-concurrency")) {
        try {
            slots = std::stoi(*value);
        } catch (const std::exception&) {}
    }
    // Without --spec, a server that lists off among its --spec values runs --spec auto, which
    // speculates with the artifact's bundled drafter or --draft-model's; an older server decodes plainly.
    const std::optional<std::string> spec = flag_value(args, "--spec");
    const bool speculative =
        spec ? *spec != "off" && *spec != "none"
             : serve_options().lists_value("--spec", "off") && !has_flag(args, "--lossless") &&
                   (model.has_mtp || model.has_dflash || model.has_dflash2 || has_flag(args, "--draft-model"));
    return {{"default_generation_settings",
             {{"id", 0},
              {"id_task", -1},
              // --max-context bounds each request, as llama.cpp's per-slot n_ctx does.
              {"n_ctx", context},
              {"speculative", speculative},
              {"is_processing", loaded && router_->in_flight() > 0},
              {"params", {{"n_predict", -1}, {"max_tokens", default_output_tokens(args, context)}}}}},
            {"total_slots", slots},
            {"model_path", path_utf8(model.path)},
            {"model_alias", effective_alias(config_id, model, scan)},
            {"chat_template_caps",
             {{"supports_tools", true},
              {"supports_tool_calls", true},
              {"supports_parallel_tool_calls", true},
              {"supports_system_role", true},
              {"supports_preserve_reasoning", has_flag(args, "--preserve-thinking")}}},
            {"modalities", {{"vision", has_flag(args, "--vision")}, {"audio", false}}},
            {"build_info", "infernix-launcher " + options_.version},
            {"is_sleeping", false}};
}

void Launcher::publish_model_event(const std::string& model, const std::string& event, Json data) {
    Json body = {{"model", model}, {"event", event}};
    if (!data.is_null()) { body["data"] = std::move(data); }
    model_events_.publish_text(EventHub::format_data(body));
}

void Launcher::handle_props(const Request& request, Response& response, ApiCall& call) {
    call.quiet = true;
    if (!store_.settings().llamacpp_api) {
        response.status = 404; // as an unknown path
        return;
    }
    if (!client_authorized(request, response, false, call)) { return; }
    // Router mode names the model (?model=); single-model mode describes the loaded one, here the
    // loaded configuration, else the default one. Nothing is loaded for the answer.
    const std::string name = request.get_param_value("model");
    std::optional<std::string> key;
    if (!name.empty()) {
        key = resolve_model(name);
        if (!key) {
            llamacpp_error(response, 400, "invalid_request_error", "model '" + name + "' not found");
            return;
        }
    } else {
        const InstanceSnapshot instance = supervisor_->snapshot();
        if (!instance.model_key.empty() &&
            (instance.state == InstanceState::Ready || instance.state == InstanceState::Starting)) {
            key = instance.model_key;
        } else {
            key = store_.settings().default_model;
        }
    }
    std::lock_guard lock(library_mutex_);
    const std::optional<ConfigRef> ref = key ? find_config_locked(*key) : std::nullopt;
    if (!ref) {
        // Nothing to describe: llama-server's router answer.
        send_json(response, {{"role", "router"},
                             {"model_alias", "infernix-launcher"},
                             {"model_path", "none"},
                             {"default_generation_settings", {{"params", Json::object()}, {"n_ctx", 0}}},
                             {"build_info", "infernix-launcher " + options_.version}});
        return;
    }
    send_json(response, props_json(ref->id, *ref->model, library_));
}

void Launcher::handle_models_load(const Request& request, Response& response, ApiCall& call) {
    if (!store_.settings().llamacpp_api) {
        call.quiet      = true; // a probe of a path the launcher does not answer
        response.status = 404;
        return;
    }
    if (!client_authorized(request, response, false, call)) { return; }
    bool stream = false, parsed = false;
    const std::string name = json_body_model(request.body, stream, parsed);
    if (!name.empty()) { call.model = name; }
    if (!parsed) {
        llamacpp_error(response, 400, "invalid_request_error", "the request body must be a JSON object");
        return;
    }
    const std::optional<std::string> key = resolve_model(name);
    if (!key) {
        llamacpp_error(response, 404, "not_found_error", "model is not found");
        return;
    }
    call.key                        = key;
    const InstanceSnapshot instance = supervisor_->snapshot();
    if ((instance.model_key == *key &&
         (instance.state == InstanceState::Ready || instance.state == InstanceState::Starting)) ||
        router_->loading_model() == key) {
        llamacpp_error(response, 400, "invalid_request_error", "model is already running");
        return;
    }
    // Queued like a request (the loaded model's running requests finish first); it returns at once.
    router_->request_load(*key, false);
    send_json(response, {{"success", true}});
}

void Launcher::handle_models_unload(const Request& request, Response& response, ApiCall& call) {
    if (!store_.settings().llamacpp_api) {
        call.quiet      = true;
        response.status = 404;
        return;
    }
    if (!client_authorized(request, response, false, call)) { return; }
    bool stream = false, parsed = false;
    const std::string name = json_body_model(request.body, stream, parsed);
    if (!name.empty()) { call.model = name; }
    const std::optional<std::string> key = parsed ? resolve_model(name) : std::nullopt;
    if (!key) {
        llamacpp_error(response, 400, "invalid_request_error", "model is not found");
        return;
    }
    call.key                        = key;
    const InstanceSnapshot instance = supervisor_->snapshot();
    if (instance.model_key != *key ||
        (instance.state != InstanceState::Ready && instance.state != InstanceState::Starting)) {
        llamacpp_error(response, 400, "invalid_request_error", "model is not running");
        return;
    }
    router_->request_unload();
    send_json(response, {{"success", true}});
}

void Launcher::handle_models_sse(const Request& request, Response& response, ApiCall& call) {
    call.quiet = true;
    if (!store_.settings().llamacpp_api) {
        response.status = 404;
        return;
    }
    if (!client_authorized(request, response, false, call)) { return; }
    auto subscription = model_events_.subscribe();
    response.status   = 200;
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
    response.set_chunked_content_provider(
        "text/event-stream",
        [subscription](size_t, httplib::DataSink& sink) {
            const std::optional<std::string> event = subscription->next(std::chrono::seconds(15));
            if (!event) {
                if (subscription->closed()) {
                    sink.done();
                    return true;
                }
                static constexpr std::string_view kPing = ": ping\n\n"; // llama.cpp clients skip comments
                return sink.write(kPing.data(), kPing.size());
            }
            return sink.write(event->data(), event->size());
        },
        [subscription](bool) { subscription->close(); });
}

void Launcher::handle_generation(const Request& request, Response& response, ApiCall& call) {
    const bool anthropic = request.path.find("/messages") != std::string::npos;
    if (!client_authorized(request, response, anthropic, call)) {
        call.error = "invalid API key";
        return;
    }
    bool stream = false, parsed = false;
    const std::string name = json_body_model(request.body, stream, parsed);
    call.stream            = stream;
    if (!name.empty()) { call.model = name; }
    if (!parsed) {
        call.error = "the body is not a JSON object";
        if (anthropic) {
            anthropic_error(response, 400, "invalid_request_error", "the request body must be a JSON object");
        } else {
            openai_error(response, 400, "invalid_request_error", "invalid_json", "the request body must be a JSON object");
        }
        return;
    }
    std::string routed_by;
    const std::optional<std::string> key = route_request(name, routed_by);
    call.routed_by                       = routed_by.empty() ? std::nullopt : std::optional(routed_by);
    if (!key) {
        const std::string message = name.empty() ? "the request names no model and none is loaded or set as default"
                                                 : "The model '" + name + "' is not in the launcher's library";
        call.error = message;
        if (anthropic) {
            anthropic_error(response, 404, "not_found_error", message);
        } else {
            openai_error(response, 404, "invalid_request_error", "model_not_found", message);
        }
        return;
    }
    call.key                     = key;
    const std::uint64_t activity = activity_begin(request, call);
    call.id                      = activity;
    const auto finished          = [this, activity](const ProxyOutcome& outcome) {
        const bool failed = !outcome.error.empty() || outcome.status >= 400 || outcome.status == 0;
        activity_update(activity, failed ? "failed" : "done", outcome.status,
                        outcome.error.empty() ? std::nullopt : std::optional(outcome.error));
    };
    const auto refuse = [&](const Router::Refusal& refusal) {
        activity_update(activity, "failed", refusal.status, refusal.message);
        call.error = refusal.code + ": " + refusal.message;
        // Whole seconds, as clients (Claude Code among them) expect.
        if (refusal.retry_after_s > 0) { response.set_header("Retry-After", std::to_string(refusal.retry_after_s)); }
        if (anthropic) {
            anthropic_error(response, refusal.status == 499 ? 499 : 529, "overloaded_error", refusal.message);
        } else {
            openai_error(response, refusal.status, "server_error", refusal.code, refusal.message);
        }
    };
    // A stream that would wait for a model load gets its answer started at once, kept alive while
    // the model loads; a model held after a failed load is refused before any stream starts.
    if (stream) {
        if (std::shared_ptr<Router::Lease> ready = router_->try_acquire(*key)) {
            activity_update(activity, "running", 0, std::nullopt);
            proxy(request, response, std::move(ready), call, finished);
            return;
        }
        if (Router::Refusal held; router_->held(*key, held)) {
            refuse(held);
            return;
        }
        std::string alias = *key;
        {
            std::lock_guard lock(library_mutex_);
            if (const std::optional<ConfigRef> ref = find_config_locked(*key)) {
                alias = effective_alias(ref->id, *ref->model, library_);
            }
        }
        const std::string config_id = *key;
        proxy_after_load(
            request, response,
            [this, config_id, activity](const std::function<bool()>& cancelled, Router::Refusal& refusal) {
                std::shared_ptr<Router::Lease> lease = router_->acquire(config_id, cancelled, refusal);
                if (lease) { activity_update(activity, "running", 0, std::nullopt); }
                return lease;
            },
            anthropic ? StreamDialect::Anthropic : StreamDialect::OpenAI, "loading " + alias, call, finished);
        return;
    }
    Router::Refusal refusal;
    std::shared_ptr<Router::Lease> lease =
        router_->acquire(*key, [&request] { return request.is_connection_closed(); }, refusal);
    if (!lease) {
        refuse(refusal);
        return;
    }
    activity_update(activity, "running", 0, std::nullopt);
    proxy(request, response, std::move(lease), call, finished);
}

std::optional<std::string> Launcher::route_request(const std::string& name, std::string& routed_by) const {
    const Settings settings         = store_.settings();
    const InstanceSnapshot instance = supervisor_->snapshot();
    // The loaded (or loading) model, else the default model.
    const auto loaded = [&]() -> std::optional<std::string> {
        if (!instance.model_key.empty() &&
            (instance.state == InstanceState::Ready || instance.state == InstanceState::Starting)) {
            return instance.model_key;
        }
        if (std::optional<std::string> loading = router_->loading_model()) { return loading; }
        return std::nullopt;
    };
    const auto default_model = [&]() -> std::optional<std::string> {
        if (!settings.default_model) { return std::nullopt; }
        std::lock_guard lock(library_mutex_);
        if (find_config_locked(*settings.default_model)) { return settings.default_model; }
        return std::nullopt;
    };
    if (settings.pin_loaded) {
        if (std::optional<std::string> key = loaded()) {
            routed_by = "pinned";
            return key;
        }
    }
    if (const ModelRoute* route = match_route(settings.model_routes, name)) {
        std::optional<std::string> key;
        if (route->target == "loaded") {
            key = loaded();
            if (!key) { key = default_model(); }
        } else {
            std::lock_guard lock(library_mutex_);
            if (find_config_locked(route->target)) { key = route->target; }
        }
        if (key) {
            routed_by = "route";
            return key;
        }
    }
    if (std::optional<std::string> key = resolve_model(name)) {
        routed_by = "name";
        return key;
    }
    if (settings.unknown_model == "loaded") {
        // Like infernix-serve --any-model-name: the loaded (or loading) model, else the default.
        std::optional<std::string> key = loaded();
        if (!key) { key = default_model(); }
        if (key) {
            routed_by = "loaded";
            return key;
        }
    }
    return std::nullopt;
}

void Launcher::handle_stateful(const Request& request, Response& response, ApiCall& call) {
    if (!client_authorized(request, response, false, call)) {
        call.error = "invalid API key";
        return;
    }
    // Stored Responses live in the loaded server; nothing is loaded for them.
    const InstanceSnapshot instance = supervisor_->snapshot();
    if (instance.model_key.empty() || instance.state != InstanceState::Ready) {
        call.error = "no model is loaded";
        openai_error(response, 404, "invalid_request_error", "not_found",
                     "no model is loaded; stored responses live in the loaded model's server");
        return;
    }
    call.key = instance.model_key;
    Router::Refusal refusal;
    std::shared_ptr<Router::Lease> lease =
        router_->acquire(instance.model_key, [&request] { return request.is_connection_closed(); }, refusal);
    if (!lease) {
        call.error = refusal.code + ": " + refusal.message;
        if (refusal.retry_after_s > 0) { response.set_header("Retry-After", std::to_string(refusal.retry_after_s)); }
        openai_error(response, refusal.status, "server_error", refusal.code, refusal.message);
        return;
    }
    proxy(request, response, std::move(lease), call, nullptr);
}

std::function<void(const ProxyOutcome&)> Launcher::exchange_finished(const Request& request, ApiCall& call,
                                                                     std::function<void(const ProxyOutcome&)> finished) {
    // The exchange ends after this handler returns (streams); its log line is written then.
    call.proxied      = true;
    const bool bodies = store_.settings().log_bodies;
    std::string request_body = bodies ? request.body : std::string();
    return [this, call, bodies, request_body = std::move(request_body), finished = std::move(finished)](
               const ProxyOutcome& outcome) {
        if (finished) { finished(outcome); }
        ApiCall ended = call;
        if (!outcome.error.empty()) { ended.error = outcome.error; }
        log_api_request(ended, outcome.status, outcome.first_byte_ms, outcome.bytes_out);
        if (bodies) { log_request_body(ended, request_body, outcome.status, outcome.body, outcome.body_truncated); }
    };
}

void Launcher::proxy(const Request& request, Response& response, std::shared_ptr<Router::Lease> lease, ApiCall& call,
                     std::function<void(const ProxyOutcome&)> finished) {
    const int port = lease->port();
    if (call.id != 0) { response.set_header("X-Infernix-Request-Id", std::to_string(call.id)); }
    const std::size_t capture = store_.settings().log_bodies ? kBodyCaptureLimit : 0;
    forward_request(request, response, port, std::move(lease), exchange_finished(request, call, std::move(finished)),
                    call.arrived, capture, call.id != 0 ? "launcher-" + std::to_string(call.id) : std::string());
}

void Launcher::proxy_after_load(const Request& request, Response& response, AcquireLease acquire, StreamDialect dialect,
                                const std::string& loading_note, ApiCall& call,
                                std::function<void(const ProxyOutcome&)> finished) {
    if (call.id != 0) { response.set_header("X-Infernix-Request-Id", std::to_string(call.id)); }
    const std::size_t capture = store_.settings().log_bodies ? kBodyCaptureLimit : 0;
    forward_after_load(request, response, std::move(acquire), dialect, loading_note,
                       exchange_finished(request, call, std::move(finished)), call.arrived, capture,
                       call.id != 0 ? "launcher-" + std::to_string(call.id) : std::string());
}

void Launcher::register_routes() {
    const Settings settings = store_.settings();
    server_.new_task_queue  = [] { return new httplib::ThreadPool(16, 512); };
    server_.set_payload_max_length(512ull << 20);
    server_.set_read_timeout(std::chrono::minutes(5));
    server_.set_write_timeout(std::chrono::minutes(10));
    server_.set_keep_alive_max_count(1000);
    if (settings.cors) {
        server_.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                                     {"Access-Control-Allow-Headers", "*"},
                                     {"Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS"}});
    }
    server_.Options(R"(.*)", [](const Request&, Response& response) { response.status = 204; });
    server_.set_exception_handler([](const Request&, Response& response, std::exception_ptr error) {
        std::string message = "internal error";
        try {
            if (error) { std::rethrow_exception(error); }
        } catch (const std::exception& failure) { message = failure.what(); } catch (...) {}
        send_error(response, 500, "internal_error", message);
    });
    register_public_routes();
    register_management_routes();

    if (options_.ui_dir) {
        server_.set_mount_point("/", path_utf8(*options_.ui_dir));
    } else {
        const auto ui = [](const Request&, Response& response) {
            response.set_header("Cache-Control", "no-cache");
            if (assets::kUiGzipSize == 0) {
                response.set_content(
                    "<!doctype html><meta charset=utf-8><title>Infernix Launcher</title><body style=\"font-family:"
                    "system-ui;background:#111;color:#eee;padding:2rem\"><h1>Infernix Launcher</h1><p>This build has no "
                    "web UI (apps/launcher/assets/index.html.gz was missing). The OpenAI and Anthropic endpoints and "
                    "/launcher/api still work.</p>",
                    "text/html; charset=utf-8");
                return;
            }
            response.set_header("Content-Encoding", "gzip");
            response.set_content(reinterpret_cast<const char*>(assets::kUiGzip), assets::kUiGzipSize,
                                 "text/html; charset=utf-8");
        };
        server_.Get("/", ui);
        server_.Get("/index.html", ui);
    }
}

void Launcher::register_public_routes() {
    server_.Get(R"((?:/v1)?/health)", [this](const Request&, Response& response) {
        const InstanceSnapshot instance = supervisor_->snapshot();
        send_json(response, {{"status", "ok"},
                             {"state", state_name(instance.state)},
                             {"model", instance.alias.empty() ? Json(nullptr) : Json(instance.alias)}});
    });
    // Every public request becomes one event-log line (a proxied one when its exchange ends).
    using PublicHandler = void (Launcher::*)(const Request&, Response&, ApiCall&);
    const auto logged   = [this](PublicHandler handler) {
        return [this, handler](const Request& q, Response& s) {
            ApiCall call;
            call.arrived    = std::chrono::steady_clock::now();
            call.method     = q.method;
            call.path       = q.path;
            call.client     = q.remote_addr;
            call.user_agent = q.get_header_value("User-Agent");
            call.bytes_in   = q.body.size();
            (this->*handler)(q, s, call);
            if (call.proxied) { return; }
            log_api_request(call, s.status, std::nullopt, s.body.size());
            if (store_.settings().log_bodies && !call.quiet) { log_request_body(call, q.body, s.status, s.body, false); }
        };
    };
    // llama-server's router API when Settings.llamacpp_api is on (else 404, as unknown paths);
    // before /models/{id}, which would take /models/sse for a model id.
    server_.Get(R"((?:/v1)?/props)", logged(&Launcher::handle_props));
    server_.Post("/models/load", logged(&Launcher::handle_models_load));
    server_.Post("/models/unload", logged(&Launcher::handle_models_unload));
    server_.Get("/models/sse", logged(&Launcher::handle_models_sse));
    // Also without /v1, for tools given the bare address as their base URL (llama-swap and
    // llama.cpp answer /models too).
    server_.Get(R"((?:/v1){0,2}/models)", logged(&Launcher::handle_models));
    server_.Get(R"((?:/v1){0,2}/models/(.+))", logged(&Launcher::handle_models));
    server_.Post(
        R"((?:/v1)?/v1/(chat/completions|completions|responses|responses/input_tokens|responses/compact|messages|messages/count_tokens|decide|systemone))",
        logged(&Launcher::handle_generation));
    // Stored Responses live in the loaded server.
    server_.Get(R"((?:/v1)?/v1/responses/([^/]+)(/input_items)?)", logged(&Launcher::handle_stateful));
    server_.Post(R"((?:/v1)?/v1/responses/([^/]+)/cancel)", logged(&Launcher::handle_stateful));
    server_.Delete(R"((?:/v1)?/v1/responses/([^/]+))", logged(&Launcher::handle_stateful));
}

void Launcher::register_management_routes() {
    using Handler = std::function<void(const Request&, Response&)>;
    const auto admin = [this](Handler handler) {
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
    };
    const auto body_json = [](const Request& request) {
        if (request.body.empty()) { return Json::object(); }
        Json value = Json::parse(request.body);
        if (!value.is_object()) { throw std::invalid_argument("the request body must be a JSON object"); }
        return value;
    };
    const std::string api = "/launcher/api";

    server_.Get(api + "/state", admin([this](const Request&, Response& s) { send_json(s, state_json()); }));
    server_.Get(api + "/settings",
                admin([this](const Request&, Response& s) { send_json(s, settings_to_json(store_.settings())); }));
    server_.Put(api + "/settings", admin([this, body_json](const Request& q, Response& s) {
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
    server_.Post(api + "/dirs", admin([this, body_json](const Request& q, Response& s) {
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
        publish("library", library_json());
        publish_model_event("*", "models_reload"); // a model id or the listing may have changed
    };
    server_.Get(api + config_path + "/config", admin([this, with_config](const Request& q, Response& s) {
                    with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                        send_json(s, model_config_json(ref.id, *ref.model, scan));
                    });
                }));
    server_.Put(api + config_path + "/config",
                admin([this, with_config, body_json, parse_args, parse_env, check_alias, config_changed](const Request& q,
                                                                                                       Response& s) {
                    const Json patch = body_json(q);
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
                        if (patch.contains("args")) {
                            Args args = parse_args(patch["args"]);
                            // The default configuration follows the defaults until it differs; a named
                            // one keeps its own.
                            if (!named && args == model_default_args(m, scan)) {
                                stored.args.reset();
                            } else {
                                stored.args = std::move(args);
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
                 admin([this, with_config, body_json, parse_args, parse_env, check_alias, config_changed](const Request& q,
                                                                                                        Response& s) {
                     const Json body        = body_json(q);
                     const std::string name = trim(json_or<std::string>(body, "name", ""));
                     if (name.empty()) { throw std::invalid_argument("a configuration needs a name"); }
                     Json config;
                     with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                         // A copy of `from` (one of this artifact's configurations), else of the defaults.
                         std::string from = json_or<std::string>(body, "from", ref.id);
                         if (artifact_key_of(from) != ref.model->key || !find_config_locked(from)) { from = ref.id; }
                         StoredModelConfig stored;
                         stored.name = name;
                         stored.args = body.contains("args") ? parse_args(body["args"])
                                                             : effective_args(from, *ref.model, scan);
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
                       publish("library", library_json());
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
    server_.Post(api + config_path + "/preview", admin([this, with_config, body_json, parse_env](const Request& q, Response& s) {
                     const Json body = body_json(q);
                     with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                         Args args = effective_args(ref.id, *ref.model, scan);
                         if (body.contains("args") && body["args"].is_array()) {
                             args.clear();
                             for (const Json& arg : body["args"]) {
                                 if (arg.is_string()) { args.push_back(arg.get<std::string>()); }
                             }
                         }
                         const EnvVars env = body.contains("env") ? parse_env(body["env"]) : effective_env(ref.id);
                         send_json(s, preview_json(ref.id, *ref.model, scan, std::move(args), env));
                     });
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
    server_.Post(api + "/scripts/import", admin([this, body_json](const Request& q, Response& s) {
                     const Json body = body_json(q);
                     send_json(s, import_script_json(json_or<std::string>(body, "text", ""),
                                                     json_or<std::string>(body, "file_name", "")));
                 }));
    server_.Post(api + config_path + "/plan", admin([this, with_config, body_json, parse_args, parse_env](const Request& q, Response& s) {
                     const Json body = body_json(q);
                     std::string id;
                     ScannedModel model;
                     Args args;
                     EnvVars env;
                     with_config(q, [&](const ConfigRef& ref, const LibraryScan& scan) {
                         id    = ref.id;
                         model = *ref.model;
                         args  = body.contains("args") ? parse_args(body["args"]) : effective_args(ref.id, *ref.model, scan);
                         env   = body.contains("env") ? parse_env(body["env"]) : effective_env(ref.id);
                     });
                     send_json(s, fit_plan_json(id, model, args, env));
                 }));
    server_.Post(api + config_path + "/load", admin([this, with_config, body_json](const Request& q, Response& s) {
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

    // Live state.
    server_.Get(api + "/instance", admin([this](const Request&, Response& s) { send_json(s, instance_json()); }));
    server_.Get(api + "/logs", admin([this](const Request& q, Response& s) {
                    std::uint64_t after = 0;
                    std::size_t limit   = 2000;
                    if (q.has_param("after")) { after = std::stoull(q.get_param_value("after")); }
                    if (q.has_param("limit")) { limit = std::clamp<std::size_t>(std::stoull(q.get_param_value("limit")), 1, 20000); }
                    send_json(s, logs_.page(after, limit));
                }));
    server_.Get(api + "/eventlog", admin([this](const Request& q, Response& s) {
                    std::uint64_t after = 0;
                    std::size_t limit   = 2000;
                    if (q.has_param("after")) { after = std::stoull(q.get_param_value("after")); }
                    if (q.has_param("limit")) { limit = std::clamp<std::size_t>(std::stoull(q.get_param_value("limit")), 1, 5000); }
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

    // Catalog and downloads.
    server_.Get(api + "/catalog", admin([this](const Request&, Response& s) { send_json(s, catalog_); }));
    server_.Get(api + "/hf/files", admin([this](const Request& q, Response& s) {
                    send_json(s, hf_files_json(trim(q.get_param_value("repo"))));
                }));
    server_.Get(api + "/downloads", admin([this](const Request&, Response& s) { send_json(s, downloads_->list()); }));
    server_.Post(api + "/downloads", admin([this, body_json](const Request& q, Response& s) {
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
                     // the process and the update script takes over.
                     std::thread([this] {
                         std::this_thread::sleep_for(std::chrono::milliseconds(500));
                         server_.stop();
                     }).detach();
                 }));

    // Disk: deleting an artifact's files, a drafter or an n-gram volume.
    const auto check_confirm = [body_json](const Request& q, const std::string& file_name) {
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
                failed.push_back(path_utf8(file) + (error ? " (" + error.message() + ")" : ""));
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
                       if (supervisor_->snapshot().artifact_key == key &&
                           supervisor_->snapshot().state != InstanceState::Idle &&
                           supervisor_->snapshot().state != InstanceState::Failed) {
                           throw std::invalid_argument(file_name + " is loaded: unload it first");
                       }
                       check_confirm(q, file_name);
                       remove_files(files, file_name);
                       for (const auto& [id, config] : store_.named_configs(key)) { store_.erase_model_config(id); }
                       store_.erase_model_config(key);
                       request_scan();
                       send_json(s, library_json());
                   }));
    server_.Delete(api + R"(/(drafters|ngram)/([0-9a-f]{32}))", admin([this, check_confirm, remove_files](const Request& q, Response& s) {
                       std::filesystem::path path;
                       std::string file_name;
                       {
                           std::lock_guard lock(library_mutex_);
                           const std::string key = q.matches[2].str();
                           if (q.matches[1].str() == "drafters") {
                               for (const ScannedDrafter& drafter : library_.drafters) {
                                   if (drafter.key == key) { path = drafter.path, file_name = drafter.file_name; }
                               }
                           } else {
                               for (const ScannedNgramVolume& volume : library_.ngram_volumes) {
                                   if (volume.key == key) { path = volume.path, file_name = volume.file_name; }
                               }
                           }
                       }
                       if (path.empty()) { throw std::invalid_argument("no such file in the library"); }
                       // A running server may have it open.
                       const InstanceSnapshot instance = supervisor_->snapshot();
                       if (instance.state != InstanceState::Idle && instance.state != InstanceState::Failed) {
                           for (const std::string& arg : instance.argv) {
                               if (iequals(arg, path_utf8(path))) {
                                   throw std::invalid_argument(file_name + " is in use by " + instance.alias + ": unload it first");
                               }
                           }
                       }
                       check_confirm(q, file_name);
                       remove_files({path}, file_name);
                       request_scan();
                       send_json(s, library_json());
                   }));

    // API keys and sign-in.
    server_.Post(api + "/keys", admin([this, body_json](const Request& q, Response& s) {
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
                    send_json(s, chats_.put(q.matches[1].str(), Json::parse(q.body)));
                }));
    server_.Delete(api + R"(/chats/([A-Za-z0-9_-]{1,64}))", admin([this](const Request& q, Response& s) {
                       chats_.remove(q.matches[1].str());
                       send_json(s, {{"ok", true}});
                   }));

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

    // Folder picker.
    server_.Get(api + "/fs", admin([](const Request& q, Response& s) { send_json(s, list_folder(q.get_param_value("path"))); }));
    server_.Post(api + "/fs/mkdir", admin([body_json](const Request& q, Response& s) {
                     const std::string path = trim(json_or<std::string>(body_json(q), "path", ""));
                     if (path.empty()) { throw std::invalid_argument("no folder given"); }
                     std::error_code error;
                     std::filesystem::create_directories(utf8_path(path), error);
                     if (error) { throw std::invalid_argument("cannot create " + path + ": " + error.message()); }
                     send_json(s, list_folder(path));
                 }));

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
