#include "launcher/launcher.h"

#include "launcher/assets.h"
#include "launcher/hf_hub.h"
#include "launcher/http_access.h"
#include "serve/cors_headers.h"
#include "launcher/load_advice.h"
#include "launcher/model_defaults.h"
#include "launcher/option_support.h"
#include "launcher/process.h"
#include "launcher/proxy.h"
#include "launcher/route_support.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <unistd.h>
#endif

namespace infernix::launcher {
namespace {

// The console writer's output, written to the handle itself rather than through stdout's FILE: a write a console
// holds (a QuickEdit selection) then holds no C runtime lock, so the exit's stream flush is not held up by it.
void write_console(const std::vector<std::string>& lines) {
    std::string text;
    for (const std::string& line : lines) {
#ifdef _WIN32
        text += line + "\r\n";
#else
        text += line + "\n";
#endif
    }
#ifdef _WIN32
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    for (std::size_t at = 0; out != nullptr && out != INVALID_HANDLE_VALUE && at < text.size();) {
        DWORD wrote = 0;
        if (WriteFile(out, text.data() + at, static_cast<DWORD>(std::min<std::size_t>(text.size() - at, 1U << 20)), &wrote,
                      nullptr) == 0 || wrote == 0) {
            break;
        }
        at += wrote;
    }
#else
    for (std::size_t at = 0; at < text.size();) {
        const ssize_t wrote = ::write(STDOUT_FILENO, text.data() + at, text.size() - at);
        if (wrote < 0 && errno == EINTR) { continue; }
        if (wrote <= 0) { break; }
        at += static_cast<std::size_t>(wrote);
    }
#endif
}

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

// The API key a request sends, as `Authorization: Bearer`, `x-api-key` or `api-key`; "" for none.
std::string presented_key(const httplib::Request& request) {
    std::string bearer = request.get_header_value("Authorization");
    if (bearer.size() > 7 && iequals(bearer.substr(0, 7), "Bearer ")) { return trim(bearer.substr(7)); }
    for (const char* header : {"x-api-key", "api-key"}) {
        if (std::string value = trim(request.get_header_value(header)); !value.empty()) { return value; }
    }
    return {};
}

int pick_free_port() {
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    probe.stop();
    if (port <= 0) { throw std::runtime_error("no free loopback port for infernix-serve"); }
    return port;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Construction

Launcher::Launcher(LauncherOptions options)
    : options_(std::move(options)), data_lock_(options_.data_dir), store_(options_.data_dir),
      event_log_(resolve_log_dir(store_.settings(), options_.data_dir)),
      logs_(resolve_log_dir(store_.settings(), options_.data_dir) / kServeLogDir),
      applied_log_dir_(resolve_log_dir(store_.settings(), options_.data_dir)), usage_(options_.data_dir),
      chats_(options_.data_dir / "chats") {
    try {
        catalog_ = Json::parse(std::string(reinterpret_cast<const char*>(assets::kCatalogJson), assets::kCatalogJsonSize));
    } catch (const Json::exception&) { catalog_ = {{"entries", Json::array()}}; }

    // The console shows the event log (but its debug records) and the server's errors; the web
    // UI shows everything. The listeners only queue: the server's output pipe is read on the
    // thread that calls the log's.
    event_log_.set_listener([this](const EventRecord& record) {
        if (record.level != "debug") { print_console(format_event_line(record).substr(11)); } // without the date
        std::lock_guard lock(pending_logs_mutex_);
        pending_events_.push_back(record);
    });
    logs_.set_listener([this](const std::vector<LogRecord>& records) {
        if (supervisor_) {
            for (const LogRecord& record : records) { supervisor_->observe(record); }
        }
        for (const LogRecord& record : records) {
            if (record.source == "server" && (record.level == "error" || record.level == "critical")) {
                char head[48];
                std::snprintf(head, sizeof(head), "%s  %-5s  serve | ", local_clock(record.t).c_str(),
                              record.level.c_str());
                print_console(head + record.text);
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
        // The instance starting now ends (the monitor kills it within its next poll); one not started yet is never
        // started (load_model asks load_cancelled first).
        .cancel_load =
            [this] {
                supervisor_->end_load(supervisor_->snapshot().instance, "cancelled", "the load was cancelled from the launcher");
            },
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

    ablation_ = std::make_unique<Ablation>(
        options_.data_dir, options_.ablate_tool,
        Ablation::Hooks{
            .version     = options_.version,
            .take_gpu    = [this](const std::function<bool()>& abort) { return take_gpu_for_ablation(abort); },
            .release_gpu = [this] { release_gpu_from_ablation(); },
            .changed     = [this](const Json& state) { publish("ablation", state); },
            .event       = [this](const std::string& level, const std::string& message) { event(level, "ablation", message); },
            .toast       = [this](const std::string& level, const std::string& message) { toast(level, message); },
            .exported    = [this](const std::filesystem::path& folder) { download_finished(ablation_->sidecars_dir(), folder); },
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

    scanner_        = std::thread([this] { scan_loop(); });
    log_flusher_    = std::thread([this] { flush_logs_loop(); });
    console_writer_ = std::thread([queue = console_] { console_loop(queue); });
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
              settings.allow_remote_admin
                  ? "listening on " + bound_host_ +
                        " with remote management and no API key: anyone who can reach the port can use the models, "
                        "change every setting and choose the program the launcher runs"
                  : "listening on " + bound_host_ + " without an API key: anyone who can reach the port can use the models");
    }
    if (settings.autoload) {
        std::optional<std::string> key = settings.default_model ? settings.default_model : store_.last_model();
        if (key) {
            autoloader_ = std::thread([this, key = *key] {
                // The first scan (after `infernix-serve --help`, however long both take) makes the model
                // known; a load requested before it would fail and hold the model off for two minutes.
                {
                    std::unique_lock lock(library_mutex_);
                    scan_cv_.wait(lock, [this] { return library_.scanned_at != 0 || stopping_; });
                    if (stopping_) { return; }
                }
                router_->request_load(key, false);
            });
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
    // Saved first as well: a closed console window, a sign-out or a shutdown ends the process about 5 s from now,
    // whatever the model's stop below still waits for.
    usage_.flush();
    store_.update_settings([](Settings&) {});
    events_.close_all();
    model_events_.close_all();
    server_.stop();
    // An ablation run first: the tool stops its engines (and keeps its finished work) before the GPU is let go.
    if (ablation_) { ablation_->shutdown(); }
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
    if (autoloader_.joinable()) { autoloader_.join(); }
    {
        std::lock_guard lock(install_stopper_mutex_);
        if (install_stopper_.joinable()) { install_stopper_.join(); }
    }
    // The server's log file and the console get what is still queued.
    stop_flushing_ = true;
    if (log_flusher_.joinable()) { log_flusher_.join(); }
    if (console_writer_.joinable()) {
        // A few seconds for the last lines; a console held by a QuickEdit selection does not hold up the exit (the
        // writer keeps its queue alive and is ended with the process).
        std::unique_lock lock(console_->mutex);
        console_->stopping = true;
        console_->cv.notify_all();
        const bool done = console_->cv.wait_for(lock, std::chrono::seconds(3), [this] { return console_->done; });
        lock.unlock();
        if (done) {
            console_writer_.join();
        } else {
            console_writer_.detach();
        }
    }
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
    // Through the shutdown too (the server's last lines while it stops), then once more after it.
    for (;;) {
        const bool last = stop_flushing_;
        if (!last) { std::this_thread::sleep_for(std::chrono::milliseconds(150)); }
        // A failed save of the memory file (a full disk, a read-only folder, an antivirus lock): the change is
        // in memory only, so the user is told rather than finding it gone at the next start.
        if (const std::optional<std::string> failure = store_.take_save_failure()) {
            event("error", "settings", "the settings file could not be saved; the change is lost at the next start "
                                        "unless a later save succeeds: " + *failure);
            toast("error", "Settings could not be saved: " + *failure);
        }
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
        if (!batch.empty()) {
            // Listeners run outside the log lock, so records from two threads can arrive swapped.
            std::sort(batch.begin(), batch.end(), [](const LogRecord& a, const LogRecord& b) { return a.seq < b.seq; });
            Json lines = Json::array();
            for (const LogRecord& record : batch) { lines.push_back(to_json(record)); }
            publish("log", {{"lines", lines}});
        }
        logs_.write_files();
        event_log_.write_files(); // the event and request-content logs: their only writer
        if (last) { return; }
    }
}

void Launcher::print_console(std::string line) {
    {
        std::lock_guard lock(console_->mutex);
        console_->lines.push_back(std::move(line));
    }
    console_->cv.notify_all();
}

void Launcher::console_loop(const std::shared_ptr<ConsoleQueue>& queue) {
    std::unique_lock lock(queue->mutex);
    for (;;) {
        queue->cv.wait(lock, [&] { return queue->stopping || !queue->lines.empty(); });
        if (queue->lines.empty()) { break; } // stopping, and everything printed
        std::vector<std::string> lines;
        lines.swap(queue->lines);
        lock.unlock();
        write_console(lines);
        lock.lock();
    }
    queue->done = true;
    queue->cv.notify_all();
}

Json Launcher::state_json() const {
    const Settings settings = store_.settings();
    const std::string base  = ui_url();
    static const std::string run = random_token(16); // this process: its sequence numbers start afresh
    return {{"version", options_.version},
            {"run", run},
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
    // Runs `infernix-serve --help` now, so the first API calls find it cached.
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
        if (requested) { publish_library(); }
        const Settings settings = store_.settings();
        std::vector<std::filesystem::path> dirs;
        for (const std::string& dir : settings.model_dirs) { dirs.push_back(utf8_path(dir)); }
        LibraryScan scan = scan_library(dirs, settings.scan_depth);
        lock.lock();
        library_  = std::move(scan);
        scanning_ = false;
        lock.unlock();
        scan_cv_.notify_all(); // the autoload waits for the first scan
        Json compare = library_json();
        compare.erase("scanned_at");
        const std::string digest = compare.dump();
        lock.lock();
        const bool changed = digest != library_digest_;
        library_digest_    = digest;
        lock.unlock();
        if (changed || requested) {
            publish_library();
            publish("activity", activity_json());
        }
        if (changed) { publish_model_event("*", "models_reload"); }
        lock.lock();
    }
}

Args Launcher::effective_args(const std::string& config_id, const ScannedModel& model, const LibraryScan& scan) const {
    const std::optional<StoredModelConfig> stored = store_.model_config(config_id);
    if (stored && stored->args) {
        // A saved value for an option this model no longer takes never reaches the command line.
        Args args = *stored->args;
        strip_unsupported_options(args, model);
        return args;
    }
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
    return default_args(scan, model);
}

std::uint64_t Launcher::advertised_context(const std::string& config_id, const ScannedModel& model,
                                           const Args& args) const {
    if (!context_auto(args)) { return context_length(args); }
    const InstanceSnapshot instance = supervisor_->snapshot();
    if (instance.model_key == config_id && instance.state == InstanceState::Ready) {
        const Json metrics = supervisor_->metrics();
        if (metrics.is_object() && metrics.contains("engine") && metrics["engine"].is_object()) {
            const Json& resolved = metrics["engine"].value("max_context", Json());
            if (resolved.is_number_unsigned() && resolved.get<std::uint64_t>() > 0) { return resolved.get<std::uint64_t>(); }
        }
    }
    return model.native_context != 0 ? std::min<std::uint64_t>(model.native_context, 262144) : 8192;
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
    const Args args             = effective_args(config_id, model, scan); // without options this model does not take
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
            {"context_length", advertised_context(config_id, model, args)},
            {"context_auto", context_auto(args)},
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
    // Each folder's share of the library (its drive's room comes with the scan: no drive is touched
    // under the library lock).
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
        for (const ScannedAdapter& adapter : library_.adapters) {
            if (same_or_inside(adapter.path, path, 64)) { bytes += adapter.bytes; }
        }
        for (const ScannedSidecar& sidecar : library_.sidecars) {
            if (inside(sidecar.path)) { bytes += sidecar.bytes; }
        }
        dir["library_bytes"] = bytes;
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

namespace {
constexpr const char* kAblationHoldsGpu =
    "an ablation run has the GPU (the Ablation page): a model loads once it ends or is stopped";
}

bool Launcher::ablation_holds_gpu() const { return gpu_gate_.load(); }

bool Launcher::take_gpu_for_ablation(const std::function<bool()>& abort) {
    {
        std::lock_guard gate(gpu_gate_mutex_); // after a server start in progress
        gpu_gate_ = true;
    }
    event("info", "ablation", "the ablation run takes the GPU: the model is unloaded, and none loads until the run ends");
    router_->request_unload();
    // A load the unload cancelled ends in load_model, whose start it passed (or not) under the gate's lock.
    while (!stopping_ && !abort()) {
        const InstanceSnapshot instance = supervisor_->snapshot();
        if (!router_->loading_model() && (instance.state == InstanceState::Idle || instance.state == InstanceState::Failed)) {
            publish_instance();
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    release_gpu_from_ablation();
    return false;
}

void Launcher::release_gpu_from_ablation() {
    gpu_gate_ = false;
    if (router_) { router_->clear_holds(); }
    publish_instance();
}

AblationRun Launcher::ablation_run_for(const std::string& config_id) {
    LibraryScan scan;
    std::string id;
    {
        std::lock_guard lock(library_mutex_);
        const std::optional<ConfigRef> ref = find_config_locked(config_id);
        if (!ref) { throw std::invalid_argument("no configuration " + config_id); }
        id   = ref->id;
        scan = library_;
    }
    const ScannedModel* model = scan.model(artifact_key_of(id));
    if (model == nullptr) { throw std::invalid_argument("the model is not in the library"); }
    if (!model->missing_files.empty()) { throw std::invalid_argument(model->file_name + " has missing or incomplete parts"); }
    if (model->family == "unknown") {
        throw std::invalid_argument(model->file_name + " is a " + model->model_type + " model, which Infernix does not run");
    }
    const std::filesystem::path executable = serve_executable();
    std::error_code error;
    if (!std::filesystem::is_regular_file(executable, error)) {
        throw std::invalid_argument("infernix-serve was not found at " + path_utf8(executable) + " (Settings, Server)");
    }
    const ServeOptions serve = serve_options();
    const TakesValue takes   = [&serve](std::string_view flag) { return serve.takes_value(flag); };
    Args args                = effective_args(id, *model, scan);
    strip_managed_flags(args, takes);
    remove_flag(args, "--ablate", takes); // a fit measures the plain model
    AblationRun run;
    run.config_id  = id;
    run.model_name = effective_alias(id, *model, scan);
    // Absolute, as the tool starts its engines elsewhere; the other relative paths are the data folder's (--launch-dir).
    run.argv = {path_utf8(executable), path_utf8(std::filesystem::absolute(model->path, error))};
    run.argv.insert(run.argv.end(), args.begin(), args.end());
    for (const auto& [name, value] : effective_env(id)) { run.env.emplace_back(name, value); }
    return run;
}

Router::Loaded Launcher::load_model(const std::string& key) {
    Router::Loaded result;
    if (ablation_holds_gpu()) {
        // Not a failure of the model: it is not held off once the run ends.
        result.error     = kAblationHoldsGpu;
        result.cancelled = true;
        return result;
    }
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
        spec.port_retry = settings.child_port == 0 && attempt < kPortAttempts;
        if (router_ && router_->load_cancelled()) {
            result.error     = alias + ": the load was cancelled from the launcher";
            result.cancelled = true;
            return result;
        }
        try {
            // Under the gate's lock: an ablation run that takes the GPU waits for this start, and its unload ends it.
            std::lock_guard gate(gpu_gate_mutex_);
            if (gpu_gate_) {
                result.error     = kAblationHoldsGpu;
                result.cancelled = true;
                return result;
            }
            supervisor_->start(spec);
        } catch (const std::exception& failure) {
            result.error = failure.what();
            return result;
        }
        // start() first joins the previous instance's monitor (which can sit in a 20 s plan run): an Unload or another
        // load in that window found the supervisor idle and stopped nothing, so the new instance stops here.
        if (router_ && router_->load_cancelled()) {
            supervisor_->stop(std::chrono::seconds(settings.stop_timeout_s));
            result.error     = alias + ": the load was cancelled from the launcher";
            result.cancelled = true;
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
    if (!ready && router_ && router_->load_cancelled()) {
        if (snapshot.state == InstanceState::Starting) { supervisor_->stop(std::chrono::seconds(settings.stop_timeout_s)); }
        result.error     = alias + ": the load was cancelled from the launcher";
        result.cancelled = true;
        return result;
    }
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
            {"incident", incident},
            {"ablation_holds_gpu", ablation_holds_gpu()}};
}

void Launcher::publish_instance() {
    std::lock_guard lock(publish_instance_mutex_);
    publish("instance", instance_json());
}

void Launcher::publish_library() {
    std::lock_guard lock(publish_library_mutex_);
    publish("library", library_json());
}

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
    // Two settings saves at once (handlers run in parallel) apply the folders in order: the last one read is the
    // one both logs end up in.
    bool moved = false;
    std::string error;
    std::filesystem::path dir;
    {
        std::lock_guard lock(log_dir_mutex_);
        dir = log_dir();
        if (dir == applied_log_dir_) { return; }
        applied_log_dir_ = dir;
        moved            = event_log_.set_dir(dir, error);
        logs_.set_dir(dir / kServeLogDir);
    }
    if (moved) {
        event("info", "settings", "the logs are now written to " + path_utf8(dir));
    } else {
        event("error", "settings", "the log folder cannot be written (" + error + "); events stay in memory only");
        toast("error", "The log folder cannot be written: " + error);
    }
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
        } else if (snapshot.incident == "cancelled") {
            event("info", "model", alias + ": the load was cancelled"); // an unload or another load: no restart
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
    return snapshot.port_retry && !snapshot.incident && classify_load_failure(failure_detail(snapshot)) == "port";
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
        if (!reading->gpus.empty()) { // the inference GPU, listed first
            facts.vram_total = reading->gpus.front().vram_total;
            facts.vram_used  = reading->gpus.front().vram_used;
        }
    }
    Json suggestions = Json::array();
    if (model && kind == "vram" && !stopping_) {
        // The server's own plan (about a second; the failed server's memory is free again) names the
        // largest context that fits exactly, where the refusal's figures give an estimate. It runs here,
        // on the supervisor's thread, so the next load starts only after it and does not fill the GPU
        // the plan measures; the next load and every model switch wait for it, so it gets 20 s (a plan
        // that hangs, a driver wedged by the out-of-memory failure, say, leaves the estimate). The
        // dashboard shows the failure meanwhile.
        publish_instance();
        const Json plan = fit_plan_json(snapshot.model_key, *model, args, env, std::chrono::seconds(20));
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
    } else if (router_ && router_->unloaded(snapshot.instance)) {
        action = "not restarted: it was unloaded";
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
    // An Unload landing since the check above still wins.
    if (restart && !(router_ && router_->restart(snapshot.model_key, snapshot.instance))) {
        restart = false;
        action  = "not restarted: it was unloaded";
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
    // JSON bodies are kept as JSON; anything else (a stream's events, a body nested too deep to store as
    // JSON) as text.
    const auto value = [](const std::string& text) {
        try {
            return parse_client_json(text);
        } catch (const std::exception&) { return Json(text); }
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
    // A PEFT LoRA adapter's two files (--lora), at the repository's root. Every adapter's files have these names, and
    // adapters of one rank and module set have the same sizes: one is in the library only in the folder a download puts
    // it in, named for its owner and repository (adapter_folder_name), so another owner's adapter of the same name is
    // not taken for it.
    const auto adapter_file = [](const std::string& path) {
        return path == "adapter_config.json" || path == "adapter_model.safetensors";
    };
    const bool adapter = !has_model && std::count_if(listing.files.begin(), listing.files.end(), [&](const HfFile& file) {
                                           return adapter_file(file.path);
                                       }) == 2;
    const std::string folder = adapter_folder_name(listing.repo);
    const auto adapter_present = [&](const HfFile& file) -> Json {
        for (const std::string& dir : settings.model_dirs) {
            const std::filesystem::path root = utf8_path(dir);
            for (const std::filesystem::path& candidate :
                 {root / utf8_path(folder) / utf8_path(file.path), root / utf8_path(file.path)}) {
                if (!iequals(path_utf8(candidate.parent_path().filename()), folder)) { continue; }
                std::error_code status;
                if (std::filesystem::is_regular_file(candidate, status) &&
                    std::filesystem::file_size(candidate, status) == file.size) {
                    return path_utf8(candidate);
                }
            }
        }
        return nullptr;
    };
    Json files = Json::array();
    for (const HfFile& file : listing.files) {
        const std::string lowered = lower(file.path);
        std::string role          = "other";
        if (adapter && adapter_file(file.path)) {
            files.push_back({{"path", file.path},
                             {"bytes", file.size},
                             {"sha256", file.sha256 ? Json(*file.sha256) : Json(nullptr)},
                             {"role", "adapter"},
                             {"default_selected", true},
                             {"present_locally", adapter_present(file)}});
            continue;
        }
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
    // An adapter's config (about 1 kB) says its base model and rank before its weights are downloaded.
    Json adapter_facts = nullptr, adapter_error = nullptr;
    if (adapter) {
        const HttpResult config =
            http_get(hf_resolve_url(listing.repo, listing.revision, "adapter_config.json"), store_.hf_token(), 30);
        if (!config.error.empty() || config.status != 200) {
            adapter_error = "adapter_config.json: " +
                            (config.error.empty() ? "HTTP " + std::to_string(config.status) : config.error);
        } else {
            try {
                adapter_facts = to_json(parse_adapter_config(Json::parse(config.body)));
            } catch (const Json::exception&) { adapter_error = "adapter_config.json is not JSON"; }
        }
    }
    return {{"repo", listing.repo},
            {"revision", listing.revision},
            {"kind", adapter ? "adapter" : kind},
            {"files", files},
            {"adapter", adapter_facts},
            {"adapter_error", adapter_error}};
}

Json Launcher::hf_adapters_json(const std::string& base, const std::string& search) const {
    if (search.size() > 100) { throw std::invalid_argument("the search is longer than 100 characters"); }
    // Harness Hub (huggingface.co/harness-hub) trains adapters for agent harnesses (Pi, OpenCode, Hermes): they come
    // first, whatever their downloads.
    constexpr const char* kHarnessHub = "harness-hub";
    const std::string token                 = store_.hf_token();
    const std::vector<HfAdapter> featured   = hf_search_adapters(base, kHarnessHub, search, 50, token);
    const std::vector<HfAdapter> downloaded = hf_search_adapters(base, "", search, 60, token);
    Json adapters = Json::array();
    std::set<std::string> listed;
    for (const std::vector<HfAdapter>* list : {&featured, &downloaded}) {
        for (const HfAdapter& adapter : *list) {
            if (!listed.insert(lower(adapter.repo)).second) { continue; }
            adapters.push_back({{"repo", adapter.repo},
                                {"downloads", adapter.downloads},
                                {"likes", adapter.likes},
                                {"last_modified", adapter.last_modified},
                                {"gated", adapter.gated},
                                {"harness_hub", iequals(adapter.repo.substr(0, adapter.repo.find('/')), kHarnessHub)}});
        }
    }
    return {{"base", base}, {"search", search}, {"adapters", adapters}};
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
    // `Access-Control-Allow-Origin: *`): http_access.h's page_refusal.
    const std::optional<std::string> refusal =
        page_refusal(request.get_header_value("Host"), request.get_header_value("Origin"), has_key);
    if (!refusal) { return true; }
    if (response) { send_error(*response, 403, "forbidden", *refusal); }
    return false;
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


void Launcher::register_routes() {
    const Settings settings = store_.settings();
    server_.new_task_queue  = [] { return new httplib::ThreadPool(16, 512); };
    server_.set_payload_max_length(512ull << 20);
    server_.set_read_timeout(std::chrono::minutes(5));
    server_.set_write_timeout(std::chrono::minutes(10));
    server_.set_keep_alive_max_count(1000);
    // Clients reach the models through this server, so its keep-alive window is the one their
    // pools meet (the proxy opens a fresh connection to infernix-serve for every request). As
    // infernix-serve's default: the advertised 360 s outlasts Bun's fetch idle timeout (5 minutes
    // on a minute wheel), and the close follows the advertised timeout by 10 s, so neither a
    // header-blind pool (opencode) nor one that follows the header writes a request onto a
    // connection this server is closing; a client that probes a pooled socket can still lose a
    // request written while a close is in flight (cpp-httplib's default of 5 s closed it between
    // almost every pair of turns). An idle connection holds one of the pool's 512 workers.
    server_.set_keep_alive_timeout(std::chrono::seconds(360));
    server_.set_keep_alive_timeout_buffer(10);
    if (settings.cors) {
        // infernix-serve's headers (Authorization named, "*" for the rest of a page's headers) and the launcher's own
        // api-key.
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Allow-Headers", std::string(serve::kCorsAllowHeaders) + ", api-key"},
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


} // namespace infernix::launcher
