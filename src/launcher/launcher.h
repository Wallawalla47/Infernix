#pragma once

// The launcher service: one HTTP port for the web UI, its management API (/launcher/api) and the
// OpenAI- and Anthropic-compatible endpoints coding tools use. Requests name a model; the router
// loads it (draining the previous one), and the proxy forwards to the managed infernix-serve. The
// supervisor's watchdog reports crashes, lock-ups and stalls; the launcher restarts the model (at
// most kRestartLimit times in kRestartWindow) and records everything in its event log.

#include "launcher/artifact_scan.h"
#include "launcher/chats.h"
#include "launcher/downloads.h"
#include "launcher/event_log.h"
#include "launcher/events.h"
#include "launcher/log_buffer.h"
#include "launcher/proxy.h"
#include "launcher/router.h"
#include "launcher/serve_help.h"
#include "launcher/store.h"
#include "launcher/supervisor.h"
#include "launcher/telemetry.h"
#include "launcher/updates.h"
#include "launcher/usage.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace infernix::launcher {

struct LauncherOptions {
    std::filesystem::path data_dir;
    std::optional<std::string> host;
    std::optional<int> port;
    std::optional<std::filesystem::path> serve_executable;
    std::optional<std::filesystem::path> ui_dir;
    bool open_browser = true;
    std::string version;
    // The command-line arguments, for starting again after an update.
    std::vector<std::string> restart_args;
};

class Launcher {
public:
    explicit Launcher(LauncherOptions options);
    ~Launcher();

    // Binds the listen address; false (with `error`) when it is taken or invalid.
    bool bind(std::string& error);
    // Serves until stop(); blocks.
    void serve();
    // Orderly shutdown: the managed server is stopped (and saves its caches), downloads pause.
    void stop();

    [[nodiscard]] std::string ui_url() const;
    [[nodiscard]] std::string event_log_file() const;

private:
    struct ProxyRecord {
        std::uint64_t id = 0;
        std::int64_t t   = 0;
        std::string method;
        std::string path;
        std::optional<std::string> model;
        std::optional<std::string> model_key;
        bool stream = false;
        int status  = 0;
        std::string state = "waiting";
        std::optional<std::int64_t> duration_ms;
        std::optional<std::string> error;
        std::string client;
        std::optional<std::string> key_name;
        std::optional<std::string> routed_by;
        std::optional<std::string> label; // the client's own X-Infernix-Request-Label, forwarded as it is
        Json progress = nullptr; // the server's live progress while it runs (RequestProgress)
        bool usage_counted = false;
    };

    // Library and configurations. A configuration id is an artifact key (its default
    // configuration) or "<artifact key>@<8 hex>" (a named one).
    struct ConfigRef {
        std::string id;
        const ScannedModel* model = nullptr; // into library_ (the library lock held)
    };
    void request_scan();
    void scan_loop();
    [[nodiscard]] Json library_json() const;
    [[nodiscard]] std::optional<ConfigRef> find_config_locked(const std::string& config_id) const;
    [[nodiscard]] std::vector<std::string> config_ids(const ScannedModel& model) const;
    [[nodiscard]] std::string config_name(const std::string& config_id) const;
    [[nodiscard]] std::string config_default_alias(const std::string& config_id, const ScannedModel& model,
                                                   const LibraryScan& scan) const;
    [[nodiscard]] Json model_config_json(const std::string& config_id, const ScannedModel& model,
                                         const LibraryScan& scan) const;
    [[nodiscard]] Args effective_args(const std::string& config_id, const ScannedModel& model,
                                      const LibraryScan& scan) const;
    [[nodiscard]] EnvVars effective_env(const std::string& config_id) const;
    [[nodiscard]] Args model_default_args(const ScannedModel& model, const LibraryScan& scan) const;
    // The configuration using this model id, if any.
    [[nodiscard]] std::optional<std::string> alias_owner(const std::string& alias) const;
    // POST /scripts/import: the script's server commands, matched to the library.
    [[nodiscard]] Json import_script_json(const std::string& text, const std::string& file_name) const;
    // POST /models/{id}/plan: infernix-serve --plan for these parameters, as a FitPlan.
    [[nodiscard]] Json fit_plan_json(const std::string& config_id, const ScannedModel& model, const Args& args,
                                     const EnvVars& env) const;
    [[nodiscard]] std::string effective_alias(const std::string& config_id, const ScannedModel& model,
                                              const LibraryScan& scan) const;
    // Not under the library lock: it may run --help and looks for the files the arguments name.
    [[nodiscard]] Json preview_json(const std::string& config_id, const ScannedModel& model, const std::string& alias,
                                    Args args, const EnvVars& env) const;
    [[nodiscard]] std::filesystem::path serve_executable() const;
    [[nodiscard]] ServeOptions serve_options() const;
    [[nodiscard]] std::optional<std::string> resolve_model(const std::string& name) const;

    // Model lifecycle and the watchdog's incidents.
    Router::Loaded load_model(const std::string& key);
    [[nodiscard]] Json instance_json() const;
    void publish_instance();
    void on_instance_changed(const InstanceSnapshot& snapshot);
    void handle_incident(const InstanceSnapshot& snapshot);
    // A load that did not fit ("vram" or "ram"): the error and the launch-parameter changes to try.
    void handle_load_failure(const InstanceSnapshot& snapshot, const std::string& kind, const std::string& detail);
    // Why a server that failed ended: its error and its last warnings and errors.
    [[nodiscard]] std::string failure_detail(const InstanceSnapshot& snapshot) const;
    // Whether `snapshot` failed only because another program took its automatic port first, which
    // load_model retries on another port: nothing is reported for it.
    [[nodiscard]] bool port_retry_due(const InstanceSnapshot& snapshot) const;
    // While a model loads: ends the load when free RAM stays below Settings::load_ram_floor_mib.
    void guard_load_memory(const SystemReading& reading);

    // The event log.
    void event(const std::string& level, const std::string& category, const std::string& message,
               Json fields = nullptr);
    [[nodiscard]] std::filesystem::path log_dir() const;
    [[nodiscard]] Json log_paths_json() const;
    void apply_log_dir();
    // One public API request, for the event log (copied: a stream's log line is written after the
    // handler has returned).
    struct ApiCall {
        std::uint64_t id = 0; // the activity record's; 0 when refused before one
        std::chrono::steady_clock::time_point arrived;
        std::string method;
        std::string path;
        std::string client;
        std::string user_agent;
        std::uint64_t bytes_in = 0;
        std::optional<std::string> model; // the name the request gave
        std::optional<std::string> key;   // the model that served it
        bool stream  = false;
        bool proxied = false; // the proxy writes the line when the exchange ends
        bool quiet   = false; // a debug record (model lists, which tools poll)
        std::optional<std::string> error;
        std::optional<std::string> key_name;  // the API key it sent
        std::optional<std::string> routed_by; // name, route, pinned, loaded
        bool from_ui = false;                 // the launcher's own page (the chat)
    };
    void log_api_request(const ApiCall& call, int status, std::optional<std::int64_t> first_byte_ms, std::uint64_t bytes_out);
    void log_request_body(const ApiCall& call, const std::string& request_body, int status,
                          const std::string& response_body, bool truncated);

    // HTTP.
    void register_routes();
    void register_management_routes();
    void register_public_routes();
    // The management API: this computer, or (remote management on) anyone while no key exists and
    // an admin key or its sign-in once one does; always after page_checks. Otherwise the error is written (401 "sign_in" when signing in helps).
    bool admin_allowed(const httplib::Request& request, httplib::Response& response);
    // The checks against other web pages (Origin) and DNS rebinding (Host); writes a 403 when given
    // a response.
    bool page_checks(const httplib::Request& request, httplib::Response* response, bool has_key) const;
    // The admin key a request presents or is signed in with.
    std::optional<ApiKey> admin_key(const httplib::Request& request);
    Json session_json(const httplib::Request& request);
    // A request may use the models: no keys exist, it sent one (named in `call`), or it comes from
    // the launcher's own page. Otherwise the error is written.
    bool client_authorized(const httplib::Request& request, httplib::Response& response, bool anthropic, ApiCall& call);
    // The launcher's own page: same origin, and a Host the management API would accept.
    [[nodiscard]] bool from_own_page(const httplib::Request& request);
    // Where a request naming `name` goes: pinned model, a route, a model id, or the unknown-name
    // policy; `routed_by` says which.
    [[nodiscard]] std::optional<std::string> route_request(const std::string& name, std::string& routed_by) const;
    [[nodiscard]] std::optional<std::filesystem::path> prefix_cache_file(const std::string& config_id, const Args& args) const;
    std::function<void(const ProxyOutcome&)> exchange_finished(const httplib::Request& request, ApiCall& call,
                                                               std::function<void(const ProxyOutcome&)> finished);
    void proxy_after_load(const httplib::Request& request, httplib::Response& response, AcquireLease acquire,
                          StreamDialect dialect, const std::string& loading_note, ApiCall& call,
                          std::function<void(const ProxyOutcome&)> finished);
    void handle_generation(const httplib::Request& request, httplib::Response& response, ApiCall& call);
    void handle_stateful(const httplib::Request& request, httplib::Response& response, ApiCall& call);
    void handle_models(const httplib::Request& request, httplib::Response& response, ApiCall& call);
    // llama-server's router API (Settings.llamacpp_api): /props, /models/load, /models/unload and
    // /models/sse, answered from the launcher's own state; generation stays on the /v1 routes.
    void handle_props(const httplib::Request& request, httplib::Response& response, ApiCall& call);
    void handle_models_load(const httplib::Request& request, httplib::Response& response, ApiCall& call);
    void handle_models_unload(const httplib::Request& request, httplib::Response& response, ApiCall& call);
    void handle_models_sse(const httplib::Request& request, httplib::Response& response, ApiCall& call);
    // A configuration as llama-server's GET /props describes its loaded model, without loading it.
    // Not under the library lock: it may run --help.
    [[nodiscard]] Json props_json(const std::string& config_id, const ScannedModel& model, const Args& args,
                                  const std::string& alias) const;
    // A /models/sse event: `model_status` (loading, loaded, unloaded with a failed load's exit
    // code) or, for the model "*", `models_reload` when the list changes.
    void publish_model_event(const std::string& model, const std::string& event, Json data = nullptr);
    void proxy(const httplib::Request& request, httplib::Response& response, std::shared_ptr<Router::Lease> lease,
               ApiCall& call, std::function<void(const ProxyOutcome&)> finished);
    // `claude_code`: the id is Claude Code's spelling of the model ("<model id>:claude-code") unless
    // the model id already passes its filter.
    [[nodiscard]] Json model_entry(const std::string& config_id, const ScannedModel& model, const LibraryScan& scan,
                                   bool claude_code = false) const;

    // Proxy activity.
    std::uint64_t activity_begin(const httplib::Request& request, const ApiCall& call);
    // The server's metrics: running requests' progress and finished requests' usage, matched to
    // activity records by their labels.
    void absorb_metrics(const Json& metrics);
    void activity_update(std::uint64_t id, const std::string& state, int status, const std::optional<std::string>& error);
    [[nodiscard]] Json activity_json() const;

    // Events.
    void publish(const std::string& name, const Json& data);
    // Publishes the new log and event records to the web UI and writes the server's log file, every
    // 150 ms until stop() has ended everything that logs.
    void flush_logs_loop();
    // A line for the launcher's console, printed in order by console_loop(): a console that does not
    // take output (a QuickEdit selection holds it) delays only that thread.
    void print_console(std::string line);
    void console_loop();
    void toast(const std::string& level, const std::string& message);
    [[nodiscard]] Json state_json() const;
    [[nodiscard]] Json hf_files_json(const std::string& repo) const;
    void download_finished(const std::filesystem::path& dest_dir, const std::filesystem::path& target_dir);

    static constexpr std::size_t kRestartLimit = 3;
    static constexpr std::chrono::minutes kRestartWindow{15};
    // Request and response bodies kept per request when Settings::log_bodies is on.
    static constexpr std::size_t kBodyCaptureLimit = 16u << 20;

    LauncherOptions options_;
    Store store_;
    EventLog event_log_;
    LogBuffer logs_;
    std::filesystem::path applied_log_dir_;
    EventHub events_;
    EventHub model_events_; // llama.cpp's /models/sse
    Json catalog_;

    mutable std::mutex library_mutex_;
    LibraryScan library_;
    bool scanning_ = false;
    std::string library_digest_;
    std::condition_variable scan_cv_;
    bool scan_requested_ = true;

    std::unique_ptr<Supervisor> supervisor_;
    std::unique_ptr<Router> router_;
    std::unique_ptr<Telemetry> telemetry_;
    std::unique_ptr<Downloads> downloads_;

    mutable std::mutex activity_mutex_;
    std::deque<ProxyRecord> activity_;
    std::uint64_t next_activity_ = 1;
    // The server's request rows already counted in the usage book ("<instance>:<id>").
    std::set<std::string> counted_rows_;
    std::deque<std::string> counted_order_;

    std::mutex pending_logs_mutex_;
    std::vector<LogRecord> pending_logs_;
    std::vector<EventRecord> pending_events_;
    std::atomic<bool> stop_flushing_{false};

    std::mutex console_mutex_;
    std::condition_variable console_cv_;
    std::vector<std::string> console_lines_;
    bool console_stopping_ = false;

    // The instance transitions already reported, the latest incident and the recent restarts.
    mutable std::mutex incident_mutex_;
    std::uint32_t seen_instance_ = 0;
    InstanceState seen_state_    = InstanceState::Idle;
    Json last_incident_          = nullptr;
    std::deque<std::chrono::steady_clock::time_point> restarts_;
    int low_ram_samples_ = 0; // telemetry thread only

    // Sign-ins of admin keys (cookie token -> key id), until the launcher stops.
    std::mutex sessions_mutex_;
    std::map<std::string, std::string> sessions_;

    UsageBook usage_;
    ChatStore chats_;
    std::unique_ptr<ModelUpdates> model_updates_;
    std::unique_ptr<InfernixUpdate> infernix_update_;

    std::atomic<bool> restart_required_{false};
    // load_model's current start may be retried on another port (an automatic port, attempts left).
    std::atomic<bool> port_retry_allowed_{false};
    std::atomic<bool> stopping_{false};
    std::mutex stop_mutex_;
    bool stopped_ = false;
    std::thread scanner_;
    std::thread log_flusher_;
    std::thread console_writer_;
    httplib::Server server_;
    std::string bound_host_;
    int bound_port_ = 0;
};

} // namespace infernix::launcher
