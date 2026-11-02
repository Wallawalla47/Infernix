#include "launcher/supervisor.h"

#include "launcher/args.h"

#include <httplib.h>

#include <csignal>
#include <cstdio>

namespace infernix::launcher {
namespace {

constexpr auto kStartingPoll   = std::chrono::milliseconds(250);
constexpr auto kServingPoll    = std::chrono::milliseconds(1000);
constexpr auto kWatchdogStop   = std::chrono::seconds(15);

// The work a server has done: prompt tokens computed, tokens generated and replayed, requests ended.
std::optional<std::uint64_t> progress_counter(const Json& metrics) {
    const Json totals = metrics.value("totals", Json::object());
    if (!totals.is_object()) { return std::nullopt; }
    std::uint64_t sum = 0;
    for (const char* key : {"computed_prefill_tokens", "output_tokens", "decode_tokens", "replayed_tokens", "requests"}) {
        sum += json_or<std::uint64_t>(totals, key, 0);
    }
    return sum;
}

} // namespace

std::string exit_status_meaning(int code, bool serving) {
    // infernix-serve's own exit statuses first. Status 1 is its startup failure; a server ended while
    // serving gets it too when another program ends it (taskkill /F, Task Manager).
    switch (code) {
    case 0: return "stopped";
    case 1: return serving ? "exited unexpectedly (status 1: ended by another program, or a fatal error)" : "failed to start";
    case 2: return "stopped: the inference engine failed";
    case 130: return "stopped before its orderly stop finished";
    default: break;
    }
    char hex[16];
    std::snprintf(hex, sizeof(hex), "0x%08X", static_cast<unsigned>(code));
#ifdef _WIN32
    // Windows' loader statuses: the server never ran (the launcher's error mode keeps the system's dialog away).
    // The release zip carries every DLL but the Visual C++ runtime (candidates SV-01).
    const auto unloadable = [&](const char* problem) {
        return std::string("could not start: ") + problem + " (" + hex +
               "). Download the Windows release zip from https://github.com/Wallawalla47/Infernix/releases/latest "
               "and keep all its DLLs beside infernix-serve.exe; the Microsoft Visual C++ Redistributable (x64) "
               "supplies the rest";
    };
    switch (static_cast<unsigned>(code)) {
    case 0xC0000135u: return unloadable("a DLL it needs is missing");
    case 0xC0000139u: return unloadable("a DLL beside it is from another version");
    case 0xC000007Bu: return unloadable("a DLL it loads is damaged or not 64-bit");
    case 0xC0000142u: return unloadable("a DLL it loads failed to initialize");
    case 3: return "crashed: aborted (exit status 3)";
    case 0xC0000005u: return std::string("crashed: access violation (") + hex + ")";
    case 0xC000001Du: return std::string("crashed: illegal instruction (") + hex + ")";
    case 0xC0000094u: return std::string("crashed: integer division by zero (") + hex + ")";
    case 0xC00000FDu: return std::string("crashed: stack overflow (") + hex + ")";
    case 0xC0000374u: return std::string("crashed: heap corruption (") + hex + ")";
    case 0xC0000409u: return std::string("crashed: fail-fast or aborted (") + hex + ")";
    case 0xC0000017u: return std::string("crashed: out of memory (") + hex + ")";
    case 0xE06D7363u: return std::string("crashed: unhandled C++ exception (") + hex + ")";
    case 0xC000013Au: return std::string("was ended by Ctrl+C (") + hex + ")";
    case 0x40010004u: return std::string("was ended by another program or the system (") + hex + ")";
    default: break;
    }
    if ((static_cast<unsigned>(code) & 0xC0000000u) == 0xC0000000u) {
        return std::string("crashed (") + hex + ")";
    }
    return "exited with status " + std::to_string(code);
#else
    if (code > 128 && code < 128 + 65) {
        switch (code - 128) {
        case SIGSEGV: return "crashed: segmentation fault (SIGSEGV)";
        case SIGABRT: return "crashed: aborted (SIGABRT)";
        case SIGBUS: return "crashed: bus error (SIGBUS)";
        case SIGILL: return "crashed: illegal instruction (SIGILL)";
        case SIGFPE: return "crashed: arithmetic error (SIGFPE)";
        // systemd-oomd (on by default on Ubuntu 22.04+ and Fedora) kills by memory pressure and leaves no kernel
        // log line (Strata #4967).
        case SIGKILL:
            return "was killed (SIGKILL; under memory pressure, perhaps by the kernel's out-of-memory killer, "
                   "`journalctl -k`, or by systemd-oomd, `journalctl -u systemd-oomd`)";
        case SIGTERM: return "was terminated (SIGTERM)";
        default: return "was ended by signal " + std::to_string(code - 128);
        }
    }
    return "exited with status " + std::to_string(code);
#endif
}

const char* state_name(InstanceState state) noexcept {
    switch (state) {
    case InstanceState::Idle: return "idle";
    case InstanceState::Starting: return "starting";
    case InstanceState::Ready: return "ready";
    case InstanceState::Stopping: return "stopping";
    case InstanceState::Failed: return "failed";
    }
    return "idle";
}

Supervisor::Supervisor(LogBuffer& logs, Callbacks callbacks) : logs_(logs), callbacks_(std::move(callbacks)) {}

Supervisor::~Supervisor() { stop(std::chrono::seconds(30)); }

void Supervisor::notify() {
    cv_.notify_all();
    if (callbacks_.changed) { callbacks_.changed(); }
}

void Supervisor::start(const LaunchSpec& spec) {
    std::uint32_t instance = 0;
    {
        std::lock_guard lock(mutex_);
        if (shutting_down_) { throw std::runtime_error("the launcher is shutting down"); }
        if (state_.state == InstanceState::Starting || state_.state == InstanceState::Ready ||
            state_.state == InstanceState::Stopping) {
            throw std::logic_error("an instance is already running");
        }
        instance          = next_instance_++;
        state_            = {};
        state_.state      = InstanceState::Starting;
        state_.instance   = instance;
        state_.model_key    = spec.model_key;
        state_.artifact_key = spec.artifact_key;
        state_.config_name  = spec.config_name;
        state_.alias        = spec.alias;
        state_.name       = spec.name;
        state_.port       = spec.port;
        state_.started_at = now_ms();
        state_.phase      = "starting infernix-serve";
        load_verdict_.reset();
        state_.argv       = {path_utf8(spec.executable)};
        state_.argv.insert(state_.argv.end(), spec.args.begin(), spec.args.end());
        stop_requested_   = false;
        metrics_          = nullptr;
    }
    if (monitor_.joinable()) { monitor_.join(); }
    child_.reset();
    logs_.begin_instance(instance, spec.alias);
    logs_.add(instance, "info", "launcher", "loading " + spec.alias + " on port " + std::to_string(spec.port));
    logs_.add(instance, "debug", "launcher", command_line_for_display(state_.argv));
    try {
        child_ = ChildProcess::start(
            spec.executable, spec.args, spec.working_dir,
            [this, instance](std::string_view chunk) { logs_.append_output(instance, chunk); }, spec.env);
    } catch (const std::exception& error) {
        {
            std::lock_guard lock(mutex_);
            state_.state = InstanceState::Failed;
            state_.error = error.what();
        }
        logs_.add(instance, "error", "launcher", error.what());
        logs_.end_instance();
        notify();
        throw;
    }
    {
        std::lock_guard lock(mutex_);
        state_.pid = child_->pid();
    }
    monitor_ = std::thread([this, instance] { monitor(instance); });
    notify();
}

bool Supervisor::wait_ready(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    cv_.wait_until(lock, deadline, [this] { return state_.state != InstanceState::Starting || shutting_down_; });
    return state_.state == InstanceState::Ready && !shutting_down_;
}

void Supervisor::shutdown() {
    {
        std::lock_guard lock(mutex_);
        shutting_down_ = true;
    }
    cv_.notify_all();
}

void Supervisor::end_load(std::uint32_t instance, std::string kind, std::string message) {
    {
        std::lock_guard lock(mutex_);
        if (state_.instance != instance || state_.state != InstanceState::Starting || load_verdict_) { return; }
        load_verdict_.emplace(std::move(kind), std::move(message));
    }
    cv_.notify_all();
}

void Supervisor::stop(std::chrono::seconds timeout) {
    std::uint32_t instance = 0;
    std::string alias;
    bool was_ready = false;
    int port       = 0;
    {
        std::lock_guard lock(mutex_);
        alias     = state_.alias;
        was_ready = state_.state == InstanceState::Ready && child_ && !child_->exit_code();
        port      = state_.port;
        if (!child_ || state_.state == InstanceState::Idle || state_.state == InstanceState::Failed) {
            // Nothing running; still reap a finished monitor below.
        } else {
            stop_requested_ = true;
            state_.state    = InstanceState::Stopping;
            state_.phase    = "stopping (running requests are cancelled; the prefix cache is saved)";
        }
        instance = state_.instance;
    }
    if (was_ready && callbacks_.metrics) {
        // One last reading: requests that ended since the last poll (a switch follows the last one
        // at once) are counted from it.
        httplib::Client client("127.0.0.1", port);
        client.set_connection_timeout(std::chrono::seconds(1));
        client.set_read_timeout(std::chrono::seconds(2));
        const httplib::Headers headers = {{"Accept", "application/json"}};
        if (const auto response = client.Get("/metrics?format=json", headers); response && response->status == 200) {
            try {
                callbacks_.metrics(Json::parse(response->body));
            } catch (const Json::exception&) {}
        }
    }
    if (child_ && !child_->exit_code()) {
        notify();
        logs_.add(instance, "info", "launcher", "stopping " + alias);
        if (!child_->request_stop()) {
            logs_.add(instance, "warn", "launcher", "could not send the stop signal; ending the process");
            child_->kill();
        }
        if (!child_->wait_for(std::chrono::duration_cast<std::chrono::milliseconds>(timeout))) {
            logs_.add(instance, "warn", "launcher",
                      "infernix-serve did not stop within " + std::to_string(timeout.count()) + " s; ending it");
            child_->kill();
            child_->wait_for(std::chrono::seconds(10));
        }
    }
    if (monitor_.joinable()) { monitor_.join(); }
    child_.reset();
}

void Supervisor::monitor(std::uint32_t instance) {
    int port = 0;
    {
        std::lock_guard lock(mutex_);
        port = state_.port;
    }
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::seconds(2));
    client.set_read_timeout(std::chrono::seconds(5));
    // One kept-alive connection keeps a server worker for the polls however busy the others are.
    client.set_keep_alive(true);

    using Clock = std::chrono::steady_clock;
    // Unanswered polls in a row, and since when: a lock-up needs several (one failed poll right
    // after the computer wakes from sleep is not one).
    std::optional<Clock::time_point> unanswered_since;
    int unanswered_polls  = 0;
    bool reported_idle    = false;
    bool server_busy      = false; // requests in the server at its last answer
    Clock::time_point last_poll      = Clock::now();
    Clock::time_point progress_since = Clock::now();
    std::optional<std::uint64_t> progress;
    // The watchdog's verdict on this instance: its kind and the reason.
    std::optional<std::pair<std::string, std::string>> incident;

    for (;;) {
        InstanceState state;
        {
            std::lock_guard lock(mutex_);
            state = state_.state;
        }
        if (const std::optional<int> code = child_->exit_code()) {
            // Its last lines (why it ended: infernix-serve logs a start failure and exits at once)
            // are read after the exit is seen. Not under the lock: each line passes observe().
            child_->wait_output(std::chrono::seconds(2));
            std::string error;
            {
                std::lock_guard lock(mutex_);
                const bool expected = stop_requested_;
                state_.exit_code    = *code;
                state_.pid.reset();
                state_.usage.reset();
                if (expected) {
                    state_.state = InstanceState::Idle;
                    state_.phase = "stopped";
                } else if (incident) {
                    error           = incident->second;
                    state_.state    = InstanceState::Failed;
                    state_.error    = error;
                    state_.incident = incident->first;
                    state_.phase    = "ended by the watchdog";
                } else {
                    // The last error (or warning) record says why; else the exit status does. A start's
                    // last line says how far it got; a serving server's last line is only its last
                    // request, and an old warning is not why it ended now.
                    const bool serving = state != InstanceState::Starting;
                    std::string reason;
                    for (const LogRecord& record : logs_.recent(instance, 3, true)) {
                        if (!serving || record.t >= now_ms() - 10000) { reason = record.text; }
                    }
                    if (reason.empty() && !serving) {
                        for (const LogRecord& record : logs_.recent(instance, 1, false)) { reason = record.text; }
                    }
                    error = "infernix-serve " + exit_status_meaning(*code, serving) + (reason.empty() ? "" : ": " + reason);
                    state_.state = InstanceState::Failed;
                    state_.error = error;
                    if (state == InstanceState::Ready) { state_.incident = "crash"; }
                    state_.phase = state == InstanceState::Starting ? "failed to start" : "stopped unexpectedly";
                }
                metrics_ = nullptr;
            }
            if (!error.empty()) { logs_.add(instance, "error", "launcher", error); }
            logs_.end_instance();
            notify();
            if (callbacks_.metrics) { callbacks_.metrics(Json(nullptr)); }
            return;
        }
        if (state == InstanceState::Starting && !incident) {
            {
                std::lock_guard lock(mutex_);
                if (load_verdict_) {
                    incident = std::move(load_verdict_);
                    load_verdict_.reset();
                }
            }
            if (incident) {
                logs_.add(instance, "error", "launcher", incident->second + "; ending the load");
                {
                    std::lock_guard lock(mutex_);
                    state_.phase = "ending the load";
                }
                notify();
                child_->kill();
                child_->wait_for(std::chrono::seconds(10));
                continue;
            }
            if (auto response = client.Get("/health"); response && response->status == 200) {
                {
                    std::lock_guard lock(mutex_);
                    if (state_.state == InstanceState::Starting) {
                        state_.state    = InstanceState::Ready;
                        state_.ready_at = now_ms();
                        state_.phase    = "ready";
                    }
                }
                progress_since = Clock::now();
                logs_.add(instance, "info", "launcher", "ready: " + snapshot().alias);
                notify();
                continue;
            }
            std::unique_lock lock(mutex_);
            cv_.wait_for(lock, kStartingPoll, [this] { return stop_requested_; });
            continue;
        }
        if (state == InstanceState::Ready && !incident) {
            const WatchdogConfig watch = callbacks_.watchdog ? callbacks_.watchdog() : WatchdogConfig{};
            httplib::Headers headers   = {{"Accept", "application/json"}};
            const auto response        = client.Get("/metrics?format=json", headers);
            const Clock::time_point now = Clock::now();
            // A long gap between polls (the computer slept): nothing ran meanwhile, so the stall
            // clock starts again.
            if (now - last_poll > std::chrono::seconds(15)) { progress_since = now; }
            last_poll = now;
            if (response && response->status == 200) {
                if (unanswered_since && reported_idle) {
                    logs_.add(instance, "info", "launcher", "infernix-serve answers again");
                }
                unanswered_since.reset();
                unanswered_polls = 0;
                reported_idle    = false;
                try {
                    Json snapshot = Json::parse(response->body);
                    {
                        std::lock_guard lock(mutex_);
                        metrics_ = snapshot;
                    }
                    if (callbacks_.metrics) { callbacks_.metrics(snapshot); }
                    // A stall: requests in the server, yet nothing processed for the stall timeout. An
                    // idle server (nothing running or waiting) never stalls.
                    const Json live         = snapshot.value("live", Json::object());
                    const auto busy         = json_or<std::uint64_t>(live, "running", 0) + json_or<std::uint64_t>(live, "waiting", 0);
                    server_busy             = busy > 0;
                    const auto counter      = progress_counter(snapshot);
                    if (busy == 0 || !counter || !progress || *counter != *progress) {
                        progress       = counter;
                        progress_since = now;
                    } else if (watch.enabled && watch.stall_timeout_s > 0 &&
                               now - progress_since >= std::chrono::seconds(watch.stall_timeout_s)) {
                        incident = {"stall", "infernix-serve stopped making progress: " + std::to_string(busy) +
                                                 " request(s) in it, yet no prompt or output token processed for " +
                                                 std::to_string(watch.stall_timeout_s) + " s"};
                    }
                } catch (const Json::exception&) {}
            } else {
                if (!unanswered_since) { unanswered_since = now; }
                ++unanswered_polls;
                const bool long_enough = unanswered_polls >= 3 &&
                                         now - *unanswered_since >= std::chrono::seconds(watch.hang_timeout_s);
                const std::string why  = response ? "HTTP " + std::to_string(response->status)
                                                  : httplib::to_string(response.error());
                if (watch.enabled && long_enough && (watch.busy || server_busy)) {
                    const auto silent = std::chrono::duration_cast<std::chrono::seconds>(now - *unanswered_since).count();
                    incident = {"hang", "infernix-serve stopped responding: no answer for " + std::to_string(silent) +
                                            " s with requests waiting on it (" + why + ")"};
                } else if (watch.enabled && long_enough && !reported_idle) {
                    // Idle: left running. A request that comes and waits on it ends it then.
                    reported_idle = true;
                    const std::string notice = "infernix-serve has not answered for " +
                                               std::to_string(watch.hang_timeout_s) + " s while idle (" + why +
                                               "); it is left running, and is restarted if a request waits on it";
                    logs_.add(instance, "warn", "launcher", notice);
                    if (callbacks_.notice) { callbacks_.notice(notice); }
                }
            }
            if (incident) {
                // Asked to stop in order first (it may still save its caches), then ended.
                logs_.add(instance, "error", "launcher", incident->second + "; ending it");
                {
                    std::lock_guard lock(mutex_);
                    state_.phase = "not responding: ending it";
                }
                notify();
                if (!child_->request_stop() || !child_->wait_for(kWatchdogStop)) {
                    child_->kill();
                    child_->wait_for(std::chrono::seconds(10));
                }
                continue;
            }
            const std::optional<ProcessUsage> usage = child_->usage();
            {
                std::lock_guard lock(mutex_);
                state_.usage = usage;
            }
        }
        std::unique_lock lock(mutex_);
        cv_.wait_for(lock, kServingPoll, [this] { return stop_requested_; });
        if (stop_requested_) {
            // Wait for the process to end; the exit branch above reports it.
            lock.unlock();
            child_->wait_for(std::chrono::milliseconds(200));
        }
    }
}

void Supervisor::observe(const LogRecord& record) {
    if (record.source != "server") { return; }
    bool changed = false;
    {
        std::lock_guard lock(mutex_);
        if (record.instance != state_.instance) { return; }
        if (!serve_build_ && record.text.starts_with("build ")) { serve_build_ = record.text.substr(6); }
        if (state_.state == InstanceState::Starting && record.level == "info") {
            std::string phase = record.text;
            if (phase.starts_with("engine | ")) { phase.erase(0, 9); }
            if (phase.size() > 200) { phase = phase.substr(0, 197) + "..."; }
            state_.phase = phase;
            changed      = true;
        }
    }
    if (changed) { notify(); }
}

InstanceSnapshot Supervisor::snapshot() const {
    std::lock_guard lock(mutex_);
    return state_;
}

Json Supervisor::metrics() const {
    std::lock_guard lock(mutex_);
    return metrics_;
}

std::optional<std::string> Supervisor::serve_build() const {
    std::lock_guard lock(mutex_);
    return serve_build_;
}

} // namespace infernix::launcher
