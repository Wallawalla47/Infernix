#pragma once

// One managed infernix-serve at a time: starts it, watches /health until it is ready, polls
// /metrics?format=json while it serves, records its exit, and stops it in order (CTRL_BREAK /
// SIGTERM, so it cancels its requests and saves its prefix cache and expert state) with a kill
// after the timeout.
//
// The watchdog: while the server is ready, a crash (an exit nobody asked for), a lock-up (requests in
// it or waiting for it, and no answer to the metrics poll for the hang timeout) or a stall (requests
// in it, yet no prompt or output token processed for the stall timeout) ends the instance as Failed
// with the incident's kind and reason; a locked-up or stalled server is asked to stop and ended after
// 15 s. An idle server is never ended, whatever it answers: one that stops answering while idle is
// reported once and left running (after the computer sleeps, say); a request that then waits on it
// ends it at once. Restarting is the owner's decision.

#include "launcher/log_buffer.h"
#include "launcher/process.h"
#include "launcher/util.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace infernix::launcher {

enum class InstanceState { Idle, Starting, Ready, Stopping, Failed };
[[nodiscard]] const char* state_name(InstanceState state) noexcept;

struct LaunchSpec {
    std::string model_key;    // the configuration id
    std::string artifact_key;
    std::string config_name;  // empty for the default configuration
    std::string alias;
    std::string name;
    std::filesystem::path executable;
    // argv[1..]: the artifact, the user's launch parameters and the managed flags.
    std::vector<std::string> args;
    EnvOverlay env;
    std::filesystem::path working_dir;
    int port = 0;
};

struct InstanceSnapshot {
    InstanceState state      = InstanceState::Idle;
    std::uint32_t instance   = 0;
    std::string model_key; // the configuration id
    std::string artifact_key;
    std::string config_name;
    std::string alias;
    std::string name;
    std::optional<std::uint32_t> pid;
    int port                 = 0;
    std::optional<std::int64_t> started_at;
    std::optional<std::int64_t> ready_at;
    std::string phase;
    std::vector<std::string> argv;
    std::optional<int> exit_code;
    std::optional<std::string> error;
    // Failed while ready: "crash", "hang" or "stall" (the watchdog's incident).
    std::optional<std::string> incident;
    std::optional<ProcessUsage> usage;
};

struct WatchdogConfig {
    bool enabled        = true;
    int hang_timeout_s  = 60;
    int stall_timeout_s = 300; // 0: off
    bool busy           = false; // requests are on their way to the server (the launcher's view)
};

// "access violation (0xC0000005)", "killed by SIGSEGV": what an exit status means, for a server that
// was starting or (`serving`) had started.
[[nodiscard]] std::string exit_status_meaning(int code, bool serving);

class Supervisor {
public:
    struct Callbacks {
        std::function<void()> changed;                 // state, phase or error changed
        std::function<void(const Json&)> metrics;      // a /metrics?format=json snapshot
        std::function<WatchdogConfig()> watchdog;      // read at every poll; none: defaults
        std::function<void(const std::string&)> notice; // something the event log should say
    };

    Supervisor(LogBuffer& logs, Callbacks callbacks);
    ~Supervisor();

    // Starts `spec`; the previous instance must be stopped. Throws when the process cannot start.
    void start(const LaunchSpec& spec);
    // Waits until the instance is ready (true), or failed, stopped or past `deadline` (false).
    bool wait_ready(std::chrono::steady_clock::time_point deadline);
    // Orderly stop, killed after `timeout`; returns once the process has ended. No-op when idle.
    // start/stop are called from one thread at a time (the router's, then the launcher's at exit).
    void stop(std::chrono::seconds timeout);
    // The launcher is exiting: a pending wait_ready returns false now and start refuses. Any thread.
    void shutdown();
    // Ends `instance` while it loads (the launcher's RAM floor): it fails with `kind` and `message`.
    // Any thread; ignored once that instance is ready or gone.
    void end_load(std::uint32_t instance, std::string kind, std::string message);

    [[nodiscard]] InstanceSnapshot snapshot() const;
    [[nodiscard]] Json metrics() const;
    [[nodiscard]] std::optional<std::string> serve_build() const;
    // A record of the current instance's output (startup milestones become the phase line).
    void observe(const LogRecord& record);

private:
    void monitor(std::uint32_t instance);
    void notify();

    LogBuffer& logs_;
    Callbacks callbacks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    InstanceSnapshot state_;
    std::unique_ptr<ChildProcess> child_;
    std::thread monitor_;
    bool stop_requested_ = false;
    bool stop_forced_    = false; // the stop ended the process itself (no signal, or past its timeout)
    bool shutting_down_  = false;
    std::optional<std::pair<std::string, std::string>> load_verdict_; // end_load's kind and reason
    Json metrics_;
    std::optional<std::string> serve_build_;
    std::uint32_t next_instance_ = 1;
};

} // namespace infernix::launcher
