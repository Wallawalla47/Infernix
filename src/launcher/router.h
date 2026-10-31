#pragma once

// Decides which model is loaded: requests (and the UI's Load) queue as epochs, one per run of
// requests for the same model, in arrival order. The front epoch's model is loaded and admits its
// requests; once a later epoch waits for another model, the front admits nothing new, its running
// requests finish (or are cancelled after the drain timeout), and the next model is loaded. A
// request for the model already loaded with nothing queued is admitted at once.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace infernix::launcher {

class Router {
public:
    struct Loaded {
        std::optional<std::string> error;
        int port               = 0;
        std::uint32_t instance = 0;
    };

    struct Hooks {
        // Loads `key` (stopping whatever runs), blocking until it is ready; the error otherwise.
        std::function<Loaded(const std::string& key)> load;
        // Stops the current instance (blocking).
        std::function<void()> stop;
        std::function<void()> changed;
        // Seconds: drain timeout and request wait limit; minutes the loaded `key` may sit idle
        // before it is unloaded (0: never).
        std::function<int()> drain_timeout_s;
        std::function<int()> load_timeout_s;
        std::function<int(const std::string& key)> idle_unload_min;
    };

    class Lease {
    public:
        Lease(Router& router, int port, std::uint32_t instance) : router_(router), port_(port), instance_(instance) {}
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease() { router_.release(); }
        [[nodiscard]] int port() const noexcept { return port_; }
        [[nodiscard]] std::uint32_t instance() const noexcept { return instance_; }

    private:
        Router& router_;
        int port_;
        std::uint32_t instance_;
    };

    struct Refusal {
        int status = 503;
        std::string code;
        std::string message;
        // Seconds a client should wait before trying again (a `Retry-After` header); 0 for none.
        int retry_after_s = 0;
    };

    // How long a model whose load failed refuses requests at once instead of loading again: coding
    // tools retry 5xx answers (Claude Code up to 10 times), and each retry would otherwise start
    // another load that fails the same way. Over 60 s, so Claude Code reports it instead of retrying.
    static constexpr std::chrono::seconds kFailedLoadHold{120};

    explicit Router(Hooks hooks);
    ~Router();

    // Waits until `key` is loaded and admits the request; a Refusal when its load failed, the
    // wait timed out or `cancelled` turned true.
    [[nodiscard]] std::shared_ptr<Lease> acquire(const std::string& key, const std::function<bool()>& cancelled,
                                                 Refusal& refusal);
    // Admits the request only when `key` is loaded and nothing is queued; nullptr otherwise (nothing
    // is queued either).
    [[nodiscard]] std::shared_ptr<Lease> try_acquire(const std::string& key);
    // The end of the last request (or the last admission, or the load), Unix ms; 0 before any.
    [[nodiscard]] std::int64_t last_activity_ms() const;
    // The UI's Load (and the watchdog's restart): queues `key` like a request; `force` cancels running
    // requests at once. It ends the hold of a failed load of `key`.
    void request_load(const std::string& key, bool force);
    // Ends the hold of a failed load of `key` (its configuration changed).
    void clear_hold(const std::string& key);
    // Ends every hold (the launcher's settings, the server port among them, changed).
    void clear_holds();
    // Whether `key` is held after a failed load, with the refusal its requests get meanwhile.
    [[nodiscard]] bool held(const std::string& key, Refusal& refusal);
    // Stops the loaded model now; queued requests are refused.
    void request_unload();
    // The supervisor's instance `instance` ended (stopped, failed or crashed): its model no longer
    // serves. A later instance's state is not affected.
    void instance_ended(std::uint32_t instance);

    [[nodiscard]] int in_flight() const;
    [[nodiscard]] int waiting() const;
    // The model a queued switch will load next, if any.
    [[nodiscard]] std::optional<std::string> pending_model() const;
    [[nodiscard]] std::optional<std::string> loading_model() const;
    void shutdown();

private:
    struct Epoch {
        std::uint64_t id = 0;
        std::string key;
        int waiting = 0;
        bool forced = false;
        bool failed = false;
        std::string error;
    };

    // A failed load's refusal, kept for kFailedLoadHold.
    struct Hold {
        std::chrono::steady_clock::time_point until;
        std::string error;
    };

    void release();
    void run();
    void touch_locked();
    std::shared_ptr<Epoch> push_epoch_locked(const std::string& key);
    // Whole seconds left in `key`'s hold (rounded up); 0 when none is active.
    int hold_remaining_s_locked(const std::string& key);
    bool held_locked(const std::string& key, Refusal& refusal);

    Hooks hooks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Epoch>> epochs_;
    std::map<std::string, Hold> holds_;
    std::uint64_t next_epoch_ = 1;
    std::string ready_key_;
    int ready_port_              = 0;
    std::uint32_t ready_instance_ = 0;
    std::uint32_t last_ended_     = 0; // the newest instance reported ended
    std::string loading_key_;
    int in_flight_               = 0;
    bool unload_requested_       = false;
    bool stopping_               = false;
    std::optional<std::chrono::steady_clock::time_point> drain_since_;
    std::chrono::steady_clock::time_point last_activity_ = std::chrono::steady_clock::now();
    std::int64_t last_activity_ms_                        = 0;
    std::thread worker_;
};

} // namespace infernix::launcher
