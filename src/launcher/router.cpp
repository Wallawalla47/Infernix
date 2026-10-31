#include "launcher/router.h"

#include <algorithm>

namespace infernix::launcher {

using Clock = std::chrono::steady_clock;

Router::Router(Hooks hooks) : hooks_(std::move(hooks)) {
    worker_ = std::thread([this] { run(); });
}

Router::~Router() { shutdown(); }

void Router::shutdown() {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) { return; }
        stopping_ = true;
        for (const auto& epoch : epochs_) {
            epoch->failed = true;
            epoch->error  = "the launcher is shutting down";
        }
        epochs_.clear();
    }
    cv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

void Router::touch_locked() {
    last_activity_    = Clock::now();
    last_activity_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
}

std::int64_t Router::last_activity_ms() const {
    std::lock_guard lock(mutex_);
    return last_activity_ms_;
}

int Router::hold_remaining_s_locked(const std::string& key) {
    const auto hold = holds_.find(key);
    if (hold == holds_.end()) { return 0; }
    const auto left = hold->second.until - Clock::now();
    if (left <= Clock::duration::zero()) {
        holds_.erase(hold);
        return 0;
    }
    return static_cast<int>(std::chrono::ceil<std::chrono::seconds>(left).count());
}

bool Router::held_locked(const std::string& key, Refusal& refusal) {
    const int left = hold_remaining_s_locked(key);
    if (left == 0 || ready_key_ == key) { return false; }
    refusal = {503, "model_load_failed",
               holds_[key].error + " (the launcher loads this model again after " + std::to_string(left) +
                   " s, or at once when it is loaded from the web UI or its configuration changes)",
               left};
    return true;
}

bool Router::held(const std::string& key, Refusal& refusal) {
    std::lock_guard lock(mutex_);
    return held_locked(key, refusal);
}

void Router::clear_hold(const std::string& key) {
    std::lock_guard lock(mutex_);
    holds_.erase(key);
}

void Router::clear_holds() {
    std::lock_guard lock(mutex_);
    holds_.clear();
}

std::shared_ptr<Router::Lease> Router::try_acquire(const std::string& key) {
    std::lock_guard lock(mutex_);
    if (stopping_ || !epochs_.empty() || unload_requested_ || ready_key_ != key) { return nullptr; }
    ++in_flight_;
    touch_locked();
    return std::make_shared<Lease>(*this, ready_port_, ready_instance_);
}

std::shared_ptr<Router::Epoch> Router::push_epoch_locked(const std::string& key) {
    auto epoch = std::make_shared<Epoch>();
    epoch->id  = next_epoch_++;
    epoch->key = key;
    epochs_.push_back(epoch);
    return epoch;
}

std::shared_ptr<Router::Lease> Router::acquire(const std::string& key, const std::function<bool()>& cancelled,
                                               Refusal& refusal) {
    std::unique_lock lock(mutex_);
    if (stopping_) {
        refusal = {503, "shutting_down", "the launcher is shutting down"};
        return nullptr;
    }
    if (epochs_.empty() && !unload_requested_ && ready_key_ == key) {
        ++in_flight_;
        touch_locked();
        return std::make_shared<Lease>(*this, ready_port_, ready_instance_);
    }
    if (held_locked(key, refusal)) { return nullptr; }
    std::shared_ptr<Epoch> epoch =
        !epochs_.empty() && epochs_.back()->key == key ? epochs_.back() : push_epoch_locked(key);
    ++epoch->waiting;
    lock.unlock();
    cv_.notify_all();
    if (hooks_.changed) { hooks_.changed(); }
    lock.lock();

    const auto deadline =
        Clock::now() + std::chrono::seconds(hooks_.load_timeout_s() + hooks_.drain_timeout_s() + 30);
    std::shared_ptr<Lease> lease;
    for (;;) {
        if (epoch->failed) {
            refusal = {503, "model_load_failed", epoch->error, hold_remaining_s_locked(key)};
            break;
        }
        // A pending unload (a restart's first half) must not admit into the server it is stopping.
        if (!epochs_.empty() && epochs_.front() == epoch && !unload_requested_ && ready_key_ == key) {
            ++in_flight_;
            touch_locked();
            lease          = std::make_shared<Lease>(*this, ready_port_, ready_instance_);
            break;
        }
        if (cancelled && cancelled()) {
            refusal = {499, "client_closed", "the client closed the connection while the model was loading"};
            break;
        }
        if (Clock::now() >= deadline) {
            // The load goes on; a retry in a few seconds waits for it again.
            refusal = {503, "model_load_timeout", "timed out waiting for the model to load", 10};
            break;
        }
        cv_.wait_for(lock, std::chrono::milliseconds(500));
    }
    --epoch->waiting;
    lock.unlock();
    cv_.notify_all();
    if (hooks_.changed) { hooks_.changed(); }
    return lease;
}

void Router::release() {
    {
        std::lock_guard lock(mutex_);
        --in_flight_;
        touch_locked();
    }
    cv_.notify_all();
    if (hooks_.changed) { hooks_.changed(); }
}

void Router::request_load(const std::string& key, bool force) {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) { return; }
        holds_.erase(key);
        if (epochs_.empty() && !unload_requested_ && ready_key_ == key) { return; }
        std::shared_ptr<Epoch> epoch =
            !epochs_.empty() && epochs_.back()->key == key ? epochs_.back() : push_epoch_locked(key);
        epoch->forced = epoch->forced || force;
        if (force) {
            // Everything queued before it gives way too.
            for (const auto& other : epochs_) {
                if (other != epoch) {
                    other->failed = true;
                    other->error  = "another model was loaded from the launcher";
                }
            }
            epochs_.clear();
            epochs_.push_back(epoch);
        }
    }
    cv_.notify_all();
    if (hooks_.changed) { hooks_.changed(); }
}

void Router::request_unload() {
    {
        std::lock_guard lock(mutex_);
        for (const auto& epoch : epochs_) {
            epoch->failed = true;
            epoch->error  = "the model was unloaded from the launcher";
        }
        epochs_.clear();
        unload_requested_ = true;
    }
    cv_.notify_all();
    if (hooks_.changed) { hooks_.changed(); }
}

void Router::instance_ended(std::uint32_t instance) {
    {
        std::lock_guard lock(mutex_);
        // An instance can end between becoming ready and its load returning here; run() checks.
        last_ended_ = std::max(last_ended_, instance);
        if (instance != ready_instance_) { return; }
        ready_key_.clear();
        ready_port_     = 0;
        ready_instance_ = 0;
    }
    cv_.notify_all();
}

int Router::in_flight() const {
    std::lock_guard lock(mutex_);
    return in_flight_;
}

int Router::waiting() const {
    std::lock_guard lock(mutex_);
    int total = 0;
    for (const auto& epoch : epochs_) { total += epoch->waiting; }
    return total;
}

std::optional<std::string> Router::pending_model() const {
    std::lock_guard lock(mutex_);
    for (const auto& epoch : epochs_) {
        if (epoch->key != ready_key_) { return epoch->key; }
    }
    return std::nullopt;
}

std::optional<std::string> Router::loading_model() const {
    std::lock_guard lock(mutex_);
    if (loading_key_.empty()) { return std::nullopt; }
    return loading_key_;
}

void Router::run() {
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        cv_.wait_for(lock, std::chrono::milliseconds(250));
        if (stopping_) { break; }

        if (unload_requested_) {
            unload_requested_ = false;
            drain_since_.reset();
            ready_key_.clear();
            ready_instance_ = 0;
            lock.unlock();
            hooks_.stop();
            lock.lock();
            continue;
        }

        if (epochs_.empty()) {
            if (ready_key_.empty() || in_flight_ > 0) { continue; }
            // The loaded model's idle limit (its configuration's, else the global one), asked
            // without the lock held.
            const std::string key = ready_key_;
            lock.unlock();
            const int idle_min = hooks_.idle_unload_min ? hooks_.idle_unload_min(key) : 0;
            lock.lock();
            if (idle_min > 0 && ready_key_ == key && epochs_.empty() && in_flight_ == 0 &&
                Clock::now() - last_activity_ > std::chrono::minutes(idle_min)) {
                ready_key_.clear();
                ready_instance_ = 0;
                lock.unlock();
                hooks_.stop();
                lock.lock();
            }
            continue;
        }

        const std::shared_ptr<Epoch> front = epochs_.front();
        if (front->failed) {
            epochs_.pop_front();
            cv_.notify_all();
            continue;
        }
        if (ready_key_ == front->key) {
            drain_since_.reset();
            // Its requests admit themselves; once none waits, the epoch is done.
            if (front->waiting == 0) {
                epochs_.pop_front();
                lock.unlock();
                cv_.notify_all();
                if (hooks_.changed) { hooks_.changed(); }
                lock.lock();
            }
            continue;
        }

        // Another model must be loaded: drain the running requests first.
        if (in_flight_ > 0 && !front->forced) {
            if (!drain_since_) { drain_since_ = Clock::now(); }
            if (Clock::now() - *drain_since_ < std::chrono::seconds(hooks_.drain_timeout_s())) { continue; }
        }
        drain_since_.reset();
        const std::string key = front->key;
        loading_key_          = key;
        ready_key_.clear();
        ready_instance_ = 0;
        lock.unlock();
        if (hooks_.changed) { hooks_.changed(); }
        Loaded loaded = hooks_.load(key);
        lock.lock();
        loading_key_.clear();
        if (!loaded.error && loaded.instance <= last_ended_) {
            loaded.error = "infernix-serve stopped as soon as it became ready (see the console)";
        }
        if (loaded.error) {
            front->failed = true;
            front->error  = *loaded.error;
            holds_[key]   = {Clock::now() + kFailedLoadHold, *loaded.error};
            if (!epochs_.empty() && epochs_.front() == front) { epochs_.pop_front(); }
        } else {
            ready_key_      = key;
            ready_port_     = loaded.port;
            ready_instance_ = loaded.instance;
            touch_locked(); // the idle clock starts at the load
        }
        lock.unlock();
        cv_.notify_all();
        if (hooks_.changed) { hooks_.changed(); }
        lock.lock();
    }
}

} // namespace infernix::launcher
