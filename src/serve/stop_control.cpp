#include "serve/stop_control.h"

#include <string>
#include <utility>

namespace ninfer::serve {

namespace {

const std::string kWithin =
    "Press Ctrl+C again within " + std::to_string(StopControl::kConfirmWindow.count()) + " s to ";

} // namespace

StopControl::StopControl(bool saves_prefix_cache, StopControlActions actions)
    : saves_prefix_cache_(saves_prefix_cache), actions_(std::move(actions)) {}

void StopControl::serve(std::function<void()> stop_server) {
    std::lock_guard lock(mutex_);
    stop_server_ = std::move(stop_server);
    phase_       = Phase::Serving;
    confirm_until_.reset();
}

void StopControl::end_serving() {
    std::lock_guard lock(mutex_);
    stop_server_ = nullptr;
    if (phase_ == Phase::Serving) { phase_ = Phase::Stopping; }
}

void StopControl::finish() {
    std::lock_guard lock(mutex_);
    phase_       = Phase::Inactive;
    stop_server_ = nullptr;
    confirm_until_.reset();
    actions_.show({});
}

bool StopControl::handle(StopEvent event, Clock::time_point now) {
    std::lock_guard lock(mutex_);
    if (phase_ == Phase::Inactive) { return false; }
    if (event == StopEvent::Terminate) {
        if (phase_ == Phase::Serving) { begin_stop_locked(false); }
        return true;
    }
    // The stop already asked for confirmation: one more Ctrl+C exits.
    if (phase_ == Phase::Stopping) {
        exit_locked();
        return true;
    }
    if (!confirm_until_ || now > *confirm_until_) {
        confirm_until_ = now + kConfirmWindow;
        actions_.show(prompt_line_locked());
        return true;
    }
    begin_stop_locked(true);
    return true;
}

std::optional<StopControl::Clock::time_point> StopControl::expire(Clock::time_point now) {
    std::lock_guard lock(mutex_);
    if (confirm_until_ && now > *confirm_until_) {
        confirm_until_.reset();
        actions_.show(resting_line_locked());
    }
    return confirm_until_;
}

void StopControl::begin_stop_locked(bool interrupt) {
    phase_ = Phase::Stopping;
    confirm_until_.reset();
    actions_.record(
        {.severity = OperationalSeverity::Info,
         .message =
             std::string(interrupt ? "Ctrl+C: " : "") +
             "stopping | running and queued requests are cancelled" +
             (saves_prefix_cache_ ? " | Press Ctrl+C again to exit without saving the prefix cache"
                                  : "")});
    actions_.show(resting_line_locked());
    // Only closes a socket and signals the Engine's worker; it does not wait.
    if (stop_server_) { stop_server_(); }
}

void StopControl::exit_locked() {
    std::string message = "Ctrl+C: exiting before the stop finished";
    if (saves_prefix_cache_) {
        const PrefixCacheSaveControl::Abandon result =
            actions_.abandon_save ? actions_.abandon_save()
                                  : PrefixCacheSaveControl::Abandon::Unsaved;
        switch (result) {
        case PrefixCacheSaveControl::Abandon::Unsaved:
            message = "Ctrl+C: exiting without saving the prefix cache | the previous file is kept";
            break;
        case PrefixCacheSaveControl::Abandon::Saved:
            message += " | the prefix cache was already saved";
            break;
        case PrefixCacheSaveControl::Abandon::StillWriting:
            message = "Ctrl+C: exiting without saving the prefix cache | the save did not stop in "
                      "time, so its unfinished .tmp file may remain; the previous file is kept";
            break;
        }
    }
    actions_.record({.severity = OperationalSeverity::Warning, .message = std::move(message)});
    actions_.exit_now();
}

StopConsoleLine StopControl::prompt_line_locked() const {
    return {.text   = kWithin + (saves_prefix_cache_ ? "save the prefix cache and close" : "close"),
            .prompt = true};
}

StopConsoleLine StopControl::resting_line_locked() const {
    if (phase_ != Phase::Stopping) { return {}; }
    return {.text = saves_prefix_cache_
                        ? "Closing: saving the prefix cache | Press Ctrl+C again to exit without "
                          "saving"
                        : "Closing | Press Ctrl+C again to exit at once"};
}

} // namespace ninfer::serve
