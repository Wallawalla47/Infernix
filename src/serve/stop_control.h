#pragma once

// infernix-serve's stop policy. Ctrl+C (SIGINT) asks for a confirming second press within
// kConfirmWindow, shown on the console's transient bottom line, so one stray press never cancels
// running requests. Ctrl+Break, closing the console window and SIGTERM stop at once. The stop
// closes the listener and fails running and queued requests; the Engine then saves the prefix
// cache (--prefix-cache-file). While that runs, one more Ctrl+C abandons the save, whose
// unfinished file the Engine deletes, and exits at once.
//
// This is the state machine only: platform code delivers events with the current time, calls
// expire() when the pending deadline passes, and supplies the console and process actions.

#include "infernix/types.h"
#include "serve/operational_log.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>

namespace infernix::serve {

enum class StopEvent : std::uint8_t {
    Interrupt, // Ctrl+C / SIGINT: needs a confirming second press
    Terminate, // Ctrl+Break, closing the console window, SIGTERM
};

// The console's transient bottom line. Empty text clears it.
struct StopConsoleLine {
    std::string text;
    // A prompt waiting for a second Ctrl+C. Output without a bottom line (redirected, or not a
    // terminal) logs prompts instead, since nothing else would show them.
    bool prompt = false;
};

struct StopControlActions {
    std::function<void(const StopConsoleLine&)> show;
    std::function<void(const OperationalRecord&)> record;
    // Abandons the Engine's prefix cache save before an early exit and waits, briefly, for its
    // unfinished file to be deleted. Called only with a prefix cache file.
    std::function<PrefixCacheSaveControl::Abandon()> abandon_save;
    // Ends the process at once; production never returns from it.
    std::function<void()> exit_now;
};

class StopControl {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr std::chrono::seconds kConfirmWindow{5};

    StopControl(bool saves_prefix_cache, StopControlActions actions);

    StopControl(const StopControl&)            = delete;
    StopControl& operator=(const StopControl&) = delete;

    // Stop events reach `stop_server` from here until end_serving(). It must not block.
    void serve(std::function<void()> stop_server);
    // listen() returned. The Engine may still be failing requests and saving the prefix cache,
    // so events keep their stopping meaning, but none reaches the server again.
    void end_serving();
    // Nothing runs any more: events fall back to their default action and the line is cleared.
    void finish();

    // Returns false while inactive: the caller then applies the event's default action.
    bool handle(StopEvent event, Clock::time_point now);
    // Withdraws a Ctrl+C whose confirmation window has passed. Returns the pending deadline.
    std::optional<Clock::time_point> expire(Clock::time_point now);

private:
    enum class Phase : std::uint8_t { Inactive, Serving, Stopping };

    void begin_stop_locked(bool interrupt);
    void exit_locked();
    [[nodiscard]] StopConsoleLine prompt_line_locked() const;
    [[nodiscard]] StopConsoleLine resting_line_locked() const;

    const bool saves_prefix_cache_;
    const StopControlActions actions_;
    std::mutex mutex_;
    Phase phase_ = Phase::Inactive;
    std::function<void()> stop_server_;
    std::optional<Clock::time_point> confirm_until_;
};

} // namespace infernix::serve
