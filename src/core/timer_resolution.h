#pragma once

// While one of these exists the process's timed waits resolve to about a millisecond. On Windows a timed wait
// (std::condition_variable::wait_for, Sleep) otherwise ends at the system timer's default 15.625 ms tick: the
// Engine worker's 1 ms wait between cycles that ran nothing measured 15.7 ms (median; 2.0 ms at 1 ms resolution,
// RTX 5090 host, Windows 11, 2026-11-16), so every poll a binding or transfer waits through cost a tick
// (candidates QN-27). Since Windows 10 2004 the request is this process's only. Elsewhere it does nothing.

namespace infernix {

class TimerResolution {
public:
    TimerResolution() noexcept;
    ~TimerResolution() noexcept;
    TimerResolution(const TimerResolution&)            = delete;
    TimerResolution& operator=(const TimerResolution&) = delete;

private:
    bool raised_ = false;
};

} // namespace infernix
