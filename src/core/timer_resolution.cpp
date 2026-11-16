#include "core/timer_resolution.h"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <timeapi.h>
#endif

namespace infernix {

TimerResolution::TimerResolution() noexcept {
#ifdef _WIN32
    raised_ = timeBeginPeriod(1) == TIMERR_NOERROR;
#endif
}

TimerResolution::~TimerResolution() noexcept {
#ifdef _WIN32
    if (raised_) { timeEndPeriod(1); }
#endif
}

} // namespace infernix
