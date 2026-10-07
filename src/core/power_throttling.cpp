#include "core/power_throttling.h"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include <mutex>

namespace ninfer {

void exempt_process_from_power_throttling() noexcept {
#if defined(_WIN32)
    static std::once_flag once;
    try {
        std::call_once(once, [] {
            PROCESS_POWER_THROTTLING_STATE state{};
            state.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
            state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
            state.StateMask   = 0; // controlled and off: never throttled
            (void)SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
        });
    } catch (...) {}
#endif
}

} // namespace ninfer
