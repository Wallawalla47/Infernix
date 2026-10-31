#include "core/io_priority.h"

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
#include <optional>

namespace infernix {

std::optional<IoPriorityRaise> raise_low_io_priority() noexcept {
#if defined(_WIN32)
    static std::once_flag once;
    static std::optional<IoPriorityRaise> raised;
    try {
        std::call_once(once, [] {
            // ProcessIoPriority (PROCESSINFOCLASS 33) through ntdll, which every process has loaded; a
            // process may set its own I/O priority up to normal without a privilege.
            using Query = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
            using Set   = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG);
            constexpr ULONG kProcessIoPriority = 33;
            const HMODULE ntdll                = GetModuleHandleW(L"ntdll.dll");
            if (ntdll == nullptr) { return; }
            const auto query = reinterpret_cast<Query>(GetProcAddress(ntdll, "NtQueryInformationProcess"));
            const auto set   = reinterpret_cast<Set>(GetProcAddress(ntdll, "NtSetInformationProcess"));
            ULONG priority   = 0;
            if (query == nullptr || set == nullptr ||
                query(GetCurrentProcess(), kProcessIoPriority, &priority, sizeof(priority), nullptr) < 0 ||
                priority >= static_cast<ULONG>(kIoPriorityNormal)) {
                return;
            }
            IoPriorityRaise out{.before = static_cast<int>(priority), .after = static_cast<int>(priority)};
            ULONG normal = kIoPriorityNormal;
            if (set(GetCurrentProcess(), kProcessIoPriority, &normal, sizeof(normal)) >= 0 &&
                query(GetCurrentProcess(), kProcessIoPriority, &priority, sizeof(priority), nullptr) >= 0) {
                out.after = static_cast<int>(priority);
            }
            raised = out;
        });
    } catch (...) {}
    return raised;
#else
    return std::nullopt;
#endif
}

} // namespace infernix
