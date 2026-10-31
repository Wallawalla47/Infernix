// The engine's I/O priority raise (core/io_priority.h): a process that inherited Windows I/O priority
// Low (as one started by a Task Scheduler task does) is raised to normal and told so; elsewhere the
// call does nothing.

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

#include <cstdio>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

#if defined(_WIN32)
using Query = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
using Set   = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG);
constexpr ULONG kProcessIoPriority = 33;

int io_priority() {
    const auto query = reinterpret_cast<Query>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationProcess"));
    ULONG priority   = 0;
    return query(GetCurrentProcess(), kProcessIoPriority, &priority, sizeof(priority), nullptr) >= 0
               ? static_cast<int>(priority)
               : -1;
}

bool set_io_priority(ULONG priority) {
    const auto set = reinterpret_cast<Set>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtSetInformationProcess"));
    return set(GetCurrentProcess(), kProcessIoPriority, &priority, sizeof(priority)) >= 0;
}
#endif

} // namespace

int main() {
#if defined(_WIN32)
    // What a low-priority parent hands down: the process lowers itself (no privilege needed to lower).
    check(set_io_priority(1) && io_priority() == 1, "the test process could not take I/O priority Low");
    const auto raised = infernix::raise_low_io_priority();
    check(raised && raised->before == 1 && raised->after == infernix::kIoPriorityNormal,
          "an inherited Low I/O priority was not reported and raised to normal");
    check(io_priority() == infernix::kIoPriorityNormal, "the process does not run at normal I/O priority");
    const auto again = infernix::raise_low_io_priority();
    check(again && again->before == 1, "a second call did not return the first call's outcome");
#else
    check(!infernix::raise_low_io_priority(), "a platform without Windows I/O priorities reported a raise");
#endif
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("I/O priority checks passed\n");
    return 0;
}
