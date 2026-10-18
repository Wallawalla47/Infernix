#include "core/host_load.h"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/resource.h>
#    include <unistd.h>

#    include <cstdio>
#endif

#include <algorithm>

namespace infernix {
namespace {

#if defined(_WIN32)
std::uint64_t filetime_ns(const FILETIME& t) noexcept {
    return ((static_cast<std::uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime) * 100ULL;
}
#endif

} // namespace

HostCpuTimes host_cpu_times() noexcept {
    HostCpuTimes out;
    out.at = std::chrono::steady_clock::now();
#if defined(_WIN32)
    FILETIME idle{}, kernel{}, user{};
    if (GetSystemTimes(&idle, &kernel, &user)) {
        // Kernel time includes the idle time.
        const std::uint64_t total = filetime_ns(kernel) + filetime_ns(user);
        const std::uint64_t idle_ns = filetime_ns(idle);
        out.busy_ns = total > idle_ns ? total - idle_ns : 0;
    }
    FILETIME created{}, exited{}, own_kernel{}, own_user{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &own_kernel, &own_user)) {
        out.own_ns = filetime_ns(own_kernel) + filetime_ns(own_user);
    }
    out.cpus = std::max<std::uint32_t>(1, GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
#else
    if (std::FILE* stat = std::fopen("/proc/stat", "r")) {
        unsigned long long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
        if (std::fscanf(stat, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &user, &nice, &system, &idle, &iowait,
                        &irq, &softirq, &steal) >= 4) {
            const long hz = sysconf(_SC_CLK_TCK);
            const std::uint64_t ticks = user + nice + system + irq + softirq + steal;
            out.busy_ns = hz > 0 ? ticks * (1000000000ULL / static_cast<std::uint64_t>(hz)) : 0;
        }
        std::fclose(stat);
    }
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        const auto ns = [](const timeval& t) {
            return static_cast<std::uint64_t>(t.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(t.tv_usec) * 1000ULL;
        };
        out.own_ns = ns(usage.ru_utime) + ns(usage.ru_stime);
    }
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    out.cpus = cpus > 0 ? static_cast<std::uint32_t>(cpus) : 1U;
#endif
    return out;
}

double other_process_cpu_share(const HostCpuTimes& begin, const HostCpuTimes& end) noexcept {
    const double wall_ns = std::chrono::duration<double, std::nano>(end.at - begin.at).count();
    if (!(wall_ns > 0.0) || end.busy_ns < begin.busy_ns) { return 0.0; }
    const double busy = static_cast<double>(end.busy_ns - begin.busy_ns);
    const double own  = end.own_ns > begin.own_ns ? static_cast<double>(end.own_ns - begin.own_ns) : 0.0;
    const double share = (busy - own) / (wall_ns * static_cast<double>(std::max<std::uint32_t>(1, end.cpus)));
    return std::clamp(share, 0.0, 1.0);
}

} // namespace infernix
