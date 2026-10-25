#include "core/cpu_topology.h"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__linux__)
#    include <pthread.h>
#    include <sched.h>

#    include <fstream>
#    include <map>
#    include <sstream>
#    include <string>
#    include <utility>
#endif

#include <bit>

namespace infernix {
namespace {

#if defined(__linux__)
// A sysfs CPU list ("0-7,16,18-19") as a mask; CPUs past 63 make the list unusable (returns false).
bool parse_cpu_list(const std::string& text, std::uint64_t& out) {
    out = 0;
    std::stringstream in(text);
    std::string part;
    while (std::getline(in, part, ',')) {
        if (part.empty() || part == "\n") { continue; }
        const auto dash = part.find('-');
        const int lo    = std::stoi(part.substr(0, dash));
        const int hi    = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
        if (lo < 0 || hi > 63 || lo > hi) { return false; }
        for (int c = lo; c <= hi; ++c) { out |= std::uint64_t{1} << c; }
    }
    return true;
}

bool read_text(const std::string& path, std::string& out) {
    std::ifstream file(path);
    if (!file) { return false; }
    std::getline(file, out);
    return true;
}
#endif

} // namespace

std::vector<PhysicalCore> physical_cores() {
    std::vector<PhysicalCore> cores;
#if defined(_WIN32)
    if (GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) > 64) { return cores; }
    DWORD bytes = 0;
    (void)GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes == 0) { return cores; }
    std::vector<std::uint8_t> buffer(bytes);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bytes)) { return cores; }
    for (DWORD at = 0; at < bytes;) {
        const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + at);
        if (entry->Relationship == RelationProcessorCore && entry->Processor.GroupCount == 1 &&
            entry->Processor.GroupMask[0].Group == 0) {
            cores.push_back({static_cast<std::uint64_t>(entry->Processor.GroupMask[0].Mask), entry->Processor.EfficiencyClass});
        }
        at += entry->Size;
    }
#elif defined(__linux__)
    std::uint64_t online = 0, performance = 0;
    std::string text;
    if (!read_text("/sys/devices/system/cpu/online", text) || !parse_cpu_list(text, online)) { return cores; }
    // Hybrid Intel CPUs list their P-cores here; elsewhere every core has the same class.
    const bool hybrid = read_text("/sys/devices/cpu_core/cpus", text) && parse_cpu_list(text, performance);
    std::map<std::pair<int, int>, std::size_t> index; // (package, core id) -> cores[i]
    for (int cpu = 0; cpu < 64; ++cpu) {
        if (((online >> cpu) & 1U) == 0) { continue; }
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::string package, core;
        if (!read_text(base + "physical_package_id", package) || !read_text(base + "core_id", core)) { return {}; }
        const auto key = std::make_pair(std::stoi(package), std::stoi(core));
        const auto [it, added] = index.emplace(key, cores.size());
        if (added) {
            cores.push_back({0, static_cast<std::uint8_t>(hybrid && ((performance >> cpu) & 1U) != 0 ? 1 : 0)});
        }
        cores[it->second].cpus |= std::uint64_t{1} << cpu;
    }
#endif
    return cores;
}

int first_cpu(std::uint64_t cpus) noexcept { return cpus == 0 ? -1 : std::countr_zero(cpus); }

std::uint64_t set_current_thread_cpus(std::uint64_t cpus) noexcept {
    if (cpus == 0) { return 0; }
#if defined(_WIN32)
    return static_cast<std::uint64_t>(SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(cpus)));
#elif defined(__linux__)
    cpu_set_t before;
    CPU_ZERO(&before);
    std::uint64_t previous = 0;
    if (pthread_getaffinity_np(pthread_self(), sizeof(before), &before) == 0) {
        for (int c = 0; c < 64; ++c) {
            if (CPU_ISSET(c, &before)) { previous |= std::uint64_t{1} << c; }
        }
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 0; c < 64; ++c) {
        if (((cpus >> c) & 1U) != 0) { CPU_SET(c, &set); }
    }
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0 ? previous : 0;
#else
    return 0;
#endif
}

} // namespace infernix
