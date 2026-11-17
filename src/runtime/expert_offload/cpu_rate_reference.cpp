// The CPU expert team's rate reference (placement.h: load_cpu_rate_reference, save_cpu_rate_reference,
// cpu_rate_machine_key): a settled reading an earlier start left beside the saved expert state, which lets a start
// whose own readings agree with it skip the settling pause (settle_cpu_rates).

#include "runtime/expert_offload/placement.h"

#include "core/host_memory.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#    include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#    include <cpuid.h>
#endif

namespace infernix::runtime::expert_offload {

namespace {

// The CPU's brand string (CPUID leaves 0x80000002-4), trimmed; empty where there is none.
std::string cpu_brand() {
    char brand[49] = {};
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4] = {};
    __cpuid(info, static_cast<int>(0x80000000U));
    if (static_cast<unsigned>(info[0]) < 0x80000004U) { return {}; }
    for (int i = 0; i < 3; ++i) {
        __cpuid(info, static_cast<int>(0x80000002U + static_cast<unsigned>(i)));
        std::memcpy(brand + 16 * i, info, 16);
    }
#elif defined(__x86_64__) || defined(__i386__)
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid(0x80000000U, &a, &b, &c, &d) == 0 || a < 0x80000004U) { return {}; }
    for (unsigned i = 0; i < 3; ++i) {
        unsigned regs[4] = {};
        if (__get_cpuid(0x80000002U + i, &regs[0], &regs[1], &regs[2], &regs[3]) == 0) { return {}; }
        std::memcpy(brand + 16 * i, regs, 16);
    }
#endif
    std::string out(brand);
    const auto first = out.find_first_not_of(' ');
    if (first == std::string::npos) { return {}; }
    return out.substr(first, out.find_last_not_of(' ') - first + 1);
}

std::string today() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char text[16];
    std::strftime(text, sizeof(text), "%Y-%m-%d", &local);
    return text;
}

} // namespace

std::string cpu_rate_machine_key() {
    const std::uint64_t gib = host_memory_snapshot().total_physical >> 30;
    return "cpu " + cpu_brand() + " | logical " + std::to_string(std::thread::hardware_concurrency()) + " | ram " +
           std::to_string(gib) + " GiB";
}

// The file: one line a key, "<key>\t<experts/s>\t<expert columns/s>\t<date>". Keys hold no tab or line break.
std::optional<CpuRateReference> load_cpu_rate_reference(const std::filesystem::path& path, const std::string& key) {
    try {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream fields(line);
            std::string stored, experts, columns, when;
            if (!std::getline(fields, stored, '\t') || stored != key || !std::getline(fields, experts, '\t') ||
                !std::getline(fields, columns, '\t') || !std::getline(fields, when)) {
                continue;
            }
            CpuRateReference out{.experts = std::stod(experts), .columns = std::stod(columns), .when = when};
            if (out.experts > 0.0 && out.columns > 0.0) { return out; }
        }
    } catch (...) {}
    return std::nullopt;
}

void save_cpu_rate_reference(const std::filesystem::path& path, const std::string& key, const SettledCpuRates& rates) {
    if (path.empty() || rates.readings.empty() || rates.confirmed || rates.busy_share >= 0.0 ||
        key.find_first_of("\t\n\r") != std::string::npos) {
        return;
    }
    try {
        const std::filesystem::path& file = path;
        std::vector<std::string> kept;
        {
            std::ifstream in(file);
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty() && line.rfind(key + '\t', 0) != 0) { kept.push_back(line); }
            }
        }
        char numbers[96];
        std::snprintf(numbers, sizeof(numbers), "\t%.0f\t%.0f\t", rates.experts, rates.columns);
        kept.push_back(key + numbers + today());
        std::filesystem::path temporary = file;
        temporary += ".tmp";
        {
            std::ofstream out(temporary, std::ios::trunc);
            for (const std::string& line : kept) { out << line << '\n'; }
            if (!out) { return; }
        }
        std::filesystem::rename(temporary, file);
    } catch (...) {}
}

} // namespace infernix::runtime::expert_offload
