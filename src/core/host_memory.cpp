#include "core/host_memory.h"

#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fstream>
#include <sstream>
#include <string>
#endif

namespace ninfer {

std::uint64_t available_host_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!::GlobalMemoryStatusEx(&status)) {
        throw std::runtime_error("GlobalMemoryStatusEx failed");
    }
    return status.ullAvailPhys;
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::istringstream fields(line.substr(13));
            std::uint64_t kib = 0;
            if (fields >> kib) { return kib * 1024; }
        }
    }
    throw std::runtime_error("/proc/meminfo has no MemAvailable");
#endif
}

} // namespace ninfer
