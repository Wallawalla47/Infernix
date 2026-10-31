#include "launcher/telemetry.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace infernix::launcher {
namespace {

// ---- NVML, resolved at run time --------------------------------------------------------------

using NvmlDevice = void*;
struct NvmlMemory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};
struct NvmlUtilization {
    unsigned int gpu;
    unsigned int memory;
};

constexpr int kNvmlSuccess        = 0;
constexpr int kNvmlTemperatureGpu = 0;
constexpr int kNvmlClockSm        = 1;
constexpr int kNvmlClockMem       = 2;
constexpr int kNvmlPcieTx         = 0;
constexpr int kNvmlPcieRx         = 1;

struct Nvml {
    void* library = nullptr;
    int (*init)()                                         = nullptr;
    int (*shutdown)()                                     = nullptr;
    int (*count)(unsigned int*)                           = nullptr;
    int (*handle)(unsigned int, NvmlDevice*)              = nullptr;
    int (*name)(NvmlDevice, char*, unsigned int)          = nullptr;
    int (*utilization)(NvmlDevice, NvmlUtilization*)      = nullptr;
    int (*memory)(NvmlDevice, NvmlMemory*)                = nullptr;
    int (*temperature)(NvmlDevice, int, unsigned int*)    = nullptr;
    int (*power)(NvmlDevice, unsigned int*)               = nullptr;
    int (*power_limit)(NvmlDevice, unsigned int*)         = nullptr;
    int (*clock)(NvmlDevice, int, unsigned int*)          = nullptr;
    int (*fan)(NvmlDevice, unsigned int*)                 = nullptr;
    int (*pcie_throughput)(NvmlDevice, int, unsigned int*) = nullptr;
    int (*link_generation)(NvmlDevice, unsigned int*)     = nullptr;
    int (*link_width)(NvmlDevice, unsigned int*)          = nullptr;
    int (*driver_version)(char*, unsigned int)            = nullptr;
    std::vector<NvmlDevice> devices;
    bool ready = false;

    void* symbol(const char* name) const {
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
        return dlsym(library, name);
#endif
    }

    template <class F>
    void bind(F& target, const char* name) {
        target = reinterpret_cast<F>(symbol(name));
    }

    void open() {
#ifdef _WIN32
        library = LoadLibraryW(L"nvml.dll");
        if (library == nullptr) {
            library = LoadLibraryW(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        }
#else
        library = dlopen("libnvidia-ml.so.1", RTLD_NOW);
#endif
        if (library == nullptr) { return; }
        bind(init, "nvmlInit_v2");
        bind(shutdown, "nvmlShutdown");
        bind(count, "nvmlDeviceGetCount_v2");
        bind(handle, "nvmlDeviceGetHandleByIndex_v2");
        bind(name, "nvmlDeviceGetName");
        bind(utilization, "nvmlDeviceGetUtilizationRates");
        bind(memory, "nvmlDeviceGetMemoryInfo");
        bind(temperature, "nvmlDeviceGetTemperature");
        bind(power, "nvmlDeviceGetPowerUsage");
        bind(power_limit, "nvmlDeviceGetEnforcedPowerLimit");
        bind(clock, "nvmlDeviceGetClockInfo");
        bind(fan, "nvmlDeviceGetFanSpeed");
        bind(pcie_throughput, "nvmlDeviceGetPcieThroughput");
        bind(link_generation, "nvmlDeviceGetCurrPcieLinkGeneration");
        bind(link_width, "nvmlDeviceGetCurrPcieLinkWidth");
        bind(driver_version, "nvmlSystemGetDriverVersion");
        if (init == nullptr || count == nullptr || handle == nullptr || init() != kNvmlSuccess) { return; }
        unsigned int n = 0;
        if (count(&n) != kNvmlSuccess) { return; }
        for (unsigned int i = 0; i < n; ++i) {
            NvmlDevice device = nullptr;
            if (handle(i, &device) == kNvmlSuccess) { devices.push_back(device); }
        }
        ready = true;
    }

    ~Nvml() {
        if (ready && shutdown != nullptr) { shutdown(); }
#ifdef _WIN32
        if (library != nullptr) { FreeLibrary(static_cast<HMODULE>(library)); }
#else
        if (library != nullptr) { dlclose(library); }
#endif
    }

    std::string device_name(NvmlDevice device) const {
        char buffer[128] = {};
        if (name == nullptr || name(device, buffer, sizeof(buffer)) != kNvmlSuccess) { return "NVIDIA GPU"; }
        return buffer;
    }

    GpuReading read(std::uint32_t index, NvmlDevice device) const {
        GpuReading gpu;
        gpu.index = index;
        gpu.name  = device_name(device);
        unsigned int value = 0;
        NvmlUtilization util{};
        if (utilization != nullptr && utilization(device, &util) == kNvmlSuccess) {
            gpu.usage     = util.gpu;
            gpu.mem_usage = util.memory;
        }
        NvmlMemory mem{};
        if (memory != nullptr && memory(device, &mem) == kNvmlSuccess) {
            gpu.vram_total = mem.total;
            gpu.vram_used  = mem.used;
        }
        if (temperature != nullptr && temperature(device, kNvmlTemperatureGpu, &value) == kNvmlSuccess) {
            gpu.temp_c = value;
        }
        if (power != nullptr && power(device, &value) == kNvmlSuccess) { gpu.power_w = value / 1000.0; }
        if (power_limit != nullptr && power_limit(device, &value) == kNvmlSuccess) {
            gpu.power_limit_w = value / 1000.0;
        }
        if (clock != nullptr && clock(device, kNvmlClockSm, &value) == kNvmlSuccess) { gpu.sm_clock_mhz = value; }
        if (clock != nullptr && clock(device, kNvmlClockMem, &value) == kNvmlSuccess) { gpu.mem_clock_mhz = value; }
        if (fan != nullptr && fan(device, &value) == kNvmlSuccess) { gpu.fan_pct = value; }
        if (pcie_throughput != nullptr) {
            if (pcie_throughput(device, kNvmlPcieRx, &value) == kNvmlSuccess) { gpu.pcie_rx_bps = value * 1024.0; }
            if (pcie_throughput(device, kNvmlPcieTx, &value) == kNvmlSuccess) { gpu.pcie_tx_bps = value * 1024.0; }
        }
        if (link_generation != nullptr && link_generation(device, &value) == kNvmlSuccess) { gpu.pcie_gen = value; }
        if (link_width != nullptr && link_width(device, &value) == kNvmlSuccess) { gpu.pcie_width = value; }
        return gpu;
    }
};

#ifdef _WIN32

// ---- HWiNFO shared memory (Global\HWiNFO_SENS_SM2) -------------------------------------------

#pragma pack(push, 1)
struct HwinfoHeader {
    std::uint32_t signature;
    std::uint32_t version;
    std::uint32_t revision;
    std::int64_t poll_time;
    std::uint32_t sensor_offset;
    std::uint32_t sensor_size;
    std::uint32_t sensor_count;
    std::uint32_t reading_offset;
    std::uint32_t reading_size;
    std::uint32_t reading_count;
};
struct HwinfoReading {
    std::uint32_t type;
    std::uint32_t sensor_index;
    std::uint32_t reading_id;
    char label_original[128];
    char label_user[128];
    char unit[16];
    double value;
    double minimum;
    double maximum;
    double average;
};
#pragma pack(pop)

constexpr std::uint32_t kHwinfoActive   = 0x53695748; // "SiWH"
constexpr std::uint32_t kHwinfoTemperature = 1;

class Hwinfo {
public:
    ~Hwinfo() { close(); }

    std::optional<double> cpu_temperature() {
        if (view_ == nullptr && !open()) { return std::nullopt; }
        const auto* header = static_cast<const HwinfoHeader*>(view_);
        if (header->signature != kHwinfoActive || header->reading_size < sizeof(HwinfoReading)) {
            close();
            return std::nullopt;
        }
        static constexpr std::string_view kPreferred[] = {"CPU Package", "CPU (Tctl/Tdie)", "CPU Die (average)",
                                                          "Core Max", "CPU (Tdie)", "CPU (Tctl)"};
        std::optional<double> best;
        std::size_t best_rank = std::size(kPreferred);
        const auto* base      = static_cast<const char*>(view_) + header->reading_offset;
        for (std::uint32_t i = 0; i < header->reading_count; ++i) {
            const auto* reading = reinterpret_cast<const HwinfoReading*>(base + std::size_t{i} * header->reading_size);
            if (reading->type != kHwinfoTemperature) { continue; }
            const std::string_view label(reading->label_original, strnlen(reading->label_original, 128));
            for (std::size_t rank = 0; rank < best_rank; ++rank) {
                if (label == kPreferred[rank]) {
                    const std::string_view unit(reading->unit, strnlen(reading->unit, 16));
                    double celsius = reading->value;
                    if (unit.find('F') != std::string_view::npos) { celsius = (celsius - 32.0) * 5.0 / 9.0; }
                    best      = celsius;
                    best_rank = rank;
                    break;
                }
            }
        }
        return best;
    }

private:
    bool open() {
        // Opening is cheap; retry at most every 10 s while HWiNFO is not sharing.
        const auto now = std::chrono::steady_clock::now();
        if (now < next_try_) { return false; }
        next_try_ = now + std::chrono::seconds(10);
        mapping_  = OpenFileMappingW(FILE_MAP_READ, FALSE, L"Global\\HWiNFO_SENS_SM2");
        if (mapping_ == nullptr) { return false; }
        view_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        if (view_ == nullptr) {
            close();
            return false;
        }
        return true;
    }

    void close() {
        if (view_ != nullptr) { UnmapViewOfFile(view_); }
        if (mapping_ != nullptr) { CloseHandle(mapping_); }
        view_    = nullptr;
        mapping_ = nullptr;
    }

    HANDLE mapping_ = nullptr;
    const void* view_ = nullptr;
    std::chrono::steady_clock::time_point next_try_{};
};

// ---- Core Temp shared memory (CoreTempMappingObjectEx / CoreTempMappingObject) -----------------

#pragma pack(push, 4)
struct CoreTempData {
    std::uint32_t load[256];
    std::uint32_t tj_max[128];
    std::uint32_t core_count;
    std::uint32_t cpu_count;
    float temperature[256];
    float vid;
    float cpu_speed;
    float fsb_speed;
    float multiplier;
    char cpu_name[100];
    std::uint8_t fahrenheit;
    std::uint8_t delta_to_tj_max;
};
#pragma pack(pop)

class CoreTemp {
public:
    ~CoreTemp() { close(); }

    std::optional<double> cpu_temperature() {
        if (view_ == nullptr && !open()) { return std::nullopt; }
        const auto* data = static_cast<const CoreTempData*>(view_);
        const std::uint32_t cores = std::min<std::uint32_t>(data->core_count * std::max<std::uint32_t>(data->cpu_count, 1), 256);
        if (data->core_count == 0 || data->core_count > 256) {
            close();
            return std::nullopt;
        }
        std::optional<double> hottest;
        for (std::uint32_t i = 0; i < cores; ++i) {
            double value = data->temperature[i];
            if (data->delta_to_tj_max != 0) { value = data->tj_max[std::min<std::uint32_t>(i / data->core_count, 127)] - value; }
            if (data->fahrenheit != 0) { value = (value - 32.0) * 5.0 / 9.0; }
            if (value > 0.0 && value < 150.0 && (!hottest || value > *hottest)) { hottest = value; }
        }
        return hottest;
    }

private:
    bool open() {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_try_) { return false; }
        next_try_ = now + std::chrono::seconds(10);
        for (const wchar_t* name : {L"CoreTempMappingObjectEx", L"CoreTempMappingObject"}) {
            mapping_ = OpenFileMappingW(FILE_MAP_READ, FALSE, name);
            if (mapping_ != nullptr) { break; }
        }
        if (mapping_ == nullptr) { return false; }
        view_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, sizeof(CoreTempData));
        if (view_ == nullptr) {
            close();
            return false;
        }
        return true;
    }

    void close() {
        if (view_ != nullptr) { UnmapViewOfFile(view_); }
        if (mapping_ != nullptr) { CloseHandle(mapping_); }
        view_    = nullptr;
        mapping_ = nullptr;
    }

    HANDLE mapping_   = nullptr;
    const void* view_ = nullptr;
    std::chrono::steady_clock::time_point next_try_{};
};

// ---- ACPI thermal zones through PDH (built into Windows; no tool, no administrator) -------------

class ThermalZones {
public:
    ThermalZones() {
        if (PdhOpenQueryW(nullptr, 0, &query_) != ERROR_SUCCESS) { return; }
        if (PdhAddEnglishCounterW(query_, L"\\Thermal Zone Information(*)\\High Precision Temperature", 0,
                                  &counter_) == ERROR_SUCCESS) {
            tenths_ = true;
        } else if (PdhAddEnglishCounterW(query_, L"\\Thermal Zone Information(*)\\Temperature", 0, &counter_) !=
                   ERROR_SUCCESS) {
            counter_ = nullptr;
            return;
        }
        PdhCollectQueryData(query_);
    }
    ~ThermalZones() {
        if (query_ != nullptr) { PdhCloseQuery(query_); }
    }

    std::optional<double> read() {
        if (counter_ == nullptr || PdhCollectQueryData(query_) != ERROR_SUCCESS) { return std::nullopt; }
        DWORD bytes = 0;
        DWORD count = 0;
        if (PdhGetFormattedCounterArrayW(counter_, PDH_FMT_DOUBLE, &bytes, &count, nullptr) != PDH_MORE_DATA) {
            return std::nullopt;
        }
        std::vector<std::byte> buffer(bytes);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
        if (PdhGetFormattedCounterArrayW(counter_, PDH_FMT_DOUBLE, &bytes, &count, items) != ERROR_SUCCESS) {
            return std::nullopt;
        }
        std::optional<double> hottest;
        for (DWORD i = 0; i < count; ++i) {
            const double kelvin = tenths_ ? items[i].FmtValue.doubleValue / 10.0 : items[i].FmtValue.doubleValue;
            const double celsius = kelvin - 273.15;
            if (celsius > -40.0 && celsius < 150.0 && (!hottest || celsius > *hottest)) { hottest = celsius; }
        }
        return hottest;
    }

private:
    PDH_HQUERY query_     = nullptr;
    PDH_HCOUNTER counter_ = nullptr;
    bool tenths_          = false;
};

std::uint64_t filetime_ticks(const FILETIME& t) {
    return (static_cast<std::uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
}

std::string wide_to_utf8(const wchar_t* text) { return path_utf8(std::filesystem::path(text)); }

#else

std::optional<std::string> read_line(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::string line;
    if (!in || !std::getline(in, line)) { return std::nullopt; }
    return trim(line);
}

std::optional<double> hwmon_cpu_temperature() {
    std::error_code error;
    std::optional<double> fallback;
    for (const auto& entry : std::filesystem::directory_iterator("/sys/class/hwmon", error)) {
        const std::optional<std::string> name = read_line(entry.path() / "name");
        if (!name || (*name != "coretemp" && *name != "k10temp" && *name != "zenpower")) { continue; }
        for (int i = 1; i < 64; ++i) {
            const auto input = entry.path() / ("temp" + std::to_string(i) + "_input");
            const std::optional<std::string> value = read_line(input);
            if (!value) { continue; }
            const double celsius = std::stod(*value) / 1000.0;
            const std::optional<std::string> label = read_line(entry.path() / ("temp" + std::to_string(i) + "_label"));
            if (label && (*label == "Package id 0" || *label == "Tctl" || *label == "Tdie")) { return celsius; }
            if (!fallback) { fallback = celsius; }
        }
    }
    return fallback;
}

#endif

} // namespace

// ---- Sources ---------------------------------------------------------------------------------

struct Telemetry::Sources {
    Nvml nvml;
#ifdef _WIN32
    Hwinfo hwinfo;
    CoreTemp core_temp;
    std::unique_ptr<ThermalZones> zones;
    std::uint64_t idle = 0, kernel = 0, user = 0;
#else
    std::uint64_t total = 0, idle = 0;
#endif
    std::optional<double> last_server_cpu_seconds;
    std::chrono::steady_clock::time_point last_server_time{};

    double cpu_usage() {
#ifdef _WIN32
        FILETIME idle_time{}, kernel_time{}, user_time{};
        if (!GetSystemTimes(&idle_time, &kernel_time, &user_time)) { return 0.0; }
        const std::uint64_t i = filetime_ticks(idle_time), k = filetime_ticks(kernel_time), u = filetime_ticks(user_time);
        const std::uint64_t total_delta = (k - kernel) + (u - user);
        const std::uint64_t idle_delta  = i - idle;
        const bool first                = kernel == 0;
        idle = i;
        kernel = k;
        user = u;
        if (first || total_delta == 0) { return 0.0; }
        return std::clamp(100.0 * (1.0 - static_cast<double>(idle_delta) / static_cast<double>(total_delta)), 0.0, 100.0);
#else
        std::ifstream stat("/proc/stat");
        std::string cpu;
        std::uint64_t values[10] = {};
        stat >> cpu;
        std::uint64_t sum = 0;
        for (std::uint64_t& v : values) {
            stat >> v;
            sum += v;
        }
        const std::uint64_t idle_now = values[3] + values[4];
        const bool first             = total == 0;
        const std::uint64_t total_delta = sum - total, idle_delta = idle_now - idle;
        total = sum;
        idle  = idle_now;
        if (first || total_delta == 0) { return 0.0; }
        return std::clamp(100.0 * (1.0 - static_cast<double>(idle_delta) / static_cast<double>(total_delta)), 0.0, 100.0);
#endif
    }

    void memory(std::uint64_t& total_bytes, std::uint64_t& available_bytes) {
#ifdef _WIN32
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            total_bytes     = status.ullTotalPhys;
            available_bytes = status.ullAvailPhys;
        }
#else
        std::ifstream meminfo("/proc/meminfo");
        std::string key;
        std::uint64_t kib = 0;
        std::string unit;
        while (meminfo >> key >> kib >> unit) {
            if (key == "MemTotal:") { total_bytes = kib * 1024; }
            if (key == "MemAvailable:") { available_bytes = kib * 1024; }
        }
#endif
    }

    std::pair<std::optional<double>, std::optional<std::string>> cpu_temperature() {
#ifdef _WIN32
        // A sensor tool's package reading when one shares it; else the zone Windows itself reports.
        if (const std::optional<double> value = hwinfo.cpu_temperature()) { return {value, "hwinfo"}; }
        if (const std::optional<double> value = core_temp.cpu_temperature()) { return {value, "coretemp"}; }
        if (!zones) { zones = std::make_unique<ThermalZones>(); }
        if (const std::optional<double> value = zones->read()) { return {value, "thermal-zone"}; }
        return {std::nullopt, std::nullopt};
#else
        if (const std::optional<double> value = hwmon_cpu_temperature()) { return {value, "hwmon"}; }
        return {std::nullopt, std::nullopt};
#endif
    }
};

Json to_json(const SystemReading& r) {
    const auto opt = [](const std::optional<double>& v) { return v ? Json(*v) : Json(nullptr); };
    Json gpus      = Json::array();
    for (const GpuReading& g : r.gpus) {
        gpus.push_back({{"index", g.index},
                        {"name", g.name},
                        {"usage", opt(g.usage)},
                        {"mem_usage", opt(g.mem_usage)},
                        {"vram_total", g.vram_total},
                        {"vram_used", g.vram_used},
                        {"temp_c", opt(g.temp_c)},
                        {"power_w", opt(g.power_w)},
                        {"power_limit_w", opt(g.power_limit_w)},
                        {"sm_clock_mhz", opt(g.sm_clock_mhz)},
                        {"mem_clock_mhz", opt(g.mem_clock_mhz)},
                        {"fan_pct", opt(g.fan_pct)},
                        {"pcie_rx_bps", opt(g.pcie_rx_bps)},
                        {"pcie_tx_bps", opt(g.pcie_tx_bps)},
                        {"pcie_gen", opt(g.pcie_gen)},
                        {"pcie_width", opt(g.pcie_width)}});
    }
    Json server = nullptr;
    if (r.server) {
        server = {{"working_set", r.server->working_set},
                  {"private_bytes", r.server->private_bytes},
                  {"cpu", opt(r.server->cpu)}};
    }
    return {{"t", r.t},
            {"cpu",
             {{"usage", r.cpu_usage},
              {"temp_c", opt(r.cpu_temp_c)},
              {"temp_source", r.cpu_temp_source ? Json(*r.cpu_temp_source) : Json(nullptr)}}},
            {"ram",
             {{"total", r.ram_total}, {"used", r.ram_total - std::min(r.ram_total, r.ram_available)}, {"available", r.ram_available}}},
            {"gpus", gpus},
            {"server", server}};
}

Json to_json(const SystemFacts& f) {
    Json gpus = Json::array();
    for (const auto& [index, gpu] : f.gpus) {
        gpus.push_back({{"index", index}, {"name", gpu.first}, {"vram_total", gpu.second}});
    }
    return {{"hostname", f.hostname},
            {"os", f.os},
            {"cpu_name", f.cpu_name},
            {"cores", f.cores},
            {"threads", f.threads},
            {"ram_total", f.ram_total},
            {"gpus", gpus},
            {"driver_version", f.driver_version ? Json(*f.driver_version) : Json(nullptr)},
            {"cpu_temp_source", f.cpu_temp_source ? Json(*f.cpu_temp_source) : Json(nullptr)}};
}

namespace {

SystemFacts read_facts() {
    SystemFacts facts;
#ifdef _WIN32
    wchar_t host[256] = {};
    DWORD host_size   = 256;
    if (GetComputerNameExW(ComputerNameDnsHostname, host, &host_size)) { facts.hostname = wide_to_utf8(host); }
    using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
    if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
        if (auto get = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"))) {
            RTL_OSVERSIONINFOW version{};
            version.dwOSVersionInfoSize = sizeof(version);
            if (get(&version) == 0) {
                facts.os = std::string(version.dwBuildNumber >= 22000 ? "Windows 11" : "Windows 10") + " (" +
                           std::to_string(version.dwMajorVersion) + "." + std::to_string(version.dwMinorVersion) +
                           "." + std::to_string(version.dwBuildNumber) + ")";
            }
        }
    }
    wchar_t cpu[256] = {};
    DWORD cpu_size   = sizeof(cpu);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     L"ProcessorNameString", RRF_RT_REG_SZ, nullptr, cpu, &cpu_size) == ERROR_SUCCESS) {
        facts.cpu_name = trim(wide_to_utf8(cpu));
    }
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    std::vector<std::byte> buffer(length);
    if (length != 0 && GetLogicalProcessorInformationEx(
                           RelationProcessorCore,
                           reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length)) {
        for (DWORD offset = 0; offset < length;) {
            const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
            ++facts.cores;
            offset += info->Size;
        }
    }
    facts.threads = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
#else
    char host[256] = {};
    if (gethostname(host, sizeof(host)) == 0) { facts.hostname = host; }
    std::ifstream release("/etc/os-release");
    std::string line;
    while (std::getline(release, line)) {
        if (line.starts_with("PRETTY_NAME=")) {
            facts.os = line.substr(12);
            facts.os.erase(std::remove(facts.os.begin(), facts.os.end(), '"'), facts.os.end());
        }
    }
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::set<std::pair<std::string, std::string>> cores;
    std::string physical;
    while (std::getline(cpuinfo, line)) {
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) { continue; }
        const std::string key = trim(line.substr(0, colon));
        const std::string value = trim(line.substr(colon + 1));
        if (key == "model name" && facts.cpu_name.empty()) { facts.cpu_name = value; }
        if (key == "physical id") { physical = value; }
        if (key == "core id") { cores.insert({physical, value}); }
    }
    facts.threads = static_cast<std::uint32_t>(sysconf(_SC_NPROCESSORS_ONLN));
    facts.cores   = cores.empty() ? facts.threads : static_cast<std::uint32_t>(cores.size());
#endif
    return facts;
}

} // namespace

Telemetry::Telemetry(ServerProbe server, Listener listener)
    : server_(std::move(server)), listener_(std::move(listener)), sources_(std::make_unique<Sources>()) {
    facts_ = read_facts();
    sources_->nvml.open();
    for (std::size_t i = 0; i < sources_->nvml.devices.size(); ++i) {
        const GpuReading gpu = sources_->nvml.read(static_cast<std::uint32_t>(i), sources_->nvml.devices[i]);
        facts_.gpus.push_back({gpu.index, {gpu.name, gpu.vram_total}});
    }
    if (sources_->nvml.driver_version != nullptr) {
        char version[96] = {};
        if (sources_->nvml.driver_version(version, sizeof(version)) == kNvmlSuccess) { facts_.driver_version = version; }
    }
    std::uint64_t total = 0, available = 0;
    sources_->memory(total, available);
    facts_.ram_total = total;
    worker_          = std::thread([this] { run(); });
}

Telemetry::~Telemetry() { stop(); }

void Telemetry::stop() {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) { return; }
        stopping_ = true;
    }
    if (worker_.joinable()) { worker_.join(); }
}

SystemFacts Telemetry::facts() const {
    std::lock_guard lock(mutex_);
    return facts_;
}

Json Telemetry::history() const {
    std::lock_guard lock(mutex_);
    Json samples = Json::array();
    for (const SystemReading& reading : history_) { samples.push_back(to_json(reading)); }
    return {{"interval_ms", 1000}, {"samples", samples}};
}

std::optional<SystemReading> Telemetry::latest() const {
    std::lock_guard lock(mutex_);
    if (history_.empty()) { return std::nullopt; }
    return history_.back();
}

void Telemetry::run() {
    auto next = std::chrono::steady_clock::now();
    for (;;) {
        {
            std::lock_guard lock(mutex_);
            if (stopping_) { return; }
        }
        SystemReading reading;
        reading.t         = now_ms();
        reading.cpu_usage = sources_->cpu_usage();
        sources_->memory(reading.ram_total, reading.ram_available);
        const auto [temperature, source] = sources_->cpu_temperature();
        reading.cpu_temp_c      = temperature;
        reading.cpu_temp_source = source;
        for (std::size_t i = 0; i < sources_->nvml.devices.size(); ++i) {
            reading.gpus.push_back(sources_->nvml.read(static_cast<std::uint32_t>(i), sources_->nvml.devices[i]));
        }
        if (server_) {
            if (const auto probe = server_()) {
                ServerReading server = probe->first;
                const double cpu_seconds = probe->second;
                const auto now           = std::chrono::steady_clock::now();
                if (sources_->last_server_cpu_seconds && facts_.threads != 0) {
                    const double elapsed = std::chrono::duration<double>(now - sources_->last_server_time).count();
                    if (elapsed > 0.0 && cpu_seconds >= *sources_->last_server_cpu_seconds) {
                        server.cpu = std::clamp(100.0 * (cpu_seconds - *sources_->last_server_cpu_seconds) /
                                                    (elapsed * facts_.threads),
                                                0.0, 100.0);
                    }
                }
                sources_->last_server_cpu_seconds = cpu_seconds;
                sources_->last_server_time        = now;
                reading.server                    = server;
            } else {
                sources_->last_server_cpu_seconds.reset();
            }
        }
        {
            std::lock_guard lock(mutex_);
            if (reading.cpu_temp_source) { facts_.cpu_temp_source = reading.cpu_temp_source; }
            history_.push_back(reading);
            while (history_.size() > kHistory) { history_.pop_front(); }
        }
        if (listener_) { listener_(reading); }
        next += std::chrono::seconds(1);
        const auto now = std::chrono::steady_clock::now();
        if (next < now) { next = now; }
        while (std::chrono::steady_clock::now() < next) {
            {
                std::lock_guard lock(mutex_);
                if (stopping_) { return; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

} // namespace infernix::launcher
