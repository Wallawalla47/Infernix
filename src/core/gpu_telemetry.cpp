#include "core/gpu_telemetry.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace infernix {

std::string gpu_clock_event_reason_names(std::uint64_t reasons) {
    // nvmlClocksEventReason* bits 0x1 .. 0x100.
    static constexpr std::array<std::string_view, 9> kNames{
        "idle",        "application-clocks", "sw-power-cap", "hw-slowdown",    "sync-boost",
        "sw-thermal",  "hw-thermal",         "power-brake",  "display-clocks",
    };
    if (reasons == 0) { return "none"; }
    std::string out;
    for (std::size_t bit = 0; bit < 64; ++bit) {
        if ((reasons & (1ULL << bit)) == 0) { continue; }
        if (!out.empty()) { out += ','; }
        out += bit < kNames.size() ? std::string(kNames[bit]) : "bit" + std::to_string(bit);
    }
    return out;
}

struct GpuTelemetry::Library {
    struct ViolationTime { // nvmlViolationTime_t
        unsigned long long reference_time;
        unsigned long long violation_time;
    };
    static constexpr unsigned kClockSm = 1, kClockMemory = 2;  // nvmlClockType_t
    static constexpr unsigned kTemperatureGpu = 0;             // nvmlTemperatureSensors_t
    static constexpr unsigned kPolicyPower = 0, kPolicyThermal = 1; // nvmlPerfPolicyType_t

#ifdef _WIN32
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
    int (*init)()                                          = nullptr;
    int (*shutdown)()                                      = nullptr;
    int (*by_pci)(const char*, void**)                     = nullptr;
    int (*clock)(void*, unsigned, unsigned*)               = nullptr;
    int (*power)(void*, unsigned*)                         = nullptr;
    int (*energy)(void*, unsigned long long*)              = nullptr;
    int (*reasons)(void*, unsigned long long*)             = nullptr;
    int (*temperature)(void*, unsigned, unsigned*)         = nullptr;
    int (*violation)(void*, unsigned, ViolationTime*)      = nullptr;
    int (*link_generation)(void*, unsigned*)               = nullptr;
    int (*link_width)(void*, unsigned*)                    = nullptr;
    int (*max_link_generation)(void*, unsigned*)           = nullptr;
    int (*max_link_width)(void*, unsigned*)                = nullptr;
    void* device     = nullptr;
    bool initialized = false;

    explicit Library(int cuda_device) {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, cuda_device) != cudaSuccess) { return; }
#ifdef _WIN32
        handle = ::LoadLibraryA("nvml.dll");
        if (handle == nullptr) { return; }
        const auto symbol = [&](const char* name) { return reinterpret_cast<void*>(::GetProcAddress(handle, name)); };
#else
        handle = ::dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (handle == nullptr) { return; }
        const auto symbol = [&](const char* name) { return ::dlsym(handle, name); };
#endif
        init        = reinterpret_cast<int (*)()>(symbol("nvmlInit_v2"));
        shutdown    = reinterpret_cast<int (*)()>(symbol("nvmlShutdown"));
        by_pci      = reinterpret_cast<int (*)(const char*, void**)>(symbol("nvmlDeviceGetHandleByPciBusId_v2"));
        clock       = reinterpret_cast<int (*)(void*, unsigned, unsigned*)>(symbol("nvmlDeviceGetClockInfo"));
        power       = reinterpret_cast<int (*)(void*, unsigned*)>(symbol("nvmlDeviceGetPowerUsage"));
        energy      = reinterpret_cast<int (*)(void*, unsigned long long*)>(symbol("nvmlDeviceGetTotalEnergyConsumption"));
        temperature = reinterpret_cast<int (*)(void*, unsigned, unsigned*)>(symbol("nvmlDeviceGetTemperature"));
        violation   = reinterpret_cast<int (*)(void*, unsigned, ViolationTime*)>(symbol("nvmlDeviceGetViolationStatus"));
        using LinkQuery = int (*)(void*, unsigned*);
        link_generation =
            reinterpret_cast<LinkQuery>(symbol("nvmlDeviceGetCurrPcieLinkGeneration"));
        link_width = reinterpret_cast<LinkQuery>(symbol("nvmlDeviceGetCurrPcieLinkWidth"));
        max_link_generation =
            reinterpret_cast<LinkQuery>(symbol("nvmlDeviceGetMaxPcieLinkGeneration"));
        max_link_width = reinterpret_cast<LinkQuery>(symbol("nvmlDeviceGetMaxPcieLinkWidth"));
        // The current name, then the one older drivers export.
        reasons = reinterpret_cast<int (*)(void*, unsigned long long*)>(symbol("nvmlDeviceGetCurrentClocksEventReasons"));
        if (reasons == nullptr) {
            reasons = reinterpret_cast<int (*)(void*, unsigned long long*)>(
                symbol("nvmlDeviceGetCurrentClocksThrottleReasons"));
        }
        if (init == nullptr || shutdown == nullptr || by_pci == nullptr || init() != 0) {
            release();
            return;
        }
        initialized = true;
        char bus[32];
        std::snprintf(bus, sizeof(bus), "%08x:%02x:%02x.0", props.pciDomainID, props.pciBusID, props.pciDeviceID);
        if (by_pci(bus, &device) != 0) { device = nullptr; }
    }

    ~Library() { release(); }

    void release() noexcept {
        if (initialized) { (void)shutdown(); }
        initialized = false;
        device      = nullptr;
        if (handle == nullptr) { return; }
#ifdef _WIN32
        ::FreeLibrary(handle);
#else
        ::dlclose(handle);
#endif
        handle = nullptr;
    }
};

GpuTelemetry::GpuTelemetry(int cuda_device) : library_(std::make_unique<Library>(cuda_device)) {}

GpuTelemetry::~GpuTelemetry() = default;

GpuTelemetrySample GpuTelemetry::sample() const {
    GpuTelemetrySample out;
    const Library& l = *library_;
    if (l.device == nullptr) { return out; }
    out.valid = true;
    unsigned value = 0;
    if (l.clock != nullptr && l.clock(l.device, Library::kClockSm, &value) == 0) { out.sm_clock_mhz = value; }
    if (l.clock != nullptr && l.clock(l.device, Library::kClockMemory, &value) == 0) { out.memory_clock_mhz = value; }
    if (l.temperature != nullptr && l.temperature(l.device, Library::kTemperatureGpu, &value) == 0) {
        out.temperature_c = value;
    }
    if (l.power != nullptr && l.power(l.device, &value) == 0) { out.power_watts = static_cast<double>(value) * 1e-3; }
    unsigned long long wide = 0;
    if (l.energy != nullptr && l.energy(l.device, &wide) == 0) {
        out.has_energy         = true;
        out.energy_millijoules = wide;
    }
    if (l.reasons != nullptr && l.reasons(l.device, &wide) == 0) { out.clock_event_reasons = wide; }
    Library::ViolationTime power{}, thermal{};
    if (l.violation != nullptr && l.violation(l.device, Library::kPolicyPower, &power) == 0 &&
        l.violation(l.device, Library::kPolicyThermal, &thermal) == 0) {
        out.has_violations       = true;
        out.power_violation_ns   = power.violation_time;
        out.thermal_violation_ns = thermal.violation_time;
    }
    return out;
}

GpuPcieLink GpuTelemetry::pcie_link() const {
    GpuPcieLink out;
    const Library& l = *library_;
    if (l.device == nullptr) { return out; }
    const auto read = [&](int (*query)(void*, unsigned*), std::uint32_t& field) {
        unsigned value = 0;
        if (query != nullptr && query(l.device, &value) == 0) { field = value; }
    };
    read(l.link_generation, out.generation);
    read(l.link_width, out.width);
    read(l.max_link_generation, out.max_generation);
    read(l.max_link_width, out.max_width);
    return out;
}

} // namespace infernix
