#pragma once

// The dashboard's system readings, sampled once a second into a 10-minute history: CPU usage and
// temperature, RAM, and each NVIDIA GPU through NVML (part of every NVIDIA driver, loaded at run
// time, so its absence only hides the GPU tiles). Everything works without extra software. CPU
// temperature on Windows: the CPU package sensor from HWiNFO or Core Temp shared memory when one of
// them happens to run, else the ACPI thermal zone Windows itself reports (no tool or administrator
// needed; a motherboard zone, not the package); Linux reads the kernel's hwmon (coretemp, k10temp).
// A machine with no readable sensor reports none.

#include "launcher/util.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace infernix::launcher {

struct GpuReading {
    std::uint32_t index = 0; // NVML's
    std::string name;
    std::optional<double> usage;
    std::optional<double> mem_usage;
    std::uint64_t vram_total = 0;
    std::uint64_t vram_used  = 0;
    std::optional<double> temp_c;
    std::optional<double> power_w;
    std::optional<double> power_limit_w;
    std::optional<double> sm_clock_mhz;
    std::optional<double> mem_clock_mhz;
    std::optional<double> fan_pct;
    std::optional<double> pcie_rx_bps;
    std::optional<double> pcie_tx_bps;
    std::optional<double> pcie_gen;
    std::optional<double> pcie_width;
};

struct ServerReading {
    std::uint64_t working_set   = 0;
    std::uint64_t private_bytes = 0;
    std::optional<double> cpu;
};

struct SystemReading {
    std::int64_t t = 0;
    double cpu_usage = 0.0;
    std::optional<double> cpu_temp_c;
    std::optional<std::string> cpu_temp_source;
    std::uint64_t ram_total     = 0;
    std::uint64_t ram_available = 0;
    std::vector<GpuReading> gpus;
    std::optional<ServerReading> server;
};

[[nodiscard]] Json to_json(const SystemReading& reading);

// The order the launcher lists GPUs in (readings, facts, the dashboard's gauges, default launch
// parameters, load advice): the inference GPU first, then the others in NVML order. infernix-serve
// names its device on no endpoint the launcher reads, so the inference GPU is the NVIDIA device with
// the most VRAM, the first of equals in NVML order: CUDA numbers the fastest device first while NVML
// orders by PCI bus, so on a PC with a smaller second card that is infernix-serve's default
// --device 0. Returns positions in `gpus`.
[[nodiscard]] std::vector<std::size_t> gpu_listing_order(const std::vector<GpuReading>& gpus);

struct SystemFacts {
    std::string hostname;
    std::string os;
    std::string cpu_name;
    std::uint32_t cores   = 0;
    std::uint32_t threads = 0;
    std::uint64_t ram_total = 0;
    // NVML index, name and VRAM of each GPU, in gpu_listing_order (the inference GPU first).
    std::vector<std::pair<std::uint32_t, std::pair<std::string, std::uint64_t>>> gpus;
    // Whether the inference GPU drives a display (NVML); empty when NVML cannot tell.
    std::optional<bool> inference_gpu_display;
    std::optional<std::string> driver_version;
    std::optional<std::string> cpu_temp_source;
};

[[nodiscard]] Json to_json(const SystemFacts& facts);

class Telemetry {
public:
    static constexpr std::size_t kHistory = 600;
    // `server` returns the managed process's memory and CPU seconds, when one runs.
    using ServerProbe = std::function<std::optional<std::pair<ServerReading, double>>()>;
    using Listener    = std::function<void(const SystemReading&)>;

    Telemetry(ServerProbe server, Listener listener);
    ~Telemetry();

    [[nodiscard]] SystemFacts facts() const;
    [[nodiscard]] Json history() const;
    [[nodiscard]] std::optional<SystemReading> latest() const;
    void stop();

private:
    struct Sources;
    void run();

    ServerProbe server_;
    Listener listener_;
    std::unique_ptr<Sources> sources_;
    mutable std::mutex mutex_;
    std::deque<SystemReading> history_;
    SystemFacts facts_;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace infernix::launcher
