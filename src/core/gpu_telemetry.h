#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace infernix {

// One NVML reading of a device. A field NVML cannot read keeps its default (`valid` false when the
// library or the device is unavailable).
struct GpuTelemetrySample {
    bool valid                       = false;
    std::uint32_t sm_clock_mhz       = 0;
    std::uint32_t memory_clock_mhz   = 0;
    std::uint32_t temperature_c      = 0;
    double power_watts               = 0.0; // instantaneous board power
    bool has_energy                  = false;
    std::uint64_t energy_millijoules = 0; // cumulative since the driver loaded
    // nvmlClocksEventReasons bits at the reading (idle, application clocks, power cap, slowdowns...).
    std::uint64_t clock_event_reasons = 0;
    bool has_violations               = false;
    // Cumulative time the clocks were held below their target by the power and thermal policies.
    std::uint64_t power_violation_ns   = 0;
    std::uint64_t thermal_violation_ns = 0;
};

// The device's PCIe link through NVML; 0: unknown. An idle GPU lowers its current generation to
// save power, so the current link is read right after work; the maximum is what the device and the
// system allow (nvmlDeviceGetMaxPcieLink*).
struct GpuPcieLink {
    std::uint32_t generation     = 0;
    std::uint32_t width          = 0;
    std::uint32_t max_generation = 0;
    std::uint32_t max_width      = 0;
};

// The set reasons' names, comma-separated ("none" for 0), in bit order.
[[nodiscard]] std::string gpu_clock_event_reason_names(std::uint64_t reasons);

// NVML through the driver's library, loaded at runtime so its absence never stops the caller: every
// sample is then invalid. One instance per reading thread.
class GpuTelemetry {
public:
    explicit GpuTelemetry(int cuda_device);
    ~GpuTelemetry();

    GpuTelemetry(const GpuTelemetry&)            = delete;
    GpuTelemetry& operator=(const GpuTelemetry&) = delete;

    [[nodiscard]] GpuTelemetrySample sample() const;
    [[nodiscard]] GpuPcieLink pcie_link() const;

private:
    struct Library;
    std::unique_ptr<Library> library_;
};

} // namespace infernix
