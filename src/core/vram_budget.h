#pragma once

// Physical facts about one CUDA device's video memory: device-wide free memory, this process's OS
// budget (Windows WDDM, through DXGI) and whether the device drives a display. No policy: sizing
// decisions live with their owners (docs/maintainer/qwen3_8-flash-next-design.md §19.3.7).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace ninfer {

enum class DisplayState : std::uint8_t {
    Headless, // no output attached and no display reported
    Attached, // an output is attached, or the driver reports an active display
    Unknown,  // neither source could tell; callers treat it as Attached
};

struct VramSnapshot {
    std::size_t device_free  = 0; // cudaMemGetInfo (device-wide under WDDM)
    std::size_t device_total = 0;
    // Windows WDDM: this process's local-memory budget and usage (DXGI QueryVideoMemoryInfo).
    bool has_budget            = false;
    std::uint64_t local_budget = 0;
    std::uint64_t local_usage  = 0;
    DisplayState display       = DisplayState::Unknown;
    int outputs                = -1; // DXGI outputs of the adapter; -1 unknown
};

// One device's source. query() may be called from any thread on which the device is current.
class VramBudgetSource {
public:
    virtual ~VramBudgetSource()             = default;
    [[nodiscard]] virtual VramSnapshot query() = 0;
};

// The source for `device`: DXGI and NVML where present (both loaded or created lazily; their
// absence never fails, it only leaves fields unknown), else CUDA alone.
[[nodiscard]] std::unique_ptr<VramBudgetSource> open_vram_budget_source(int device);

// A device allocation step that the driver placed partly in system memory (the Windows WDDM sysmem
// fallback) lowers device free memory by less than it allocated. The bytes of a step of `bytes`
// missing from the drop in free memory beyond `tolerance`; 0: resident.
inline constexpr std::uint64_t kSpillToleranceBytes = 32ULL << 20;
[[nodiscard]] std::uint64_t spill_shortfall(std::uint64_t free_before, std::uint64_t free_after,
                                            std::uint64_t bytes,
                                            std::uint64_t tolerance = kSpillToleranceBytes) noexcept;

// Checks device allocation steps against a source's free memory. A shortfall is confirmed by a
// second reading, since another program can change free memory during the step.
class SpillGuard {
public:
    explicit SpillGuard(VramBudgetSource& source) : source_(source) {}
    // Records free memory before a step.
    void begin() { before_ = source_.query().device_free; }
    // After a step that allocated `bytes`: the bytes not placed in device memory (0: resident).
    [[nodiscard]] std::uint64_t end(std::uint64_t bytes);

private:
    VramBudgetSource& source_;
    std::uint64_t before_ = 0;
};

namespace testing {
// Replaces every source opened afterwards with `fake` (tests of sizing and monitoring); an empty
// function restores the real sources.
void set_vram_budget_source(std::function<VramSnapshot()> fake);
} // namespace testing

} // namespace ninfer
