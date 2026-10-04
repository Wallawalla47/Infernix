#include "core/vram_budget.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_6.h>
#else
#include <dlfcn.h>
#endif

namespace ninfer {
namespace {

std::mutex& fake_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::function<VramSnapshot()>& fake_source() {
    static std::function<VramSnapshot()> fake;
    return fake;
}

// NVML through the driver's library, loaded at runtime so its absence never stops startup.
class Nvml {
public:
    explicit Nvml(const cudaDeviceProp& props) {
#ifdef _WIN32
        library_ = ::LoadLibraryA("nvml.dll");
        if (library_ == nullptr) { return; }
        const auto symbol = [&](const char* name) { return reinterpret_cast<void*>(::GetProcAddress(library_, name)); };
#else
        library_ = ::dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (library_ == nullptr) { return; }
        const auto symbol = [&](const char* name) { return ::dlsym(library_, name); };
#endif
        init_         = reinterpret_cast<int (*)()>(symbol("nvmlInit_v2"));
        shutdown_     = reinterpret_cast<int (*)()>(symbol("nvmlShutdown"));
        by_pci_       = reinterpret_cast<int (*)(const char*, void**)>(symbol("nvmlDeviceGetHandleByPciBusId_v2"));
        display_      = reinterpret_cast<int (*)(void*, int*)>(symbol("nvmlDeviceGetDisplayActive"));
        if (init_ == nullptr || shutdown_ == nullptr || by_pci_ == nullptr || display_ == nullptr ||
            init_() != 0) {
            release();
            return;
        }
        initialized_ = true;
        char bus[32];
        std::snprintf(bus, sizeof(bus), "%08x:%02x:%02x.0", props.pciDomainID, props.pciBusID, props.pciDeviceID);
        if (by_pci_(bus, &device_) != 0) { device_ = nullptr; }
    }

    ~Nvml() { release(); }

    Nvml(const Nvml&)            = delete;
    Nvml& operator=(const Nvml&) = delete;

    // 1 active, 0 inactive, -1 unknown.
    [[nodiscard]] int display_active() const {
        if (device_ == nullptr) { return -1; }
        int active = 0;
        return display_(device_, &active) == 0 ? (active != 0 ? 1 : 0) : -1;
    }

private:
    void release() noexcept {
        if (initialized_) { (void)shutdown_(); }
        initialized_ = false;
        device_      = nullptr;
        if (library_ == nullptr) { return; }
#ifdef _WIN32
        ::FreeLibrary(library_);
#else
        ::dlclose(library_);
#endif
        library_ = nullptr;
    }

#ifdef _WIN32
    HMODULE library_ = nullptr;
#else
    void* library_ = nullptr;
#endif
    int (*init_)()                         = nullptr;
    int (*shutdown_)()                     = nullptr;
    int (*by_pci_)(const char*, void**)    = nullptr;
    int (*display_)(void*, int*)           = nullptr;
    void* device_                          = nullptr;
    bool initialized_                      = false;
};

class DeviceSource final : public VramBudgetSource {
public:
    explicit DeviceSource(int device) : device_(device) {
        if (cudaGetDeviceProperties(&props_, device) != cudaSuccess) {
            throw std::runtime_error("VRAM budget: cannot read the device's properties");
        }
        nvml_ = std::make_unique<Nvml>(props_);
#ifdef _WIN32
        budget_event_ = ::CreateEventW(nullptr, FALSE, FALSE, nullptr);
        open_adapter();
#endif
    }

    ~DeviceSource() override {
#ifdef _WIN32
        release_adapter();
        if (budget_event_ != nullptr) { ::CloseHandle(budget_event_); }
#endif
    }

    void* change_event() override {
#ifdef _WIN32
        return budget_event_;
#else
        return nullptr;
#endif
    }

    VramSnapshot query() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        VramSnapshot out;
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            throw std::runtime_error("VRAM budget: cudaMemGetInfo failed");
        }
        out.device_free  = free_bytes;
        out.device_total = total_bytes;
#ifdef _WIN32
        // An adapter's output list is a snapshot of its factory: recreate both after a hot-plug.
        if (factory_ != nullptr && !factory_->IsCurrent()) {
            release_adapter();
            open_adapter();
        }
        if (adapter_ != nullptr) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(adapter_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
                out.has_budget   = true;
                out.local_budget = info.Budget;
                out.local_usage  = info.CurrentUsage;
            }
            out.outputs = outputs_;
        }
#endif
        const int active = nvml_->display_active();
        if (out.outputs > 0 || active == 1) {
            out.display = DisplayState::Attached;
        } else if (out.outputs == 0 && active != 1) {
            out.display = DisplayState::Headless;
        } else if (out.outputs < 0 && active == 0) {
            out.display = DisplayState::Headless;
        } else {
            out.display = DisplayState::Unknown;
        }
        return out;
    }

private:
#ifdef _WIN32
    void open_adapter() {
        IDXGIFactory4* factory = nullptr;
        if (FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) { return; }
        LUID luid{};
        std::memcpy(&luid, props_.luid, sizeof(luid));
        IDXGIAdapter3* adapter = nullptr;
        if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter)))) {
            factory->Release();
            return;
        }
        factory_ = factory;
        adapter_ = adapter;
        if (budget_event_ != nullptr &&
            FAILED(adapter_->RegisterVideoMemoryBudgetChangeNotificationEvent(budget_event_, &budget_cookie_))) {
            budget_cookie_ = 0;
        }
        outputs_ = 0;
        for (UINT i = 0;; ++i) {
            IDXGIOutput* output = nullptr;
            if (adapter_->EnumOutputs(i, &output) == DXGI_ERROR_NOT_FOUND) { break; }
            if (output != nullptr) {
                output->Release();
                ++outputs_;
            }
        }
    }

    void release_adapter() noexcept {
        if (adapter_ != nullptr && budget_cookie_ != 0) {
            adapter_->UnregisterVideoMemoryBudgetChangeNotification(budget_cookie_);
            budget_cookie_ = 0;
        }
        if (adapter_ != nullptr) { adapter_->Release(); }
        if (factory_ != nullptr) { factory_->Release(); }
        adapter_ = nullptr;
        factory_ = nullptr;
        outputs_ = -1;
    }

    IDXGIFactory4* factory_ = nullptr;
    IDXGIAdapter3* adapter_ = nullptr;
    int outputs_            = -1;
    HANDLE budget_event_    = nullptr;
    DWORD budget_cookie_    = 0;
#endif
    int device_ = 0;
    cudaDeviceProp props_{};
    std::unique_ptr<Nvml> nvml_;
    std::mutex mutex_;
};

class FakeSource final : public VramBudgetSource {
public:
    explicit FakeSource(std::function<VramSnapshot()> fake) : fake_(std::move(fake)) {}
    VramSnapshot query() override { return fake_(); }

private:
    std::function<VramSnapshot()> fake_;
};

} // namespace

std::uint64_t spill_shortfall(std::uint64_t free_before, std::uint64_t free_after, std::uint64_t bytes,
                              std::uint64_t tolerance) noexcept {
    const std::uint64_t dropped = free_before > free_after ? free_before - free_after : 0;
    const std::uint64_t missing = bytes > dropped ? bytes - dropped : 0;
    return missing > tolerance ? missing : 0;
}

std::uint64_t SpillGuard::end(std::uint64_t bytes) {
    if (spill_shortfall(before_, source_.query().device_free, bytes) == 0) { return 0; }
    return spill_shortfall(before_, source_.query().device_free, bytes);
}

std::unique_ptr<VramBudgetSource> open_vram_budget_source(int device) {
    {
        const std::lock_guard<std::mutex> lock(fake_mutex());
        if (fake_source()) { return std::make_unique<FakeSource>(fake_source()); }
    }
    return std::make_unique<DeviceSource>(device);
}

namespace testing {
void set_vram_budget_source(std::function<VramSnapshot()> fake) {
    const std::lock_guard<std::mutex> lock(fake_mutex());
    fake_source() = std::move(fake);
}
} // namespace testing

} // namespace ninfer
