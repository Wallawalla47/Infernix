#pragma once

// WSL2 (Linux under Windows GPU paravirtualization, /dev/dxg): the Windows adapter's shared system
// memory and this process's use of it, read through DXCore (libdxcore.so, loaded at runtime). The
// WDDM page-locking limit (VramSnapshot::shared_system) holds there as on Windows, which reads the
// same facts through DXGI (vram_budget.cpp). Not built on Windows.

#include <cstdint>
#include <memory>

namespace infernix {

class DxcoreAdapter {
public:
    // The adapter whose LUID is `luid` (8 bytes; nullptr when unknown), else the only hardware
    // NVIDIA adapter. Null without /dev/dxg or libdxcore, or without a unique match; a failure
    // never throws, it only leaves the facts unknown.
    [[nodiscard]] static std::unique_ptr<DxcoreAdapter> open(const unsigned char* luid);

    ~DxcoreAdapter();
    DxcoreAdapter(const DxcoreAdapter&)            = delete;
    DxcoreAdapter& operator=(const DxcoreAdapter&) = delete;

    // DXCoreAdapterProperty::SharedSystemMemory (never 0 on an opened adapter).
    [[nodiscard]] std::uint64_t shared_system() const noexcept { return shared_; }
    // This process's non-local (system memory) usage, which counts every page-locked byte; false
    // when the query fails.
    [[nodiscard]] bool nonlocal_usage(std::uint64_t& usage) const;

private:
    DxcoreAdapter() = default;

    void* library_ = nullptr;
    void* adapter_ = nullptr; // IDXCoreAdapter
    std::uint64_t shared_ = 0;
};

} // namespace infernix
