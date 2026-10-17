#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace infernix {

// Physical Host memory the OS can hand out now without paging: free plus reclaimable file cache
// (Windows ullAvailPhys, Linux MemAvailable). Callers that lock large Host blocks check it first,
// since pinned pages cannot be paged out and an oversized lock pushes other processes to swap.
[[nodiscard]] std::uint64_t available_host_memory_bytes();

// One reading of the Host memory a startup ledger plans against.
struct HostMemorySnapshot {
    std::uint64_t total_physical = 0;
    // As available_host_memory_bytes, lowered on Linux to what the process's cgroup still allows.
    std::uint64_t available_physical = 0;
    // Memory the OS can still commit (Windows ullAvailPageFile: the commit limit less what is
    // committed; Linux: MemAvailable plus free swap, as Linux commits lazily).
    std::uint64_t available_commit = 0;
    // This process's private committed bytes (Windows PrivateUsage, Linux VmRSS).
    std::uint64_t process_private = 0;
};

[[nodiscard]] HostMemorySnapshot host_memory_snapshot();

// Keeps at least `bytes` of this process resident (Windows: a hard minimum working set), so a small
// RAM reserve trims other programs before Infernix's own pageable memory. Empty on success and where
// the OS has no such control (Linux); otherwise the OS's reason (non-fatal: the caller logs it).
[[nodiscard]] std::string reserve_process_working_set(std::uint64_t bytes);

// Host memory pinned after it is filled. The block is committed pageable memory: prefault() touches
// pages (the OS zeroes a page at its first touch) and pin() then page-locks the whole block and maps
// it for the device in one call (cudaHostRegister, portable and mapped). cudaMallocHost commits,
// zeroes and locks in one serial call (0.11 s/GiB on an i9-13900K, 7 s for Qwen3.8-Flash-Next's
// experts); here threads zero the pages while a reader fills them, and the registration costs about
// 0.003 s/GiB. The device reaches the block at device_data(), which differs from data() where the
// driver cannot map the host address (WDDM).
class RegisteredHostBuffer {
public:
    explicit RegisteredHostBuffer(std::size_t bytes);
    ~RegisteredHostBuffer();
    RegisteredHostBuffer(const RegisteredHostBuffer&)            = delete;
    RegisteredHostBuffer& operator=(const RegisteredHostBuffer&) = delete;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    // Writes one byte of every page of [begin, end) that nothing has written yet, so its first
    // touch (the zeroing) happens here; it must not overlap bytes already filled.
    void prefault(std::size_t begin, std::size_t end) const noexcept;
    // Page-locks and maps the block for the device; once.
    void pin();
    [[nodiscard]] std::byte* device_data() const noexcept { return device_; } // null before pin()

private:
    std::byte* data_   = nullptr;
    std::size_t size_  = 0;
    std::byte* device_ = nullptr;
};

// Host memory kept in RAM for the CPU without page-locking it for the device: on Windows the
// process's hard minimum working set grows by its size and its pages are VirtualLock'ed; on Linux
// they are mlock'ed. The device cannot read it, so it does not count toward the limit Windows puts
// on page-locked memory (the adapter's shared system memory, VramSnapshot::shared_system). Committed
// at construction and page-aligned; locking touches (zeroes) every page. When the OS refuses the
// lock, the memory still works but may be paged out under memory pressure: lock_failure() says why.
class ResidentHostBuffer {
public:
    explicit ResidentHostBuffer(std::size_t bytes);
    ~ResidentHostBuffer();
    ResidentHostBuffer(const ResidentHostBuffer&)            = delete;
    ResidentHostBuffer& operator=(const ResidentHostBuffer&) = delete;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    // Empty when every page is locked in RAM; otherwise the OS's reason.
    [[nodiscard]] const std::string& lock_failure() const noexcept { return lock_failure_; }

private:
    std::byte* data_  = nullptr;
    std::size_t size_ = 0;
    std::size_t raised_ = 0; // Windows: what the minimum working set grew by
    std::string lock_failure_;
};

} // namespace infernix
