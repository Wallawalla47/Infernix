#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

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
    // This process's resident bytes (Windows WorkingSetSize, Linux VmRSS).
    std::uint64_t process_resident = 0;
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
    // Page-locks and maps the block for the device; once. One registration of the whole block; when
    // the driver refuses it and `boundaries` (ascending offsets inside the block where an object
    // begins) are given, slices of at most kSliceBytes that end only at those offsets, each mapped
    // on its own (Basalt #73/#82: a Windows host refused 39.5 GiB whole and pinned it in slices).
    // Throws when a slice is refused too. `slice_bytes` is for tests.
    static constexpr std::size_t kSliceBytes = std::size_t{4} << 30;
    void pin(std::span<const std::size_t> boundaries = {}, std::size_t slice_bytes = kSliceBytes);
    // Where the device reaches byte `offset` (null before pin()). Contiguous within a slice, so an
    // object that begins at one of pin()'s boundaries reads from here to its end.
    [[nodiscard]] std::byte* device_at(std::size_t offset) const noexcept;
    [[nodiscard]] std::byte* device_data() const noexcept { return device_at(0); }
    // Registrations the block took: 1 whole, more in slices, 0 before pin().
    [[nodiscard]] std::size_t registrations() const noexcept { return slices_.size(); }

private:
    struct Slice {
        std::size_t begin  = 0;
        std::size_t bytes  = 0;
        std::byte* device = nullptr;
    };
    // Registers [begin, begin + bytes) and appends it to slices_; false when the driver refuses it.
    bool register_slice(std::size_t begin, std::size_t bytes, std::string& failure);
    void unregister() noexcept;

    std::byte* data_  = nullptr;
    std::size_t size_ = 0;
    std::vector<Slice> slices_;
};

namespace testing {
// Makes RegisteredHostBuffer::pin refuse every registration larger than `bytes` (0: none), as a
// driver that will not page-lock one large block would (sliced-pinning tests).
void refuse_host_registrations_above(std::size_t bytes);
} // namespace testing

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

// A range of this process's address space reserved with no memory behind it: nothing is committed
// and any access faults. A plan-only materialization (artifact::materialize_geometry) binds weights
// at addresses inside one, so their views have the shape and offsets a load gives them while no
// byte is read, uploaded or pinned. Zero bytes reserves nothing (data() is null).
class AddressReservation {
public:
    explicit AddressReservation(std::size_t bytes);
    ~AddressReservation();
    AddressReservation(const AddressReservation&)            = delete;
    AddressReservation& operator=(const AddressReservation&) = delete;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    std::byte* data_  = nullptr;
    std::size_t size_ = 0;
};

} // namespace infernix
