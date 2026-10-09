#pragma once

#include "runtime/contract/request.h"
#include <cstddef>
#include <cstdint>

namespace infernix::runtime {

// Physical quantities are sampled from stores, never a scheduler-owned occupancy ledger.
struct ContextResourceUsage {
    std::uint32_t state_slots                                                   = 0;
    std::uint32_t main_kv_pages                                                 = 0;
    std::uint32_t backend_kv_pages                                              = 0;
    friend bool operator==(ContextResourceUsage, ContextResourceUsage) noexcept = default;
};

struct ResourceReservation {
    bool reserved = false;
    ContextResourceUsage shortage;

    explicit operator bool() const noexcept { return reserved; }
};

enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

// Work a binding did: StateImage copies from a cached snapshot, Host snapshot restores and
// private partial-tail page copies.
struct ContextOperationCounts {
    std::uint64_t state_forks            = 0;
    std::uint64_t state_restores         = 0;
    std::uint64_t partial_tail_cow_pages = 0;
};

struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace infernix::runtime
