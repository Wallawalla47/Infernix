#pragma once

#include "infernix/types.h"
#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace infernix::runtime {

// Automatic capacity takes what `available_runtime_bytes` leaves after `automatic_headroom_bytes`;
// explicit capacity ignores the headroom.
[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes,
                                                       std::size_t automatic_headroom_bytes);

// The `available_runtime_bytes` resolve_kv_capacity takes for free device memory `device_free` and, under
// an OS budget (Windows WDDM), `budget_left` (the budget less this process's usage). An explicit
// capacity must fit both: min(device_free, budget_left). An automatic one leaves the headroom of free
// device memory only and `budget_margin_bytes` of the budget, as the expert frames do
// (expert_offload::size_expert_frames), so that the capacity it resolves is
// min(device_free - headroom, budget_left - margin) and a budget below free memory is not stacked on
// the headroom: the value returned is that plus the headroom resolve_kv_capacity takes off again.
[[nodiscard]] std::size_t kv_available_bytes(bool automatic, std::size_t device_free,
                                             std::optional<std::uint64_t> budget_left,
                                             std::size_t automatic_headroom_bytes,
                                             std::size_t budget_margin_bytes) noexcept;

// The resolution of a plan whose runtime reservation grew beside `ordinary` (the same policy's
// resolution of the plan without the growth), when the growth keeps the KV capacity that
// resolution must keep: an explicit capacity whole, an automatic one down to every lane's full
// context (`lane_page_groups`) or to the ordinary capacity when that is smaller. Empty when the
// grown plan does not fit. Startup uses it to take an idle-wide-prefill width only where it fits.
[[nodiscard]] std::optional<KvCapacityResolution>
resolve_kv_capacity_beside(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                           std::size_t available_runtime_bytes,
                           std::size_t automatic_headroom_bytes,
                           const KvCapacityResolution& ordinary, std::uint64_t lane_page_groups);

// A Program's MemorySummary with the KV capacity its engine's startup resolved (Engine::memory_summary, for the
// generation and the causal-scoring engines alike).
[[nodiscard]] MemorySummary with_kv_capacity(MemorySummary summary, const KvCapacityResolution& resolution);

} // namespace infernix::runtime
