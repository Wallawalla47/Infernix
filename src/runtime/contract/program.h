#pragma once

#include "infernix/types.h"

#include <concepts>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <utility>

namespace infernix::runtime {

// What EngineCore needs of a model family's Program beyond its rounds and transactions: maintenance
// outside rounds (the expert cache's and the RAM tiers' resizes), the round thread's CPU placement, the
// prefix cache's SSD tier and periodic save, and what a finished or refused request reports. Every
// family's RuntimeTypes must satisfy it (EngineCore static_asserts it), so a Program that lacks a
// member fails to compile instead of silently losing the feature. A family without the feature
// implements the member as a no-op (an unpinned round thread, no SSD tier, no Program refusal).
template <class Types>
concept EngineProgramContract = requires(typename Types::Program& program, const typename Types::Program& view,
                                         typename Types::FinishResult& finished,
                                         typename Types::BindingReservation& reservation) {
    // Recovery from a RecoverableExecutionError (or a stall, YJ-02): the Program discards the failed
    // requests' state and keeps serving; false when it could not.
    { program.recover_after_failure() } -> std::same_as<bool>;
    // The resize of resources outside rounds: the Program asks for maintenance through the waker and the
    // worker runs it under the execution lock, maintain() while idle and maintain_busy() between rounds
    // while requests run (the RAM tiers' resize, which must not wait for an idle moment).
    program.maintain();
    program.maintain_busy();
    program.set_maintenance_waker(std::function<void()>{});
    // The logical CPUs the round thread runs on (0: unpinned).
    { view.engine_thread_cpus() } -> std::convertible_to<std::uint64_t>;
    // The SSD tier: landed transfers applied and write-back continued (true when a promotion landed), a
    // source waiting for one, write-back due while idle, and the grid an idle-time pass resumes on.
    { program.hybrid_poll_storage() } -> std::same_as<bool>;
    { view.hybrid_source_pending() } -> std::same_as<bool>;
    { view.hybrid_storage_due() } -> std::same_as<bool>;
    { view.hybrid_resume_grid() } -> std::convertible_to<std::uint32_t>;
    // The periodic prefix-cache save (empty when there was nothing to save).
    program.save_prefix_cache_now(CancellationView{});
    // The expert cache's cumulative counters (RuntimeStats::expert_cache; empty without one).
    { view.expert_cache_totals() } -> std::convertible_to<std::optional<ExpertCacheTotals>>;
    // A finished request's expert cache counters and the prompt tokens its snapshots made reusable.
    { finished.expert_cache } -> std::convertible_to<std::optional<ExpertCacheStats>>;
    { finished.cache_created_prompt_tokens } -> std::convertible_to<std::optional<std::uint32_t>>;
    // A request no state of the Engine admits fails at once with the Program's own error.
    { reservation.refusal } -> std::convertible_to<std::exception_ptr>;
};

} // namespace infernix::runtime
