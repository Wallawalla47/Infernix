#pragma once

// Host side of the pipelined groupwise sliced-K kernel (sliced_k_pipelined.cuh): which route the
// Q4/Q5/Q6 RowSplit sliced-K launches take, and the CTA shape (block rows, cp.async ring depth) of
// a pipelined launch.
//
// The routes:
// - OneWave (every launch): the pipelined kernel, whose block rows may also double, so a call whose
//   CTAs need a second wave at the schedule's rows can fit one. The column-invariant GEMVs it serves
//   ran 15-33 % faster; the reproducible route's decode +14.6 % on the groupwise 27B (MTP 3), the
//   default route's unchanged (2026-10-25).
// - Pipelined: the pipelined kernel at the schedule's own block rows, with a 1-4 slot ring in dynamic
//   shared memory sized for the launch and a vectorised A decode.
// - Release: the original sliced-K kernels.
// Every shape keeps the schedule's K warps and reduction, so each route returns the release kernel's
// bits; Pipelined and Release remain for the tests that check it. INFERNIX_TMP_GW_SHAPE=<rows>x<stages>
// (for example 32x3, a temporary switch) forces one shape wherever it is admissible, for the op sweeps
// that tune the rule.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace infernix::ops::detail {

enum class SlicedKRoute : std::uint8_t { Release, Pipelined, OneWave };

[[nodiscard]] SlicedKRoute groupwise_sliced_k_route() noexcept;

// Tests compare the routes in one process: an override replaces the environment's route and forced
// shape until reset (std::nullopt). Not for production code.
struct SlicedKShape {
    int block_rows = 0;
    int stages     = 0;

    bool operator==(const SlicedKShape&) const = default;
};

void set_groupwise_sliced_k_route_for_testing(std::optional<SlicedKRoute> route,
                                              SlicedKShape forced = {});

// INFERNIX_TMP_GW_SHAPE, or {0, 0} when unset (the rule chooses).
[[nodiscard]] SlicedKShape forced_sliced_k_shape() noexcept;

// One admissible shape of a launch: its CTAs, the CTAs one SM holds at its threads, registers and
// shared memory, and the weight bytes one ring slot holds.
struct SlicedKCandidate {
    SlicedKShape shape;
    std::int64_t ctas               = 0;
    int resident                    = 0;
    std::int64_t weight_slot_bytes  = 0;
};

// The rule (tiles16's lesson: a second wave of even two CTAs costs a narrow call 25-30 %): the
// fewest waves (ceil(ctas / (resident x SMs))); among those the most weight bytes in flight per SM
// (min(resident, ceil(ctas / SMs)) x stages x slot bytes, Little's law for a latency-bound
// stream); then the fewer block rows (more CTAs) and the shallower ring. Candidates with no
// resident CTA are skipped. Returns the chosen index; throws when none is usable.
[[nodiscard]] std::size_t choose_sliced_k_shape(std::span<const SlicedKCandidate> candidates,
                                                int multiprocessors);

} // namespace infernix::ops::detail
