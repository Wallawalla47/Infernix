#pragma once

// Engine::plan's shared steps (infernix::EnginePlan): the device and Host readings, the refusal a
// start would give, the Qwen3.5 family's capacity resolution beside the weights, the RAM totals and
// the largest --max-context. The architectures' constructions (construct_model, construct_qwen4_exp)
// run their own startup sequence up to the Program and fill the plan through these, so the sizing
// formulas stay the start's own (resolve_kv_capacity, the RAM ledger, size_expert_frames).

#include "core/host_memory.h"
#include "core/vram_budget.h"
#include "infernix/types.h"
#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace infernix::runtime {

// The device's name and size, and the free memory and OS budget of a reading; the Host's memory and
// the reserve (--ram-free-mib).
void plan_readings(EnginePlan& plan, std::string gpu_name, const VramSnapshot& vram, const HostMemorySnapshot& host,
                   const EngineOptions& options);

// A start's memory refusal: the plan does not fit, and the first refusal is its error.
void plan_refuse(EnginePlan& plan, std::string refusal);

// The free device memory a Qwen3.5-family start sizes its Program against, read where the start reads
// it after the weights: the plan's reading there less the weights' arena it did not allocate (and with
// --vision-offload on what the weights' pool maps beyond it, the arena's last partial chunk: the overlay
// window borrows evicted chunks).
[[nodiscard]] std::size_t planned_free_after_weights(std::size_t free_now, std::uint64_t weights_bytes,
                                                     std::uint64_t beside_weights_bytes = 0) noexcept;

// Qwen3.5 family: resolves `policy` on `curve` against `free_after_weights` with `headroom` as a start
// does (resolve_kv_capacity) and records the curve, the resolution and the device totals in plan.vram
// (weights_bytes must be set). Returns the resolution, or empty after plan_refuse with the start's
// own message when it does not fit.
[[nodiscard]] std::optional<KvCapacityResolution> plan_capacity(EnginePlan& plan, const KvCapacityPolicy& policy,
                                                                const SequenceCapacityCurve& curve,
                                                                std::size_t free_after_weights, std::size_t headroom);

// The RAM totals: `planned` (everything the start plans in RAM) and what the available memory leaves
// beside it and the reserve.
void plan_ram_totals(EnginePlan& plan, std::uint64_t planned) noexcept;

// The largest context in [1, upper] that `fits`, which must hold for every smaller context once it
// holds for one; a context the options cannot take may throw std::invalid_argument (not fitting).
// The search starts at `hint` (the context asked for), so a smallest context the planner refuses for
// other reasons does not hide the answer. Empty when none fits.
[[nodiscard]] std::optional<std::uint32_t> largest_fitting_context(std::uint32_t upper, std::uint32_t hint,
                                                                   const std::function<bool(std::uint32_t)>& fits);

} // namespace infernix::runtime
