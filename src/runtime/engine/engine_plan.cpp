#include "runtime/engine/engine_plan.h"

#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace infernix::runtime {

void plan_readings(EnginePlan& plan, std::string gpu_name, const VramSnapshot& vram, const HostMemorySnapshot& host,
                   const EngineOptions& options) {
    plan.gpu.name        = std::move(gpu_name);
    plan.gpu.total_bytes = vram.device_total;
    plan.gpu.free_bytes  = vram.device_free;
    plan.gpu.budget_bytes.reset();
    if (vram.has_budget) { plan.gpu.budget_bytes = vram.local_budget; }
    plan.ram.total_bytes     = host.total_physical;
    plan.ram.available_bytes = host.available_physical;
    plan.ram.ram_free_bytes  = options.ram_free_bytes;
}

void plan_refuse(EnginePlan& plan, std::string refusal) {
    if (plan.fits) { plan.error = std::move(refusal); }
    plan.fits = false;
}

std::size_t planned_free_after_weights(std::size_t free_now, std::uint64_t weights_bytes,
                                       std::uint64_t beside_weights_bytes) noexcept {
    const std::uint64_t taken = weights_bytes + beside_weights_bytes;
    return free_now > taken ? static_cast<std::size_t>(free_now - taken) : 0;
}

std::optional<KvCapacityResolution> plan_capacity(EnginePlan& plan, const KvCapacityPolicy& policy,
                                                  const SequenceCapacityCurve& curve,
                                                  std::size_t free_after_weights, std::size_t headroom) {
    EnginePlanVram& vram           = plan.vram;
    vram.minimum_reservation_bytes = curve.minimum_device_reservation_bytes;
    vram.bytes_per_page_group      = curve.bytes_per_additional_main_page_group;
    vram.page_group_tokens         = curve.main_page_tokens;
    // resolve_kv_capacity applies the headroom to an automatic capacity only.
    vram.headroom_bytes = policy.mode == KvCapacityMode::Automatic ? headroom : 0;
    try {
        const KvCapacityResolution resolution = resolve_kv_capacity(policy, curve, free_after_weights, headroom);
        vram.kv_tokens     = resolution.resolved_tokens;
        vram.planned_bytes = vram.weights_bytes + resolution.runtime_reservation_bytes;
        const std::uint64_t used =
            static_cast<std::uint64_t>(resolution.runtime_reservation_bytes) + vram.headroom_bytes;
        vram.left_bytes = free_after_weights > used ? free_after_weights - used : 0;
        return resolution;
    } catch (const std::invalid_argument& refusal) {
        vram.kv_tokens     = 0;
        vram.kv_bytes      = 0;
        vram.planned_bytes = vram.weights_bytes + curve.minimum_device_reservation_bytes;
        vram.left_bytes    = 0;
        plan_refuse(plan, refusal.what());
        return std::nullopt;
    }
}

void plan_ram_totals(EnginePlan& plan, std::uint64_t planned) noexcept {
    plan.ram.planned_bytes   = planned;
    const std::uint64_t used = plan.ram.ram_free_bytes + planned;
    plan.ram.left_bytes      = plan.ram.available_bytes > used ? plan.ram.available_bytes - used : 0;
}

std::optional<std::uint32_t> largest_fitting_context(std::uint32_t upper, std::uint32_t hint,
                                                     const std::function<bool(std::uint32_t)>& fits) {
    const auto test = [&](std::uint32_t context) {
        try {
            return fits(context);
        } catch (const std::invalid_argument&) { return false; }
    };
    if (upper == 0) { return std::nullopt; }
    hint = std::clamp<std::uint32_t>(hint, 1, upper);
    // test(low) holds and test(high) does not.
    std::uint32_t low = 0, high = 0;
    if (test(hint)) {
        if (hint == upper || test(upper)) { return upper; }
        low  = hint;
        high = upper;
    } else {
        if (hint == 1 || !test(1)) { return std::nullopt; }
        low  = 1;
        high = hint;
    }
    while (high - low > 1) {
        const std::uint32_t middle = low + (high - low) / 2;
        if (test(middle)) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

} // namespace infernix::runtime
