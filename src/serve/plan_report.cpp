#include "serve/plan_report.h"

#include <nlohmann/json.hpp>

namespace infernix::serve {
namespace {

using Json = nlohmann::json;

template <class T>
Json optional_json(const std::optional<T>& value) {
    return value ? Json(*value) : Json(nullptr);
}

const char* tier_name(infernix::EnginePlanExpertTier tier) noexcept {
    return tier == infernix::EnginePlanExpertTier::Full ? "full" : "ssd";
}

const char* layout_name(infernix::ExpertRamLayout layout) noexcept {
    return layout == infernix::ExpertRamLayout::Exclusive ? "exclusive" : "full";
}

const char* residency_name(infernix::NgramResidency residency) noexcept {
    switch (residency) {
    case infernix::NgramResidency::Ram:
        return "ram";
    case infernix::NgramResidency::Volume:
        return "volume";
    case infernix::NgramResidency::Auto:
        break;
    }
    return "auto";
}

std::string dump(const Json& value) {
    // Messages may quote a path; replace any byte that is not UTF-8 rather than fail the report.
    return value.dump(-1, ' ', false, Json::error_handler_t::replace) + '\n';
}

} // namespace

std::string render_plan_json(const infernix::EnginePlan& plan) {
    const infernix::EnginePlanGpu& gpu   = plan.gpu;
    const infernix::EnginePlanVram& vram = plan.vram;
    const infernix::EnginePlanRam& ram   = plan.ram;
    const Json root{
        {"schema", kPlanSchema},
        {"fits", plan.fits},
        // Free memory is read live and what a start measures only once loaded is estimated.
        {"estimated", true},
        {"architecture", plan.architecture},
        {"error", plan.fits ? Json(nullptr) : Json(plan.error)},
        {"gpu",
         Json{{"name", gpu.name},
              {"total_bytes", gpu.total_bytes},
              {"free_bytes", gpu.free_bytes},
              {"budget_bytes", optional_json(gpu.budget_bytes)}}},
        {"vram",
         Json{{"weights_bytes", vram.weights_bytes},
              {"kv_bytes", vram.kv_bytes},
              {"kv_tokens", vram.kv_tokens},
              {"minimum_reservation_bytes", vram.minimum_reservation_bytes},
              {"bytes_per_page_group", optional_json(vram.bytes_per_page_group)},
              {"page_group_tokens", optional_json(vram.page_group_tokens)},
              {"workspace_bytes", optional_json(vram.workspace_bytes)},
              {"graph_allowance_bytes", optional_json(vram.graph_allowance_bytes)},
              {"expert_cache_bytes", optional_json(vram.expert_cache_bytes)},
              {"expert_frames", optional_json(vram.expert_frames)},
              {"headroom_bytes", vram.headroom_bytes},
              {"planned_bytes", vram.planned_bytes},
              {"left_bytes", vram.left_bytes}}},
        {"ram",
         Json{{"total_bytes", ram.total_bytes},
              {"available_bytes", ram.available_bytes},
              {"ram_free_bytes", ram.ram_free_bytes},
              {"experts_bytes", optional_json(ram.experts_bytes)},
              {"pinned_bytes", optional_json(ram.pinned_bytes)},
              {"host_context_bytes", optional_json(ram.host_context_bytes)},
              {"ngram_bytes", optional_json(ram.ngram_bytes)},
              {"ngram_residency", ram.ngram_residency ? Json(residency_name(*ram.ngram_residency)) : Json(nullptr)},
              {"tier", ram.tier ? Json(tier_name(*ram.tier)) : Json(nullptr)},
              {"expert_ram_layout",
               ram.expert_ram_layout ? Json(layout_name(*ram.expert_ram_layout)) : Json(nullptr)},
              {"planned_bytes", ram.planned_bytes},
              {"left_bytes", ram.left_bytes}}},
        {"max_context", optional_json(plan.max_context)},
    };
    return dump(root);
}

std::string render_plan_error_json(std::string_view message) {
    return dump(Json{{"schema", kPlanSchema}, {"error", std::string(message)}});
}

} // namespace infernix::serve
