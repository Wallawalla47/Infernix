// The Qwen3.5-122B-A10B's calibrated expert order (calibrated_order.h): the generated table and its shape guard.
#include "models/qwen3_5/program/calibrated_order.h"

#include "runtime/expert_offload/expert_state.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen3_5 {
namespace {

// kTableLayers x kTableExperts expert ids generated for top-kTableTopK routing (0 x 0: none generated yet).
#include "models/qwen3_5/program/calibrated_order.inc"

static_assert(kTableLayers == 0 || (kTableLayers == kCalibratedOrderShape.layers &&
                                    kTableExperts == kCalibratedOrderShape.experts &&
                                    kTableTopK == kCalibratedOrderShape.top_k),
              "calibrated_order.inc was generated for another geometry than kCalibratedOrderShape");
static_assert(kTable.size() == static_cast<std::size_t>(kTableLayers) * kTableExperts);

} // namespace

std::optional<runtime::expert_offload::expert_cache::SavedState>
calibrated_order_for(const ExpertOrderShape& model, const ExpertOrderShape& table, std::span<const std::uint16_t> order) {
    if (table.layers == 0 || table.experts == 0) { return std::nullopt; }
    if (order.size() != static_cast<std::size_t>(table.layers) * table.experts) {
        throw std::invalid_argument("calibrated expert order: " + std::to_string(order.size()) + " entries for " +
                                    std::to_string(table.layers) + " layers of " + std::to_string(table.experts) +
                                    " experts");
    }
    if (!(model == table)) { return std::nullopt; }
    return runtime::expert_offload::expert_cache::calibrated_order_state(
        table.layers, table.experts, [&](std::uint32_t layer, std::uint32_t rank) -> std::uint32_t {
            return order[static_cast<std::size_t>(layer) * table.experts + rank];
        });
}

const runtime::expert_offload::expert_cache::SavedState* calibrated_expert_order(const ExpertOrderShape& model) {
    static const auto order = []() -> std::optional<runtime::expert_offload::expert_cache::SavedState> {
        if (kTableLayers == 0) { return std::nullopt; }
        return calibrated_order_for(kCalibratedOrderShape, kCalibratedOrderShape,
                                    std::span<const std::uint16_t>(kTable.data(), kTable.size()));
    }();
    return order && model == kCalibratedOrderShape ? &*order : nullptr;
}

bool has_calibrated_order_table() noexcept { return kTableLayers != 0; }

} // namespace infernix::models::qwen3_5
