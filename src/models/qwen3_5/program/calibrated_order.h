#pragma once

// The calibrated expert order of the Qwen3.5-122B-A10B geometry (Ornith-1.5-122B-NVFP4, its UNCENSORED export and
// any other artifact of the shape: formats and re-quantized drafters do not change the routed experts' keys): each
// layer's routed experts from the highest cache priority to the lowest. The expert cache seeds its frames, and the
// SSD tier its RAM, from it when no saved state is usable (runtime::expert_offload::ExpertOffload::warm_start;
// docs/maintainer/qwen3_5-122b-offload-design.md §9.2; candidates §23 NU-43), as Qwen3.8-Flash-Next does from its
// own (models/qwen4_exp/program/calibrated_order.h). The table (calibrated_order.inc) is generated from route traces
// of a mixed task set by tools/expert_cache_replay/calibrated_order.py; while it is empty no model has an order and
// a first session starts with an empty cache, as before.

#include "runtime/expert_offload/expert_cache.h"

#include <cstdint>
#include <optional>
#include <span>

namespace infernix::models::qwen3_5 {

// The geometry an order ranks: the router's input and expert count, the cache's keys (layers x experts) and the
// experts' shape. An order applies to a model only when every field is equal.
struct ExpertOrderShape {
    std::uint32_t hidden = 0, layers = 0, experts = 0, top_k = 0, intermediate = 0, shared_intermediate = 0;
    friend bool operator==(const ExpertOrderShape&, const ExpertOrderShape&) = default;
};
// The Qwen3.5-122B-A10B geometry the built-in table ranks (H 3072, 48 layers, 256 experts top-8, I 1024, shared 1024).
inline constexpr ExpertOrderShape kCalibratedOrderShape{3072, 48, 256, 8, 1024, 1024};

// The built-in order for a model of shape `model` as a saved state (built once per process): null when `model` is
// not kCalibratedOrderShape or no order has been generated.
[[nodiscard]] const runtime::expert_offload::expert_cache::SavedState* calibrated_expert_order(const ExpertOrderShape& model);
// Whether calibrated_order.inc holds a generated order.
[[nodiscard]] bool has_calibrated_order_table() noexcept;

// The shape guard over a table: `order` holds table.layers rows of table.experts expert ids (row = layer, highest
// priority first). The order as a saved state (expert_cache::calibrated_order_state) when `model` equals `table` and
// the table is not empty, else nullopt. Throws std::invalid_argument when the table's size differs from its shape or
// a row is not a permutation of the experts.
[[nodiscard]] std::optional<runtime::expert_offload::expert_cache::SavedState>
calibrated_order_for(const ExpertOrderShape& model, const ExpertOrderShape& table, std::span<const std::uint16_t> order);

} // namespace infernix::models::qwen3_5
