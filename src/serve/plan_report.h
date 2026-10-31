#pragma once

// infernix-serve --plan FILE: the Engine's sizing of a start (infernix::Engine::plan) as one JSON
// object, written to FILE, and the process's exit status. The key names are a consumer contract: the
// launcher reads them (docs/serving.md, "Sizing a start without loading").

#include "infernix/types.h"

#include <string>
#include <string_view>

namespace infernix::serve {

inline constexpr int kPlanSchema = 1;
// infernix-serve --plan's exit statuses: the start fits, it does not, or no plan could be made.
inline constexpr int kPlanFitsExitStatus      = 0;
inline constexpr int kPlanDoesNotFitExitStatus = 3;
inline constexpr int kPlanFailedExitStatus     = 1;

// The plan as one JSON object (schema kPlanSchema): sizes in bytes, null where the architecture or the
// options have no such quantity.
[[nodiscard]] std::string render_plan_json(const infernix::EnginePlan& plan);
// The object a failed plan leaves: {"schema": 1, "error": message}.
[[nodiscard]] std::string render_plan_error_json(std::string_view message);

} // namespace infernix::serve
