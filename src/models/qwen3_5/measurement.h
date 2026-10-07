#pragma once

#include "models/qwen3_5/model.h"

#include <string>

namespace infernix::models::qwen3_5 {

// Measurement applicability, independent of checkpoint names, object IDs, payload values and
// device addresses. This key never selects or admits an execution implementation.
[[nodiscard]] std::string prefill_signature(const Model& model);

} // namespace infernix::models::qwen3_5
