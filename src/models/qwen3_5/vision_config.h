#pragma once

#include "artifact/schema.h"
#include "models/qwen3_5/config.h"

#include <span>
#include <string_view>

namespace infernix::models::qwen3_5 {

// The Vision tower's config (a Qwen3.5-family tower), accepting the listed model_type values; also
// read by Qwen4Exp, whose tower differs only in its output width (bound on the merger).
[[nodiscard]] VisionConfig parse_vision_config(const artifact::Json& value,
                                               std::span<const std::string_view> model_types);

} // namespace infernix::models::qwen3_5
