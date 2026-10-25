#pragma once

#include "infernix/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace infernix::product {

// --logits-dtype fp32|bf16 (EngineOptions::logits_dtype).
inline LogitsDtype parse_logits_dtype(std::string_view text) {
    if (text == "fp32") { return LogitsDtype::FP32; }
    if (text == "bf16") { return LogitsDtype::BF16; }
    throw std::invalid_argument("--logits-dtype takes fp32 or bf16, not " + std::string(text));
}

inline const char* logits_dtype_name(LogitsDtype dtype) { return dtype == LogitsDtype::BF16 ? "bf16" : "fp32"; }

} // namespace infernix::product
