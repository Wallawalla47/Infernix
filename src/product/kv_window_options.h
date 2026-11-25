#pragma once

#include "infernix/kv_format.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace infernix::product {

// --kv-window-dtype auto|bf16|bf16p|int8 (EngineOptions::kv_window_cache, Inkling 2's sliding windows): auto is empty
// (the --kv-dtype's default).
inline std::optional<KvCacheStorage> parse_kv_window_dtype(std::string_view text) {
    if (text == "auto") { return std::nullopt; }
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "bf16p") { return KvCacheStorage::Bf16Packed; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    throw std::invalid_argument("--kv-window-dtype takes auto, bf16, bf16p or int8, not " + std::string(text));
}

inline const char* kv_window_dtype_name(const std::optional<KvCacheStorage>& window) {
    return window ? kv_cache_cli_name(*window) : "auto";
}

} // namespace infernix::product
