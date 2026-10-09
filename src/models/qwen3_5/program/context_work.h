#pragma once

#include "runtime/contract/resources.h"
#include <cstddef>
#include <cstdint>
#include <chrono>

namespace infernix::models::qwen3_5::detail {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point started) noexcept;

std::int32_t checked_i32(std::uint32_t value, const char* label);

std::uint32_t kv_pages_for_frontier(std::uint32_t frontier) noexcept;

} // namespace infernix::models::qwen3_5::detail
