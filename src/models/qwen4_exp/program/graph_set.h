#pragma once

// Qwen3.8-Flash-Next's Program captures its CUDA graph set at startup instead of at each graph's first use
// (ProgramImpl::capture_graph_set, program/graph_set.cpp; candidates §23 NU-04, design §19.3.37). This seam lets the
// startup-capture identity test run the first-use arm beside it in one process.

#include <optional>

namespace infernix::models::qwen4_exp::testing {

// Turns the startup capture off (false) or on for the Programs constructed after the call; nullopt (the default)
// keeps it on. For tests only.
void set_startup_capture(std::optional<bool> capture) noexcept;
[[nodiscard]] std::optional<bool> startup_capture() noexcept;

} // namespace infernix::models::qwen4_exp::testing
