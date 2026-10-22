#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"

namespace infernix::models::qwen3_5::execution {

// gate_up_routes: the fast-path routes of the dense gate/up projection (linear_swiglu); the
// workspace query and every call must pass the same flags. down_routes: the dense down projection's
// (linear_add; its A16 routes take no workspace).
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false,
                                              bool wide_verification = false,
                                              ops::LinearRoutes gate_up_routes = ops::LinearRoutes::Exact);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp = false, bool wide_verification = false,
         ops::LinearRoutes gate_up_routes = ops::LinearRoutes::Exact,
         ops::LinearRoutes down_routes    = ops::LinearRoutes::Exact);

} // namespace infernix::models::qwen3_5::execution
