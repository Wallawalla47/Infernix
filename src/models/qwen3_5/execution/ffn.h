#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/offloaded_moe.h"
#include "models/qwen3_5/execution/parameters.h"

namespace infernix::models::qwen3_5::execution {

// gate_up_routes: the fast-path routes of the dense gate/up projection (linear_swiglu); the
// workspace query and every call must pass the same flags. down_routes: the dense down projection's
// (linear_add; its A16 routes take no workspace). An offloaded MoE layer (OffloadedMoeParameters)
// needs `offloaded`: the Program's experts and the call's route; its workspace is sized for
// `expert_wide`, the experts' wide route (OffloadedExperts::wide_a16 of every call). An `mtp` call of
// a resident MoE outside the SparseMoe geometry (the 122B drafter) with `offloaded` reads its pooled
// experts when the Program has them (OffloadedExperts::drafter_slots).
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false,
                                              bool wide_verification = false,
                                              ops::LinearRoutes gate_up_routes = ops::LinearRoutes::Exact,
                                              ops::offloaded_moe::WideA16 expert_wide =
                                                  ops::offloaded_moe::WideA16::kExact);
// residual += MLP(Offset rmsnorm(residual)) in ops::dense_mlp_fused_rmsnorm_nvfp4's three launches (SI-02+: the
// NVFP4 A4 residual activations of verification rounds, and the A4 activations of prompt calls of 256 columns or
// more, encoded by their producers), bit for bit rmsnorm() and then ffn() with the same
// wide_verification, where that Op serves the dense layer's projections, policies and width; elsewhere it
// launches nothing and returns false. ffn_workspace_bytes covers its calls.
bool fused_dense_mlp(Tensor& residual, const Tensor& norm_weight, float eps, const FfnParameters& parameters,
                     WorkspaceArena& workspace, cudaStream_t stream, bool wide_verification);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp = false, bool wide_verification = false,
         ops::LinearRoutes gate_up_routes = ops::LinearRoutes::Exact,
         ops::LinearRoutes down_routes    = ops::LinearRoutes::Exact,
         const OffloadedMoeContext* offloaded = nullptr);

} // namespace infernix::models::qwen3_5::execution
