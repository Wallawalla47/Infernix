#include "models/qwen3_5/execution/ffn.h"

#include "core/layout.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/residual_add.h"
#include "infernix/ops/resident_moe.h"
#include "infernix/ops/silu_mul.h"

#include <array>
#include <stdexcept>
#include <type_traits>

namespace infernix::models::qwen3_5::execution {
namespace {

// Qwen3.5 MoE routes top-8 (Parameters checks the configuration).
constexpr std::int32_t kMoeTopK = 8;

template <class A>
constexpr bool kRuns = std::is_same_v<A, WorkspaceArena>;

// A sparse MoE outside the closed SparseMoe Op's geometry (the Qwen3.5-122B-A10B drafter: H 3072,
// 256 experts of I 1024, row-split banks), one sequence of allocations for execution and sizing:
// FP32 router and shared-gate logits read at BF16 as the model computes them (projection_fp32,
// moe_route with RouterLogits::kBf16), the routed experts resident (resident_moe_experts), the
// shared expert, and y = bf16(sum w y_i + s y_shared) added to the residual. Every step depends
// only on its own column, so the route is column-invariant whenever the shared expert's
// projections run their column-invariant policy.
template <class A>
void resident_moe(A& a, const Tensor& hidden, const ops::SparseMoeWeights& p, Tensor* residual,
                  DeviceExecutionView execution) {
    const cudaStream_t s  = execution.stream;
    const std::int32_t T  = hidden.ne[1];
    const std::int32_t H  = p.router_shared_gate.k, E = p.router_shared_gate.n - 1;
    const std::int32_t I  = p.routed_down.k, Is = p.shared_down.k;
    const std::int32_t KT = kMoeTopK * T;
    auto scope            = a.scope();
    Tensor logits         = a.alloc(DType::FP32, {E + 1, T});
    ops::MoeRouting routing{a.alloc(DType::I32, {kMoeTopK, T}), a.alloc(DType::FP32, {kMoeTopK, T}),
                            a.alloc(DType::FP32, {T})};
    Tensor outputs              = a.alloc(DType::BF16, {H, KT});
    const std::size_t expert_ws = ops::resident_moe_workspace_bytes(KT, I);
    const DeviceSpan experts    = a.alloc_bytes(expert_ws);
    Tensor gate_up              = a.alloc(DType::BF16, {2 * Is, T});
    Tensor product              = a.alloc(DType::BF16, {Is, T});
    Tensor shared               = a.alloc(DType::BF16, {H, T});
    Tensor y                    = a.alloc(DType::BF16, {H, T});
    if constexpr (kRuns<A>) {
        const Tensor router(const_cast<void*>(p.router_shared_gate.qdata), DType::BF16, {H, E + 1});
        const std::array<const Tensor*, 1> rows{&router};
        ops::projection_fp32(hidden, rows, logits, s);
        ops::moe_route(logits, kMoeTopK, routing, s, ops::RouterLogits::kBf16);
        ops::resident_moe_experts(hidden, routing.ids, p.routed_gate_up, p.routed_down, E, I, experts.data,
                                  expert_ws, outputs, s);
        {
            auto call = a.scope();
            ops::linear(hidden, p.shared_gate_up, gate_up, p.policy, a, s);
        }
        ops::silu_mul(gate_up.slice(0, 0, Is), gate_up.slice(0, Is, Is), product, s);
        {
            auto call = a.scope();
            ops::linear(product, p.shared_down, shared, p.policy, a, s);
        }
        ops::moe_combine(outputs, routing, shared, y, s);
        ops::residual_add(y, *residual, s);
    } else {
        (void)residual;
        (void)execution;
        {
            auto call = a.scope();
            (void)a.alloc_bytes(ops::linear_workspace_capacity_bytes(
                p.shared_gate_up.qtype, p.shared_gate_up.n, p.shared_gate_up.k, p.policy, 1, T));
        }
        {
            auto call = a.scope();
            (void)a.alloc_bytes(ops::linear_workspace_capacity_bytes(
                p.shared_down.qtype, p.shared_down.n, p.shared_down.k, p.policy, 1, T));
        }
    }
}

} // namespace

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp, bool wide_verification,
                                ops::LinearRoutes gate_up_routes) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* offloaded = std::get_if<OffloadedMoeParameters>(&parameters)) {
        return offloaded_moe_workspace_bytes(*offloaded, first, last);
    }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        if (!ops::sparse_moe_geometry(*moe)) {
            // Every allocation grows with T: the interval's widest call bounds it.
            WorkspaceLayoutBuilder layout;
            Tensor x(nullptr, DType::BF16, {moe->router_shared_gate.k, last});
            resident_moe(layout, x, *moe, nullptr, DeviceExecutionView{});
            return layout.peak_bytes(1);
        }
        return ops::sparse_moe_workspace_capacity_bytes(
            moe->routed_gate_up.qtype, moe->routed_down.qtype, moe->policy, first, last);
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    if (mtp || gu.qtype == QType::Q6_G64_FP16) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        if (mtp) {
            (void)layout.alloc(DType::BF16, {down.n, last});
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        } else {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last, gate_up_routes));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, residual_projection_policy(p.down, wide_verification),
                first, last));
        }
    }
    return layout.peak_bytes(1);
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp, bool wide_verification, ops::LinearRoutes gate_up_routes,
         ops::LinearRoutes down_routes, const OffloadedMoeContext* offloaded) {
    const auto stream  = execution.stream;
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (const auto* moe = std::get_if<OffloadedMoeParameters>(&parameters)) {
        if (offloaded == nullptr) { throw std::logic_error("FFN: an offloaded MoE layer needs the Program's experts"); }
        offloaded_moe(hidden, *moe, residual, *offloaded, workspace, execution);
        return;
    }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        if (!ops::sparse_moe_geometry(*moe)) {
            resident_moe(workspace, hidden, *moe, &residual, execution);
            return;
        }
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        execution);
        return;
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (mtp || gu.qtype == QType::Q6_G64_FP16) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, p.gate_up.policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        if (mtp) {
            Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
            ops::linear(activation, down, delta, p.down.policy, workspace, stream);
            ops::residual_add(delta, residual, stream);
            return;
        }
        ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation, p.gate_up.policy, workspace, stream,
                           gate_up_routes);
    }
    ops::linear_add(activation, down, residual,
                    residual_projection_policy(p.down, wide_verification), workspace, stream,
                    down_routes);
}

} // namespace infernix::models::qwen3_5::execution
