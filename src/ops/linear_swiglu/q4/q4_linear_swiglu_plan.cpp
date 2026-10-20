#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include "infernix/ops/linear.h"
#include "infernix/ops/silu_mul.h"
#include "core/layout.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_w4a8.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

// Under LinearRoutes::Q4W4A8, calls of at least this many columns take the W4A8 route; narrower
// calls keep the exact route's schedule and bits. 256 is the crossover upstream NInfer measured on
// an RTX 5090 (giveen/ninfer-ext e99059abb); the route sweep of infernix_q4_linear_swiglu_bench
// (--routes exact and w4a8 over --t-sweep) sets it for this kernel.
constexpr std::int32_t kW4A8MinCols = 256;

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4LinearSwiGluScheduleId schedule;
};

constexpr Q4LinearSwiGluProblem kShape{34816, 17408, 5120, 5120, 1};

constexpr std::array<RouteSpec, 10> kRoutes{{
    {{1, 1}, Q4LinearSwiGluScheduleId::GemvPair},
    {{2, 32}, Q4LinearSwiGluScheduleId::SmallTTiled},
    {{33, 128}, Q4LinearSwiGluScheduleId::Materialized},
    {{129, 168}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail},
    {{169, 224}, Q4LinearSwiGluScheduleId::Materialized},
    {{225, 256}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{257, 384}, Q4LinearSwiGluScheduleId::Materialized},
    {{385, 512}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{513, 640}, Q4LinearSwiGluScheduleId::Materialized},
    {{641, kAnyCols}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "Q4 LinearSwiGLU routes must be exact, contiguous, and closed");

bool supported_shape(const Q4LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == kShape.gate_up_rows &&
           problem.output_rows == kShape.output_rows && problem.k == kShape.k &&
           problem.padded_k == kShape.padded_k;
}

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows, std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}

std::size_t w4a8_workspace_bytes(std::int32_t cols, std::int32_t k) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_q4_w4a8_workspace(layout, cols, k);
    return layout.peak_bytes(1);
}

bool selects_w4a8(LinearRoutes routes, std::int32_t cols) noexcept {
    return has_route(routes, LinearRoutes::Q4W4A8) && cols >= kW4A8MinCols;
}

} // namespace

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        return "linear_swiglu.q4.gemv.paired_rows";
    case Q4LinearSwiGluScheduleId::SmallTTiled:
        return "linear_swiglu.q4.mma.small_t.tiled";
    case Q4LinearSwiGluScheduleId::Materialized:
        return "linear_swiglu.q4.materialized";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c128";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c128.narrow_tail";
    case Q4LinearSwiGluScheduleId::W4A8:
        return "linear_swiglu.q4.w4a8.mma_s8";
    case Q4LinearSwiGluScheduleId::SmallTTiledColumnInvariant:
        return "linear_swiglu.q4.mma.small_t.column_invariant";
    }
    return "linear_swiglu.q4.unknown";
}

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem,
                                                 LinearRoutes routes, bool column_invariant) {
    if (!q4_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "q4 linear_swiglu: exact problem or column count is not admitted");
    }
    if (!valid_linear_routes(routes)) {
        throw std::invalid_argument("q4 linear_swiglu: invalid route flags");
    }
    if (column_invariant) { return {Q4LinearSwiGluScheduleId::SmallTTiledColumnInvariant, 0, routes}; }
    if (selects_w4a8(routes, problem.cols)) {
        return {Q4LinearSwiGluScheduleId::W4A8, w4a8_workspace_bytes(problem.cols, problem.k),
                routes};
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4LinearSwiGluPlan plan{route.schedule, 0, routes};
        switch (route.schedule) {
        case Q4LinearSwiGluScheduleId::GemvPair:
        case Q4LinearSwiGluScheduleId::SmallTTiled:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail:
            return plan;
        case Q4LinearSwiGluScheduleId::Materialized:
            plan.workspace_bytes = materialized_workspace_bytes(problem.gate_up_rows, problem.cols);
            return plan;
        case Q4LinearSwiGluScheduleId::W4A8:
        case Q4LinearSwiGluScheduleId::SmallTTiledColumnInvariant:
            break; // not exact-route catalog entries
        }
    }
    throw std::logic_error("q4 linear_swiglu: admitted problem has no covering route");
}

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols, LinearRoutes routes,
                                                      bool column_invariant) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q4 linear_swiglu: invalid column interval");
    }
    if (column_invariant) {
        (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols}, routes, true);
        return 0;
    }
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, min_cols}, routes);
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols}, routes);

    // Columns below the W4A8 threshold (all of them without the flag) run the exact catalog; the
    // W4A8 workspace grows with the column count, so its largest call is the interval's last.
    const bool w4a8               = selects_w4a8(routes, max_cols);
    const std::int32_t exact_last = w4a8 ? std::min(max_cols, kW4A8MinCols - 1) : max_cols;
    std::size_t maximum           = 0;
    for (const RouteSpec& route : kRoutes) {
        if (min_cols > exact_last) { break; }
        if (route.cols.last < min_cols || route.cols.first > exact_last) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, exact_last);
        maximum                     = std::max(maximum, q4_linear_swiglu_resolve_plan(
                                        {gate_up_rows, output_rows, k, padded_k, endpoint})
                                                            .workspace_bytes);
    }
    if (w4a8) {
        maximum = std::max(
            maximum,
            q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols}, routes)
                .workspace_bytes);
    }
    return maximum;
}

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream, bool column_invariant) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan resolved = q4_linear_swiglu_resolve_plan(problem, plan.routes, column_invariant);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q4 linear_swiglu: plan does not match the exact problem");
    }

    switch (plan.schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        q4_linear_swiglu_gemv_pair_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTTiled:
        q4_linear_swiglu_small_t_tiled_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::Materialized: {
        auto scratch_scope = ws.scope();
        Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
        linear(x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_tail_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::W4A8:
        q4_linear_swiglu_w4a8_launch(x, w, out, ws, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTTiledColumnInvariant:
        q4_linear_swiglu_small_t_invariant_launch(x, w, out, stream);
        return;
    }
    throw std::logic_error("q4 linear_swiglu: unknown schedule");
}

void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream, LinearRoutes routes, bool column_invariant) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan plan = q4_linear_swiglu_resolve_plan(problem, routes, column_invariant);
    q4_linear_swiglu_execute_plan(plan, x, w, out, ws, stream, column_invariant);
}

} // namespace infernix::ops::detail
