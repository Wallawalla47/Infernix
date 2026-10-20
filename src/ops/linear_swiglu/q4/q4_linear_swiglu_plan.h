#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

enum class Q4LinearSwiGluScheduleId {
    GemvPair,
    SmallTTiled,
    Materialized,
    MmaSplitHalfPairR32C128,
    MmaSplitHalfPairR32C128Tail,
    W4A8, // the q4-w4a8 fast path (LinearRoutes::Q4W4A8)
    // LinearPolicy::A16ColumnInvariant: the small-T sliced-K family (8 K-warps, pairwise reduction) at
    // every width, its 8-column instance for one column and 32-column slices beyond 32, so a column's
    // bits are those of the 2..32-column exact route whatever the call's width.
    SmallTTiledColumnInvariant,
};

struct Q4LinearSwiGluProblem {
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4LinearSwiGluPlan {
    Q4LinearSwiGluScheduleId schedule;
    std::size_t workspace_bytes;
    // The route flags the plan was resolved with.
    LinearRoutes routes = LinearRoutes::Exact;
};

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept;

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept;
// Without LinearRoutes::Q4W4A8 (or below its column threshold) the exact route is selected, the
// same schedule whatever other flags are set. `column_invariant` (LinearPolicy::A16ColumnInvariant)
// selects SmallTTiledColumnInvariant at every width and ignores the route flags.
Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem,
                                                 LinearRoutes routes = LinearRoutes::Exact,
                                                 bool column_invariant = false);

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols, LinearRoutes routes,
                                                      bool column_invariant = false);

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream,
                                   bool column_invariant = false);
void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream, LinearRoutes routes = LinearRoutes::Exact,
                               bool column_invariant = false);

} // namespace infernix::ops::detail
