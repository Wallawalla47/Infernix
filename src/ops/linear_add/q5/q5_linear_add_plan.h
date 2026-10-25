#pragma once

#include "core/weight.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

enum class Q5LinearAddScheduleId {
    Split2ExactResidual,
    SlicedR16T8W4S2,
    SlicedR16T16W4S2,
    SlicedR16T24W4S2,
    SlicedR32T32W4S2,
    SlicedR32T24W4S2Pairwise,
    SlicedR32T32W4S1,
    SlicedR32T32W2S2,
    SlicedR32T64W2S1,
    MmaResidualR32T32K128,
    MmaResidualR32T128,
    MmaResidualR64T128,
    MmaResidualR64T128Tail,
    // Request-invariant exact route: one sliced-K reduction (four K warps, linear order) at every
    // width, so a column's bits do not depend on how many columns share the call.
    SlicedColumnInvariant,
    // The q5-ksplit-mma fast path (LinearRoutes::Q5KSplit): eight K warps with the pairwise
    // reduction on the pipelined sliced-K kernel, a different FP32 summation order. KSplitW8 serves
    // the release route's 2..16 columns; KSplitColumnInvariant replaces SlicedColumnInvariant at
    // every width (one reduction, so still column-invariant).
    KSplitW8,
    KSplitColumnInvariant,
};

struct Q5LinearAddProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q5LinearAddPlan {
    Q5LinearAddScheduleId schedule;
    std::size_t workspace_bytes;
};

const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept;

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept;
// `ksplit`: LinearRoutes::Q5KSplit (the q5-ksplit-mma fast path).
Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem,
                                           bool column_invariant = false, bool ksplit = false);

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols,
                                                   bool column_invariant = false);

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream,
                                bool column_invariant = false, bool ksplit = false);
void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream, bool column_invariant = false,
                            bool ksplit = false);

} // namespace infernix::ops::detail
