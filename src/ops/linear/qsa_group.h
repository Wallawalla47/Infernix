#pragma once

// linear_qsa_group's format routes (include/infernix/ops/linear.h): the [13952,2560] QSA group projection
// of Qwen3.8-Flash-Next into its six row segments, on the routes linear() selects for that problem.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// The segments' rows: query, gate, key, value, index query, index key.
inline constexpr std::int32_t kQsaGroupRows[kQsaGroupSegments] = {6144, 6144, 512, 512, 512, 128};

struct QsaGroupParts {
    __nv_bfloat16* data[kQsaGroupSegments];
};

// Q8 (Dense8): the exact Q8 routes of [13952,2560], or LinearRoutes::F16Accumulate's from its admission
// (needs `workspace`).
void q8_qsa_group(const Tensor& x, const Weight& w, const QsaGroupParts& parts, LinearRoutes routes,
                  WorkspaceArena* workspace, cudaStream_t stream);
// The f16-accumulate route of q8_qsa_group (ops/linear/q8/q8_f16acc.cu).
void q8_qsa_group_f16acc(const Tensor& x, const Weight& w, const QsaGroupParts& parts, WorkspaceArena& workspace,
                         cudaStream_t stream);
// MXFP8 (FP4-Local): mxfp8_dispatch's routes of [13952,2560] (A16, f16-accumulate, block32 pair).
void mxfp8_qsa_group(const Tensor& x, const Weight& w, const QsaGroupParts& parts, LinearRoutes routes,
                     WorkspaceArena* workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
