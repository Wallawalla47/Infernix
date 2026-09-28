#pragma once

// The fast NVFP4 prompt kernel: each warp keeps 16 query rows, scores and output in registers, QK
// runs on block-scaled FP4 Tensor Cores, V decodes in registers, and a launch whose row blocks
// alone would leave SMs idle splits every row block's keys across CTAs (fast_tiled_mma.cuh).

#include "core/arena.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"

namespace ninfer::ops::detail {

void nvfp4_kv_fast_tiled_attention(const CausalAttentionOperands&, Nvfp4KvReadView,
                                   WorkspaceArena&, cudaStream_t);

} // namespace ninfer::ops::detail
