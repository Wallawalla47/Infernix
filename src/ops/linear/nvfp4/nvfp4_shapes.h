#pragma once
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

namespace infernix::ops::detail {
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
    // A16ColumnInvariant: a column's arithmetic independent of the call's other columns (null: none).
    Nvfp4Launch invariant = nullptr;
    // Writes FP32 as well as BF16 outputs (a vocabulary head's logits).
    bool fp32_output = false;
};

extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
// The 27B output head (NVIDIA's NVFP4-QAD artifact): FP32 or BF16 logits, A16 only.
extern const Nvfp4LinearShape kNvfp4N248320K5120;
// A DSpine drafter's NVFP4 projections beyond the target's shapes (the 27B's: attention output, feature
// projection), A16 only.
extern const Nvfp4LinearShape kNvfp4N5120K4096;
extern const Nvfp4LinearShape kNvfp4N5120K25600;
} // namespace infernix::ops::detail
