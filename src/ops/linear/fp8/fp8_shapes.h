#pragma once
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

namespace infernix::ops::detail {
struct Fp8LinearShape {
    std::int32_t n, k;
    Fp8Launch a16;
    void (*a8)(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t);
    bool (*uses_a8)(std::int32_t min_tokens, std::int32_t max_tokens);
    std::size_t (*partial_capacity_bytes)(std::int32_t max_tokens) = nullptr;
    // A16ColumnInvariant: one schedule for every T; null where the shape registers none.
    Fp8Launch invariant = nullptr;
    // fp8-a8-pair (LinearRoutes::Fp8A8Pair): the two-term E4M3 activation route, selected at
    // T >= pair_min_tokens when the flag is set and no single-term A8 route is selected; null
    // where the problem registers none.
    void (*pair)(const Tensor&, const Weight&, Tensor&, Fp8A8PairWorkspace, cudaStream_t) = nullptr;
    std::int32_t pair_min_tokens = 0;
};

extern const Fp8LinearShape kFp8N14336K5120;
extern const Fp8LinearShape kFp8N16384K5120;
extern const Fp8LinearShape kFp8N34816K5120;
extern const Fp8LinearShape kFp8N5120K6144;
extern const Fp8LinearShape kFp8N5120K17408;
extern const Fp8LinearShape kFp8N248320K5120;
// Qwen3.8-Flash-Next (recipe C, W8A16: A16 routes, plus the fp8-a8-pair route).
extern const Fp8LinearShape kFp8N16384K2560;
extern const Fp8LinearShape kFp8N13312K2560;
extern const Fp8LinearShape kFp8N2560K6144;
extern const Fp8LinearShape kFp8N1280K2560;
extern const Fp8LinearShape kFp8N2560K640;
} // namespace infernix::ops::detail
