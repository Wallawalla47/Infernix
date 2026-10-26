#pragma once
#include "ops/linear/f16acc/f16acc_plan.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

namespace infernix::ops::detail {
using Fp8F16Launch = void (*)(const Tensor&, const Weight&, Tensor&, F16AccumulateWorkspace, cudaStream_t);

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
    // fp8-a8-pair-block32 (LinearRoutes::Fp8A8PairBlock32): the two-term route with per-32-value
    // block scales, admitted where the column-scaled pair is (pair_min_tokens) and taking precedence
    // over it; null where the problem registers none.
    void (*pair_block32)(const Tensor&, const Weight&, Tensor&, Fp8A8PairBlock32Workspace,
                         cudaStream_t) = nullptr;
    // f16-accumulate-prefill (LinearRoutes::F16Accumulate, ops/linear/f16acc/f16acc_plan.h): FP16
    // Tensor Cores with FP16 chunk sums and an FP32 carry, selected at T >= f16_min_tokens when the flag
    // is set and no single-term A8 route is selected; it takes precedence over both pair routes. Null
    // where the problem registers none.
    Fp8F16Launch f16 = nullptr;
    std::int32_t f16_min_tokens = 0;
};

// The f16-accumulate route's admission on every problem that registers it: prefill widths, where the
// encoder's extra pass over x is small beside the GEMM. Not yet timed: the op bench (linear_bench
// --routes f16) sets it per problem, as each kPairMinTokens was set.
inline constexpr std::int32_t kFp8F16MinTokens = 256;

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
// Qwen3.5-122B-A10B fp8_e4m3fn_row_f32 problems (dealignai's Ornith-1.5-122B-UNCENSORED-NVFP4, W8A16):
// the shared expert's down projection and the vocabulary head.
extern const Fp8LinearShape kFp8F32N3072K1024;
extern const Fp8LinearShape kFp8F32N248320K3072;
} // namespace infernix::ops::detail
