#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/f16acc/f16acc_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

using Mxfp8Launch    = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
using Mxfp8F16Launch = void (*)(const Tensor&, const Weight&, Tensor&, F16AccumulateWorkspace, cudaStream_t);
using Mxfp8PairLaunch = void (*)(const Tensor&, const Weight&, Tensor&, Fp8A8PairBlock32Workspace, cudaStream_t);

// One registered MXFP8 problem: its exact (A16) route for every width, column-invariant at every T,
// the f16-accumulate-prefill route (LinearRoutes::F16Accumulate, ops/linear/f16acc/f16acc_plan.h)
// from f16_min_tokens, and fp8-a8-pair-block32's (LinearRoutes::Fp8A8PairBlock32: the activation as two
// E4M3 terms with UE8M0 scales per 32 values, the weight's own group scales in the same block-scaled MMA,
// fp8_a8_pair_tma_mma.cuh) from pair_min_tokens; f16-accumulate takes precedence where both are set.
struct Mxfp8LinearShape {
    std::int32_t n, k;
    Mxfp8Launch a16;
    Mxfp8F16Launch f16           = nullptr;
    std::int32_t f16_min_tokens  = 0;
    Mxfp8PairLaunch pair_block32 = nullptr;
    std::int32_t pair_min_tokens = 0;
};

// The f16-accumulate route's admission (prefill widths). Not yet timed: the op bench (linear_bench
// --qtype mxfp8 --routes f16) sets it per problem.
inline constexpr std::int32_t kMxfp8F16MinTokens = 256;

// The block32 pair's admission per problem: recipe C's measured column-scaled pair admissions for the
// same [N,K] (fp8/shapes, kPairMinTokens; [13952,2560] takes [13312,2560]'s). Not yet timed on MXFP8, whose
// A16 route is a BF16 MMA rather than recipe C's A16 one: the op bench (linear_bench --qtype mxfp8
// --routes block32) sets them.
inline constexpr std::int32_t kMxfp8PairMinTokensN16384 = 33;
inline constexpr std::int32_t kMxfp8PairMinTokensN13952 = 97;
inline constexpr std::int32_t kMxfp8PairMinTokensN2560  = 513;
inline constexpr std::int32_t kMxfp8PairMinTokensN1280  = 1025;

// Qwen3.8-Flash-Next (NVIDIA FP4-Local, W8A16): the GDN q/k/v/z, QSA query/gate/key/value with the
// indexer, GDN and QSA output and shared-expert gate/up and down projections.
extern const Mxfp8LinearShape kMxfp8N16384K2560;
extern const Mxfp8LinearShape kMxfp8N13952K2560;
extern const Mxfp8LinearShape kMxfp8N2560K6144;
extern const Mxfp8LinearShape kMxfp8N1280K2560;
extern const Mxfp8LinearShape kMxfp8N2560K640;

} // namespace infernix::ops::detail
