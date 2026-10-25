// Qwen3.8-Flash-Next QSA query, gate, key, value and indexer projections, one MXFP8 parent with BF16
// activations (W8A16, NVIDIA FP4-Local). Every width runs the Tensor Core route, whose schedules
// accumulate a column in one order, so a column has the same bits at every T and with either producer
// (mxfp8_a16_tma_mma.cuh). 13,952 rows admit row tiles up to 128. The selection is
// bench/ops/mxfp8_flash_next_sweep.cu's (RTX 5090, 2026-10-20: the fastest schedule per width, merged
// within 2 %): cp.async to 192 columns, TMA from there.
#include "ops/linear/common/output.cuh"
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "ops/linear/mxfp8/mxfp8_shapes.h"
#include "ops/linear/f16acc/f16acc_mma.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"
#include "ops/linear/qsa_group.h"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<13952, 2560>;

// The selections into any output: linear's [13952,T], or the QSA group's six segments
// (linear_qsa_group). The output map changes only where a value is stored, so both give the same bits.
template <class Output>
void select_a16(const Mxfp8Operands& p, Output output, cudaStream_t stream) {
    const int tokens = p.tokens;
    if (tokens <= 4) return mxfp8_tma<Geometry, Mxfp8TmaT16R32>(p, output, stream);
    if (tokens <= 6) return mxfp8_tma<Geometry, Mxfp8P4<Mxfp8TmaT16R128>>(p, output, stream);
    if (tokens <= 16) return mxfp8_tma<Geometry, Mxfp8P2<Mxfp8TmaT16R128>>(p, output, stream);
    if (tokens <= 64) return mxfp8_tma<Geometry, Fp8A16RingT32R64S3>(p, output, stream);
    if (tokens <= 128) return mxfp8_tma<Geometry, Mxfp8P2<Mxfp8TmaT64R64>>(p, output, stream);
    if (tokens <= 192) return mxfp8_tma<Geometry, Mxfp8P4<Fp8A16RingT64R128>>(p, output, stream);
    if (tokens <= 256) return mxfp8_tma<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(p, output, stream);
    if (tokens <= 768) return mxfp8_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(p, output, stream);
    if (tokens <= 1024) return mxfp8_tma<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(p, output, stream);
    mxfp8_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(p, output, stream);
}

// f16-accumulate-prefill: the exact route's TMA tiles at the same widths (not yet swept for this route).
template <class Output>
void select_f16(const Tensor& x, const Weight& weight, Output output, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 768) return mxfp8_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(x, weight, output, scratch, stream);
    if (tokens <= 1024) return mxfp8_f16acc<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(x, weight, output, scratch, stream);
    mxfp8_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(x, weight, output, scratch, stream);
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    select_a16(mxfp8_operands(x, weight), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, stream);
}

void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    select_f16(x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, scratch, stream);
}

// fp8-a8-pair-block32: the activation as two E4M3 terms with per-32 UE8M0 scales, the weight's own group
// scales in the same block-scaled MMA (fp8_a8_pair_tma_mma.cuh); recipe C's pair tiles, not yet swept here.
void launch_pair_block32(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8PairBlock32Workspace scratch,
                         cudaStream_t stream) {
    fp8_a8_pair_block32_gemm<Geometry, true>(x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
                                             scratch, stream);
}
} // namespace

const Mxfp8LinearShape kMxfp8N13952K2560{13952, 2560, launch_a16, launch_f16, kMxfp8F16MinTokens, launch_pair_block32, kMxfp8PairMinTokensN13952};

// linear_qsa_group's MXFP8 route: mxfp8_dispatch's choice (f16-accumulate first, then the block32 pair,
// from their admissions; `routes` already masked under column invariance), into the six segments.
void mxfp8_qsa_group(const Tensor& x, const Weight& weight, const QsaGroupParts& parts, LinearRoutes routes,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    const LinearBf16SegmentedOutput<6144, 6144, 512, 512, 512, 128> output{
        {parts.data[0], parts.data[1], parts.data[2], parts.data[3], parts.data[4], parts.data[5]}};
    if (has_route(routes, LinearRoutes::F16Accumulate) && x.ne[1] >= kMxfp8F16MinTokens) {
        if (workspace == nullptr) {
            throw std::invalid_argument("mxfp8 qsa group: the f16-accumulate route requires caller workspace");
        }
        auto scope = workspace->scope();
        return select_f16(x, weight, output, allocate_f16_accumulate_workspace(*workspace, x.ne[1], weight.k), stream);
    }
    if (has_route(routes, LinearRoutes::Fp8A8PairBlock32) && x.ne[1] >= kMxfp8PairMinTokensN13952) {
        if (workspace == nullptr) {
            throw std::invalid_argument("mxfp8 qsa group: the block32 pair route requires caller workspace");
        }
        auto scope = workspace->scope();
        return fp8_a8_pair_block32_gemm<Geometry, true>(
            x, weight, output, allocate_fp8_a8_pair_block32_workspace(*workspace, x.ne[1], weight.k), stream);
    }
    select_a16(mxfp8_operands(x, weight), output, stream);
}
} // namespace infernix::ops::detail
