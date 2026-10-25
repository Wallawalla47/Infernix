// The q5-ksplit-mma fast path (LinearRoutes::Q5KSplit): Q5 LinearAdd [5120,6144] and
// [5120,17408] with eight K warps. Each warp contracts one 64-wide group of every 512-wide K tile
// on the BF16 tensor cores and the CTA reduces the eight FP32 partials pairwise, the structure of
// NInfer's K-split residual kernel (giveen/ninfer-ext f7dabfe80, restored for 2..16 columns in
// e01b11645: 27B MTP K = 3 decode 143.7 -> 164.2 tok/s on his RTX 5090), here on the pipelined
// sliced-K kernel with its one-wave shape rule. Halving the K loop's tiles against the release
// four K warps changes the FP32 summation order, so it is a fast path (bits change, not the
// semantics). A column's bits do not depend on the call's width: every capacity runs the same
// reduction, and calls wider than 16 columns run 16-column token tiles.
#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "ops/linear/q5/q5_sliced_k_launch.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {

template <int Capacity>
using KSplitSchedule = Q5A16SlicedKMmaSchedule<16, Capacity, 8, 2, Cache::cg, Cache::ca, 2, 0,
                                               Capacity, Q5SlicedKReduction::Pairwise>;

template <int Capacity>
void launch_ksplit(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(residual.data);
    const auto stride = std::int64_t(residual.nb[1] / sizeof(__nv_bfloat16));
    launch_q5_a16_sliced_k_pipelined<KSplitSchedule<Capacity>>(
        q5_linear_operands(x, w), LinearBf16StridedOutput{data, stride, 0},
        LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}

} // namespace

void q5_linear_add_ksplit_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                 cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < 1) { throw std::invalid_argument("q5 linear_add K-split requires T >= 1"); }
    if (cols <= 8) {
        launch_ksplit<8>(x, w, residual, stream);
    } else {
        launch_ksplit<16>(x, w, residual, stream);
    }
}

} // namespace infernix::ops::detail
