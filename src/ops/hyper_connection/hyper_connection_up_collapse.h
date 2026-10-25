#pragma once

// The prefill route of hyper_connection_mix's up projection and collapse for Q8_G32_FP16 RowSplit
// or BF16 W_up (docs/maintainer/qwen3_8-flash-next-design.md §19.3.23): one Q8 (or BF16) MMA tile per
// (16 features x 4 streams, token block) computes u = W_up m and collapses it in the tile's epilogue,
// so u never reaches global memory. Each u value accumulates over K in the order every MMA tile of
// its format uses and is rounded to BF16 before the collapse, as the composed route's linear +
// collapse do: the result is bit-identical to theirs. Private to the Op.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// Whether the fused up + collapse serves this call: W_up Q8_G32_FP16 RowSplit, or BF16 with rank a
// multiple of 64, [streams * hidden, rank], 4 streams, hidden a multiple of 16, and more than 96
// columns (narrower calls are as fast unfused).
[[nodiscard]] bool hc_up_collapse_supported(const Weight& up, std::int32_t hidden, std::int32_t streams,
                                            std::int32_t columns) noexcept;

// Whether hyper_connection_mix takes the fused route for a BF16 W_up (a temporary switch until the
// prefill A/B; the Q8 route is always taken).
[[nodiscard]] bool hc_bf16_up_collapse_enabled() noexcept;

// x[d, t] = bf16((1/S) * sum_s sigmoid(bf16((W_up m)[s*H + d, t])) * normalized[s*H + d, t]), the sum in
// stream order. m: BF16 [rank, T]; normalized: BF16 [S*H, T]; x: BF16 [H, T].
void hc_up_collapse(const Tensor& m, const Weight& up, const Tensor& normalized, std::int32_t streams, Tensor& x,
                    cudaStream_t stream);

// The composed route's collapse: x from u = W_up m (BF16 [S*H, T]) and normalized, the same expression
// as hc_up_collapse's epilogue.
void hc_collapse(const Tensor& u, const Tensor& normalized, std::int32_t streams, Tensor& x, cudaStream_t stream);

} // namespace infernix::ops::detail
