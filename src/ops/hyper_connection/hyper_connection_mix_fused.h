#pragma once

// The fused route of hyper_connection_mix (docs/maintainer/qwen3_8-flash-next-design.md §8.3
// K1a/K1b, §19.3.3 Phase 1b) for Q8_G32_FP16 RowSplit and BF16 contiguous weights. Private to the Op.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// Columns the fused route serves in one call.
inline constexpr std::int32_t kHcMixFusedMaxColumns = 16;

// Whether the fused route serves this problem: both weights Q8_G32_FP16 RowSplit or both BF16
// contiguous, the registered geometry (4 streams of 2560, rank a multiple of 8 up to 512) and 1..16
// columns.
[[nodiscard]] bool hc_mix_fused_supported(const Weight& down, const Weight& up, std::int32_t hidden,
                                          std::int32_t streams, std::int32_t rank, std::int32_t columns) noexcept;

// FP32 split-K partials [S][down rows][T] and per-stream inverse RMS [S][T].
[[nodiscard]] std::size_t hc_mix_fused_workspace_bytes(std::int32_t down_rows, std::int32_t streams,
                                                       std::int32_t columns) noexcept;

// --a8-activations: x's 8-bit copy for a GGUF projection of x (ggml_linear_a8's column words, column t at words + t *
// stride; slice code sums with `sums`). Null words: none.
struct HcMixA8Columns {
    std::int32_t* words = nullptr;
    std::int32_t stride = 0;
    bool sums           = false;
};

// Whether the fused route writes x's 8-bit copy for this call: its two-or-more-column kernels (v4), whose norm
// weights are 16-byte aligned.
[[nodiscard]] bool hc_mix_fused_a8_supported(const Tensor& norm_weight, std::int32_t columns) noexcept;

// K1a then K1b. `partials` holds hc_mix_fused_workspace_bytes; inject (FP32 [S, T]) may be null; `a8` (words set)
// only where hc_mix_fused_a8_supported.
void hc_mix_fused(const Tensor& residual, const Tensor& norm_weight, const Weight& down, const Weight& up,
                  std::int32_t streams, std::int32_t rank, float eps, Tensor& x, Tensor* inject, void* partials,
                  cudaStream_t stream, HcMixA8Columns a8 = {});

} // namespace infernix::ops::detail
