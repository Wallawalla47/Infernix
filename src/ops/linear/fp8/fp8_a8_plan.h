#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {

struct Fp8A8Workspace {
    std::uint8_t* codes = nullptr;
    float* scales       = nullptr;
    float* partials     = nullptr;
    // The A8 residual profile: each token's E4M3 encoding of what `codes` left, scaled by 16
    // relative to its row scale (fp8_a8_quantize writes it when it is not null).
    std::uint8_t* residual = nullptr;
};

inline std::size_t fp8_a8_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 A8 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("fp8 A8 workspace size overflow");
    }
    return count * bytes_per_token;
}

template <class Arena>
Fp8A8Workspace allocate_fp8_a8_workspace(Arena& arena, std::int32_t tokens, std::int32_t input_rows,
                                         std::size_t partial_bytes = 0, bool residual = false) {
    if (input_rows <= 0 || (input_rows % 32) != 0) {
        throw std::invalid_argument("fp8 A8 workspace: invalid K");
    }
    const std::size_t code_bytes =
        fp8_a8_checked_bytes(tokens, static_cast<std::size_t>(input_rows));
    const DeviceSpan codes  = arena.alloc_bytes(code_bytes, 256);
    const DeviceSpan scales = arena.alloc_bytes(fp8_a8_checked_bytes(tokens, sizeof(float)), 256);
    float* partials =
        partial_bytes ? static_cast<float*>(arena.alloc_bytes(partial_bytes, 256).data) : nullptr;
    std::uint8_t* residual_codes =
        residual ? static_cast<std::uint8_t*>(arena.alloc_bytes(code_bytes, 256).data) : nullptr;
    return {static_cast<std::uint8_t*>(codes.data), static_cast<float*>(scales.data), partials,
            residual_codes};
}

inline std::size_t fp8_a8_workspace_capacity_bytes(std::int32_t tokens, std::int32_t input_rows,
                                                   std::size_t partial_bytes = 0,
                                                   bool residual = false) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fp8_a8_workspace(layout, tokens, input_rows, partial_bytes, residual);
    return layout.peak_bytes(1);
}

void launch_fp8_a8_quantize(const Tensor& x, const Weight& weight, Fp8A8Workspace workspace,
                            cudaStream_t stream);

// fp8-a8-pair (LinearRoutes::Fp8A8Pair): each activation column as two E4M3 terms with their own
// power-of-two per-column scales, x = scales_hi * codes_hi + scales_lo * codes_lo. The high term
// scales the column's maximum to at most 448 by a power of two; the low term quantizes the high
// term's exact residual x - scales_hi * codes_hi the same way. A BF16 value has eight significant
// bits, four in each term, so the pair is exact for every value within 2^14 of its column maximum.
struct Fp8A8PairWorkspace {
    std::uint8_t* codes_hi = nullptr;
    float* scales_hi       = nullptr;
    std::uint8_t* codes_lo = nullptr;
    float* scales_lo       = nullptr;
};

template <class Arena>
Fp8A8PairWorkspace allocate_fp8_a8_pair_workspace(Arena& arena, std::int32_t tokens,
                                                  std::int32_t input_rows) {
    if (input_rows <= 0 || (input_rows % 32) != 0) {
        throw std::invalid_argument("fp8 A8 pair workspace: invalid K");
    }
    const std::size_t code_bytes =
        fp8_a8_checked_bytes(tokens, static_cast<std::size_t>(input_rows));
    const std::size_t scale_bytes = fp8_a8_checked_bytes(tokens, sizeof(float));
    const DeviceSpan codes_hi     = arena.alloc_bytes(code_bytes, 256);
    const DeviceSpan scales_hi    = arena.alloc_bytes(scale_bytes, 256);
    const DeviceSpan codes_lo     = arena.alloc_bytes(code_bytes, 256);
    const DeviceSpan scales_lo    = arena.alloc_bytes(scale_bytes, 256);
    return {static_cast<std::uint8_t*>(codes_hi.data), static_cast<float*>(scales_hi.data),
            static_cast<std::uint8_t*>(codes_lo.data), static_cast<float*>(scales_lo.data)};
}

inline std::size_t fp8_a8_pair_workspace_capacity_bytes(std::int32_t tokens,
                                                        std::int32_t input_rows) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fp8_a8_pair_workspace(layout, tokens, input_rows);
    return layout.peak_bytes(1);
}

// Writes both terms of every column of x (K = weight.k) in one pass.
void launch_fp8_a8_pair_quantize(const Tensor& x, const Weight& weight,
                                 Fp8A8PairWorkspace workspace, cudaStream_t stream);


} // namespace infernix::ops::detail
