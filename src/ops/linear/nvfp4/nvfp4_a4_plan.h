#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {

struct Nvfp4A4Workspace {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    // Extent of the scale plane. The tiled layout writes over whole tiles, so it reaches past the
    // real token count; carrying the size lets the quantizer check that rather than trust that the
    // caller allocated through allocate_nvfp4_a4_workspace.
    std::size_t scale_bytes = 0;
    // The A4 residual profile stores `terms` row-major planes, plane p at codes + p * code_plane and
    // scales + p * scale_plane; A4 has one.
    int terms               = 1;
    std::size_t code_plane  = 0;
    std::size_t scale_plane = 0;
};

// LinearPolicy::AllowA4Residual's term count. Two terms passed the decode gate with DFlash2 and
// n-gram copies on the official 27B (TV excess 4.2e-5 against a 2.4e-3 budget, INT8 KV), and run
// 1,787 / 1,217 / 887 tok/s there at one, two and four requests against three terms' 1,696 /
// 1,074 / 783 (RTX 5090, 2026-10-14): a third term costs time the gate does not need.
inline constexpr int kNvfp4A4ResidualTerms = 2;

inline std::size_t nvfp4_a4_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 A4 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("nvfp4 A4 workspace size overflow");
    }
    return count * bytes_per_token;
}

template <class Arena>
Nvfp4A4Workspace allocate_nvfp4_a4_workspace(Arena& arena, std::int32_t tokens,
                                             std::int32_t input_rows, int terms = 1) {
    if (input_rows <= 0 || (input_rows % 64) != 0) {
        throw std::invalid_argument("nvfp4 A4 workspace: invalid K");
    }
    if (terms != 1 && terms != kNvfp4A4ResidualTerms) {
        throw std::invalid_argument("nvfp4 A4 workspace: invalid activation planes");
    }
    const std::size_t code_bytes =
        nvfp4_a4_checked_bytes(tokens, static_cast<std::size_t>(input_rows) / 2);
    if (terms > 1) {
        // Residual scale planes are allocated over whole tiles like the one-term plane, so either
        // the row-major (MMA) or the 128-token tiled (TMA) layout fits. Each plane's extent is a
        // multiple of 16 bytes for every K that is a multiple of 256.
        const std::size_t scale_plane = nvfp4_a4_checked_bytes(
            nvfp4_a4_padded_tokens(tokens), static_cast<std::size_t>(input_rows) / 16);
        const DeviceSpan codes  = arena.alloc_bytes(code_bytes * terms, 256);
        const DeviceSpan scales = arena.alloc_bytes(scale_plane * terms, 256);
        return {static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(scales.data),
                scale_plane * terms, terms, code_bytes, scale_plane};
    }
    // The tiled layout addresses whole tiles, so the scale plane is allocated for the padded token
    // count: at most 255 tokens of scales, under 0.3 MiB on the widest registered K, and paid
    // whichever layout the quantizer then writes.
    const std::size_t scale_bytes = nvfp4_a4_checked_bytes(
        nvfp4_a4_padded_tokens(tokens), static_cast<std::size_t>(input_rows) / 16);
    const DeviceSpan codes  = arena.alloc_bytes(code_bytes, 256);
    const DeviceSpan scales = arena.alloc_bytes(scale_bytes, 256);
    return {static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(scales.data),
            scale_bytes};
}

inline std::size_t nvfp4_a4_workspace_capacity_bytes(std::int32_t tokens, std::int32_t input_rows,
                                                     int terms = 1) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_nvfp4_a4_workspace(layout, tokens, input_rows, terms);
    return layout.peak_bytes(1);
}

// reciprocal_quotient selects the reciprocal-multiply quotient formulation. It is measured
// faster only on latency-bound small grids; callers opt in explicitly per measured route and
// the default keeps historical division behavior. A workspace with more than one term is quantized
// as the A4 residual profile (row-major only): plane 0 holds the A4 encoding, and each later plane
// the NVFP4 encoding of what the earlier ones left.
void launch_nvfp4_a4_quantize(const Tensor& x, const Weight& weight, Nvfp4A4Workspace workspace,
                              Nvfp4ScaleLayout layout, cudaStream_t stream,
                              bool reciprocal_quotient = false);


} // namespace infernix::ops::detail
