#pragma once

// infernix::ops::detail - The bf16_p12 Linear routes registered per shape (bf16_p12_n*_k*.cu), from the
// bench/ops/bf16_p12_sweep.cu sweep on an RTX 5090 (2026-11-01, recipe A's tensors): the packed skinny GEMV, the
// register-decode stream MMA (bf16_p12_stream.cuh) and the packed TMA MMA (bf16_p12_tma.cuh), each taken over the
// widths where it is the fastest packed route.

#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"

#include <cstdint>

namespace infernix::ops::detail {

// Qwen3.8-Flash-Next (recipe A): the GDN query, key, value and z projections [16384, 2560], the QSA query, gate,
// key, value and index projections [13952, 2560] and the GDN and QSA output projections [2560, 6144].
[[nodiscard]] P12Launch select_bf16_p12_n16384_k2560(std::int32_t tokens);
[[nodiscard]] P12Launch select_bf16_p12_n13952_k2560(std::int32_t tokens);
[[nodiscard]] P12Launch select_bf16_p12_n2560_k6144(std::int32_t tokens);

// The registered route of a [n, k] call of `tokens` columns; null for a shape without one.
[[nodiscard]] P12Launch select_bf16_p12_launch(std::int32_t n, std::int32_t k, std::int32_t tokens);

} // namespace infernix::ops::detail
