#pragma once

#include "ops/linear/bf16/bf16_launch.h"

namespace infernix::ops::detail {

[[nodiscard]] Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n256_k5120(std::int32_t tokens);
// Vision tower: patch embedding, fused QKV, attention output, MLP fc1/fc2, merger fc1, and the
// merger fc2 of the 2560- (Qwen3.8-Flash-Next) and 5120-wide (Qwen3.8 27B, e.g. Quasar) text
// models. T <= 8 keeps the fallback's skinny GEMV below; larger T run the BF16 TMA MMA template,
// in its tail-capable form where N or K is 4304.
[[nodiscard]] Bf16Launch select_bf16_n1152_k1536(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n3456_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1152_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n4304_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1152_k4304(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n4608_k4608(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2560_k4608(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k4608(std::int32_t tokens);
// Qwen3.8-Flash-Next text projections stored BF16: the QSA query/gate/key/value/index group and the
// GDN a/b projections (prefill; T <= 8 keeps the skinny GEMV).
[[nodiscard]] Bf16Launch select_bf16_n13952_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n96_k2560(std::int32_t tokens);

// Runtime-shape GEMM fallback for any (n, k) the table above does not specialise.
[[nodiscard]] Bf16Launch select_bf16_general_launch(std::int32_t tokens);

} // namespace infernix::ops::detail
