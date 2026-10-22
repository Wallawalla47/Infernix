#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

void bf16_gdn_norm_gating_proj_27_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                         Tensor& h, const Weight& a_weight, const Weight& b_weight,
                                         const Tensor& alog, const Tensor& bias, Tensor& g,
                                         Tensor& beta, bool column_invariant,
                                         cudaStream_t stream);
// The 35B-A3B control [64,2048] (32 heads): the column-invariant fused route, 256 threads at every T.
void bf16_gdn_norm_gating_proj_35_invariant_launch(const Tensor& x, const Tensor& norm_weight,
                                                   float eps, Tensor& h, const Weight& a_weight,
                                                   const Weight& b_weight, const Tensor& alog,
                                                   const Tensor& bias, Tensor& g, Tensor& beta,
                                                   cudaStream_t stream);

// Qwen3.5-122B-A10B's control [128,3072] (64 heads): the fused SIMT norm and control, 384 threads at
// every T (one 8-element load per thread), so a column's sums are column-invariant at every width.
void bf16_gdn_norm_gating_proj_122_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                          Tensor& h, const Weight& a_weight, const Weight& b_weight,
                                          const Tensor& alog, const Tensor& bias, Tensor& g,
                                          Tensor& beta, cudaStream_t stream);

enum class Bf16GdnGatingTokenVariant {
    None,
    Full,
    Predicated,
};

void bf16_gdn_gating_proj_gemv_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, const Tensor& A_log,
                                      const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                      cudaStream_t stream);
void bf16_gdn_gating_proj_small_t_split10_launch(const Tensor& x, const Weight& a_weight,
                                                 const Weight& b_weight, const Tensor& A_log,
                                                 const Tensor& dt_bias, void* workspace,
                                                 std::size_t workspace_bytes, Tensor& g,
                                                 Tensor& beta, cudaStream_t stream);

// Cooperative launchers return false without submitting work only when the selected device cannot
// make one complete token tile resident. The Op wrapper owns the non-cooperative fallback.
[[nodiscard]] bool bf16_gdn_gating_proj_mma_split8_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_mma_split4_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_mma_split2_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream);

void bf16_gdn_gating_proj_35_simt_c4_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_35_simt_c8_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split32_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_norm_gating_proj_35_mma_split32_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Tensor& norm_weight, float eps,
    Tensor& h, const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
    const Tensor& dt_bias, void* workspace, Tensor& g, Tensor& beta,
    std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split16_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split8_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split4_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_35_mma_split2_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
// Qwen3.5-122B-A10B's MMA control routes: split_k 16, 8, 4 or 2 (cooperative; false when a token tile
// cannot be resident) or 1 (unsplit, always true).
[[nodiscard]] bool bf16_gdn_gating_proj_122_mma_launch(
    std::int32_t split_k, Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
void bf16_gdn_gating_proj_35_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace infernix::ops::detail
