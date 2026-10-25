#pragma once

#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/linear.h"

namespace infernix::models::qwen4_exp::execution {

// The fast paths the Forward runs (models/qwen4_exp/program/fast_paths.h selects them): each field
// is false or Exact on the reproducible route (--reproducible).
struct ForwardFastPaths {
    // The dense projections' fast-route flags: q8-sliced-k (GDN q/k/v/z, GDN and QSA output),
    // fp8-a8-pair or fp8-a8-pair-block32 (recipe C's FP8 rows), ggml-bf16-prefill (GGUF imports) and
    // f16-accumulate-prefill (FP8 and MXFP8). Each flag reaches only the problems that register it; the
    // prefill flags only prefill calls (Forward::dense_routes).
    ops::LinearRoutes dense_routes = ops::LinearRoutes::Exact;
    // gdn-conv-fused: the GDN projection and its causal convolution in one kernel (FP32 current token).
    bool gdn_conv_fused = false;
    // qsa-prepare-fused: one QSA prepare kernel (norms, rotations, K/V and index-key appends).
    bool qsa_prepare_fused = false;
    // expert-a4-residual (2) or expert-a4-residual-3 (3): W4A16 experts' wide calls take the A4
    // residual route of that many NVFP4 terms per x and h row (ops::offloaded_moe::WideA16::kA4Residual,
    // kA4Residual3, design §19.3.23); 0 keeps them exact. No effect on W4A4 experts.
    int expert_a4_terms = 0;
    // expert-a4-residual-decode: W4A16 experts' narrow calls (at most eight columns per expert), on the
    // GPU and in the CPU expert engine, take two NVFP4 terms per x and h row with exact sums
    // (ops::offloaded_moe::NarrowA16::kA4Residual); false keeps them exact. No effect on W4A4 experts.
    bool expert_a4_narrow = false;
    // gdn-f16-accumulate: the chunked GDN calls' arithmetic.
    ops::GdnChunkArithmetic gdn_arithmetic = ops::GdnChunkArithmetic::Exact;
    // router-mma: prefill calls of at least kRouterMmaMinColumns take the tensor-core router.
    bool router_mma = false;
};

// router-mma's narrowest call: prefill chunks and walk calls, never decode or verification rounds.
inline constexpr std::int32_t kRouterMmaMinColumns = 256;

} // namespace infernix::models::qwen4_exp::execution
