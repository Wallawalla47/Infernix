#pragma once

#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/linear.h"

namespace infernix::models::qwen4_exp::execution {

// The fast paths the Forward runs (models/qwen4_exp/program/fast_paths.h selects them): each field
// is false or Exact on the reproducible route (--reproducible).
struct ForwardFastPaths {
    // The dense projections' fast-route flags: q8-sliced-k (GDN q/k/v/z, GDN and QSA output) and
    // fp8-a8-pair (recipe C's FP8 rows). Each flag reaches only the problems that register it.
    ops::LinearRoutes dense_routes = ops::LinearRoutes::Exact;
    // gdn-conv-fused: the GDN projection and its causal convolution in one kernel (FP32 current token).
    bool gdn_conv_fused = false;
    // qsa-prepare-fused: one QSA prepare kernel (norms, rotations, K/V and index-key appends).
    bool qsa_prepare_fused = false;
    // expert-a4-residual (2) or expert-a4-residual-3 (3): W4A16 experts' wide calls take the A4
    // residual route of that many NVFP4 terms per x and h row (ops::offloaded_moe::WideA16::kA4Residual,
    // kA4Residual3, design §19.3.23); 0 keeps them exact. No effect on W4A4 experts.
    int expert_a4_terms = 0;
    // gdn-f16-accumulate: the chunked GDN calls' arithmetic.
    ops::GdnChunkArithmetic gdn_arithmetic = ops::GdnChunkArithmetic::Exact;
};

} // namespace infernix::models::qwen4_exp::execution
