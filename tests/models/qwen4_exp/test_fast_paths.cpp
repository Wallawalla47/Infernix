// Qwen3.8-Flash-Next fast-path selection (models/qwen4_exp/program/fast_paths.h): where each fast
// path's kernels run, which configurations the registry qualifies it for, and how a configuration is
// described. The selection itself (select_fast_paths) needs loaded parameters; the startup log and
// the quality gate's runs cover it.

#include "models/qwen4_exp/program/fast_paths.h"

#include <iostream>
#include <string>

namespace {

using infernix::FastPath;
using infernix::KvCacheStorage;
using infernix::QType;
using infernix::models::qwen4_exp::DenseKernels;
using infernix::models::qwen4_exp::FastPathConfiguration;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    namespace q = infernix::models::qwen4_exp;
    int failures = 0;

    const infernix::EngineOptions options;
    const FastPathConfiguration dense8{DenseKernels::Q8, true, KvCacheStorage::Int8Group64};
    failures += check(q::fast_path_applies(FastPath::Q8SlicedK, dense8, options), "q8-sliced-k runs on Q8 dense weights");
    for (const DenseKernels dense : {DenseKernels::Bf16, DenseKernels::Fp8Row, DenseKernels::Ggml}) {
        failures += check(!q::fast_path_applies(FastPath::Q8SlicedK, {dense, true, KvCacheStorage::Int8Group64}, options),
                          "q8-sliced-k has no kernels for other dense formats");
    }
    // The Qwen3.8-27B attention paths have no Flash-Next kernels.
    failures += check(!q::fast_path_applies(FastPath::PromptPv8, dense8, options), "prompt-pv8 does not apply");
    failures += check(!q::fast_path_applies(FastPath::DecodeSplit64, dense8, options), "decode-split64 does not apply");

    failures += check(q::describe(dense8) == "Q8 dense, 4-bit expert activations, int8 KV",
                      "configuration description changed");
    failures += check(q::describe({DenseKernels::Fp8Row, false, KvCacheStorage::BFloat16}) ==
                          "FP8 dense, 16-bit expert activations, bf16 KV",
                      "configuration description changed");
    // Every GGUF block format is one dense kernel family, so every GGUF import shares a configuration.
    for (const QType qtype : {QType::GGML_Q4_K, QType::GGML_Q6_K, QType::GGML_IQ3_S, QType::GGML_IQ4_XS}) {
        failures += check(q::dense_kernels(qtype) == DenseKernels::Ggml, "a GGUF block format is not the GGUF family");
    }
    failures += check(q::dense_kernels(QType::Q8_G32_FP16) == DenseKernels::Q8 &&
                          q::dense_kernels(QType::FP8_E4M3FN_ROW_BF16) == DenseKernels::Fp8Row &&
                          q::dense_kernels(QType::BF16) == DenseKernels::Bf16,
                      "a dense format maps to the wrong kernel family");
    failures += check(q::describe({DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}) ==
                          "GGUF dense, 16-bit expert activations, int8 KV",
                      "GGUF configuration description changed");

    failures += check(q::fast_path_applies(FastPath::GdnConvFused, dense8, options), "gdn-conv-fused runs on Q8 dense weights");
    failures += check(!q::fast_path_applies(FastPath::GdnConvFused, {DenseKernels::Bf16, true, KvCacheStorage::Int8Group64}, options),
                      "gdn-conv-fused has no kernels for BF16 dense weights");
    failures += check(q::fast_path_applies(FastPath::QsaPrepareFused, dense8, options), "qsa-prepare-fused runs on INT8 KV");
    failures += check(!q::fast_path_applies(FastPath::QsaPrepareFused, {DenseKernels::Q8, true, KvCacheStorage::Vq2}, options),
                      "qsa-prepare-fused has no VQ2 kernel");

    // The kernel paths are qualified with Q8 dense weights and 4-bit experts on every KV format their
    // kernels serve: qsa-prepare-fused has none for VQ2 or K4V2.
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                                    KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        const FastPathConfiguration config{DenseKernels::Q8, true, kv};
        failures += check(q::fast_path_qualified(FastPath::Q8SlicedK, config) &&
                              q::fast_path_qualified(FastPath::GdnConvFused, config),
                          "the dense kernel paths are qualified on every KV format");
        failures += check(q::fast_path_qualified(FastPath::QsaPrepareFused, config) ==
                              q::fast_path_applies(FastPath::QsaPrepareFused, config, options),
                          "qsa-prepare-fused is qualified exactly where its kernel runs");
    }
    for (const FastPath path : {FastPath::Q8SlicedK, FastPath::GdnConvFused, FastPath::QsaPrepareFused}) {
        failures += check(!q::fast_path_qualified(path, {DenseKernels::Q8, false, KvCacheStorage::Int8Group64}),
                          "a kernel path stays unqualified with 16-bit expert activations");
    }
    // fp8-a8-pair runs on recipe C (FP8 dense, 16-bit expert activations) where its own decode gate
    // passed; INT8 fails its tail and BF16 its prefill gate.
    for (const KvCacheStorage kv : {KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                    KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value})
        failures += check(q::fast_path_qualified(FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, false, kv}),
                          "fp8-a8-pair is qualified on recipe C with every KV format but INT8 and BF16");
    for (const KvCacheStorage kv : {KvCacheStorage::Int8Group64, KvCacheStorage::BFloat16})
        failures += check(!q::fast_path_qualified(FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, false, kv}),
                          "fp8-a8-pair stays unqualified on recipe C with INT8 or BF16 KV");
    failures += check(!q::fast_path_qualified(FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, true, KvCacheStorage::Int8Group64}),
                      "fp8-a8-pair is not qualified with 4-bit expert activations");

    // The request-coupling paths are qualified with Q8 dense weights and 4-bit experts on every KV
    // format: speculation and cache resume by their joint arm, preemption by its increment arm.
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                                    KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        const FastPathConfiguration config{DenseKernels::Q8, true, kv};
        for (const FastPath path : {FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume, FastPath::Preemption}) {
            failures += check(q::fast_path_qualified(path, config),
                              "the request-coupling paths are qualified on every KV format");
            failures += check(!q::fast_path_qualified(path, {DenseKernels::Fp8Row, false, kv}),
                              "the request-coupling paths are not qualified on recipe C");
        }
    }

    q::FastPathSelection selection{.configuration = dense8, .paths = {}};
    const auto exact = selection.forward();
    failures += check(exact.dense_routes == infernix::ops::LinearRoutes::Exact && !exact.gdn_conv_fused &&
                          !exact.qsa_prepare_fused,
                      "the exact route without a selected path");
    selection.paths = {FastPath::Q8SlicedK, FastPath::GdnConvFused, FastPath::QsaPrepareFused};
    const auto fast = selection.forward();
    failures += check(fast.dense_routes == infernix::ops::LinearRoutes::Q8SlicedK && fast.gdn_conv_fused &&
                          fast.qsa_prepare_fused,
                      "each selected path sets its Forward flag");
    // q8-sliced-k's reduction order depends on the round's width; the fused kernels are per column.
    failures += check(!selection.request_invariant(), "q8-sliced-k couples a request to its round");
    selection.paths = {FastPath::GdnConvFused, FastPath::QsaPrepareFused};
    failures += check(selection.request_invariant(), "the fused kernels keep a request invariant");

    // expert-a4-residual acts only on W4A16 experts (recipe C), and routes an expert by its column
    // count in the call, which batched rounds share between requests.
    const FastPathConfiguration recipe_c{DenseKernels::Fp8Row, false, KvCacheStorage::Int8Group64};
    failures += check(q::fast_path_applies(FastPath::ExpertA4Residual, recipe_c, options),
                      "expert-a4-residual runs on W4A16 experts");
    failures += check(!q::fast_path_applies(FastPath::ExpertA4Residual, dense8, options),
                      "expert-a4-residual does not apply to W4A4 experts");
    for (const DenseKernels dense : {DenseKernels::Ggml, DenseKernels::Bf16}) {
        failures += check(!q::fast_path_applies(FastPath::ExpertA4Residual, {dense, false, KvCacheStorage::Int8Group64}, options),
                          "expert-a4-residual has no kernel for GGUF or block-FP8 experts");
    }
    failures += check(!exact.expert_a4_residual, "the exact route keeps W4A16 experts exact");
    selection.paths = {FastPath::ExpertA4Residual};
    failures += check(selection.forward().expert_a4_residual, "expert-a4-residual sets its Forward flag");
    failures += check(!selection.request_invariant(), "expert-a4-residual couples a request to its round");
    // fp8-a8-pair admits a dense call by its width, which batched rounds share.
    selection.paths = {FastPath::Fp8A8Pair};
    failures += check(!selection.request_invariant(), "fp8-a8-pair couples a request to its round");
    selection.paths = {FastPath::ExpertA4Residual};
    for (const auto kv : {KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::ExpertA4Residual, {DenseKernels::Fp8Row, false, kv}),
                          "expert-a4-residual is qualified on recipe C with every quantized KV format");
    }
    failures += check(!q::fast_path_qualified(FastPath::ExpertA4Residual,
                                              {DenseKernels::Fp8Row, false, KvCacheStorage::BFloat16}),
                      "expert-a4-residual fails on BF16 KV, where recipe C's exact route is the reference");
    failures += check(!q::fast_path_qualified(FastPath::ExpertA4Residual,
                                              {DenseKernels::Q8, false, KvCacheStorage::Int8Group64}),
                      "expert-a4-residual is qualified only on the dense format it was gated with");

    // sampled-mtp-drafts acts on the MTP drafter only; gdn-f16-accumulate on every artifact's GDN
    // prefill, per call, so it keeps a request invariant.
    infernix::EngineOptions mtp = options;
    mtp.speculative.backend     = infernix::SpeculativeBackend::Mtp;
    failures += check(q::fast_path_applies(FastPath::SampledMtpDrafts, dense8, mtp) &&
                          !q::fast_path_applies(FastPath::SampledMtpDrafts, dense8, options),
                      "sampled-mtp-drafts applies with the MTP drafter only");
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::SampledMtpDrafts, {DenseKernels::Q8, true, kv}),
                          "sampled-mtp-drafts is qualified on Dense8 with every KV format");
    }
    // The other Flash-Next configurations: IQ3_S (GGUF dense, 16-bit experts), the official FP8 (BF16
    // dense, 16-bit experts), recipe A (BF16 dense, 4-bit experts) and recipe C (FP8 dense).
    const DenseKernels kGgml = DenseKernels::Ggml;
    const DenseKernels kBf16 = DenseKernels::Bf16;
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        const FastPathConfiguration iq3s{kGgml, false, kv};
        const FastPathConfiguration fp8{kBf16, false, kv};
        const FastPathConfiguration recipe_a{kBf16, true, kv};
        // Sampled drafts change no logit: every configuration whose drafter they were timed on or by.
        for (const FastPathConfiguration& config : {iq3s, fp8, recipe_a, FastPathConfiguration{DenseKernels::Fp8Row, false, kv}}) {
            failures += check(q::fast_path_qualified(FastPath::SampledMtpDrafts, config),
                              "sampled-mtp-drafts is qualified on every Flash-Next configuration");
        }
        for (const FastPathConfiguration& config : {iq3s, fp8, recipe_a}) {
            for (const FastPath path : {FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume, FastPath::Preemption}) {
                failures += check(q::fast_path_qualified(path, config),
                                  "the coupling paths are qualified on IQ3_S, FP8 and recipe A");
            }
        }
        // FP16 GDN: faster on IQ3_S and FP8 prefill, not on Dense8's; IQ3_S with BF16 KV fails its gates.
        failures += check(q::fast_path_qualified(FastPath::GdnF16Accumulate, iq3s) == (kv != KvCacheStorage::BFloat16) &&
                              q::fast_path_qualified(FastPath::GdnF16Accumulate, fp8) &&
                              !q::fast_path_qualified(FastPath::GdnF16Accumulate, {DenseKernels::Q8, true, kv}),
                          "gdn-f16-accumulate is registered where it was measured faster and passed");
    }
    failures += check(exact.gdn_arithmetic == infernix::ops::GdnChunkArithmetic::Exact,
                      "the exact route keeps the GDN prefill exact");
    selection.paths = {FastPath::GdnF16Accumulate};
    failures += check(q::fast_path_applies(FastPath::GdnF16Accumulate, recipe_c, options) &&
                          selection.forward().gdn_arithmetic == infernix::ops::GdnChunkArithmetic::Fp16Accumulate,
                      "gdn-f16-accumulate applies to every artifact and sets its Forward flag");
    failures += check(selection.request_invariant(), "gdn-f16-accumulate keeps a request invariant");

    // ggml-bf16-prefill acts on GGUF dense projections only, sets the BF16 route flag, and couples a
    // prompt's arithmetic to where its calls start.
    failures += check(q::fast_path_applies(FastPath::GgmlBf16Prefill, {kGgml, false, KvCacheStorage::Int8Group64}, options),
                      "ggml-bf16-prefill runs on GGUF dense weights");
    for (const DenseKernels dense : {DenseKernels::Q8, DenseKernels::Bf16, DenseKernels::Fp8Row}) {
        failures += check(!q::fast_path_applies(FastPath::GgmlBf16Prefill, {dense, false, KvCacheStorage::Int8Group64}, options),
                          "ggml-bf16-prefill has no route for non-GGUF dense weights");
    }
    failures += check(!infernix::ops::has_route(exact.dense_routes, infernix::ops::LinearRoutes::GgmlBf16Wide),
                      "the exact route keeps GGUF projections exact");
    selection.paths = {FastPath::GgmlBf16Prefill};
    failures += check(infernix::ops::has_route(selection.forward().dense_routes, infernix::ops::LinearRoutes::GgmlBf16Wide),
                      "ggml-bf16-prefill sets its route flag");
    failures += check(!selection.request_invariant(), "ggml-bf16-prefill couples a prompt to its call boundaries");
    for (const auto kv : {KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::GgmlBf16Prefill, {kGgml, false, kv}),
                          "ggml-bf16-prefill is qualified on GGUF imports with every KV format that passed its gate");
    }
    failures += check(!q::fast_path_qualified(FastPath::GgmlBf16Prefill, {kGgml, false, KvCacheStorage::BFloat16}),
                      "ggml-bf16-prefill is not qualified with BF16 KV (it fails the gates there)");

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Qwen3.8-Flash-Next fast-path selection\n";
    return failures == 0 ? 0 : 1;
}
