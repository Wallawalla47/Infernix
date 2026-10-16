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
    const FastPathConfiguration dense8{QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64};
    failures += check(q::fast_path_applies(FastPath::Q8SlicedK, dense8, options), "q8-sliced-k runs on Q8 dense weights");
    for (const QType dense : {QType::BF16, QType::FP8_E4M3FN_ROW_BF16}) {
        failures += check(!q::fast_path_applies(FastPath::Q8SlicedK, {dense, true, KvCacheStorage::Int8Group64}, options),
                          "q8-sliced-k has no kernels for other dense formats");
    }
    // The Qwen3.8-27B attention paths have no Flash-Next kernels.
    failures += check(!q::fast_path_applies(FastPath::PromptPv8, dense8, options), "prompt-pv8 does not apply");
    failures += check(!q::fast_path_applies(FastPath::DecodeSplit64, dense8, options), "decode-split64 does not apply");

    failures += check(q::describe(dense8) == "Q8 dense, 4-bit expert activations, int8 KV",
                      "configuration description changed");
    failures += check(q::describe({QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::BFloat16}) ==
                          "FP8 dense, 16-bit expert activations, bf16 KV",
                      "configuration description changed");

    failures += check(q::fast_path_applies(FastPath::GdnConvFused, dense8, options), "gdn-conv-fused runs on Q8 dense weights");
    failures += check(!q::fast_path_applies(FastPath::GdnConvFused, {QType::BF16, true, KvCacheStorage::Int8Group64}, options),
                      "gdn-conv-fused has no kernels for BF16 dense weights");
    failures += check(q::fast_path_applies(FastPath::QsaPrepareFused, dense8, options), "qsa-prepare-fused runs on INT8 KV");
    failures += check(!q::fast_path_applies(FastPath::QsaPrepareFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Vq2}, options),
                      "qsa-prepare-fused has no VQ2 kernel");

    // The kernel paths are qualified with Q8 dense weights and 4-bit experts on every KV format their
    // kernels serve: qsa-prepare-fused has none for VQ2 or K4V2.
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                                    KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        const FastPathConfiguration config{QType::Q8_G32_FP16, true, kv};
        failures += check(q::fast_path_qualified(FastPath::Q8SlicedK, config) &&
                              q::fast_path_qualified(FastPath::GdnConvFused, config),
                          "the dense kernel paths are qualified on every KV format");
        failures += check(q::fast_path_qualified(FastPath::QsaPrepareFused, config) ==
                              q::fast_path_applies(FastPath::QsaPrepareFused, config, options),
                          "qsa-prepare-fused is qualified exactly where its kernel runs");
    }
    for (const FastPath path : {FastPath::Q8SlicedK, FastPath::GdnConvFused, FastPath::QsaPrepareFused}) {
        failures += check(!q::fast_path_qualified(path, {QType::Q8_G32_FP16, false, KvCacheStorage::Int8Group64}),
                          "a kernel path stays unqualified with 16-bit expert activations");
    }
    failures += check(!q::fast_path_qualified(FastPath::Fp8A8Pair, {QType::FP8_E4M3FN_ROW_BF16, true, KvCacheStorage::Int8Group64}),
                      "fp8-a8-pair is qualified at op level only, not on any artifact");

    // The request-coupling paths, per KV format: speculation and cache resume passed their joint arm
    // on these formats (FP8's and K8V4's failed), preemption its increment arm on all of them.
    struct Coupling {
        KvCacheStorage kv;
        bool spec_and_cache;
        bool preemption;
    };
    for (const Coupling& c : {Coupling{KvCacheStorage::BFloat16, true, true}, Coupling{KvCacheStorage::Int8Group64, true, true},
                              Coupling{KvCacheStorage::Fp8E4M3Row256, false, true},
                              Coupling{KvCacheStorage::Nvfp4Group16, true, true},
                              Coupling{KvCacheStorage::Fp8KeyNvfp4Value, false, true},
                              Coupling{KvCacheStorage::Vq2, true, true},
                              Coupling{KvCacheStorage::Q4KeyVq2Value, true, true}}) {
        const FastPathConfiguration config{QType::Q8_G32_FP16, true, c.kv};
        failures += check(q::fast_path_qualified(FastPath::AdaptiveSpeculation, config) == c.spec_and_cache &&
                              q::fast_path_qualified(FastPath::FreeCacheResume, config) == c.spec_and_cache,
                          "adaptive-speculation and free-cache-resume follow their joint arm per KV format");
        failures += check(q::fast_path_qualified(FastPath::Preemption, config) == c.preemption,
                          "preemption follows its increment arm per KV format");
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
    const FastPathConfiguration recipe_c{QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::Int8Group64};
    failures += check(q::fast_path_applies(FastPath::ExpertA4Residual, recipe_c, options),
                      "expert-a4-residual runs on W4A16 experts");
    failures += check(!q::fast_path_applies(FastPath::ExpertA4Residual, dense8, options),
                      "expert-a4-residual does not apply to W4A4 experts");
    failures += check(!exact.expert_a4_residual, "the exact route keeps W4A16 experts exact");
    selection.paths = {FastPath::ExpertA4Residual};
    failures += check(selection.forward().expert_a4_residual, "expert-a4-residual sets its Forward flag");
    failures += check(!selection.request_invariant(), "expert-a4-residual couples a request to its round");
    for (const auto kv : {KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::ExpertA4Residual, {QType::FP8_E4M3FN_ROW_BF16, false, kv}),
                          "expert-a4-residual is qualified on recipe C with every quantized KV format");
    }
    failures += check(!q::fast_path_qualified(FastPath::ExpertA4Residual,
                                              {QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::BFloat16}),
                      "expert-a4-residual fails on BF16 KV, where recipe C's exact route is the reference");
    failures += check(!q::fast_path_qualified(FastPath::ExpertA4Residual,
                                              {QType::Q8_G32_FP16, false, KvCacheStorage::Int8Group64}),
                      "expert-a4-residual is qualified only on the dense format it was gated with");

    // sampled-mtp-drafts acts on the MTP drafter only; gdn-f16-accumulate on every artifact's GDN
    // prefill, per call, so it keeps a request invariant.
    infernix::EngineOptions mtp = options;
    mtp.speculative.backend     = infernix::SpeculativeBackend::Mtp;
    failures += check(q::fast_path_applies(FastPath::SampledMtpDrafts, dense8, mtp) &&
                          !q::fast_path_applies(FastPath::SampledMtpDrafts, dense8, options),
                      "sampled-mtp-drafts applies with the MTP drafter only");
    failures += check(exact.gdn_arithmetic == infernix::ops::GdnChunkArithmetic::Exact,
                      "the exact route keeps the GDN prefill exact");
    selection.paths = {FastPath::GdnF16Accumulate};
    failures += check(q::fast_path_applies(FastPath::GdnF16Accumulate, recipe_c, options) &&
                          selection.forward().gdn_arithmetic == infernix::ops::GdnChunkArithmetic::Fp16Accumulate,
                      "gdn-f16-accumulate applies to every artifact and sets its Forward flag");
    failures += check(selection.request_invariant(), "gdn-f16-accumulate keeps a request invariant");

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Qwen3.8-Flash-Next fast-path selection\n";
    return failures == 0 ? 0 : 1;
}
