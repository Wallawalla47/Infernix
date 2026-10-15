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

    // The kernel paths are qualified only where they passed the gate: Q8 dense, 4-bit experts, INT8 KV.
    for (const FastPath path : {FastPath::Q8SlicedK, FastPath::GdnConvFused, FastPath::QsaPrepareFused}) {
        failures += check(q::fast_path_qualified(path, dense8), "a kernel path is qualified on NVFP4 Dense8, INT8 KV");
        failures += check(!q::fast_path_qualified(path, {QType::Q8_G32_FP16, true, KvCacheStorage::Nvfp4Group16}),
                          "a kernel path stays unqualified on an ungated KV format");
        failures += check(!q::fast_path_qualified(path, {QType::Q8_G32_FP16, false, KvCacheStorage::Int8Group64}),
                          "a kernel path stays unqualified with 16-bit expert activations");
    }
    failures += check(!q::fast_path_qualified(FastPath::Fp8A8Pair, {QType::FP8_E4M3FN_ROW_BF16, true, KvCacheStorage::Int8Group64}),
                      "fp8-a8-pair is qualified at op level only, not on any artifact");

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

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Qwen3.8-Flash-Next fast-path selection\n";
    return failures == 0 ? 0 : 1;
}
