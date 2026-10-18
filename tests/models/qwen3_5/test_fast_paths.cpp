// Fast-path selection (models/qwen3_5/program/planning/fast_paths.h): where each fast path's kernels
// run, which configurations the registry qualifies it for, and how a configuration is described.
// The selection itself (select_fast_paths) needs a loaded model; the engine real tests and startup
// logs cover it.

#include "models/qwen3_5/program/planning/fast_paths.h"

#include <iostream>
#include <string>

namespace {

using infernix::FastPath;
using infernix::KvCacheStorage;
using infernix::models::qwen3_5::ActivationClass;
using infernix::models::qwen3_5::FastPathConfiguration;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    namespace q = infernix::models::qwen3_5;
    int failures = 0;
    const infernix::EngineOptions options;

    // decode-split64: INT8 KV with 24 query heads only.
    const FastPathConfiguration int8_24{24, ActivationClass::A4, KvCacheStorage::Int8Group64};
    failures += check(q::fast_path_applies(FastPath::DecodeSplit64, int8_24, options, true),
                      "decode-split64 must apply to 24 heads with INT8 KV");
    failures += check(!q::fast_path_applies(FastPath::DecodeSplit64,
                                            {16, ActivationClass::A4, KvCacheStorage::Int8Group64}, options, true),
                      "decode-split64 must not apply to 16 heads");
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(!q::fast_path_applies(FastPath::DecodeSplit64, {24, ActivationClass::A4, kv}, options, true),
                          "decode-split64 must apply to INT8 KV only");
    }

    // prompt-pv8: the KV formats whose prompt kernel has an 8-bit P*V form.
    for (const auto kv : {KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_applies(FastPath::PromptPv8, {24, ActivationClass::A4, kv}, options, true),
                          "prompt-pv8 must apply where an 8-bit P*V form exists");
    }
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256}) {
        failures += check(!q::fast_path_applies(FastPath::PromptPv8, {24, ActivationClass::A4, kv}, options, true),
                          "prompt-pv8 must not apply to BF16 or FP8 KV");
    }
    infernix::EngineOptions original = options;
    original.original_int8_prefill_kernel = true;
    failures += check(!q::fast_path_applies(FastPath::PromptPv8, int8_24, original, true),
                      "prompt-pv8 needs the fast INT8 prompt kernel");
    original                                = options;
    original.original_nvfp4_prefill_kernel  = true;
    failures += check(!q::fast_path_applies(FastPath::PromptPv8,
                                            {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}, original, true),
                      "prompt-pv8 needs the fast NVFP4 prompt kernel");

    // The request-coupling paths apply only where the exact route can be request-invariant, and
    // speculation and cache resume only where those run.
    // adaptive-mtp and idle-wide-prefill couple requests too, but no exact route runs their
    // behaviour, so they remain selectable fast paths where invariance is impossible (below).
    for (const FastPath path : infernix::kFastPaths) {
        if (!q::couples_requests(path) || path == FastPath::AdaptiveMtp || path == FastPath::IdleWidePrefill) {
            continue;
        }
        failures += check(!q::fast_path_applies(path, int8_24, options, false),
                          std::string(infernix::fast_path_name(path)) +
                              " must not apply where request invariance is impossible");
    }
    for (const FastPath path :
         {FastPath::BatchedDecode, FastPath::BatchSplitAttention, FastPath::Preemption}) {
        failures += check(q::fast_path_applies(path, int8_24, options, true),
                          std::string(infernix::fast_path_name(path)) + " must apply when capable");
    }
    // Low-precision batched decode needs an artifact whose activations go below 16 bits.
    for (const FastPath path : {FastPath::BatchedA8Decode, FastPath::BatchedA4Decode,
                                FastPath::BatchedA4ResidualDecode, FastPath::BatchedA8ResidualDecode}) {
        failures += check(!q::fast_path_applies(path,
                                                {24, ActivationClass::A16, KvCacheStorage::Int8Group64},
                                                options, true),
                          std::string(infernix::fast_path_name(path)) +
                              " must not apply at 16-bit activations");
        failures += check(q::fast_path_applies(path, int8_24, options, true),
                          std::string(infernix::fast_path_name(path)) +
                              " must apply at 4-bit activations");
    }
    for (const FastPath path : {FastPath::BatchedA4Decode, FastPath::BatchedA4ResidualDecode}) {
        failures += check(!q::fast_path_applies(
                              path, {24, ActivationClass::A8, KvCacheStorage::Int8Group64}, options,
                              true),
                          std::string(infernix::fast_path_name(path)) +
                              " must not apply at 8-bit activations");
    }
    infernix::EngineOptions plain = options;
    plain.speculative             = {};
    plain.context_cache.enabled   = false;
    failures += check(!q::fast_path_applies(FastPath::AdaptiveSpeculation, int8_24, plain, true),
                      "adaptive-speculation must not apply without speculation");
    failures += check(!q::fast_path_applies(FastPath::FreeCacheResume, int8_24, plain, true),
                      "free-cache-resume must not apply without the prefix cache");
    infernix::EngineOptions coupled         = plain;
    coupled.speculative.backend             = infernix::SpeculativeBackend::Mtp;
    coupled.context_cache.enabled           = true;
    failures += check(q::fast_path_applies(FastPath::AdaptiveSpeculation, int8_24, coupled, true),
                      "adaptive-speculation must apply with speculation");
    failures += check(q::fast_path_applies(FastPath::FreeCacheResume, int8_24, coupled, true),
                      "free-cache-resume must apply with the prefix cache");
    failures += check(!q::couples_requests(FastPath::PromptPv8) &&
                          !q::couples_requests(FastPath::DecodeSplit64),
                      "prompt-pv8 and decode-split64 do not couple requests");

    // The request-coupling paths and both residual routes are qualified on every KV format
    // (4-bit activations); the one-term 4- and 8-bit paths failed the decode gate.
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                          KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Q4KeyVq2Value,
                          KvCacheStorage::Vq2}) {
        for (const FastPath path :
             {FastPath::BatchedDecode, FastPath::BatchedA4ResidualDecode,
              FastPath::BatchedA8ResidualDecode, FastPath::BatchSplitAttention,
              FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume, FastPath::Preemption}) {
            failures += check(q::fast_path_qualified(path, {24, ActivationClass::A4, kv}),
                              std::string(infernix::fast_path_name(path)) + " is qualified");
        }
        for (const FastPath path : {FastPath::BatchedA4Decode, FastPath::BatchedA8Decode}) {
            failures += check(!q::fast_path_qualified(path, {24, ActivationClass::A4, kv}),
                              std::string(infernix::fast_path_name(path)) + " is not qualified");
        }
    }
    failures += check(!q::fast_path_qualified(FastPath::BatchedDecode,
                                              {16, ActivationClass::A4, KvCacheStorage::Int8Group64}),
                      "batched-decode is not qualified for an unmeasured head count");

    // Request invariance needs capability and no coupling path.
    q::FastPathSelection selection{.configuration = int8_24, .paths = {}, .invariant_capable = true};
    failures += check(selection.request_invariant(), "no coupling path selected is invariant");
    selection.paths = {FastPath::PromptPv8, FastPath::DecodeSplit64};
    failures += check(selection.request_invariant(), "prompt-pv8 and decode-split64 keep invariance");
    selection.paths.push_back(FastPath::Preemption);
    failures += check(!selection.request_invariant() && !selection.invariant(FastPath::Preemption) &&
                          selection.invariant(FastPath::BatchedDecode),
                      "a coupling path gives up its own part of invariance only");
    selection.paths.clear();
    selection.invariant_capable = false;
    failures += check(!selection.request_invariant() && !selection.invariant(FastPath::BatchedDecode),
                      "an incapable artifact is not invariant");

    // Qualification is per configuration: the measured one, not its neighbours.
    failures += check(q::fast_path_qualified(FastPath::DecodeSplit64, int8_24),
                      "decode-split64 is qualified for 24 heads, 4-bit activations, INT8 KV");
    for (const auto activations : {ActivationClass::A8, ActivationClass::A16}) {
        failures += check(!q::fast_path_qualified(FastPath::DecodeSplit64,
                                                  {24, activations, KvCacheStorage::Int8Group64}),
                          "decode-split64 is not qualified at higher activation precision");
    }
    for (const auto kv : {KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A4, kv}),
                          "prompt-pv8 is qualified for 24 heads and 4-bit activations on these KV formats");
        failures += check(!q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A16, kv}),
                          "prompt-pv8 is not qualified at 16-bit activations");
    }
    failures += check(q::describe(int8_24) == "24 heads, 4-bit activations, int8 KV",
                      "configuration description changed");

    failures += check(q::fast_path_applies(FastPath::IdleWidePrefill, int8_24, options, false),
                      "idle-wide-prefill applies where request invariance is impossible");
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                    KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                    KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::IdleWidePrefill, {24, ActivationClass::A4, kv}),
                          "idle-wide-prefill is qualified where it passed its gate");
        failures += check(q::fast_path_qualified(FastPath::GdnF16Accumulate, {24, ActivationClass::A4, kv}),
                          "gdn-f16-accumulate is qualified where it passed its gate");
    }
    failures += check(q::fast_path_applies(FastPath::GdnF16Accumulate, int8_24, options, false) &&
                          !q::couples_requests(FastPath::GdnF16Accumulate),
                      "gdn-f16-accumulate applies everywhere and keeps a request invariant");
    failures += check(!q::fast_path_qualified(FastPath::IdleWidePrefill,
                                              {24, ActivationClass::A16, KvCacheStorage::Int8Group64}),
                      "idle-wide-prefill is not qualified at 16-bit activations");
    auto mtp                 = options;
    mtp.speculative.backend  = infernix::SpeculativeBackend::Mtp;
    failures += check(!q::fast_path_applies(FastPath::AdaptiveMtp, int8_24, mtp, false),
                      "adaptive-mtp must not apply to a fixed --draft-tokens N");
    mtp.speculative.draft_tokens_auto = true;
    failures += check(q::fast_path_applies(FastPath::AdaptiveMtp, int8_24, mtp, false),
                      "adaptive-mtp applies to --draft-tokens auto where request invariance is impossible");
    failures += check(!q::fast_path_applies(FastPath::AdaptiveMtp, int8_24, options, true),
                      "adaptive-mtp needs the MTP backend");
    // Where adaptive-mtp does not run, auto drafts the configuration's fastest fixed length.
    failures += check(q::auto_fixed_draft_tokens(int8_24) == 5 &&
                          q::auto_fixed_draft_tokens({24, ActivationClass::A4, KvCacheStorage::Vq2}) == 5,
                      "auto drafts five on the NVFP4 27B");
    failures += check(q::auto_fixed_draft_tokens({24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}) == 3 &&
                          q::auto_fixed_draft_tokens({16, ActivationClass::A16, KvCacheStorage::Int8Group64}) == 3,
                      "auto drafts three on the groupwise 27B and the 35B-A3B");

    // The groupwise 27B takes prompt-pv8 by composition (model-quality.md 6.1): the NVFP4 27B's
    // kernels, shapes and clear gain, on the same KV formats.
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A16, kv, true}) ==
                              q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A4, kv}),
                          "the groupwise 27B takes prompt-pv8 where the NVFP4 27B has it");
    }

    // The groupwise 27B's and the 35B-A3B's own evidence: idle-wide-prefill on every KV format, the
    // 27B's GDN path everywhere and decode-split64 on INT8; the 35B's prompt-pv8 on each format it was
    // timed on and no gdn-f16-accumulate (its gain was within spread). adaptive-mtp: the NVFP4 27B and
    // the 35B on every format.
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::IdleWidePrefill, {24, ActivationClass::A16, kv, true}) &&
                              q::fast_path_qualified(FastPath::GdnF16Accumulate, {24, ActivationClass::A16, kv, true}) &&
                              q::fast_path_qualified(FastPath::IdleWidePrefill, {16, ActivationClass::A16, kv}) &&
                              !q::fast_path_qualified(FastPath::GdnF16Accumulate, {16, ActivationClass::A16, kv}),
                          "the groupwise 27B and 35B prefill paths");
        failures += check(q::fast_path_qualified(FastPath::PromptPv8, {16, ActivationClass::A16, kv}) ==
                              q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A4, kv}),
                          "the 35B's prompt-pv8 on every format the kernel serves");
        failures += check(q::fast_path_qualified(FastPath::AdaptiveMtp, {24, ActivationClass::A4, kv}) &&
                              q::fast_path_qualified(FastPath::AdaptiveMtp, {16, ActivationClass::A16, kv}),
                          "adaptive-mtp on the NVFP4 27B and the 35B");
    }
    failures += check(q::fast_path_qualified(FastPath::DecodeSplit64, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}) &&
                          q::fast_path_qualified(FastPath::PromptPv8, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}),
                      "the groupwise 27B's decode-split64 and the 35B's prompt-pv8 on INT8 KV");

    // q4-w4a8: only where the text MLP gate/up is groupwise Q4; it fails its gate, so nowhere qualified.
    const FastPathConfiguration groupwise{24, ActivationClass::A16, KvCacheStorage::Int8Group64, true};
    failures += check(q::fast_path_applies(FastPath::Q4W4A8, groupwise, options, false),
                      "q4-w4a8 must apply to a groupwise Q4 gate/up");
    failures += check(!q::fast_path_applies(FastPath::Q4W4A8, int8_24, options, true),
                      "q4-w4a8 must not apply without a Q4 gate/up");
    failures += check(!q::fast_path_qualified(FastPath::Q4W4A8, groupwise),
                      "q4-w4a8 failed its gate");
    failures += check(q::describe(groupwise) == "24 heads, 16-bit activations, int8 KV, Q4 gate/up",
                      "groupwise configuration description changed");

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " fast-path selection\n";
    return failures == 0 ? 0 : 1;
}
