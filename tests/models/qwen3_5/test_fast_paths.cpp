// Fast-path selection (models/qwen3_5/program/planning/fast_paths.h): where each fast path's kernels
// run, which configurations the registry qualifies it for, how a configuration is described, and the
// routes --a8-activations gives it, which the decode and verification copies of the parameters keep.
// The selection itself (select_fast_paths) needs a loaded model; the engine real tests and startup
// logs cover it.

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/planning/fast_paths.h"

#include <iostream>
#include <string>

namespace {

using infernix::FastPath;
using infernix::KvCacheStorage;
using infernix::models::qwen3_5::ActivationClass;
using infernix::models::qwen3_5::FastPathConfiguration;
using infernix::models::qwen3_5::GateUpFormat;

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
    const FastPathConfiguration int8_24{24, ActivationClass::A4, KvCacheStorage::Int8Group64, GateUpFormat::Nvfp4};
    failures += check(q::fast_path_applies(FastPath::DecodeSplit64, int8_24, options),
                      "decode-split64 must apply to 24 heads with INT8 KV");
    failures += check(!q::fast_path_applies(FastPath::DecodeSplit64,
                                            {16, ActivationClass::A4, KvCacheStorage::Int8Group64}, options),
                      "decode-split64 must not apply to 16 heads");
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(!q::fast_path_applies(FastPath::DecodeSplit64, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}, options),
                          "decode-split64 must apply to INT8 KV only");
    }

    // prompt-pv8: the KV formats whose prompt kernel has an 8-bit P*V form.
    for (const auto kv : {KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_applies(FastPath::PromptPv8, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}, options),
                          "prompt-pv8 must apply where an 8-bit P*V form exists");
    }
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256}) {
        failures += check(!q::fast_path_applies(FastPath::PromptPv8, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}, options),
                          "prompt-pv8 must not apply to BF16 or FP8 KV");
    }
    infernix::EngineOptions original = options;
    original.original_int8_prefill_kernel = true;
    failures += check(!q::fast_path_applies(FastPath::PromptPv8, int8_24, original),
                      "prompt-pv8 needs the fast INT8 prompt kernel");
    original                                = options;
    original.original_nvfp4_prefill_kernel  = true;
    failures += check(!q::fast_path_applies(FastPath::PromptPv8,
                                            {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16, GateUpFormat::Nvfp4}, original),
                      "prompt-pv8 needs the fast NVFP4 prompt kernel");

    // The request-coupling paths apply on every artifact (every one has the exact route's invariant
    // counterpart; speculation and cache resume only where those run). adaptive-mtp and
    // idle-wide-prefill couple requests too, but no exact route runs their behaviour (below).
    infernix::EngineOptions serving       = options;
    serving.speculative.backend           = infernix::SpeculativeBackend::Mtp;
    serving.context_cache.enabled         = true;
    for (const FastPath path : {FastPath::BatchedDecode, FastPath::SingleColumnDecode, FastPath::BatchSplitAttention,
                                FastPath::Preemption, FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume}) {
        failures += check(q::fast_path_applies(path, int8_24, serving),
                          std::string(infernix::fast_path_name(path)) + " must apply on every artifact");
    }
    // Low-precision batched decode needs an artifact whose activations go below 16 bits.
    for (const FastPath path : {FastPath::BatchedA8Decode, FastPath::BatchedA8SubsetDecode, FastPath::BatchedA4Decode,
                                FastPath::BatchedA4ResidualDecode, FastPath::BatchedA8ResidualDecode}) {
        failures += check(!q::fast_path_applies(path,
                                                {24, ActivationClass::A16, KvCacheStorage::Int8Group64, GateUpFormat::Nvfp4},
                                                options),
                          std::string(infernix::fast_path_name(path)) +
                              " must not apply at 16-bit activations");
        failures += check(q::fast_path_applies(path, int8_24, options),
                          std::string(infernix::fast_path_name(path)) +
                              " must apply at 4-bit activations");
    }
    for (const FastPath path : {FastPath::BatchedA4Decode, FastPath::BatchedA4ResidualDecode}) {
        failures += check(!q::fast_path_applies(
                              path, {24, ActivationClass::A8, KvCacheStorage::Int8Group64, GateUpFormat::Nvfp4}, options),
                          std::string(infernix::fast_path_name(path)) +
                              " must not apply at 8-bit activations");
    }
    infernix::EngineOptions plain = options;
    plain.speculative             = {};
    plain.context_cache.enabled   = false;
    failures += check(!q::fast_path_applies(FastPath::AdaptiveSpeculation, int8_24, plain),
                      "adaptive-speculation must not apply without speculation");
    failures += check(!q::fast_path_applies(FastPath::FreeCacheResume, int8_24, plain),
                      "free-cache-resume must not apply without the prefix cache");
    infernix::EngineOptions coupled         = plain;
    coupled.speculative.backend             = infernix::SpeculativeBackend::Mtp;
    coupled.context_cache.enabled           = true;
    failures += check(q::fast_path_applies(FastPath::AdaptiveSpeculation, int8_24, coupled),
                      "adaptive-speculation must apply with speculation");
    failures += check(q::fast_path_applies(FastPath::FreeCacheResume, int8_24, coupled),
                      "free-cache-resume must apply with the prefix cache");
    failures += check(!q::couples_requests(FastPath::PromptPv8) &&
                          !q::couples_requests(FastPath::DecodeSplit64),
                      "prompt-pv8 and decode-split64 do not couple requests");
    // single-column-decode: a step's route depends on whether another request shares its round.
    failures += check(q::couples_requests(FastPath::SingleColumnDecode),
                      "single-column-decode couples requests");
    failures += check(infernix::parse_fast_path("single-column-decode") == FastPath::SingleColumnDecode,
                      "single-column-decode is parsed by name");
    // batched-a8-subset-decode: a round's route depends on its width, which other requests share.
    failures += check(q::couples_requests(FastPath::BatchedA8SubsetDecode),
                      "batched-a8-subset-decode couples requests");
    failures += check(infernix::parse_fast_path("batched-a8-subset-decode") == FastPath::BatchedA8SubsetDecode,
                      "batched-a8-subset-decode is parsed by name");

    // The request-coupling paths and both residual routes are qualified on every KV format
    // (4-bit activations); the one-term 4- and 8-bit paths failed the decode gate (the 8-bit one is
    // registered anyway).
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                          KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Q4KeyVq2Value,
                          KvCacheStorage::Vq2}) {
        for (const FastPath path :
             {FastPath::BatchedDecode, FastPath::BatchedA4ResidualDecode,
              FastPath::BatchedA8ResidualDecode, FastPath::BatchSplitAttention,
              FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume, FastPath::Preemption}) {
            failures += check(q::fast_path_qualified(path, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                              std::string(infernix::fast_path_name(path)) + " is qualified");
        }
        for (const FastPath path : {FastPath::BatchedA4Decode, FastPath::BatchedA8SubsetDecode}) {
            failures += check(!q::fast_path_qualified(path, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                              std::string(infernix::fast_path_name(path)) + " is not qualified");
        }
        // batched-a8-decode by the owner's decision (2026-10-24), although it failed its decode gate.
        failures += check(q::fast_path_qualified(FastPath::BatchedA8Decode, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                          "batched-a8-decode is registered");
    }
    failures += check(!q::fast_path_qualified(FastPath::BatchedDecode,
                                              {16, ActivationClass::A4, KvCacheStorage::Int8Group64}),
                      "batched-decode is not qualified for an unmeasured head count");
    // The groupwise 27B and the 35B-A3B: the three coupling paths each timed faster passed on every format
    // (2026-10-22); batched-decode and batch-split-attention, mixed or slower in their timing, are not listed.
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                          KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Q4KeyVq2Value,
                          KvCacheStorage::Vq2}) {
        for (const q::FastPathConfiguration config :
             {q::FastPathConfiguration{24, ActivationClass::A16, kv, GateUpFormat::GroupwiseQ4}, q::FastPathConfiguration{16, ActivationClass::A16, kv}}) {
            for (const FastPath path : {FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume, FastPath::Preemption}) {
                failures += check(q::fast_path_qualified(path, config),
                                  std::string(infernix::fast_path_name(path)) + " is qualified for " + q::describe(config));
            }
            for (const FastPath path : {FastPath::BatchedDecode, FastPath::BatchSplitAttention}) {
                failures += check(!q::fast_path_qualified(path, config),
                                  std::string(infernix::fast_path_name(path)) + " is not qualified for " + q::describe(config));
            }
        }
        // single-column-decode: the groupwise 27B only (the 35B-A3B was 1 % slower at four requests).
        failures += check(q::fast_path_qualified(FastPath::SingleColumnDecode, {24, ActivationClass::A16, kv, GateUpFormat::GroupwiseQ4}) &&
                              !q::fast_path_qualified(FastPath::SingleColumnDecode, {16, ActivationClass::A16, kv}) &&
                              !q::fast_path_qualified(FastPath::SingleColumnDecode, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                          "single-column-decode is qualified for the groupwise 27B only");
    }

    // Request invariance needs no coupling path.
    q::FastPathSelection selection{.configuration = int8_24, .paths = {}};
    failures += check(selection.request_invariant(), "no coupling path selected is invariant");
    selection.paths = {FastPath::PromptPv8, FastPath::DecodeSplit64};
    failures += check(selection.request_invariant(), "prompt-pv8 and decode-split64 keep invariance");
    selection.paths.push_back(FastPath::Preemption);
    failures += check(!selection.request_invariant() && !selection.invariant(FastPath::Preemption) &&
                          selection.invariant(FastPath::BatchedDecode),
                      "a coupling path gives up its own part of invariance only");

    // Qualification is per configuration: the measured one, not its neighbours.
    failures += check(q::fast_path_qualified(FastPath::DecodeSplit64, int8_24),
                      "decode-split64 is qualified for 24 heads, 4-bit activations, INT8 KV");
    failures += check(!q::fast_path_qualified(FastPath::DecodeSplit64, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}),
                      "decode-split64 is not qualified on 16 heads");
    for (const auto kv : {KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                          "prompt-pv8 is qualified for 24 heads and 4-bit activations on these KV formats");
        failures += check(q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A16, kv, GateUpFormat::Nvfp4}) &&
                              !q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A8, kv, GateUpFormat::Nvfp4}),
                          "prompt-pv8 is qualified for the NVFP4 27B under --a16-activations, not at 8 bits");
    }
    failures += check(q::describe(int8_24) == "24 heads, 4-bit activations, int8 KV, NVFP4 gate/up",
                      "configuration description changed");

    failures += check(q::fast_path_applies(FastPath::IdleWidePrefill, int8_24, options),
                      "idle-wide-prefill applies on every artifact");
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                    KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                    KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::IdleWidePrefill, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                          "idle-wide-prefill is qualified where it passed its gate");
        failures += check(q::fast_path_qualified(FastPath::GdnF16Accumulate, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                          "gdn-f16-accumulate is qualified where it passed its gate");
    }
    failures += check(q::fast_path_applies(FastPath::GdnF16Accumulate, int8_24, options) &&
                          !q::couples_requests(FastPath::GdnF16Accumulate),
                      "gdn-f16-accumulate applies everywhere and keeps a request invariant");
    failures += check(q::fast_path_applies(FastPath::Fp16GdnState, int8_24, options) &&
                          !q::couples_requests(FastPath::Fp16GdnState) &&
                          q::fast_path_qualified(FastPath::Fp16GdnState, int8_24) &&
                          q::fast_path_qualified(FastPath::Fp16GdnState,
                                                 {24, ActivationClass::A4, KvCacheStorage::BFloat16, GateUpFormat::Nvfp4}),
                      "fp16-gdn-state applies everywhere, keeps a request invariant and is qualified where it "
                      "was measured faster (the 27Bs, BF16 KV included since 2026-11-19)");
    // The NVFP4 27B under --a8-activations (24 heads, 8-bit activations at every width), where measured faster.
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                    KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                    KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        const q::FastPathConfiguration a8_27b{24, ActivationClass::A8, kv, GateUpFormat::Nvfp4};
        for (const FastPath path : {FastPath::IdleWidePrefill, FastPath::GdnF16Accumulate, FastPath::Fp16GdnState,
                                    FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume, FastPath::Preemption}) {
            failures += check(q::fast_path_qualified(path, a8_27b),
                              std::string(infernix::fast_path_name(path)) + " is qualified under --a8-activations");
        }
        for (const FastPath path : {FastPath::BatchedDecode, FastPath::SingleColumnDecode,
                                    FastPath::BatchSplitAttention, FastPath::PromptPv8, FastPath::AdaptiveMtp,
                                    FastPath::AdaptiveDFlash, FastPath::GumbelSampling, FastPath::BatchedA8Decode,
                                    FastPath::BatchedA8ResidualDecode}) {
            failures += check(!q::fast_path_qualified(path, a8_27b),
                              std::string(infernix::fast_path_name(path)) + " stays off under --a8-activations");
        }
    }
    failures += check(q::fast_path_qualified(FastPath::DecodeSplit64, {24, ActivationClass::A8, KvCacheStorage::Int8Group64, GateUpFormat::Nvfp4}) &&
                          q::auto_fixed_draft_tokens({24, ActivationClass::A8, KvCacheStorage::Int8Group64, GateUpFormat::Nvfp4},
                                                     infernix::SpeculativeBackend::Mtp) == 5,
                      "decode-split64 and five MTP drafts under --a8-activations");
    // NVIDIA's NVFP4 27B under --a16-activations (24 heads, 16-bit activations, NVFP4 gate/up), registered where each
    // path was measured faster (2026-11-19); the paths that were not stay off.
    {
        for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                        KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                        KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                        KvCacheStorage::Q4KeyVq2Value}) {
            const q::FastPathConfiguration a16_27b{24, ActivationClass::A16, kv, GateUpFormat::Nvfp4};
            for (const FastPath path : {FastPath::BatchedDecode, FastPath::Fp16GdnState, FastPath::IdleWidePrefill,
                                        FastPath::GdnF16Accumulate, FastPath::AdaptiveSpeculation,
                                        FastPath::FreeCacheResume, FastPath::Preemption}) {
                failures += check(q::fast_path_qualified(path, a16_27b),
                                  std::string(infernix::fast_path_name(path)) + " is qualified for the NVFP4 27B under "
                                                                      "--a16-activations");
            }
            for (const FastPath path : {FastPath::SingleColumnDecode, FastPath::BatchSplitAttention,
                                        FastPath::AdaptiveMtp, FastPath::AdaptiveDFlash, FastPath::GumbelSampling}) {
                failures += check(!q::fast_path_qualified(path, a16_27b),
                                  std::string(infernix::fast_path_name(path)) + " was not faster for the NVFP4 27B under "
                                                                      "--a16-activations");
            }
        }
        const q::FastPathConfiguration a16_27b{24, ActivationClass::A16, KvCacheStorage::Int8Group64, GateUpFormat::Nvfp4};
        failures += check(q::fast_path_qualified(FastPath::DecodeSplit64, a16_27b), "decode-split64 under --a16");
        // Adaptive drafting takes the NVFP4 27B's round costs and batch length at any activation precision, its MTP
        // fixed length five; the n-gram copy windows stay on the 4-bit class (not timed elsewhere).
        failures += check(q::auto_fixed_draft_tokens(a16_27b, infernix::SpeculativeBackend::Mtp) == 5 &&
                              q::dflash_width_costs(a16_27b, infernix::SpeculativeBackend::DFlash2).proposal == 0.18 &&
                              q::dflash_batch_drafts(a16_27b, infernix::SpeculativeBackend::DFlash2) == 7 &&
                              q::auto_ngram_settings(a16_27b, infernix::SpeculativeBackend::DFlash2, true).window == 0,
                          "the NVFP4 27B under --a16-activations drafts with its own round costs");
    }
    // expert-a4-residual (either term count) acts on offloaded W4A16 experts (Qwen3.5-122B-A10B: 32 heads, 8-bit
    // FP8 activations) only, and it couples a column to its call. Two terms are registered there on every KV
    // format (measured faster, 2026-10-28); three terms stay a candidate.
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                    KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                    KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        const q::FastPathConfiguration offloaded{32, ActivationClass::A8, kv, GateUpFormat::Other, true};
        for (const FastPath path : {FastPath::ExpertA4Residual, FastPath::ExpertA4Residual3}) {
            failures += check(q::fast_path_applies(path, offloaded, options),
                              "expert-a4-residual applies to the 122B's offloaded experts");
        }
        failures += check(q::fast_path_qualified(FastPath::ExpertA4Residual, offloaded) &&
                              !q::fast_path_qualified(FastPath::ExpertA4Residual3, offloaded),
                          "two-term expert-a4-residual is registered on the 122B, three terms are not");
        // free-cache-resume and preemption measured faster there (2026-11-03: continuing-turn TTFT, a short pool);
        // gdn-f16-accumulate level.
        failures += check(q::fast_path_qualified(FastPath::FreeCacheResume, offloaded) &&
                              q::fast_path_qualified(FastPath::Preemption, offloaded) &&
                              !q::fast_path_qualified(FastPath::GdnF16Accumulate, offloaded),
                          "free-cache-resume and preemption are registered on the 122B, gdn-f16-accumulate is not");
    }
    for (const KvCacheStorage kv : {KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16,
                                    KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::PromptPv8, {32, ActivationClass::A8, kv, GateUpFormat::Other, true}),
                          "prompt-pv8 is registered on the 122B on every format its kernel serves (measured faster)");
    }
    failures += check(!q::fast_path_applies(FastPath::ExpertA4Residual, int8_24, options) &&
                          !q::fast_path_applies(FastPath::ExpertA4Residual3, int8_24, options),
                      "expert-a4-residual needs offloaded experts");
    failures += check(q::couples_requests(FastPath::ExpertA4Residual) && q::couples_requests(FastPath::ExpertA4Residual3),
                      "expert-a4-residual couples a column to its expert's column count in the call");
    auto mtp                 = options;
    mtp.speculative.backend  = infernix::SpeculativeBackend::Mtp;
    failures += check(!q::fast_path_applies(FastPath::AdaptiveMtp, int8_24, mtp),
                      "adaptive-mtp must not apply to a fixed --draft-tokens N");
    mtp.speculative.draft_tokens_auto = true;
    failures += check(q::fast_path_applies(FastPath::AdaptiveMtp, int8_24, mtp),
                      "adaptive-mtp applies to --draft-tokens auto");
    failures += check(!q::fast_path_applies(FastPath::AdaptiveMtp, int8_24, options),
                      "adaptive-mtp needs the MTP backend");
    // Where adaptive-mtp does not run, auto drafts the configuration's fastest fixed length.
    using infernix::SpeculativeBackend;
    failures += check(q::auto_fixed_draft_tokens(int8_24, SpeculativeBackend::Mtp) == 5 &&
                          q::auto_fixed_draft_tokens({24, ActivationClass::A4, KvCacheStorage::Vq2, GateUpFormat::Nvfp4},
                                                     SpeculativeBackend::Mtp) == 5,
                      "auto drafts five on the NVFP4 27B");
    failures += check(q::auto_fixed_draft_tokens({24, ActivationClass::A16, KvCacheStorage::Int8Group64, GateUpFormat::GroupwiseQ4},
                                                 SpeculativeBackend::Mtp) == 3 &&
                          q::auto_fixed_draft_tokens({16, ActivationClass::A16, KvCacheStorage::Int8Group64},
                                                     SpeculativeBackend::Mtp) == 3,
                      "auto drafts three on the groupwise 27B and the 35B-A3B");
    // adaptive-dflash: --draft-tokens auto's DFlash and DFlash2 policy; where it does not run, auto
    // verifies the block drafter's fastest fixed length, and a block round proposes the
    // configuration's block.
    auto block                          = options;
    block.speculative.backend           = SpeculativeBackend::DFlash2;
    failures += check(!q::fast_path_applies(FastPath::AdaptiveDFlash, int8_24, block),
                      "adaptive-dflash must not apply to a fixed --draft-tokens N");
    block.speculative.draft_tokens_auto = true;
    failures += check(q::fast_path_applies(FastPath::AdaptiveDFlash, int8_24, block),
                      "adaptive-dflash applies to DFlash2 --draft-tokens auto");
    block.speculative.backend = SpeculativeBackend::DFlash;
    failures += check(q::fast_path_applies(FastPath::AdaptiveDFlash, int8_24, block),
                      "adaptive-dflash applies to DFlash --draft-tokens auto");
    failures += check(!q::fast_path_applies(FastPath::AdaptiveDFlash, int8_24, mtp),
                      "adaptive-dflash needs a block drafter");
    failures += check(q::auto_fixed_draft_tokens(int8_24, SpeculativeBackend::DFlash2) == 7 &&
                          q::auto_fixed_draft_tokens({24, ActivationClass::A16, KvCacheStorage::Int8Group64, GateUpFormat::GroupwiseQ4},
                                                     SpeculativeBackend::DFlash2) == 7 &&
                          q::auto_fixed_draft_tokens({16, ActivationClass::A16, KvCacheStorage::Int8Group64},
                                                     SpeculativeBackend::DFlash) == 3,
                      "auto verifies the block drafter's fastest fixed length where adaptive-dflash does not run");
    // Offloaded experts (the 122B-A10B) price a verified draft by its expert misses, keyed on the
    // offload, not on its 32 query heads.
    {
        const q::FastPathConfiguration offloaded{32, ActivationClass::A16, KvCacheStorage::BFloat16, GateUpFormat::Other, true};
        const q::FastPathConfiguration resident32{32, ActivationClass::A16, KvCacheStorage::BFloat16};
        failures += check(q::auto_fixed_draft_tokens(offloaded, SpeculativeBackend::DFlash2) == 1 &&
                              q::auto_fixed_draft_tokens(resident32, SpeculativeBackend::DFlash2) == 3,
                          "offloaded experts draft one under a forced fixed length");
        failures += check(q::auto_fixed_draft_tokens(offloaded, SpeculativeBackend::Mtp) == 2 &&
                              !q::auto_drafts_nothing(true, SpeculativeBackend::Mtp) &&
                              q::auto_drafts_nothing(true, SpeculativeBackend::DFlash2) &&
                              !q::auto_drafts_nothing(false, SpeculativeBackend::DFlash2),
                          "offloaded experts' auto drafts two gated MTP drafts and no DFlash2 draft");
        failures += check(q::dflash_width_costs(offloaded, SpeculativeBackend::DFlash2).per_draft == 0.72 &&
                              q::dflash_width_costs(resident32, SpeculativeBackend::DFlash2).per_draft == 0.15,
                          "offloaded experts price a draft column at 0.72 of a plain round");
    }
    // --spec auto takes the bundled drafter measured fastest: with offloaded experts MTP (DFlash2's auto
    // decodes plain there), else DFlash2 (the Qwen3.8-27Bs), then MTP (the 35B-A3B beside DFlash, the
    // Qwen3.6-27Bs alone), then DFlash; none without a drafter worth binding.
    {
        const q::BundledDrafters all{.mtp = true, .dflash = true, .dflash2 = true};
        const q::BundledDrafters qwen3_8_27b{.mtp = true, .dflash2 = true};
        const q::BundledDrafters qwen3_6_35b{.mtp = true, .dflash = true};
        failures += check(q::auto_speculative_backend(true, qwen3_8_27b) == SpeculativeBackend::Mtp &&
                              q::auto_speculative_backend(true, all) == SpeculativeBackend::Mtp &&
                              q::auto_speculative_backend(true, {.dflash2 = true}) == SpeculativeBackend::None &&
                              q::auto_speculative_backend(true, {}) == SpeculativeBackend::None,
                          "--spec auto with offloaded experts did not take MTP alone");
        failures += check(q::auto_speculative_backend(false, qwen3_8_27b) == SpeculativeBackend::DFlash2 &&
                              q::auto_speculative_backend(false, all) == SpeculativeBackend::DFlash2 &&
                              q::auto_speculative_backend(false, qwen3_6_35b) == SpeculativeBackend::Mtp &&
                              q::auto_speculative_backend(false, {.mtp = true}) == SpeculativeBackend::Mtp &&
                              q::auto_speculative_backend(false, {.dflash = true}) == SpeculativeBackend::DFlash &&
                              q::auto_speculative_backend(false, {}) == SpeculativeBackend::None,
                          "--spec auto did not rank resident drafters DFlash2, MTP, DFlash");
    }
    failures += check(q::dflash_auto_ceiling(int8_24, SpeculativeBackend::DFlash2) == 15 &&
                          q::dflash_auto_ceiling({16, ActivationClass::A16, KvCacheStorage::Int8Group64},
                                                 SpeculativeBackend::DFlash) == 7,
                      "a block round proposes the configuration's block");
    failures += check(q::dflash_batch_drafts(int8_24, SpeculativeBackend::DFlash2) == 7 &&
                          q::dflash_batch_drafts({16, ActivationClass::A16, KvCacheStorage::Int8Group64},
                                                 SpeculativeBackend::DFlash) == 0,
                      "the NVFP4 27B's batched rounds run plain at 7 drafts");
    // --ngram-draft-tokens auto: copies beside the NVFP4 27B's adaptive DFlash2 rounds only; the
    // 35B-A3B's were not faster, and nothing else was measured.
    failures += check(q::auto_ngram_settings(int8_24, SpeculativeBackend::DFlash2, true).window != 0 &&
                          q::auto_ngram_settings(int8_24, SpeculativeBackend::DFlash2, false).window == 0 &&
                          q::auto_ngram_settings(int8_24, SpeculativeBackend::Mtp, true).window == 0 &&
                          q::auto_ngram_settings({16, ActivationClass::A16, KvCacheStorage::Int8Group64},
                                                 SpeculativeBackend::DFlash, true)
                                  .window == 0,
                      "the automatic n-gram window is the NVFP4 27B's beside adaptive DFlash2");

    // The groupwise 27B takes prompt-pv8 by composition (model-quality.md 6.1): the NVFP4 27B's
    // kernels, shapes and clear gain, on the same KV formats.
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A16, kv, GateUpFormat::GroupwiseQ4}) ==
                              q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                          "the groupwise 27B takes prompt-pv8 where the NVFP4 27B has it");
    }

    // The groupwise 27B's and the 35B-A3B's own evidence: idle-wide-prefill on every KV format, the
    // 27B's GDN path everywhere and decode-split64 on INT8; the 35B's prompt-pv8 on each format it was
    // timed on and no gdn-f16-accumulate (its gain was within spread). adaptive-mtp: the NVFP4 27B and
    // the 35B on every format.
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::IdleWidePrefill, {24, ActivationClass::A16, kv, GateUpFormat::GroupwiseQ4}) &&
                              q::fast_path_qualified(FastPath::GdnF16Accumulate, {24, ActivationClass::A16, kv, GateUpFormat::GroupwiseQ4}) &&
                              q::fast_path_qualified(FastPath::IdleWidePrefill, {16, ActivationClass::A16, kv}) &&
                              !q::fast_path_qualified(FastPath::GdnF16Accumulate, {16, ActivationClass::A16, kv}),
                          "the groupwise 27B and 35B prefill paths");
        failures += check(q::fast_path_qualified(FastPath::PromptPv8, {16, ActivationClass::A16, kv}) ==
                              q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                          "the 35B's prompt-pv8 on every format the kernel serves");
        failures += check(q::fast_path_qualified(FastPath::AdaptiveMtp, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}) &&
                              q::fast_path_qualified(FastPath::AdaptiveMtp, {16, ActivationClass::A16, kv}),
                          "adaptive-mtp on the NVFP4 27B and the 35B");
        failures += check(q::fast_path_qualified(FastPath::AdaptiveDFlash, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}) &&
                              q::fast_path_qualified(FastPath::AdaptiveDFlash, {16, ActivationClass::A16, kv}) &&
                              !q::fast_path_qualified(FastPath::AdaptiveDFlash,
                                                      {24, ActivationClass::A16, kv, GateUpFormat::GroupwiseQ4}),
                          "adaptive-dflash on the NVFP4 27B and the 35B, not the groupwise 27B");
    }
    failures += check(q::fast_path_qualified(FastPath::DecodeSplit64, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, GateUpFormat::GroupwiseQ4}) &&
                          q::fast_path_qualified(FastPath::PromptPv8, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}),
                      "the groupwise 27B's decode-split64 and the 35B's prompt-pv8 on INT8 KV");

    // q4-w4a8: only where the text MLP gate/up is groupwise Q4; it fails its gate, so nowhere qualified.
    const FastPathConfiguration groupwise{24, ActivationClass::A16, KvCacheStorage::Int8Group64, GateUpFormat::GroupwiseQ4};
    failures += check(q::fast_path_applies(FastPath::Q4W4A8, groupwise, options),
                      "q4-w4a8 must apply to a groupwise Q4 gate/up");
    failures += check(!q::fast_path_applies(FastPath::Q4W4A8, int8_24, options),
                      "q4-w4a8 must not apply without a Q4 gate/up");
    failures += check(!q::fast_path_qualified(FastPath::Q4W4A8, groupwise),
                      "q4-w4a8 failed its gate");
    failures += check(q::describe(groupwise) == "24 heads, 16-bit activations, int8 KV, Q4 gate/up",
                      "groupwise configuration description changed");

    // --a8-activations' routes (A8Routes): the groupwise Q4 MLP gate/up's prefill calls take q4-w4a8's int8
    // route, offloaded experts their 8-bit decode and prefill routes; the NVFP4 27B (its FP8 and NVFP4 sites
    // already take the activations the artifact permits) and the 35B-A3B have none, so they refuse the flag.
    {
        const q::A8Routes groupwise_a8 = q::a8_activation_routes(groupwise);
        failures += check(groupwise_a8.q4_gate_up && !groupwise_a8.experts && !groupwise_a8.experts_prefill &&
                              groupwise_a8.describe().find("decode and verification: none") != std::string::npos,
                          "the groupwise Q4 gate/up takes the 8-bit prefill route only");
        FastPathConfiguration offloaded_a8 = int8_24;
        offloaded_a8.offloaded_experts     = true;
        const q::A8Routes experts_a8       = q::a8_activation_routes(offloaded_a8);
        failures += check(experts_a8.experts && experts_a8.experts_prefill && !experts_a8.q4_gate_up &&
                              experts_a8.describe().find("prefill: the routed experts") != std::string::npos,
                          "offloaded experts take the 8-bit decode and prefill routes");
        failures += check(!q::a8_activation_routes(int8_24).any() &&
                              !q::a8_activation_routes({16, ActivationClass::A16, KvCacheStorage::Int8Group64}).any() &&
                              !q::a8_activation_routes({24, ActivationClass::A16, KvCacheStorage::Int8Group64}).any(),
                          "the NVFP4 27B, the 35B-A3B and a groupwise 27B without Q4 gate/up have no 8-bit route");
    }

    // sampled-mtp-drafts: MTP rounds of resident experts without n-gram copies; not yet qualified on
    // any Qwen3.5 configuration (a candidate only).
    mtp = options; // reset: the sampled-drafts cases below start from the plain options
    mtp.speculative.backend      = infernix::SpeculativeBackend::Mtp;
    mtp.speculative.draft_tokens = 3;
    infernix::EngineOptions mtp_ngram = mtp;
    mtp_ngram.speculative.ngram_draft_tokens = 16;
    infernix::EngineOptions dflash2 = options;
    dflash2.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
    dflash2.speculative.draft_tokens = 7;
    FastPathConfiguration offloaded = int8_24;
    offloaded.offloaded_experts     = true;
    failures += check(q::fast_path_applies(FastPath::SampledMtpDrafts, int8_24, mtp) &&
                          !q::fast_path_applies(FastPath::SampledMtpDrafts, int8_24, mtp_ngram) &&
                          !q::fast_path_applies(FastPath::SampledMtpDrafts, int8_24, dflash2) &&
                          !q::fast_path_applies(FastPath::SampledMtpDrafts, int8_24, options) &&
                          !q::fast_path_applies(FastPath::SampledMtpDrafts, offloaded, mtp),
                      "sampled-mtp-drafts applies to resident MTP rounds without copies only");
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16}) {
        failures += check(!q::fast_path_qualified(FastPath::SampledMtpDrafts, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}) &&
                              !q::fast_path_qualified(FastPath::SampledMtpDrafts, {16, ActivationClass::A16, kv}),
                          "sampled-mtp-drafts is not qualified on Qwen3.5");
    }
    // router-mma: the offloaded MoE layers' router only (the resident MoE models' SparseMoe Op has its own
    // MMA router), a candidate only.
    failures += check(q::fast_path_applies(FastPath::RouterMma, offloaded, options) &&
                          !q::fast_path_applies(FastPath::RouterMma, int8_24, options) &&
                          !q::fast_path_qualified(FastPath::RouterMma, offloaded),
                      "router-mma applies to offloaded experts only and is qualified nowhere on Qwen3.5");
    // block-verification: chains with proposal laws (DFlash2, and MTP under sampled-mtp-drafts); a
    // row's acceptance reads only that row, and it is a candidate only.
    infernix::EngineOptions dflash = options;
    dflash.speculative.backend      = infernix::SpeculativeBackend::DFlash;
    dflash.speculative.draft_tokens = 7;
    failures += check(q::fast_path_applies(FastPath::BlockVerification, int8_24, dflash2) &&
                          q::fast_path_applies(FastPath::BlockVerification, int8_24, mtp) &&
                          !q::fast_path_applies(FastPath::BlockVerification, int8_24, dflash) &&
                          !q::fast_path_applies(FastPath::BlockVerification, int8_24, options) &&
                          !q::couples_requests(FastPath::BlockVerification) &&
                          !q::fast_path_qualified(FastPath::BlockVerification, int8_24),
                      "block-verification applies to DFlash2 and MTP chains, keeps a request invariant "
                      "and is qualified nowhere");
    failures += check(infernix::parse_fast_path("block-verification") == FastPath::BlockVerification,
                      "block-verification parses");
    // gumbel-sampling (candidates ZH-01): MTP rounds only (DFlash2's equality acceptance timed slower), a row's draw
    // reads only that row; registered on the NVFP4 27B's configuration with INT8 KV (2026-11-16) and the formats below
    // its precision (2026-11-19); not BF16, the groupwise 27B (timed slower) or the 35B-A3B.
    failures += check(!q::fast_path_applies(FastPath::GumbelSampling, int8_24, dflash2) &&
                          q::fast_path_applies(FastPath::GumbelSampling, int8_24, mtp) &&
                          !q::fast_path_applies(FastPath::GumbelSampling, int8_24, dflash) &&
                          !q::fast_path_applies(FastPath::GumbelSampling, int8_24, options) &&
                          !q::couples_requests(FastPath::GumbelSampling) &&
                          q::fast_path_qualified(FastPath::GumbelSampling, int8_24) &&
                          q::fast_path_qualified(FastPath::GumbelSampling,
                                                 {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16, GateUpFormat::Nvfp4}) &&
                          !q::fast_path_qualified(FastPath::GumbelSampling,
                                                  {24, ActivationClass::A4, KvCacheStorage::BFloat16, GateUpFormat::Nvfp4}) &&
                          !q::fast_path_qualified(FastPath::GumbelSampling,
                                                  {24, ActivationClass::A16, KvCacheStorage::Int8Group64, GateUpFormat::GroupwiseQ4}) &&
                          !q::fast_path_qualified(FastPath::GumbelSampling,
                                                  {16, ActivationClass::A16, KvCacheStorage::Int8Group64}) &&
                          infernix::parse_fast_path("gumbel-sampling") == FastPath::GumbelSampling,
                      "gumbel-sampling applies to MTP rounds, keeps a request invariant, is registered on the NVFP4 "
                      "27B with INT8 KV and the formats below its precision only and parses");
    // univer-trees: fixed-table tree rounds of the drafters that build their trees (DFlash2, DFlash);
    // not automatic widths, not MTP's leaf trees, not chains.
    infernix::EngineOptions dflash2_trees = dflash2, dflash_trees = dflash, mtp_trees = mtp,
                            auto_trees = dflash2;
    dflash2_trees.speculative.draft_tree_nodes = {16};
    dflash_trees.speculative.draft_tree_nodes  = {12};
    mtp_trees.speculative.draft_tree_nodes     = {12};
    auto_trees.speculative.draft_tree_auto     = true;
    failures += check(q::fast_path_applies(FastPath::UniverTrees, int8_24, dflash2_trees) &&
                          q::fast_path_applies(FastPath::UniverTrees, int8_24, dflash_trees) &&
                          !q::fast_path_applies(FastPath::UniverTrees, int8_24, mtp_trees) &&
                          !q::fast_path_applies(FastPath::UniverTrees, int8_24, auto_trees) &&
                          !q::fast_path_applies(FastPath::UniverTrees, int8_24, dflash2) &&
                          !q::couples_requests(FastPath::UniverTrees) &&
                          !q::fast_path_qualified(FastPath::UniverTrees, int8_24),
                      "univer-trees applies to fixed DFlash2/DFlash tree tables only, keeps a request "
                      "invariant and is qualified nowhere");
    failures += check(infernix::parse_fast_path("univer-trees") == FastPath::UniverTrees, "univer-trees parses");
    // gdn-tree-closed-form: every engine with tree rounds (any drafter, fixed or automatic widths).
    failures += check(q::fast_path_applies(FastPath::GdnTreeClosedForm, int8_24, dflash2_trees) &&
                          q::fast_path_applies(FastPath::GdnTreeClosedForm, int8_24, dflash_trees) &&
                          q::fast_path_applies(FastPath::GdnTreeClosedForm, int8_24, mtp_trees) &&
                          q::fast_path_applies(FastPath::GdnTreeClosedForm, int8_24, auto_trees) &&
                          !q::fast_path_applies(FastPath::GdnTreeClosedForm, int8_24, dflash2) &&
                          !q::couples_requests(FastPath::GdnTreeClosedForm) &&
                          q::fast_path_qualified(FastPath::GdnTreeClosedForm, int8_24) &&
                          q::fast_path_qualified(FastPath::GdnTreeClosedForm,
                                                 {16, ActivationClass::A16, KvCacheStorage::Nvfp4Group16}) &&
                          !q::fast_path_qualified(FastPath::GdnTreeClosedForm, groupwise),
                      "gdn-tree-closed-form applies to every tree configuration, keeps a request invariant "
                      "and is qualified on the NVFP4 27B and the 35B-A3B");
    failures += check(infernix::parse_fast_path("gdn-tree-closed-form") == FastPath::GdnTreeClosedForm,
                      "gdn-tree-closed-form parses");
    // q5-ksplit-mma: the 16-bit-activation (groupwise) artifacts' Q5 residual projections, on by the owner's
    // decision where it measured faster (the groupwise 27B, INT8 KV) ahead of its gate; it keeps a request
    // invariant. fp16-gdn-state likewise there.
    failures += check(q::fast_path_applies(FastPath::Q5KSplitMma, groupwise, options),
                      "q5-ksplit-mma must apply to the groupwise configuration");
    failures += check(!q::fast_path_applies(FastPath::Q5KSplitMma,
                                            {24, ActivationClass::A4, KvCacheStorage::Int8Group64, GateUpFormat::Nvfp4}, options),
                      "q5-ksplit-mma must not apply to 4-bit-activation artifacts");
    // Both run on the KV formats below INT8's precision too (the owner's decision, 2026-11-01), and on BF16 where timed
    // faster (2026-11-19).
    failures += check(q::fast_path_qualified(FastPath::Q5KSplitMma, groupwise) &&
                          q::fast_path_qualified(FastPath::Q5KSplitMma,
                                                 {24, ActivationClass::A16, KvCacheStorage::BFloat16, GateUpFormat::GroupwiseQ4}) &&
                          q::fast_path_qualified(FastPath::Fp16GdnState,
                                                 {24, ActivationClass::A16, KvCacheStorage::BFloat16, GateUpFormat::GroupwiseQ4}),
                      "q5-ksplit-mma and fp16-gdn-state are qualified on the groupwise 27B with BF16 KV (timed faster)");
    failures += check(q::fast_path_qualified(FastPath::Fp16GdnState, groupwise),
                      "fp16-gdn-state is qualified on the groupwise 27B with INT8 KV");
    for (const KvCacheStorage kv : {KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                    KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::Q5KSplitMma, {24, ActivationClass::A16, kv, GateUpFormat::GroupwiseQ4}) &&
                              q::fast_path_qualified(FastPath::Fp16GdnState, {24, ActivationClass::A16, kv, GateUpFormat::GroupwiseQ4}) &&
                              q::fast_path_qualified(FastPath::Fp16GdnState, {24, ActivationClass::A4, kv, GateUpFormat::Nvfp4}),
                          "q5-ksplit-mma and fp16-gdn-state are qualified on the KV formats below INT8's precision");
    }
    failures += check(!q::fast_path_qualified(FastPath::Fp16GdnState, {16, ActivationClass::A16, KvCacheStorage::Vq2}),
                      "fp16-gdn-state stays off the 35B-A3B");
    failures += check(!q::couples_requests(FastPath::Q5KSplitMma),
                      "q5-ksplit-mma must not couple requests");

    // --a8-activations binds the text projections A8Only, 8-bit at every width: the column-invariant copy that decode and
    // verification run without batched-decode keeps them (until 2026-11-19 it made them A16ColumnInvariant, so the flag
    // decoded at 16 bits), while every other projection there takes its column-invariant route.
    {
        namespace ex = infernix::models::qwen3_5::execution;
        using P = infernix::ops::LinearPolicy;
        ex::TextParameters text;
        text.output_head.policy = P::AllowA8;
        ex::GdnParameters gdn;
        gdn.projection    = ex::LinearParameters{.policy = P::A8Only};
        gdn.output.policy = P::A8Only;
        ex::BlockParameters block;
        block.mixer = gdn;
        block.ffn   = ex::DenseParameters{.gate_up = {.policy = P::A8Only}, .down = {.policy = P::AllowA4}};
        text.layers.push_back(block);
        const ex::TextParameters invariant = ex::column_invariant_text(text);
        const auto& g = std::get<ex::GdnParameters>(invariant.layers.front().mixer);
        const auto& d = std::get<ex::DenseParameters>(invariant.layers.front().ffn);
        failures += check(std::get<ex::LinearParameters>(g.projection).policy == P::A8Only &&
                              g.output.policy == P::A8Only && d.gate_up.policy == P::A8Only &&
                              ex::projection_policy(g.control) == P::A16ColumnInvariant &&
                              d.down.policy == P::A16ColumnInvariant &&
                              invariant.output_head.policy == P::A16ColumnInvariant,
                          "the column-invariant decode copy keeps A8Only and makes the other projections invariant");
    }

    // The configuration of bound weights (fast_path_configuration's Model-free form). The gate/up format tells the NVFP4
    // 27B under --a16-activations, on which its rows and drafting constants were measured, from 16-bit 27Bs of the same
    // heads with another gate/up format (the Q6 gate/up recipe, a Q8_0 GGUF import): those get none of them.
    {
        using infernix::QType;
        using infernix::QuantLayout;
        using P = infernix::ops::LinearPolicy;
        const auto parent = [](QType format, QuantLayout layout) {
            infernix::WeightParent out;
            out.geometry.format = format;
            out.geometry.layout = layout;
            return out;
        };
        const infernix::WeightParent q4 = parent(QType::Q4_G64_FP16, QuantLayout::RowSplit),
                                     q5 = parent(QType::Q5_G64_FP16, QuantLayout::RowSplit),
                                     q6 = parent(QType::Q6_G64_FP16, QuantLayout::RowSplit),
                                     q8 = parent(QType::Q8_G32_FP16, QuantLayout::RowSplit),
                                     nvfp4 = parent(QType::NVFP4, QuantLayout::Contiguous),
                                     nvfp4_mul = parent(QType::NVFP4_MUL, QuantLayout::Contiguous),
                                     fp8 = parent(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::Contiguous);
        const auto bind = [](std::string name, const infernix::WeightParent& weights, P policy) {
            return q::BoundWeight{.name = std::move(name),
                                  .view = {.shape = {1, 1}, .parts = {{&weights, 0, 1}}},
                                  .uses = {{.input = "x", .policy = policy}}};
        };
        infernix::EngineOptions int8 = options;
        int8.kv_cache                = KvCacheStorage::Int8Group64;
        // Text layers of an attention projection and an MLP (gate/up from `gate_up[layer]`, down like the projection).
        const auto configuration = [&](const infernix::WeightParent& projection,
                                       std::initializer_list<const infernix::WeightParent*> gate_up, P projection_policy,
                                       P mlp_policy) {
            std::vector<q::BoundWeight> bound;
            q::TextWeights text;
            std::size_t layer = 0;
            for (const infernix::WeightParent* mlp : gate_up) {
                const std::string prefix = "text/layers/" + std::to_string(layer++) + "/";
                const std::size_t first  = bound.size();
                bound.push_back(bind(prefix + "attention/query", projection, projection_policy));
                bound.push_back(bind(prefix + "mlp/gate", *mlp, mlp_policy));
                bound.push_back(bind(prefix + "mlp/up", *mlp, mlp_policy));
                bound.push_back(bind(prefix + "mlp/down", projection, mlp_policy));
                text.layers.push_back(q::BlockWeights{
                    .ffn = q::DenseWeights{q::WeightId{first + 1}, q::WeightId{first + 2}, q::WeightId{first + 3}}});
            }
            return q::fast_path_configuration(bound, text, 24, false, int8);
        };
        // NVIDIA's NVFP4 27B (NVFP4 with a multiplier beside FP8 projections) and NInfer's (NVFP4 with a divisor);
        // --a16-activations binds every Use A16Only.
        const FastPathConfiguration nvidia    = configuration(fp8, {&nvfp4_mul}, P::AllowA8, P::AllowA4);
        const FastPathConfiguration nvidia_16 = configuration(fp8, {&nvfp4_mul}, P::A16Only, P::A16Only);
        const FastPathConfiguration ninfer_16 = configuration(fp8, {&nvfp4}, P::A16Only, P::A16Only);
        // orcarouter's NVFP4 27B: FP8 in its last MLPs.
        const FastPathConfiguration orcarouter = configuration(fp8, {&nvfp4, &fp8}, P::AllowA8, P::AllowA4);
        const FastPathConfiguration q6_recipe  = configuration(q5, {&q6}, P::A16Only, P::A16Only);
        const FastPathConfiguration gguf_q8_0  = configuration(q8, {&q8}, P::A16Only, P::A16Only);
        const FastPathConfiguration groupwise_recipe = configuration(q5, {&q4}, P::A16Only, P::A16Only);
        const FastPathConfiguration nvfp4_16{24, ActivationClass::A16, KvCacheStorage::Int8Group64, GateUpFormat::Nvfp4};
        const FastPathConfiguration other_16{24, ActivationClass::A16, KvCacheStorage::Int8Group64, GateUpFormat::Other};
        failures += check(nvidia == FastPathConfiguration{24, ActivationClass::A4, KvCacheStorage::Int8Group64,
                                                          GateUpFormat::Nvfp4} &&
                              orcarouter == nvidia && nvidia_16 == nvfp4_16 && ninfer_16 == nvfp4_16,
                          "the NVFP4 27Bs are configurations of NVFP4 gate/up");
        failures += check(q6_recipe == other_16 && gguf_q8_0 == other_16,
                          "a 16-bit 27B of a Q6 or Q8 gate/up is not the NVFP4 27B's configuration");
        failures += check(groupwise_recipe == FastPathConfiguration{24, ActivationClass::A16, KvCacheStorage::Int8Group64,
                                                                    GateUpFormat::GroupwiseQ4},
                          "the groupwise recipe is a configuration of Q4 gate/up");
        failures += check(q::describe(nvidia_16) == "24 heads, 16-bit activations, int8 KV, NVFP4 gate/up" &&
                              q::describe(q6_recipe) == "24 heads, 16-bit activations, int8 KV",
                          "the gate/up format is described");
        for (const FastPath path : {FastPath::BatchedDecode, FastPath::Fp16GdnState, FastPath::IdleWidePrefill,
                                    FastPath::GdnF16Accumulate, FastPath::PromptPv8, FastPath::AdaptiveSpeculation,
                                    FastPath::FreeCacheResume, FastPath::Preemption, FastPath::DecodeSplit64}) {
            failures += check(q::fast_path_qualified(path, nvidia_16) && !q::fast_path_qualified(path, q6_recipe) &&
                                  !q::fast_path_qualified(path, gguf_q8_0),
                              std::string(infernix::fast_path_name(path)) +
                                  " is registered for the NVFP4 27B under --a16-activations only, not a Q6 or Q8 "
                                  "gate/up 27B");
        }
        failures += check(q::auto_fixed_draft_tokens(nvidia_16, SpeculativeBackend::Mtp) == 5 &&
                              q::auto_fixed_draft_tokens(q6_recipe, SpeculativeBackend::Mtp) == 3 &&
                              q::dflash_width_costs(q6_recipe, SpeculativeBackend::DFlash2).proposal == 0.69 &&
                              q::dflash_batch_drafts(q6_recipe, SpeculativeBackend::DFlash2) == 0 &&
                              q::dflash_batch_drafts(nvidia_16, SpeculativeBackend::DFlash2) == 7,
                          "a Q6 gate/up 27B keeps the generic drafting constants, not the NVFP4 27B's");
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " fast-path selection\n";
    return failures == 0 ? 0 : 1;
}
