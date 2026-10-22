#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/program/speculative/dflash_width_policy.h"
#include "infernix/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace infernix::models::qwen3_5 {

// The narrowest activation precision any text-decoder linear computes with.
enum class ActivationClass : std::uint8_t { A4, A8, A16 };

// What a fast path is qualified by: the precision that sets the default route's noise floor
// (activations and KV format), the attention geometry the kernels run in, and whether the text MLP
// gate/up weights are groupwise Q4, the format the q4-w4a8 kernels run on
// (docs/maintainer/model-quality.md section 6).
struct FastPathConfiguration {
    std::uint32_t query_heads   = 0;
    ActivationClass activations = ActivationClass::A16;
    KvCacheStorage kv           = KvCacheStorage::BFloat16;
    bool q4_gate_up             = false;
    // Routed experts held outside device memory (Qwen3.5-122B-A10B): a verified draft column brings
    // its own expert misses, which prices speculation differently.
    bool offloaded_experts = false;

    bool operator==(const FastPathConfiguration&) const = default;
};

// For example "24 heads, 4-bit activations, int8 KV" (", Q4 gate/up" appended when set).
[[nodiscard]] std::string describe(const FastPathConfiguration& configuration);

[[nodiscard]] FastPathConfiguration fast_path_configuration(const Model& model,
                                                            const EngineOptions& options);

// Whether the path's kernels run in this configuration at all. The request-coupling paths give up a
// part of the exact route's invariance, which every artifact has (every decode projection runs a
// column-invariant route, or its exact route once per column). adaptive-mtp, adaptive-dflash and
// idle-wide-prefill couple requests with behaviour no exact route runs.
[[nodiscard]] bool fast_path_applies(FastPath path, const FastPathConfiguration& configuration,
                                     const EngineOptions& options);

// A path that couples a request's arithmetic to the other requests of the Engine.
[[nodiscard]] constexpr bool couples_requests(FastPath path) noexcept {
    switch (path) {
    case FastPath::PromptPv8:
    case FastPath::DecodeSplit64:
    case FastPath::Q8SlicedK:
    case FastPath::GdnConvFused:
    case FastPath::QsaPrepareFused:
    case FastPath::Fp8A8Pair:
    case FastPath::Q4W4A8:
    case FastPath::SampledMtpDrafts:
    case FastPath::GdnF16Accumulate: return false;
    case FastPath::ExpertA4Residual:
    case FastPath::ExpertA4Residual3:
    case FastPath::BatchedDecode:
    case FastPath::BatchedA8Decode:
    case FastPath::BatchedA4Decode:
    case FastPath::BatchedA4ResidualDecode:
    case FastPath::BatchedA8ResidualDecode:
    case FastPath::SingleColumnDecode:
    case FastPath::BatchSplitAttention:
    case FastPath::AdaptiveSpeculation:
    case FastPath::FreeCacheResume:
    case FastPath::Preemption:
    case FastPath::AdaptiveMtp:
    case FastPath::AdaptiveDFlash:
    case FastPath::IdleWidePrefill:
    case FastPath::GgmlBf16Prefill: return true;
    }
    return false;
}

// Whether the path passed the acceptance gate in this configuration (the registry, section 6.7).
[[nodiscard]] bool fast_path_qualified(FastPath path, const FastPathConfiguration& configuration);

// Whether --draft-tokens auto decodes without a drafter: where no draft length was measured faster
// than plain decode for either drafter. The engine then turns speculation off before it binds the
// drafter (normalize_engine_options), so the drafter's VRAM stays with the expert cache.
[[nodiscard]] bool auto_drafts_nothing(bool offloaded_experts) noexcept;

// The fixed draft length --draft-tokens auto runs where its policy (adaptive-mtp, adaptive-dflash)
// does not (not qualified for the configuration, or fast paths disabled): the single length
// measured fastest overall for the configuration and drafter (section 6.7).
[[nodiscard]] std::uint32_t auto_fixed_draft_tokens(const FastPathConfiguration& configuration,
                                                    SpeculativeBackend backend) noexcept;

// The block a DFlash or DFlash2 --draft-tokens auto round proposes (its ceiling), unless given: the
// 27B's DFlash2 15, the 35B-A3B's DFlash 7 (its proposals of 16 columns cost more than the drafts
// they add; section 6.7, adaptive-dflash).
[[nodiscard]] std::uint32_t dflash_auto_ceiling(const FastPathConfiguration& configuration,
                                                SpeculativeBackend backend) noexcept;

// adaptive-dflash's round costs for the configuration and block drafter, in plain rounds: the
// proposal of the whole block and each verified draft (dflash_width_policy.h), measured per
// configuration (section 6.7, adaptive-dflash).
[[nodiscard]] detail::DFlashWidthCosts dflash_width_costs(const FastPathConfiguration& configuration,
                                                          SpeculativeBackend backend) noexcept;

// adaptive-dflash's batch length: rounds of several rows run plain neural rounds at this many
// drafts (one graph, no host wait) and the width policy chooses for one row only; 0 leaves
// batched rounds to the policy too (section 6.7, adaptive-dflash).
[[nodiscard]] std::uint32_t dflash_batch_drafts(const FastPathConfiguration& configuration,
                                                SpeculativeBackend backend) noexcept;

// --ngram-draft-tokens auto's copy window (0: no copies), the window of rounds of several requests
// (0: the same) and minimum match.
struct NgramSettings {
    std::uint32_t window       = 0;
    std::uint32_t batch_window = 0;
    std::uint32_t min_match    = 12;

    bool operator==(const NgramSettings&) const = default;
};

// The configuration's measured n-gram settings beside its drafter, where speculation is adaptive
// (copy rounds take the narrowest copy family; with fixed speculation every round would verify
// the window, and auto keeps copies off). `adaptive_drafts`: --draft-tokens auto's policy acts
// (adaptive-mtp or adaptive-dflash), so neural rounds draft their own lengths (section 6.7).
[[nodiscard]] NgramSettings auto_ngram_settings(const FastPathConfiguration& configuration,
                                                SpeculativeBackend backend, bool adaptive_drafts) noexcept;

struct FastPathSelection {
    FastPathConfiguration configuration;
    // In kFastPaths order.
    std::vector<FastPath> paths;

    [[nodiscard]] bool selected(FastPath path) const noexcept;
    // Request invariance holds for `path`'s concern: the path is not selected.
    [[nodiscard]] bool invariant(FastPath path) const noexcept { return !selected(path); }
    // A request's outputs depend only on the request, the artifact, the options and the build.
    [[nodiscard]] bool request_invariant() const noexcept;
};

// The qualified paths that apply when options.fast_paths is set, plus every
// options.fast_path_candidates entry (qualification runs), which must apply.
[[nodiscard]] FastPathSelection select_fast_paths(const Model& model, const EngineOptions& options);

} // namespace infernix::models::qwen3_5
