#pragma once

#include "models/qwen3_5/model.h"
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

    bool operator==(const FastPathConfiguration&) const = default;
};

// For example "24 heads, 4-bit activations, int8 KV" (", Q4 gate/up" appended when set).
[[nodiscard]] std::string describe(const FastPathConfiguration& configuration);

[[nodiscard]] FastPathConfiguration fast_path_configuration(const Model& model,
                                                            const EngineOptions& options);

// Whether the path's kernels run in this configuration at all. The request-coupling paths that give
// up a part of the exact route's invariance (batched-decode through preemption) apply only where the
// exact route can be request-invariant (`invariant_capable`: every decode projection registers a
// column-invariant route, execution::column_invariant_text_supported); elsewhere their behaviour is
// the exact route's. adaptive-mtp and idle-wide-prefill couple requests with behaviour no exact
// route runs, so they apply either way.
[[nodiscard]] bool fast_path_applies(FastPath path, const FastPathConfiguration& configuration,
                                     const EngineOptions& options, bool invariant_capable);

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
    case FastPath::BatchedDecode:
    case FastPath::BatchedA8Decode:
    case FastPath::BatchedA4Decode:
    case FastPath::BatchedA4ResidualDecode:
    case FastPath::BatchedA8ResidualDecode:
    case FastPath::BatchSplitAttention:
    case FastPath::AdaptiveSpeculation:
    case FastPath::FreeCacheResume:
    case FastPath::Preemption:
    case FastPath::AdaptiveMtp:
    case FastPath::IdleWidePrefill:
    case FastPath::GgmlBf16Prefill: return true;
    }
    return false;
}

// Whether the path passed the acceptance gate in this configuration (the registry, section 6.7).
[[nodiscard]] bool fast_path_qualified(FastPath path, const FastPathConfiguration& configuration);

// The fixed MTP draft length --draft-tokens auto runs where the adaptive-mtp policy does not (not
// qualified for the configuration, or fast paths disabled): the single length measured fastest
// overall for the configuration (section 6.7, adaptive-mtp), else 3.
[[nodiscard]] std::uint32_t auto_fixed_draft_tokens(const FastPathConfiguration& configuration) noexcept;

struct FastPathSelection {
    FastPathConfiguration configuration;
    // In kFastPaths order.
    std::vector<FastPath> paths;
    bool invariant_capable = false;

    [[nodiscard]] bool selected(FastPath path) const noexcept;
    // Request invariance holds for `path`'s concern: the Model supports it and the path is not
    // selected.
    [[nodiscard]] bool invariant(FastPath path) const noexcept {
        return invariant_capable && !selected(path);
    }
    // A request's outputs depend only on the request, the artifact, the options and the build.
    [[nodiscard]] bool request_invariant() const noexcept;
};

// The qualified paths that apply when options.fast_paths is set, plus every
// options.fast_path_candidates entry (qualification runs), which must apply.
[[nodiscard]] FastPathSelection select_fast_paths(const Model& model, const EngineOptions& options,
                                                  bool invariant_capable);

} // namespace infernix::models::qwen3_5
