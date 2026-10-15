// Fast-path selection (EngineOptions::fast_paths): which numerics-changing routes run for a loaded
// model, from the registry of configurations each passed the acceptance gate in
// (docs/maintainer/model-quality.md section 6.7).
#include "models/qwen3_5/program/planning/fast_paths.h"

#include "infernix/ops/linear.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace infernix::models::qwen3_5 {
namespace {

struct Qualification {
    FastPath path;
    FastPathConfiguration configuration;
};

// Every configuration a fast path passed the acceptance gate in; the evidence is in
// docs/maintainer/model-quality.md section 6.7. A path runs only in a listed configuration.
constexpr std::array kQualified{
    Qualification{FastPath::DecodeSplit64, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // The request-coupling paths, qualified jointly per KV format at concurrency 8, DFlash2 with
    // n-gram copies at concurrency 4, and off-grid and endpoint cache resumes (2026-10-14).
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::BatchSplitAttention, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::BatchSplitAttention, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::BatchSplitAttention,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::BatchSplitAttention, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveSpeculation,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // Two-term A4 residual MLP activations in wide rounds, jointly with the coupling paths above
    // (DFlash2 with n-gram copies at concurrency 4, 2026-10-14).
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // Two-term A8 residual activations in the FP8 input projections' wide rounds: gated on INT8
    // and NVFP4 KV (the least headroom), K8V4 and K4V2 by their bounds plus INT8's increment
    // (2026-10-15).
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // Wider prefill steps while no lane waits to decode: the idle width (3,584) against 896- and
    // 1,792-token chunks, prefill gate, quick corpus (2026-10-15).
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // Preemption with Replay under a two-lane pool at concurrency 8 (2026-10-14).
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
};

const char* kv_name(KvCacheStorage kv) {
    switch (kv) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value: return "k8v4";
    case KvCacheStorage::Vq2: return "vq2";
    case KvCacheStorage::Q4KeyVq2Value: return "k4v2";
    }
    return "unknown";
}

const char* activation_name(ActivationClass activations) {
    switch (activations) {
    case ActivationClass::A4: return "4-bit";
    case ActivationClass::A8: return "8-bit";
    case ActivationClass::A16: return "16-bit";
    }
    return "unknown";
}

} // namespace

std::string describe(const FastPathConfiguration& configuration) {
    return std::to_string(configuration.query_heads) + " heads, " +
           activation_name(configuration.activations) + " activations, " +
           kv_name(configuration.kv) + " KV" + (configuration.q4_gate_up ? ", Q4 gate/up" : "");
}

FastPathConfiguration fast_path_configuration(const Model& model, const EngineOptions& options) {
    ActivationClass activations = ActivationClass::A16;
    // Text-decoder weights only: drafter and vision weights change proposals and encodings, not
    // the precision of the decoder's own arithmetic.
    for (const BoundWeight& weight : model.weight_data()) {
        if (!std::string_view(weight.name).starts_with("text/")) { continue; }
        for (const WeightUse& use : weight.uses) {
            if (ops::allows_a4(use.policy)) {
                activations = ActivationClass::A4;
            } else if (ops::allows_a8(use.policy) && activations == ActivationClass::A16) {
                activations = ActivationClass::A8;
            }
        }
    }
    // A text layer whose dense MLP gate and up are both RowSplit Q4_G64_FP16 (the groupwise recipes).
    const auto groupwise_q4 = [&](WeightId id) {
        const WeightView& view = model.weight(id).view;
        return !view.parts.empty() &&
               std::all_of(view.parts.begin(), view.parts.end(), [](const WeightRegion& part) {
                   return part.parent != nullptr &&
                          part.parent->geometry.format == QType::Q4_G64_FP16 &&
                          part.parent->geometry.layout == QuantLayout::RowSplit;
               });
    };
    bool q4_gate_up = false;
    for (const BlockWeights& layer : model.weights().text.layers) {
        if (const auto* dense = std::get_if<DenseWeights>(&layer.ffn)) {
            q4_gate_up = q4_gate_up || (groupwise_q4(dense->gate) && groupwise_q4(dense->up));
        }
    }
    const auto& attention = model.config().text.attention;
    return {.query_heads = attention ? static_cast<std::uint32_t>(attention->num_attention_heads) : 0U,
            .activations = activations,
            .kv          = options.kv_cache,
            .q4_gate_up  = q4_gate_up};
}

bool fast_path_applies(FastPath path, const FastPathConfiguration& configuration,
                       const EngineOptions& options, bool invariant_capable) {
    switch (path) {
    case FastPath::BatchedDecode:
    case FastPath::BatchSplitAttention:
    case FastPath::Preemption: return invariant_capable;
    case FastPath::BatchedA8Decode:
    case FastPath::BatchedA8ResidualDecode:
        return invariant_capable && configuration.activations != ActivationClass::A16;
    case FastPath::BatchedA4Decode:
    case FastPath::BatchedA4ResidualDecode:
        return invariant_capable && configuration.activations == ActivationClass::A4;
    case FastPath::AdaptiveSpeculation:
        return invariant_capable && options.speculative.backend != SpeculativeBackend::None;
    case FastPath::FreeCacheResume: return invariant_capable && options.context_cache.enabled;
    case FastPath::PromptPv8:
        switch (configuration.kv) {
        case KvCacheStorage::Int8Group64: return !options.original_int8_prefill_kernel;
        case KvCacheStorage::Nvfp4Group16: return !options.original_nvfp4_prefill_kernel;
        case KvCacheStorage::Fp8KeyNvfp4Value:
        case KvCacheStorage::Vq2:
        case KvCacheStorage::Q4KeyVq2Value: return true;
        case KvCacheStorage::BFloat16:
        case KvCacheStorage::Fp8E4M3Row256: return false;
        }
        return false;
    case FastPath::DecodeSplit64:
        return configuration.kv == KvCacheStorage::Int8Group64 && configuration.query_heads == 24;
    case FastPath::Q8SlicedK:
    case FastPath::GdnConvFused:
    case FastPath::QsaPrepareFused: return false; // Qwen3.8-Flash-Next kernels only
    // It runs only where speculation is adaptive (adaptive-speculation selected, or an artifact the
    // exact route cannot make request-invariant), see ExecutionSchedule::adaptive_mtp.
    case FastPath::AdaptiveMtp: return options.speculative.backend == SpeculativeBackend::Mtp;
    // Causal scoring has no idle steps (each window's chunks are the caller's); the path is a no-op
    // there, which lets the gate name it while its fast run prefills at --fast-chunk.
    case FastPath::IdleWidePrefill: return true;
    // The FP8 sites of the Qwen3.8-27B recipes already permit single-term A8 at prefill widths.
    case FastPath::Fp8A8Pair: return false;
    // The W4A8 gate/up kernel runs only on groupwise Q4 weights (the text MLP's linear_swiglu).
    case FastPath::Q4W4A8: return configuration.q4_gate_up;
    }
    return false;
}

bool fast_path_qualified(FastPath path, const FastPathConfiguration& configuration) {
    return std::any_of(kQualified.begin(), kQualified.end(), [&](const Qualification& entry) {
        return entry.path == path && entry.configuration == configuration;
    });
}

bool FastPathSelection::selected(FastPath path) const noexcept {
    return std::find(paths.begin(), paths.end(), path) != paths.end();
}

bool FastPathSelection::request_invariant() const noexcept {
    return invariant_capable && std::none_of(paths.begin(), paths.end(), couples_requests);
}

FastPathSelection select_fast_paths(const Model& model, const EngineOptions& options,
                                    bool invariant_capable) {
    FastPathSelection out{.configuration     = fast_path_configuration(model, options),
                          .paths             = {},
                          .invariant_capable = invariant_capable};
    for (const FastPath path : options.fast_path_candidates) {
        if (!fast_path_applies(path, out.configuration, options, invariant_capable)) {
            throw std::invalid_argument(std::string("fast path ") + fast_path_name(path) +
                                        " does not apply to " + describe(out.configuration));
        }
    }
    for (const FastPath path : kFastPaths) {
        const bool candidate = std::find(options.fast_path_candidates.begin(),
                                         options.fast_path_candidates.end(),
                                         path) != options.fast_path_candidates.end();
        const bool qualified = options.fast_paths &&
                               fast_path_applies(path, out.configuration, options,
                                                 invariant_capable) &&
                               fast_path_qualified(path, out.configuration);
        if (candidate || qualified) { out.paths.push_back(path); }
    }
    return out;
}

} // namespace infernix::models::qwen3_5
