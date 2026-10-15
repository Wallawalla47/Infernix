// Qwen3.8-Flash-Next fast-path selection (EngineOptions::fast_paths): which numerics-changing routes
// run for a loaded artifact, from the registry of configurations each passed the acceptance gate in
// (docs/maintainer/model-quality.md section 6.7).
#include "models/qwen4_exp/program/fast_paths.h"

#include "models/qwen4_exp/execution/parameters.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <variant>

namespace infernix::models::qwen4_exp {
namespace {

struct Qualification {
    FastPath path;
    FastPathConfiguration configuration;
};

// Every configuration a fast path passed the acceptance gate in; the evidence is in
// docs/maintainer/model-quality.md section 6.7. A path runs only in a listed configuration.
constexpr std::array kQualified{
    // NVIDIA NVFP4 Dense8, INT8 KV: MTP at concurrency 2, and off-grid and endpoint cache resumes
    // (2026-10-14).
    Qualification{FastPath::AdaptiveSpeculation, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::FreeCacheResume, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
    // Preemption with Replay under a 388-page pool that certainly runs short (2026-10-14).
    Qualification{FastPath::Preemption, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
    // NInfer's Q8 dense tables, fused GDN convolution and fused QSA prepare, jointly: prefill (quick
    // corpus, four chunk sizes) and decode (plain at concurrency 8, MTP at 2) (2026-10-15).
    Qualification{FastPath::Q8SlicedK, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::GdnConvFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::QsaPrepareFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
};

const char* dense_name(QType dense) {
    switch (dense) {
    case QType::Q8_G32_FP16: return "Q8";
    case QType::BF16: return "BF16";
    case QType::FP8_E4M3FN_ROW_BF16: return "FP8";
    case QType::NVFP4: return "NVFP4";
    default: return "other";
    }
}

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

} // namespace

std::string describe(const FastPathConfiguration& configuration) {
    return std::string(dense_name(configuration.dense)) + " dense, " +
           (configuration.a4_experts ? "4-bit" : "16-bit") + " expert activations, " +
           kv_name(configuration.kv) + " KV";
}

FastPathConfiguration fast_path_configuration(const execution::Parameters& parameters,
                                              const EngineOptions& options) {
    FastPathConfiguration out{.dense      = QType::BF16,
                              .a4_experts = !parameters.model.config().text.moe.a16_experts,
                              .kv         = options.kv_cache};
    // The dense layers share one format; the GDN q/k/v/z projection is the one the fast routes serve.
    for (const execution::BlockParameters& block : parameters.layers) {
        if (const auto* gdn = std::get_if<execution::GdnParameters>(&block.mixer)) {
            out.dense = gdn->projection.weight.qtype;
            break;
        }
    }
    return out;
}

bool fast_path_applies(FastPath path, const FastPathConfiguration& configuration,
                       const EngineOptions& options) {
    switch (path) {
    // The request-coupling behaviours: two-row speculative rounds, free cache resume, preemption.
    // Flash-Next's decode is already request-invariant at every batch, so the batched-decode and
    // attention paths have nothing to change.
    case FastPath::AdaptiveSpeculation: return options.speculative.backend == SpeculativeBackend::Mtp;
    case FastPath::FreeCacheResume: return options.context_cache.enabled;
    case FastPath::Preemption: return true;
    case FastPath::BatchedDecode:
    case FastPath::BatchedA8Decode:
    case FastPath::BatchedA4Decode:
    case FastPath::BatchedA4ResidualDecode:
    case FastPath::BatchedA8ResidualDecode:
    case FastPath::BatchSplitAttention: return false;
    case FastPath::PromptPv8:
    case FastPath::DecodeSplit64: return false; // Qwen3.8-27B attention kernels only
    // Flash-Next chooses its draft lengths and prefill widths itself; groupwise Q4 is not a dense format.
    case FastPath::AdaptiveMtp:
    case FastPath::IdleWidePrefill:
    case FastPath::Q4W4A8: return false;
    // Recipe C's row-FP8 dense projections run 16-bit activations (A16Only).
    case FastPath::Fp8A8Pair: return configuration.dense == QType::FP8_E4M3FN_ROW_BF16;
    case FastPath::Q8SlicedK:
    case FastPath::GdnConvFused: return configuration.dense == QType::Q8_G32_FP16;
    case FastPath::QsaPrepareFused:
        // ops::qsa_prepare_supports's KV formats (bf16, int8, fp8, nvfp4, k8v4); VQ2 and K4V2 append
        // inside qsa_attention and keep the composed prepare.
        return configuration.kv == KvCacheStorage::Int8Group64 || configuration.kv == KvCacheStorage::BFloat16 ||
               configuration.kv == KvCacheStorage::Fp8E4M3Row256 || configuration.kv == KvCacheStorage::Nvfp4Group16 ||
               configuration.kv == KvCacheStorage::Fp8KeyNvfp4Value;
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
    return std::none_of(paths.begin(), paths.end(), [](FastPath path) {
        // q8-sliced-k picks its reduction order by the round's column count, so a decode column's bits
        // depend on the other requests of its round.
        return path == FastPath::AdaptiveSpeculation || path == FastPath::FreeCacheResume ||
               path == FastPath::Preemption || path == FastPath::Q8SlicedK;
    });
}

execution::ForwardFastPaths FastPathSelection::forward() const noexcept {
    ops::LinearRoutes routes = ops::LinearRoutes::Exact;
    if (selected(FastPath::Q8SlicedK)) { routes = routes | ops::LinearRoutes::Q8SlicedK; }
    if (selected(FastPath::Fp8A8Pair)) { routes = routes | ops::LinearRoutes::Fp8A8Pair; }
    return {.dense_routes      = routes,
            .gdn_conv_fused    = selected(FastPath::GdnConvFused),
            .qsa_prepare_fused = selected(FastPath::QsaPrepareFused)};
}

FastPathSelection select_fast_paths(const execution::Parameters& parameters, const EngineOptions& options) {
    FastPathSelection out{.configuration = fast_path_configuration(parameters, options), .paths = {}};
    for (const FastPath path : options.fast_path_candidates) {
        if (!fast_path_applies(path, out.configuration, options)) {
            throw std::invalid_argument(std::string("fast path ") + fast_path_name(path) +
                                        " does not apply to " + describe(out.configuration));
        }
    }
    for (const FastPath path : kFastPaths) {
        const bool candidate = std::find(options.fast_path_candidates.begin(),
                                         options.fast_path_candidates.end(),
                                         path) != options.fast_path_candidates.end();
        const bool qualified = options.fast_paths && fast_path_applies(path, out.configuration, options) &&
                               fast_path_qualified(path, out.configuration);
        if (candidate || qualified) { out.paths.push_back(path); }
    }
    return out;
}

} // namespace infernix::models::qwen4_exp
