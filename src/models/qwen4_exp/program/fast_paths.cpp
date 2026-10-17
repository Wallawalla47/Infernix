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
    // The other formats (2026-10-16/17, 256 segments): MTP at concurrency 2 with every request resuming
    // from a warm 3,000-token entry, against the exact route; preemption by its increment over the
    // qualified paths under the same short pool (model-quality.md 6.1). FP8 and K8V4 failed the
    // joint arm narrowly at 256 segments and passed it at 1,024 (2026-10-17).
    Qualification{FastPath::AdaptiveSpeculation, {QType::Q8_G32_FP16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::FreeCacheResume, {QType::Q8_G32_FP16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::Preemption, {QType::Q8_G32_FP16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveSpeculation, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::FreeCacheResume, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Preemption, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveSpeculation, {QType::Q8_G32_FP16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::FreeCacheResume, {QType::Q8_G32_FP16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Preemption, {QType::Q8_G32_FP16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveSpeculation, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::FreeCacheResume, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Preemption, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveSpeculation, {QType::Q8_G32_FP16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::FreeCacheResume, {QType::Q8_G32_FP16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::Preemption, {QType::Q8_G32_FP16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveSpeculation, {QType::Q8_G32_FP16, true, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::FreeCacheResume, {QType::Q8_G32_FP16, true, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::Preemption, {QType::Q8_G32_FP16, true, KvCacheStorage::Q4KeyVq2Value}},
    // NInfer's Q8 dense tables, fused GDN convolution and fused QSA prepare, jointly: prefill (quick
    // corpus, four chunk sizes) and decode (plain at concurrency 8, MTP at 2) (2026-10-15).
    Qualification{FastPath::Q8SlicedK, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::GdnConvFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::QsaPrepareFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
    // The same three on the other formats qsa_prepare serves: prefill (quick corpus, 896 and 7,168
    // tokens) on each, decode plain on BF16 (the lowest floor) and NVFP4 (the lossiest KV) (2026-10-15).
    Qualification{FastPath::Q8SlicedK, {QType::Q8_G32_FP16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::Q8SlicedK, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Q8SlicedK, {QType::Q8_G32_FP16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Q8SlicedK, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::GdnConvFused, {QType::Q8_G32_FP16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::GdnConvFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::GdnConvFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::GdnConvFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::QsaPrepareFused, {QType::Q8_G32_FP16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::QsaPrepareFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::QsaPrepareFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::QsaPrepareFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    // VQ2 and K4V2 keep the composed QSA prepare; the dense paths leave attention and the KV cache alone,
    // so they are registered from the INT8 increments (model-quality.md 6.1, step 3).
    Qualification{FastPath::Q8SlicedK, {QType::Q8_G32_FP16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::Q8SlicedK, {QType::Q8_G32_FP16, true, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::GdnConvFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::GdnConvFused, {QType::Q8_G32_FP16, true, KvCacheStorage::Q4KeyVq2Value}},
    // Recipe C (FP8 dense, W4A16 experts): two-term A4 expert activations. Prefill (quick corpus,
    // 896 and 7,168 tokens) on every format, decode (MTP at concurrency 2, 256 segments) on INT8, the
    // passing format with the lowest floor; the path leaves attention and the KV cache alone, so the
    // others are registered from the INT8 increments (model-quality.md 6.1, step 3). BF16 KV fails:
    // there recipe C's exact route is the reference itself (floor 0) (2026-10-15).
    Qualification{FastPath::ExpertA4Residual, {QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::ExpertA4Residual, {QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::ExpertA4Residual, {QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::ExpertA4Residual, {QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::ExpertA4Residual, {QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::ExpertA4Residual, {QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::Q4KeyVq2Value}},
    // Recipe C: fp8-a8-pair (power-of-two column scales) beside two-term expert-a4-residual,
    // decode gated on each format directly (MTP at concurrency 2, 256 segments, 2026-10-17), since
    // INT8, the lowest floor, fails its tail. Prefill passes on every format but BF16; FP8, NVFP4
    // and K4V2 await their 1,024-segment decode gates (model-quality.md 6.7).
    Qualification{FastPath::Fp8A8Pair, {QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Fp8A8Pair, {QType::FP8_E4M3FN_ROW_BF16, false, KvCacheStorage::Vq2}},
    // NVIDIA NVFP4 Dense8: sampled MTP drafts (2026-10-17). They change no logit, so the gate is the
    // sampler's distribution (2,000 seeded requests per arm, chi-square per position against plain
    // decoding, INT8 KV) and greedy identity (INT8, NVFP4), which no KV format affects; speed on
    // INT8 (MTP 4, T 1.0 / top-p 0.95 / top-k 20, geomean +2.7 %, 5 of 6 cells).
    Qualification{FastPath::SampledMtpDrafts, {QType::Q8_G32_FP16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::SampledMtpDrafts, {QType::Q8_G32_FP16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::SampledMtpDrafts, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::SampledMtpDrafts, {QType::Q8_G32_FP16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::SampledMtpDrafts, {QType::Q8_G32_FP16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::SampledMtpDrafts, {QType::Q8_G32_FP16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::SampledMtpDrafts, {QType::Q8_G32_FP16, true, KvCacheStorage::Q4KeyVq2Value}},
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
    // The drafter's sampled proposals (the program checks the sparse acceptance's token domain).
    case FastPath::SampledMtpDrafts: return options.speculative.backend == SpeculativeBackend::Mtp;
    // The GDN layers normalize Q/K; their chunked calls serve prefill.
    case FastPath::GdnF16Accumulate: return true;
    // Recipe C's row-FP8 dense projections run 16-bit activations (A16Only).
    case FastPath::Fp8A8Pair: return configuration.dense == QType::FP8_E4M3FN_ROW_BF16;
    case FastPath::Q8SlicedK:
    case FastPath::GdnConvFused: return configuration.dense == QType::Q8_G32_FP16;
    // The wide route of W4A16 experts (weight-only exports such as recipe C).
    case FastPath::ExpertA4Residual: return !configuration.a4_experts;
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
        // expert-a4-residual routes an expert, and fp8-a8-pair a dense problem, by its column count in
        // the call, which batched rounds share between requests.
        return path == FastPath::AdaptiveSpeculation || path == FastPath::FreeCacheResume ||
               path == FastPath::Preemption || path == FastPath::Q8SlicedK ||
               path == FastPath::ExpertA4Residual || path == FastPath::Fp8A8Pair;
    });
}

execution::ForwardFastPaths FastPathSelection::forward() const noexcept {
    ops::LinearRoutes routes = ops::LinearRoutes::Exact;
    if (selected(FastPath::Q8SlicedK)) { routes = routes | ops::LinearRoutes::Q8SlicedK; }
    if (selected(FastPath::Fp8A8Pair)) { routes = routes | ops::LinearRoutes::Fp8A8Pair; }
    return {.dense_routes       = routes,
            .gdn_conv_fused     = selected(FastPath::GdnConvFused),
            .qsa_prepare_fused  = selected(FastPath::QsaPrepareFused),
            .expert_a4_residual = selected(FastPath::ExpertA4Residual),
            .gdn_arithmetic     = selected(FastPath::GdnF16Accumulate) ? ops::GdnChunkArithmetic::Fp16Accumulate
                                                                       : ops::GdnChunkArithmetic::Exact};
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
