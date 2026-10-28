#pragma once

#include "core/weight.h"
#include "models/qwen4_exp/execution/fast_path_flags.h"
#include "infernix/types.h"

#include <cstddef>
#include <string>
#include <vector>

namespace infernix::models::qwen4_exp {

namespace execution {
class Parameters;
}

// The dense kernels an artifact's GDN q/k/v/z projection runs: one family per kernel set, so every
// GGUF block format (one exact A16 kernel family, design 19.3.25) is one configuration.
enum class DenseKernels : std::uint8_t { Bf16, Q8, Fp8Row, Nvfp4, Ggml, Mxfp8, Other };

[[nodiscard]] DenseKernels dense_kernels(QType qtype) noexcept;

// What a Qwen3.8-Flash-Next fast path is qualified by (docs/maintainer/model-quality.md section 6):
// the kernels it runs on (the dense family and whether the routed experts take 4-bit activations)
// and the KV format. Approval follows the kernels (6.1): every artifact of a configuration shares it.
struct FastPathConfiguration {
    DenseKernels dense = DenseKernels::Bf16;
    bool a4_experts    = false; // W4A4 experts (NVFP4 with activation scales), else 16-bit activations
    KvCacheStorage kv  = KvCacheStorage::Int8Group64;

    bool operator==(const FastPathConfiguration&) const = default;
};

// For example "Q8 dense, 4-bit expert activations, int8 KV".
[[nodiscard]] std::string describe(const FastPathConfiguration& configuration);

[[nodiscard]] FastPathConfiguration fast_path_configuration(const execution::Parameters& parameters,
                                                            const EngineOptions& options);

// Whether the path acts in this configuration under these options at all.
[[nodiscard]] bool fast_path_applies(FastPath path, const FastPathConfiguration& configuration,
                                     const EngineOptions& options);

// Whether the path passed the acceptance gate in this configuration (the registry, section 6.7).
[[nodiscard]] bool fast_path_qualified(FastPath path, const FastPathConfiguration& configuration);

struct FastPathSelection {
    FastPathConfiguration configuration;
    // In kFastPaths order.
    std::vector<FastPath> paths;

    [[nodiscard]] bool selected(FastPath path) const noexcept;
    // The Forward's flags for the selected paths.
    [[nodiscard]] execution::ForwardFastPaths forward() const noexcept;
    // No request-coupling path (adaptive-speculation, dflash2-pairs, free-cache-resume, preemption, q8-sliced-k,
    // expert-a4-residual and its forms, fp8-a8-pair, fp8-a8-pair-block32, ggml-bf16-prefill,
    // f16-accumulate-prefill, router-mma, screened-lm-head) is selected:
    // a request's output depends only on the request, the artifact, the options and the build.
    [[nodiscard]] bool request_invariant() const noexcept;
};

// The qualified paths that apply when options.fast_paths is set, plus every
// options.fast_path_candidates entry (qualification runs), which must apply.
[[nodiscard]] FastPathSelection select_fast_paths(const execution::Parameters& parameters,
                                                  const EngineOptions& options);

// --a8-activations (EngineOptions::a8_activations; LOSSY, opt-in, never a fast path; design §19.3.31): the
// 8-bit activation routes an artifact takes, by call width. A route stays only where it measured faster than
// the route it replaces; a8_activation_routes sets each on one line, which withdraws it.
struct A8Routes {
    // A GGUF import's dense projections (their count; 0 for other artifacts), which ggml_dense and
    // ggml_dense_prefill cover.
    std::size_t ggml_projections = 0;
    // Decode and verification calls: a GGUF import's dense projections in calls of 1-31 columns
    // (ops::LinearRoutes::GgmlA8), and the routed experts' calls of at most 128 columns with their CPU-served
    // misses: GGUF banks (ops::MoeGgmlBank::a8) and NVFP4 W4A16 banks (ops::offloaded_moe::NarrowA16::kA8,
    // design §16.2.3).
    bool ggml_dense    = false;
    bool ggml_experts  = false;
    bool nvfp4_experts = false;
    // Prefill calls: a GGUF import's dense projections in calls of 32 or more columns on the same codec
    // (ops::LinearRoutes::GgmlA8Wide, ahead of ggml-bf16-prefill's BF16 tiles); MXFP8 dense projections
    // (NVIDIA FP4-Local) on fp8-a8-pair-block32's route, two E4M3 terms per 32 activation values
    // (ops::LinearRoutes::Fp8A8PairBlock32, from 33 to 1,025 columns by problem), in place of
    // f16-accumulate-prefill's FP16 route; and the routed experts' calls of more than 128 columns on the same
    // codec (ops::offloaded_moe::WideA16::kA8, which needs their 8-bit narrow route), in place of
    // expert-a4-residual's A4 terms, ggml-bf16-prefill's BF16 planes or the exact route.
    bool ggml_dense_prefill = false;
    bool mxfp8_dense        = false;
    bool experts_prefill    = false;

    [[nodiscard]] bool any() const noexcept {
        return ggml_dense || ggml_experts || nvfp4_experts || ggml_dense_prefill || mxfp8_dense || experts_prefill;
    }
    // Adds the routes to the Forward's flags.
    void apply(execution::ForwardFastPaths& fast) const noexcept;
    // What takes 8-bit activations in decode and in prefill (the startup line).
    [[nodiscard]] std::string describe() const;
};

// The routes of a configuration whose routed experts' gate/up plane is `expert_format`, with `ggml_projections`
// GGUF dense projections; `expert_a4_narrow`: expert-a4-residual-decode is selected (its two A4 terms keep the
// W4A16 experts' narrow calls).
[[nodiscard]] A8Routes a8_activation_routes(const FastPathConfiguration& configuration, QType expert_format,
                                            std::size_t ggml_projections, bool expert_a4_narrow) noexcept;
// The same for loaded parameters beside their selected fast paths.
[[nodiscard]] A8Routes a8_activation_routes(const execution::Parameters& parameters, const FastPathSelection& fast);

} // namespace infernix::models::qwen4_exp
