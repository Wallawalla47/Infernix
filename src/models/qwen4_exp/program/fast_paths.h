#pragma once

#include "core/weight.h"
#include "models/qwen4_exp/execution/fast_path_flags.h"
#include "infernix/types.h"

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

} // namespace infernix::models::qwen4_exp
