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
// (activations and KV format) and the attention geometry the kernels run in
// (docs/maintainer/model-quality.md section 6).
struct FastPathConfiguration {
    std::uint32_t query_heads   = 0;
    ActivationClass activations = ActivationClass::A16;
    KvCacheStorage kv           = KvCacheStorage::BFloat16;

    bool operator==(const FastPathConfiguration&) const = default;
};

// For example "24 heads, 4-bit activations, int8 KV".
[[nodiscard]] std::string describe(const FastPathConfiguration& configuration);

[[nodiscard]] FastPathConfiguration fast_path_configuration(const Model& model,
                                                            const EngineOptions& options);

// Whether the path's kernels run in this configuration at all.
[[nodiscard]] bool fast_path_applies(FastPath path, const FastPathConfiguration& configuration,
                                     const EngineOptions& options);

// Whether the path passed the acceptance gate in this configuration (the registry, section 6.7).
[[nodiscard]] bool fast_path_qualified(FastPath path, const FastPathConfiguration& configuration);

struct FastPathSelection {
    FastPathConfiguration configuration;
    // In kFastPaths order.
    std::vector<FastPath> paths;

    [[nodiscard]] bool selected(FastPath path) const noexcept;
};

// The qualified paths that apply when options.fast_paths is set, plus every
// options.fast_path_candidates entry (qualification runs), which must apply.
[[nodiscard]] FastPathSelection select_fast_paths(const Model& model, const EngineOptions& options);

} // namespace infernix::models::qwen3_5
