#include "runtime/contract/sampling.h"

#include <cmath>
#include <stdexcept>

namespace infernix::runtime {
namespace {

void validate(const ResolvedSamplingParameters& sampling, bool full_vocabulary) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.temperature < 0.0F || sampling.temperature > 2.0F) {
        throw std::invalid_argument("temperature must be in [0,2]");
    }
    if (sampling.top_k < (full_vocabulary ? 0 : 1) || sampling.top_k > 20) {
        throw std::invalid_argument(full_vocabulary ? "resolved top_k must be in [0,20]"
                                                    : "resolved top_k must be in [1,20]");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
    if (sampling.presence_penalty < -2.0F || sampling.presence_penalty > 2.0F) {
        throw std::invalid_argument("presence_penalty must be in [-2,2]");
    }
    if (sampling.frequency_penalty < -2.0F || sampling.frequency_penalty > 2.0F) {
        throw std::invalid_argument("frequency_penalty must be in [-2,2]");
    }
}

} // namespace

ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                            SamplingMode mode, const SamplingOverrides& overrides,
                                            bool constrained_output) {
    SamplingPreset preset = defaults.for_mode(mode);
    if (constrained_output) {
        preset.presence_penalty  = 0.0F;
        preset.frequency_penalty = 0.0F;
    }
    ResolvedSamplingParameters resolved{
        .temperature       = overrides.temperature.value_or(preset.temperature),
        .top_k             = overrides.top_k.value_or(preset.top_k),
        .top_p             = overrides.top_p.value_or(preset.top_p),
        .min_p             = overrides.min_p.value_or(preset.min_p),
        .presence_penalty  = overrides.presence_penalty.value_or(preset.presence_penalty),
        .frequency_penalty = overrides.frequency_penalty.value_or(preset.frequency_penalty),
        .seed              = overrides.seed.value_or(0),
    };
    // Zero means no limit. A family whose sampler takes the whole vocabulary (ModelSamplingDefaults::full_vocabulary)
    // keeps it; the others' pipelines have an exact top-20 candidate domain, so zero is that cap there.
    if (resolved.top_k == 0 && !defaults.full_vocabulary) { resolved.top_k = 20; }
    validate(resolved, defaults.full_vocabulary);
    return resolved;
}

} // namespace infernix::runtime
