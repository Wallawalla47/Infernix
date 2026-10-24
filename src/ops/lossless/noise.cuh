#pragma once

// The lossless route's noise twin (EngineOptions::lossless_noise_seed; docs/maintainer/model-quality.md, IQS-2
// noise floor): every correctly rounded result is replaced by a stochastic rounding of the same exact value,
// rounding away from the nearest neighbour with probability equal to the exact value's distance from it in
// units of the gap between the two neighbours. It models an ideal BF16 implementation that is off by at most
// one unit in the last place at every rounding, and nothing else.
//
// The seed is a __constant__ of each translation unit that rounds (one copy per CUDA module); each such unit
// defines its setter with INFERNIX_LOSSLESS_NOISE_UNIT, and ops::lossless_set_noise_seed sets them all. A seed
// of 0 (the default) keeps round to nearest even. The draw is keyed by the value being rounded, so equal exact
// values round alike within one run.

#include "core/device.h"

#include <cstdint>

namespace infernix::ops::lossless {

static __constant__ unsigned long long noise_seed;
// The share of the stochastic roundings that move (1: the twin's full noise; less: a weaker noise, for the
// attribution of an engine's error to the noise level it is equivalent to).
static __constant__ float noise_scale = 1.0F;

__device__ __forceinline__ bool noise_active() { return noise_seed != 0ULL; }

__device__ __forceinline__ std::uint64_t noise_mix(std::uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// A uniform 32-bit draw keyed by the rounded value.
__device__ __forceinline__ std::uint32_t noise_draw(std::uint64_t key) {
    return static_cast<std::uint32_t>(noise_mix(noise_seed ^ noise_mix(key)) >> 32);
}

// Whether a value at `fraction` (in [0, 1]) of the way from the nearest neighbour to the other one moves
// to the other one.
__device__ __forceinline__ bool noise_moves(double fraction, std::uint64_t key) {
    if (!(fraction > 0.0)) { return false; }
    return static_cast<double>(noise_draw(key)) < fraction * static_cast<double>(noise_scale) * 4294967296.0;
}

} // namespace infernix::ops::lossless

#define INFERNIX_LOSSLESS_NOISE_UNIT(tag)                                                                   \
    namespace infernix::ops::lossless {                                                                     \
    void set_noise_seed_##tag(unsigned long long seed, float scale) {                                       \
        CUDA_CHECK(cudaMemcpyToSymbol(noise_seed, &seed, sizeof(seed)));                                    \
        CUDA_CHECK(cudaMemcpyToSymbol(noise_scale, &scale, sizeof(scale)));                                 \
    }                                                                                                       \
    }
