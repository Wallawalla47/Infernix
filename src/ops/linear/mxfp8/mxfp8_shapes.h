#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

using Mxfp8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// One registered MXFP8 problem: its exact (A16) route for every width, column-invariant at every T.
struct Mxfp8LinearShape {
    std::int32_t n, k;
    Mxfp8Launch a16;
};

// Qwen3.8-Flash-Next (NVIDIA FP4-Local, W8A16): the GDN q/k/v/z, QSA query/gate/key/value with the
// indexer, GDN and QSA output and shared-expert gate/up and down projections.
extern const Mxfp8LinearShape kMxfp8N16384K2560;
extern const Mxfp8LinearShape kMxfp8N13952K2560;
extern const Mxfp8LinearShape kMxfp8N2560K6144;
extern const Mxfp8LinearShape kMxfp8N1280K2560;
extern const Mxfp8LinearShape kMxfp8N2560K640;

} // namespace infernix::ops::detail
