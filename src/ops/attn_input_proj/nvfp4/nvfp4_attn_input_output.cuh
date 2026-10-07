#pragma once
#include "ops/linear/common/output.cuh"

namespace infernix::ops::detail {
using Nvfp4AttentionInputOutput = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;
} // namespace infernix::ops::detail
