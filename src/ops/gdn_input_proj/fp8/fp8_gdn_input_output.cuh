#pragma once
#include "ops/linear/common/output.cuh"

namespace infernix::ops::detail {
using Fp8GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;
} // namespace infernix::ops::detail
