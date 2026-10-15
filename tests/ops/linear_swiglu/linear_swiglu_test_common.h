#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace infernix::test::linear_swiglu {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,    // FP8 E4M3 activations
    A8Int, // int8 activations with one absmax scale per token (the q4-w4a8 route)
    A4,
    // A16 compute under LinearPolicy::A16ColumnInvariant: one schedule at every T.
    A16ColumnInvariant,
    // LinearPolicy::AllowA4Residual: A16 up to 8 columns, then NVFP4 residual terms.
    A4Residual,
    // LinearPolicy::AllowA8Residual: A16 up to 16 columns, then two E4M3 terms.
    A8Residual,
};

enum class ActivationPattern : std::uint8_t {
    Mixed,             // one dense token, then sparse tokens
    Int8Representable, // int8 per-token quantization is exact (A8Int only; A16 criterion)
};

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ActivationCompute activation_compute;
    // The fast-path route flags every call passes.
    ops::LinearRoutes routes  = ops::LinearRoutes::Exact;
    ActivationPattern pattern = ActivationPattern::Mixed;
};

// exact_route_cases: extents at which the profile's routes must reproduce the exact route's
// (LinearRoutes::Exact) output bit for bit; each must lie within token_cases' largest extent.
int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                std::span<const std::int32_t> graph_cases       = {},
                std::span<const std::int32_t> exact_route_cases = {});

} // namespace infernix::test::linear_swiglu
