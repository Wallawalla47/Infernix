#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace infernix::test::linear_swiglu {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
    // A16 compute under LinearPolicy::A16ColumnInvariant: one schedule at every T.
    A16ColumnInvariant,
    // LinearPolicy::AllowA4Residual: A16 up to 8 columns, then NVFP4 residual terms.
    A4Residual,
    // LinearPolicy::AllowA8Residual: A16 up to 16 columns, then two E4M3 terms.
    A8Residual,
};

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ActivationCompute activation_compute;
};

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                std::span<const std::int32_t> graph_cases = {});

} // namespace infernix::test::linear_swiglu
