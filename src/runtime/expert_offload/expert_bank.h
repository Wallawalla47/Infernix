#pragma once

// A layer's routed-expert bank as the offloaded MoE reads it (docs/maintainer/qwen3_5-122b-offload-
// design.md §7: the shared expert-offload component that Qwen3.8-Flash-Next's and Qwen3.5's Programs
// instantiate). The planes are views into the Model's materialized weights; the scalars are derived
// once at load.

#include "core/weight_view.h"
#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cstdint>
#include <string>
#include <vector>

namespace infernix::runtime::expert_offload {

// W4A4 (Qwen3.8-Flash-Next design §16.2): alpha is derived once from the stored words, alpha =
// fl32(weight_scale_2 * input_scale), so every route (GPU frame, GPU staging, CPU) reads the same
// values. W4A16 (§16.2.1): no input scales; alpha holds the stored multipliers themselves.
struct ExpertBank {
    ExpertBankPlanes planes;
    ops::offloaded_moe::ExpertActivation activation = ops::offloaded_moe::ExpertActivation::kA4;
    std::vector<ops::offloaded_moe::ExpertScales> scales; // [experts]

    [[nodiscard]] ops::offloaded_moe::ExpertShape shape() const noexcept {
        return {static_cast<int>(planes.hidden), static_cast<int>(planes.intermediate)};
    }
};

// A GGUF import's bank: its records' ggml formats for the CPU route (invalid for an NVFP4 bank).
[[nodiscard]] inline ops::offloaded_moe::GgmlExpertFormat ggml_format(const ExpertBank& bank) noexcept {
    if (bank.planes.gate_up_format == QType::NVFP4_MUL) { return {}; }
    return {.gate_up        = bank.planes.gate_up_format,
            .down           = bank.planes.down_format,
            .gate_row_bytes = static_cast<std::uint32_t>(bank.planes.gate_row_bytes),
            .down_row_bytes = static_cast<std::uint32_t>(bank.planes.down_row_bytes)};
}

// The scalars of a W4A16 nvfp4_expert_rg16_v1 bank (Qwen3.8-Flash-Next design §16.2.1): no input
// scales; alpha_* are the record's stored multipliers. Throws std::invalid_argument naming `name`
// unless every multiplier is positive finite FP32.
[[nodiscard]] std::vector<ops::offloaded_moe::ExpertScales> a16_expert_scales(const ExpertBankPlanes& planes,
                                                                             const std::string& name);

// A complete W4A16 nvfp4_expert_rg16_v1 bank of `experts` x [hidden, intermediate] in one
// host-readable parent (a pinned routed bank), with its scalars from its scale tail.
[[nodiscard]] ExpertBank a16_nvfp4_bank(const WeightView& view, std::uint32_t experts, std::uint32_t hidden,
                                        std::uint32_t intermediate, const std::string& name);
// The same bank whose records the host cannot read, built from `multipliers`, the caller's copy of the
// bank's scale tail ([experts][3] FP32, kept alive as long as the bank): left in the artifact for the
// SSD tier (artifact::Residency::Streamed; no records), or resident in device memory only (a shared
// expert; device_records).
[[nodiscard]] ExpertBank a16_nvfp4_streamed_bank(const WeightView& view, const float* multipliers,
                                                 std::uint32_t experts, std::uint32_t hidden,
                                                 std::uint32_t intermediate, const std::string& name);
[[nodiscard]] ExpertBank a16_nvfp4_device_bank(const WeightView& view, const float* multipliers,
                                               std::uint32_t experts, std::uint32_t hidden,
                                               std::uint32_t intermediate, const std::string& name);

} // namespace infernix::runtime::expert_offload
