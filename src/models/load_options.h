#pragma once

#include "ninfer/types.h"

#include <string_view>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose                  = EnginePurpose::Generation;
    bool vision                            = false;
    bool vision_offload                    = false;
    std::uint32_t vision_max_merged_tokens = 32768;
    SpeculativeBackend speculative         = SpeculativeBackend::None;
    ProposalHead proposal_head             = ProposalHead::Full;
    float rope_yarn_factor                 = 1.0F;

    bool operator==(const LoadOptions&) const = default;

    // Vision offload keeps the vision tower in pinned system RAM and streams it through borrowed
    // device memory per encode window; requires vision plus an evictable weight ladder.
    [[nodiscard]] bool overlay_vision() const noexcept { return vision && vision_offload; }

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept { return dflash() || dflash2(); }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash:
            return "dflash";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        }
        return {};
    }
};

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose                  = options.purpose,
            .vision                   = options.enable_vision,
            .vision_offload           = options.vision_offload,
            .vision_max_merged_tokens = options.vision_max_merged_tokens,
            .speculative              = options.speculative.backend,
            .proposal_head            = options.speculative.proposal_head,
            .rope_yarn_factor         = options.rope_yarn_factor};
}

} // namespace ninfer::models
