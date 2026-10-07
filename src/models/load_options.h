#pragma once

#include "models/registry.h"
#include "infernix/types.h"

#include <string_view>

namespace infernix::models {

struct LoadOptions {
    EnginePurpose purpose                  = EnginePurpose::Generation;
    bool vision                            = false;
    bool vision_offload                    = false;
    std::uint32_t vision_max_merged_tokens = 32768;
    SpeculativeBackend speculative         = SpeculativeBackend::None;
    ProposalHead proposal_head             = ProposalHead::Full;
    float rope_yarn_factor                 = 1.0F;
    // Every Use binds A16Only, whatever activation precision the artifact permits.
    bool a16_activations = false;
    // Qwen4Exp: the routed expert banks stay in the artifact and are read in place by the SSD
    // expert tier (design §19.3.7) instead of being pinned in host memory.
    bool stream_experts = false;

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

// --vision-offload auto: on for Qwen4Exp, whose expert cache lends the encode window's device
// memory (design §19.3.2); off for Qwen3.5, which borrows it from an evictable weight ladder.
[[nodiscard]] constexpr bool resolve_vision_offload(VisionOffload mode, Architecture architecture) noexcept {
    return mode == VisionOffload::On || (mode == VisionOffload::Auto && architecture == Architecture::Qwen4Exp);
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options, Architecture architecture) noexcept {
    return {.purpose                  = options.purpose,
            .vision                   = options.enable_vision,
            .vision_offload           = resolve_vision_offload(options.vision_offload, architecture),
            .vision_max_merged_tokens = options.vision_max_merged_tokens,
            .speculative              = options.speculative.backend,
            .proposal_head            = options.speculative.proposal_head,
            .rope_yarn_factor         = options.rope_yarn_factor,
            .a16_activations          = options.a16_activations};
}

} // namespace infernix::models
