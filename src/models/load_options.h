#pragma once

#include "models/registry.h"
#include "infernix/types.h"

#include <algorithm>
#include <filesystem>
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
    // The lossless route (EngineOptions::lossless): every Use binds A16Only; the model runs its lossless
    // route.
    bool lossless = false;
    // The design reference (EngineOptions::reference, with lossless): every Use keeps its declared
    // activation permission, which the lossless route applies as the design (lossless-route.md §3b), and
    // K and V cross the KV format reference_kv's codec (bf16 without the reference).
    bool reference              = false;
    KvCacheStorage reference_kv = KvCacheStorage::BFloat16;
    // Qwen4Exp: routed expert banks whose 4-bit activations are optional run them (W4A4).
    bool expert_a4_activations = false;
    // Offloaded routed expert banks (Qwen4Exp, Qwen3.5 with offloaded experts) stay in the artifact
    // and are read in place by the SSD expert tier (Flash-Next design §19.3.7) instead of being
    // pinned in host memory.
    bool stream_experts = false;
    // Qwen4Exp DFlash2: the companion artifact of the drafter (SpeculativeOptions::draft_model).
    std::filesystem::path draft_model;
    // Qwen4Exp: the artifact's 2-bit screens (--screen-heads), each bound when present for its consumer (an
    // artifact without them loads as before): the MTP drafter's proposal head (Generation with --spec mtp
    // --lm-head-draft) and the LM head (Generation with fast paths, where screened-lm-head may be selected).
    // screened_lm_head (the fast path asked for as a candidate) needs the LM head's screen: an artifact
    // without it is refused.
    bool proposal_head_screen = false;
    bool lm_head_screen       = false;
    bool screened_lm_head     = false;
    // Qwen3.5 (EngineOptions::embedding_host): the token embedding stays in pinned host memory, gathered
    // zero-copy, as an artifact with offloaded experts already keeps it.
    bool embedding_host = false;

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

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options, Architecture architecture) {
    return {.purpose                  = options.purpose,
            .vision                   = options.enable_vision,
            .vision_offload           = resolve_vision_offload(options.vision_offload, architecture),
            .vision_max_merged_tokens = options.vision_max_merged_tokens,
            .speculative              = options.speculative.backend,
            .proposal_head            = options.speculative.proposal_head,
            .rope_yarn_factor         = options.rope_yarn_factor,
            .lossless                 = options.lossless,
            .reference                = options.reference,
            .reference_kv             = options.reference ? options.kv_cache : KvCacheStorage::BFloat16,
            .expert_a4_activations    = options.expert_a4_activations,
            .draft_model              = options.speculative.draft_model,
            // The screens are Qwen4Exp's (a Qwen3.5 Program compares these options with its loaded ones).
            .proposal_head_screen     = architecture == Architecture::Qwen4Exp &&
                                    options.purpose == EnginePurpose::Generation &&
                                    options.speculative.backend == SpeculativeBackend::Mtp &&
                                    options.speculative.proposal_head == ProposalHead::Optimized,
            .lm_head_screen           = architecture == Architecture::Qwen4Exp &&
                                    options.purpose == EnginePurpose::Generation && options.fast_paths &&
                                    !options.lossless,
            .screened_lm_head         = std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(),
                                                  FastPath::ScreenedLmHead) != options.fast_path_candidates.end(),
            .embedding_host           = options.embedding_host};
}

} // namespace infernix::models
