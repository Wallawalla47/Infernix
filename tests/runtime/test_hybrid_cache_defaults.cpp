#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"
#include "runtime/engine/expert_memory.h"
#include "runtime/engine/model_instance.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

using infernix::EngineOptions;
using infernix::runtime::normalize_engine_options;

} // namespace

int main() {
    int failures = 0;

    // The prefix cache derives every tuning value from the rest of the configuration. With the
    // default Host tier (8 GiB) at concurrency 2 and a 2048-token chunk: one resident snapshot per
    // lane plus one staging slot = 3, 8 taps, ladder max(4096, 2 * 2048) = 4096 and minimum gap
    // max(1024, 2048) = 2048. The snapshot slots are the extra Device StateImages.
    {
        EngineOptions options;
        options.max_concurrency                  = 2;
        options.prefill_chunk                    = 2048;
        const EngineOptions normalized           = normalize_engine_options(options);
        const infernix::ContextCacheOptions& out = normalized.context_cache;
        failures +=
            check(out.host_capacity_bytes == infernix::kDefaultHybridHostCacheBytes &&
                      out.hybrid.device_snapshot_slots == 3U && out.hybrid.max_new_taps == 8U &&
                      out.hybrid.tap_ladder_tokens == 4096U &&
                      out.hybrid.tap_min_gap_tokens == 2048U,
                  "prefix cache defaults did not derive from concurrency, chunk and Host tier");
    }

    // Without a Host tier Device slots are the only snapshot storage: two spare slots and a
    // two-tap budget. A large chunk coarsens the ladder: max(4096, 2 * 8192) and max(1024, 8192).
    {
        EngineOptions options;
        options.max_concurrency                   = 8;
        options.prefill_chunk                     = 8192;
        options.context_cache.host_capacity_bytes = 0;
        const infernix::ContextCacheOptions out   = normalize_engine_options(options).context_cache;
        failures +=
            check(out.host_capacity_bytes == 0U && out.hybrid.device_snapshot_slots == 10U &&
                      out.hybrid.max_new_taps == 2U && out.hybrid.tap_ladder_tokens == 16384U &&
                      out.hybrid.tap_min_gap_tokens == 8192U,
                  "Device-only prefix cache defaults are wrong");
    }

    // Explicit overrides are kept verbatim.
    {
        EngineOptions options;
        options.max_concurrency                            = 4;
        options.context_cache.hybrid.device_snapshot_slots = 12;
        options.context_cache.hybrid.max_new_taps          = 3;
        options.context_cache.hybrid.tap_ladder_tokens     = 8192;
        options.context_cache.hybrid.tap_min_gap_tokens    = 512;
        const infernix::ContextCacheOptions out = normalize_engine_options(options).context_cache;
        failures += check(out.hybrid.device_snapshot_slots == 12U &&
                              out.hybrid.max_new_taps == 3U &&
                              out.hybrid.tap_ladder_tokens == 8192U &&
                              out.hybrid.tap_min_gap_tokens == 512U,
                          "explicit prefix cache overrides were not preserved");
    }

    // A disabled cache keeps no snapshot slots and no Host tier, and refuses capacities for them.
    {
        EngineOptions options;
        options.context_cache.enabled           = false;
        const infernix::ContextCacheOptions out = normalize_engine_options(options).context_cache;
        failures += check(out.host_capacity_bytes == 0U && out.hybrid.device_snapshot_slots == 0U,
                          "a disabled prefix cache kept snapshot or Host capacity");
    }
    for (int variant = 0; variant < 2; ++variant) {
        EngineOptions options;
        options.context_cache.enabled = false;
        if (variant == 0) { options.context_cache.host_capacity_bytes = 1ULL << 30; }
        if (variant == 1) { options.context_cache.hybrid.device_snapshot_slots = 2; }
        bool rejected = false;
        try {
            (void)normalize_engine_options(options);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "a disabled prefix cache accepted a capacity");
    }

    // Out-of-range tuning is rejected.
    for (int variant = 0; variant < 3; ++variant) {
        EngineOptions options;
        if (variant == 0) { options.context_cache.hybrid.device_snapshot_slots = 0; }
        if (variant == 1) { options.context_cache.hybrid.device_snapshot_slots = 65; }
        if (variant == 2) { options.context_cache.hybrid.tap_min_gap_tokens = 16; }
        bool rejected = false;
        try {
            (void)normalize_engine_options(options);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "prefix cache normalization accepted an invalid capacity");
    }

    // The Host budget buys whole slabs; it must hold one snapshot (image slabs + tail slab) plus
    // one block. A 2 MiB slab and a 7-slab image need 9 slabs = 18 MiB; 0 disables the tier.
    {
        infernix::models::qwen3_5::detail::HybridHostLayout layout;
        layout.slab_bytes  = 2ULL << 20;
        layout.image_slabs = 7;
        using infernix::models::qwen3_5::detail::hybrid_host_slabs;
        failures += check(hybrid_host_slabs(layout, 0) == 0, "a zero Host budget must disable");
        failures += check(hybrid_host_slabs(layout, 18ULL << 20) == 9U,
                          "the minimum Host budget did not buy exactly one snapshot and a block");
        failures += check(hybrid_host_slabs(layout, (21ULL << 20) - 1U) == 10U,
                          "a Host budget must buy whole slabs");
        bool rejected = false;
        try {
            (void)hybrid_host_slabs(layout, (18ULL << 20) - 1U);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "a Host budget below one snapshot was accepted");
    }

    // Qwen3.8-Flash-Next resolves its own defaults: the common 8 GiB Host tier, no Device snapshot
    // slots (its snapshots are Host-born), the common tap budget; Qwen3.5 keeps C + 1 slots.
    {
        EngineOptions options;
        options.max_concurrency = 2;
        const infernix::ContextCacheOptions qwen4 =
            normalize_engine_options(options, infernix::models::Architecture::Qwen4Exp).context_cache;
        failures += check(qwen4.host_capacity_bytes == infernix::kDefaultHybridHostCacheBytes &&
                              qwen4.hybrid.device_snapshot_slots == 0U && qwen4.hybrid.max_new_taps == 8U,
                          "Qwen3.8-Flash-Next prefix cache defaults are wrong");
        const infernix::ContextCacheOptions qwen35 = normalize_engine_options(options).context_cache;
        failures += check(qwen35.host_capacity_bytes == infernix::kDefaultHybridHostCacheBytes &&
                              qwen35.hybrid.device_snapshot_slots == 3U,
                          "Qwen3.5 prefix cache defaults changed");
        // A Qwen3.5 artifact with offloaded experts keeps one Device slot beside its Host tier (the frames
        // the others would take; candidates §22 VM-03), and the per-lane rule without a Host tier.
        const infernix::ContextCacheOptions offloaded =
            normalize_engine_options(options, infernix::models::Architecture::Qwen3_5, true).context_cache;
        failures += check(offloaded.hybrid.device_snapshot_slots == 1U,
                          "offloaded Qwen3.5 with a Host tier: one Device snapshot slot");
        EngineOptions no_tier                          = options;
        no_tier.context_cache.host_capacity_bytes      = 0;
        const infernix::ContextCacheOptions device_only =
            normalize_engine_options(no_tier, infernix::models::Architecture::Qwen3_5, true).context_cache;
        failures += check(device_only.hybrid.device_snapshot_slots == 4U,
                          "offloaded Qwen3.5 without a Host tier: C + 2 Device snapshot slots");
        // --host-context-mib auto: Qwen3.8-Flash-Next keeps the size unset for its RAM plan (and still has a Host
        // tier, so its other defaults are the tier's); Qwen3.5 takes the common default.
        EngineOptions automatic = options;
        automatic.context_cache.host_capacity_auto = true;
        const infernix::ContextCacheOptions qwen4_auto =
            normalize_engine_options(automatic, infernix::models::Architecture::Qwen4Exp).context_cache;
        failures += check(!qwen4_auto.host_capacity_bytes && qwen4_auto.host_capacity_auto &&
                              qwen4_auto.hybrid.max_new_taps == 8U,
                          "Qwen3.8-Flash-Next must keep an automatic Host tier unsized for its RAM plan");
        const infernix::ContextCacheOptions qwen35_auto = normalize_engine_options(automatic).context_cache;
        failures += check(qwen35_auto.host_capacity_bytes == infernix::kDefaultHybridHostCacheBytes &&
                              !qwen35_auto.host_capacity_auto,
                          "Qwen3.5 must take the default Host tier for --host-context-mib auto");
        // The automatic size: 0.4 of what the plan leaves beyond --ram-free-mib, in 256 MiB steps, between 1 and 8 GiB.
        constexpr std::uint64_t kMiB = 1ULL << 20;
        failures += check(infernix::runtime::auto_host_context_bytes(0) == 1024 * kMiB, "an automatic Host tier has a 1 GiB floor");
        failures += check(infernix::runtime::auto_host_context_bytes(10240 * kMiB) == 4096 * kMiB,
                          "an automatic Host tier takes 0.4 of 10 GiB left");
        failures += check(infernix::runtime::auto_host_context_bytes(15155 * kMiB) == 5888 * kMiB,
                          "an automatic Host tier rounds 0.4 of the RAM left down to 256 MiB");
        failures += check(infernix::runtime::auto_host_context_bytes(40960 * kMiB) == 8192 * kMiB,
                          "an automatic Host tier stops at 8 GiB");
        // The prefill chunk defaults per model: 16,384 for Qwen3.8-Flash-Next, whose ladder keeps
        // the 4,096-token chunk's spacing (max(4096, 2 * 4096) and max(1024, 4096)), and 1,024
        // for Qwen3.5 (ladder max(4096, 2 * 1024), gap max(1024, 1024)).
        const EngineOptions qwen4_options =
            normalize_engine_options(options, infernix::models::Architecture::Qwen4Exp);
        failures += check(qwen4_options.prefill_chunk == 16384U && qwen4.hybrid.tap_ladder_tokens == 8192U &&
                              qwen4.hybrid.tap_min_gap_tokens == 4096U,
                          "Qwen3.8-Flash-Next prefill chunk or ladder defaults are wrong");
        failures += check(normalize_engine_options(options).prefill_chunk == 1024U &&
                              qwen35.hybrid.tap_ladder_tokens == 4096U &&
                              qwen35.hybrid.tap_min_gap_tokens == 1024U,
                          "Qwen3.5 prefill chunk or ladder defaults changed");
        // Tool-call holds: on by default for Qwen3.5, absent for Qwen3.8-Flash-Next, explicit
        // values (zero turns them off) kept, and a disabled cache has none.
        failures += check(qwen35.hybrid.tool_call_hold == infernix::kDefaultToolCallHold &&
                              qwen4.hybrid.tool_call_hold == std::chrono::milliseconds{0},
                          "tool-call hold defaults are wrong");
        EngineOptions off = options;
        off.context_cache.hybrid.tool_call_hold = std::chrono::milliseconds{0};
        failures += check(normalize_engine_options(off).context_cache.hybrid.tool_call_hold ==
                              std::chrono::milliseconds{0},
                          "an explicit zero tool-call hold was not kept");
        EngineOptions disabled;
        disabled.context_cache.enabled = false;
        failures += check(normalize_engine_options(disabled).context_cache.hybrid.tool_call_hold ==
                              std::chrono::milliseconds{0},
                          "a disabled prefix cache kept a tool-call hold");
        // Next-turn prefill (hybrid-prefix-cache-spec §7.9): on by default for Qwen3.5, off for Qwen3.8-Flash-Next
        // (and refused on its reproducible route), an explicit zero kept, none without a cache, and a value
        // outside [0, 1 h] refused.
        failures += check(qwen35.hybrid.next_turn_prefill_quiet == infernix::kDefaultNextTurnPrefillQuiet &&
                              qwen4.hybrid.next_turn_prefill_quiet == std::chrono::milliseconds{0},
                          "next-turn prefill defaults are wrong");
        {
            EngineOptions flash_next                                = options;
            flash_next.context_cache.hybrid.next_turn_prefill_quiet = std::chrono::milliseconds{150};
            failures += check(normalize_engine_options(flash_next, infernix::models::Architecture::Qwen4Exp)
                                      .context_cache.hybrid.next_turn_prefill_quiet == std::chrono::milliseconds{150},
                              "Qwen3.8-Flash-Next's explicit next-turn quiet period was not kept");
            flash_next.fast_paths = false;
            bool refused          = false;
            try {
                (void)normalize_engine_options(flash_next, infernix::models::Architecture::Qwen4Exp);
            } catch (const std::invalid_argument&) { refused = true; }
            failures += check(refused, "Qwen3.8-Flash-Next's reproducible route accepted the next-turn prefill");
        }
        EngineOptions quiet_off                              = options;
        quiet_off.context_cache.hybrid.next_turn_prefill_quiet = std::chrono::milliseconds{0};
        failures += check(normalize_engine_options(quiet_off).context_cache.hybrid.next_turn_prefill_quiet ==
                              std::chrono::milliseconds{0},
                          "an explicit zero next-turn quiet period was not kept");
        failures += check(normalize_engine_options(disabled).context_cache.hybrid.next_turn_prefill_quiet ==
                              std::chrono::milliseconds{0},
                          "a disabled prefix cache kept the next-turn prefill");
        for (int variant = 0; variant < 3; ++variant) {
            EngineOptions bad = options;
            bad.context_cache.hybrid.next_turn_prefill_quiet =
                std::chrono::milliseconds(variant == 0 ? -1 : variant == 1 ? 3'600'001 : 150);
            if (variant == 2) {
                bad                       = EngineOptions{};
                bad.context_cache.enabled = false;
                bad.context_cache.hybrid.next_turn_prefill_quiet = std::chrono::milliseconds{150};
            }
            bool rejected = false;
            try {
                (void)normalize_engine_options(bad);
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected, "an unservable next-turn quiet period was accepted");
        }
    }
    // --draft-tokens auto takes the model's most MTP drafts unless given a ceiling (Qwen3.8-Flash-Next 7,
    // Qwen3.5 5); the block drafters take it too (Qwen3.8-Flash-Next's DFlash2 7, the Qwen3.5 planner
    // resolves its blocks). adaptive-mtp and adaptive-dflash are auto's policy, so a fixed length
    // refuses them, and auto without a drafter is refused.
    {
        EngineOptions options;
        options.speculative.backend           = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens_auto = true;
        failures += check(normalize_engine_options(options, infernix::models::Architecture::Qwen4Exp)
                                      .speculative.draft_tokens == 7U &&
                              normalize_engine_options(options).speculative.draft_tokens == 5U,
                          "--draft-tokens auto did not resolve to the model's most drafts");
        options.speculative.draft_tokens = 3;
        failures += check(normalize_engine_options(options).speculative.draft_tokens == 3U,
                          "an automatic draft length lost its ceiling");
        EngineOptions fixed               = options;
        fixed.speculative.draft_tokens_auto = false;
        fixed.fast_path_candidates        = {infernix::FastPath::AdaptiveMtp};
        bool rejected = false;
        try {
            (void)normalize_engine_options(fixed);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "adaptive-mtp was accepted with a fixed draft length");
        fixed.speculative.backend   = infernix::SpeculativeBackend::DFlash2;
        fixed.fast_path_candidates  = {infernix::FastPath::AdaptiveDFlash};
        rejected = false;
        try {
            (void)normalize_engine_options(fixed);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "adaptive-dflash was accepted with a fixed draft length");
        EngineOptions dflash                  = options;
        dflash.speculative.backend            = infernix::SpeculativeBackend::DFlash2;
        dflash.speculative.draft_tokens       = 0;
        failures += check(normalize_engine_options(dflash).speculative.draft_tokens == 0U &&
                              normalize_engine_options(dflash, infernix::models::Architecture::Qwen4Exp)
                                      .speculative.draft_tokens == 7U,
                          "--draft-tokens auto with DFlash2 did not leave Qwen3.5's block to the planner "
                          "or take Qwen3.8-Flash-Next's 7");
        EngineOptions none          = options;
        none.speculative.backend    = infernix::SpeculativeBackend::None;
        rejected = false;
        try {
            (void)normalize_engine_options(none);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "--draft-tokens auto was accepted without a drafter");
    }
    // --expert-tail-skip skips offloaded experts: Qwen3.8-Flash-Next only.
    {
        EngineOptions options;
        options.expert_tail_skip = 7;
        bool rejected = false;
        try {
            (void)normalize_engine_options(options);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "--expert-tail-skip was accepted on Qwen3.5");
        failures += check(normalize_engine_options(options, infernix::models::Architecture::Qwen4Exp).expert_tail_skip == 7U,
                          "--expert-tail-skip did not reach Qwen3.8-Flash-Next");
        options.fast_paths = false;
        rejected           = false;
        try {
            (void)normalize_engine_options(options, infernix::models::Architecture::Qwen4Exp);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "--expert-tail-skip was accepted on the exact route");
    }
    for (int variant = 0; variant < 3; ++variant) {
        EngineOptions options;
        options.context_cache.hybrid.tool_call_hold = std::chrono::seconds(variant == 0 ? -1 : 30);
        if (variant == 2) { options.context_cache.enabled = false; }
        bool rejected = false;
        try {
            (void)normalize_engine_options(options);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected == (variant != 1), "a tool-call hold was misjudged");
    }
    // Lossy Host state makes a resume depend on eviction, so the exact route refuses it.
    for (const auto precision :
         {infernix::HybridHostStatePrecision::Fp32, infernix::HybridHostStatePrecision::Fp24,
          infernix::HybridHostStatePrecision::Bf16}) {
        for (const bool fast_paths : {true, false}) {
            EngineOptions options;
            options.context_cache.hybrid.host_state_precision = precision;
            options.fast_paths                                = fast_paths;
            bool rejected = false;
            try {
                (void)normalize_engine_options(options);
            } catch (const std::invalid_argument&) { rejected = true; }
            const bool lossy = precision != infernix::HybridHostStatePrecision::Fp32;
            failures += check(rejected == (lossy && !fast_paths),
                              "the exact route misjudged a Host state precision");
        }
    }
    // Qwen3.8-Flash-Next refuses what its binding cannot serve: no Host tier, Device snapshot
    // slots, a tool-call hold and a packed Host state.
    for (int variant = 0; variant < 4; ++variant) {
        EngineOptions options;
        if (variant == 0) { options.context_cache.host_capacity_bytes = 0; }
        if (variant == 1) { options.context_cache.hybrid.device_snapshot_slots = 1; }
        if (variant == 2) { options.context_cache.hybrid.tool_call_hold = std::chrono::seconds(30); }
        if (variant == 3) {
            options.context_cache.hybrid.host_state_precision = infernix::HybridHostStatePrecision::Fp24;
        }
        bool rejected = false;
        try {
            (void)normalize_engine_options(options, infernix::models::Architecture::Qwen4Exp);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "Qwen3.8-Flash-Next normalization accepted an unservable cache");
    }
    {
        EngineOptions options;
        options.context_cache.enabled = false;
        bool accepted = true;
        try {
            (void)normalize_engine_options(options, infernix::models::Architecture::Qwen4Exp);
        } catch (const std::invalid_argument&) { accepted = false; }
        failures += check(accepted, "Qwen3.8-Flash-Next refused a disabled context cache");
    }

    // The same normalization owns the speculative draft-confidence cut (--spec-min-p): Qwen3.8-Flash-Next's
    // MTP and DFlash2 rounds and Qwen3.5's MTP rounds have it, so every other combination is refused, not
    // ignored.
    for (int variant = 0; variant < 7; ++variant) {
        EngineOptions options;
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.draft_min_p  = variant == 3 ? 1.5F : 0.25F;
        auto architecture                = infernix::models::Architecture::Qwen4Exp;
        if (variant == 1 || variant == 5) { architecture = infernix::models::Architecture::Qwen3_5; }
        if (variant == 2) { options.speculative.backend = infernix::SpeculativeBackend::DFlash; }
        if (variant == 4) { options.speculative.draft_min_p = 0.0F; }
        if (variant == 5 || variant == 6) { options.speculative.backend = infernix::SpeculativeBackend::DFlash2; }
        bool rejected = false;
        try {
            const EngineOptions out = normalize_engine_options(options, architecture);
            failures += check(out.speculative.draft_min_p == options.speculative.draft_min_p,
                              "normalization changed --spec-min-p");
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected == (variant == 2 || variant == 3 || variant == 5),
                          "--spec-min-p was misjudged for its model, backend or range");
    }
    // --vram-headroom-mib with an explicit KV capacity: refused for Qwen3.5 (the headroom is what
    // automatic KV leaves), accepted with offloaded experts (what the expert cache leaves).
    {
        EngineOptions options;
        options.artifact_path       = "model.infernix";
        options.kv_capacity         = infernix::KvCapacityPolicy::explicit_capacity(8192);
        options.vram_headroom_bytes = std::size_t{1} << 30;
        bool rejected               = false;
        try {
            infernix::runtime::validate_engine_options(options, infernix::models::Architecture::Qwen3_5);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "--vram-headroom-mib with an explicit KV capacity was accepted on Qwen3.5");
        rejected = false;
        try {
            infernix::runtime::validate_engine_options(options, infernix::models::Architecture::Qwen3_5, true);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(!rejected, "--vram-headroom-mib was refused with offloaded experts");
    }
    // --draft-tokens auto on a Qwen3.5 artifact with offloaded experts: MTP keeps its drafter (its gated
    // drafts beat plain decode there), DFlash2 decodes without one (no policy beat plain decode); a fixed
    // length keeps its drafter.
    {
        EngineOptions options;
        options.speculative.backend           = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens_auto = true;
        const EngineOptions mtp = normalize_engine_options(options, infernix::models::Architecture::Qwen3_5, true);
        failures += check(mtp.speculative.backend == infernix::SpeculativeBackend::Mtp && mtp.speculative.draft_tokens_auto,
                          "--draft-tokens auto lost the MTP drafter with offloaded experts");
        options.speculative.backend = infernix::SpeculativeBackend::DFlash2;
        const EngineOptions block = normalize_engine_options(options, infernix::models::Architecture::Qwen3_5, true);
        failures += check(block.speculative.backend == infernix::SpeculativeBackend::None &&
                              !block.speculative.draft_tokens_auto && block.speculative.draft_tokens == 0,
                          "--draft-tokens auto kept the DFlash2 drafter with offloaded experts");
        options.speculative.backend           = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens_auto = false;
        options.speculative.draft_tokens      = 1;
        failures += check(normalize_engine_options(options, infernix::models::Architecture::Qwen3_5, true)
                                  .speculative.backend == infernix::SpeculativeBackend::Mtp,
                          "a fixed draft length lost its drafter with offloaded experts");
    }
    // --spec auto: the Engine picks the drafter from the artifact (Qwen3.8-Flash-Next: DFlash2 with a
    // --draft-model, else MTP; the Qwen3.5 family's measured order) before normalization, which refuses
    // an unresolved auto; without a drafter it decodes plain and drops every drafter setting.
    {
        using infernix::SpeculativeBackend;
        using infernix::models::Architecture;
        using infernix::runtime::resolve_auto_speculative_backend;
        const infernix::models::qwen3_5::BundledDrafters mtp_dflash2{.mtp = true, .dflash2 = true};
        EngineOptions options;
        options.speculative.backend_auto      = true;
        options.speculative.draft_tokens_auto = true;
        options.speculative.proposal_head     = infernix::ProposalHead::Optimized;
        const auto resolved = [&](EngineOptions in, Architecture architecture, bool offloaded,
                                  const infernix::models::qwen3_5::BundledDrafters& drafters) {
            resolve_auto_speculative_backend(in, architecture, offloaded, drafters);
            return in;
        };
        EngineOptions flash_next_dflash2           = options;
        flash_next_dflash2.speculative.draft_model = "drafter.infernix";
        const EngineOptions with_drafter           = resolved(flash_next_dflash2, Architecture::Qwen4Exp, false, {.mtp = true});
        const EngineOptions without_drafter        = resolved(options, Architecture::Qwen4Exp, false, {.mtp = true});
        failures += check(with_drafter.speculative.backend == SpeculativeBackend::DFlash2 &&
                              !with_drafter.speculative.backend_auto &&
                              without_drafter.speculative.backend == SpeculativeBackend::Mtp &&
                              without_drafter.speculative.draft_tokens_auto &&
                              without_drafter.speculative.proposal_head == infernix::ProposalHead::Optimized,
                          "--spec auto on Qwen3.8-Flash-Next did not take DFlash2 with --draft-model, else MTP");
        failures += check(resolved(options, Architecture::Qwen3_5, false, mtp_dflash2).speculative.backend ==
                                  SpeculativeBackend::DFlash2 &&
                              resolved(options, Architecture::Qwen3_5, true, mtp_dflash2).speculative.backend ==
                                  SpeculativeBackend::Mtp,
                          "--spec auto on a Qwen3.5 artifact did not take its measured drafter");
        EngineOptions trees                    = options;
        trees.speculative.draft_tree_auto      = true;
        trees.speculative.ngram_draft_tokens_auto = true;
        const EngineOptions plain = resolved(trees, Architecture::Qwen3_5, false, {});
        failures += check(plain.speculative.backend == SpeculativeBackend::None && !plain.speculative.backend_auto &&
                              !plain.speculative.draft_tokens_auto &&
                              plain.speculative.proposal_head == infernix::ProposalHead::Full &&
                              !plain.speculative.draft_tree_auto && !plain.speculative.ngram_draft_tokens_auto,
                          "--spec auto without a drafter kept drafter settings");
        failures += check(normalize_engine_options(plain).speculative.backend == SpeculativeBackend::None &&
                              normalize_engine_options(resolved(options, Architecture::Qwen3_5, false, mtp_dflash2))
                                      .speculative.backend == SpeculativeBackend::DFlash2,
                          "a resolved --spec auto did not normalize");
        bool rejected = false;
        try {
            (void)normalize_engine_options(options);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "an unresolved --spec auto was normalized");
        for (const bool fixed : {true, false}) {
            EngineOptions bad = options;
            if (fixed) {
                bad.speculative.draft_tokens_auto = false;
                bad.speculative.draft_tokens      = 3;
            } else {
                bad.speculative.draft_min_p = 0.5F;
            }
            rejected = false;
            try {
                (void)resolved(bad, Architecture::Qwen3_5, false, mtp_dflash2);
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected, fixed ? "--spec auto was resolved with a fixed draft length"
                                              : "--spec auto was resolved with --spec-min-p");
        }
        EngineOptions scoring = options;
        scoring.purpose       = infernix::EnginePurpose::CausalScoring;
        failures += check(normalize_engine_options(resolved(scoring, Architecture::Qwen3_5, false, mtp_dflash2))
                                  .speculative.backend == SpeculativeBackend::None,
                          "causal scoring did not drop --spec auto");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
