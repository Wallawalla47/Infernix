#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"
#include "runtime/engine/model_instance.h"

#include <chrono>
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
    }
    // --draft-tokens auto takes the model's most drafts unless given a ceiling (Qwen3.8-Flash-Next 7,
    // Qwen3.5 5); adaptive-mtp is auto's policy, so a fixed length refuses it.
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
        EngineOptions dflash               = options;
        dflash.speculative.backend         = infernix::SpeculativeBackend::DFlash2;
        rejected = false;
        try {
            (void)normalize_engine_options(dflash);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "--draft-tokens auto was accepted without MTP");
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

    // The same normalization owns the speculative draft-confidence gate (--spec-min-p): only
    // Qwen3.8-Flash-Next's MTP rounds have it, so every other combination is refused, not ignored.
    for (int variant = 0; variant < 5; ++variant) {
        EngineOptions options;
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.draft_min_p  = variant == 3 ? 1.5F : 0.25F;
        auto architecture                = infernix::models::Architecture::Qwen4Exp;
        if (variant == 1) { architecture = infernix::models::Architecture::Qwen3_5; }
        if (variant == 2) { options.speculative.backend = infernix::SpeculativeBackend::DFlash; }
        if (variant == 4) { options.speculative.draft_min_p = 0.0F; }
        bool rejected = false;
        try {
            const EngineOptions out = normalize_engine_options(options, architecture);
            failures += check(out.speculative.draft_min_p == options.speculative.draft_min_p,
                              "normalization changed --spec-min-p");
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected == (variant == 1 || variant == 2 || variant == 3),
                          "--spec-min-p was misjudged for its model, backend or range");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
