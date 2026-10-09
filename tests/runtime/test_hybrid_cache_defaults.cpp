#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"
#include "runtime/engine/model_instance.h"

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

    // Qwen3.8-Flash-Next resolves its own defaults: a 4 GiB Host tier, no Device snapshot slots
    // (its snapshots are Host-born), the common tap budget; Qwen3.5 keeps C + 1 slots and 8 GiB.
    {
        EngineOptions options;
        options.max_concurrency = 2;
        const infernix::ContextCacheOptions qwen4 =
            normalize_engine_options(options, infernix::models::Architecture::Qwen4Exp).context_cache;
        failures += check(qwen4.host_capacity_bytes == infernix::kDefaultQwen4ExpHybridHostCacheBytes &&
                              qwen4.hybrid.device_snapshot_slots == 0U && qwen4.hybrid.max_new_taps == 8U,
                          "Qwen3.8-Flash-Next prefix cache defaults are wrong");
        const infernix::ContextCacheOptions qwen35 = normalize_engine_options(options).context_cache;
        failures += check(qwen35.host_capacity_bytes == infernix::kDefaultHybridHostCacheBytes &&
                              qwen35.hybrid.device_snapshot_slots == 3U,
                          "Qwen3.5 prefix cache defaults changed");
    }
    // ... and refuses what its binding cannot serve: no Host tier and Device snapshot slots.
    for (int variant = 0; variant < 2; ++variant) {
        EngineOptions options;
        if (variant == 0) { options.context_cache.host_capacity_bytes = 0; }
        if (variant == 1) { options.context_cache.hybrid.device_snapshot_slots = 1; }
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

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
