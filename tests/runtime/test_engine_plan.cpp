// Engine::plan's shared steps (runtime/engine/engine_plan.h): a plan resolves a capacity curve against the
// free memory less the weights exactly as a start does (resolve_kv_capacity), refuses in the start's own
// words, keeps its first refusal, takes the start's expert-pool slack off its frames and its weights' pool
// beside the arena, totals its RAM, and finds the largest --max-context a start accepts. Pure: no GPU.

#include "core/evictable_weight_pool.h"
#include "runtime/engine/engine_plan.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/expert_offload/memory_plan.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace infernix;
using namespace infernix::runtime;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::string refusal_of(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve, std::size_t free_bytes,
                       std::size_t headroom) {
    try {
        (void)resolve_kv_capacity(policy, curve, free_bytes, headroom);
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

// A Qwen3.5-like family of curves: 64-token page groups, a fixed part and a per-page part, the smallest
// capacity one lane's context (at least one page per lane).
SequenceCapacityCurve curve_for(std::uint32_t context, std::uint32_t lanes) {
    const std::uint32_t pages = (context + 63U) / 64U;
    const std::uint32_t least = pages > lanes ? pages : lanes;
    return SequenceCapacityCurve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = least,
        .maximum_main_page_groups             = least * lanes + 64,
        .minimum_device_reservation_bytes     = 4000 + 100ULL * least,
        .bytes_per_additional_main_page_group = 100,
    };
}

} // namespace

int main() {
    int failures = 0;

    // ---- an automatic capacity that fits: the start's resolution, and the device totals ----
    {
        EnginePlan plan;
        plan.vram.weights_bytes          = 20000;
        const SequenceCapacityCurve curve = curve_for(8192, 1); // 128 pages: 16,800 bytes at least
        const std::size_t free_bytes     = planned_free_after_weights(60000, plan.vram.weights_bytes);
        failures += check(free_bytes == 40000, "the free memory a start sizes against excludes the weights");
        const auto resolved = plan_capacity(plan, KvCapacityPolicy::automatic(), curve, free_bytes, 1000);
        const auto start    = resolve_kv_capacity(KvCapacityPolicy::automatic(), curve, free_bytes, 1000);
        failures += check(resolved && resolved->main_page_groups == start.main_page_groups &&
                              resolved->runtime_reservation_bytes == start.runtime_reservation_bytes,
                          "a plan resolves the capacity a start resolves");
        failures += check(plan.fits && plan.error.empty(), "a fitting capacity leaves the plan fitting");
        failures += check(plan.vram.kv_tokens == start.resolved_tokens && plan.vram.headroom_bytes == 1000 &&
                              plan.vram.minimum_reservation_bytes == curve.minimum_device_reservation_bytes &&
                              plan.vram.bytes_per_page_group == 100U && plan.vram.page_group_tokens == 64U,
                          "the curve and the resolution are recorded");
        failures += check(plan.vram.planned_bytes == 20000 + start.runtime_reservation_bytes &&
                              plan.vram.left_bytes == free_bytes - start.runtime_reservation_bytes - 1000,
                          "planned bytes and what is left beside the headroom");
    }

    // ---- an automatic capacity that does not fit: the start's own refusal ----
    {
        EnginePlan plan;
        plan.vram.weights_bytes          = 20000;
        const SequenceCapacityCurve curve = curve_for(65536, 1); // 1,024 pages: 106,400 bytes at least
        const auto resolved = plan_capacity(plan, KvCapacityPolicy::automatic(), curve, 40000, 1000);
        failures += check(!resolved && !plan.fits, "a capacity beyond the free memory does not fit");
        failures += check(plan.error == refusal_of(KvCapacityPolicy::automatic(), curve, 40000, 1000) &&
                              !plan.error.empty(),
                          "the plan's error is the start's refusal");
        failures += check(plan.vram.kv_tokens == 0 && plan.vram.left_bytes == 0 &&
                              plan.vram.planned_bytes == 20000 + curve.minimum_device_reservation_bytes,
                          "a refused capacity plans the smallest reservation and leaves nothing");
    }

    // ---- an explicit capacity carries no headroom; the first refusal is kept ----
    {
        EnginePlan plan;
        const SequenceCapacityCurve curve = curve_for(8192, 1);
        const auto resolved =
            plan_capacity(plan, KvCapacityPolicy::explicit_capacity(8192), curve, 16800, 5000);
        failures += check(resolved && resolved->main_page_groups == 128 && plan.vram.headroom_bytes == 0,
                          "an explicit capacity resolves without headroom, as a start's does");
        plan_refuse(plan, "first");
        plan_refuse(plan, "second");
        failures += check(!plan.fits && plan.error == "first", "the first refusal is the plan's error");
    }

    // ---- what a start keeps off its expert frames, which the plan's frames leave off too ----
    {
        constexpr std::uint64_t kChunk = 64ULL << 20;
        // A GGUF import's layers of three record sizes: three pools, two chunks of slack in average frames.
        const std::uint64_t strides[] = {6'291'456, 5'242'880, 6'291'456, 4'194'304, 5'242'880};
        failures += check(expert_offload::frame_pools(strides) == 3, "one pool per distinct record stride");
        // 128 MiB in frames of their mean stride (5,452,596 bytes): 24 whole frames.
        failures += check(expert_offload::pool_slack_frames(3, kChunk, 5'452'596) == 24,
                          "one chunk of slack per pool beyond the first, in average frames");
        const std::uint64_t nvfp4[] = {7'340'032, 7'340'032, 7'340'032};
        failures += check(expert_offload::frame_pools(nvfp4) == 1 &&
                              expert_offload::pool_slack_frames(1, kChunk, 7'340'032) == 0,
                          "one record size: one pool and no slack");
        failures += check(expert_offload::pool_slack_frames(3, kChunk, 0) == 0, "no frame size: no slack");
    }

    // ---- --vision-offload on: the weights' pool beside the arena, and its pinned tail mirror ----
    {
        constexpr std::size_t kChunk = EvictableWeightPool::kChunkBytes;
        // The pool maps the arena in whole chunks; an overlay window borrows evicted chunks, nothing more.
        failures += check(EvictableWeightPool::mapped_bytes(10 * kChunk + 5) == 11 * kChunk &&
                              EvictableWeightPool::mapped_bytes(10 * kChunk) == 10 * kChunk,
                          "the weights' pool maps the arena in whole chunks");
        failures += check(planned_free_after_weights(100 * kChunk, 10 * kChunk + 5,
                                                     EvictableWeightPool::mapped_bytes(10 * kChunk + 5) -
                                                         (10 * kChunk + 5)) == 89 * kChunk,
                          "a plan sizes beside the pool's chunks, not beside a window's staging");
        // The loader starts the evictable tail on a chunk (Binder::finish), so the mirror is the whole tail.
        failures += check(EvictableWeightPool::mirror_bytes(10 * kChunk + 5, 3 * kChunk + 5) == 3 * kChunk + 5,
                          "the mirror holds the evictable tail's weights");
        failures += check(EvictableWeightPool::mirror_bytes(10 * kChunk + 5, 3 * kChunk + 9) == 3 * kChunk + 5,
                          "the mirror starts at the tail's first whole chunk");
        failures += check(EvictableWeightPool::mirror_bytes(10 * kChunk, 0) == 0, "no ladder: no mirror");
    }

    // ---- the RAM totals ----
    {
        EnginePlan plan;
        plan.ram.available_bytes = 100;
        plan.ram.ram_free_bytes  = 30;
        plan_ram_totals(plan, 50);
        failures += check(plan.ram.planned_bytes == 50 && plan.ram.left_bytes == 20, "RAM left beside the reserve");
        plan_ram_totals(plan, 90);
        failures += check(plan.ram.left_bytes == 0, "an overplanned RAM leaves nothing, not a wrapped count");
        failures += check(planned_free_after_weights(100, 60, 50) == 0, "weights beyond the free memory leave none");
    }

    // ---- the largest --max-context: the boundary a start's own resolution draws ----
    {
        const std::size_t free_bytes = 40000;
        const std::size_t headroom   = 1000;
        const auto fits_automatic    = [&](std::uint32_t context) {
            (void)resolve_kv_capacity(KvCapacityPolicy::automatic(), curve_for(context, 2), free_bytes, headroom);
            return true;
        };
        // 4000 + 100 * pages + 1000 <= 40000: 350 pages, 22,400 tokens; from a context that fits and
        // from one that does not.
        for (const std::uint32_t asked : {8192U, 65536U, 22400U, 22401U, 1U}) {
            const auto largest = largest_fitting_context(1U << 20, asked, fits_automatic);
            failures += check(largest == 22400U, "the largest context is the last whole page that fits");
        }
        failures += check(fits_automatic(22400) && refusal_of(KvCapacityPolicy::automatic(), curve_for(22401, 2),
                                                              free_bytes, headroom) != "",
                          "one more token needs another page that does not fit");
        // A context the options cannot take (beyond the model's positions) throws: it does not fit.
        const auto bounded = largest_fitting_context(1U << 20, 8192, [&](std::uint32_t context) {
            if (context > 16384) { throw std::invalid_argument("max_context exceeds the configured position capacity"); }
            return fits_automatic(context);
        });
        failures += check(bounded == 16384U, "the model's positions bound the context");
        // A planner that refuses tiny contexts (below a prefill chunk, say) does not hide the answer.
        const auto floor = largest_fitting_context(1U << 20, 8192, [&](std::uint32_t context) {
            if (context < 128) { throw std::invalid_argument("context below the chunk"); }
            return fits_automatic(context);
        });
        failures += check(floor == 22400U, "the search starts at the asked context");
        failures += check(largest_fitting_context(1U << 20, 8192, [](std::uint32_t) { return false; }) == std::nullopt,
                          "nothing fits: no context");
        failures += check(largest_fitting_context(4096, 8192, [](std::uint32_t) { return true; }) == 4096U,
                          "everything fits: the bound");
    }

    return failures == 0 ? 0 : 1;
}
