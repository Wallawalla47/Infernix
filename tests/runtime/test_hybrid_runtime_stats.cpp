// The hybrid prefix cache's counters reach RuntimeStats from every family's cache that keeps them: Qwen3.8-Flash-Next
// retries a tap the Host tier refused for room (hybrid-prefix-cache-spec §7.4), so its taps_retried is published
// although its cache keeps none of Qwen3.5's Device-slot counters (those stay 0). One gate on
// `device_slot_refusals` for the whole group published Flash-Next's retried taps as 0.
#include "models/qwen4_exp/program/program.h"
#include "runtime/engine/context_cache/hybrid_resource_manager.h"

#include <iostream>

namespace {

struct StubProgram {
    infernix::models::qwen4_exp::HybridPrefixCacheStats stats;
    [[nodiscard]] infernix::models::qwen4_exp::HybridPrefixCacheStats hybrid_stats() const { return stats; }
};

struct StubModel {
    using Program         = StubProgram;
    using RequestBasePlan = int;
    using SourceCandidate = int;
    using ResumeState     = int;
};

} // namespace

int main() {
    StubProgram program;
    program.stats.taps_created = 5;
    program.stats.taps_skipped = 3;
    program.stats.taps_retried = 2;
    infernix::RuntimeStats out;
    infernix::runtime::HybridResourceManager<StubModel>{}.populate_runtime_stats(program, out);
    const bool ok = out.hybrid_taps_created == 5 && out.hybrid_taps_skipped == 3 && out.hybrid_taps_retried == 2 &&
                    out.hybrid_device_slot_refusals == 0 && out.hybrid_taps_skipped_no_slot == 0;
    if (!ok) {
        std::cerr << "FAIL: Flash-Next's retried taps did not reach RuntimeStats (taps_retried "
                  << out.hybrid_taps_retried << ")\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
