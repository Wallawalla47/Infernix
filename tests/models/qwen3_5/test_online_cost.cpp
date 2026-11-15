// The prefix cache's online cost corrections (runtime/engine/context_cache/online_cost.h), host only:
// - the Host-to-Device rate: the first large restore's bytes over its copy time replaces the compiled coefficient
//   (51.5 GB/s compiled against a link that moves 27.5 GB/s priced a restore at half its time); later restores
//   move it by a quarter each; a small restore, an unreadable interval and an implausible rate change nothing;
// - the prefill scale: 1 until three prompts were measured, then the measured seconds over the predicted ones,
//   weighted by length and bounded; it scales the model's prefill terms and leaves its transfer terms alone.

#include "runtime/engine/context_cache/online_cost.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

using infernix::runtime::blend_h2d_rate;
using infernix::runtime::OnlinePrefillScale;
using infernix::runtime::prefix_cache::CacheCostModel;

int check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; }
    return condition ? 0 : 1;
}

bool near(double value, double expected, double tolerance = 1.0e-9) {
    return std::abs(value - expected) <= tolerance * std::max(1.0, std::abs(expected));
}

} // namespace

int main() {
    int failures = 0;

    // 5.5 GB in 0.2 s: 27.5 GB/s, taken whole as the first sample.
    const std::uint64_t restore = 5'500'000'000ULL;
    const double first          = blend_h2d_rate(0.0, restore, 0.2);
    failures += check(near(first, 27.5e9), "the first large restore sets the measured rate");
    // A later restore at 23.5 GB/s moves it a quarter of the way.
    failures += check(near(blend_h2d_rate(first, 4'700'000'000ULL, 0.2), 26.5e9),
                      "a later restore moves the rate by a quarter of the difference");
    failures += check(blend_h2d_rate(first, (64ULL << 20) - 1U, 0.001) == first &&
                          blend_h2d_rate(0.0, (64ULL << 20) - 1U, 0.001) == 0.0,
                      "a restore under 64 MiB is no link sample");
    failures += check(blend_h2d_rate(first, restore, 0.0) == first && blend_h2d_rate(first, restore, -1.0) == first,
                      "an interval that is not positive is no sample");
    failures += check(blend_h2d_rate(first, restore, 100.0) == first && blend_h2d_rate(first, restore, 1.0e-3) == first,
                      "a rate under 1 GB/s or over 200 GB/s is no sample");

    // A restore priced with the measured rate against the compiled one.
    CacheCostModel model;
    model.h2d_bytes_per_second = 51.5e9;
    const double compiled      = model.restore_seconds(restore);
    model.h2d_bytes_per_second = first;
    failures += check(near(model.restore_seconds(restore), model.transfer_batch_seconds + 0.2) &&
                          model.restore_seconds(restore) > 1.8 * compiled,
                      "the measured rate prices the restore at its own time");

    OnlinePrefillScale scale;
    CacheCostModel base;
    base.chunk_seconds          = 0.040;
    base.chunk_tokens           = 1024;
    base.token_seconds          = 2.4e-4;
    base.attention_pair_seconds = 7.0e-9;
    const auto predicted        = [&](std::uint32_t tokens) { return base.prefill_seconds(0, tokens); };
    scale.observe(2.0 * predicted(4096), predicted(4096));
    scale.observe(2.0 * predicted(8192), predicted(8192));
    failures += check(scale.scale() == 1.0 && near(scale.apply(base).token_seconds, base.token_seconds),
                      "two prompts do not move the scale");
    scale.observe(2.0 * predicted(2048), predicted(2048));
    failures += check(near(scale.scale(), 2.0), "an artifact twice as slow as the model scales it by two");
    const CacheCostModel scaled = scale.apply(base);
    failures += check(near(scaled.prefill_seconds(1000, 5000), 2.0 * base.prefill_seconds(1000, 5000)) &&
                          scaled.h2d_bytes_per_second == base.h2d_bytes_per_second &&
                          scaled.transfer_batch_seconds == base.transfer_batch_seconds &&
                          scaled.chunk_tokens == base.chunk_tokens,
                      "the scale applies to every prefill term and to no transfer term");

    // Length-weighted: a long prompt at 1.0 outweighs a short one at 3.0.
    OnlinePrefillScale weighted;
    weighted.observe(predicted(65536), predicted(65536));
    weighted.observe(predicted(65536), predicted(65536));
    weighted.observe(3.0 * predicted(1024), predicted(1024));
    failures += check(weighted.scale() > 1.0 && weighted.scale() < 1.02,
                      "the scale is the ratio of the summed seconds, not the mean of the ratios");

    // Bounded, and unusable samples are ignored.
    OnlinePrefillScale bounded;
    for (int i = 0; i < 4; ++i) { bounded.observe(1000.0, 1.0); }
    failures += check(bounded.scale() == OnlinePrefillScale::kMaximumScale, "the scale is bounded above");
    OnlinePrefillScale ignored;
    ignored.observe(0.0, 1.0);
    ignored.observe(1.0, 0.0);
    ignored.observe(-1.0, 1.0);
    failures += check(ignored.samples() == 0 && ignored.scale() == 1.0, "a non-positive time is no sample");

    // The newest prompts decide: after a change of speed the scale follows it.
    OnlinePrefillScale moving;
    for (int i = 0; i < 3; ++i) { moving.observe(predicted(4096), predicted(4096)); }
    for (int i = 0; i < 40; ++i) { moving.observe(1.5 * predicted(4096), predicted(4096)); }
    failures += check(moving.scale() > 1.45 && moving.scale() <= 1.5, "older prompts fade from the scale");

    if (failures == 0) { std::cout << "online cost tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
