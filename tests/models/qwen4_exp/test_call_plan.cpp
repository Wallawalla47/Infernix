// The prefill call planner of Qwen4Exp's prefix cache (program/prefix/call_plan.h; design §19.3.1,
// test U2), host only. Checks: the cold grid for F = 0 and the shifted grid after a resume; calls
// cover [F, n) in order without overlap or gaps and never exceed the chunk; kept exact taps end a
// call; boundary taps are kept at any cost; an automatic tap is kept when the call it starts is
// CPU-served and it adds at most the split budget (the generation opener's tail) and demoted to
// flexible otherwise (a split before a GPU-staged call, even one that only shifts the grid); the layer
// walk's spans (none for a cut
// between walk-eligible calls, one for a cut that leaves a narrow call inside the walk, one per
// 65,536 tokens); Δ accounting against an independent enumeration of call widths; determinism;
// the remaining calls planned again at a narrower chunk (segment_ends) keep the exact taps' ends;
// refused inputs; Vision span membership. The oracle for
// random plans re-derives the calls from the kept taps with a plain loop and re-runs the admission
// rule with its own cost sum.

#include "models/qwen4_exp/program/prefix/call_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pc = infernix::runtime::prefix_cache;
namespace q4 = infernix::models::qwen4_exp::prefix;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

q4::CallCost production_cost(std::uint32_t chunk) {
    q4::CallCost cost;
    cost.model.chunk_seconds       = 1.28;
    cost.model.chunk_tokens        = chunk;
    cost.model.token_seconds       = 1.17e-3;
    cost.model.call_route_fraction = 10.0 / 512.0;
    return cost;
}

pc::PlannedTap exact(std::uint32_t p, bool boundary = false) {
    return {.position = p, .placement = pc::TapPlacement::Exact, .boundary = boundary};
}
pc::PlannedTap flexible(std::uint32_t p) { return {.position = p, .placement = pc::TapPlacement::Flexible}; }

// Oracle: calls of [F, n) cut at `cuts`, by stepping each segment.
std::vector<std::uint32_t> oracle_ends(std::uint32_t F, std::uint32_t n, std::uint32_t c, std::vector<std::uint32_t> cuts) {
    cuts.push_back(n);
    std::vector<std::uint32_t> out;
    std::uint32_t at = F;
    for (const std::uint32_t cut : cuts) {
        while (at < cut) {
            at = std::min(at + c, cut);
            out.push_back(at);
        }
    }
    return out;
}

double oracle_seconds(std::uint32_t F, const std::vector<std::uint32_t>& ends, const q4::CallCost& cost) {
    double total = 0;
    for (std::size_t i = 0; i < ends.size(); ++i) {
        const std::uint32_t w = ends[i] - (i == 0 ? F : ends[i - 1]);
        total += w <= cost.served_columns ? cost.served_call_seconds : cost.model.call_seconds(w);
    }
    return total;
}

bool contains(const std::vector<std::uint32_t>& v, std::uint32_t x) { return std::find(v.begin(), v.end(), x) != v.end(); }

} // namespace

int main() {
    try {
        const q4::CallCost cost = production_cost(4096);
        // Cold grid and a resume's shifted grid, no taps.
        check(q4::plan_calls(0, 10000, 4096, {}, cost).ends == std::vector<std::uint32_t>{4096, 8192, 10000},
              "F = 0 without taps is today's grid");
        check(q4::plan_calls(300, 10000, 4096, {}, cost).ends == std::vector<std::uint32_t>{4396, 8492, 10000},
              "a resume steps by the chunk from its frontier");
        check(q4::plan_calls(9999, 10000, 4096, {}, cost).ends == std::vector<std::uint32_t>{10000}, "a one-token suffix");

        // The generation opener 5 tokens before the end: one CPU-served call more, kept.
        {
            const pc::PlannedTap taps[] = {exact(9995)};
            const q4::CallPlan plan = q4::plan_calls(0, 10000, 4096, taps, cost);
            check(plan.ends == std::vector<std::uint32_t>{4096, 8192, 9995, 10000} &&
                      plan.taps[0].placement == pc::TapPlacement::Exact,
                  "the opener splits off a 5-token call");
            const double delta = plan.seconds - q4::plan_calls(0, 10000, 4096, {}, cost).seconds;
            check(delta > 0 && delta <= cost.split_budget, "the opener's Δ is within the split budget (" +
                                                                std::to_string(delta) + " s)");
        }
        // An automatic tap before a GPU-staged call is demoted even where it only shifts the grid (same
        // call count); a boundary tap there is kept and the next segment's grid starts at it.
        {
            const pc::PlannedTap taps[] = {exact(2000)};
            const q4::CallPlan plan = q4::plan_calls(0, 10000, 4096, taps, cost);
            check(plan.ends == std::vector<std::uint32_t>{4096, 8192, 10000} &&
                      plan.taps[0].placement == pc::TapPlacement::Flexible,
                  "an automatic split before a GPU-staged call is demoted");
            const pc::PlannedTap boundary[] = {exact(2000, true)};
            check(q4::plan_calls(0, 10000, 4096, boundary, cost).ends == std::vector<std::uint32_t>{2000, 6096, 10000},
                  "a boundary split starts the next segment's grid");
        }
        // A mid-chunk automatic tap of a one-call suffix costs a whole call: demoted to flexible.
        {
            const pc::PlannedTap taps[] = {exact(2000)};
            const q4::CallPlan plan = q4::plan_calls(0, 4096, 4096, taps, cost);
            check(plan.ends == std::vector<std::uint32_t>{4096} && plan.taps[0].placement == pc::TapPlacement::Flexible,
                  "an expensive automatic split is demoted");
        }
        // Explicit and structural taps (boundary) are kept at any cost.
        {
            const pc::PlannedTap taps[] = {exact(2000, true)};
            const q4::CallPlan plan = q4::plan_calls(0, 4096, 4096, taps, cost);
            check(plan.ends == std::vector<std::uint32_t>{2000, 4096} && plan.taps[0].placement == pc::TapPlacement::Exact,
                  "a boundary tap splits regardless of cost");
        }
        // With the layer walk's spans priced (one per run of calls of at least 256 columns, cut at
        // 65,536 tokens): a cut between walk-eligible calls is captured inside the span and costs no
        // span, a cut that leaves a narrower call inside the walk costs one, and the opener's cut
        // before a CPU-served tail costs none.
        {
            q4::CallCost spans = cost;
            spans.span_seconds = 1.67;
            const auto extra   = [&](std::uint32_t n, std::span<const pc::PlannedTap> taps) {
                return q4::plan_calls(0, n, 4096, taps, spans).seconds - q4::plan_calls(0, n, 4096, taps, cost).seconds;
            };
            const pc::PlannedTap shift[] = {exact(2000, true)};
            check(std::fabs(extra(10000, shift) - 1.67) < 1e-9, "a boundary split between walk-eligible calls keeps the span");
            const pc::PlannedTap opener[] = {exact(9995)};
            check(q4::plan_calls(0, 10000, 4096, opener, spans).taps[0].placement == pc::TapPlacement::Exact &&
                      std::fabs(extra(10000, opener) - 1.67) < 1e-9,
                  "a cut before a CPU-served call costs no span");
            const pc::PlannedTap narrow[] = {exact(4196)};
            check(q4::plan_calls(0, 10000, 4096, narrow, spans).taps[0].placement == pc::TapPlacement::Flexible,
                  "a cut that leaves a 100-column call inside the walk is demoted");
            const pc::PlannedTap narrow_boundary[] = {exact(4196, true)};
            check(std::fabs(extra(10000, narrow_boundary) - 2 * 1.67) < 1e-9,
                  "a boundary cut that leaves a 100-column call inside the walk is charged a second span");
            check(std::fabs(extra(140000, {}) - 3 * 1.67) < 1e-9, "a 140,000-token prompt walks three spans");
        }
        // Flexible taps never cut.
        {
            const pc::PlannedTap taps[] = {flexible(1000), flexible(9000)};
            check(q4::plan_calls(0, 10000, 4096, taps, cost).ends == std::vector<std::uint32_t>{4096, 8192, 10000},
                  "flexible taps leave the calls unchanged");
        }

        // Random plans against the oracle.
        std::mt19937 rng(20261005);
        int bad = 0;
        for (int trial = 0; trial < 4000 && bad < 5; ++trial) {
            const std::uint32_t c = std::uniform_int_distribution<std::uint32_t>(1, 4)(rng) == 1 ? 7U : 4096U;
            const std::uint32_t n = std::uniform_int_distribution<std::uint32_t>(2, 30000)(rng);
            const std::uint32_t F = std::uniform_int_distribution<std::uint32_t>(0, n - 1)(rng);
            const q4::CallCost cc = production_cost(c);
            std::vector<pc::PlannedTap> taps;
            for (std::uint32_t p = F + 1; p < n; p += std::uniform_int_distribution<std::uint32_t>(1, 6000)(rng)) {
                const int kind = std::uniform_int_distribution<int>(0, 2)(rng);
                taps.push_back(kind == 0 ? flexible(p) : exact(p, kind == 2));
            }
            const q4::CallPlan plan = q4::plan_calls(F, n, c, taps, cc);
            // Admission re-run with the oracle's costs.
            std::vector<std::uint32_t> cuts;
            for (const auto& t : taps) {
                if (t.placement == pc::TapPlacement::Exact && t.boundary) { cuts.push_back(t.position); }
            }
            std::vector<pc::TapPlacement> placements;
            for (const auto& t : taps) {
                if (t.placement != pc::TapPlacement::Exact || t.boundary) {
                    placements.push_back(t.placement);
                    continue;
                }
                std::vector<std::uint32_t> with = cuts;
                with.insert(std::upper_bound(with.begin(), with.end(), t.position), t.position);
                const std::vector<std::uint32_t> with_ends = oracle_ends(F, n, c, with);
                const double delta = oracle_seconds(F, with_ends, cc) - oracle_seconds(F, oracle_ends(F, n, c, cuts), cc);
                // The call the tap starts: the first end past it.
                std::uint32_t next = n;
                for (const std::uint32_t e : with_ends) {
                    if (e > t.position) {
                        next = e;
                        break;
                    }
                }
                if (next - t.position <= cc.served_columns && delta <= cc.split_budget) {
                    cuts = with;
                    placements.push_back(pc::TapPlacement::Exact);
                } else {
                    placements.push_back(pc::TapPlacement::Flexible);
                }
            }
            bool ok = plan.ends == oracle_ends(F, n, c, cuts) && plan.taps.size() == taps.size();
            for (std::size_t i = 0; ok && i < taps.size(); ++i) {
                ok = plan.taps[i].position == taps[i].position && plan.taps[i].placement == placements[i] &&
                     (placements[i] != pc::TapPlacement::Exact || contains(plan.ends, taps[i].position));
            }
            // Coverage: strictly increasing ends from past F to n, each call at most a chunk.
            std::uint32_t at = F;
            for (const std::uint32_t end : plan.ends) {
                ok = ok && end > at && end - at <= c;
                at = end;
            }
            ok = ok && at == n && plan.seconds == q4::plan_calls(F, n, c, taps, cc).seconds &&
                 plan.ends == q4::plan_calls(F, n, c, taps, cc).ends;
            if (!ok) {
                ++bad;
                check(false, "random plan F=" + std::to_string(F) + " n=" + std::to_string(n) + " c=" +
                                 std::to_string(c) + " taps=" + std::to_string(taps.size()));
            }
            // The remaining calls planned again at a narrower chunk from a call boundary (the default
            // route's cut when the cache shrinks mid-prompt): the oracle's grid from there, every kept
            // exact tap past it still ending a call.
            const std::size_t done = std::uniform_int_distribution<std::size_t>(0, plan.ends.size() - 1)(rng);
            const std::uint32_t from = done == 0 ? F : plan.ends[done - 1];
            const std::uint32_t narrower = std::uniform_int_distribution<std::uint32_t>(1, c)(rng);
            std::vector<std::uint32_t> remaining;
            for (const auto& t : plan.taps) {
                if (t.placement == pc::TapPlacement::Exact && t.position > from) { remaining.push_back(t.position); }
            }
            const std::vector<std::uint32_t> again = q4::segment_ends(from, n, narrower, remaining);
            bool replanned = again == oracle_ends(from, n, narrower, remaining);
            for (const std::uint32_t p : remaining) { replanned = replanned && contains(again, p); }
            if (!replanned && bad < 5) {
                ++bad;
                check(false, "replanned F=" + std::to_string(from) + " n=" + std::to_string(n) + " c=" +
                                 std::to_string(narrower) + " cuts=" + std::to_string(remaining.size()));
            }
        }

        // Refused inputs.
        const auto refuses = [&](auto&& call) {
            try {
                call();
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        const pc::PlannedTap unsorted[] = {exact(50), exact(40)};
        const pc::PlannedTap at_frontier[] = {exact(10)};
        const pc::PlannedTap at_end[] = {exact(100)};
        check(refuses([&] { (void)q4::plan_calls(0, 100, 64, unsorted, cost); }), "unsorted taps are refused");
        check(refuses([&] { (void)q4::plan_calls(10, 100, 64, at_frontier, cost); }), "a tap at the frontier is refused");
        check(refuses([&] { (void)q4::plan_calls(0, 100, 64, at_end, cost); }), "a tap at the prompt end is refused");
        check(refuses([&] { (void)q4::plan_calls(100, 100, 64, {}, cost); }), "an empty suffix is refused");
        check(refuses([&] { (void)q4::plan_calls(0, 100, 0, {}, cost); }), "a zero chunk is refused");

        // Vision spans: only frontiers strictly inside are excluded.
        const pc::TapExclusion spans[] = {{10, 74}, {100, 164}};
        check(!q4::inside_exclusion(10, spans) && q4::inside_exclusion(11, spans) && q4::inside_exclusion(73, spans) &&
                  !q4::inside_exclusion(74, spans) && !q4::inside_exclusion(99, spans) && q4::inside_exclusion(163, spans),
              "a frontier at an item's first or end token is outside it");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "qwen4_exp call plan checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
