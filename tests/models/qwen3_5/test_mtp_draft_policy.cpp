#include "models/qwen3_5/program/speculative/mtp_draft_policy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// adaptive-mtp draft-length policy (MtpDraftPolicy): the acceptance estimate, the choice of a
// longer length when drafts are accepted and a shorter one when they are not, hysteresis, the
// periodic probe of the next longer length, the common length of batch rounds among the batch
// lengths, and a request's estimate surviving a save and restore (preemption).
namespace {

using infernix::models::qwen3_5::detail::kMtpPolicyMaximumDrafts;
using infernix::models::qwen3_5::detail::mtp_batch_draft_length;
using infernix::models::qwen3_5::detail::MtpAcceptanceEstimate;
using infernix::models::qwen3_5::detail::MtpDraftPolicy;
using infernix::models::qwen3_5::detail::MtpLaneDraftState;

constexpr std::uint32_t kCeiling = 5;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool close_to(double a, double b) {
    return std::abs(a - b) <= 1e-12 * std::max(1.0, std::abs(b));
}

// Round times shaped like a dense target: a fixed verification cost, a little per verified
// column and a drafter step per draft.
void measure_linear(MtpDraftPolicy& policy) {
    for (std::uint32_t batch = 1; batch <= policy.max_batch(); ++batch) {
        for (const std::uint32_t k : policy.lengths(batch)) {
            policy.set_round_seconds(k, batch, 0.010 + 0.0004 * batch * (k + 1U) + 0.0012 * k);
        }
    }
}

MtpLaneDraftState lane_with(std::uint32_t extent, std::uint32_t accepted, int rounds) {
    MtpLaneDraftState lane;
    for (int i = 0; i < rounds; ++i) { lane.acceptance.observe(extent, accepted); }
    return lane;
}

void batch_lengths() {
    check(mtp_batch_draft_length(1, 5) && mtp_batch_draft_length(3, 5) &&
              mtp_batch_draft_length(5, 5),
          "1..3 and the ceiling are batch lengths");
    check(!mtp_batch_draft_length(4, 5) && !mtp_batch_draft_length(0, 5) &&
              !mtp_batch_draft_length(6, 5),
          "4 below a ceiling of 5, 0 and lengths past the ceiling are not");
    check(mtp_batch_draft_length(1, 1), "a ceiling of 1 batches at 1");

    const MtpDraftPolicy policy(kCeiling, 4);
    check(policy.lengths(1) == std::vector<std::uint32_t>{1, 2, 3, 4, 5},
          "one row chooses among every length up to the ceiling");
    check(policy.lengths(2) == std::vector<std::uint32_t>{1, 2, 3, 5} &&
              policy.lengths(4) == std::vector<std::uint32_t>{1, 2, 3, 5},
          "several rows choose among the batch lengths");
    check(policy.lengths(5).empty(), "a batch past the concurrency has no length");
}

void acceptance_estimate() {
    const MtpAcceptanceEstimate fresh{};
    const double p = MtpAcceptanceEstimate::kPriorAcceptance;
    check(close_to(fresh.expected_tokens(0), 1.0), "a round without drafts commits one token");
    check(close_to(fresh.expected_tokens(3), 1.0 + p + p * p + p * p * p),
          "before any round every position takes the prior");

    MtpAcceptanceEstimate all;
    MtpAcceptanceEstimate none;
    for (int i = 0; i < 40; ++i) {
        all.observe(5, 5);
        none.observe(5, 0);
    }
    check(all.acceptance(0) > 0.95 && all.acceptance(4) > 0.95,
          "drafts always accepted give an acceptance near one");
    check(none.acceptance(0) < 0.15, "drafts never accepted give an acceptance near zero");
    check(all.expected_tokens(5) > 5.5 && none.expected_tokens(5) < 1.2,
          "expected tokens follow the acceptance");
    check(all.expected_tokens(kMtpPolicyMaximumDrafts + 4) ==
              all.expected_tokens(kMtpPolicyMaximumDrafts),
          "lengths past the tracked positions are clamped");

    // Only the first draft was ever tested: deeper positions take the pooled acceptance.
    MtpAcceptanceEstimate first_only;
    for (int i = 0; i < 20; ++i) { first_only.observe(1, 1); }
    check(close_to(first_only.acceptance(3), first_only.acceptance(4)),
          "untested positions share the pooled acceptance");

    // A phase change: the estimate follows the recent rounds.
    MtpAcceptanceEstimate changing = all;
    for (int i = 0; i < 30; ++i) { changing.observe(5, 0); }
    check(changing.acceptance(0) < 0.3, "decay forgets an earlier phase");
}

void longer_when_accepted_shorter_when_not() {
    MtpDraftPolicy policy(kCeiling, 1);
    measure_linear(policy);
    check(policy.measured(1), "every length of one row is measured");

    MtpLaneDraftState high = lane_with(5, 5, 30);
    check(policy.choose(high) == kCeiling, "high acceptance chooses the ceiling");
    MtpLaneDraftState low = lane_with(5, 0, 30);
    check(policy.choose(low) == 1, "low acceptance chooses one draft");
    // Accepted about half the time: an interior length.
    MtpLaneDraftState half;
    for (int i = 0; i < 30; ++i) { half.acceptance.observe(5, (i % 2 == 0) ? 1U : 2U); }
    const std::uint32_t middle = policy.choose(half);
    check(middle > 1 && middle < kCeiling, "middling acceptance chooses an interior length");

    // Without measured times the fixed model decides, and the prior starts at 3 drafts.
    MtpDraftPolicy modelled(kCeiling, 1);
    check(!modelled.measured(1), "nothing is measured before startup timing");
    check(close_to(modelled.round_seconds(2, 1), 1.0 + 2.0 * MtpDraftPolicy::kModelDraftCost),
          "the fixed model costs 1 + 0.2 K");
    MtpLaneDraftState fresh;
    check(modelled.choose(fresh) == 3, "the prior under the fixed model starts at 3 drafts");
}

void hysteresis() {
    MtpDraftPolicy policy(kCeiling, 1);
    MtpLaneDraftState lane = lane_with(5, 3, 30);
    const double e3 = lane.acceptance.expected_tokens(3);
    const double e4 = lane.acceptance.expected_tokens(4);
    const auto set = [&](double four_over_three) {
        for (const std::uint32_t k : policy.lengths(1)) { policy.set_round_seconds(k, 1, 100.0); }
        policy.set_round_seconds(3, 1, 1.0);
        policy.set_round_seconds(4, 1, e4 / (e3 * four_over_three));
    };
    set(0.9);
    check(policy.choose(lane) == 3, "3 drafts lead");
    set(1.01);
    check(policy.choose(lane) == 3, "a lead under the margin keeps the current length");
    set(1.05);
    check(policy.choose(lane) == 4, "a lead past the margin switches");
    set(1.01);
    check(policy.choose(lane) == 4, "the margin also protects the new length");
}

void probes() {
    MtpDraftPolicy policy(kCeiling, 1);
    for (const std::uint32_t k : policy.lengths(1)) { policy.set_round_seconds(k, 1, 10.0); }
    policy.set_round_seconds(2, 1, 1.0);
    MtpLaneDraftState lane = lane_with(5, 2, 30);
    std::vector<std::uint32_t> chosen;
    for (std::uint32_t i = 0; i < 2 * MtpDraftPolicy::kProbePeriod + 2 * MtpDraftPolicy::kProbeRounds;
         ++i) {
        chosen.push_back(policy.choose(lane));
    }
    // Choices 1..15 run 2 drafts, the 16th and 17th probe 3, then 2 again until the next probe.
    bool ok = true;
    for (std::size_t i = 0; i < chosen.size(); ++i) {
        const std::size_t first  = MtpDraftPolicy::kProbePeriod - 1U;
        const std::size_t second = 2U * MtpDraftPolicy::kProbePeriod + MtpDraftPolicy::kProbeRounds - 2U;
        const bool probing       = (i >= first && i < first + MtpDraftPolicy::kProbeRounds) ||
                             (i >= second && i < second + MtpDraftPolicy::kProbeRounds);
        ok = ok && chosen[i] == (probing ? 3U : 2U);
    }
    check(ok, "every probe period runs the next longer length for two rounds");
    check(lane.memory.draft_length == 2, "a probe does not replace the chosen length");

    // At the ceiling there is nothing longer to probe.
    MtpDraftPolicy top(kCeiling, 1);
    for (const std::uint32_t k : top.lengths(1)) { top.set_round_seconds(k, 1, 1.0); }
    MtpLaneDraftState accepted = lane_with(5, 5, 30);
    bool at_ceiling = true;
    for (std::uint32_t i = 0; i < 3 * MtpDraftPolicy::kProbePeriod; ++i) {
        at_ceiling = at_ceiling && top.choose(accepted) == kCeiling;
    }
    check(at_ceiling, "the ceiling is never probed past");
}

void batch_choice() {
    MtpDraftPolicy policy(kCeiling, 4);
    measure_linear(policy);
    const MtpLaneDraftState high = lane_with(5, 5, 30);
    const MtpLaneDraftState low  = lane_with(5, 0, 30);
    const MtpLaneDraftState mid  = lane_with(5, 3, 30);

    std::array<const MtpAcceptanceEstimate*, 4> rows{&high.acceptance, &high.acceptance,
                                                     &high.acceptance, &high.acceptance};
    check(policy.choose_batch(std::span<const MtpAcceptanceEstimate* const>(rows.data(), 4)) ==
              kCeiling,
          "a batch of accepting rows runs the ceiling");
    rows = {&low.acceptance, &low.acceptance, &low.acceptance, &low.acceptance};
    check(policy.choose_batch(std::span<const MtpAcceptanceEstimate* const>(rows.data(), 4)) == 1,
          "a batch of rejecting rows runs one draft");

    // Rows that reach about 3 drafts, where 4 would be best for one row: the batch takes 3 or
    // the ceiling, never 4.
    MtpDraftPolicy narrow(kCeiling, 2);
    for (const std::uint32_t k : narrow.lengths(1)) { narrow.set_round_seconds(k, 1, 1.0 + 0.1 * k); }
    for (const std::uint32_t k : narrow.lengths(2)) { narrow.set_round_seconds(k, 2, 1.0 + 0.1 * k); }
    std::array<const MtpAcceptanceEstimate*, 2> pair{&mid.acceptance, &mid.acceptance};
    bool never_four = true;
    for (int i = 0; i < 40; ++i) {
        const std::uint32_t k =
            narrow.choose_batch(std::span<const MtpAcceptanceEstimate* const>(pair.data(), 2));
        never_four = never_four && narrow.serves(k, 2) && k != 4;
    }
    check(never_four, "batch rounds choose only batch lengths");

    // An expensive long batch round: the policy measures each batch size on its own.
    MtpDraftPolicy costly(kCeiling, 2);
    measure_linear(costly);
    costly.set_round_seconds(kCeiling, 2, 1.0);
    std::array<const MtpAcceptanceEstimate*, 2> accepting{&high.acceptance, &high.acceptance};
    check(costly.choose_batch(std::span<const MtpAcceptanceEstimate* const>(accepting.data(), 2)) ==
              3,
          "a slow ceiling at batch 2 falls back to the longest short length");
    MtpLaneDraftState single = high;
    check(costly.choose(single) == kCeiling, "the batch time does not affect one row");

    // Mixed rows sum their expected tokens.
    std::array<const MtpAcceptanceEstimate*, 2> mixed{&high.acceptance, &low.acceptance};
    const double rate5 =
        policy.rate(std::span<const MtpAcceptanceEstimate* const>(mixed.data(), 2), 5);
    check(close_to(rate5, (high.acceptance.expected_tokens(5) + low.acceptance.expected_tokens(5)) /
                          policy.round_seconds(5, 2)),
          "a batch's rate is its rows' summed expected tokens over the batch round time");
}

void save_and_restore() {
    MtpDraftPolicy policy(kCeiling, 1);
    measure_linear(policy);
    MtpLaneDraftState lane = lane_with(5, 5, 20);
    for (int i = 0; i < 5; ++i) { (void)policy.choose(lane); }
    // Preemption moves the lane state into the paused request; resume moves it back.
    const MtpLaneDraftState saved = lane;
    MtpLaneDraftState restored    = saved;
    check(restored == lane, "a restored state equals the saved one");
    std::vector<std::uint32_t> original;
    std::vector<std::uint32_t> resumed;
    for (int i = 0; i < 2 * static_cast<int>(MtpDraftPolicy::kProbePeriod); ++i) {
        lane.acceptance.observe(5, static_cast<std::uint32_t>(i % 6));
        restored.acceptance.observe(5, static_cast<std::uint32_t>(i % 6));
        original.push_back(policy.choose(lane));
        resumed.push_back(policy.choose(restored));
    }
    check(original == resumed, "a restored request makes the choices it would have made");

    // A request that lost its estimate (a fresh state) starts from the prior instead.
    MtpLaneDraftState fresh;
    MtpLaneDraftState learned = saved;
    check(policy.choose(learned) == kCeiling && policy.choose(fresh) != kCeiling,
          "the learned estimate, not the prior, steers a restored request");
}

void invalid_inputs() {
    bool threw = false;
    try {
        MtpDraftPolicy policy(0, 1);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a zero ceiling is rejected");
    threw = false;
    try {
        MtpDraftPolicy policy(kCeiling, 2);
        policy.set_round_seconds(4, 2, 1.0);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a length a batch size does not serve cannot be timed");
    threw = false;
    try {
        MtpDraftPolicy policy(kCeiling, 2);
        policy.set_round_seconds(2, 1, 0.0);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a round time must be positive");
    threw = false;
    try {
        MtpDraftPolicy policy(kCeiling, 2);
        const MtpLaneDraftState lane{};
        std::array<const MtpAcceptanceEstimate*, 1> one{&lane.acceptance};
        (void)policy.choose_batch(std::span<const MtpAcceptanceEstimate* const>(one.data(), 1));
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a batch choice needs two or more rows");

    MtpDraftPolicy single(1, 1);
    MtpLaneDraftState lane = lane_with(1, 1, 10);
    check(single.choose(lane) == 1 && single.choose(lane) == 1, "a ceiling of 1 always drafts 1");
}

// Round-cost learning: cycles of a + b K recover the cost b / a, idle gaps are skipped, a fit
// needs spread lengths, and a cheaper learned draft lengthens the choice the fixed model makes.
void learned_cost() {
    MtpDraftPolicy policy(kCeiling, 2);
    check(!policy.learned_draft_cost(1).has_value(), "no cost before any cycle");
    for (int i = 0; i < 40; ++i) { policy.observe_cycle(3, 1, 0.012 + 0.003 * 3); }
    check(!policy.learned_draft_cost(1).has_value(), "one length alone determines no cost");
    for (int i = 0; i < 40; ++i) {
        const std::uint32_t k = 1U + static_cast<std::uint32_t>(i % static_cast<int>(kCeiling));
        policy.observe_cycle(k, 1, 0.012 + 0.0008 * k);
        if (i % 10 == 0) { policy.observe_cycle(k, 1, 1.0); } // an idle gap
    }
    const auto cost = policy.learned_draft_cost(1);
    check(cost.has_value() && *cost > 0.03 && *cost < 0.2, "the fit learns a draft cost near 0.07 rounds");
    check(!policy.learned_draft_cost(2).has_value(), "each batch size learns its own cost");
    check(close_to(policy.round_seconds(4, 1), 1.0 + 4.0 * *cost), "a learned cost replaces the fixed model");

    // About 60 % first-draft acceptance, falling off after: the fixed 0.2 model stops at two
    // drafts, the learned ~0.07 drafts further.
    constexpr std::array<std::uint32_t, 10> kAccepted{0, 1, 0, 2, 0, 1, 3, 0, 5, 1};
    MtpLaneDraftState fixed_lane;
    for (int round = 0; round < 3; ++round) {
        for (const std::uint32_t accepted : kAccepted) { fixed_lane.acceptance.observe(5, accepted); }
    }
    MtpLaneDraftState learned_lane = fixed_lane;
    const MtpDraftPolicy modelled(kCeiling, 1);
    const std::uint32_t fixed_choice   = modelled.choose(fixed_lane);
    const std::uint32_t learned_choice = policy.choose(learned_lane);
    check(learned_choice > fixed_choice, "a cheaper learned draft chooses a longer length (fixed " +
                                             std::to_string(fixed_choice) + ", learned " +
                                             std::to_string(learned_choice) + ")");
}

} // namespace

int main() {
    try {
        batch_lengths();
        acceptance_estimate();
        longer_when_accepted_shorter_when_not();
        hysteresis();
        probes();
        batch_choice();
        save_and_restore();
        invalid_inputs();
        learned_cost();
    } catch (const std::exception& error) {
        std::cerr << "MTP draft policy test threw: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " MTP draft policy check(s) failed\n";
        return 1;
    }
    std::cout << "MTP draft policy tests passed\n";
    return 0;
}
