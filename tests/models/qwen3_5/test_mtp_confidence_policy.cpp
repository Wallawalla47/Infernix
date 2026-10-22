#include "models/qwen3_5/program/speculative/mtp_confidence_policy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Confidence-gated MTP extents (MtpConfidencePolicy): the rate-maximizing prefix against an
// exhaustive evaluation, cheap columns verifying every draft and costly ones only the confident
// prefix, acceptance learned per probability bucket, the round-cost fit (and its gap filter), the
// fixed-threshold mode and invalid inputs.
namespace {

using infernix::models::qwen3_5::detail::MtpConfidencePolicy;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// The rule's choice evaluated exhaustively: every prefix's expected tokens over its cost.
std::uint32_t exhaustive(const MtpConfidencePolicy& policy, std::span<const float> q, std::uint32_t ceiling,
                         std::uint32_t batch) {
    const double c     = policy.column_cost(batch);
    std::uint32_t best = 0;
    double best_rate   = 1.0;
    const auto n       = std::min<std::uint32_t>(ceiling, static_cast<std::uint32_t>(q.size()));
    for (std::uint32_t e = 1; e <= n; ++e) {
        double tokens = 1.0;
        for (std::uint32_t j = 1; j <= e; ++j) {
            double reach = 1.0;
            for (std::uint32_t i = 0; i < j; ++i) { reach *= policy.acceptance(q[i]); }
            tokens += reach;
        }
        const double rate = tokens / (1.0 + c * e);
        if (rate > best_rate) {
            best      = e;
            best_rate = rate;
        }
    }
    return best;
}

void rule_matches_exhaustive() {
    const std::vector<std::vector<float>> drafts{
        {0.99F, 0.98F, 0.95F}, {0.4F, 0.99F, 0.99F}, {0.999F, 0.6F, 0.999F}, {0.85F, 0.85F, 0.85F}, {}, {0.2F}};
    for (const double cost : {0.0, 0.05, 0.2, 0.7, 1.5}) {
        const MtpConfidencePolicy policy(cost);
        for (const auto& q : drafts) {
            for (std::uint32_t ceiling = 0; ceiling <= 3; ++ceiling) {
                check(policy.extent(q, ceiling, 1) == exhaustive(policy, q, ceiling, 1),
                      "extent equals the exhaustive rate maximum at cost " + std::to_string(cost));
            }
        }
    }
}

void cheap_and_costly_columns() {
    const std::array<float, 3> confident{0.999F, 0.995F, 0.99F}, mixed{0.999F, 0.5F, 0.999F};
    const std::array<float, 3> doubtful{0.3F, 0.999F, 0.999F};
    const MtpConfidencePolicy cheap(0.05), costly(0.7);
    check(cheap.extent(mixed, 3, 1) == 3, "cheap columns verify past a doubtful draft");
    check(costly.extent(confident, 3, 1) == 3, "costly columns still verify a confident chain");
    check(costly.extent(mixed, 3, 1) == 1, "costly columns stop at the first doubtful draft");
    check(costly.extent(doubtful, 3, 1) == 0, "costly columns skip a chain whose first draft is doubtful");
    check(costly.extent(confident, 2, 1) == 2, "the ceiling bounds the extent");
}

void learns_acceptance() {
    MtpConfidencePolicy policy(0.7);
    const double prior = policy.acceptance(0.92F);
    check(std::abs(prior - 0.925) < 1e-12, "a bucket's prior is its midpoint");
    // Drafts of probability 0.92 that are always rejected: their acceptance falls, others are kept.
    const std::array<float, 1> q{0.92F};
    for (int i = 0; i < 64; ++i) { policy.observe(q, 1, 0); }
    check(policy.acceptance(0.92F) < 0.1, "rejected drafts lower their bucket's acceptance");
    check(std::abs(policy.acceptance(0.999F) - 0.9985) < 1e-12, "other buckets keep their prior");
    check(policy.extent(q, 1, 1) == 0, "a learned-rejected bucket is not verified");
    // An untested draft (after a rejected one) is not counted.
    MtpConfidencePolicy second(0.7);
    const std::array<float, 2> pair{0.92F, 0.6F};
    for (int i = 0; i < 64; ++i) { second.observe(pair, 2, 0); }
    check(std::abs(second.acceptance(0.6F) - 0.6) < 1e-12, "a draft after a rejection is not tested");
    for (int i = 0; i < 64; ++i) { second.observe(pair, 2, 2); }
    check(second.acceptance(0.6F) > 0.9, "accepted drafts raise their bucket's acceptance");
}

void learned_cost() {
    MtpConfidencePolicy policy(0.7);
    check(policy.column_cost(1) == 0.7, "the prior column cost before any round");
    // Rounds of t = 10 ms + 1 ms per verified column: c = 0.1.
    for (int i = 0; i < 200; ++i) {
        const auto columns = static_cast<std::uint32_t>(i % 4);
        policy.observe_round(1, columns, 0.010 + 0.001 * columns);
    }
    check(std::abs(policy.column_cost(1) - 0.1) < 1e-6, "the fit learns the column cost");
    // A gap (an idle wait between rounds) is not a round cost.
    policy.observe_round(1, 0, 1.0);
    check(std::abs(policy.column_cost(1) - 0.1) < 1e-6, "a gap is skipped");
    // Batches of two rows: the cost of one row's columns counts both rows' rounds.
    for (int i = 0; i < 200; ++i) {
        const auto columns = static_cast<std::uint32_t>(i % 5);
        policy.observe_round(2, columns, 0.020 + 0.001 * columns);
    }
    check(std::abs(policy.column_cost(2) - 0.1) < 1e-6, "a batch's column cost is per row of the round");
    check(std::abs(policy.column_cost(1) - 0.1) < 1e-6, "batch sizes fit apart");
}

void invalid_inputs() {
    bool threw = false;
    try {
        (void)MtpConfidencePolicy(-0.1);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a negative column cost is refused");
}

} // namespace

int main() {
    try {
        rule_matches_exhaustive();
        cheap_and_costly_columns();
        learns_acceptance();
        learned_cost();
        invalid_inputs();
    } catch (const std::exception& error) {
        std::cerr << "MTP confidence policy test threw: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "MTP confidence policy: all checks passed\n";
    return 0;
}
