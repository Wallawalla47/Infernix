// The block drafters' calibrated draft length (runtime/speculative/draft_length.h): the calibrated
// length against a direct evaluation of expected tokens per round time, the expected tokens of a
// prefix, the calibration's updates, and a two-row round's width (pair_draft_width) against the
// best of every pair of row lengths under the padded round's cost.

#include "runtime/speculative/draft_length.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using infernix::runtime::speculative::calibrated_draft_length;
using infernix::runtime::speculative::DraftCalibration;
using infernix::runtime::speculative::expected_draft_tokens;
using infernix::runtime::speculative::pair_draft_width;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

// The k in [0, most] with the highest expected tokens per round time, by exhaustive evaluation.
int brute_force(const DraftCalibration& c, const std::vector<float>& p, int most, double w, double f) {
    int best = 0;
    double best_score = 1.0 / (1.0 + f);
    for (int k = 1; k <= most; ++k) {
        double tokens = 1.0;
        for (int j = 1; j <= k; ++j) {
            double reach = 1.0;
            for (int i = 0; i < j; ++i) { reach *= c.accept[DraftCalibration::bin(p[static_cast<std::size_t>(i)])]; }
            tokens += reach;
        }
        const double score = tokens / (1.0 + f + w * k);
        if (score > best_score + 1e-12) {
            best_score = score;
            best       = k;
        }
    }
    return best;
}

// A two-row round's best expected tokens per round time over every pair of row lengths (ka, kb),
// ka, kb in [least_x, n_x], the round costing 1 + f + w * max(ka, kb) (the shorter row padded).
double best_pair_score(const std::vector<double>& ta, int least_a, int na, const std::vector<double>& tb, int least_b,
                       int nb, double w, double f) {
    double best = -1.0;
    for (int ka = least_a; ka <= na; ++ka) {
        for (int kb = least_b; kb <= nb; ++kb) {
            const double score = (ta[static_cast<std::size_t>(ka)] + tb[static_cast<std::size_t>(kb)]) /
                                 (1.0 + f + w * std::max(ka, kb));
            best = std::max(best, score);
        }
    }
    return best;
}

// Row tokens for pair_draft_width over k in [0, most]: a row with n drafts verifies min(k, n).
std::vector<double> row_tokens(const DraftCalibration& c, const std::vector<float>& p, int n, int most) {
    std::vector<double> out;
    const std::vector<float> drafts(p.begin(), p.begin() + n);
    for (int k = 0; k <= most; ++k) { out.push_back(expected_draft_tokens(c, drafts, k)); }
    return out;
}

// A copy row of n drafts at acceptance a: 1 + a + ... + a^min(k, n).
std::vector<double> copy_tokens(double a, int n, int most) {
    std::vector<double> out;
    double tokens = 1.0, reach = 1.0;
    for (int k = 0; k <= most; ++k) {
        if (k > 0 && k <= n) {
            reach *= a;
            tokens += reach;
        }
        out.push_back(tokens);
    }
    return out;
}

} // namespace

int main() {
    try {
        DraftCalibration c;
        c.reset();
        require(std::abs(c.accept[0] - 0.05) < 1e-12 && std::abs(c.accept[9] - 0.95) < 1e-12, "decile centres");
        require(DraftCalibration::bin(0.0F) == 0 && DraftCalibration::bin(0.999F) == 9 &&
                    DraftCalibration::bin(1.0F) == 9 && DraftCalibration::bin(-1.0F) == 0,
                "probabilities map to deciles 0..9");
        // Confident drafts are all verified; unlikely ones are not; a low draft ends the run.
        const std::vector<float> sure(7, 0.97F), doubtful(7, 0.05F);
        require(calibrated_draft_length(c, sure, 7, 0.31, 0.1) == 7, "seven confident drafts are verified");
        require(calibrated_draft_length(c, doubtful, 7, 0.31, 0.1) == 0, "unlikely drafts are not verified");
        const std::vector<float> broken{0.97F, 0.96F, 0.05F, 0.97F, 0.97F, 0.97F, 0.97F};
        require(calibrated_draft_length(c, broken, 7, 0.31, 0.1) == 2, "a doubtful third draft ends the run");
        require(calibrated_draft_length(c, sure, 3, 0.31, 0.1) == 3, "the length never exceeds `most`");
        // Against the exhaustive choice over a grid of probabilities and costs.
        std::uint32_t state = 7;
        for (int trial = 0; trial < 2000; ++trial) {
            std::vector<float> p(7);
            for (float& v : p) {
                state = state * 1664525U + 1013904223U;
                v     = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
            }
            const double w = 0.1 + 0.05 * (trial % 9), f = 0.05 * (trial % 4);
            const int most = 1 + trial % 7;
            require(calibrated_draft_length(c, p, most, w, f) == brute_force(c, p, most, w, f),
                    "the policy's choice equals the exhaustive one (trial " + std::to_string(trial) + ")");
            // The expected tokens of every prefix, against the direct sum of products.
            for (int k = 0; k <= 9; ++k) {
                double direct = 1.0;
                for (int j = 1; j <= std::min(k, 7); ++j) {
                    double reach = 1.0;
                    for (int i = 0; i < j; ++i) { reach *= c.accept[DraftCalibration::bin(p[static_cast<std::size_t>(i)])]; }
                    direct += reach;
                }
                require(std::abs(expected_draft_tokens(c, p, k) - direct) < 1e-12,
                        "expected tokens of a prefix (trial " + std::to_string(trial) + ", k " + std::to_string(k) + ")");
            }
        }
        // Updates: an accepted draft raises its decile, the first rejected lowers its own, later
        // drafts are untouched.
        DraftCalibration u;
        u.reset();
        const std::array<float, 4> probabilities{0.92F, 0.71F, 0.55F, 0.33F};
        u.observe(probabilities, 4, 1, 0.5);
        require(std::abs(u.accept[9] - (0.5 * 0.95 + 0.5)) < 1e-12, "an accepted draft raises its decile");
        require(std::abs(u.accept[7] - 0.5 * 0.75) < 1e-12, "the first rejected draft lowers its decile");
        require(u.accept[5] == 0.55 && u.accept[3] == 0.35, "drafts after the first rejection are untouched");
        u.reset();
        u.observe(probabilities, 2, 2, 0.5);
        require(u.accept[9] > 0.95 && u.accept[7] > 0.75 && u.accept[5] == 0.55,
                "an all-accepted round raises every verified draft's decile and nothing past it");

        // Two-row rounds (dflash2-pairs): two confident rows verify both drafts, two doubtful rows none,
        // and a copy row's whole proposal bounds the width from below.
        const auto sure_row = row_tokens(c, sure, 2, 2), doubtful_row = row_tokens(c, doubtful, 2, 2);
        require(pair_draft_width(sure_row, sure_row, 0, 2, 0.47, 0.1) == 2, "two confident rows verify two drafts");
        require(pair_draft_width(doubtful_row, doubtful_row, 0, 2, 0.47, 0.1) == 0, "two doubtful rows verify none");
        require(pair_draft_width(doubtful_row, copy_tokens(0.9, 2, 2), 2, 2, 0.47, 0.1) == 2,
                "a two-token copy row is verified whole");
        require(pair_draft_width(sure_row, sure_row, 0, 1, 0.47, 0.1) == 1, "the width never exceeds `most`");
        // Against every pair of row lengths: verifying each row's drafts up to the shared width loses
        // nothing, since the padded round costs the same whichever row is shorter.
        for (int trial = 0; trial < 4000; ++trial) {
            std::vector<float> pa(2), pb(2);
            for (float& v : pa) {
                state = state * 1664525U + 1013904223U;
                v     = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
            }
            for (float& v : pb) {
                state = state * 1664525U + 1013904223U;
                v     = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
            }
            const int na = trial % 3, nb = (trial / 3) % 3, most = std::max(na, nb);
            const double w = 0.2 + 0.05 * (trial % 9), f = 0.05 * (trial % 4);
            // Every fourth trial row b is a copy row (fixed extent nb, verified whole).
            const bool copy = trial % 4 == 3;
            const auto ta = row_tokens(c, pa, na, most);
            const auto tb = copy ? copy_tokens(0.5 + 0.1 * (trial % 5), nb, most) : row_tokens(c, pb, nb, most);
            const int least = copy ? nb : 0;
            const int k     = pair_draft_width(ta, tb, least, most, w, f);
            require(k >= least && k <= most, "the width lies in [least, most] (trial " + std::to_string(trial) + ")");
            const double score = (ta[static_cast<std::size_t>(k)] + tb[static_cast<std::size_t>(k)]) / (1.0 + f + w * k);
            const double best  = best_pair_score(ta, 0, na, tb, copy ? nb : 0, nb, w, f);
            require(std::abs(score - best) < 1e-12,
                    "the width's score equals the best pair of row lengths (trial " + std::to_string(trial) + ")");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("draft length policy checks passed\n");
    return 0;
}
