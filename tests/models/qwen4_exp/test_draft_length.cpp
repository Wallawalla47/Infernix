// Qwen4Exp's DFlash2 draft-length policy (program/draft_length.h): the calibrated length against a
// direct evaluation of expected tokens per round time, and the calibration's updates.

#include "models/qwen4_exp/program/draft_length.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using infernix::models::qwen4_exp::detail::calibrated_draft_length;
using infernix::models::qwen4_exp::detail::SelectorCalibration;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

// The k in [0, most] with the highest expected tokens per round time, by exhaustive evaluation.
int brute_force(const SelectorCalibration& c, const std::vector<float>& p, int most, double w, double f) {
    int best = 0;
    double best_score = 1.0 / (1.0 + f);
    for (int k = 1; k <= most; ++k) {
        double tokens = 1.0;
        for (int j = 1; j <= k; ++j) {
            double reach = 1.0;
            for (int i = 0; i < j; ++i) { reach *= c.accept[SelectorCalibration::bin(p[static_cast<std::size_t>(i)])]; }
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

} // namespace

int main() {
    try {
        SelectorCalibration c;
        c.reset();
        require(std::abs(c.accept[0] - 0.05) < 1e-12 && std::abs(c.accept[9] - 0.95) < 1e-12, "decile centres");
        require(SelectorCalibration::bin(0.0F) == 0 && SelectorCalibration::bin(0.999F) == 9 &&
                    SelectorCalibration::bin(1.0F) == 9 && SelectorCalibration::bin(-1.0F) == 0,
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
        }
        // Updates: an accepted draft raises its decile, the first rejected lowers its own, later
        // drafts are untouched.
        SelectorCalibration u;
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
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp draft length policy checks passed\n");
    return 0;
}
