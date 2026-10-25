// llama.cpp's KL-divergence format and statistics (apps/certify/llama_kld.h) against an independent binary64
// evaluation of their definitions: the base row's log-probabilities, each position's KL over the base tokens
// above -16, the target's NLL and probability change, the top-token rule, and the summary's lines.

#include "llama_kld.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace llama = infernix::certify::llama;
using infernix::TokenId;

int failures = 0;

void require(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

std::vector<float> random_logits(std::mt19937& rng, std::size_t vocab, float spread) {
    std::normal_distribution<float> normal(0.0F, spread);
    std::vector<float> out(vocab);
    for (float& value : out) { value = normal(rng); }
    return out;
}

std::vector<double> log_softmax(const std::vector<float>& logits) {
    const double max = *std::max_element(logits.begin(), logits.end());
    double sum       = 0;
    for (const float value : logits) { sum += std::exp(static_cast<double>(value) - max); }
    std::vector<double> out(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) { out[i] = static_cast<double>(logits[i]) - max - std::log(sum); }
    return out;
}

std::vector<std::uint16_t> encoded(const std::vector<float>& logits) {
    std::vector<std::uint16_t> row(2 * ((logits.size() + 1) / 2) + 4);
    llama::encode_row(logits, row);
    return row;
}

void check_encoding() {
    std::mt19937 rng(1);
    for (const std::size_t vocab : {std::size_t{1001}, std::size_t{4096}}) {
        const std::vector<float> logits = random_logits(rng, vocab, 6.0F);
        const std::vector<std::uint16_t> row = encoded(logits);
        const std::vector<double> exact      = log_softmax(logits);
        const float max = *std::max_element(logits.begin(), logits.end());
        const float min = std::max(*std::min_element(logits.begin(), logits.end()), max - 16);
        const double scale = llama::row_scale(row);
        require(std::abs(scale - (static_cast<double>(max) - min) / 65535) < 1e-9, "scale spans the 16-nat window");
        double worst = 0;
        for (std::size_t i = 0; i < vocab; ++i) {
            const double decoded = llama::row_log_prob(row, static_cast<TokenId>(i));
            if (logits[i] > min) {
                worst = std::max(worst, std::abs(decoded - exact[i]));
            } else {
                require(row[4 + i] == 0 && decoded >= exact[i] - 1e-5, "a logit below the window reads as the floor");
            }
        }
        // Half a step of the uint16 grid, plus binary32 rounding of the window's ends.
        require(worst <= scale / 2 + 1e-5, "codes hold the log-probabilities to half a step");
        const auto top = static_cast<std::size_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        require(row[4 + top] == 65535, "the most probable token takes the top code");
        if (vocab % 2 == 1) { require(row.back() == 0, "an odd vocabulary's padding word is zero"); }
    }
}

void check_file(const std::filesystem::path& directory) {
    std::mt19937 rng(2);
    llama::BaseHeader header{.context = 8, .vocab = 5, .chunks = 2, .tokens = {1, 2, 3, 4, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 1, 2}};
    const std::filesystem::path path = directory / "base.kld";
    std::vector<std::vector<std::uint16_t>> written;
    {
        llama::BaseWriter writer(path, header);
        for (std::uint32_t r = 0; r < 2 * header.rows_per_chunk(); ++r) {
            const std::vector<float> logits = random_logits(rng, 5, 2.0F);
            written.push_back(encoded(logits));
            writer.write_row(logits);
        }
        writer.finish();
    }
    llama::BaseReader reader(path);
    require(reader.header().context == 8 && reader.header().vocab == 5 && reader.header().chunks == 2 &&
                reader.header().tokens == header.tokens,
            "the header round-trips");
    require(header.rows_per_chunk() == 3 && header.first_row() == 4 && header.row_words() == 10, "llama.cpp's row geometry");
    for (int c = 0; c < 2; ++c) {
        const std::vector<std::uint16_t> rows = reader.next_chunk();
        for (std::uint32_t k = 0; k < 3; ++k) {
            require(std::equal(written[c * 3 + k].begin(), written[c * 3 + k].end(), rows.begin() + k * 10), "rows round-trip");
        }
    }
    reader.finish();
    // A truncated file is refused when it is opened.
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 2);
    bool refused = false;
    try {
        llama::BaseReader truncated(path);
    } catch (const std::runtime_error&) { refused = true; }
    require(refused, "a truncated base file is refused");
}

// The definitions in binary64: the base's decoded log-probabilities, the evaluated run's log-softmax.
struct Oracle {
    double kld = 0, nll = 0, nll_base = 0, p_diff = 0;
};

Oracle oracle(const std::vector<std::uint16_t>& base, const std::vector<float>& logits, TokenId target) {
    const std::vector<double> q = log_softmax(logits);
    Oracle out;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const double p = llama::row_log_prob(base, static_cast<TokenId>(i));
        if (p > -16.0) { out.kld += std::exp(p) * (p - q[i]); }
    }
    out.nll      = -q[static_cast<std::size_t>(target)];
    out.nll_base = -static_cast<double>(llama::row_log_prob(base, target));
    out.p_diff   = std::exp(-out.nll) - std::exp(-out.nll_base);
    return out;
}

bool close(double a, double b, double relative, double absolute) { return std::abs(a - b) <= absolute + relative * std::abs(b); }

void check_positions() {
    std::mt19937 rng(3);
    const std::size_t vocab = 3000;
    for (int trial = 0; trial < 8; ++trial) {
        const std::vector<float> reference = random_logits(rng, vocab, 3.0F);
        std::vector<float> evaluated       = reference;
        std::normal_distribution<float> noise(0.0F, trial < 4 ? 0.02F : 0.5F);
        for (float& value : evaluated) { value += noise(rng); }
        const std::vector<std::uint16_t> base = encoded(reference);
        const TokenId target                  = static_cast<TokenId>(rng() % vocab);
        const llama::Position full            = llama::compare_logits(evaluated, base, target);
        const Oracle want                     = oracle(base, evaluated, target);
        // llama.cpp sums binary32 terms p (log p - (l - L)): about 1e-6 nats of absolute rounding.
        require(close(full.kld, want.kld, 1e-4, 5e-6), "KL matches the definition");
        require(close(full.nll, want.nll, 1e-6, 1e-5) && close(full.nll_base, want.nll_base, 1e-6, 1e-6), "NLLs match");
        require(close(full.p_diff, want.p_diff, 1e-5, 1e-7), "the probability change matches");
        const auto top_of = [](const std::vector<float>& v) {
            return static_cast<TokenId>(std::max_element(v.begin(), v.end()) - v.begin());
        };
        require(full.same_top == (top_of(evaluated) == llama::base_top(base, static_cast<std::int32_t>(vocab))),
                "same top compares the most probable tokens");

        // From observations of the KL sum's tokens, as the decode arm and llama-compare have them.
        const std::vector<TokenId> support = llama::kl_support(base, static_cast<std::int32_t>(vocab));
        require(std::is_sorted(support.begin(), support.end()), "the support is ascending");
        const std::vector<double> q = log_softmax(evaluated);
        std::vector<float> observed;
        for (const TokenId token : support) { observed.push_back(static_cast<float>(q[static_cast<std::size_t>(token)])); }
        const llama::Position from_observed = llama::compare_observed(
            base, support, observed, static_cast<float>(q[static_cast<std::size_t>(target)]), full.same_top, target);
        require(close(from_observed.kld, want.kld, 1e-4, 5e-6), "observed KL matches the definition");
        require(close(from_observed.p_diff, want.p_diff, 1e-5, 1e-7) && from_observed.same_top == full.same_top,
                "observed values match");
    }
    // Equal maxima: the lowest token id is the most probable, on both sides.
    std::vector<float> logits(16, 0.0F);
    logits[3] = logits[7] = 5.0F;
    const std::vector<std::uint16_t> base_tie = encoded(logits);
    require(llama::base_top(base_tie, 16) == 3, "the base breaks a tie by the lowest id");
    require(llama::compare_logits(logits, base_tie, 3).same_top, "equal maxima agree on the lowest id");
    std::vector<float> other = logits;
    other[3]                 = 4.0F; // the evaluated run's top is now 7
    require(!llama::compare_logits(other, base_tie, 3).same_top, "a different top token is a flip");
}

void check_summary() {
    llama::Summary few;
    for (int i = 0; i < 99; ++i) { few.add(llama::Position{.kld = 0.01, .p_diff = 0, .same_top = true, .nll = 1, .nll_base = 1}); }
    require(few.text().empty(), "fewer than 100 positions print no statistics, as in llama.cpp");

    llama::Summary summary;
    double kld_sum = 0;
    for (int i = 0; i < 1000; ++i) {
        const llama::Position position{.kld      = 1e-4 * i,
                                       .p_diff   = static_cast<float>((i % 7) - 3) * 1e-3F,
                                       .same_top = i % 50 != 0,
                                       .nll      = 2.0F + static_cast<float>(i % 5) * 0.1F,
                                       .nll_base = 2.0F + static_cast<float>(i % 3) * 0.1F};
        summary.add(position);
        kld_sum += position.kld;
    }
    const std::string text = summary.text();
    const auto has         = [&](const std::string& line) { return text.find(line) != std::string::npos; };
    char expected[128];
    std::snprintf(expected, sizeof expected, "Mean    KLD: %10.6lf \xC2\xB1", kld_sum / 1000);
    require(has(expected), "the mean KLD line");
    // Median of 1000 sorted values: the mean of the 500th and 501st.
    std::snprintf(expected, sizeof expected, "Median  KLD: %10.6f\n", static_cast<double>(0.5F * (0.0499F + 0.05F)));
    require(has(expected), "the median KLD line");
    require(has("Maximum KLD:   0.099900\n") && has("Minimum KLD:   0.000000\n"), "the KLD extremes");
    std::snprintf(expected, sizeof expected, "Same top p: %6.3lf \xC2\xB1 %5.3lf %%\n", 98.0, 100.0 * std::sqrt(0.98 * 0.02 / 999));
    require(has(expected), "the same top p line");
    require(has("====== KL divergence statistics ======\n") && has("RMS \xCE\x94p    :"), "the section headings");
    require(close(summary.to_json().at("clipped_mean_kld").get<double>(), kld_sum / 1000, 1e-6, 0), "clipping at 0.1 leaves these values");
}

} // namespace

int main() {
    try {
        check_encoding();
        check_file(std::filesystem::temp_directory_path());
        check_positions();
        check_summary();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "llama_kld: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
