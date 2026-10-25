#include "llama_kld.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

// The arithmetic below follows llama.cpp's tools/perplexity/perplexity.cpp (build b10221, MIT licence):
// binary32 logits and log-probabilities, binary64 sums, expf, and its order of operations, so a row this code
// writes is the row llama-perplexity would write for the same logits and a summary is the one it prints.

namespace infernix::certify::llama {

namespace {

constexpr char kMagic[8] = {'_', 'l', 'o', 'g', 'i', 't', 's', '_'};
constexpr std::size_t kRowHeaderWords = 4;

// llama.cpp's nearest_int: round half to even through the binary32 magic constant (|value| < 2^22).
int nearest_int(float value) {
    const float shifted = value + 12582912.F;
    std::int32_t bits   = 0;
    std::memcpy(&bits, &shifted, sizeof bits);
    return (bits & 0x007fffff) - 0x00400000;
}

const std::uint16_t* codes(std::span<const std::uint16_t> row) { return row.data() + kRowHeaderWords; }

std::pair<double, double> mean_and_uncertainty(double sum, double sum2, std::uint64_t count) {
    if (count < 1) { return {0.0, 0.0}; }
    const double f = sum / static_cast<double>(count);
    double df      = sum2 / static_cast<double>(count) - f * f;
    df             = df > 0 && count > 10 ? std::sqrt(df / static_cast<double>(count - 1)) : 0.0;
    return {f, df};
}

double covariance(double suma, double sumb, double sumab, std::uint64_t count) {
    if (count < 10) { return 0.0; }
    const double n = static_cast<double>(count);
    double var     = sumab / n - (suma / n) * (sumb / n);
    var /= n - 1;
    return var;
}

float percentile(const std::vector<float>& values, float fraction) {
    if (fraction <= 0) { return values.front(); }
    if (fraction >= 1) { return values.back(); }
    float p                 = fraction * static_cast<float>(values.size() - 1);
    const std::size_t index = static_cast<std::size_t>(p);
    p -= static_cast<float>(index);
    return (1 - p) * values[index] + p * values[std::min(index + 1, values.size() - 1)];
}

float median(const std::vector<float>& values) {
    const std::size_t n = values.size();
    return n % 2 == 0 ? 0.5F * (values[n / 2] + values[n / 2 - 1]) : values[n / 2];
}

template <class... Args>
void line(std::string& out, const char* format, Args... args) {
    char buffer[256];
    const int written = std::snprintf(buffer, sizeof buffer, format, args...);
    if (written < 0 || static_cast<std::size_t>(written) >= sizeof buffer) { throw std::logic_error("summary line overflow"); }
    out += buffer;
}

void require_row(std::span<const std::uint16_t> row, std::size_t vocab) {
    if (row.size() != 2 * ((vocab + 1) / 2) + kRowHeaderWords) { throw std::invalid_argument("llama base row has the wrong size"); }
}

} // namespace

void encode_row(std::span<const float> logits, std::span<std::uint16_t> row) {
    if (logits.empty()) { throw std::invalid_argument("llama base row needs logits"); }
    require_row(row, logits.size());
    const std::size_t n_vocab = logits.size();
    float max_logit           = logits[0];
    float min_logit           = logits[0];
    for (std::size_t i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit      = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (std::size_t i = 0; i < n_vocab; ++i) { sum_exp += std::exp(logits[i] - max_logit); }
    const float log_sum_exp  = static_cast<float>(std::log(sum_exp));
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale        = (max_logit - min_logit) / 65535.F;
    std::memcpy(row.data(), &scale, sizeof scale);
    std::memcpy(row.data() + 2, &min_log_prob, sizeof min_log_prob);
    std::uint16_t* out = row.data() + kRowHeaderWords;
    std::fill(out, row.data() + row.size(), std::uint16_t{0});
    if (scale != 0) {
        const float inv_scale = 1 / scale;
        for (std::size_t i = 0; i < n_vocab; ++i) {
            out[i] = logits[i] > min_logit ? static_cast<std::uint16_t>(nearest_int(inv_scale * (logits[i] - min_logit))) : 0;
        }
    }
}

float row_scale(std::span<const std::uint16_t> row) noexcept {
    float value = 0;
    std::memcpy(&value, row.data(), sizeof value);
    return value;
}

float row_min_log_prob(std::span<const std::uint16_t> row) noexcept {
    float value = 0;
    std::memcpy(&value, row.data() + 2, sizeof value);
    return value;
}

float row_log_prob(std::span<const std::uint16_t> row, TokenId token) noexcept {
    return row_scale(row) * static_cast<float>(codes(row)[token]) + row_min_log_prob(row);
}

Position compare_logits(std::span<const float> logits, std::span<const std::uint16_t> base, TokenId target) {
    require_row(base, logits.size());
    if (target < 0 || static_cast<std::size_t>(target) >= logits.size()) { throw std::invalid_argument("target outside the vocabulary"); }
    const std::size_t n_vocab = logits.size();
    float max_logit           = logits[0];
    std::size_t imax          = 0;
    for (std::size_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax      = i;
        }
    }
    double sum_exp = 0.0;
    for (std::size_t i = 0; i < n_vocab; ++i) { sum_exp += std::exp(logits[i] - max_logit); }
    const float log_sum_exp = static_cast<float>(std::log(sum_exp));
    const float nll         = max_logit + log_sum_exp - logits[static_cast<std::size_t>(target)];
    const float nll_base    = -row_log_prob(base, target);
    max_logit += log_sum_exp;
    double sum               = 0;
    std::size_t imax_base    = 0;
    float p_log_base_max     = 0;
    for (std::size_t i = 0; i < n_vocab; ++i) {
        const float p_log_base = row_log_prob(base, static_cast<TokenId>(i));
        if (i == 0 || p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            imax_base      = i;
        }
        if (p_log_base > -16.F) {
            const float p_base = std::exp(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    const float p_base = std::exp(-nll_base);
    const float p      = std::exp(-nll);
    return Position{.kld = sum, .p_diff = p - p_base, .same_top = imax == imax_base, .nll = nll, .nll_base = nll_base};
}

std::vector<TokenId> kl_support(std::span<const std::uint16_t> base, std::int32_t vocab) {
    require_row(base, static_cast<std::size_t>(vocab));
    std::vector<TokenId> out;
    for (TokenId i = 0; i < vocab; ++i) {
        if (row_log_prob(base, i) > -16.F) { out.push_back(i); }
    }
    return out;
}

TokenId base_top(std::span<const std::uint16_t> base, std::int32_t vocab) {
    require_row(base, static_cast<std::size_t>(vocab));
    TokenId top    = 0;
    float best     = 0;
    for (TokenId i = 0; i < vocab; ++i) {
        const float value = row_log_prob(base, i);
        if (i == 0 || value > best) {
            best = value;
            top  = i;
        }
    }
    return top;
}

Position compare_observed(std::span<const std::uint16_t> base, std::span<const TokenId> support,
                          std::span<const float> log_probs, float target_log_prob, bool same_top, TokenId target) {
    if (support.size() != log_probs.size()) { throw std::invalid_argument("one log-probability per support token is required"); }
    const float nll      = -target_log_prob;
    const float nll_base = -row_log_prob(base, target);
    double sum           = 0;
    for (std::size_t k = 0; k < support.size(); ++k) {
        const float p_log_base = row_log_prob(base, support[k]);
        const float p_base     = std::exp(p_log_base);
        sum += p_base * (p_log_base - log_probs[k]);
    }
    const float p_base = std::exp(-nll_base);
    const float p      = std::exp(-nll);
    return Position{.kld = sum, .p_diff = p - p_base, .same_top = same_top, .nll = nll, .nll_base = nll_base};
}

void Summary::add(const Position& value) {
    sum_nll_ += value.nll;
    sum_nll2_ += value.nll * value.nll;
    sum_nll_base_ += value.nll_base;
    sum_nll_base2_ += value.nll_base * value.nll_base;
    sum_nll_nll_base_ += value.nll * value.nll_base;
    sum_kld_ += value.kld;
    sum_kld2_ += value.kld * value.kld;
    ++count_;
    same_top_ += value.same_top ? 1U : 0U;
    sum_p_diff_ += value.p_diff;
    const double p_diff2 = value.p_diff * value.p_diff;
    sum_p_diff2_ += p_diff2;
    sum_p_diff4_ += p_diff2 * p_diff2;
    kld_.push_back(static_cast<float>(value.kld));
    p_diff_.push_back(value.p_diff);
}

std::string Summary::text() const {
    // llama.cpp prints no statistics for fewer than 100 positions.
    if (count_ < 100) { return {}; }
    std::vector<float> kld = kld_, p_diff = p_diff_;
    std::sort(kld.begin(), kld.end());
    std::sort(p_diff.begin(), p_diff.end());
    std::string out;
    line(out, "====== Perplexity statistics ======\n");
    const auto log_ppl      = mean_and_uncertainty(sum_nll_, sum_nll2_, count_);
    const double ppl_val    = std::exp(log_ppl.first);
    const double ppl_unc    = ppl_val * log_ppl.second;
    line(out, "Mean PPL(Q)                   : %10.6lf \xC2\xB1 %10.6lf\n", ppl_val, ppl_unc);
    const auto log_ppl_base   = mean_and_uncertainty(sum_nll_base_, sum_nll_base2_, count_);
    const double ppl_base_val = std::exp(log_ppl_base.first);
    const double ppl_base_unc = ppl_base_val * log_ppl_base.second;
    line(out, "Mean PPL(base)                : %10.6lf \xC2\xB1 %10.6lf\n", ppl_base_val, ppl_base_unc);
    const double log_ppl_cov = covariance(sum_nll_, sum_nll_base_, sum_nll_nll_base_, count_);
    const double log_ppl_cor = log_ppl_cov / (log_ppl.second * log_ppl_base.second);
    line(out, "Cor(ln(PPL(Q)), ln(PPL(base))): %6.2lf%%\n", 100.0 * log_ppl_cor);
    const double log_ppl_ratio_val = log_ppl.first - log_ppl_base.first;
    const double log_ppl_ratio_unc = std::sqrt(log_ppl.second * log_ppl.second + log_ppl_base.second * log_ppl_base.second -
                                               2.0 * log_ppl_cov);
    line(out, "Mean ln(PPL(Q)/PPL(base))     : %10.6lf \xC2\xB1 %10.6lf\n", log_ppl_ratio_val, log_ppl_ratio_unc);
    const double ppl_ratio_val = std::exp(log_ppl_ratio_val);
    const double ppl_ratio_unc = ppl_ratio_val * log_ppl_ratio_unc;
    line(out, "Mean PPL(Q)/PPL(base)         : %10.6lf \xC2\xB1 %10.6lf\n", ppl_ratio_val, ppl_ratio_unc);
    const double ppl_cov      = ppl_val * ppl_base_val * log_ppl_cov;
    const double ppl_diff_val = ppl_val - ppl_base_val;
    const double ppl_diff_unc = std::sqrt(ppl_unc * ppl_unc + ppl_base_unc * ppl_base_unc - 2.0 * ppl_cov);
    line(out, "Mean PPL(Q)-PPL(base)         : %10.6lf \xC2\xB1 %10.6lf\n", ppl_diff_val, ppl_diff_unc);
    line(out, "\n");

    line(out, "====== KL divergence statistics ======\n");
    const auto kl_div = mean_and_uncertainty(sum_kld_, sum_kld2_, count_);
    line(out, "Mean    KLD: %10.6lf \xC2\xB1 %10.6lf\n", kl_div.first, kl_div.second);
    line(out, "Maximum KLD: %10.6f\n", static_cast<double>(kld.back()));
    line(out, "99.9%%   KLD: %10.6f\n", static_cast<double>(percentile(kld, 0.999F)));
    line(out, "99.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kld, 0.990F)));
    line(out, "95.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kld, 0.950F)));
    line(out, "90.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kld, 0.900F)));
    line(out, "Median  KLD: %10.6f\n", static_cast<double>(median(kld)));
    line(out, "10.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kld, 0.100F)));
    line(out, " 5.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kld, 0.050F)));
    line(out, " 1.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kld, 0.010F)));
    line(out, " 0.1%%   KLD: %10.6f\n", static_cast<double>(percentile(kld, 0.001F)));
    line(out, "Minimum KLD: %10.6f\n", static_cast<double>(kld.front()));
    line(out, "\n");

    line(out, "====== Token probability statistics ======\n");
    const auto mean_p_diff = mean_and_uncertainty(sum_p_diff_, sum_p_diff2_, count_);
    line(out, "Mean    \xCE\x94p: %6.3lf \xC2\xB1 %5.3lf %%\n", 100.0 * mean_p_diff.first, 100.0 * mean_p_diff.second);
    line(out, "Maximum \xCE\x94p: %6.3lf%%\n", 100.0 * p_diff.back());
    constexpr std::pair<const char*, float> upper[] = {{"99.9", 0.999F}, {"99.0", 0.990F}, {"95.0", 0.950F},
                                                      {"90.0", 0.900F}, {"75.0", 0.750F}};
    for (const auto& [label, fraction] : upper) {
        line(out, "%s%%   \xCE\x94p: %6.3lf%%\n", label, 100.0 * percentile(p_diff, fraction));
    }
    line(out, "Median  \xCE\x94p: %6.3lf%%\n", 100.0 * median(p_diff));
    constexpr std::pair<const char*, float> lower[] = {{"25.0", 0.250F}, {"10.0", 0.100F}, {" 5.0", 0.050F},
                                                      {" 1.0", 0.010F}, {" 0.1", 0.001F}};
    for (const auto& [label, fraction] : lower) {
        line(out, "%s%%   \xCE\x94p: %6.3lf%%\n", label, 100.0 * percentile(p_diff, fraction));
    }
    line(out, "Minimum \xCE\x94p: %6.3lf%%\n", 100.0 * p_diff.front());
    const auto p_diff_mse        = mean_and_uncertainty(sum_p_diff2_, sum_p_diff4_, count_);
    const double p_diff_rms_val  = std::sqrt(p_diff_mse.first);
    const double p_diff_rms_unc  = 0.5 / p_diff_rms_val * p_diff_mse.second;
    line(out, "RMS \xCE\x94p    : %6.3lf \xC2\xB1 %5.3lf %%\n", 100.0 * p_diff_rms_val, 100.0 * p_diff_rms_unc);
    const double same_top_p = static_cast<double>(same_top_) / static_cast<double>(count_);
    line(out, "Same top p: %6.3lf \xC2\xB1 %5.3lf %%\n", 100.0 * same_top_p,
         100.0 * std::sqrt(same_top_p * (1.0 - same_top_p) / static_cast<double>(count_ - 1)));
    return out;
}

nlohmann::json Summary::to_json() const {
    nlohmann::json out{{"positions", count_}};
    if (count_ == 0) { return out; }
    std::vector<float> kld = kld_, p_diff = p_diff_;
    std::sort(kld.begin(), kld.end());
    std::sort(p_diff.begin(), p_diff.end());
    const auto log_ppl      = mean_and_uncertainty(sum_nll_, sum_nll2_, count_);
    const auto log_ppl_base = mean_and_uncertainty(sum_nll_base_, sum_nll_base2_, count_);
    const auto kl_div       = mean_and_uncertainty(sum_kld_, sum_kld2_, count_);
    const auto p_diff_mse   = mean_and_uncertainty(sum_p_diff2_, sum_p_diff4_, count_);
    double clipped          = 0;
    for (const float value : kld_) { clipped += std::min(static_cast<double>(value), 0.1); }
    out["mean_ppl"]          = std::exp(log_ppl.first);
    out["mean_ppl_base"]     = std::exp(log_ppl_base.first);
    out["mean_ln_ppl_ratio"] = log_ppl.first - log_ppl_base.first;
    out["mean_kld"]          = kl_div.first;
    out["mean_kld_unc"]      = kl_div.second;
    out["median_kld"]        = median(kld);
    out["kld_percentiles"]   = {{"99.9", percentile(kld, 0.999F)}, {"99.0", percentile(kld, 0.990F)},
                                {"95.0", percentile(kld, 0.950F)}, {"90.0", percentile(kld, 0.900F)},
                                {"10.0", percentile(kld, 0.100F)}, {"5.0", percentile(kld, 0.050F)},
                                {"1.0", percentile(kld, 0.010F)},  {"0.1", percentile(kld, 0.001F)}};
    out["max_kld"]           = kld.back();
    out["min_kld"]           = kld.front();
    // The clipped mean: min(KL, 0.1 nats) averaged over every position.
    out["clipped_mean_kld"]  = clipped / static_cast<double>(count_);
    out["rms_p_diff"]        = std::sqrt(p_diff_mse.first);
    out["median_p_diff"]     = median(p_diff);
    out["same_top"]          = same_top_;
    out["same_top_p"]        = static_cast<double>(same_top_) / static_cast<double>(count_);
    return out;
}

BaseWriter::BaseWriter(const std::filesystem::path& path, const BaseHeader& header)
    : path_(path), out_(path, std::ios::binary | std::ios::trunc) {
    if (!out_) { throw std::runtime_error("cannot write " + path.string()); }
    if (header.context < 2 || header.vocab <= 0 || header.chunks <= 0 ||
        header.tokens.size() != static_cast<std::size_t>(header.chunks) * header.context) {
        throw std::invalid_argument("llama base header is inconsistent");
    }
    const std::int32_t context = static_cast<std::int32_t>(header.context);
    out_.write(kMagic, sizeof kMagic);
    out_.write(reinterpret_cast<const char*>(&context), sizeof context);
    out_.write(reinterpret_cast<const char*>(&header.vocab), sizeof header.vocab);
    out_.write(reinterpret_cast<const char*>(&header.chunks), sizeof header.chunks);
    out_.write(reinterpret_cast<const char*>(header.tokens.data()),
               static_cast<std::streamsize>(header.tokens.size() * sizeof(TokenId)));
    row_.resize(header.row_words());
    remaining_ = static_cast<std::uint64_t>(header.chunks) * header.rows_per_chunk();
}

void BaseWriter::write_row(std::span<const float> logits) {
    if (remaining_ == 0) { throw std::logic_error(path_.string() + ": every row is already written"); }
    encode_row(logits, row_);
    out_.write(reinterpret_cast<const char*>(row_.data()), static_cast<std::streamsize>(row_.size() * sizeof(std::uint16_t)));
    if (!out_) { throw std::runtime_error("cannot write " + path_.string()); }
    --remaining_;
}

void BaseWriter::finish() {
    if (remaining_ != 0) { throw std::logic_error(path_.string() + ": " + std::to_string(remaining_) + " rows are missing"); }
    out_.flush();
    if (!out_) { throw std::runtime_error("cannot write " + path_.string()); }
    out_.close();
}

BaseReader::BaseReader(const std::filesystem::path& path) : path_(path), in_(path, std::ios::binary) {
    if (!in_) { throw std::runtime_error("cannot read " + path.string()); }
    char magic[sizeof kMagic] = {};
    std::int32_t context      = 0;
    in_.read(magic, sizeof magic);
    in_.read(reinterpret_cast<char*>(&context), sizeof context);
    in_.read(reinterpret_cast<char*>(&header_.vocab), sizeof header_.vocab);
    in_.read(reinterpret_cast<char*>(&header_.chunks), sizeof header_.chunks);
    if (!in_ || std::memcmp(magic, kMagic, sizeof kMagic) != 0) {
        throw std::runtime_error(path.string() + " is not a llama.cpp log-probability (_logits_) file");
    }
    if (context < 2 || header_.vocab <= 0 || header_.chunks <= 0) { throw std::runtime_error(path.string() + ": invalid header"); }
    header_.context = static_cast<std::uint32_t>(context);
    header_.tokens.resize(static_cast<std::size_t>(header_.chunks) * header_.context);
    in_.read(reinterpret_cast<char*>(header_.tokens.data()), static_cast<std::streamsize>(header_.tokens.size() * sizeof(TokenId)));
    if (!in_) { throw std::runtime_error(path.string() + ": truncated tokens"); }
    const std::uintmax_t expected = sizeof kMagic + 3 * sizeof(std::int32_t) + header_.tokens.size() * sizeof(TokenId) +
                                    static_cast<std::uintmax_t>(header_.chunks) * header_.rows_per_chunk() * header_.row_words() *
                                        sizeof(std::uint16_t);
    if (std::filesystem::file_size(path) != expected) {
        throw std::runtime_error(path.string() + " holds " + std::to_string(std::filesystem::file_size(path)) + " bytes; its header needs " +
                                 std::to_string(expected));
    }
}

std::vector<std::uint16_t> BaseReader::next_chunk() {
    if (read_chunks_ == header_.chunks) { throw std::logic_error(path_.string() + ": every chunk is already read"); }
    std::vector<std::uint16_t> rows(static_cast<std::size_t>(header_.rows_per_chunk()) * header_.row_words());
    in_.read(reinterpret_cast<char*>(rows.data()), static_cast<std::streamsize>(rows.size() * sizeof(std::uint16_t)));
    if (!in_) { throw std::runtime_error(path_.string() + ": truncated chunk " + std::to_string(read_chunks_)); }
    ++read_chunks_;
    return rows;
}

void BaseReader::finish() {
    if (read_chunks_ != header_.chunks || in_.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(path_.string() + ": not every chunk was read");
    }
}

PositionWriter::PositionWriter(const std::filesystem::path& path) : out_(path) {
    if (!out_) { throw std::runtime_error("cannot write " + path.string()); }
    out_ << "chunk,position,kld,p_diff,same_top,nll,nll_base\n";
    out_.precision(9);
}

void PositionWriter::add(std::uint32_t chunk, std::uint32_t position, const Position& value) {
    out_ << chunk << ',' << position << ',' << value.kld << ',' << value.p_diff << ',' << (value.same_top ? 1 : 0) << ','
         << value.nll << ',' << value.nll_base << '\n';
    ++rows_;
}

} // namespace infernix::certify::llama
