#pragma once

// Next-token distribution comparison for ninfer-perplexity: a reference run saves each scored
// position's most probable tokens, and a later run of the same corpus windows measures its
// KL divergence from them.

#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace ninfer::perplexity {

// Most probable tokens a reference keeps per scored position.
inline constexpr std::uint32_t kReferenceTopTokens = 32;

// The scoring a reference describes; a comparison must repeat it exactly.
struct ReferenceHeader {
    std::string corpus_id;
    std::string mode;
    std::uint32_t context = 0;
    std::uint32_t stride  = 0;
    std::uint32_t top_k   = kReferenceTopTokens;
};

// One scored window: its stream, target range and input fingerprint, then per scored position
// the target token's log-probability and the top_k most probable tokens, most probable first.
struct ReferenceWindow {
    std::uint32_t stream_index = 0;
    std::uint64_t target_begin = 0;
    std::uint64_t target_end   = 0;
    std::uint64_t input_hash   = 0;
    std::vector<float> target_logprobs;
    std::vector<TokenId> top_ids;
    std::vector<float> top_logprobs;
};

// FNV-1a over the window's input token ids.
[[nodiscard]] std::uint64_t token_fingerprint(std::span<const TokenId> tokens) noexcept;

// Native-endian binary reference file: a header, the windows in scoring order, and an end marker.
class ReferenceWriter {
public:
    ReferenceWriter(const std::filesystem::path& path, const ReferenceHeader& header);
    void write(const ReferenceWindow& window);
    void finish();

private:
    std::filesystem::path path_;
    std::ofstream out_;
    std::uint32_t top_k_ = 0;
};

class ReferenceReader {
public:
    explicit ReferenceReader(const std::filesystem::path& path);
    [[nodiscard]] const ReferenceHeader& header() const noexcept { return header_; }
    // The next window; throws when the reference has no more windows.
    [[nodiscard]] ReferenceWindow next();
    // Throws unless every window has been read.
    void finish();

private:
    std::filesystem::path path_;
    std::ifstream in_;
    ReferenceHeader header_;
};

// One position's comparison with its reference.
struct PositionDivergence {
    // KL(reference || evaluated) between the two distributions coarsened to the reference's top
    // tokens plus one bucket for every other token: a lower bound of the full KL divergence.
    double kl = 0.0;
    // The reference's top-token probability mass (how much of it the bound resolves).
    double reference_top_mass = 0.0;
    bool top1_match           = false;
    // Evaluated minus reference negative log-likelihood of the target token.
    double delta_nll = 0.0;
};

// reference_logprobs and evaluated_logprobs hold the reference's top tokens' log-probabilities
// under each distribution, in the reference's order.
[[nodiscard]] PositionDivergence position_divergence(std::span<const float> reference_logprobs,
                                                     std::span<const float> evaluated_logprobs,
                                                     bool top1_match, float reference_target,
                                                     float evaluated_target);

// Divergence statistics by context length: the number of tokens before the predicted one, in
// buckets [0,1K), [1K,2K), [2K,4K), ... doubling.
class DivergenceAggregate {
public:
    void add(std::uint64_t context_tokens, const PositionDivergence& value);
    void add(const DivergenceAggregate& other);
    [[nodiscard]] std::uint64_t positions() const noexcept { return overall_.count; }
    [[nodiscard]] double mean_kl() const;
    [[nodiscard]] double top1_agreement() const;
    // Overall statistics with KL quantiles, and per-bucket statistics.
    [[nodiscard]] nlohmann::json to_json() const;
    // Mean KL by bucket, "label mean" pairs for a log line.
    [[nodiscard]] std::string bucket_summary() const;

private:
    struct Sums {
        std::uint64_t count        = 0;
        std::uint64_t top1_matches = 0;
        double kl                  = 0.0;
        double kl_squares          = 0.0;
        double delta_nll           = 0.0;
        double reference_top_mass  = 0.0;
        void add(const PositionDivergence& value);
        void add(const Sums& other);
        [[nodiscard]] nlohmann::json to_json() const;
    };

    Sums overall_;
    std::vector<Sums> buckets_;
    std::vector<double> values_;
};

} // namespace ninfer::perplexity
