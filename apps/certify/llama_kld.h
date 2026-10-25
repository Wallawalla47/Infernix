#pragma once

// llama.cpp's KL-divergence measurement (tools/perplexity/perplexity.cpp of build b10221), reproduced so
// that Infernix and llama.cpp are scored by one definition: the
// `_logits_` base file llama-perplexity --kl-divergence-base writes and reads, the per-position values its
// --kl-divergence mode computes against a base row, and its final summary, printed in its format.
//
// A base file holds "_logits_", the context n_ctx (u32), the vocabulary (i32), the chunk count (i32) and the
// chunks' tokens (i32, n_ctx per chunk); then, per chunk, one row for each position n_ctx/2 .. n_ctx-2 (the
// row of position r predicts token r + 1). A row is row_words() uint16 words: the first four hold two
// binary32 values, the scale s and the minimum log-probability m, then one code c per token, its
// log-probability being s * c + m (codes are 0 for logits more than 16 below the maximum), padded to an even
// count. All native-endian, as llama.cpp writes it.

#include "infernix/types.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace infernix::certify::llama {

struct BaseHeader {
    std::uint32_t context = 0;
    std::int32_t vocab    = 0;
    std::int32_t chunks   = 0;
    std::vector<TokenId> tokens; // chunks x context

    [[nodiscard]] std::uint32_t first_row() const noexcept { return context / 2; }
    [[nodiscard]] std::uint32_t rows_per_chunk() const noexcept { return context - 1 - context / 2; }
    [[nodiscard]] std::size_t row_words() const noexcept { return 2 * ((static_cast<std::size_t>(vocab) + 1) / 2) + 4; }
    [[nodiscard]] std::span<const TokenId> chunk(std::size_t index) const {
        return std::span<const TokenId>(tokens).subspan(index * context, context);
    }
};

// One row as llama.cpp's writer encodes a logits row (its log_softmax with a uint16 output).
void encode_row(std::span<const float> logits, std::span<std::uint16_t> row);

[[nodiscard]] float row_scale(std::span<const std::uint16_t> row) noexcept;
[[nodiscard]] float row_min_log_prob(std::span<const std::uint16_t> row) noexcept;
// The base log-probability of `token`, s * c + m in binary32.
[[nodiscard]] float row_log_prob(std::span<const std::uint16_t> row, TokenId token) noexcept;

class BaseWriter {
public:
    BaseWriter(const std::filesystem::path& path, const BaseHeader& header);
    // The rows of the next position, in order: rows_per_chunk() per chunk.
    void write_row(std::span<const float> logits);
    // Throws unless every row has been written.
    void finish();

private:
    std::filesystem::path path_;
    std::ofstream out_;
    std::vector<std::uint16_t> row_;
    std::uint64_t remaining_ = 0;
};

class BaseReader {
public:
    explicit BaseReader(const std::filesystem::path& path);
    [[nodiscard]] const BaseHeader& header() const noexcept { return header_; }
    // The next chunk's rows, rows_per_chunk() x row_words() words.
    [[nodiscard]] std::vector<std::uint16_t> next_chunk();
    // Throws unless every chunk has been read and nothing follows.
    void finish();

private:
    std::filesystem::path path_;
    std::ifstream in_;
    BaseHeader header_;
    std::int32_t read_chunks_ = 0;
};

// One position, as llama.cpp's --kl-divergence mode computes it against a base row.
struct Position {
    double kld       = 0.0;   // sum over tokens with base log-probability above -16 of p_base (log p_base - log q)
    float p_diff     = 0.0F;  // q(target) - p_base(target)
    bool same_top    = false; // the evaluated and base most probable tokens agree (each: lowest id among equals)
    float nll        = 0.0F;  // -log q(target)
    float nll_base   = 0.0F;  // -log p_base(target)
};

// From the evaluated run's whole logits row, in llama.cpp's arithmetic.
[[nodiscard]] Position compare_logits(std::span<const float> logits, std::span<const std::uint16_t> base, TokenId target);

// The tokens of the KL sum (base log-probability above -16), ascending.
[[nodiscard]] std::vector<TokenId> kl_support(std::span<const std::uint16_t> base, std::int32_t vocab);
// The base's most probable token (the lowest id among equal codes).
[[nodiscard]] TokenId base_top(std::span<const std::uint16_t> base, std::int32_t vocab);

// From the evaluated run's log-probabilities of kl_support's tokens (in its order) and of the target, and
// whether its most probable token is base_top: what a decode-path observation or another base file gives.
[[nodiscard]] Position compare_observed(std::span<const std::uint16_t> base, std::span<const TokenId> support,
                                        std::span<const float> log_probs, float target_log_prob, bool same_top,
                                        TokenId target);

// llama.cpp's accumulation and final summary ("====== Perplexity statistics ======" onwards).
class Summary {
public:
    void add(const Position& value);
    [[nodiscard]] std::uint64_t count() const noexcept { return count_; }
    // The summary text, line for line as llama-perplexity prints it.
    [[nodiscard]] std::string text() const;
    [[nodiscard]] nlohmann::json to_json() const;

private:
    double sum_nll_ = 0, sum_nll2_ = 0, sum_nll_base_ = 0, sum_nll_base2_ = 0, sum_nll_nll_base_ = 0;
    double sum_kld_ = 0, sum_kld2_ = 0, sum_p_diff_ = 0, sum_p_diff2_ = 0, sum_p_diff4_ = 0;
    std::uint64_t same_top_ = 0, count_ = 0;
    std::vector<float> kld_, p_diff_;
};

// One row per position for the paired statistics (tools/quality): chunk, position, and Position's fields.
class PositionWriter {
public:
    explicit PositionWriter(const std::filesystem::path& path);
    void add(std::uint32_t chunk, std::uint32_t position, const Position& value);
    [[nodiscard]] std::uint64_t rows() const noexcept { return rows_; }

private:
    std::ofstream out_;
    std::uint64_t rows_ = 0;
};

} // namespace infernix::certify::llama
