#pragma once

// PLE n-gram row ids of Qwen3.8-Flash-Next (docs/maintainer/qwen3_8-flash-next-design.md §12):
// exact integer hashing of the n-grams ending at each position into one PLE layer's table.
// Specification and fixture generator: tools/flash_next/ngram.py.

#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp {

struct NgramConfig {
    std::uint32_t vocab_size                         = 0;
    std::int32_t eos_token_id                        = 0;
    std::uint32_t ngram_size                         = 0; // >= 2
    std::uint32_t heads_per_ngram                    = 0;
    std::uint64_t ngram_vocab_size_base              = 0;
    std::uint64_t make_ngram_vocab_size_divisible_by = 1;
    std::uint64_t seed                               = 1234;
};

class NgramHash {
public:
    NgramHash(const NgramConfig& config, std::uint32_t ple_layer_index);

    [[nodiscard]] std::uint32_t heads() const { return static_cast<std::uint32_t>(sizes_.size()); }
    [[nodiscard]] std::uint64_t table_rows() const { return padded_rows_; }
    [[nodiscard]] std::span<const std::uint64_t> multipliers() const { return multipliers_; }
    [[nodiscard]] std::span<const std::uint64_t> sizes() const { return sizes_; }
    [[nodiscard]] std::span<const std::uint64_t> offsets() const { return offsets_; }

    // Rows [count][heads()] for the last `count` positions of `history`, which holds at least
    // ngram_size - 1 tokens before them (EOS-padded at sequence start). An EOS in a window replaces
    // every earlier context token by EOS.
    void row_ids(std::span<const std::int32_t> history, std::size_t count, std::uint32_t* out) const;

private:
    NgramConfig config_;
    std::vector<std::uint64_t> multipliers_; // [ngram_size]
    std::vector<std::uint64_t> sizes_;       // [heads], primes
    std::vector<std::uint64_t> offsets_;     // [heads]
    std::uint64_t padded_rows_ = 0;
};

} // namespace infernix::models::qwen4_exp
