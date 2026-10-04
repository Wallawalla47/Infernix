#include "models/qwen4_exp/frontend/ngram_hash.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::uint64_t kGamma  = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kPrime1 = 10007;

std::uint64_t splitmix64(std::uint64_t x) {
    x += kGamma;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

bool is_prime(std::uint64_t v) {
    if (v < 2) { return false; }
    if (v % 2 == 0) { return v == 2; }
    for (std::uint64_t d = 3; d * d <= v; d += 2) {
        if (v % d == 0) { return false; }
    }
    return true;
}

} // namespace

NgramHash::NgramHash(const NgramConfig& config, std::uint32_t layer) : config_(config) {
    if (config.ngram_size < 2 || config.heads_per_ngram == 0 || config.vocab_size == 0 ||
        config.ngram_vocab_size_base == 0 || config.make_ngram_vocab_size_divisible_by == 0) {
        throw std::invalid_argument("qwen4_exp n-gram config is incomplete");
    }
    const std::uint64_t max_long   = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    const std::uint64_t half_bound = std::max<std::uint64_t>(1, max_long / config.vocab_size / 2);
    const std::uint64_t base       = config.seed + kPrime1 * layer; // wraps like the 64-bit reference
    for (std::uint32_t i = 0; i < config.ngram_size; ++i) {
        multipliers_.push_back(2 * (splitmix64(base + kGamma * (i + 1)) % half_bound) + 1);
    }
    const std::uint64_t heads = static_cast<std::uint64_t>(config.ngram_size - 1) * config.heads_per_ngram;
    std::uint64_t prime = config.ngram_vocab_size_base - 1, total = 0;
    for (std::uint64_t i = 0; i < (static_cast<std::uint64_t>(layer) + 1) * heads; ++i) {
        do { ++prime; } while (!is_prime(prime));
        if (i >= static_cast<std::uint64_t>(layer) * heads) {
            sizes_.push_back(prime);
            offsets_.push_back(total);
            total += prime;
        }
    }
    const std::uint64_t d = config.make_ngram_vocab_size_divisible_by;
    padded_rows_          = (total + d - 1) / d * d;
    if (padded_rows_ > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("qwen4_exp n-gram table exceeds u32 row ids");
    }
}

void NgramHash::row_ids(std::span<const std::int32_t> history, std::size_t count, std::uint32_t* out) const {
    const std::size_t n_ctx = config_.ngram_size;
    if (history.size() < count + n_ctx - 1) { throw std::invalid_argument("n-gram history too short"); }
    std::uint64_t context[32];
    if (n_ctx > 32) { throw std::invalid_argument("ngram_size above 32"); }
    for (std::size_t p = history.size() - count; p < history.size(); ++p) {
        bool closed = false; // an EOS among history[p - s .. p - 1]
        for (std::size_t s = 0; s < n_ctx; ++s) {
            if (s > 0 && history[p - s] == config_.eos_token_id) { closed = true; }
            const std::int32_t token = closed ? config_.eos_token_id : history[p - s];
            if (token < 0 || static_cast<std::uint32_t>(token) >= config_.vocab_size) {
                throw std::invalid_argument("n-gram token outside the vocabulary");
            }
            context[s] = static_cast<std::uint64_t>(token);
        }
        std::uint64_t mixed = context[0] * multipliers_[0];
        for (std::size_t n = 2; n <= n_ctx; ++n) {
            mixed ^= context[n - 1] * multipliers_[n - 1];
            const std::size_t base = (n - 2) * config_.heads_per_ngram;
            for (std::size_t j = base; j < base + config_.heads_per_ngram; ++j) {
                *out++ = static_cast<std::uint32_t>(mixed % sizes_[j] + offsets_[j]);
            }
        }
    }
}

} // namespace ninfer::models::qwen4_exp
