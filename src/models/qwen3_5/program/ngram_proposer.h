#pragma once

#include "models/qwen3_5/ngram.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

namespace ninfer::models::qwen3_5::detail {

// A bounded, position-indexed proposal corpus. It never owns target state.
// Absolute positions make overwritten ring entries rejectable before a read.
class NgramProposer {
public:
    using Token = std::int32_t;

    using Match = NgramMatch;

    [[nodiscard]] static bool maximal(const Match& match, std::size_t history_size,
                                      std::uint32_t maximum) {
        return match.matched == history_size && match.tokens.size() == maximum;
    }

    explicit NgramProposer(std::size_t token_capacity = 1U << 20,
                           std::size_t bucket_count   = 1U << 19)
        : tokens_(power_of_two(token_capacity) && token_capacity >= 64 ? token_capacity : 0, -1),
          buckets_(power_of_two(bucket_count) ? bucket_count : 0),
          token_mask_(token_capacity - 1U) {
        if (tokens_.empty() || buckets_.empty()) {
            throw std::invalid_argument("invalid ngram corpus capacity");
        }
    }

    void boundary() { append(-1); }

    void set_boundaries(std::vector<Token> tokens) {
        boundaries_ = std::move(tokens);
        std::sort(boundaries_.begin(), boundaries_.end());
    }

    void append(Token token) {
        if (!store(token)) { return; }
        for (const auto n : widths_) {
            if (insertable(n)) { insert(hash_at(end_ - n, n), n); }
        }
    }

    // Indexes a whole span between boundaries. Each window is hashed once from the input, a few
    // tokens before its insert, so its bucket line is already on its way; an inserted window
    // never spans a boundary, so the input holds exactly the tokens the ring then holds, and the
    // index is the one token-by-token append() builds.
    void ingest(std::span<const Token> tokens) {
        boundary();
        constexpr std::size_t kLookahead = 8;
        std::array<std::array<std::uint64_t, widths_.size()>, kLookahead> ahead{};
        const auto hash_windows = [&](std::size_t end) {
            auto& hashes = ahead[end % kLookahead];
            for (std::size_t w = 0; w < widths_.size(); ++w) {
                if (end < widths_[w]) { continue; }
                hashes[w] = hash(tokens.subspan(end - widths_[w], widths_[w]), widths_[w]);
                prefetch(&buckets_[hashes[w] & (buckets_.size() - 1)]);
            }
        };
        for (std::size_t end = 1; end <= std::min(kLookahead, tokens.size()); ++end) {
            hash_windows(end);
        }
        for (std::size_t end = 1; end <= tokens.size(); ++end) {
            const std::array<std::uint64_t, widths_.size()> hashes = ahead[end % kLookahead];
            if (end + kLookahead <= tokens.size()) { hash_windows(end + kLookahead); }
            if (!store(tokens[end - 1])) { continue; }
            for (std::size_t w = 0; w < widths_.size(); ++w) {
                if (insertable(widths_[w])) { insert(hashes[w], widths_[w]); }
            }
        }
        boundary();
    }

    [[nodiscard]] Match propose_for_round(std::span<const Token> history, std::uint32_t maximum,
                                          std::uint32_t neural_drafts,
                                          std::uint32_t minimum_match = 12) const {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        return finish_round(propose(history, maximum + 2U, minimum_match), maximum, neural_drafts);
    }

    [[nodiscard]] static Match finish_round(Match match, std::uint32_t maximum,
                                            std::uint32_t neural_drafts) {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        const auto available = static_cast<std::uint32_t>(match.tokens.size());
        // The target emits a bonus after the drafts. Leave one source token beyond it
        // for neural completion; source endings may need different surrounding text.
        const auto drafts = std::min(maximum, available > 2U ? available - 2U : 0U);
        if (available < maximum + 2U && drafts <= neural_drafts) { return {}; }
        match.tokens.resize(drafts);
        return match;
    }

    [[nodiscard]] Match propose(std::span<const Token> history, std::uint32_t maximum,
                                std::uint32_t minimum_match = 12) const {
        Match best;
        if (maximum == 0) { return best; }
        for (const auto n : widths_) {
            if (history.size() < n) { continue; }
            const auto tail = history.last(n);
            if (std::find_if(tail.begin(), tail.end(), [](Token t) { return t < 0; }) !=
                tail.end()) {
                continue;
            }
            const auto& bucket = buckets_[hash(tail, n) & (buckets_.size() - 1)];
            for (const auto entry : bucket) {
                if (entry.width != n || entry.end < n || entry.end - n < live_begin() ||
                    entry.end >= end_) {
                    continue;
                }
                bool equal = true;
                for (std::uint32_t j = 0; j < n; ++j) {
                    if (at(entry.end - n + j) != tail[j]) {
                        equal = false;
                        break;
                    }
                }
                if (!equal) { continue; }
                std::uint32_t matched = n;
                while (matched < history.size() && entry.end - matched > live_begin()) {
                    const auto token = at(entry.end - matched - 1);
                    if (token < 0 || token != history[history.size() - matched - 1]) { break; }
                    ++matched;
                }
                if (matched < minimum_match || matched < best.matched) { continue; }
                std::vector<Token> draft;
                for (std::uint64_t p = entry.end; p < end_ && draft.size() < maximum; ++p) {
                    const auto token = at(p);
                    if (token < 0) { break; }
                    draft.push_back(token);
                }
                if (!draft.empty() &&
                    (matched > best.matched || draft.size() > best.tokens.size())) {
                    best = {std::move(draft), matched};
                }
            }
        }
        return best;
    }

private:
    struct Entry {
        std::uint64_t end   = 0;
        std::uint32_t width = 0;
    };

    static constexpr std::array<std::uint32_t, 3> widths_{16, 8, 4};
    std::vector<Token> tokens_;
    std::vector<Token> boundaries_;
    std::vector<std::array<Entry, 4>> buckets_;
    std::uint64_t token_mask_    = 0;
    std::uint64_t end_           = 0;
    std::uint64_t segment_start_ = 0;

    [[nodiscard]] static bool power_of_two(std::size_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    }

    static void prefetch(const void* address) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(address, 1);
#else
        (void)address;
#endif
    }

    // Writes the next ring token; false when it is a boundary, which starts a new segment.
    bool store(Token token) {
        if (std::binary_search(boundaries_.begin(), boundaries_.end(), token)) { token = -1; }
        tokens_[end_ & token_mask_] = token;
        ++end_;
        if (token < 0) {
            segment_start_ = end_;
            return false;
        }
        return true;
    }

    // Whether the width-n window ending at the ring end lies in one live segment.
    [[nodiscard]] bool insertable(std::uint32_t n) const {
        return end_ - segment_start_ >= n && end_ - live_begin() >= n;
    }

    void insert(std::uint64_t window_hash, std::uint32_t n) {
        auto& bucket = buckets_[window_hash & (buckets_.size() - 1)];
        for (std::size_t i = bucket.size() - 1; i > 0; --i) { bucket[i] = bucket[i - 1]; }
        bucket[0] = {end_, n};
    }

    [[nodiscard]] std::uint64_t live_begin() const {
        return end_ > tokens_.size() ? end_ - tokens_.size() : 0;
    }

    [[nodiscard]] Token at(std::uint64_t p) const { return tokens_[p & token_mask_]; }

    static std::uint64_t mix(std::uint64_t h, Token token) {
        return (h ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
    }

    static std::uint64_t finalize_hash(std::uint64_t value) {
        // Buckets mask low bits; avalanche patterned IDs before selecting a bucket.
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    static std::uint64_t hash(std::span<const Token> tokens, std::uint32_t n) {
        std::uint64_t h = 1469598103934665603ULL ^ n;
        for (auto token : tokens) { h = mix(h, token); }
        return finalize_hash(h);
    }

    [[nodiscard]] std::uint64_t hash_at(std::uint64_t p, std::uint32_t n) const {
        std::uint64_t h = 1469598103934665603ULL ^ n;
        for (std::uint32_t j = 0; j < n; ++j) { h = mix(h, at(p + j)); }
        return finalize_hash(h);
    }
};

} // namespace ninfer::models::qwen3_5::detail
