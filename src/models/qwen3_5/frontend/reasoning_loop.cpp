#include "models/qwen3_5/frontend/reasoning_loop.h"
#include "text/unicode.h"

#include <algorithm>
#include <unordered_map>

namespace infernix::models::qwen3_5 {
namespace {

std::uint64_t fnv1a(std::string_view text) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const char c : text) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

bool word_byte(unsigned char c) {
    return c >= 0x80 || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

// Python's `\s`, which Strata's split drops. unicode::is_whitespace covers the Unicode space
// categories; U+001C-U+001F and U+0085 are whitespace to Python and not to it.
bool space_codepoint(std::int32_t value) {
    return (value >= 0x1c && value <= 0x1f) || value == 0x85 ||
           text::unicode_internal::is_whitespace(value);
}

// A non-ASCII byte only shows what it is once its whole character has arrived: one token can hold one
// byte of a multi-byte character. `complete` is false for that trailing part.
struct Codepoint {
    std::int32_t value = 0;
    std::size_t length = 1;
    bool complete      = true;
};

// Lenient UTF-8 over raw token bytes: a byte that cannot start a sequence, or one whose sequence runs
// past the end of the buffer, decodes as a length-1 code point of its own.
Codepoint codepoint_at(std::string_view text, std::size_t at) {
    const auto first = static_cast<unsigned char>(text[at]);
    if (first < 0x80U) { return {first, 1, true}; }
    std::size_t length = 0;
    std::int32_t value = 0;
    if ((first & 0xe0U) == 0xc0U) {
        length = 2;
        value  = first & 0x1fU;
    } else if ((first & 0xf0U) == 0xe0U) {
        length = 3;
        value  = first & 0x0fU;
    } else if ((first & 0xf8U) == 0xf0U) {
        length = 4;
        value  = first & 0x07U;
    } else {
        return {first, 1, true};
    }
    if (at + length > text.size()) { return {first, 1, false}; }
    for (std::size_t i = 1; i < length; ++i) {
        const auto next = static_cast<unsigned char>(text[at + i]);
        if ((next & 0xc0U) != 0x80U) { return {first, 1, true}; }
        value = (value << 6) | static_cast<std::int32_t>(next & 0x3fU);
    }
    return {value, length, true};
}

// `\w` in Strata's split: a letter, a number, or '_'.
bool word_codepoint(std::int32_t value) {
    return value == '_' || text::unicode_internal::is_letter(value) ||
           text::unicode_internal::is_number(value);
}

// The bytes accumulated at a word boundary are one word to the byte rule above; Strata's split can
// make several of them. A run of word characters is one word, every other non-space character is a
// word of its own, and whitespace separates.
// Words are lowercased as Strata's `str.lower()` does: ASCII on arrival (append_words), every other
// complete character by its simple lowercase mapping here.
void flush_word(std::string& partial, std::vector<std::uint64_t>& words) {
    const std::string_view text = partial;
    std::string word;
    const auto emit_word = [&] {
        if (!word.empty()) {
            words.push_back(fnv1a(word));
            word.clear();
        }
    };
    for (std::size_t at = 0; at < text.size();) {
        const Codepoint point = codepoint_at(text, at);
        if (!point.complete) { // cut by the last token: the bytes stand as a word of their own
            emit_word();
            words.push_back(fnv1a(text.substr(at)));
            partial.clear();
            return;
        }
        if (word_codepoint(point.value)) {
            if (point.length == 1) {
                word.push_back(text[at]);
            } else {
                word += text::unicode_internal::codepoint_to_utf8(
                    text::unicode_internal::simple_lowercase(point.value));
            }
            at += point.length;
            continue;
        }
        emit_word();
        if (!space_codepoint(point.value)) {
            words.push_back(fnv1a(text.substr(at, point.length)));
        }
        at += point.length;
    }
    emit_word();
    partial.clear();
}

struct TailSpan {
    std::size_t begin      = 0;
    std::size_t characters = 0;
};

// Where the last `limit` characters of `text` begin, walking back over continuations so that a
// multi-byte character is never cut in half.
TailSpan tail_span(std::string_view text, std::size_t limit) {
    TailSpan span;
    span.begin = text.size();
    while (span.begin > 0 && span.characters < limit) {
        std::size_t at = span.begin - 1U;
        while (at > 0 && (static_cast<unsigned char>(text[at]) & 0xc0U) == 0x80U) { --at; }
        span.begin = at;
        ++span.characters;
    }
    return span;
}

} // namespace

void append_words(std::string_view bytes, std::string& partial, std::vector<std::uint64_t>& words) {
    for (const char raw : bytes) {
        const auto c = static_cast<unsigned char>(raw);
        if (word_byte(c)) {
            partial.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : raw);
            continue;
        }
        if (!partial.empty()) { flush_word(partial, words); }
        if (!space_codepoint(c)) { words.push_back(fnv1a(std::string_view(&raw, 1))); }
    }
}

double repeated_passage_coverage(std::span<const std::uint64_t> words) {
    const std::size_t n = words.size();
    if (n < kLoopWindowWords || n < kLoopPassageWords) { return 0.0; }
    // Polynomial hash of every passage, rolled along the words.
    constexpr std::uint64_t kBase = 0x9E3779B97F4A7C15ULL;
    std::uint64_t top = 1; // kBase^(kLoopPassageWords - 1)
    for (std::size_t i = 1; i < kLoopPassageWords; ++i) { top *= kBase; }
    const std::size_t passages = n - kLoopPassageWords + 1;
    std::vector<std::uint64_t> hashes(passages);
    std::uint64_t hash = 0;
    for (std::size_t i = 0; i < kLoopPassageWords; ++i) { hash = hash * kBase + words[i]; }
    hashes[0] = hash;
    for (std::size_t i = 1; i < passages; ++i) {
        hash      = (hash - words[i - 1] * top) * kBase + words[i + kLoopPassageWords - 1];
        hashes[i] = hash;
    }
    std::unordered_map<std::uint64_t, std::uint32_t> counts;
    counts.reserve(passages);
    for (const std::uint64_t h : hashes) { ++counts[h]; }

    // Mark the window's words that lie inside a repeated passage (a difference array over starts).
    const std::size_t window_start = n - kLoopWindowWords;
    std::vector<int> delta(kLoopWindowWords + 1, 0);
    const std::size_t first = window_start >= kLoopPassageWords - 1 ? window_start - (kLoopPassageWords - 1) : 0;
    for (std::size_t i = first; i < passages; ++i) {
        if (counts[hashes[i]] < kLoopRepeats) { continue; }
        const std::size_t begin = std::max(i, window_start) - window_start;
        const std::size_t end   = std::min(i + kLoopPassageWords, n) - window_start;
        ++delta[begin];
        --delta[end];
    }
    std::size_t covered = 0;
    int depth           = 0;
    for (std::size_t p = 0; p < kLoopWindowWords; ++p) {
        depth += delta[p];
        covered += depth > 0 ? 1U : 0U;
    }
    return static_cast<double>(covered) / static_cast<double>(kLoopWindowWords);
}

void append_tail(std::string_view bytes, std::string& tail) {
    tail.append(bytes);
    // A character is at most four bytes, so the last 4 * kLoopTailChars bytes always hold the last
    // kLoopTailChars characters; the buffer is cut only once it is twice that.
    constexpr std::size_t kKeep = 4U * kLoopTailChars;
    if (tail.size() > 2U * kKeep) { tail.erase(0, tail.size() - kKeep); }
}

bool periodic_tail(std::string_view text) {
    const TailSpan span = tail_span(text, kLoopTailChars);
    if (span.characters < kLoopTailChars) { return false; }
    const std::string_view tail = text.substr(span.begin);
    // Strata's period counts characters: the bytes of the tail's first 1..kLoopTailPeriod characters.
    // The tail starts on a character, so a byte shift by a whole number of characters that repeats
    // the tail is exactly a character period.
    std::size_t period = 0;
    for (std::size_t characters = 1; characters <= kLoopTailPeriod; ++characters) {
        do { ++period; } while (period < tail.size() && (static_cast<unsigned char>(tail[period]) & 0xc0U) == 0x80U);
        if (period >= tail.size()) { return false; }
        if (tail.substr(period) == tail.substr(0, tail.size() - period)) { return true; }
    }
    return false;
}

} // namespace infernix::models::qwen3_5
