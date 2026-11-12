#pragma once

// Repeated-passage detection over a reasoning stream, for the opt-in reasoning-loop guard
// (ThinkingControlOptions::loop). A model stuck in its thinking rewrites the same passages: a
// verification pass, a recap, a list of cases, over and over. Token penalties do not see it (each
// pass varies a little), so the measure works on words and long passages: the share of the most
// recent words that sit inside a 12-word passage seen at least three times in the recent history.
// That is Strata's measure (its #728 recovery), which also fixes the thresholds below; the
// literature's looping criterion (a 30-gram repeated 20 times) only fires far later.
// A loop can also be written as a single long word, which no word measure can see: Strata's #1753
// reads a periodic tail as full coverage, and periodic_tail below does the same. A third failure
// repeats nothing at all: thinking that degenerates into one run-on of hyphen-joined words, every
// word new and no sentence ending (Strata #5218, Qwen3.8-Flash-Next on long agent requests: the
// longest chain 0-56 characters in normal thinking against 203-1,569 in degenerate runs);
// longest_hyphen_chain below measures it. A counting loop rewrites one line with only a number
// changing ("checking item 7 of 40", "checking item 8 of 40": TensorSharp #338, 3 of 53 Aider turns on
// the Qwen3.8-27B NVFP4): when the lines are short, every 12-word passage holds a different number and
// no passage repeats, so the word measure also reads the words with each run of ASCII digits folded
// to one digit. Folded, a numbered template with a few words varying (a table, a list of cases) also
// repeats, a few times per passage; the loop's line repeats once per line, so on that reading a
// passage counts from kLoopFoldedRepeats occurrences and the guard fires at kLoopFoldedCoverage.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::models::qwen3_5 {

inline constexpr std::uint32_t kLoopCheckTokens   = 512;   // reasoning tokens between two checks
inline constexpr std::size_t kLoopPassageWords    = 12;    // words in one passage
inline constexpr std::uint32_t kLoopRepeats       = 3;     // occurrences that make a passage repeated
inline constexpr std::size_t kLoopWindowWords     = 2000;  // the recent words whose coverage is measured
inline constexpr std::size_t kLoopHistoryWords    = 30000; // how far back passages are counted
inline constexpr double kLoopCoverage             = 0.25;  // coverage at which the guard fires
// The digit-folded reading: a passage is repeated from 32 occurrences (TensorSharp closes after 32
// such lines in a row), and the guard fires once those cover half of the last kLoopWindowWords; a
// counting loop that keeps going soon covers nearly all of it.
inline constexpr std::uint32_t kLoopFoldedRepeats = 32;
inline constexpr double kLoopFoldedCoverage       = 0.5;
inline constexpr std::size_t kLoopTailChars       = 2048;  // reasoning characters read at the end
inline constexpr std::size_t kLoopTailPeriod      = 64;    // a period at most this long repeats
inline constexpr std::size_t kLoopRunOnChars      = 200;   // a hyphen chain this long is a run-on

// One word, hashed as written and with every run of ASCII digits folded to one '0' ("item17" reads
// as "item0"); the two are equal for a word without digits.
struct LoopWord {
    std::uint64_t exact  = 0;
    std::uint64_t folded = 0;

    friend bool operator==(const LoopWord&, const LoopWord&) = default;
};

// Which of a LoopWord's hashes a passage measure reads.
enum class LoopWordForm { Exact, DigitsFolded };

// Appends the words of UTF-8 `bytes` to `words`. A word is a run of letters, digits and '_' (ASCII
// letters lower-cased); every other non-space character is a word of its own, so an unseparated run
// of emoji or punctuation contributes one word per character, as it does under Strata's
// `\w+|[^\w\s]`. A character is classified once all of its bytes have arrived, so `partial` holds the
// bytes of a word a token boundary cut, completed by the next call.
void append_words(std::string_view bytes, std::string& partial, std::vector<LoopWord>& words);

// The share of the last kLoopWindowWords of `words` covered by kLoopPassageWords-word passages that
// occur at least `repeats` times in `words` (the caller passes the last kLoopHistoryWords at most),
// reading each word in `form`. Zero while fewer than kLoopWindowWords words exist.
[[nodiscard]] double repeated_passage_coverage(std::span<const LoopWord> words, LoopWordForm form,
                                               std::uint32_t repeats);

// The guard's word measure: the Exact coverage at kLoopRepeats, or the DigitsFolded coverage at
// kLoopFoldedRepeats when that reaches kLoopFoldedCoverage; the guard fires when the result reaches
// kLoopCoverage.
[[nodiscard]] double loop_passage_coverage(std::span<const LoopWord> words);

// Appends `bytes` to the bounded tail of the reasoning that periodic_tail reads. It holds at least
// kLoopTailChars characters whenever that many have been emitted.
void append_tail(std::string_view bytes, std::string& tail);

// True when the last kLoopTailChars characters of `text` are one period of at most kLoopTailPeriod
// characters repeated (Strata's #1753). The comparison is byte-exact where Strata folds case, which no loop
// this catches depends on. False while fewer than kLoopTailChars characters have been emitted.
[[nodiscard]] bool periodic_tail(std::string_view text);

// The length in bytes of the longest chain of ASCII letter runs joined by single hyphens in `text`
// (Strata's `[A-Za-z]+(?:-[A-Za-z]+)+`: at least one hyphen; a digit, '_', a space, any other byte or a
// second hyphen in a row ends the chain), 0 when there is none. Letters only, so identifiers, UUIDs and
// base64 strings, whose digits break chains, are not run-ons. The guard reads it over the reasoning's
// tail (append_tail: at least 4 x kLoopTailChars bytes, every byte since the previous check).
[[nodiscard]] std::size_t longest_hyphen_chain(std::string_view text);

} // namespace infernix::models::qwen3_5
