// The reasoning-loop measure: words split across token boundaries join; a reasoning that keeps
// rewriting the same verification passes reaches the threshold; varied work, and a short text, stay
// below it; a passage repeated twice is not a loop; a counting loop (a line changing only in a
// number) fires on the digit-folded reading, a numbered template with varying words does not.
#include "models/qwen3_5/frontend/reasoning_loop.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace infernix::models::qwen3_5;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL " << label << '\n';
        ++failures;
    }
}

std::vector<LoopWord> words_of(const std::string& text, std::size_t piece = 3) {
    std::vector<LoopWord> words;
    std::string partial;
    for (std::size_t at = 0; at < text.size(); at += piece) {
        append_words(std::string_view(text).substr(at, piece), partial, words);
    }
    append_words(" ", partial, words);
    return words;
}

// Deterministic varied prose: a new combination of words in every sentence.
std::string varied(int sentences) {
    static const char* subjects[] = {"the parser", "this branch", "the cache", "our bound", "the loop", "a tile",
                                     "the queue", "that edge", "the schema", "each lane", "the budget"};
    static const char* verbs[]    = {"handles", "skips", "rounds", "checks", "splits", "merges", "drops",
                                     "keeps", "orders", "limits", "scales", "tracks", "maps"};
    static const char* objects[]  = {"empty input", "the last token", "two columns", "a ragged width",
                                     "every retry", "the null case", "large values", "unicode names",
                                     "negative offsets", "the first page", "stale entries", "short reads"};
    std::string text;
    unsigned state = 12345;
    for (int s = 0; s < sentences; ++s) {
        state = state * 1103515245U + 12345U;
        text += std::string(subjects[(state >> 8) % 11]) + " " + verbs[(state >> 12) % 13] + " " +
                objects[(state >> 16) % 12] + " when step " + std::to_string(s) + " runs with value " +
                std::to_string((state >> 4) % 997) + ". ";
    }
    return text;
}

} // namespace

int main() {
    // Splitting: case folding, punctuation as words, a word cut by a token boundary.
    {
        std::vector<LoopWord> a, b;
        std::string pa, pb;
        append_words("Hello, World", pa, a);
        append_words("!", pa, a);
        append_words("hel", pb, b);
        append_words("lo , world!", pb, b);
        expect(a == b && a.size() == 4, "split and fold");
    }
    // A non-word character is a word of its own even without separators, whether it arrives whole or
    // one byte per call.
    {
        std::vector<LoopWord> a, b;
        std::string pa, pb;
        append_words("\xe2\x9a\xa1\xe2\x9a\xa1" "ab", pa, a);
        append_words(" ", pa, a);
        append_words("\xe2", pb, b);
        append_words("\x9a", pb, b);
        append_words("\xa1", pb, b);
        append_words("\xe2\x9a\xa1" "ab", pb, b);
        append_words(" ", pb, b);
        expect(a == b && a.size() == 3, "unseparated symbol splits per character");
    }
    // Non-ASCII letters fold as `str.lower()` does: "ПРИВЕТ Ünïcode" and "привет ünïCODE" are the
    // same words, also when a token boundary cuts a character.
    {
        expect(words_of("\xd0\x9f\xd0\xa0\xd0\x98\xd0\x92\xd0\x95\xd0\xa2 \xc3\x9cn\xc3\xaf" "code", 1) ==
                   words_of("\xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xc3\xbcn\xc3\xaf" "CODE"),
               "non-ascii case fold");
    }
    // A run of CJK letters is one word, as `\w+` reads it.
    {
        std::vector<LoopWord> words;
        std::string partial;
        append_words("\xe4\xb8\xad\xe6\x96\x87", partial, words);
        append_words(" ", partial, words);
        expect(words.size() == 1, "cjk is one word");
    }
    // ASCII punctuation keeps one word per character.
    {
        std::vector<LoopWord> words;
        std::string partial;
        append_words("?!?!?!", partial, words);
        append_words(" ", partial, words);
        expect(words.size() == 6, "punctuation per character");
    }
    // The folded hash reads every run of ASCII digits as one digit, also across token boundaries; a
    // word without digits hashes alike both ways.
    {
        const auto folded = [](const std::vector<LoopWord>& words) {
            std::vector<std::uint64_t> hashes;
            for (const LoopWord& word : words) { hashes.push_back(word.folded); }
            return hashes;
        };
        const std::vector<LoopWord> a = words_of("Item 17 of 40, v12x3");
        const std::vector<LoopWord> b = words_of("item 3 of 5, V0X999", 1);
        expect(a.size() == 6 && a != b && folded(a) == folded(b), "digit runs fold");
        const std::vector<LoopWord> plain = words_of("no digits here");
        expect(plain.size() == 3 && plain[0].exact == plain[0].folded && plain[2].exact == plain[2].folded,
               "a word without digits hashes alike");
        expect(words_of("item17", 1) == words_of("item17") && words_of("item17") != words_of("item1 7"),
               "a digit run cut by a token boundary stays one run");
    }
    // Below a full window nothing is measured.
    expect(loop_passage_coverage(words_of(varied(20))) == 0.0, "short text");
    // Varied work stays low.
    const double novel = loop_passage_coverage(words_of(varied(400)));
    expect(novel < 0.05, "varied reasoning coverage " + std::to_string(novel));
    // The same verification pass over and over, after some real work.
    const std::string pass =
        "Wait, let me double-check the edge case where the input list is empty. If the list is empty, "
        "the function returns zero, which matches the expected output. So that case is fine. ";
    std::string looping = varied(200);
    for (int i = 0; i < 40; ++i) { looping += pass; }
    const double loop = loop_passage_coverage(words_of(looping));
    expect(loop >= kLoopCoverage, "looping coverage " + std::to_string(loop));
    // The same loop written as one unseparated run of symbols: every symbol is a word of its own, so
    // the passage repeats and the measure sees it (a byte-level split makes it a single word).
    {
        std::string symbols;
        for (int i = 0; i < 3000; ++i) { symbols += "\xe2\x9a\xa1"; }
        const double symbol_loop = loop_passage_coverage(words_of(varied(150) + symbols, 1));
        expect(symbol_loop >= kLoopCoverage,
               "unseparated symbol loop coverage " + std::to_string(symbol_loop));
    }
    // A passage seen twice is not repeated.
    std::string twice = varied(300) + pass + varied(10) + pass + varied(150);
    const double recap = loop_passage_coverage(words_of(twice));
    expect(recap < kLoopCoverage, "recap coverage " + std::to_string(recap));
    // A counting loop after some real work: every 12-word passage of "Checking item N of 40..." holds a
    // different number, so no exact passage repeats; folded, the line repeats once per line and the
    // guard fires on that reading.
    double counting = 0.0;
    {
        std::string text = varied(200);
        for (int i = 1; i <= 300; ++i) { text += "Checking item " + std::to_string(i) + " of 40...\n"; }
        const std::vector<LoopWord> words = words_of(text);
        const double exact = repeated_passage_coverage(words, LoopWordForm::Exact, kLoopRepeats);
        counting           = loop_passage_coverage(words);
        expect(exact < kLoopCoverage, "a counting loop repeats no exact passage " + std::to_string(exact));
        expect(counting >= kLoopFoldedCoverage, "counting loop coverage " + std::to_string(counting));
    }
    // A long numbered template whose words vary (varied(): subject, verb and object drawn per
    // sentence) repeats each folded passage a few times: past kLoopRepeats, never kLoopFoldedRepeats.
    {
        const std::vector<LoopWord> words = words_of(varied(800));
        const double three = repeated_passage_coverage(words, LoopWordForm::DigitsFolded, kLoopRepeats);
        const double guard = loop_passage_coverage(words);
        expect(three >= kLoopFoldedCoverage && guard < kLoopCoverage,
               "a numbered template is no counting loop (folded at 3 repeats " + std::to_string(three) +
                   ", guard " + std::to_string(guard) + ")");
    }
    // A numbered checklist of distinct items is no loop either.
    {
        static const char* verbs[]      = {"verify", "check",   "confirm", "test",    "measure",
                                           "inspect", "review", "trace",   "compare", "rerun"};
        static const char* objects[]    = {"the tokenizer",    "the scheduler",  "the KV pool",     "the sampler",
                                           "the grammar",      "the prefix cache", "the HTTP layer",
                                           "the stop strings", "the tool parser", "the console log",
                                           "the request log",  "the graph cache"};
        static const char* qualifiers[] = {"on empty input",       "under two lanes",     "after a preemption",
                                           "with a stale prefix",  "at full context",     "with long tool calls",
                                           "on unicode names",     "when the queue is full",
                                           "during a model switch", "after a restart"};
        std::string checklist = varied(100);
        unsigned state        = 99;
        for (int i = 1; i <= 200; ++i) {
            state = state * 1103515245U + 12345U;
            checklist += std::to_string(i) + ". " + verbs[(state >> 8) % 10] + " " + objects[(state >> 12) % 12] +
                         " " + qualifiers[(state >> 16) % 10] + ".\n";
        }
        const double coverage = loop_passage_coverage(words_of(checklist));
        expect(coverage < kLoopCoverage, "numbered checklist coverage " + std::to_string(coverage));
    }
    // A loop written as one long word: a periodic tail reads as full coverage.
    {
        std::string digits;
        for (int i = 0; i < 200; ++i) { digits += "58747693387"; } // 2200 characters, period 11
        expect(periodic_tail(digits), "periodic tail");
        expect(!periodic_tail("58747693387"), "tail below the window");
        expect(!periodic_tail(varied(200)), "varied tail is not periodic");
    }
    // The period counts characters, not bytes: 30 CJK characters (90 bytes) repeat, 65 do not.
    {
        const auto cjk = [](int n) {
            std::string unit;
            for (int i = 0; i < n; ++i) {
                const int c = 0x4e00 + i; // three UTF-8 bytes each
                unit += static_cast<char>(0xe0 | (c >> 12));
                unit += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
                unit += static_cast<char>(0x80 | (c & 0x3f));
            }
            std::string text;
            while (text.size() < 3U * (kLoopTailChars + 2U * n)) { text += unit; }
            return text;
        };
        expect(periodic_tail(cjk(30)), "30-character period of 3-byte characters");
        expect(periodic_tail(cjk(64)), "64-character period of 3-byte characters");
        expect(!periodic_tail(cjk(65)), "65-character period is beyond the limit");
    }
    // A run-on of hyphen-joined words repeats nothing (Strata #5218): the chain measure follows
    // `[A-Za-z]+(?:-[A-Za-z]+)+`.
    {
        expect(longest_hyphen_chain("") == 0 && longest_hyphen_chain("no hyphens here") == 0, "no chain");
        expect(longest_hyphen_chain("a well-known self-check") == 10, "longest of two chains");
        expect(longest_hyphen_chain("x-") == 0 && longest_hyphen_chain("-x") == 0, "a lone hyphen joins nothing");
        expect(longest_hyphen_chain("ab--cd") == 0, "a double hyphen ends the chain");
        expect(longest_hyphen_chain("state-of-the-art.") == 16, "punctuation ends the chain");
        expect(longest_hyphen_chain("utf-8 x86-64") == 0, "digits break chains");
        expect(longest_hyphen_chain("123e4567-e89b-12d3-a456-426614174000 snake_case-name") == 9,
               "a UUID is no chain; an underscore ends one");
        expect(longest_hyphen_chain(varied(300)) == 0, "varied prose has no chain");
        static const char* parts[] = {"so",   "then", "we",    "need", "to",    "check", "maybe", "again",
                                      "the",  "list", "value", "is",   "fine",  "but",   "wait",  "hmm",
                                      "okay", "and",  "next",  "edge", "case",  "right", "now",   "sure"};
        std::string run_on = varied(40) + " ";
        unsigned state     = 11;
        for (int i = 0; i < 120; ++i) {
            state = state * 1103515245U + 12345U;
            run_on += (i == 0 ? "" : "-") + std::string(parts[(state >> 8) % 24]);
        }
        const std::size_t chain = longest_hyphen_chain(run_on);
        expect(chain >= kLoopRunOnChars, "run-on chain " + std::to_string(chain));
        expect(loop_passage_coverage(words_of(varied(200) + run_on)) < kLoopCoverage,
               "a run-on is not repeated passages");
        expect(!periodic_tail(run_on), "a run-on is not periodic");
    }
    // The tail stays bounded and keeps its characters however the bytes arrive.
    {
        std::string tail;
        const std::string symbol = "\xe2\x9a\xa1";
        for (int i = 0; i < 8000; ++i) { append_tail(symbol, tail); }
        expect(periodic_tail(tail), "bounded tail is periodic");
        expect(tail.size() <= 8U * kLoopTailChars, "tail stays bounded");
    }
    if (failures != 0) { return 1; }
    std::cout << "OK reasoning loop (varied " << novel << ", looping " << loop << ", recap " << recap
              << ", counting " << counting << ")\n";
    return 0;
}
