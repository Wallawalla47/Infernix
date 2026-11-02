// The prefix-miss record (serve/prefix_miss.h, candidates CF-08): which large re-prefills are
// explained, as a divergence from an earlier context or as an eviction of one, and which are not.

#include "serve/prefix_miss.h"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using infernix::TokenId;
using infernix::serve::PrefixMiss;
using infernix::serve::PrefixMissSettlement;
using infernix::serve::PrefixMissTracker;
using infernix::serve::settle_prefix_miss;

namespace {

int failures = 0;

void check(bool pass, const char* message) {
    if (!pass) {
        std::cerr << "FAIL " << message << '\n';
        ++failures;
    }
}

std::vector<TokenId> run(TokenId first, std::size_t count) {
    std::vector<TokenId> ids(count);
    std::iota(ids.begin(), ids.end(), first);
    return ids;
}

std::vector<TokenId> join(std::vector<TokenId> a, const std::vector<TokenId>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// Token t decodes to "<t>"; token 7 to a lone UTF-8 lead byte, as a window cut can leave.
std::string decode(std::span<const TokenId> ids) {
    std::string text;
    for (const TokenId id : ids) text += id == 7 ? std::string("\xe2") : "<" + std::to_string(id) + ">";
    return text;
}

// The bytes a window cut leaves behind: token 7 the first three bytes of a four-byte character,
// token 8 a continuation byte whose lead fell outside the window, token 9 a complete euro sign.
std::string raw_bytes(std::span<const TokenId> ids) {
    std::string text;
    for (const TokenId id : ids) {
        if (id == 7) {
            text += "\xf0\x9f\x98";
        } else if (id == 8) {
            text += "\x82";
        } else if (id == 9) {
            text += "\xe2\x82\xac";
        } else if (id == 5) {
            text += '\0';
        } else {
            text += "<" + std::to_string(id) + ">";
        }
    }
    return text;
}

// The check the productive Engine::detokenize applies to its reconstruction, written independently
// of it: an invalid or incomplete sequence throws rather than reaching the caller as bytes.
void require_utf8(const std::string& text) {
    const auto byte = [&text](const std::size_t at) { return static_cast<unsigned char>(text.at(at)); };
    const auto continuation = [&](const std::size_t at) {
        return byte(at) >= 0x80U && byte(at) <= 0xbfU;
    };
    std::size_t at = 0;
    while (at < text.size()) {
        const unsigned char lead = byte(at);
        std::size_t length       = 0;
        unsigned char lowest     = 0x80U;
        if (lead <= 0x7fU) {
            length = 1;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2; // 0xc0 and 0xc1 only encode overlong
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
            lowest = (lead == 0xe0U || lead == 0xedU) ? 0xa0U : 0x80U; // no overlong, no surrogate
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
            lowest = lead == 0xf0U ? 0x90U : 0x80U; // no overlong, none above U+10FFFF
        }
        if (length == 0 || at + length > text.size()) {
            throw std::invalid_argument("invalid or incomplete UTF-8");
        }
        for (std::size_t part = 1; part < length; ++part) {
            if (!continuation(at + part) || (part == 1 && byte(at + part) < lowest)) {
                throw std::invalid_argument("invalid UTF-8 continuation byte");
            }
        }
        at += length;
    }
}

// The route the serve took: the same bytes, then the strict check that a window cut fails before
// the recorded text is ever sanitized.
std::string strict_decode(std::span<const TokenId> ids) {
    const std::string text = raw_bytes(ids);
    require_utf8(text);
    return text;
}

} // namespace

int main() {
    // min_tokens 100, windows of 4 tokens on each side.
    PrefixMissTracker tracker(decode, 3, 100, 4);
    const std::vector<TokenId> system = run(1000, 300);
    const std::vector<TokenId> turn1  = run(5000, 50);
    const std::vector<TokenId> answer = run(9000, 40);

    // A first request has nothing to compare with.
    check(!tracker.settle(1, join(system, turn1), answer, 0, 350).has_value(), "first request explained");

    // The next turn appends to prompt + answer and reuses them: nothing to explain.
    const std::vector<TokenId> next = join(join(join(system, turn1), answer), run(7000, 120));
    check(!tracker.settle(2, next, run(9500, 10), 390, 120).has_value(), "a reused append was explained");

    // The same append recomputed from scratch: the earlier context was a prefix, so it was evicted.
    const auto evicted = tracker.settle(3, next, run(9600, 10), 0, 510);
    check(evicted && evicted->kind == PrefixMiss::Kind::Evicted && evicted->earlier_engine_request_id == 2 &&
              evicted->common_prefix_tokens == 510 && evicted->lost_reuse_tokens == 509 && !evicted->earlier &&
              !evicted->current,
          "an evicted prefix was not recorded as an eviction");

    // A client that drops the earlier answer (its reasoning, say) diverges at the answer's start.
    std::vector<TokenId> rewritten = join(join(system, turn1), run(7000, 120));
    const auto diverged            = tracker.settle(4, rewritten, {}, 344, 126);
    check(diverged && diverged->kind == PrefixMiss::Kind::Diverged && diverged->common_prefix_tokens == 350 &&
              diverged->earlier_context_tokens == 520 && diverged->lost_reuse_tokens == 6,
          "a rewrite was not recorded as a divergence at its first changed token");
    check(diverged && diverged->earlier && diverged->current && diverged->earlier->first == 346 &&
              diverged->earlier->token_ids.size() == 8 && diverged->earlier->token_ids[4] == 9000 &&
              diverged->current->token_ids[4] == 7000 && diverged->current->text.rfind("<5046>", 0) == 0,
          "the divergence windows do not straddle the divergence");

    // Below min_tokens computed, nothing is compared.
    check(!tracker.settle(5, rewritten, {}, 300, 99).has_value(), "a small recompute was explained");

    // Sharing less than half the prompt with every earlier context explains nothing.
    const std::vector<TokenId> other = join(run(1000, 100), run(20000, 400));
    check(!tracker.settle(6, other, {}, 0, 500).has_value(), "an unrelated prompt was explained");

    // Only the last `contexts` are kept: with room for 3, a context two requests back still explains
    // an eviction; with room for 2 it is gone.
    for (const std::size_t room : {3U, 2U}) {
        PrefixMissTracker recent(decode, room, 100, 4);
        const std::vector<TokenId> a = run(30000, 200);
        (void)recent.settle(1, a, {}, 0, 200);
        (void)recent.settle(2, run(40000, 200), {}, 0, 200);
        (void)recent.settle(3, run(50000, 200), {}, 0, 200);
        const bool explained = recent.settle(4, join(a, run(60000, 50)), {}, 0, 250).has_value();
        check(explained == (room == 3), "the recent contexts kept do not follow the capacity");
    }

    // A window cutting a multi-byte character keeps valid UTF-8.
    PrefixMissTracker cut(decode, 2, 1, 2);
    (void)cut.settle(1, std::vector<TokenId>{1, 2, 3, 7, 8}, {}, 0, 5);
    const auto broken = cut.settle(2, std::vector<TokenId>{1, 2, 3, 9, 8}, {}, 0, 5);
    check(broken && broken->earlier && broken->earlier->text == "<2><3>\xef\xbf\xbd<8>",
          "a cut character was not replaced");

    // The productive decoder validates what it returns, so a window that cuts a character threw
    // before the recorded text was ever sanitized: the finished request must survive that and keep
    // its window and counters, and say that the diagnosis gave up. The earlier context is
    // {1,2,7,3} (an incomplete character at its end) in both cases, so that side always collapses
    // to one U+FFFD; only this side of the divergence differs.
    struct CutCase {
        std::vector<TokenId> rewritten;
        std::string current_text;
        std::uint64_t failures;
    };
    const std::vector<CutCase> cuts = {
        {{1, 2, 9, 3}, "<1><2>\xe2\x82\xac<3>", 1}, // a complete character on this side
        {{1, 2, 8, 7}, "\xef\xbf\xbd", 2},          // cut on both sides
    };
    for (const CutCase& cut : cuts) {
        PrefixMissTracker strict(strict_decode, 2, 1, 2);
        (void)strict.settle(1, std::vector<TokenId>{1, 2, 7, 3}, {}, 0, 5);
        bool threw = false;
        std::optional<PrefixMissSettlement> settled;
        try {
            settled = settle_prefix_miss(strict, 2, cut.rewritten, {}, 0, 5);
        } catch (const std::exception&) {
            threw = true;
        }
        const PrefixMiss* record = settled && settled->miss ? &*settled->miss : nullptr;
        check(!threw, "a window cut through a character threw out of the diagnosis");
        check(settled && settled->diagnostic_failed && record &&
                  record->kind == PrefixMiss::Kind::Diverged && record->common_prefix_tokens == 2 &&
                  record->earlier && record->earlier->first == 0 &&
                  record->earlier->token_ids == std::vector<TokenId>{1, 2, 7, 3} &&
                  record->earlier->text == "\xef\xbf\xbd" && record->current &&
                  record->current->text == cut.current_text &&
                  strict.diagnostic_failures() == cut.failures,
              "a throwing decoder cost the record its window, its counters or its failure note");
    }

    // The route the serve takes: the bytes, then the replacement. Every run of bytes that cannot
    // decode stands for one U+FFFD, and the valid text around it keeps its bytes.
    {
        PrefixMissTracker bytes(raw_bytes, 2, 1, 2);
        (void)bytes.settle(1, std::vector<TokenId>{1, 2, 7, 3}, {}, 0, 5);
        const auto record = bytes.settle(2, std::vector<TokenId>{1, 2, 8, 7}, {}, 0, 5);
        check(record && record->earlier && record->earlier->text == "<1><2>\xef\xbf\xbd<3>",
              "a character truncated at the window's end did not become one U+FFFD");
        check(record && record->current &&
                  record->current->text == "<1><2>\xef\xbf\xbd\xef\xbf\xbd",
              "a leading continuation byte and a truncated character did not become two U+FFFD");
        check(bytes.diagnostic_failures() == 0, "the byte route reported no diagnostic failure");
    }

    // A NUL is a byte of the window's text, not its end.
    {
        PrefixMissTracker nul(raw_bytes, 2, 1, 2);
        (void)nul.settle(1, std::vector<TokenId>{1, 2, 5, 9}, {}, 0, 5);
        const auto record = nul.settle(2, std::vector<TokenId>{1, 2, 3, 9}, {}, 0, 5);
        std::string expected = "<1><2>";
        expected += '\0';
        expected += "\xe2\x82\xac";
        check(record && record->earlier && record->earlier->text == expected,
              "a NUL byte did not survive the window's text with the bytes after it");
    }

    // Concurrent settles share one tracker: nothing throws, and the contexts kept stay whole enough
    // to explain a later request.
    {
        PrefixMissTracker shared(raw_bytes, 16, 100, 4);
        const std::vector<TokenId> base = run(700000, 300);
        std::atomic_flag threw;
        std::vector<std::thread> workers;
        for (int worker = 0; worker < 4; ++worker) {
            workers.emplace_back([&] {
                for (int round = 0; round < 50; ++round) {
                    const auto prompt =
                        join(base, run(static_cast<TokenId>(800000 + worker * 1000 + round), 20));
                    try {
                        (void)settle_prefix_miss(shared,
                                                 static_cast<std::uint64_t>(worker * 100 + round),
                                                 prompt, {}, 0, 320);
                    } catch (const std::exception&) {
                        threw.test_and_set();
                    }
                }
            });
        }
        for (std::thread& worker : workers) { worker.join(); }
        check(!threw.test(), "concurrent settles threw");
        const auto after = shared.settle(999, join(base, run(900000, 20)), {}, 0, 320);
        check(after && after->kind == PrefixMiss::Kind::Diverged &&
                  after->common_prefix_tokens == 300 && after->earlier && after->current,
              "the contexts kept through concurrent settles no longer explain a divergence");
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " prefix_miss (" << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}
