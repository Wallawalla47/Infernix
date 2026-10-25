// The prefix-miss record (serve/prefix_miss.h, candidates CF-08): which large re-prefills are
// explained, as a divergence from an earlier context or as an eviction of one, and which are not.

#include "serve/prefix_miss.h"

#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

using infernix::TokenId;
using infernix::serve::PrefixMiss;
using infernix::serve::PrefixMissTracker;

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

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " prefix_miss (" << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}
