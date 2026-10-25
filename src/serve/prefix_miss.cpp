#include "serve/prefix_miss.h"

#include "serve/request_json.h"

#include <algorithm>
#include <utility>

namespace infernix::serve {
namespace {

std::size_t common_prefix(std::span<const infernix::TokenId> a, std::span<const infernix::TokenId> b) {
    const std::size_t n = std::min(a.size(), b.size());
    return static_cast<std::size_t>(std::mismatch(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n), b.begin()).first -
                                    a.begin());
}

// A window cut can split a multi-byte character: such bytes become U+FFFD.
std::string valid_utf8(const std::string& text) {
    return RequestJson::parse(RequestJson(text).dump(-1, ' ', false, RequestJson::error_handler_t::replace))
        .get<std::string>();
}

} // namespace

const char* prefix_miss_kind_name(PrefixMiss::Kind kind) noexcept {
    return kind == PrefixMiss::Kind::Diverged ? "diverged" : "evicted";
}

PrefixMissTracker::PrefixMissTracker(Detokenize detokenize, std::size_t contexts,
                                     std::uint32_t min_tokens, std::uint32_t window)
    : detokenize_(std::move(detokenize)), capacity_(contexts), min_tokens_(min_tokens), window_(window) {}

PrefixMissWindow PrefixMissTracker::window(std::span<const infernix::TokenId> tokens, std::size_t at) const {
    const std::size_t first = at > window_ ? at - window_ : 0;
    const std::size_t last  = std::min(tokens.size(), at + window_);
    PrefixMissWindow out;
    out.first = static_cast<std::uint32_t>(first);
    out.token_ids.assign(tokens.begin() + static_cast<std::ptrdiff_t>(first),
                         tokens.begin() + static_cast<std::ptrdiff_t>(last));
    out.text = valid_utf8(detokenize_(out.token_ids));
    return out;
}

std::optional<PrefixMiss> PrefixMissTracker::settle(std::uint64_t engine_request_id,
                                                    std::span<const infernix::TokenId> prompt,
                                                    std::span<const infernix::TokenId> output,
                                                    std::uint32_t reused_tokens,
                                                    std::uint32_t computed_prefill_tokens) {
    std::lock_guard lock(mutex_);
    std::optional<PrefixMiss> miss;
    if (computed_prefill_tokens >= min_tokens_ && !prompt.empty()) {
        // The earlier context sharing the longest prefix (the latest on ties).
        const Context* best  = nullptr;
        std::size_t shared   = 0;
        for (auto it = contexts_.rbegin(); it != contexts_.rend(); ++it) {
            const std::size_t n = common_prefix(prompt, it->tokens);
            if (best == nullptr || n > shared) {
                best   = &*it;
                shared = n;
            }
        }
        if (best != nullptr) {
            const std::size_t reusable = std::min(shared, prompt.size() - 1);
            const std::size_t lost     = reusable > reused_tokens ? reusable - reused_tokens : 0;
            const bool diverged        = shared < best->tokens.size() && shared < prompt.size();
            if ((diverged && 2 * shared >= prompt.size()) || (!diverged && lost >= min_tokens_)) {
                miss.emplace();
                miss->kind                      = diverged ? PrefixMiss::Kind::Diverged : PrefixMiss::Kind::Evicted;
                miss->earlier_engine_request_id = best->engine_request_id;
                miss->earlier_context_tokens    = static_cast<std::uint32_t>(best->tokens.size());
                miss->common_prefix_tokens      = static_cast<std::uint32_t>(shared);
                miss->reused_tokens             = reused_tokens;
                miss->computed_prefill_tokens   = computed_prefill_tokens;
                miss->lost_reuse_tokens         = static_cast<std::uint32_t>(lost);
                if (diverged) {
                    miss->earlier = window(best->tokens, shared);
                    miss->current = window(prompt, shared);
                }
            }
        }
    }
    Context context{.engine_request_id = engine_request_id, .tokens = {}};
    context.tokens.reserve(prompt.size() + output.size());
    context.tokens.insert(context.tokens.end(), prompt.begin(), prompt.end());
    context.tokens.insert(context.tokens.end(), output.begin(), output.end());
    contexts_.push_back(std::move(context));
    while (contexts_.size() > capacity_) { contexts_.pop_front(); }
    return miss;
}

} // namespace infernix::serve
