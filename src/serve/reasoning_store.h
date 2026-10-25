#pragma once

// The reasoning of the responses that called tools, by tool-call id (--restore-tool-reasoning, candidates
// DR-02). Most OpenAI-compatible agent clients send an assistant turn back without its
// reasoning_content. Inside a tool loop the chat template renders that reasoning, so the turn's tokens
// then differ from the ones the model generated, and every later turn of the loop re-prefills from
// there. With the store, such a turn takes the reasoning its response carried, and renders as
// generated. A client that sends the reasoning itself is left as it is.

#include "serve/request.h"

#include <cstddef>
#include <list>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernix::serve {

// A request's assistant turns that have tool calls and arrived without reasoning, and those of them the
// store filled.
struct ReasoningRestore {
    std::size_t missing  = 0;
    std::size_t restored = 0;
};

class ReasoningStore {
public:
    // At most `max_responses` responses and `max_bytes` of reasoning; the oldest go first.
    ReasoningStore(std::size_t max_responses, std::size_t max_bytes);

    // One response's reasoning under each of its tool-call ids (nothing without either).
    void remember(std::span<const std::string> call_ids, const std::string& reasoning);

    // Fills each assistant turn with tool calls and no reasoning whose call ids all name one remembered
    // response.
    ReasoningRestore restore(std::vector<ChatTurn>& turns) const;

    [[nodiscard]] std::size_t responses() const;

private:
    struct Entry {
        std::vector<std::string> ids;
        std::string reasoning;
    };

    std::size_t max_responses_;
    std::size_t max_bytes_;
    mutable std::mutex mutex_;
    std::list<Entry> order_; // oldest first
    std::unordered_map<std::string, std::list<Entry>::iterator> by_id_;
    std::size_t bytes_ = 0;
};

// The assistant turns with tool calls and no reasoning (ReasoningRestore::missing), without a store.
[[nodiscard]] std::size_t count_missing_reasoning(const std::vector<ChatTurn>& turns);

} // namespace infernix::serve
