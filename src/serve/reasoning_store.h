#pragma once

// The reasoning of this server's Chat Completions responses, so an assistant message sent back without it
// renders as generated (--restore-reasoning, candidates DR-02 and ZA-03). Most OpenAI-compatible clients send
// assistant messages back without their reasoning_content. Where the chat template renders that reasoning
// (every Qwen template inside a tool loop; Qwen3.8's for every closed turn unless preserve_thinking is false),
// the message's tokens then differ from the ones the model generated, and every later turn re-prefills from
// there. With the store, such a message takes the reasoning its response carried and renders as generated.
// A message that carries reasoning of its own is left as it is.
//
// A tool-calling response's reasoning is kept under its tool-call ids. An answer (a response without tool
// calls) has no id: its reasoning is kept under an AnswerKey, a 128-bit digest of the answer's text and of
// the whole conversation it answered (the request's tool definitions and every message, without reasoning).
// The same answer in another conversation, or after another prompt, has another key.

#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace infernix::serve {

struct AnswerKey {
    std::uint64_t high = 0;
    std::uint64_t low  = 0;

    [[nodiscard]] friend bool operator==(const AnswerKey&, const AnswerKey&) noexcept = default;
};

// A conversation's running digest as the template renders it, reasoning left out: the tool definitions
// (name, description, schema, examples, whether deferred), then each message's role, content (text, and each
// media item's source), tool calls (name and arguments) and loaded tools. Tool-call ids are left out too:
// the template does not render them.
class ConversationDigest {
public:
    ConversationDigest() noexcept;

    void add_tools(std::span<const ToolDefinition> tools);
    void add(const ChatTurn& turn);
    // The key of an answer whose text is `answer` following the messages added so far. Whitespace at
    // either end of `answer` does not count: the template trims it, and some clients do.
    [[nodiscard]] AnswerKey answer_key(std::string_view answer) const;

private:
    void absorb(std::uint64_t word) noexcept;
    void absorb(std::string_view bytes) noexcept;
    void absorb(const ToolDefinition& tool);

    std::uint64_t a_;
    std::uint64_t b_;
};

// The digest of a request's tool definitions and whole conversation: the prefix its answer follows.
[[nodiscard]] ConversationDigest conversation_digest(const GenerationRequest& request);

// Assistant messages by kind.
struct ReasoningCounts {
    std::size_t tool_calls = 0; // messages with tool calls
    std::size_t answers    = 0; // messages without

    [[nodiscard]] friend bool operator==(const ReasoningCounts&, const ReasoningCounts&) noexcept = default;
};

// A request's assistant messages that arrived without reasoning, and those of them the store filled.
struct ReasoningRestore {
    // Every tool-call message without reasoning, and each answer without reasoning whose response from this
    // server had some (one the store remembers).
    ReasoningCounts missing;
    // Of those, the messages given their response's reasoning (--restore-reasoning).
    ReasoningCounts restored;
};

class ReasoningStore {
public:
    // At most `max_responses` responses and `max_bytes` of reasoning; the oldest go first. Without `restore`
    // the store keeps only which answers had reasoning, to count them: no reasoning, and no tool calls.
    ReasoningStore(std::size_t max_responses, std::size_t max_bytes, bool restore);

    // One tool-calling response's reasoning under each of its tool-call ids (nothing without either, or
    // without `restore`).
    void remember_tool_calls(std::span<const std::string> call_ids, const std::string& reasoning);
    // One answer's reasoning under its key: `conversation` is the digest of the request it answered
    // (nothing without reasoning).
    void remember_answer(const ConversationDigest& conversation, std::string_view answer,
                         const std::string& reasoning);

    // Counts `request`'s assistant messages without reasoning and, with `restore`, fills each tool-call
    // message whose call ids all name one remembered response, and each answer whose key names one.
    // `conversation` receives the digest of the whole request, for remember_answer.
    ReasoningRestore restore(GenerationRequest& request, ConversationDigest& conversation) const;

    [[nodiscard]] std::size_t responses() const;

private:
    struct Entry {
        std::vector<std::string> ids;    // a tool-calling response's call ids
        std::optional<AnswerKey> answer; // an answer's key
        std::string reasoning;           // empty without restore_
    };
    struct AnswerKeyHash {
        [[nodiscard]] std::size_t operator()(const AnswerKey& key) const noexcept {
            return static_cast<std::size_t>(key.low);
        }
    };

    // Takes the mutex, evicts the oldest responses until `entry` fits both bounds, and appends it.
    void insert(Entry entry);

    std::size_t max_responses_;
    std::size_t max_bytes_;
    bool restore_;
    mutable std::mutex mutex_;
    std::list<Entry> order_; // oldest first
    std::unordered_map<std::string, std::list<Entry>::iterator> by_id_;
    std::unordered_map<AnswerKey, std::list<Entry>::iterator, AnswerKeyHash> by_answer_;
    std::size_t bytes_ = 0;
};

} // namespace infernix::serve
