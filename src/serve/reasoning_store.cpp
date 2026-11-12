#include "serve/reasoning_store.h"

#include <cstring>
#include <iterator>
#include <utility>

namespace infernix::serve {
namespace {

constexpr std::uint64_t kSeedA = 0x6e696e6665722d72ULL;
constexpr std::uint64_t kSeedB = 0x243f6a8885a308d3ULL;

// Each field is tagged and each variable-length one carries its length, so two different conversations
// never absorb the same word sequence.
enum Tag : std::uint64_t {
    kTool    = 0x746f6f6cULL,
    kMessage = 0x6d657373ULL,
    kPart    = 0x70617274ULL,
    kCall    = 0x63616c6cULL,
    kAnswer  = 0x616e7377ULL,
};

constexpr std::uint64_t mix(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

constexpr std::uint64_t rotl(std::uint64_t value, unsigned bits) noexcept {
    return (value << bits) | (value >> (64U - bits));
}

std::string_view trimmed(std::string_view text) noexcept {
    constexpr std::string_view kSpace = " \t\n\r\f\v";
    const std::size_t first           = text.find_first_not_of(kSpace);
    if (first == std::string_view::npos) { return {}; }
    return text.substr(first, text.find_last_not_of(kSpace) - first + 1);
}

bool lacks_reasoning(const ChatTurn& turn) noexcept {
    return turn.role == ChatRole::Assistant && turn.reasoning_content.empty();
}

// An answer's text (its text parts in order), or nothing for a message with tool calls or media.
std::optional<std::string> answer_text(const ChatTurn& turn) {
    if (!turn.tool_calls.empty()) { return std::nullopt; }
    std::string text;
    for (const ContentPart& part : turn.content) {
        if (part.kind != ContentKind::Text) { return std::nullopt; }
        text += part.text;
    }
    return text;
}

} // namespace

ConversationDigest::ConversationDigest() noexcept : a_(kSeedA), b_(kSeedB) {}

// Two lanes of different construction, so the 128-bit key is not one 64-bit hash twice.
void ConversationDigest::absorb(std::uint64_t word) noexcept {
    a_ = (a_ ^ mix(word)) * 0x9e3779b97f4a7c15ULL;
    a_ ^= a_ >> 29U;
    b_ = rotl(b_ + word * 0xc2b2ae3d27d4eb4fULL, 31U) * 0x9e3779b185ebca87ULL;
}

void ConversationDigest::absorb(std::string_view bytes) noexcept {
    absorb(static_cast<std::uint64_t>(bytes.size()));
    std::size_t offset = 0;
    for (; offset + sizeof(std::uint64_t) <= bytes.size(); offset += sizeof(std::uint64_t)) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + offset, sizeof(word));
        absorb(word);
    }
    if (offset < bytes.size()) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + offset, bytes.size() - offset);
        absorb(word);
    }
}

void ConversationDigest::absorb(const ToolDefinition& tool) {
    absorb(kTool);
    absorb(tool.name);
    absorb(tool.description);
    absorb(tool.input_schema_json);
    absorb(static_cast<std::uint64_t>(tool.input_examples_json.has_value()));
    if (tool.input_examples_json) { absorb(*tool.input_examples_json); }
    absorb(static_cast<std::uint64_t>(tool.deferred));
}

void ConversationDigest::add_tools(std::span<const ToolDefinition> tools) {
    absorb(static_cast<std::uint64_t>(tools.size()));
    for (const ToolDefinition& tool : tools) { absorb(tool); }
}

void ConversationDigest::add(const ChatTurn& turn) {
    absorb(kMessage);
    absorb(static_cast<std::uint64_t>(turn.role));
    absorb(static_cast<std::uint64_t>(turn.content.size()));
    for (const ContentPart& part : turn.content) {
        absorb(kPart);
        absorb(static_cast<std::uint64_t>(part.kind));
        if (part.kind == ContentKind::Text) {
            absorb(part.text);
            continue;
        }
        absorb(static_cast<std::uint64_t>(part.source.kind));
        absorb(part.source.value);
        absorb(part.source.media_type);
        absorb(std::string_view(reinterpret_cast<const char*>(part.source.bytes.data()), part.source.bytes.size()));
        absorb(static_cast<std::uint64_t>(part.image_resize_policy));
    }
    absorb(static_cast<std::uint64_t>(turn.tool_calls.size()));
    for (const ToolCall& call : turn.tool_calls) {
        absorb(kCall);
        absorb(call.name);
        absorb(call.arguments_json);
    }
    add_tools(turn.loaded_tools);
}

AnswerKey ConversationDigest::answer_key(std::string_view answer) const {
    ConversationDigest key = *this;
    key.absorb(kAnswer);
    key.absorb(trimmed(answer));
    return AnswerKey{.high = mix(key.a_), .low = mix(key.b_)};
}

ConversationDigest conversation_digest(const GenerationRequest& request) {
    ConversationDigest digest;
    digest.add_tools(request.tools);
    for (const ChatTurn& turn : request.messages) { digest.add(turn); }
    return digest;
}

ReasoningStore::ReasoningStore(std::size_t max_responses, std::size_t max_bytes, bool restore)
    : max_responses_(max_responses), max_bytes_(max_bytes), restore_(restore) {}

void ReasoningStore::insert(Entry entry) {
    std::lock_guard lock(mutex_);
    while (!order_.empty() &&
           (order_.size() >= max_responses_ || bytes_ + entry.reasoning.size() > max_bytes_)) {
        const Entry& oldest = order_.front();
        for (const std::string& id : oldest.ids) {
            const auto it = by_id_.find(id);
            if (it != by_id_.end() && it->second == order_.begin()) { by_id_.erase(it); }
        }
        if (oldest.answer) {
            const auto it = by_answer_.find(*oldest.answer);
            if (it != by_answer_.end() && it->second == order_.begin()) { by_answer_.erase(it); }
        }
        bytes_ -= oldest.reasoning.size();
        order_.pop_front();
    }
    order_.push_back(std::move(entry));
    const auto added = std::prev(order_.end());
    bytes_ += added->reasoning.size();
    for (const std::string& id : added->ids) { by_id_[id] = added; }
    if (added->answer) { by_answer_[*added->answer] = added; }
}

void ReasoningStore::remember_tool_calls(std::span<const std::string> call_ids, const std::string& reasoning) {
    if (!restore_ || call_ids.empty() || reasoning.empty() || reasoning.size() > max_bytes_ ||
        max_responses_ == 0) {
        return;
    }
    insert(Entry{.ids       = std::vector<std::string>(call_ids.begin(), call_ids.end()),
                 .answer    = {},
                 .reasoning = reasoning});
}

void ReasoningStore::remember_answer(const ConversationDigest& conversation, std::string_view answer,
                                     const std::string& reasoning) {
    if (reasoning.empty() || max_responses_ == 0 || (restore_ && reasoning.size() > max_bytes_)) { return; }
    insert(Entry{.ids       = {},
                 .answer    = conversation.answer_key(answer),
                 .reasoning = restore_ ? reasoning : std::string()});
}

ReasoningRestore ReasoningStore::restore(GenerationRequest& request, ConversationDigest& conversation) const {
    ReasoningRestore out;
    // The answers' keys first, outside the lock: each digests the conversation before its message.
    // Reasoning is not part of a key, so filling a message changes no later key.
    std::vector<std::pair<std::size_t, AnswerKey>> answers;
    conversation = ConversationDigest();
    conversation.add_tools(request.tools);
    for (std::size_t index = 0; index < request.messages.size(); ++index) {
        const ChatTurn& turn = request.messages[index];
        if (lacks_reasoning(turn)) {
            if (!turn.tool_calls.empty()) {
                ++out.missing.tool_calls;
            } else if (const std::optional<std::string> text = answer_text(turn)) {
                answers.emplace_back(index, conversation.answer_key(*text));
            }
        }
        conversation.add(turn);
    }
    if (answers.empty() && (!restore_ || out.missing.tool_calls == 0)) { return out; }

    std::lock_guard lock(mutex_);
    if (restore_) {
        for (ChatTurn& turn : request.messages) {
            if (!lacks_reasoning(turn) || turn.tool_calls.empty()) { continue; }
            const auto first = by_id_.find(turn.tool_calls.front().id);
            if (first == by_id_.end()) { continue; }
            bool same = true;
            for (const ToolCall& call : turn.tool_calls) {
                const auto it = by_id_.find(call.id);
                same          = same && it != by_id_.end() && it->second == first->second;
            }
            if (!same) { continue; }
            turn.reasoning_content = first->second->reasoning;
            ++out.restored.tool_calls;
        }
    }
    for (const auto& [index, key] : answers) {
        const auto it = by_answer_.find(key);
        if (it == by_answer_.end()) { continue; }
        ++out.missing.answers;
        if (!restore_) { continue; }
        request.messages[index].reasoning_content = it->second->reasoning;
        ++out.restored.answers;
    }
    return out;
}

std::size_t ReasoningStore::responses() const {
    std::lock_guard lock(mutex_);
    return order_.size();
}

} // namespace infernix::serve
