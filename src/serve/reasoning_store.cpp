#include "serve/reasoning_store.h"

namespace infernix::serve {
namespace {

bool lacks_reasoning(const ChatTurn& turn) {
    return turn.role == ChatRole::Assistant && !turn.tool_calls.empty() && turn.reasoning_content.empty();
}

} // namespace

ReasoningStore::ReasoningStore(std::size_t max_responses, std::size_t max_bytes)
    : max_responses_(max_responses), max_bytes_(max_bytes) {}

void ReasoningStore::remember(std::span<const std::string> call_ids, const std::string& reasoning) {
    if (call_ids.empty() || reasoning.empty() || reasoning.size() > max_bytes_ || max_responses_ == 0) { return; }
    std::lock_guard lock(mutex_);
    while (!order_.empty() && (order_.size() >= max_responses_ || bytes_ + reasoning.size() > max_bytes_)) {
        const Entry& oldest = order_.front();
        for (const std::string& id : oldest.ids) {
            const auto it = by_id_.find(id);
            if (it != by_id_.end() && it->second == order_.begin()) { by_id_.erase(it); }
        }
        bytes_ -= oldest.reasoning.size();
        order_.pop_front();
    }
    order_.push_back(Entry{.ids = std::vector<std::string>(call_ids.begin(), call_ids.end()), .reasoning = reasoning});
    bytes_ += reasoning.size();
    const auto entry = std::prev(order_.end());
    for (const std::string& id : entry->ids) { by_id_[id] = entry; }
}

ReasoningRestore ReasoningStore::restore(std::vector<ChatTurn>& turns) const {
    ReasoningRestore out;
    std::lock_guard lock(mutex_);
    for (ChatTurn& turn : turns) {
        if (!lacks_reasoning(turn)) { continue; }
        ++out.missing;
        const auto first = by_id_.find(turn.tool_calls.front().id);
        if (first == by_id_.end()) { continue; }
        bool same = true;
        for (const ToolCall& call : turn.tool_calls) {
            const auto it = by_id_.find(call.id);
            same          = same && it != by_id_.end() && it->second == first->second;
        }
        if (!same) { continue; }
        turn.reasoning_content = first->second->reasoning;
        ++out.restored;
    }
    return out;
}

std::size_t ReasoningStore::responses() const {
    std::lock_guard lock(mutex_);
    return order_.size();
}

std::size_t count_missing_reasoning(const std::vector<ChatTurn>& turns) {
    std::size_t missing = 0;
    for (const ChatTurn& turn : turns) { missing += lacks_reasoning(turn) ? 1 : 0; }
    return missing;
}

} // namespace infernix::serve
