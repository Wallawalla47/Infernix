// The tool-call reasoning store (serve/reasoning_store.h, --restore-tool-reasoning, candidates DR-02): an
// assistant tool-call turn sent back without reasoning takes its response's reasoning; a turn that has
// reasoning, an unknown id, or ids of two responses is left as it is; the oldest responses go first by count
// and by bytes.

#include "serve/reasoning_store.h"

#include <iostream>
#include <string>
#include <vector>

using infernix::ChatRole;
using infernix::serve::ChatTurn;
using infernix::serve::ReasoningStore;
using infernix::serve::ToolCall;

namespace {

int failures = 0;

void check(bool pass, const char* message) {
    if (!pass) {
        std::cerr << "FAIL " << message << '\n';
        ++failures;
    }
}

ChatTurn assistant(std::vector<std::string> ids, std::string reasoning = {}) {
    ChatTurn turn;
    turn.role = ChatRole::Assistant;
    for (std::string& id : ids) { turn.tool_calls.push_back(ToolCall{.id = std::move(id), .name = "read", .arguments_json = "{}"}); }
    turn.reasoning_content = std::move(reasoning);
    return turn;
}

ChatTurn user() {
    ChatTurn turn;
    turn.role = ChatRole::User;
    return turn;
}

} // namespace

int main() {
    {
        ReasoningStore store(16, 1 << 20);
        store.remember(std::vector<std::string>{"call_a", "call_b"}, "think one");
        store.remember(std::vector<std::string>{"call_c"}, "think two");
        store.remember(std::vector<std::string>{"call_d"}, ""); // nothing to restore
        std::vector<ChatTurn> turns{user(),
                                    assistant({"call_a", "call_b"}),
                                    assistant({"call_c"}, "the client's own"),
                                    assistant({"call_a", "call_c"}),
                                    assistant({"call_x"}),
                                    assistant({"call_d"}),
                                    assistant({"call_c"})};
        const auto result = store.restore(turns);
        check(result.missing == 5, "five tool-call turns came without reasoning");
        check(result.restored == 2, "two of them are one remembered response's");
        check(turns[1].reasoning_content == "think one", "a response's calls take its reasoning");
        check(turns[2].reasoning_content == "the client's own", "reasoning the client sent is kept");
        check(turns[3].reasoning_content.empty(), "calls of two responses take neither");
        check(turns[4].reasoning_content.empty(), "an unknown call id takes nothing");
        check(turns[5].reasoning_content.empty(), "a response without reasoning is not remembered");
        check(turns[6].reasoning_content == "think two", "a later turn of the same id takes it again");
        check(infernix::serve::count_missing_reasoning(turns) == 3, "the count alone sees the turns still without reasoning");
    }
    {
        // Count bound: the oldest response goes when a third arrives.
        ReasoningStore store(2, 1 << 20);
        store.remember(std::vector<std::string>{"a"}, "one");
        store.remember(std::vector<std::string>{"b"}, "two");
        store.remember(std::vector<std::string>{"c"}, "three");
        std::vector<ChatTurn> turns{assistant({"a"}), assistant({"b"}), assistant({"c"})};
        const auto result = store.restore(turns);
        check(store.responses() == 2, "two responses are kept");
        check(result.restored == 2 && turns[0].reasoning_content.empty() && turns[1].reasoning_content == "two" &&
                  turns[2].reasoning_content == "three",
              "the count bound evicts the oldest response");
    }
    {
        // Byte bound: 10 bytes hold two 4-byte reasonings, not a third; a reasoning past the bound is not kept.
        ReasoningStore store(16, 10);
        store.remember(std::vector<std::string>{"a"}, "aaaa");
        store.remember(std::vector<std::string>{"b"}, "bbbb");
        store.remember(std::vector<std::string>{"c"}, "cccc");
        store.remember(std::vector<std::string>{"d"}, "dddddddddddd");
        std::vector<ChatTurn> turns{assistant({"a"}), assistant({"b"}), assistant({"c"}), assistant({"d"})};
        store.restore(turns);
        check(turns[0].reasoning_content.empty() && turns[1].reasoning_content == "bbbb" &&
                  turns[2].reasoning_content == "cccc" && turns[3].reasoning_content.empty(),
              "the byte bound evicts the oldest response and refuses one larger than the store");
    }
    {
        // An id reused by a later response names the later one; evicting the earlier keeps it.
        ReasoningStore store(2, 1 << 20);
        store.remember(std::vector<std::string>{"a"}, "first");
        store.remember(std::vector<std::string>{"a"}, "second");
        store.remember(std::vector<std::string>{"b"}, "third");
        std::vector<ChatTurn> turns{assistant({"a"})};
        store.restore(turns);
        check(turns[0].reasoning_content == "second", "a reused id survives the eviction of its earlier response");
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " reasoning_store (" << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}
