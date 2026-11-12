// The reasoning store (serve/reasoning_store.h, --restore-reasoning, candidates DR-02 and ZA-03). A tool-call
// message sent back without reasoning takes its response's reasoning by call ids; one with reasoning, an
// unknown id, or ids of two responses is left as it is. An answer sent back without reasoning takes its
// response's reasoning when its text and the whole conversation before it (tool definitions and messages,
// without reasoning or call ids) are the remembered answer's; the same answer after another prompt, another
// earlier history or other tools takes nothing. Without the option nothing is filled, and only answers
// whose response had reasoning count as missing. The oldest responses go first by count and by bytes.

#include "serve/reasoning_store.h"

#include <iostream>
#include <string>
#include <vector>

using infernix::ChatRole;
using infernix::serve::ChatTurn;
using infernix::serve::ContentKind;
using infernix::serve::ContentPart;
using infernix::serve::ConversationDigest;
using infernix::serve::GenerationRequest;
using infernix::serve::ReasoningCounts;
using infernix::serve::ReasoningRestore;
using infernix::serve::ReasoningStore;
using infernix::serve::ToolCall;
using infernix::serve::ToolDefinition;

namespace {

int failures = 0;

void check(bool pass, const char* message) {
    if (!pass) {
        std::cerr << "FAIL " << message << '\n';
        ++failures;
    }
}

ChatTurn text_turn(ChatRole role, std::string text, std::string reasoning = {}) {
    ChatTurn turn;
    turn.role = role;
    turn.content.push_back(ContentPart{.kind = ContentKind::Text, .text = std::move(text)});
    turn.reasoning_content = std::move(reasoning);
    return turn;
}

ChatTurn system_message(std::string text) { return text_turn(ChatRole::System, std::move(text)); }
ChatTurn user(std::string text = "go") { return text_turn(ChatRole::User, std::move(text)); }
ChatTurn answer(std::string text, std::string reasoning = {}) {
    return text_turn(ChatRole::Assistant, std::move(text), std::move(reasoning));
}

ChatTurn calls(std::vector<std::string> ids, std::string reasoning = {}) {
    ChatTurn turn;
    turn.role = ChatRole::Assistant;
    for (std::string& id : ids) {
        turn.tool_calls.push_back(ToolCall{.id = std::move(id), .name = "read", .arguments_json = "{}"});
    }
    turn.reasoning_content = std::move(reasoning);
    return turn;
}

ChatTurn tool_result(std::string id, std::string text) {
    ChatTurn turn = text_turn(ChatRole::Tool, std::move(text));
    turn.tool_call_id = std::move(id);
    return turn;
}

GenerationRequest request_of(std::vector<ChatTurn> messages, std::vector<ToolDefinition> tools = {}) {
    GenerationRequest request;
    request.messages = std::move(messages);
    request.tools    = std::move(tools);
    return request;
}

ReasoningRestore restore(const ReasoningStore& store, GenerationRequest& request) {
    ConversationDigest conversation;
    return store.restore(request, conversation);
}

// The flow the Chat Completions route runs: restore a request, then remember its answer under the digest the
// restore produced.
void answer_request(ReasoningStore& store, std::vector<ChatTurn> messages, const std::string& text,
                    const std::string& reasoning, std::vector<ToolDefinition> tools = {}) {
    GenerationRequest request = request_of(std::move(messages), std::move(tools));
    ConversationDigest conversation;
    (void)store.restore(request, conversation);
    store.remember_answer(conversation, text, reasoning);
}

ToolDefinition tool(std::string name) {
    return ToolDefinition{
        .name = std::move(name), .description = "reads a file", .input_schema_json = R"({"type":"object"})"};
}

} // namespace

int main() {
    {
        // Tool-call messages, by call ids.
        ReasoningStore store(16, 1 << 20, true);
        store.remember_tool_calls(std::vector<std::string>{"call_a", "call_b"}, "think one");
        store.remember_tool_calls(std::vector<std::string>{"call_c"}, "think two");
        store.remember_tool_calls(std::vector<std::string>{"call_d"}, ""); // nothing to restore
        GenerationRequest request = request_of({user(),
                                                calls({"call_a", "call_b"}),
                                                calls({"call_c"}, "the client's own"),
                                                calls({"call_a", "call_c"}),
                                                calls({"call_x"}),
                                                calls({"call_d"}),
                                                calls({"call_c"})});
        const ReasoningRestore result = restore(store, request);
        const auto& turns             = request.messages;
        check(result.missing == ReasoningCounts{.tool_calls = 5, .answers = 0},
              "five tool-call messages came without reasoning");
        check(result.restored == ReasoningCounts{.tool_calls = 2, .answers = 0},
              "two of them are one remembered response's");
        check(turns[1].reasoning_content == "think one", "a response's calls take its reasoning");
        check(turns[2].reasoning_content == "the client's own", "reasoning the client sent is kept");
        check(turns[3].reasoning_content.empty(), "calls of two responses take neither");
        check(turns[4].reasoning_content.empty(), "an unknown call id takes nothing");
        check(turns[5].reasoning_content.empty(), "a response without reasoning is not remembered");
        check(turns[6].reasoning_content == "think two", "a later message of the same id takes it again");
    }
    {
        // An answer, by its text and the conversation before it.
        ReasoningStore store(16, 1 << 20, true);
        const std::vector<ToolDefinition> tools{tool("read")};
        answer_request(store, {system_message("be brief"), user("what is 2+2?")}, "4", "two and two make four", tools);

        GenerationRequest echoed =
            request_of({system_message("be brief"), user("what is 2+2?"), answer("4"), user("and 3+3?")}, tools);
        const ReasoningRestore result = restore(store, echoed);
        check(echoed.messages[2].reasoning_content == "two and two make four",
              "an answer sent back without reasoning takes its response's");
        check(result.missing == ReasoningCounts{.tool_calls = 0, .answers = 1} &&
                  result.restored == ReasoningCounts{.tool_calls = 0, .answers = 1},
              "the restored answer is counted missing and restored");

        GenerationRequest other_prompt =
            request_of({system_message("be brief"), user("what is 1+3?"), answer("4"), user("and 3+3?")}, tools);
        const ReasoningRestore other = restore(store, other_prompt);
        check(other_prompt.messages[2].reasoning_content.empty(),
              "the same answer after another prompt takes nothing");
        check(other.missing == ReasoningCounts{} && other.restored == ReasoningCounts{},
              "an answer the store does not know is not counted");

        GenerationRequest other_history =
            request_of({system_message("be terse"), user("what is 2+2?"), answer("4"), user("and 3+3?")}, tools);
        (void)restore(store, other_history);
        check(other_history.messages[2].reasoning_content.empty(),
              "the same prompt and answer after another earlier history take nothing");

        GenerationRequest other_tools =
            request_of({system_message("be brief"), user("what is 2+2?"), answer("4"), user("and 3+3?")},
                       {tool("write")});
        (void)restore(store, other_tools);
        check(other_tools.messages[2].reasoning_content.empty(),
              "the same conversation with other tool definitions takes nothing");

        GenerationRequest own =
            request_of({system_message("be brief"), user("what is 2+2?"), answer("4", "my own notes"),
                        user("and 3+3?")},
                       tools);
        const ReasoningRestore own_result = restore(store, own);
        check(own.messages[2].reasoning_content == "my own notes" && own_result.missing == ReasoningCounts{},
              "an answer the client sent with reasoning is kept and not counted");

        GenerationRequest trimmed =
            request_of({system_message("be brief"), user("what is 2+2?"), answer("\n4  \n"), user("and 3+3?")}, tools);
        (void)restore(store, trimmed);
        check(trimmed.messages[2].reasoning_content == "two and two make four",
              "whitespace at either end of the answer does not count");

        GenerationRequest changed =
            request_of({system_message("be brief"), user("what is 2+2?"), answer("4."), user("and 3+3?")}, tools);
        (void)restore(store, changed);
        check(changed.messages[2].reasoning_content.empty(), "an edited answer takes nothing");
    }
    {
        // A conversation's later answers: earlier reasoning and call ids are not part of a key, so they
        // match whether or not the client sent the earlier reasoning back, and after it renamed call ids.
        ReasoningStore store(16, 1 << 20, true);
        answer_request(store, {user("q1")}, "a1", "r1");
        answer_request(store, {user("q1"), answer("a1"), user("q2"), calls({"call_1"}), tool_result("call_1", "ok")},
                       "a2", "r2");
        GenerationRequest third = request_of({user("q1"), answer("a1", "r1"), user("q2"), calls({"call_9"}),
                                              tool_result("call_9", "ok"), answer("a2"), user("q3")});
        const ReasoningRestore result = restore(store, third);
        check(third.messages[5].reasoning_content == "r2",
              "a later answer matches whatever earlier reasoning and call ids came with it");
        check(result.missing == ReasoningCounts{.tool_calls = 1, .answers = 1} &&
                  result.restored == ReasoningCounts{.tool_calls = 0, .answers = 1},
              "the tool-call message without reasoning is counted, the answer counted and restored");

        GenerationRequest stripped = request_of({user("q1"), answer("a1"), user("q2"), calls({"call_9"}),
                                                 tool_result("call_9", "ok"), answer("a2"), user("q3")});
        const ReasoningRestore both = restore(store, stripped);
        check(stripped.messages[1].reasoning_content == "r1" && stripped.messages[5].reasoning_content == "r2" &&
                  both.restored.answers == 2,
              "every answer of the history takes its own reasoning");
    }
    {
        // Without the option: nothing is filled; answers whose response had reasoning are still counted.
        ReasoningStore store(16, 1 << 20, false);
        store.remember_tool_calls(std::vector<std::string>{"call_a"}, "think");
        answer_request(store, {user("q1")}, "a1", "r1");
        answer_request(store, {user("q1"), answer("a1"), user("q2")}, "a2", ""); // no reasoning: not counted
        GenerationRequest request =
            request_of({user("q1"), answer("a1"), user("q2"), answer("a2"), user("q3"), calls({"call_a"})});
        const ReasoningRestore result = restore(store, request);
        check(request.messages[1].reasoning_content.empty() && request.messages[5].reasoning_content.empty(),
              "without the option no message is filled");
        check(result.missing == ReasoningCounts{.tool_calls = 1, .answers = 1} && result.restored == ReasoningCounts{},
              "without the option the tool-call message and the answer that had reasoning are counted");
    }
    {
        // Count bound: the oldest response goes when a third arrives, tool calls and answers alike.
        ReasoningStore store(2, 1 << 20, true);
        answer_request(store, {user("q1")}, "a1", "one");
        store.remember_tool_calls(std::vector<std::string>{"b"}, "two");
        answer_request(store, {user("q1"), answer("a1"), user("q2")}, "a2", "three");
        GenerationRequest request = request_of({user("q1"), answer("a1"), user("q2"), answer("a2"), calls({"b"})});
        const ReasoningRestore result = restore(store, request);
        check(store.responses() == 2, "two responses are kept");
        check(request.messages[1].reasoning_content.empty() && request.messages[3].reasoning_content == "three" &&
                  request.messages[4].reasoning_content == "two" && result.missing.answers == 1,
              "the count bound evicts the oldest response, and an evicted answer is no longer counted");
    }
    {
        // Byte bound: 10 bytes hold two 4-byte reasonings, not a third; a reasoning past the bound is not kept.
        ReasoningStore store(16, 10, true);
        store.remember_tool_calls(std::vector<std::string>{"a"}, "aaaa");
        store.remember_tool_calls(std::vector<std::string>{"b"}, "bbbb");
        answer_request(store, {user("q")}, "c", "cccc");
        answer_request(store, {user("q2")}, "d", "dddddddddddd");
        GenerationRequest request = request_of({calls({"a"}), calls({"b"})});
        (void)restore(store, request);
        check(request.messages[0].reasoning_content.empty() && request.messages[1].reasoning_content == "bbbb",
              "the byte bound evicts the oldest response");
        GenerationRequest kept    = request_of({user("q"), answer("c")});
        GenerationRequest refused = request_of({user("q2"), answer("d")});
        (void)restore(store, kept);
        (void)restore(store, refused);
        check(kept.messages[1].reasoning_content == "cccc" && refused.messages[1].reasoning_content.empty(),
              "an answer within the bound is kept, one larger than the store is refused");
    }
    {
        // An id reused by a later response names the later one; evicting the earlier keeps it.
        ReasoningStore store(2, 1 << 20, true);
        store.remember_tool_calls(std::vector<std::string>{"a"}, "first");
        store.remember_tool_calls(std::vector<std::string>{"a"}, "second");
        store.remember_tool_calls(std::vector<std::string>{"b"}, "third");
        GenerationRequest request = request_of({calls({"a"})});
        (void)restore(store, request);
        check(request.messages[0].reasoning_content == "second",
              "a reused id survives the eviction of its earlier response");
    }
    {
        // The same answer to the same conversation remembered twice (a regenerated reply) gives the later
        // reasoning, also after the earlier one is evicted.
        ReasoningStore store(2, 1 << 20, true);
        answer_request(store, {user("q")}, "a", "first");
        answer_request(store, {user("q")}, "a", "second");
        answer_request(store, {user("other")}, "b", "third");
        GenerationRequest request = request_of({user("q"), answer("a"), user("next")});
        (void)restore(store, request);
        check(request.messages[1].reasoning_content == "second",
              "a regenerated answer's latest reasoning survives the eviction of the earlier one");
    }
    {
        // The digest a restore returns is the whole request's, as conversation_digest computes it.
        ReasoningStore store(16, 1 << 20, true);
        GenerationRequest request =
            request_of({system_message("s"), user("q"), answer("a", "r"), user("q2")}, {tool("read")});
        ConversationDigest from_restore;
        (void)store.restore(request, from_restore);
        const ConversationDigest direct = infernix::serve::conversation_digest(request);
        check(from_restore.answer_key("x") == direct.answer_key("x"),
              "restore returns the digest of the whole request");
        check(direct.answer_key("x") != direct.answer_key("y"), "different answers have different keys");
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " reasoning_store (" << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}
