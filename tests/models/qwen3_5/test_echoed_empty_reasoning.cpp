// A reply whose reasoning is empty, sent back as history, against the ids the model generated (candidates
// ZH-03, HF Qwen3.8-Flash-Next discussion #86). The generation prompt ends `<think>` + "\n" (id 198); a model
// that skips its thinking writes "\n" (198) and `</think>`. The next request's template renders that reply as
// `<think>\n\n</think>`, whose "\n\n" one encoding of the whole text makes one id (271), so the reply's
// endpoint (prompt plus generated ids) could not serve the next turn, which recomputed the reply. The
// Frontend now encodes each past reply apart from the generation opener before it
// (RenderedChat::generation_splits), so the history keeps 198 198.
//
// It renders each history through the real Frontend (chat template and tokenizer) exactly as the following
// request does, and requires the history to hold the prompt's ids then the generated ids, through the
// reply's `<|im_end|>`, with the same text: an empty-reasoning answer and tool call (Qwen3.8 and Qwen3.6), a
// reply with reasoning (unchanged by the split), and a thinking-off conversation, whose opener is
// `<think>\n\n</think>\n\n` and whose replies split after it.
//
//   INFERNIX_TEST_TOKENIZER_DIR   a directory holding tokenizer.json, tokenizer_config.json and
//                                 generation_config.json of a Qwen3.5-family artifact (tests/README.md)
//
// CPU only. Skipped (77) without the directory, or for a tokenizer whose "\n" and "\n\n" are not 198 and 271.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/resources.h"
#include "text/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace q  = infernix::models::qwen3_5;
using infernix::TokenId;

constexpr TokenId kNewline       = 198; // "\n"
constexpr TokenId kDoubleNewline = 271; // "\n\n"

int failures = 0;

void check(bool pass, const std::string& message) {
    if (!pass) {
        std::cerr << "FAIL " << message << '\n';
        ++failures;
    }
}

std::string read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::string ids_text(std::span<const TokenId> ids) {
    std::string text = "[";
    for (std::size_t index = 0; index < ids.size(); ++index) {
        text += (index == 0 ? "" : ",") + std::to_string(ids[index]);
    }
    return text + "]";
}

// The tokenizer files and one maintained template; the views the Frontend borrows live here.
struct Resources {
    std::string tokenizer_json, tokenizer_config_json, generation_config_json, chat_template;
    std::shared_ptr<const infernix::text::Tokenizer> tokenizer;
};

q::Frontend make_frontend(const Resources& source) {
    q::FrontendResources resources{.tokenizer_json         = source.tokenizer_json,
                                   .tokenizer_config_json  = source.tokenizer_config_json,
                                   .chat_template_jinja    = source.chat_template,
                                   .generation_config_json = source.generation_config_json};
    resources.tokenizer          = source.tokenizer;
    resources.public_token_count = static_cast<std::uint32_t>(source.tokenizer->vocab_size());
    q::FrontendOptions options;
    options.vision_enabled = false;
    options.max_context    = std::numeric_limits<std::uint32_t>::max();
    return q::make_frontend(resources, options);
}

infernix::ChatMessage message(infernix::ChatRole role, std::string text, std::string reasoning = {}) {
    infernix::ChatMessage turn;
    turn.role = role;
    if (!text.empty()) {
        turn.parts.push_back(infernix::MessagePart{
            .kind = infernix::MessagePartKind::Text, .text = std::move(text), .media = {}});
    }
    turn.reasoning_content = std::move(reasoning);
    return turn;
}

std::vector<TokenId> prompt_ids(const q::Frontend& frontend, infernix::PromptInput input) {
    const q::PreparedPrompt prepared = frontend.prepare(std::move(input));
    const std::span<const TokenId> ids = prepared.token_ids();
    return {ids.begin(), ids.end()};
}

// Where the history's earlier replies' generated ids begin (PreparedPromptData::reply_frontiers).
std::vector<std::uint32_t> reply_frontiers(const q::Frontend& frontend, infernix::PromptInput input) {
    const q::PreparedPrompt prepared = frontend.prepare(std::move(input));
    return q::PreparedPromptAccess::view(prepared).reply_frontiers;
}

// The latest reply's generated ids begin where its generation prompt ended.
void check_latest_frontier(const std::string& what, const q::Frontend& frontend, const infernix::PromptInput& echoed,
                           std::size_t prompt_ids) {
    const std::vector<std::uint32_t> frontiers = reply_frontiers(frontend, echoed);
    check(!frontiers.empty() && frontiers.back() == prompt_ids,
          what + ": the latest reply frontier is " +
              (frontiers.empty() ? std::string("absent") : std::to_string(frontiers.back())) + ", not " +
              std::to_string(prompt_ids));
}

std::vector<TokenId> concat(std::vector<TokenId> head, std::span<const TokenId> tail) {
    head.insert(head.end(), tail.begin(), tail.end());
    return head;
}

std::size_t first_difference(std::span<const TokenId> left, std::span<const TokenId> right) {
    std::size_t index = 0;
    while (index < left.size() && index < right.size() && left[index] == right[index]) { ++index; }
    return index;
}

struct Ids {
    TokenId think_close = 0;
    TokenId im_end      = 0;
    TokenId tool_call   = 0;
};

// One case: the prompt `opener` answered, the reply the model generated (`generated`), and the history the
// next request renders (`echoed`). Prints both sequences around the opener and checks the stated shape.
void empty_reasoning_case(const std::string& what, const q::Frontend& frontend, const Ids& ids,
                          const infernix::PromptInput& opener, const std::vector<TokenId>& generated_tail,
                          const infernix::PromptInput& echoed) {
    const std::vector<TokenId> prompt = prompt_ids(frontend, opener);
    const std::vector<TokenId> generated =
        concat({kNewline, ids.think_close}, generated_tail); // "\n" then </think>, as a model skipping thinking
    const std::vector<TokenId> expected = concat(prompt, generated);
    const std::vector<TokenId> history  = prompt_ids(frontend, echoed);
    const std::size_t at                = first_difference(expected, history);
    const std::size_t from              = prompt.size() >= 3 ? prompt.size() - 3 : 0;
    std::cout << what << ": prompt " << prompt.size() << " ids, ends "
              << ids_text(std::span(prompt).subspan(from)) << "; generated "
              << ids_text(std::span(generated).first(std::min<std::size_t>(generated.size(), 4)))
              << "...; the history from id " << from << ": "
              << ids_text(std::span(history).subspan(from, std::min<std::size_t>(history.size() - from, 7)))
              << "...; first difference at id " << at << '\n';

    check(!prompt.empty() && prompt.back() == kNewline, what + ": the generation prompt does not end in \"\\n\"");
    check(at == expected.size(), what + ": the history departs from the prompt and generated ids at id " +
                                     std::to_string(at) + " of " + std::to_string(expected.size()));
    check(history.size() > prompt.size() && history[prompt.size()] == kNewline &&
              history[prompt.size() + 1] == ids.think_close,
          what + ": the history does not keep the empty reasoning as 198 198 then </think>");
    check(frontend.detokenize(expected) ==
              frontend.detokenize(std::span(history).first(std::min(history.size(), expected.size()))),
          what + ": the history's text differs from the generated text");
    check_latest_frontier(what, frontend, echoed, prompt.size());
}

// A reasoning closed at once: the model writes `</think>` straight after the prompt's `<think>` "\n", and
// the template echoes it as `<think>\n\n</think>`, one "\n" more than generated, so the history departs
// from the reply's endpoint at the prompt's end. That is the latest reply frontier, where the hybrid cache's
// prompt-end tap (taken for a reply whose first id is the prompt's prompt_end_tap_token) stands.
void closed_at_once_case(const std::string& what, const q::Frontend& frontend, const Ids& ids,
                         const infernix::PromptInput& opener, const std::vector<TokenId>& generated_tail,
                         const infernix::PromptInput& echoed) {
    const std::vector<TokenId> prompt    = prompt_ids(frontend, opener);
    const std::vector<TokenId> generated = concat({ids.think_close}, generated_tail);
    const std::vector<TokenId> expected  = concat(prompt, generated);
    const std::vector<TokenId> history   = prompt_ids(frontend, echoed);
    const std::size_t at                 = first_difference(expected, history);
    std::cout << what << ": prompt " << prompt.size() << " ids; first difference at id " << at << '\n';
    check(at == prompt.size(), what + ": the history departs from the generated ids at id " + std::to_string(at) +
                                   ", not at the prompt's end " + std::to_string(prompt.size()));
    check_latest_frontier(what, frontend, echoed, prompt.size());
}

} // namespace

int main() {
    const char* directory = std::getenv("INFERNIX_TEST_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "skip: INFERNIX_TEST_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        const fs::path root(directory);
        Resources qwen3_8;
        qwen3_8.tokenizer_json         = read_file(root / "tokenizer.json");
        qwen3_8.tokenizer_config_json  = read_file(root / "tokenizer_config.json");
        qwen3_8.generation_config_json = read_file(root / "generation_config.json");
        qwen3_8.chat_template = read_file(fs::path(INFERNIX_SOURCE_DIR) / "tools" / "chat_templates" / "qwen3_8.jinja");
        qwen3_8.tokenizer     = std::make_shared<const infernix::text::Tokenizer>(infernix::text::TokenizerResources{
            qwen3_8.tokenizer_json, qwen3_8.tokenizer_config_json, qwen3_8.generation_config_json});
        Resources qwen3_6 = qwen3_8;
        qwen3_6.chat_template =
            read_file(fs::path(INFERNIX_SOURCE_DIR) / "tools" / "chat_templates" / "qwen3_6.jinja");

        const q::Frontend frontend = make_frontend(qwen3_8);
        if (frontend.tokenize_text("\n") != std::vector<TokenId>{kNewline} ||
            frontend.tokenize_text("\n\n") != std::vector<TokenId>{kDoubleNewline}) {
            std::cout << "skip: this tokenizer does not encode \"\\n\" as 198 and \"\\n\\n\" as 271\n";
            return 77;
        }
        Ids ids;
        const auto single = [&](std::string_view text) {
            const std::vector<TokenId> encoded = frontend.tokenize_text(text);
            if (encoded.size() != 1) { throw std::runtime_error("not one token: " + std::string(text)); }
            return encoded.front();
        };
        ids.think_close = single("</think>");
        ids.im_end      = single("<|im_end|>");
        ids.tool_call   = single("<tool_call>");
        std::cout << "</think> = " << ids.think_close << ", <|im_end|> = " << ids.im_end << '\n';

        using infernix::ChatRole;
        const std::string question = "What is 2+2?";
        const std::string answer   = "The answer is 4.";

        // 1. Qwen3.8, a plain answer: its next request renders every earlier answer's reasoning block.
        {
            infernix::PromptInput opener;
            opener.messages = {message(ChatRole::User, question)};
            infernix::PromptInput echoed;
            echoed.messages = {message(ChatRole::User, question), message(ChatRole::Assistant, answer),
                               message(ChatRole::User, "Thanks.")};
            const std::vector<TokenId> tail = concat(concat({kDoubleNewline}, frontend.tokenize_text(answer)),
                                                     std::vector<TokenId>{ids.im_end});
            empty_reasoning_case("qwen3_8 answer", frontend, ids, opener, tail, echoed);
            closed_at_once_case("qwen3_8 answer closed at once", frontend, ids, opener, tail, echoed);
            // The prompt opens the reasoning, so a reply whose first id is </think> is marked as not echoed.
            const q::PreparedPrompt prepared = frontend.prepare(opener);
            const auto mark                  = q::PreparedPromptAccess::view(prepared).prompt_end_tap_token;
            check(mark && *mark == ids.think_close, "a prompt opening the reasoning does not mark </think> as unechoed");
        }

        // 2. The control: a reply with reasoning, its ids the tokenizer's own segmentation of each part,
        // renders as generated through its <|im_end|>, so its endpoint serves the next turn.
        {
            const std::string reasoning = "Two plus two is four.";
            infernix::PromptInput opener;
            opener.messages = {message(ChatRole::User, question)};
            infernix::PromptInput echoed;
            echoed.messages = {message(ChatRole::User, question), message(ChatRole::Assistant, answer, reasoning),
                               message(ChatRole::User, "Thanks.")};
            const std::vector<TokenId> prompt = prompt_ids(frontend, opener);
            std::vector<TokenId> generated    = frontend.tokenize_text(reasoning + "\n");
            generated.push_back(ids.think_close);
            generated.push_back(kDoubleNewline);
            generated = concat(std::move(generated), frontend.tokenize_text(answer));
            generated.push_back(ids.im_end);
            const std::vector<TokenId> expected = concat(prompt, generated);
            const std::vector<TokenId> history  = prompt_ids(frontend, echoed);
            const std::size_t at                = first_difference(expected, history);
            std::cout << "qwen3_8 control (reasoning \"" << reasoning << "\"): first difference at id " << at << " of "
                      << expected.size() << " generated-and-prompt ids\n";
            check(at == expected.size(), "the control reply does not render as generated (first difference at id " +
                                             std::to_string(at) + ")");
            check_latest_frontier("qwen3_8 control", frontend, echoed, prompt.size());
        }

        // 2b. Thinking off: the opener is `<think>\n\n</think>\n\n` (its "\n\n" one id, as the template wrote
        // it), and the model's answer follows it; the history splits after the whole opener, so it keeps the
        // prompt's ids and then the answer's.
        {
            infernix::PromptInput opener;
            opener.messages                = {message(ChatRole::User, question)};
            opener.options.enable_thinking = false;
            infernix::PromptInput echoed   = opener;
            echoed.messages = {message(ChatRole::User, question), message(ChatRole::Assistant, answer),
                               message(ChatRole::User, "Thanks.")};
            const std::vector<TokenId> prompt = prompt_ids(frontend, opener);
            const std::vector<TokenId> expected =
                concat(concat(prompt, frontend.tokenize_text(answer)), std::vector<TokenId>{ids.im_end});
            const std::vector<TokenId> history = prompt_ids(frontend, echoed);
            const std::size_t at               = first_difference(expected, history);
            std::cout << "qwen3_8 thinking off: first difference at id " << at << " of " << expected.size()
                      << " generated-and-prompt ids\n";
            check(at == expected.size(), "a thinking-off answer does not render as generated (first difference at id " +
                                             std::to_string(at) + ")");
            check_latest_frontier("qwen3_8 thinking off", frontend, echoed, prompt.size());
            const q::PreparedPrompt prepared = frontend.prepare(opener);
            check(!q::PreparedPromptAccess::view(prepared).prompt_end_tap_token,
                  "a thinking-off prompt marks an unechoed first id");
        }

        // 3. Qwen3.8, a tool call with empty reasoning inside a tool loop, with preserve_thinking false: the
        // turn after the last user query keeps its reasoning block whatever the client asks. The call's own
        // ids are taken as the template renders them, so only the reasoning block is under test.
        {
            const std::string tool = R"({"type":"function","function":{"name":"read","parameters":)"
                                     R"({"type":"object","properties":{"path":{"type":"string"}},)"
                                     R"("required":["path"]}}})";
            infernix::PromptInput opener;
            opener.messages                  = {message(ChatRole::User, "Read a.txt.")};
            opener.options.tool_jsons        = {tool};
            opener.options.preserve_thinking = false;
            infernix::PromptInput echoed     = opener;
            infernix::ChatMessage call       = message(ChatRole::Assistant, "");
            call.tool_calls.push_back(
                infernix::ToolCall{.id = "call_0", .name = "read", .arguments_json = R"({"path":"a.txt"})"});
            infernix::ChatMessage result = message(ChatRole::Tool, "hello");
            result.tool_call_id          = "call_0";
            echoed.messages.push_back(std::move(call));
            echoed.messages.push_back(std::move(result));

            using Template = std::pair<const char*, const Resources*>;
            for (const auto& [name, resources] :
                 {Template{"qwen3_8 tool call", &qwen3_8}, Template{"qwen3_6 tool call", &qwen3_6}}) {
                const q::Frontend renderer      = make_frontend(*resources);
                const std::vector<TokenId> prompt  = prompt_ids(renderer, opener);
                const std::vector<TokenId> history = prompt_ids(renderer, echoed);
                // The reply as the template renders it after </think>: through the first <|im_end|>.
                std::size_t close = prompt.size() - 1;
                while (close < history.size() && history[close] != ids.think_close) { ++close; }
                std::size_t end = close;
                while (end < history.size() && history[end] != ids.im_end) { ++end; }
                if (end >= history.size()) {
                    check(false, std::string(name) + ": the history has no reasoning close and end after the prompt");
                    continue;
                }
                const std::vector<TokenId> tail(history.begin() + static_cast<std::ptrdiff_t>(close) + 1,
                                                history.begin() + static_cast<std::ptrdiff_t>(end) + 1);
                check(!tail.empty() && tail.front() == kDoubleNewline && tail.size() > 1 && tail[1] == ids.tool_call,
                      std::string(name) + ": the call does not follow </think> as \"\\n\\n\" and <tool_call>");
                empty_reasoning_case(name, renderer, ids, opener, tail, echoed);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " echoed_empty_reasoning (" << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}
