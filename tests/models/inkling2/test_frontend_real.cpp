// Inkling 2's frontend (src/models/inkling2/frontend/frontend.h) on the real tokenizer and the official template:
// 1. Requests: every golden conversation of tests/fixtures/inkling2/chat_template_golden.json that the Engine's
//    request types can carry (no per-message names) converted to a PromptInput (parts, reasoning, calls with their
//    arguments as JSON text, tool results by call id, declared tools, a named effort or a numeric one through
//    chat_template_kwargs) renders to Hugging Face's ids; the conversation that spells markers in client text holds
//    its twin's added tokens.
// 2. Echo: a generated turn (tests/fixtures/inkling2/output_session_golden.json) parsed by the output session and
//    sent back as the assistant message of the history renders the prompt's ids followed by the generated ids where
//    the model wrote the template's canonical form, and tokenize_chat marks exactly the reply's ids as generated; the
//    first differing position is reported otherwise (it decides where a next turn's prefix reuse ends).
// 3. Refusals: images, continuation modes, constrained output, required tool choices; context length.
//
//   INFERNIX_TEST_INKLING_TOKENIZER_DIR  tokenizer.json, tokenizer_config.json and generation_config.json

#include "models/inkling2/frontend/frontend.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
namespace ink = infernix::models::inkling2;
using infernix::ChatMessage;
using infernix::ChatRole;
using infernix::PromptInput;
using infernix::TokenId;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::optional<infernix::ReasoningEffort> effort_named(const std::string& name) {
    using E = infernix::ReasoningEffort;
    for (const E e : {E::None, E::Minimal, E::Low, E::Medium, E::High, E::XHigh, E::Max}) {
        if (infernix::reasoning_effort_name(e) == name) { return e; }
    }
    return std::nullopt;
}

// A golden context as the Engine's request types carry it; nullopt when it uses a field they lack.
std::optional<PromptInput> request_of(const Json& context) {
    PromptInput input;
    for (const Json& m : context.at("messages")) {
        if (m.contains("name")) { return std::nullopt; }
        ChatMessage message;
        const std::string role = m.at("role").get<std::string>();
        message.role = role == "system" ? ChatRole::System
                     : role == "user"   ? ChatRole::User
                     : role == "tool"   ? ChatRole::Tool
                                        : ChatRole::Assistant;
        const Json& content = m.contains("content") ? m.at("content") : Json("");
        if (content.is_string()) {
            message.parts.push_back({.text = content.get<std::string>()});
        } else {
            for (const Json& part : content) {
                message.parts.push_back({.text = part.is_string() ? part.get<std::string>() : part.at("text").get<std::string>()});
            }
        }
        if (m.contains("reasoning_content")) { message.reasoning_content = m.at("reasoning_content").get<std::string>(); }
        if (m.contains("tool_call_id")) { message.tool_call_id = m.at("tool_call_id").get<std::string>(); }
        if (m.contains("tool_calls")) {
            for (const Json& call : m.at("tool_calls")) {
                message.tool_calls.push_back({.id             = call.value("id", ""),
                                              .name           = call.at("function").at("name").get<std::string>(),
                                              .arguments_json = call.at("function").at("arguments").dump()});
            }
        }
        input.messages.push_back(std::move(message));
    }
    if (context.contains("tools")) {
        for (const Json& tool : context.at("tools")) { input.options.tool_jsons.push_back(tool.dump()); }
    }
    if (context.contains("reasoning_effort")) {
        const Json& effort = context.at("reasoning_effort");
        if (effort.is_string() && effort_named(effort.get<std::string>())) {
            input.options.reasoning_effort = effort_named(effort.get<std::string>());
        } else {
            input.options.chat_template_kwargs_json = Json{{"reasoning_effort", effort}}.dump();
        }
    }
    return input;
}

std::map<int, int> added_counts(const infernix::text::Tokenizer& tokenizer, const std::vector<TokenId>& ids) {
    std::map<int, int> counts;
    for (const int id : ids) {
        if (tokenizer.is_special_token(id)) { ++counts[id]; }
    }
    return counts;
}

bool throws(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    const char* directory = std::getenv("INFERNIX_TEST_INKLING_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "skip: INFERNIX_TEST_INKLING_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        const std::filesystem::path root(directory), source(INFERNIX_SOURCE_DIR);
        const std::string tokenizer_json = read_file(root / "tokenizer.json");
        const std::string config         = read_file(root / "tokenizer_config.json");
        const std::string generation     = read_file(root / "generation_config.json");
        const std::string chat_template  = read_file(source / "tools" / "chat_templates" / "inkling2.jinja");
        auto tokenizer = std::make_shared<const infernix::text::Tokenizer>(infernix::text::TokenizerResources{
            .tokenizer_json         = tokenizer_json,
            .tokenizer_config_json  = config,
            .generation_config_json = generation,
            .pipeline               = infernix::text::TokenizerPipeline::Inkling});
        const ink::FrontendResources resources{.tokenizer_json         = tokenizer_json,
                                               .tokenizer_config_json  = config,
                                               .chat_template_jinja    = chat_template,
                                               .generation_config_json = generation,
                                               .tokenizer              = tokenizer};
        const ink::Frontend frontend(resources, {.max_context = 8192});

        // 1. Requests against Hugging Face's ids.
        const Json templates = Json::parse(read_file(source / "tests" / "fixtures" / "inkling2" / "chat_template_golden.json"));
        std::map<std::string, std::vector<TokenId>> ids_by_name;
        int compared = 0, skipped = 0;
        for (const Json& c : templates.at("cases")) {
            const std::string name = c.at("name").get<std::string>();
            const auto input       = request_of(c.at("context"));
            if (!input) {
                ++skipped;
                continue;
            }
            const auto rendered = frontend.render(*input, 1U << 20);
            ids_by_name[name]   = rendered.ids;
            check(rendered.text == c.at("expected").get<std::string>(), name + ": text differs from transformers'");
            if (c.contains("spoof_of")) { continue; }
            ++compared;
            check(rendered.ids == c.at("ids").get<std::vector<TokenId>>(), name + ": ids differ from Hugging Face's");
        }
        for (const Json& c : templates.at("cases")) {
            if (!c.contains("spoof_of")) { continue; }
            const std::string name = c.at("name").get<std::string>(), twin = c.at("spoof_of").get<std::string>();
            // A spoof that names a message (its header spelling markers) has no request form: the template-level
            // test (infernix_inkling2_prompt_real_test) covers it.
            if (!ids_by_name.contains(name) || !ids_by_name.contains(twin)) { continue; }
            check(added_counts(*tokenizer, ids_by_name[name]) == added_counts(*tokenizer, ids_by_name[twin]),
                  name + ": client text became added tokens");
        }
        std::cout << compared << " requests render Hugging Face's ids (" << skipped
                  << " golden conversations use per-message names the request types lack)\n";

        // 2. Echo of generated turns.
        const Json outputs = Json::parse(read_file(source / "tests" / "fixtures" / "inkling2" / "output_session_golden.json"));
        PromptInput question;
        question.messages.push_back({.role = ChatRole::User, .parts = {{.text = "What is the weather?"}}});
        int echoed = 0, cases = 0;
        for (const Json& c : outputs.at("cases")) {
            const std::string name = c.at("name").get<std::string>();
            const auto generated   = c.at("ids").get<std::vector<TokenId>>();
            if (generated.empty() || generated.back() != tokenizer->find_added_token("<|content_model_end_sampling|>")->id) {
                continue; // a turn the limit cut has no history form
            }
            PromptInput prompt = question;
            if (c.at("tools").is_array()) {
                for (const Json& tool : c.at("tools")) {
                    prompt.options.tool_jsons.push_back(
                        Json{{"type", "function"}, {"function", {{"name", tool}, {"description", ""}}}}.dump());
                }
            }
            const ink::PreparedPrompt prepared = frontend.prepare(prompt);
            ink::OutputSession session         = frontend.make_output_session(prepared, {});
            const auto decision = session.preview_model(generated, 1U << 20, infernix::FinishReason::OutputLimit);
            ChatMessage reply{.role = ChatRole::Assistant};
            std::string content;
            for (const auto& delta : session.commit_preview()) {
                (delta.channel == infernix::OutputChannel::Reasoning ? reply.reasoning_content : content) += delta.text;
            }
            reply.parts.push_back({.text = content});
            for (const auto& call : session.take_tool_calls()) {
                reply.tool_calls.push_back({.id = "", .name = call.name, .arguments_json = call.arguments_json});
            }
            check(decision.accepted_tokens == generated.size(), name + ": the turn ends at its stop token");
            PromptInput history = prompt;
            history.messages.push_back(reply);
            std::vector<TokenId> expected(prepared.token_ids().begin(), prepared.token_ids().end());
            expected.insert(expected.end(), generated.begin(), generated.end());
            const auto again = frontend.render(history, 1U << 20).ids;
            ++cases;
            std::size_t at = 0;
            while (at < again.size() && at < expected.size() && again[at] == expected[at]) { ++at; }
            if (again == expected) {
                ++echoed;
                // The history's generated mask covers exactly the reply's ids.
                const infernix::ChatTokens chat = frontend.tokenize_chat(history);
                bool mask = chat.ids == again && chat.generated.size() == again.size();
                for (std::size_t i = 0; mask && i < again.size(); ++i) {
                    mask = chat.generated[i] == (i >= prepared.token_ids().size() ? 1 : 0);
                }
                check(mask, name + ": tokenize_chat marks exactly the reply's ids as generated");
            } else {
                std::cout << "  echo of \"" << name << "\": the history departs at generated position "
                          << (at - prepared.token_ids().size()) << " of " << generated.size() << '\n';
            }
        }
        std::cout << echoed << " of " << cases << " generated turns echo token for token\n";

        // 3. Refusals and limits.
        PromptInput image = question;
        image.messages.back().parts.push_back({.kind = infernix::MessagePartKind::Media});
        check(throws([&] { (void)frontend.prepare(image); }), "an image is refused");
        PromptInput continuation = question;
        continuation.options.continuation = infernix::PromptContinuationMode::ContinueFinalAssistant;
        check(throws([&] { (void)frontend.prepare(continuation); }), "a continuation mode is refused");
        const ink::PreparedPrompt plain = frontend.prepare(question);
        check(throws([&] { (void)frontend.make_output_session(plain, {}, {}, {}, infernix::OutputConstraint::json_object()); }),
              "constrained output is refused");
        check(throws([&] {
                  (void)frontend.make_output_session(plain, {}, {}, {}, {}, {.mode = infernix::ToolChoiceMode::Required});
              }),
              "a required tool choice is refused");
        check(!throws([&] { (void)frontend.make_output_session(plain, {}); }), "the default request is accepted");
        const ink::Frontend small(resources, {.max_context = 8});
        bool context_error = false;
        try {
            (void)small.prepare(question);
        } catch (const infernix::RequestError& error) {
            context_error = error.kind() == infernix::RequestErrorKind::ContextLengthExceeded;
        }
        check(context_error, "a prompt past max_context is a context-length error");
        check(frontend.count_tokens(question) == plain.token_ids().size(), "count_tokens counts the prepared ids");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " checks failed\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
