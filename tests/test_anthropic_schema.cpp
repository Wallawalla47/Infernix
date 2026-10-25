#include "serve/anthropic_messages.h"

#include "serve/generation_service.h"
#include "serve/server_tools.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json = infernix::serve::RequestJson;
using namespace infernix::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

Json base_request() {
    return Json{{"model", "claude-local"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
                {"max_tokens", 4096}};
}

RequestLimits limits() {
    RequestLimits value;
    value.default_max_tokens = 8192;
    return value;
}

AnthropicMessagesRequest parse(const Json& body) {
    return parse_anthropic_messages_request(body, limits());
}

std::string api_code(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& error) { return error.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

std::string api_param(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& error) { return error.error().param; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

ResolvedPromptSemantics semantics(const GenerationRequest& request, bool default_thinking = true) {
    ServeOptions options;
    options.enable_thinking = default_thinking;
    return resolve_prompt_semantics(request, options);
}

infernix::PromptInput prompt(const GenerationRequest& request) {
    return to_prompt_input(request, semantics(request), [](const ContentPart& part) {
        infernix::OwnedMedia media;
        media.kind =
            part.kind == ContentKind::Image ? infernix::MediaKind::Image : infernix::MediaKind::Video;
        media.bytes               = {1};
        media.image_resize_policy = part.image_resize_policy;
        return media;
    });
}

Json parse_event(const std::string& value) {
    const std::size_t newline = value.find('\n');
    if (!value.starts_with("event: ") || newline == std::string::npos || !value.ends_with("\n\n")) {
        throw std::runtime_error("invalid Anthropic SSE framing");
    }
    const std::string type   = value.substr(7, newline - 7);
    const std::string prefix = "data: ";
    if (value.compare(newline + 1U, prefix.size(), prefix) != 0) {
        throw std::runtime_error("missing Anthropic SSE data");
    }
    Json payload = Json::parse(value.substr(newline + 1U + prefix.size(),
                                            value.size() - newline - 1U - prefix.size() - 2U));
    if (payload.at("type") != type) { throw std::runtime_error("Anthropic SSE type mismatch"); }
    return payload;
}

int test_envelope_and_field_policy() {
    Json body             = base_request();
    body["stream"]        = true;
    body["temperature"]   = 0.7;
    body["top_p"]         = 0.8;
    body["top_k"]         = 20;
    body["metadata"]      = Json{{"user_id", "u"}};
    body["service_tier"]  = "auto";
    body["inference_geo"] = "anywhere";
    body["future_field"]  = Json{{"unknown", true}};
    body["output_config"] = Json{{"effort", "low"}, {"future_option", true}};

    const AnthropicMessagesRequest request = parse(body);
    int failures =
        check(request.model == "claude-local" && request.stream && request.output_tokens_explicit &&
                  request.generation.max_tokens == 4096,
              "Messages envelope fields were not preserved");
    failures += check(request.generation.sampling.temperature == 0.7 &&
                          request.generation.sampling.top_p == 0.8 &&
                          request.generation.sampling.top_k == 20 &&
                          request.generation.reasoning_effort == RequestedReasoningEffort::Low,
                      "executable Anthropic generation fields were not lowered");

    body.erase("max_tokens");
    const AnthropicMessagesRequest defaulted = parse(body);
    failures += check(!defaulted.output_tokens_explicit && defaulted.generation.max_tokens == 8192,
                      "omitted max_tokens did not use the server default");
    body["max_tokens"] = 0;
    failures += check(api_code([&] { (void)parse(body); }) == "cache_prewarm_not_supported",
                      "max_tokens=0 was accepted as a false cache prewarm");

    body                = base_request();
    body["temperature"] = 1.01;
    failures += check(api_param([&] { (void)parse(body); }) == "temperature",
                      "Anthropic temperature range was not enforced");
    body          = base_request();
    body["top_k"] = 21;
    failures += check(api_param([&] { (void)parse(body); }) == "top_k",
                      "Engine top_k range was not enforced");
    body                  = base_request();
    body["output_config"] = Json{{"format", Json{{"type", "json_schema"}}}};
    failures += check(api_param([&] { (void)parse(body); }) == "output_config.format.schema",
                      "malformed schema was accepted");
    body              = base_request();
    body["container"] = "container_1";
    failures += check(api_code([&] { (void)parse(body); }) == "container_not_supported",
                      "container execution was silently ignored");
    return failures;
}

int test_message_normalization() {
    Json body        = base_request();
    body["thinking"] = Json{{"type", "disabled"}};
    body["system"] =
        Json::array({Json{{"type", "text"}, {"text", "A"}},
                     Json{{"type", "text"},
                          {"text", "B"},
                          {"cache_control", Json{{"type", "ephemeral"}, {"ttl", "1h"}}}}});
    body["messages"] =
        Json::array({Json{{"role", "user"},
                          {"content", Json::array({Json{{"type", "text"}, {"text", "one"}},
                                                   Json{{"type", "text"}, {"text", "two"}}})}},
                     Json{{"role", "user"}, {"content", "three"}},
                     Json{{"role", "assistant"}, {"content", "prefix"}}});

    const GenerationRequest request = parse(body).generation;
    int failures                    = check(request.messages.size() == 3 &&
                                                request.messages[0].role == infernix::ChatRole::System &&
                                                request.messages[1].role == infernix::ChatRole::User &&
                                                request.messages[1].content.size() == 3,
                                            "consecutive Anthropic roles were not merged by block concatenation");
    failures += check(request.messages[0].content[0].text == "A" &&
                          request.messages[0].content[1].text == "B" &&
                          request.messages[1].content[0].text == "one" &&
                          request.messages[1].content[1].text == "two" &&
                          request.messages[1].content[2].text == "three",
                      "message normalization inserted or removed text");
    failures +=
        check(request.continuation == infernix::PromptContinuationMode::ContinueFinalAssistant,
              "final assistant text did not select continuation mode");
    const infernix::PromptInput translated = prompt(request);
    failures += check(translated.options.continuation ==
                              infernix::PromptContinuationMode::ContinueFinalAssistant &&
                          translated.context_cache.markers.size() == 1 &&
                          translated.context_cache.markers[0].location ==
                              infernix::PromptCacheMarkerLocation::LeadingInstructionBoundary,
                      "assistant continuation or system block cache boundary was lost");
    return failures;
}

int test_attribution_system_block() {
    const auto attributed = [](std::string fingerprint) {
        Json body        = base_request();
        body["thinking"] = Json{{"type", "disabled"}};
        body["system"]   = Json::array(
            {Json{{"type", "text"},
                    {"text", "x-anthropic-billing-header: cc_version=2.1; cch=" + fingerprint}},
               Json{{"type", "text"},
                    {"text", "real system"},
                    {"cache_control", Json{{"type", "ephemeral"}}}}});
        return body;
    };

    const Json first_body                 = attributed(".89c");
    const GenerationRequest first_request = parse(first_body).generation;
    const infernix::PromptInput first       = prompt(first_request);
    const infernix::PromptInput second      = prompt(parse(attributed(".8f7")).generation);
    const infernix::PromptInput counted =
        prompt(parse_anthropic_count_tokens_request(first_body).generation);

    const auto has_real_system = [](const infernix::PromptInput& input) {
        return input.messages.size() == 2 &&
               input.messages.front().role == infernix::ChatRole::System &&
               input.messages.front().parts.size() == 1 &&
               input.messages.front().parts.front().text == "real system" &&
               input.context_cache.markers.size() == 1 &&
               input.context_cache.markers.front().location ==
                   infernix::PromptCacheMarkerLocation::LeadingInstructionBoundary;
    };
    int failures =
        check(has_real_system(first) && has_real_system(second) && has_real_system(counted) &&
                  first.messages.front().parts.front().text ==
                      second.messages.front().parts.front().text &&
                  first.context_cache.markers == second.context_cache.markers &&
                  first.context_cache.markers == counted.context_cache.markers,
              "Anthropic attribution changed model input or Count Tokens lowering");

    Json attribution_only        = base_request();
    attribution_only["thinking"] = Json{{"type", "disabled"}};
    attribution_only["system"]   = Json::array(
        {Json{{"type", "text"}, {"text", "x-anthropic-billing-header: cc_version=2.1"}}});
    const GenerationRequest without_empty = parse(attribution_only).generation;
    failures += check(without_empty.messages.size() == 1 &&
                          without_empty.messages.front().role == infernix::ChatRole::User,
                      "attribution-only system produced an empty model turn");

    Json non_first        = base_request();
    non_first["thinking"] = Json{{"type", "disabled"}};
    non_first["system"] =
        Json::array({Json{{"type", "text"}, {"text", "real system"}},
                     Json{{"type", "text"}, {"text", "x-anthropic-billing-header: user text"}}});
    const GenerationRequest preserved_non_first = parse(non_first).generation;
    failures += check(preserved_non_first.messages.size() == 2 &&
                          preserved_non_first.messages.front().content.size() == 2 &&
                          preserved_non_first.messages.front().content.back().text ==
                              "x-anthropic-billing-header: user text",
                      "non-first attribution-like system text was consumed");

    Json string_system                       = base_request();
    string_system["thinking"]                = Json{{"type", "disabled"}};
    string_system["system"]                  = "x-anthropic-billing-header: ordinary string";
    const GenerationRequest preserved_string = parse(string_system).generation;
    failures += check(preserved_string.messages.size() == 2 &&
                          preserved_string.messages.front().content.size() == 1 &&
                          preserved_string.messages.front().content.front().text ==
                              "x-anthropic-billing-header: ordinary string",
                      "ordinary string system content was consumed as attribution");
    return failures;
}

// A System array shaped like Claude Code's: its attribution block (unless `attribution` is false),
// then the given blocks, each with a cache_control breakpoint.
Json claude_code_request(std::vector<std::string> blocks, bool attribution = true) {
    Json body        = base_request();
    body["thinking"] = Json{{"type", "disabled"}};
    Json system      = Json::array();
    if (attribution) {
        system.push_back(Json{{"type", "text"},
                              {"text", "x-anthropic-billing-header: cc_version=2.1.296.55c; "
                                       "cc_entrypoint=cli; cch=0a1b2;"}});
    }
    for (std::string& text : blocks) {
        system.push_back(Json{{"type", "text"},
                              {"text", std::move(text)},
                              {"cache_control", Json{{"type", "ephemeral"}}}});
    }
    body["system"] = std::move(system);
    return body;
}

AnthropicMessagesRequest parse_with(const Json& body, ClaudeCodePrompt claude_code_prompt) {
    RequestLimits value      = limits();
    value.claude_code_prompt = claude_code_prompt;
    return parse_anthropic_messages_request(body, value);
}

// The leading System turn's text as the template receives it (its blocks joined as they are).
std::string system_text(const GenerationRequest& request) {
    std::string text;
    if (request.messages.empty() || request.messages.front().role != infernix::ChatRole::System) {
        return text;
    }
    for (const ContentPart& part : request.messages.front().content) { text += part.text; }
    return text;
}

// --claude-code-prompt: the default lowers Claude Code's System prompt as sent; local rewrites one
// that carries Claude Code's attribution block and identity line, identically on every turn, and
// leaves every other System value as sent.
int test_claude_code_prompt() {
    const std::string identity = "You are Claude Code, Anthropic's official CLI for Claude.";
    // A short synthetic body with one section of each kind the rewrite handles, in Claude Code's
    // order; the heading inside the memory example's code fence does not end the Memory section.
    const std::string body_text =
        "\nYou are an agent working with the user toward their goals in this repository.\n\n"
        "IMPORTANT: Assist with authorized security testing only; refuse destructive techniques.\n\n"
        "# Harness\n"
        " - Text you output outside of tool use is displayed to the user.\n\n"
        "When you use a pronoun for someone, use they/them.\n\n"
        "# Memory\n"
        "You have a persistent file-based memory at `C:\\mem\\dir\\`. This directory already "
        "exists.\n\n"
        "## Types\n"
        "```markdown\n"
        "# Example memory title\n"
        "```\n"
        "`user`: who the user is.\n\n"
        "# Environment\n"
        " - The most recent Claude models are the Claude 5 family.\n"
        " - Claude Code is available as a CLI in the terminal.\n"
        " - Fast mode for Claude Code uses Claude Opus with faster output.\n\n"
        "# Context management\n"
        "When the conversation grows long, some of it is summarized.";
    const Json body = claude_code_request({identity, body_text});

    const GenerationRequest as_sent = parse(body).generation;
    int failures = check(as_sent.messages.front().role == infernix::ChatRole::System &&
                             as_sent.messages.front().content.size() == 2 &&
                             as_sent.messages.front().content[0].text == identity &&
                             as_sent.messages.front().content[1].text == body_text,
                         "the default did not lower Claude Code's System prompt as sent");

    const GenerationRequest local         = parse_with(body, ClaudeCodePrompt::Local).generation;
    const std::vector<ContentPart>& parts = local.messages.front().content;
    if (local.messages.front().role != infernix::ChatRole::System || parts.size() != 2) {
        return failures + check(false, "the local rewrite changed the System block count");
    }
    failures += check(parts[0].cache_boundary_after && parts[1].cache_boundary_after,
                      "the local rewrite lost a System block's cache boundary");
    const std::string opening = "You are a coding agent running on a local open-weights model";
    failures += check(parts[0].text.starts_with(opening) &&
                          parts[0].text.find("Claude") == std::string::npos,
                      "the identity block did not become the local-agent preamble");
    // The other sections pass through, and the memory how-to becomes one paragraph keeping its path.
    const std::string kept_head =
        "\n\n# Harness\n - Text you output outside of tool use is displayed to the user.\n\n"
        "# Memory\nPersistent memory lives in `C:\\mem\\dir\\`";
    const std::string kept_tail =
        "\n\n# Context management\nWhen the conversation grows long, some of it is summarized.";
    const std::string& rewritten = parts[1].text;
    failures += check(rewritten.starts_with(kept_head) && rewritten.ends_with(kept_tail) &&
                          rewritten.size() > kept_head.size() + kept_tail.size() &&
                          rewritten
                                  .substr(kept_head.size(),
                                          rewritten.size() - kept_head.size() - kept_tail.size())
                                  .find('\n') == std::string::npos,
                      "the local rewrite did not produce the expected compact System body");
    const std::string whole = system_text(local);
    for (const char* removed :
         {"billing-header", "Claude Code", "Claude 5", "security testing", "pronoun", "Fast mode",
          "# Environment", "You are an agent working", "## Types", "```", "Example memory title"}) {
        failures += check(whole.find(removed) == std::string::npos,
                          std::string("the local rewrite kept Claude-only text: ") + removed);
    }
    // Every turn renders the same System prefix, and Count Tokens counts the prompt Messages runs.
    const GenerationRequest again = parse_with(body, ClaudeCodePrompt::Local).generation;
    const GenerationRequest counted =
        parse_anthropic_count_tokens_request(body, {.claude_code_prompt = ClaudeCodePrompt::Local})
            .generation;
    failures += check(system_text(again) == whole && system_text(counted) == whole &&
                          prompt(again).context_cache.markers == prompt(local).context_cache.markers,
                      "the local rewrite is not deterministic or Count Tokens counted another prompt");

    // Interactive shape, identity and body in one block: Environment keeps its other lines, and a
    // section the rewrite does not know passes unchanged.
    const GenerationRequest interactive =
        parse_with(claude_code_request({identity +
                                        "\n\n# Environment\nPrimary working directory: D:\\x\n"
                                        " - The most recent Claude models are the Claude 5 family."
                                        "\n\n# Plan mode\nPlan before you edit."}),
                   ClaudeCodePrompt::Local)
            .generation;
    const std::string interactive_text = system_text(interactive);
    failures += check(interactive.messages.front().content.size() == 1 &&
                          interactive_text.starts_with(opening) &&
                          interactive_text.ends_with("\n\n# Environment\nPrimary working directory: "
                                                     "D:\\x\n\n# Plan mode\nPlan before you edit.") &&
                          interactive_text.find("Claude") == std::string::npos,
                      "the interactive Claude Code prompt was not rewritten section by section");

    // Anything without both the attribution block and the identity line stays as sent.
    const auto unchanged = [&](const Json& request_body, const std::string& message) {
        const std::string sent = system_text(parse(request_body).generation);
        return check(!sent.empty() &&
                         system_text(parse_with(request_body, ClaudeCodePrompt::Local).generation) ==
                             sent,
                     message);
    };
    Json string_system        = base_request();
    string_system["thinking"] = Json{{"type", "disabled"}};
    string_system["system"]   = "You are a helpful assistant.";
    failures += unchanged(string_system, "an ordinary string System prompt was rewritten");
    failures += unchanged(claude_code_request({"You are a helpful assistant."}),
                          "a prompt with Claude Code's attribution but another identity was rewritten");
    failures += unchanged(claude_code_request({identity, body_text}, false),
                          "Claude Code's identity without its attribution block was rewritten");
    failures += unchanged(claude_code_request({"You are Claude Codex, another agent.", body_text}),
                          "an identity that only begins like Claude Code's was rewritten");
    return failures;
}

Json tool_use(std::string id, std::string name = "lookup") {
    return Json{{"type", "tool_use"},
                {"id", std::move(id)},
                {"name", std::move(name)},
                {"input", Json::object()}};
}

Json tool_result(std::string id, std::string content, bool is_error = false) {
    return Json{{"type", "tool_result"},
                {"tool_use_id", std::move(id)},
                {"content", std::move(content)},
                {"is_error", is_error}};
}

int test_tool_history() {
    Json body        = base_request();
    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", Json::array({tool_use("toolu_a"), tool_use("toolu_b")})}},
         Json{{"role", "user"},
              {"content", Json::array({Json{{"type", "tool_result"},
                                            {"tool_use_id", "toolu_b"},
                                            {"content", "result B"},
                                            {"is_error", true},
                                            {"cache_control", Json{{"type", "ephemeral"}}}},
                                       tool_result("toolu_a", "result A"),
                                       Json{{"type", "text"}, {"text", "continue"}}})}}});
    const GenerationRequest normalized          = parse(body).generation;
    int failures                                = check(normalized.messages.size() == 4 &&
                                                            normalized.messages[0].role == infernix::ChatRole::Assistant &&
                                                            normalized.messages[1].role == infernix::ChatRole::Tool &&
                                                            normalized.messages[1].tool_call_id == "toolu_a" &&
                                                            normalized.messages[1].content[0].text == "result A" &&
                                                            normalized.messages[2].role == infernix::ChatRole::Tool &&
                                                            normalized.messages[2].tool_call_id == "toolu_b" &&
                                                            normalized.messages[2].content[0].text == "result B" &&
                                                            normalized.messages[2].tool_result_is_error &&
                                                            normalized.messages[2].cache_boundary_after &&
                                                            normalized.messages[2].cache_boundary_after->kind ==
                                                                infernix::PromptCacheMarkerKind::SharedStablePrefix &&
                                                            normalized.messages[3].role == infernix::ChatRole::User &&
                                                            normalized.messages[3].content[0].text == "continue",
                                                        "valid out-of-order tool results were not associated by ID");
    const infernix::PromptInput normalized_prompt = prompt(normalized);
    failures += check(normalized_prompt.messages[2].parts.size() == 2 &&
                          normalized_prompt.messages[2].parts[0].text == "[tool_error]\n" &&
                          normalized_prompt.messages[2].parts[1].text == "result B",
                      "tool result error state did not follow its ID during normalization");

    body["messages"][1]["content"] =
        Json::array({tool_result("toolu_unknown", "wrong"), tool_result("toolu_a", "A")});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history" &&
                          api_param([&] { (void)parse(body); }) == "messages",
                      "unknown tool_result ID was accepted");

    body["messages"][1]["content"] =
        Json::array({tool_result("toolu_a", "first"), tool_result("toolu_a", "duplicate")});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "duplicate tool_result ID was accepted");

    body["messages"][1]["content"] = Json::array({tool_result("toolu_a", "only one")});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "missing tool_result was accepted");

    body["messages"] =
        Json::array({Json{{"role", "assistant"}, {"content", Json::array({tool_use("toolu_a")})}},
                     Json{{"role", "user"},
                          {"content", Json::array({Json{{"type", "text"}, {"text", "before"}},
                                                   tool_result("toolu_a", "result")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "tool_result after ordinary user content was accepted");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"}, {"content", Json::array({tool_use("toolu_a")})}},
         Json{{"role", "user"}, {"content", "before"}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_a", "result")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "same-role message joining bypassed tool_result ordering");

    body["messages"] = Json::array(
        {Json{{"role", "user"},
              {"content", Json::array({tool_result("toolu_external", "imported"),
                                       Json{{"type", "text"}, {"text", "continue"}}})}}});
    const GenerationRequest truncated = parse(body).generation;
    failures += check(truncated.messages.size() == 2 &&
                          truncated.messages[0].role == infernix::ChatRole::Tool &&
                          truncated.messages[0].tool_call_id == "toolu_external" &&
                          truncated.messages[1].role == infernix::ChatRole::User,
                      "leading tool_result from a truncated history was rejected or reordered");

    // A final Assistant message is a prefill, and a prefill cannot end in an unanswered tool_use.
    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "first"}},
         Json{{"role", "assistant"},
              {"content", Json::array({Json{{"type", "text"}, {"text", "calling"}},
                                       tool_use("toolu_trailing")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history" &&
                          api_param([&] { (void)parse(body); }) == "messages",
                      "trailing assistant tool_use was accepted as a prefill");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "first"}},
         Json{{"role", "assistant"}, {"content", "ordinary"}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_orphan", "late")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "orphan tool_result inside a visible history was accepted");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", Json::array({tool_use("toolu_same"), tool_use("toolu_same", "other")})}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_same", "result")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "duplicate tool_use ID was accepted");
    failures += check(api_code([&] { (void)parse_anthropic_count_tokens_request(body); }) ==
                          "invalid_tool_history",
                      "Count Tokens did not share Messages tool-history validation");
    return failures;
}

Json ordinary_tool(bool strict = false) {
    return Json{{"name", "weather"},
                {"description", "Get weather"},
                {"input_schema", Json{{"type", "object"},
                                      {"properties", Json{{"city", Json{{"type", "string"}}}}}}},
                {"input_examples", Json::array({Json{{"city", "Paris"}}})},
                {"strict", strict}};
}

int test_tools() {
    Json body                       = base_request();
    body["tools"]                   = Json::array({ordinary_tool()});
    body["tool_choice"]             = Json{{"type", "auto"}, {"disable_parallel_tool_use", false}};
    const GenerationRequest request = parse(body).generation;
    const infernix::PromptInput translated = prompt(request);
    const Json rendered                  = Json::parse(translated.options.tool_jsons.at(0));
    int failures = check(request.uses_tools() && rendered["function"]["name"] == "weather" &&
                             rendered["function"]["input_examples"].is_array(),
                         "Anthropic tool schema/examples did not reach the Qwen prompt");
    failures += check(request.constrains_tools() &&
                          !to_request_options(request, {}, semantics(request), true)
                               .output.preserve_special_tokens,
                      "ordinary Anthropic tools did not select constrained output");

    body["tools"] = Json::array({ordinary_tool(true)});
    failures += check(parse(body).generation.tools[0].strict, "strict reaches generation");
    body["tools"][0]["strict"] = "true";
    failures += check(api_param([&] { (void)parse(body); }) == "tools", "a non-boolean strict is refused");
    body["tools"][0]["strict"] = nullptr;
    failures += check(!parse(body).generation.tools[0].strict, "a null strict means not strict");
    body["tools"][0]["strict"] = true;
    body["tool_choice"]               = Json{{"type", "none"}, {"disable_parallel_tool_use", true}};
    body["tools"][0]["allowed_callers"] = Json::array({"code_execution"});
    const GenerationRequest disabled    = parse(body).generation;
    failures += check(!disabled.uses_tools() && prompt(disabled).options.tool_jsons.size() == 1,
                      "tool_choice:none keeps prompt declarations");
    body["tools"][0]["defer_loading"] = true;
    failures += check(api_code([&] { (void)parse(body); }) == "all_tools_deferred",
                      "a request whose only tool is deferred was accepted under tool_choice:none");

    body                = base_request();
    body["tools"]       = Json::array({ordinary_tool()});
    body["tool_choice"] = Json{{"type", "any"}};
    failures += check(parse(body).generation.tool_choice.mode == ToolChoiceMode::Required,
                      "any requires at least one invocation");
    body["tool_choice"] = Json{{"type", "tool"}, {"name", "weather"}};
    failures += check(parse(body).generation.tool_choice.mode == ToolChoiceMode::Required &&
                          parse(body).generation.tool_choice.allowed_names ==
                              std::vector<std::string>{"weather"},
                      "named choice requires the selected function");
    body["tool_choice"] = Json{{"type", "auto"}, {"disable_parallel_tool_use", true}};
    failures += check(!parse(body).generation.tool_choice.parallel,
                      "disable_parallel_tool_use reaches generation");

    body          = base_request();
    body["tools"] = Json::array({Json{{"type", "web_search_20250305"}, {"name", "web_search"}}});
    failures += check(api_code([&] { (void)parse(body); }) == "anthropic_tools_not_supported",
                      "Anthropic-provided tool was treated as a user tool");
    body["tool_choice"] = Json{{"type", "none"}};
    failures += check(api_code([&] { (void)parse(body); }).empty(),
                      "inactive Anthropic-provided tool unnecessarily blocked generation");

    body          = base_request();
    body["tools"] = Json::array({ordinary_tool(), ordinary_tool()});
    failures += check(api_param([&] { (void)parse(body); }) == "tools",
                      "duplicate tool names were accepted");
    return failures;
}

int test_prompt_object_order() {
    const auto parse_raw = [](std::string_view input) {
        const std::string body =
            R"({"model":"claude-local","thinking":{"type":"disabled"},"tools":[{"name":"probe","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}}}],"messages":[{"role":"user","content":"run probe"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_order","name":"probe","input":)" +
            std::string(input) +
            R"(}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_order","content":"done"}]}],"max_tokens":32})";
        return parse(Json::parse(body)).generation;
    };

    const std::string original      = R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})";
    const GenerationRequest request = parse_raw(original);
    const auto assistant =
        std::find_if(request.messages.begin(), request.messages.end(),
                     [](const ChatTurn& turn) { return turn.role == infernix::ChatRole::Assistant; });
    const infernix::PromptInput translated = prompt(request);
    const std::string expected_tool =
        R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}},"strict":false}})";
    int failures = check(assistant != request.messages.end() && assistant->tool_calls.size() == 1 &&
                             assistant->tool_calls.front().arguments_json == original &&
                             translated.options.tool_jsons.size() == 1 &&
                             translated.options.tool_jsons.front() == expected_tool,
                         "Anthropic request changed prompt-bearing object member order");

    const GenerationRequest reordered =
        parse_raw(R"({"alpha":{"yankee":2,"bravo":true},"zeta":"last"})");
    const auto reordered_assistant =
        std::find_if(reordered.messages.begin(), reordered.messages.end(),
                     [](const ChatTurn& turn) { return turn.role == infernix::ChatRole::Assistant; });
    failures +=
        check(reordered_assistant != reordered.messages.end() &&
                  reordered_assistant->tool_calls.size() == 1 &&
                  reordered_assistant->tool_calls.front().arguments_json != original,
              "Anthropic request treated reordered tool input as the original representation");
    return failures;
}

int test_thinking_and_count_tokens() {
    Json body = base_request();
    body["thinking"] =
        Json{{"type", "enabled"}, {"budget_tokens", 1024}, {"display", "summarized"}};
    const GenerationRequest enabled = parse(body).generation;
    ServeOptions server;
    server.default_thinking_budget = 777;
    const infernix::RequestOptions options =
        to_request_options(enabled, server, semantics(enabled), true);
    int failures = check(enabled.enable_thinking == true && enabled.thinking_budget == 1024 &&
                             options.execution.thinking.budget == 1024,
                         "request Thinking budget did not reach Engine options");

    // A budget at or above max_tokens is accepted and passed on; the output limit ends thinking
    // first. Qwen Code sends a fixed budget while shrinking max_tokens to the context left.
    for (const int budget : {4096, 128000}) {
        body["thinking"]["budget_tokens"] = budget;
        const GenerationRequest over      = parse(body).generation;
        failures += check(over.enable_thinking == true && over.max_tokens == 4096 &&
                              over.thinking_budget == static_cast<std::uint32_t>(budget),
                          "Thinking budget at or above max_tokens was rejected or altered");
    }
    body["thinking"]["budget_tokens"] = 1023;
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "Thinking budget below 1024 was accepted");
    body["thinking"] = Json{{"type", "future"}};
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "unknown Thinking mode defaulted to enabled");
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "omitted"}};
    const AnthropicMessagesRequest hidden = parse(body);
    failures += check(hidden.hide_thinking && hidden.generation.enable_thinking == true &&
                          !parse(base_request()).hide_thinking,
                      "display:omitted was not parsed as hidden adaptive Thinking");
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "summarized"}};
    failures += check(!parse(body).hide_thinking, "display:summarized hid the Thinking text");
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "future"}};
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "unknown Thinking display was accepted");
    body["thinking"] = Json{{"type", "disabled"}, {"display", "omitted"}};
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "display was accepted with disabled Thinking");
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "omitted"}};

    body["max_tokens"]                        = 0;
    body["temperature"]                       = "ignored for counting";
    body["output_config"]                     = Json{{"format", Json{{"type", "json_schema"}}}};
    const AnthropicCountTokensRequest counted = parse_anthropic_count_tokens_request(body);
    failures += check(counted.generation.enable_thinking == true,
                      "Count Tokens did not share prompt-affecting Thinking parsing");

    body             = base_request();
    body["thinking"] = Json{{"type", "disabled"}};
    body["messages"].push_back(Json{{"role", "assistant"}, {"content", "prefix"}});
    failures += check(api_code([&] { (void)semantics(parse(body).generation); }).empty(),
                      "disabled-Thinking assistant prefill was rejected");
    body.erase("thinking");
    failures += check(api_code([&] { (void)semantics(parse(body).generation); }).empty(),
                      "Thinking-on assistant prefill was rejected at capability resolution");
    return failures;
}

int test_thinking_history_transport_metadata() {
    Json body = base_request();
    body["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "before"}},
                     Json{{"role", "assistant"},
                          {"content", Json::array({Json{{"type", "thinking"},
                                                        {"thinking", "thought"},
                                                        {"signature", "opaque-from-stopped-serve"}},
                                                   Json{{"type", "text"}, {"text", "answer"}}})}},
                     Json{{"role", "user"}, {"content", "after"}}});
    const GenerationRequest accepted = parse(body).generation;
    int failures                     = check(accepted.messages.size() == 3 &&
                                                 accepted.messages[1].reasoning_content == "thought" &&
                                                 accepted.messages[1].content[0].text == "answer",
                                             "Thinking history with opaque transport metadata was not lowered");

    Json changed_signature                                      = body;
    changed_signature["messages"][1]["content"][0]["signature"] = "different";
    const GenerationRequest changed_metadata = parse(changed_signature).generation;
    failures += check(changed_metadata.messages[1].reasoning_content == "thought",
                      "opaque Thinking signature changed prompt semantics");

    Json missing_signature = body;
    missing_signature["messages"][1]["content"][0].erase("signature");
    const GenerationRequest missing_metadata = parse(missing_signature).generation;
    failures += check(missing_metadata.messages[1].reasoning_content == "thought",
                      "missing Thinking signature blocked visible history");

    Json structured_signature                                      = body;
    structured_signature["messages"][1]["content"][0]["signature"] = Json{{"provider", "external"}};
    const GenerationRequest structured_metadata = parse(structured_signature).generation;
    failures += check(structured_metadata.messages[1].reasoning_content == "thought",
                      "non-semantic Thinking metadata was interpreted by request lowering");

    Json changed_thinking                                     = body;
    changed_thinking["messages"][1]["content"][0]["thinking"] = "changed";
    const GenerationRequest changed_semantics                 = parse(changed_thinking).generation;
    failures += check(changed_semantics.messages[1].reasoning_content == "changed",
                      "visible Thinking text did not own prompt semantics");

    const AnthropicCountTokensRequest counted =
        parse_anthropic_count_tokens_request(missing_signature);
    failures += check(counted.generation.messages[1].reasoning_content == "thought",
                      "Count Tokens assigned semantics to Thinking signature metadata");
    return failures;
}

int test_content_and_cache_hints() {
    Json body                       = base_request();
    body["cache_control"]           = Json{{"type", "ephemeral"}, {"ttl", "5m"}};
    body["messages"]                = Json::array({Json{
                       {"role", "user"},
                       {"content",
                        Json::array({Json{{"type", "text"},
                                          {"text", "look"},
                                          {"cache_control", Json{{"type", "ephemeral"}}}},
                                     Json{{"type", "image"},
                                          {"source", Json{{"type", "url"}, {"url", "https://example/image.png"}}},
                                          {"transformations", Json{{"oversized_image", "error"}}}}})}}});
    const GenerationRequest request = parse(body).generation;
    int failures                    = check(request.messages[0].content[1].cache_boundary_after &&
                                                infernix::has_shared_candidate_evidence(
                                 request.messages[0].content[1].cache_boundary_after->evidence,
                                 infernix::SharedCandidateEvidence::RequestedAutomatic) &&
                                                request.media_item_count() == 1 &&
                                                request.messages[0].content[1].image_resize_policy ==
                                                    infernix::ImageResizePolicy::RejectOversized,
                                            "automatic caching or image transformation policy was lost");
    const infernix::PromptInput translated = prompt(request);
    failures += check(translated.context_cache.markers.size() == 2 &&
                          translated.context_cache.markers[1].location ==
                              infernix::PromptCacheMarkerLocation::MessagePartBoundary,
                      "message-part cache boundary was not represented in PromptInput");

    body                           = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "image"}, {"source", Json{{"type", "file"}, {"file_id", "file_1"}}}}});
    failures += check(api_code([&] { (void)parse(body); }) == "files_not_supported",
                      "Files image source was flattened or ignored");
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "document"}, {"source", Json::object()}}});
    failures += check(api_code([&] { (void)parse(body); }) == "documents_not_supported",
                      "document citation semantics were silently discarded");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                            = "answer";
    outcome.reasoning                       = "thought";
    outcome.prompt_tokens                   = 100;
    outcome.completion_tokens               = 12;
    outcome.reasoning_tokens                = 5;
    outcome.finish_reason                   = infernix::FinishReason::StopString;
    outcome.matched_stop_string             = "STOP";
    outcome.reused_prompt_tokens            = 60;
    return outcome;
}

int test_aggregate_and_errors() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_test", "claude-local");
    GenerationOutcome outcome = sample_outcome();
    const Json response       = Json::parse(make_anthropic_messages_response(identity, outcome));
    int failures =
        check(response["content"].size() == 2 && response["content"][0]["type"] == "thinking" &&
                  response["content"][0]["signature"] == identity.message_id &&
                  response["stop_reason"] == "stop_sequence" && response["stop_sequence"] == "STOP",
              "aggregate Thinking or stop-sequence presentation is incomplete");
    failures += check(response["usage"]["input_tokens"] == 40 &&
                          response["usage"]["cache_read_input_tokens"] == 60 &&
                          response["usage"]["cache_creation_input_tokens"].is_null() &&
                          response["usage"]["output_tokens_details"]["thinking_tokens"] == 5,
                      "aggregate cache/reasoning usage was fabricated or lost");

    // With the prefix cache, the tokens this request made reusable leave input_tokens for
    // cache_creation_input_tokens, as in the Anthropic API (the three sum to the prompt).
    outcome.cache_created_prompt_tokens = 32;
    const Json created = Json::parse(make_anthropic_messages_response(identity, outcome));
    failures += check(created["usage"]["input_tokens"] == 8 && created["usage"]["cache_read_input_tokens"] == 60 &&
                          created["usage"]["cache_creation_input_tokens"] == 32,
                      "cache creation usage was not split out of the uncached input");
    outcome.cache_created_prompt_tokens = 500;
    failures += check(Json::parse(make_anthropic_messages_response(identity, outcome))["usage"]
                                                                                         ["cache_creation_input_tokens"] == 40,
                      "cache creation usage was not clamped to the uncached input");
    outcome.cache_created_prompt_tokens.reset();

    outcome.prompt_tokens                   = 10;
    outcome.reused_prompt_tokens            = 1000;
    const Json clamped = Json::parse(make_anthropic_messages_response(identity, outcome));
    failures += check(clamped["usage"]["input_tokens"] == 0 &&
                          clamped["usage"]["cache_read_input_tokens"] == 10,
                      "cache-read usage was not clamped to the prompt token count");

    outcome               = {};
    outcome.finish_reason = infernix::FinishReason::ContextCapacity;
    const Json empty      = Json::parse(make_anthropic_messages_response(identity, outcome));
    failures +=
        check(empty["content"].empty() && empty["stop_reason"] == "model_context_window_exceeded",
              "empty output was fabricated or context capacity was misclassified");

    ApiError overloaded;
    overloaded.status         = 429;
    overloaded.code           = "server_overloaded";
    overloaded.message        = "full";
    const ApiError normalized = normalize_anthropic_error(overloaded);
    const Json error          = Json::parse(make_anthropic_error_body(overloaded, "req_error"));
    failures += check(normalized.status == 529 && normalized.type == "overloaded_error" &&
                          error["request_id"] == "req_error" &&
                          error["error"]["type"] == "overloaded_error",
                      "Anthropic overload or request-id error mapping is wrong");
    ApiError schema_error;
    schema_error.status        = 400;
    schema_error.code          = "unsupported_json_schema";
    schema_error.param         = "tools/1/input_schema/properties/date/format";
    schema_error.message       = "unsupported schema keyword: format";
    const Json schema_response = Json::parse(make_anthropic_error_body(schema_error, "req_error"));
    failures += check(schema_response["error"]["message"] ==
                          schema_error.param + ": " + schema_error.message,
                      "Anthropic schema error lost its request field location");
    ApiError overflow;
    overflow.code           = "context_length_exceeded";
    overflow.param          = "messages";
    overflow.message        = "This model's maximum context length is 128 tokens.";
    overflow.context_length = infernix::ContextLengthDetails{.prompt_tokens = 200, .max_context = 128};
    const Json overflow_response = Json::parse(make_anthropic_error_body(overflow, "req_error"));
    failures += check(overflow_response["error"]["type"] == "invalid_request_error" &&
                          overflow_response["error"]["message"] == "prompt is too long: 200 tokens > 128 maximum",
                      "Anthropic context overflow lost the wording Claude Code matches to compact");
    // The prompt's images together over the vision budget: the same matchable sentence, then what
    // the counts are.
    ApiError vision_overflow        = overflow;
    vision_overflow.context_length  = infernix::ContextLengthDetails{
         .prompt_tokens = 40'959, .max_context = 32'768, .exact = false, .vision = true};
    const Json vision_response = Json::parse(make_anthropic_error_body(vision_overflow, "req_error"));
    const std::string vision_message = vision_response["error"]["message"].get<std::string>();
    failures += check(vision_response["error"]["type"] == "invalid_request_error" &&
                          vision_message.starts_with("prompt is too long: more than 40959 tokens > 32768 maximum (") &&
                          vision_message.find("vision budget of 32768 tokens") != std::string::npos,
                      "Anthropic vision overflow lost the compaction wording or its images note: " + vision_message);
    return failures;
}

// llama.cpp-compatible `timings` (the Chat Completions semantics): prompt counts net of the cache
// hit and N-1 generation intervals, on the aggregate body and only the terminal message_delta.
int test_timings() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_timings", "claude-local");
    GenerationOutcome outcome                   = sample_outcome();
    outcome.timings.prompt_wall_seconds         = 0.0625;
    outcome.timings.generation_wall_seconds     = 0.125;
    outcome.speculative.drafted_tokens          = 9;
    outcome.speculative.accepted_tokens         = 6;
    const Json timings = Json::parse(make_anthropic_messages_response(identity, outcome))["timings"];
    int failures       = check(
        timings["cache_n"] == 60 && timings["prompt_n"] == 40 && timings["prompt_ms"] == 62.5 &&
            timings["prompt_per_second"] == 640.0 && timings["predicted_n"] == 12 &&
            timings["predicted_ms"] == 125.0 && timings["predicted_per_second"] == 88.0 &&
            timings["draft_n"] == 9 && timings["draft_n_accepted"] == 6,
        "aggregate Anthropic timings use the exact cache split and N-1 generation intervals");

    AnthropicMessagesStream stream(identity, 100);
    std::vector<std::string> events{stream.start()};
    // Braced lists evaluate in order: reasoning, then text, then the terminal events.
    for (std::vector<std::string> values :
         {stream.reasoning_delta("thought"), stream.content_delta("answer"), stream.finish(outcome)}) {
        events.insert(events.end(), std::make_move_iterator(values.begin()),
                      std::make_move_iterator(values.end()));
    }
    int timed = 0;
    for (const std::string& wire : events) {
        const Json event = parse_event(wire);
        if (!event.contains("timings")) { continue; }
        ++timed;
        failures += check(event["type"] == "message_delta" && event["timings"]["predicted_n"] == 12,
                          "Anthropic stream timings belong to the terminal message_delta");
    }
    failures += check(timed == 1, "an Anthropic stream carries exactly one timings object");

    outcome = {};
    const Json empty = Json::parse(make_anthropic_messages_response(identity, outcome))["timings"];
    failures += check(empty["prompt_per_second"] == 0.0 && empty["predicted_per_second"] == 0.0 &&
                          !empty.contains("draft_n"),
                      "zero-token Anthropic timings report finite zero rates without draft counts");
    return failures;
}

int test_tool_call_presentation() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_tool", "claude-local");
    GenerationOutcome outcome;
    outcome.text          = "I need one more check.";
    outcome.finish_reason = infernix::FinishReason::StopToken;
    outcome.constraint    = infernix::ConstraintObservation{
           .branch = infernix::ConstraintOutputBranch::Tools, .complete = true, .terminated = true};
    const std::string arguments = R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})";
    outcome.tool_calls.push_back(infernix::GeneratedToolCall{
        .name           = "Edit",
        .arguments_json = arguments,
    });

    const Json aggregate = Json::parse(make_anthropic_messages_response(identity, outcome));
    int failures =
        check(aggregate.at("stop_reason") == "tool_use" && aggregate.at("content").size() == 2 &&
                  aggregate.at("content").at(0).at("type") == "text" &&
                  aggregate.at("content").at(1).at("type") == "tool_use" &&
                  aggregate.at("content").at(1).at("name") == "Edit" &&
                  aggregate.at("content").at(1).at("input").dump() == arguments &&
                  !aggregate.at("content").at(1).at("input").contains("replace_all"),
              "aggregate Anthropic response changed ordered tool input");

    AnthropicMessagesStream stream(identity, 10);
    std::vector<std::string> events{stream.start()};
    std::vector<std::string> content = stream.content_delta(outcome.text);
    events.insert(events.end(), std::make_move_iterator(content.begin()),
                  std::make_move_iterator(content.end()));
    std::vector<std::string> terminal = stream.finish(outcome);
    events.insert(events.end(), std::make_move_iterator(terminal.begin()),
                  std::make_move_iterator(terminal.end()));

    bool saw_edit_start   = false;
    bool saw_arguments    = false;
    bool saw_tool_stop    = false;
    int constraint_events = 0;
    for (const std::string& wire : events) {
        const Json event = parse_event(wire);
        if (event.contains("constraint")) {
            ++constraint_events;
            failures += check(event["type"] == "message_delta" &&
                                  event["constraint"]["branch"] == "tools" &&
                                  event["constraint"]["terminated"] == true,
                              "Anthropic constraint state must accompany terminal message_delta");
        }
        if (event.at("type") == "content_block_start" &&
            event.at("content_block").at("type") == "tool_use") {
            saw_edit_start = event.at("content_block").at("name") == "Edit";
        } else if (event.at("type") == "content_block_delta" &&
                   event.at("delta").at("type") == "input_json_delta") {
            saw_arguments = event.at("delta").at("partial_json").get<std::string>() == arguments;
        } else if (event.at("type") == "message_delta") {
            saw_tool_stop = event.at("delta").at("stop_reason") == "tool_use";
        }
    }
    failures += check(saw_edit_start && saw_arguments && saw_tool_stop && constraint_events == 1,
                      "Anthropic stream did not terminate the recovered Edit as tool_use");
    outcome.finish_reason          = infernix::FinishReason::OutputLimit;
    outcome.constraint->terminated = false;
    const auto partial = Json::parse(make_anthropic_messages_response(identity, outcome));
    failures += check(partial["stop_reason"] == "max_tokens" && partial["content"].size() == 2 &&
                          partial["constraint"]["complete"] == true &&
                          partial["constraint"]["terminated"] == false,
                      "completed tool before truncation lost its call or stop reason");
    return failures;
}

// A constrained call streams its block as it is generated: the text block closes, the call opens
// (`server_tool_use` for a search tool) when named, its input arrives in `input_json_delta`
// fragments and the block stops when the call completes. finish repeats nothing, and the call's
// wire id is the one it streamed with.
int test_streamed_tool_calls() {
    const AnthropicResponseIdentity identity = make_anthropic_response_identity("req_streamed", "claude-local");
    const std::vector<AnthropicToolSearch> searches{{.name = "tool_search_tool_regex"}};
    AnthropicMessagesStream stream(identity, 10, false, searches);
    std::vector<std::string> events{stream.start()};
    const auto append = [&](std::vector<std::string> values) {
        events.insert(events.end(), std::make_move_iterator(values.begin()), std::make_move_iterator(values.end()));
    };
    append(stream.content_delta("Checking."));
    append(stream.tool_call_delta({.index = 0, .name = "Edit", .arguments = R"({"zeta":)"}));
    append(stream.tool_call_delta({.index = 0, .arguments = R"("last"})", .complete = true}));
    append(stream.tool_call_delta({.index = 1, .name = "tool_search_tool_regex", .arguments = R"({"pattern":"x"})",
                                   .complete = true}));
    GenerationOutcome outcome;
    outcome.text          = "Checking.";
    outcome.finish_reason = infernix::FinishReason::StopToken;
    outcome.tool_calls    = {{.name = "Edit", .arguments_json = R"({"zeta":"last"})"},
                             {.name = "tool_search_tool_regex", .arguments_json = R"({"pattern":"x"})"}};
    const std::vector<ToolCall> calls = stream.identify_calls(outcome);
    AnthropicResponseSegment segment{.outcome = &outcome, .calls = calls, .search_results = {}};
    segment.search_results.resize(calls.size());
    append(stream.finish(segment, outcome, std::nullopt));

    std::vector<std::string> sequence;
    std::string edit_input;
    std::string edit_id;
    std::string search_type;
    for (const std::string& wire : events) {
        const Json event = parse_event(wire);
        const std::string type = event.at("type");
        if (type == "content_block_start") {
            const Json& block = event.at("content_block");
            sequence.push_back("start:" + block.at("type").get<std::string>());
            if (block.at("type") == "tool_use") { edit_id = block.at("id"); }
            if (block.contains("name") && block.at("name") == "tool_search_tool_regex") { search_type = block.at("type"); }
        } else if (type == "content_block_delta" && event.at("delta").at("type") == "input_json_delta") {
            if (event.at("index") == 1) { edit_input += event.at("delta").at("partial_json").get<std::string>(); }
            sequence.push_back("input");
        } else if (type == "content_block_stop") {
            sequence.push_back("stop");
        } else if (type == "message_delta") {
            sequence.push_back(event.at("delta").at("stop_reason").get<std::string>());
        }
    }
    const std::vector<std::string> expected{"start:text", "stop",  "start:tool_use", "input", "input", "stop",
                                            "start:server_tool_use", "input", "stop", "tool_use"};
    int failures = check(sequence == expected && edit_input == R"({"zeta":"last"})" && search_type == "server_tool_use",
                         "a streamed Anthropic call lost its block lifecycle or was repeated at finish");
    failures += check(edit_id.starts_with("toolu_") && calls[0].id == edit_id && calls[1].id.starts_with("srvtoolu_"),
                      "a streamed call's wire id differs from its segment's");

    // A call the output limit cut closes its block before message_delta with what it streamed.
    AnthropicMessagesStream cut(identity, 10);
    (void)cut.start();
    (void)cut.tool_call_delta({.index = 0, .name = "Edit", .arguments = R"({"ze)"});
    GenerationOutcome cut_outcome;
    cut_outcome.finish_reason = infernix::FinishReason::OutputLimit;
    const std::vector<std::string> cut_events = cut.finish(cut_outcome);
    failures += check(cut_events.size() == 3 && parse_event(cut_events[0]).at("type") == "content_block_stop" &&
                          parse_event(cut_events[1]).at("delta").at("stop_reason") == "max_tokens",
                      "a cut streamed call did not close before the terminal message_delta");
    return failures;
}

int test_stream() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_stream", "claude-local");
    AnthropicMessagesStream stream(identity, 100);
    const infernix::GenerationStart warm_start{
        .prompt               = infernix::PromptSummary{.prompt_tokens = 100},
        .reused_prompt_tokens = 60,
    };
    std::vector<std::string> events{stream.start(warm_start)};
    auto append_events = [&](std::vector<std::string> values) {
        events.insert(events.end(), std::make_move_iterator(values.begin()),
                      std::make_move_iterator(values.end()));
    };
    append_events(stream.reasoning_delta("thought"));
    append_events(stream.content_delta("answer"));
    append_events(stream.finish(sample_outcome()));

    std::vector<std::string> types;
    bool saw_signature        = false;
    bool start_usage_is_exact = false;
    Json terminal_usage;
    for (const std::string& value : events) {
        const Json parsed = parse_event(value);
        types.push_back(parsed.at("type").get<std::string>());
        if (parsed.at("type") == "message_start") {
            start_usage_is_exact = parsed["message"]["usage"]["input_tokens"] == 40 &&
                                   parsed["message"]["usage"]["cache_read_input_tokens"] == 60;
        }
        if (parsed.at("type") == "content_block_delta" &&
            parsed["delta"]["type"] == "signature_delta") {
            const std::string signature = parsed["delta"]["signature"].get<std::string>();
            saw_signature               = signature == identity.message_id;
        }
        if (parsed.at("type") == "message_delta") { terminal_usage = parsed.at("usage"); }
    }
    int failures = check(types.front() == "message_start" && types.back() == "message_stop" &&
                             saw_signature && start_usage_is_exact,
                         "Anthropic stream lifecycle/signature/start usage is incomplete");
    const auto signature_position =
        std::find_if(events.begin(), events.end(), [](const auto& value) {
            return value.find("signature_delta") != std::string::npos;
        });
    const auto text_position = std::find_if(events.begin(), events.end(), [](const auto& value) {
        return value.find("text_delta") != std::string::npos;
    });
    failures += check(signature_position < text_position,
                      "Thinking signature was emitted after the text block began");
    const Json aggregate =
        Json::parse(make_anthropic_messages_response(identity, sample_outcome()));
    failures +=
        check(terminal_usage == aggregate.at("usage") && terminal_usage["input_tokens"] == 40 &&
                  terminal_usage["cache_read_input_tokens"] == 60,
              "terminal stream usage did not match aggregate cache usage");

    AnthropicMessagesStream cold_stream(identity, 25);
    (void)cold_stream.start(infernix::GenerationStart{
        .prompt               = infernix::PromptSummary{.prompt_tokens = 25},
        .reused_prompt_tokens = 0,
    });
    GenerationOutcome cold;
    cold.prompt_tokens                         = 25;
    cold.completion_tokens                     = 3;
    const std::vector<std::string> cold_events = cold_stream.finish(cold);
    const Json cold_delta = parse_event(cold_events.at(cold_events.size() - 2U));
    failures += check(cold_delta["usage"]["input_tokens"] == 25 &&
                          cold_delta["usage"]["cache_read_input_tokens"] == 0 &&
                          cold_delta["usage"]["output_tokens"] == 3,
                      "cold terminal stream usage was not cumulative and exact");

    AnthropicMessagesStream pre_admission_error(identity, 25);
    const Json provisional = parse_event(pre_admission_error.start());
    failures += check(provisional["message"]["usage"]["input_tokens"] == 25 &&
                          provisional["message"]["usage"]["cache_read_input_tokens"].is_null(),
                      "pre-admission stream error prefix fabricated cache usage");
    return failures;
}

int test_constrained_decoding() {
    int failures = 0;
    auto body    = base_request();
    for (const auto& value : {Json{{"choice", {"yes", "no"}}}, Json{{"regex", "[a-z]+"}}}) {
        body["structured_outputs"] = value;
        const auto request         = parse(body).generation;
        const auto constraint =
            to_request_options(request, {}, semantics(request), true).constraint;
        failures += check(constraint == (value.contains("choice")
                                             ? infernix::OutputConstraint::choice({"yes", "no"})
                                             : infernix::OutputConstraint::regex("[a-z]+")),
                          "Anthropic choice/regex lost in Engine translation");
    }
    body["structured_outputs"] = Json{{"grammar", "root ::= \"yes\""}};
    const auto request         = parse(body);
    failures +=
        check(to_request_options(request.generation, {}, semantics(request.generation), true)
                      .constraint->source == "root ::= \"yes\"",
              "Anthropic GBNF extension was lost in Engine translation");
    body["tools"] = Json::array({ordinary_tool()});
    failures += check(api_param([&] { (void)parse(body); }) == "structured_outputs.grammar",
                      "grammar admitted active tools");
    body["tool_choice"] = Json{{"type", "none"}};
    failures += check(parse(body).generation.constraint->source == "root ::= \"yes\"",
                      "inactive tools blocked grammar");
    body["stop_sequences"] = Json::array({"yes"});
    failures += check(api_param([&] { (void)parse(body); }) == "structured_outputs.grammar",
                      "grammar admitted custom stop sequences");
    body = base_request();
    body["output_config"] =
        Json{{"format", {{"type", "json_schema"}, {"schema", {{"type", "object"}}}}}};
    const auto schema_request = parse(body).generation;
    failures += check(schema_request.constraint->kind == infernix::OutputConstraintKind::JsonSchema &&
                          schema_request.constraint_param == "output_config.format.schema",
                      "Anthropic JSON schema source lost");
    body["tools"] = Json::array({Json{{"name", "lookup"}, {"input_schema", {{"type", "object"}}}}});
    const auto combined = parse(body).generation;
    failures += check(combined.constraint && combined.uses_tools() &&
                          combined.tools[0].schema_param == "tools/0/input_schema",
                      "Anthropic JSON/tool composition or diagnostic origin lost");
    body["structured_outputs"] = Json{{"grammar", "root ::= \"x\""}};
    failures += check(api_param([&] { (void)parse(body); }) == "structured_outputs.grammar",
                      "Anthropic conflicting output formats accepted");
    return failures;
}

// Every remainder class of the Base64 codec, plus multi-byte UTF-8 and control bytes.
int test_hidden_reasoning_signature() {
    int failures = 0;
    for (const std::string reasoning :
         {std::string(), std::string("a"), std::string("ab"), std::string("abc"),
          std::string("abcd"),
          std::string("na\xC3\xAFve \xE2\x80\x94 \xE6\x80\x9D\xE8\x80\x83\n\t\"quoted\"") +
              std::string(1, '\0') + "nul"}) {
        const std::string signature = encode_thinking_signature(reasoning);
        failures += check(decode_thinking_signature(signature) == reasoning,
                          "hidden-reasoning signature did not round-trip");
    }
    // RFC 4648 test vectors pin the alphabet and padding independently of the decoder.
    failures += check(encode_thinking_signature("foobar") == "ninfer-reasoning.v1:Zm9vYmFy" &&
                          encode_thinking_signature("fooba") == "ninfer-reasoning.v1:Zm9vYmE=" &&
                          encode_thinking_signature("foob") == "ninfer-reasoning.v1:Zm9vYg==",
                      "hidden-reasoning signature is not standard Base64");
    failures +=
        check(!decode_thinking_signature("msg_0123456789abcdef") &&
                  !decode_thinking_signature("") && !decode_thinking_signature("EuYBCkQIARgCKkD"),
              "a foreign signature was interpreted as hidden reasoning");
    for (const char* malformed :
         {"ninfer-reasoning.v1:Zm9", "ninfer-reasoning.v1:Zm9v!mFy", "ninfer-reasoning.v1:Zg=A",
          "ninfer-reasoning.v1:=m9v", "ninfer-reasoning.v1:Zm9vYg==Zm9v"}) {
        bool threw = false;
        try {
            (void)decode_thinking_signature(malformed);
        } catch (const std::invalid_argument&) { threw = true; }
        failures += check(threw, std::string("malformed signature was accepted: ") + malformed);
    }
    return failures;
}

int test_hidden_reasoning_round_trip() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_hidden", "claude-local");
    const GenerationOutcome outcome = sample_outcome();

    const Json aggregate =
        Json::parse(make_anthropic_messages_response(identity, outcome, /*hide_thinking=*/true));
    const Json& block = aggregate["content"][0];
    int failures = check(block["type"] == "thinking" && block["thinking"] == "" &&
                             decode_thinking_signature(block["signature"].get<std::string>()) ==
                                 outcome.reasoning &&
                             aggregate["content"][1]["text"] == "answer",
                         "aggregate hidden Thinking leaked text or lost its signature");

    AnthropicMessagesStream stream(identity, 100, /*hide_thinking=*/true);
    std::vector<std::string> events{stream.start()};
    const auto append = [&](std::vector<std::string> values) {
        events.insert(events.end(), std::make_move_iterator(values.begin()),
                      std::make_move_iterator(values.end()));
    };
    append(stream.reasoning_delta("tho"));
    append(stream.reasoning_delta("ught"));
    append(stream.content_delta("answer"));
    append(stream.finish(outcome));
    bool saw_thinking_delta = false;
    std::optional<std::string> streamed;
    for (const std::string& wire : events) {
        const Json parsed = parse_event(wire);
        if (parsed.at("type") != "content_block_delta") { continue; }
        if (parsed["delta"]["type"] == "thinking_delta") { saw_thinking_delta = true; }
        if (parsed["delta"]["type"] == "signature_delta") {
            streamed = decode_thinking_signature(parsed["delta"]["signature"].get<std::string>());
        }
    }
    failures += check(!saw_thinking_delta && streamed == outcome.reasoning,
                      "streamed hidden Thinking leaked text or lost its signature");

    // The client returns the block it received; lowering restores the reasoning from it.
    Json body        = base_request();
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "omitted"}};
    body["messages"].push_back(
        Json{{"role", "assistant"},
             {"content", Json::array({block, Json{{"type", "text"}, {"text", "answer"}}})}});
    body["messages"].push_back(Json{{"role", "user"}, {"content", "next"}});
    const auto assistant_reasoning = [](const GenerationRequest& request) {
        const auto assistant = std::ranges::find_if(request.messages, [](const ChatTurn& turn) {
            return turn.role == infernix::ChatRole::Assistant;
        });
        return assistant == request.messages.end() ? std::optional<std::string>()
                                                   : assistant->reasoning_content;
    };
    failures += check(assistant_reasoning(parse(body).generation) == outcome.reasoning,
                      "hidden reasoning was not restored from the returned block");

    // Visible text wins, a foreign signature stays metadata, a corrupt Infernix one is a 400.
    body["messages"][1]["content"][0] =
        Json{{"type", "thinking"}, {"thinking", "visible"}, {"signature", block["signature"]}};
    failures += check(assistant_reasoning(parse(body).generation) == "visible",
                      "visible Thinking text lost to its signature");
    body["messages"][1]["content"][0] =
        Json{{"type", "thinking"}, {"thinking", ""}, {"signature", "EuYBCkQIARgCKkD"}};
    failures += check(assistant_reasoning(parse(body).generation) == std::string(),
                      "a foreign signature produced reasoning");
    body["messages"][1]["content"][0] =
        Json{{"type", "thinking"}, {"thinking", ""}, {"signature", "ninfer-reasoning.v1:Zm9"}};
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_thinking_signature",
                      "a corrupt hidden-reasoning signature was accepted");
    return failures;
}

// Deferred tools (defer_loading) and tool search.

Json deferred_tool(std::string name, std::string description) {
    return Json{{"name", std::move(name)},
                {"description", std::move(description)},
                {"input_schema",
                 Json{{"type", "object"},
                      {"properties",
                       Json{{"city", Json{{"type", "string"}, {"description", "City name"}}}}}}},
                {"defer_loading", true}};
}

Json regex_search_tool() {
    return Json{{"type", "tool_search_tool_regex_20251119"}, {"name", "tool_search_tool_regex"}};
}

std::string message_text(const infernix::ChatMessage& message) {
    std::string text;
    for (const infernix::MessagePart& part : message.parts) { text += part.text; }
    return text;
}

// Whether the first `count` messages of two prompts render the same: same roles, text, reasoning
// and calls. The Engine renders a PromptInput from these and the tool declarations alone.
bool same_messages(const infernix::PromptInput& left, const infernix::PromptInput& right,
                   std::size_t count) {
    if (left.messages.size() < count || right.messages.size() < count) { return false; }
    for (std::size_t index = 0; index < count; ++index) {
        const infernix::ChatMessage& a = left.messages[index];
        const infernix::ChatMessage& b = right.messages[index];
        if (a.role != b.role || message_text(a) != message_text(b) ||
            a.parts.size() != b.parts.size() || a.reasoning_content != b.reasoning_content ||
            a.tool_calls.size() != b.tool_calls.size()) {
            return false;
        }
        for (std::size_t call = 0; call < a.tool_calls.size(); ++call) {
            if (a.tool_calls[call].name != b.tool_calls[call].name ||
                a.tool_calls[call].arguments_json != b.tool_calls[call].arguments_json) {
                return false;
            }
        }
    }
    return true;
}

int test_deferred_tool_declarations() {
    Json listed     = base_request();
    listed["tools"] = Json::array({ordinary_tool(), regex_search_tool()});
    Json deferred   = listed;
    deferred["tools"].push_back(deferred_tool("get_forecast", "Weather forecast for a city"));

    const AnthropicMessagesRequest with    = parse(deferred);
    const infernix::PromptInput without_prompt = prompt(parse(listed).generation);
    const infernix::PromptInput with_prompt    = prompt(with.generation);
    int failures = check(with_prompt.options.tool_jsons == without_prompt.options.tool_jsons &&
                             same_messages(with_prompt, without_prompt, 1) &&
                             without_prompt.options.deferred_tool_jsons.empty() &&
                             with_prompt.options.deferred_tool_jsons.size() == 1 &&
                             Json::parse(with_prompt.options.deferred_tool_jsons[0])["function"]
                                     ["name"] == "get_forecast",
                         "a deferred tool changed the prompt or missed the output contract");
    const Json search = Json::parse(with_prompt.options.tool_jsons.at(1));
    failures += check(search["function"]["name"] == "tool_search_tool_regex" &&
                          search["function"]["parameters"]["required"] == Json::array({"pattern"}) &&
                          with.tool_searches.size() == 1 &&
                          with.tool_searches[0].name == "tool_search_tool_regex" &&
                          with.tool_searches[0].variant == AnthropicToolSearchVariant::Regex,
                      "the regex tool search tool was not declared as a listed function");
    failures += check(tool_declaration_params(with.generation) ==
                          std::vector<std::string>{"tools/0/input_schema", "tools/1/input_schema",
                                                   "tools/2/input_schema"},
                      "Engine tool-schema errors no longer map to their request tools");

    Json none           = deferred;
    none["tool_choice"] = Json{{"type", "none"}};
    const infernix::PromptInput none_prompt = prompt(parse(none).generation);
    failures += check(none_prompt.options.tool_jsons == without_prompt.options.tool_jsons &&
                          none_prompt.options.deferred_tool_jsons.size() == 1,
                      "tool_choice:none changed which tools the prompt lists");

    Json bm25     = base_request();
    bm25["tools"] = Json::array({ordinary_tool(), Json{{"type", "tool_search_tool_bm25_20251119"}},
                                 deferred_tool("get_forecast", "Weather forecast")});
    const AnthropicMessagesRequest bm25_request = parse(bm25);
    failures += check(bm25_request.tool_searches.size() == 1 &&
                          bm25_request.tool_searches[0].name == "tool_search_tool_bm25" &&
                          bm25_request.tool_searches[0].variant == AnthropicToolSearchVariant::Bm25,
                      "the BM25 tool search tool was not declared");

    Json all     = base_request();
    all["tools"] = Json::array({deferred_tool("a", "A"), deferred_tool("b", "B")});
    failures += check(api_code([&] { (void)parse(all); }) == "all_tools_deferred",
                      "a request with every tool deferred was accepted");
    Json search_deferred                      = deferred;
    search_deferred["tools"][1]["defer_loading"] = true;
    failures += check(api_code([&] { (void)parse(search_deferred); }) == "invalid_deferred_tool",
                      "a deferred tool search tool was accepted");
    Json cached                     = deferred;
    cached["tools"][2]["cache_control"] = Json{{"type", "ephemeral"}};
    failures += check(api_code([&] { (void)parse(cached); }) == "invalid_cache_control",
                      "a deferred tool with a cache breakpoint was accepted");
    Json renamed               = listed;
    renamed["tools"][1]["name"] = "search";
    failures += check(api_param([&] { (void)parse(renamed); }) == "tools",
                      "a tool search tool with another name was accepted");
    Json duplicate = listed;
    duplicate["tools"].push_back(Json{{"name", "tool_search_tool_regex"},
                                      {"input_schema", Json{{"type", "object"}}}});
    failures += check(api_param([&] { (void)parse(duplicate); }) == "tools",
                      "a user tool shadowing the search tool name was accepted");
    Json named           = deferred;
    named["tool_choice"] = Json{{"type", "tool"}, {"name", "get_forecast"}};
    failures += check(parse(named).generation.tool_choice.allowed_names ==
                          std::vector<std::string>{"get_forecast"},
                      "a named choice of a deferred tool was refused");
    return failures;
}

int test_client_tool_references() {
    Json body     = base_request();
    body["tools"] = Json::array({Json{{"name", "find_tools"},
                                      {"description", "Find tools by keyword"},
                                      {"input_schema", Json{{"type", "object"}}}},
                                 deferred_tool("get_forecast", "Weather forecast for a city")});
    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "Forecast for Paris?"}},
         Json{{"role", "assistant"},
              {"content", Json::array({tool_use("toolu_find", "find_tools")})}},
         Json{{"role", "user"},
              {"content",
               Json::array({Json{{"type", "tool_result"},
                                 {"tool_use_id", "toolu_find"},
                                 {"content",
                                  Json::array({Json{{"type", "text"}, {"text", "Found one."}},
                                               Json{{"type", "tool_reference"},
                                                    {"tool_name", "get_forecast"}},
                                               Json{{"type", "tool_reference"},
                                                    {"tool_name", "get_forecast"}}})}}})}}});
    const GenerationRequest request = parse(body).generation;
    int failures = check(request.messages.size() == 3 &&
                             request.messages[2].role == infernix::ChatRole::Tool &&
                             request.messages[2].content.size() == 1 &&
                             request.messages[2].content[0].text == "Found one." &&
                             request.messages[2].loaded_tools.size() == 1 &&
                             request.messages[2].loaded_tools[0].name == "get_forecast",
                         "tool_reference blocks did not load their tool once into the result");
    const infernix::PromptInput translated = prompt(request);
    const std::string loaded = render_loaded_tools(request.messages[2].loaded_tools);
    failures += check(translated.messages[2].parts.size() == 2 &&
                          translated.messages[2].parts[0].text == "Found one." &&
                          translated.messages[2].parts[1].text == "\n\n" + loaded &&
                          loaded.find("\"get_forecast\"") != std::string::npos &&
                          loaded.find("City name") != std::string::npos &&
                          loaded.starts_with("You now also have access to the following "
                                             "functions:\n\n<tools>\n") &&
                          loaded.ends_with("\n</tools>"),
                      "a loaded definition was not rendered after the tool result");

    body["messages"][2]["content"][0]["content"] = Json::array(
        {Json{{"type", "tool_reference"}, {"tool_name", "get_forecast"}}});
    const infernix::PromptInput only_reference = prompt(parse(body).generation);
    failures += check(only_reference.messages[2].parts.size() == 1 &&
                          only_reference.messages[2].parts[0].text == loaded,
                      "a reference-only result rendered an empty part or separator");

    body["messages"][2]["content"][0]["content"][0]["tool_name"] = "missing_tool";
    failures += check(api_code([&] { (void)parse(body); }) == "tool_reference_not_found",
                      "a reference to an undeclared tool was accepted");
    body["messages"][2]["content"] =
        Json::array({tool_result("toolu_find", "none"),
                     Json{{"type", "tool_reference"}, {"tool_name", "get_forecast"}}});
    failures += check(api_code([&] { (void)parse(body); }) == "content_block_not_supported",
                      "a tool_reference outside a tool_result was accepted");
    return failures;
}

GenerationOutcome search_round_outcome() {
    GenerationOutcome outcome;
    outcome.reasoning         = "I need a forecast tool.";
    outcome.prompt_tokens     = 40;
    outcome.completion_tokens = 12;
    outcome.reasoning_tokens  = 6;
    outcome.finish_reason     = infernix::FinishReason::StopToken;
    outcome.tool_calls.push_back(infernix::GeneratedToolCall{
        .name = "tool_search_tool_regex", .arguments_json = R"({"pattern":"forecast"})"});
    return outcome;
}

GenerationOutcome client_round_outcome() {
    GenerationOutcome outcome;
    outcome.text                            = "Let me check.";
    outcome.prompt_tokens                   = 90;
    outcome.completion_tokens               = 9;
    outcome.finish_reason                   = infernix::FinishReason::StopToken;
    outcome.reused_prompt_tokens            = 52;
    outcome.tool_calls.push_back(infernix::GeneratedToolCall{
        .name = "get_forecast", .arguments_json = R"({"city":"Paris"})"});
    return outcome;
}

Json search_request_body() {
    Json body        = base_request();
    body["tools"]    = Json::array({ordinary_tool(), regex_search_tool(),
                                    deferred_tool("get_forecast", "Weather forecast for a city"),
                                    deferred_tool("get_tides", "Tide tables for a harbour")});
    body["messages"] = Json::array({Json{{"role", "user"}, {"content", "Forecast for Paris?"}}});
    return body;
}

int test_tool_search_execution() {
    const AnthropicMessagesRequest request = parse(search_request_body());
    AnthropicToolSearchExecutor executor(request);
    int failures = check(executor.search_tool("tool_search_tool_regex") != nullptr &&
                             executor.search_tool("weather") == nullptr &&
                             executor.names() == std::vector<std::string>{"tool_search_tool_regex"},
                         "the executor did not recognize the declared search tool");
    const auto run = [&](std::string arguments) {
        return executor.execute(ToolCall{.id             = "srvtoolu_test",
                                         .name           = "tool_search_tool_regex",
                                         .arguments_json = std::move(arguments)});
    };
    const AnthropicToolSearchResult found = run(R"({"pattern":"forecast|tide"})");
    failures += check(found.content["type"] == "tool_search_tool_search_result" &&
                          found.content["tool_references"].size() == 2 &&
                          found.content["tool_references"][0] ==
                              Json{{"type", "tool_reference"}, {"tool_name", "get_forecast"}} &&
                          found.turn.role == infernix::ChatRole::Tool &&
                          found.turn.tool_call_id == "srvtoolu_test" &&
                          found.turn.content.empty() && found.turn.loaded_tools.size() == 2,
                      "a search did not return and load the matching deferred tools");
    failures += check(run(R"({"pattern":"forecast|tide","limit":1})")
                              .content["tool_references"]
                              .size() == 1,
                      "the search limit was not applied");
    // Listed tools are not in the catalog: "weather" is listed, so nothing matches.
    const AnthropicToolSearchResult listed_only = run(R"({"pattern":"^weather$"})");
    failures += check(listed_only.content["tool_references"].empty() &&
                          listed_only.turn.loaded_tools.empty() &&
                          listed_only.turn.content.at(0).text ==
                              "No deferred tools matched the search.",
                      "an empty search did not say so");
    const AnthropicToolSearchResult invalid = run(R"({"pattern":"("})");
    failures += check(invalid.content["type"] == "tool_search_tool_result_error" &&
                          invalid.content["error_code"] == "invalid_tool_input" &&
                          invalid.turn.tool_result_is_error &&
                          invalid.turn.content.at(0).text.starts_with(
                              "Tool search failed (invalid_tool_input): Invalid regular "
                              "expression pattern: missing ),"),
                      "an invalid pattern did not become an error result");
    for (const char* arguments :
         {R"({"pattern":"x","limit":0})", R"({"pattern":"x","limit":10001})", R"({"query":"x"})"}) {
        failures += check(run(arguments).content["error_code"] == "invalid_tool_input",
                          std::string("invalid search input was accepted: ") + arguments);
    }
    // History lowering of the same content gives the same turn.
    const ChatTurn lowered =
        lower_anthropic_tool_search_result(found.content, "srvtoolu_test", request.generation);
    failures += check(lowered.loaded_tools.size() == found.turn.loaded_tools.size() &&
                          lowered.loaded_tools[0].name == found.turn.loaded_tools[0].name &&
                          lowered.loaded_tools[0].input_schema_json ==
                              found.turn.loaded_tools[0].input_schema_json,
                      "replayed search content lowered differently from the executed search");
    return failures;
}

// A search round, its continuation, and the client's next request that replays the response:
// the replay renders the continuation's messages unchanged, so the prefix cache serves it.
int test_tool_search_round_trip() {
    const AnthropicMessagesRequest request = parse(search_request_body());
    AnthropicToolSearchExecutor executor(request);
    const GenerationOutcome first = search_round_outcome();
    ServerToolRound round;
    round.outcome                            = first;
    round.calls                              = identify_anthropic_tool_calls(first, request.tool_searches);
    const AnthropicToolSearchResult executed = executor.execute(round.calls.at(0));
    round.results.push_back(executed.turn);
    GenerationRequest continuation = request.generation;
    append_server_round(continuation, round, executor.names());
    int failures = check(round.calls[0].id.starts_with("srvtoolu_") &&
                             continuation.messages.size() == 3 &&
                             continuation.messages[1].role == infernix::ChatRole::Assistant &&
                             continuation.messages[1].reasoning_content == first.reasoning &&
                             continuation.messages[2].role == infernix::ChatRole::Tool &&
                             continuation.max_tokens == 4096 - first.completion_tokens &&
                             continuation.continuation ==
                                 infernix::PromptContinuationMode::NewAssistantTurn,
                         "the continuation did not append the search round");
    const infernix::PromptInput continued = prompt(continuation);

    const GenerationOutcome second = client_round_outcome();
    const std::vector<ToolCall> second_calls =
        identify_anthropic_tool_calls(second, request.tool_searches);
    const std::vector<AnthropicResponseSegment> segments{
        AnthropicResponseSegment{.outcome        = &round.outcome,
                                 .calls          = round.calls,
                                 .search_results = {executed.content}},
        AnthropicResponseSegment{
            .outcome = &second, .calls = second_calls, .search_results = {std::nullopt}}};
    const GenerationOutcome totals = combine_round_outcomes(std::span(&round, 1), second);
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_search", "claude-local");
    const Json response =
        Json::parse(make_anthropic_messages_response(identity, segments, totals, std::nullopt));
    const Json& content = response.at("content");
    std::vector<std::string> types;
    for (const Json& block : content) { types.push_back(block.at("type").get<std::string>()); }
    failures += check(types == std::vector<std::string>{"thinking", "server_tool_use",
                                                        "tool_search_tool_result", "text",
                                                        "tool_use"} &&
                          content[1].at("id") == content[2].at("tool_use_id") &&
                          content[1].at("input") == Json{{"pattern", "forecast"}} &&
                          content[2].at("content").at("tool_references").size() == 1 &&
                          response.at("stop_reason") == "tool_use",
                      "the response did not show the search between the generations");
    failures += check(response.at("usage").at("output_tokens") == 21 &&
                          response.at("usage").at("cache_read_input_tokens") == 52 &&
                          response.at("usage").at("input_tokens") == 130 - 52,
                      "response usage did not sum the generations");

    Json replay = search_request_body();
    replay["messages"].push_back(Json{{"role", "assistant"}, {"content", content}});
    replay["messages"].push_back(Json{
        {"role", "user"},
        {"content", Json::array({tool_result(content[4].at("id").get<std::string>(), "Sunny")})}});
    const GenerationRequest replayed = parse(replay).generation;
    const infernix::PromptInput replayed_prompt = prompt(replayed);
    failures += check(replayed.messages.size() == 5 &&
                          replayed.messages[3].role == infernix::ChatRole::Assistant &&
                          replayed.messages[3].tool_calls.size() == 1 &&
                          replayed.messages[3].tool_calls[0].name == "get_forecast" &&
                          replayed.messages[4].role == infernix::ChatRole::Tool &&
                          replayed.messages[4].content[0].text == "Sunny",
                      "the replayed response was not lowered as two generations");
    failures += check(replayed_prompt.options.tool_jsons == continued.options.tool_jsons &&
                          replayed_prompt.options.deferred_tool_jsons ==
                              continued.options.deferred_tool_jsons &&
                          same_messages(replayed_prompt, continued, continued.messages.size()),
                      "the replayed history did not render the continuation's prompt prefix");

    // Count Tokens shares the lowering.
    failures += check(parse_anthropic_count_tokens_request(replay).generation.messages.size() == 5,
                      "Count Tokens did not lower tool search history");
    return failures;
}

int test_tool_search_history_errors() {
    const Json body = search_request_body();
    const Json search_use{{"type", "server_tool_use"},
                          {"id", "srvtoolu_1"},
                          {"name", "tool_search_tool_regex"},
                          {"input", Json{{"pattern", "forecast"}}}};
    const Json search_result{
        {"type", "tool_search_tool_result"},
        {"tool_use_id", "srvtoolu_1"},
        {"content", Json{{"type", "tool_search_tool_search_result"},
                         {"tool_references", Json::array({Json{{"type", "tool_reference"},
                                                               {"tool_name", "get_forecast"}}})}}}};
    const auto with_history = [&](Json assistant, std::optional<Json> user = std::nullopt) {
        Json request = body;
        request["messages"].push_back(Json{{"role", "assistant"}, {"content", std::move(assistant)}});
        if (user) { request["messages"].push_back(Json{{"role", "user"}, {"content", *user}}); }
        return request;
    };

    // A response paused after a search: the next generation opens a new assistant turn.
    const GenerationRequest paused =
        parse(with_history(Json::array({search_use, search_result}))).generation;
    int failures = check(paused.messages.size() == 3 &&
                             paused.messages.back().role == infernix::ChatRole::Tool &&
                             paused.continuation ==
                                 infernix::PromptContinuationMode::NewAssistantTurn,
                         "a history ending in a search result was not resumed as a new turn");

    failures += check(api_code([&] { (void)parse(with_history(Json::array({search_use}))); }) ==
                          "invalid_tool_history",
                      "a server_tool_use without its result was accepted");
    failures += check(api_code([&] {
                          (void)parse(with_history(Json::array({search_use, search_result}),
                                                   Json::array({tool_result("srvtoolu_1", "x")})));
                      }) == "invalid_tool_history",
                      "a client tool_result for a server search was accepted");
    failures +=
        check(api_code([&] {
                  (void)parse(with_history(Json::array({search_result, search_use})));
              }) == "invalid_tool_history",
              "a search result before its call was accepted");
    failures += check(api_code([&] {
                          (void)parse(with_history(Json::array(
                              {search_use, tool_use("toolu_x", "weather"), search_result,
                               Json{{"type", "text"}, {"text", "next"}}})));
                      }) == "invalid_tool_history",
                      "a client call left unanswered inside a continued message was accepted");
    Json unknown = search_result;
    unknown["content"]["tool_references"][0]["tool_name"] = "missing_tool";
    failures += check(api_code([&] {
                          (void)parse(with_history(Json::array({search_use, unknown})));
                      }) == "tool_reference_not_found",
                      "a search result naming an undeclared tool was accepted");
    Json no_search     = with_history(Json::array({search_use, search_result}));
    no_search["tools"] = Json::array({ordinary_tool()});
    failures += check(api_code([&] { (void)parse(no_search); }) == "server_tools_not_supported",
                      "a search history without a declared search tool was accepted");

    // A search and a client call in one generation: results follow call order.
    const GenerationRequest mixed =
        parse(with_history(Json::array({tool_use("toolu_w", "weather"), search_use, search_result}),
                           Json::array({tool_result("toolu_w", "Rain")})))
            .generation;
    failures += check(mixed.messages.size() == 4 && mixed.messages[1].tool_calls.size() == 2 &&
                          mixed.messages[2].tool_call_id == "toolu_w" &&
                          mixed.messages[3].tool_call_id == "srvtoolu_1" &&
                          mixed.messages[3].loaded_tools.size() == 1,
                      "results of a mixed generation did not follow call order");
    return failures;
}

int test_tool_search_stream() {
    const AnthropicMessagesRequest request = parse(search_request_body());
    AnthropicToolSearchExecutor executor(request);
    const GenerationOutcome first  = search_round_outcome();
    const GenerationOutcome second = client_round_outcome();
    AnthropicResponseSegment search_segment{
        .outcome = &first,
        .calls   = identify_anthropic_tool_calls(first, request.tool_searches),
        .search_results = {}};
    search_segment.search_results.push_back(executor.execute(search_segment.calls.at(0)).content);
    const AnthropicResponseSegment final_segment{
        .outcome        = &second,
        .calls          = identify_anthropic_tool_calls(second, request.tool_searches),
        .search_results = {std::nullopt}};
    ServerToolRound round{.outcome = first, .calls = search_segment.calls, .results = {}};
    const GenerationOutcome totals = combine_round_outcomes(std::span(&round, 1), second);

    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_search_stream", "claude-local");
    AnthropicMessagesStream stream(identity, first.prompt_tokens);
    std::vector<std::string> events{stream.start(infernix::GenerationStart{
        .prompt = infernix::PromptSummary{.prompt_tokens = 40}, .reused_prompt_tokens = 0})};
    const auto append_events = [&](std::vector<std::string> values) {
        events.insert(events.end(), std::make_move_iterator(values.begin()),
                      std::make_move_iterator(values.end()));
    };
    append_events(stream.reasoning_delta("I need a "));
    append_events(stream.segment(search_segment));
    append_events(stream.content_delta("Let me "));
    append_events(stream.finish(final_segment, totals, std::nullopt));

    std::vector<std::string> blocks;
    std::vector<int> indices;
    int starts      = 0;
    bool reasoning  = false;
    bool arguments  = false;
    Json terminal;
    for (const std::string& wire : events) {
        const Json event = parse_event(wire);
        if (event.at("type") == "message_start") { ++starts; }
        if (event.at("type") == "content_block_start") {
            blocks.push_back(event.at("content_block").at("type").get<std::string>());
            indices.push_back(event.at("index").get<int>());
        }
        if (event.at("type") == "content_block_delta" &&
            event.at("delta").at("type") == "thinking_delta") {
            reasoning = reasoning || event.at("delta").at("thinking") == "forecast tool.";
        }
        if (event.at("type") == "content_block_delta" &&
            event.at("delta").at("type") == "input_json_delta" && blocks.back() == "server_tool_use") {
            arguments = event.at("delta").at("partial_json") == R"({"pattern":"forecast"})";
        }
        if (event.at("type") == "message_delta") { terminal = event; }
    }
    int failures = check(starts == 1 &&
                             blocks == std::vector<std::string>{"thinking", "server_tool_use",
                                                                "tool_search_tool_result", "text",
                                                                "tool_use"} &&
                             indices == std::vector<int>{0, 1, 2, 3, 4} && reasoning && arguments,
                         "the stream did not show the search between the generations in order");
    failures += check(terminal.at("delta").at("stop_reason") == "tool_use" &&
                          terminal.at("usage").at("output_tokens") == 21,
                      "the stream's terminal usage did not sum the generations");

    // A response that reached the generation limit pauses instead of claiming a tool call.
    const AnthropicResponseSegment paused{.outcome        = &first,
                                          .calls          = search_segment.calls,
                                          .search_results = search_segment.search_results};
    const Json pause = Json::parse(make_anthropic_messages_response(
        identity, std::span(&paused, 1), first, std::string_view("pause_turn")));
    failures += check(pause.at("stop_reason") == "pause_turn" && pause.at("content").size() == 3,
                      "a paused response did not report pause_turn with its search");
    return failures;
}

// Claude Code auto mode's client-side classifier: the judged transcript ends with a System turn
// stating the decision rules, with a small output budget and stop sequences beside it.
int test_auto_mode_classifier_body() {
    Json body              = base_request();
    body["thinking"]       = Json{{"type", "disabled"}};
    body["system"]         = "You are an action-safety evaluator.";
    body["max_tokens"]     = 2112;
    body["stop_sequences"] = Json::array({"STOP"});
    // The adapter admits an inline System only after a User turn and final or before an
    // Assistant (anthropic_messages_request.cpp), which is exactly the transcript shape the
    // real classifier sends.
    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "delete the build cache"}},
         Json{{"role", "assistant"}, {"content", "rm -rf build"}},
         Json{{"role", "user"}, {"content", "Should this action run?"}},
         Json{{"role", "system"}, {"content", "Judge the last action against the rules."}}});

    const AnthropicMessagesRequest parsed = parse(body);
    const GenerationRequest& request      = parsed.generation;
    int failures = check(parsed.model == "claude-local" && request.max_tokens == 2112 &&
                             request.stop_strings.size() == 1 && request.stop_strings[0] == "STOP",
                         "classifier envelope fields were not preserved");
    failures += check(
        request.messages.size() == 5 && request.messages[0].role == infernix::ChatRole::System &&
            request.messages[1].role == infernix::ChatRole::User &&
            request.messages[2].role == infernix::ChatRole::Assistant &&
            request.messages[4].role == infernix::ChatRole::System &&
            request.messages[4].content.front().text == "Judge the last action against the rules.",
        "the classifier's mid-conversation System turn was not parsed role-faithfully");
    const infernix::PromptInput translated = prompt(request);
    failures += check(translated.messages.size() == 5 &&
                          translated.messages[4].role == infernix::ChatRole::System,
                      "PromptInput lowering lost or hoisted the mid-conversation System turn");
    return failures;
}

// Claude Code auto mode's server-side review additionally carries a safeguards object; Infernix
// ignores it like every field it does not implement, leaving the rest of the request untouched.
int test_auto_mode_server_review_safeguards_body() {
    Json body        = base_request();
    body["thinking"] = Json{{"type", "disabled"}};
    body["system"]   = "Review this action for approval.";
    body["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "run the deploy script"}},
                     Json{{"role", "assistant"}, {"content", "deploying now"}},
                     Json{{"role", "user"}, {"content", "Was that within approval?"}},
                     Json{{"role", "system"}, {"content", "Approve or deny the last action."}}});
    body["safeguards"] =
        Json{{"rules", Json::array({"no-secrets"})}, {"future_check", Json{{"enabled", true}}}};
    body["context_management"] = Json{{"edits", Json::array()}};

    const AnthropicMessagesRequest reviewed = parse(body);
    Json plain                              = body;
    plain.erase("safeguards");
    plain.erase("context_management");
    const AnthropicMessagesRequest without = parse(plain);
    int failures =
        check(reviewed.model == without.model &&
                  reviewed.generation.messages.size() == without.generation.messages.size(),
              "safeguards review body was not parsed like the same request without them");
    failures +=
        check(reviewed.generation.messages.size() == 5 &&
                  reviewed.generation.messages[4].role == infernix::ChatRole::System &&
                  reviewed.generation.messages[4].content.front().text ==
                      "Approve or deny the last action.",
              "the review's mid-conversation System turn was altered by the safeguards fields");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_constrained_decoding();
    failures += test_envelope_and_field_policy();
    failures += test_message_normalization();
    failures += test_attribution_system_block();
    failures += test_claude_code_prompt();
    failures += test_tool_history();
    failures += test_tools();
    failures += test_prompt_object_order();
    failures += test_thinking_and_count_tokens();
    failures += test_thinking_history_transport_metadata();
    failures += test_content_and_cache_hints();
    failures += test_aggregate_and_errors();
    failures += test_timings();
    failures += test_tool_call_presentation();
    failures += test_streamed_tool_calls();
    failures += test_stream();
    failures += test_hidden_reasoning_signature();
    failures += test_hidden_reasoning_round_trip();
    failures += test_auto_mode_classifier_body();
    failures += test_auto_mode_server_review_safeguards_body();
    failures += test_deferred_tool_declarations();
    failures += test_client_tool_references();
    failures += test_tool_search_execution();
    failures += test_tool_search_round_trip();
    failures += test_tool_search_history_errors();
    failures += test_tool_search_stream();
    if (failures != 0) {
        std::cerr << failures << " Anthropic adapter checks failed\n";
        return 1;
    }
    std::cout << "Anthropic adapter checks passed\n";
    return 0;
}
