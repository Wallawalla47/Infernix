// /v1/score protocol: all-or-nothing validation, the explicit chat-template choice and the response
// shape. No Engine: the scores themselves come from Engine::score_tokens (perplexity's path).
#include "serve/score.h"

#include <iostream>
#include <string>

namespace {

using namespace infernix::serve;
using Json = RequestJson;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::string api_code(const Json& body) {
    try {
        (void)parse_score_request(body);
    } catch (const ApiException& exception) {
        return exception.error().code.empty() ? std::string("<no code>") : exception.error().code;
    }
    return {};
}

std::string api_param(const Json& body) {
    try {
        (void)parse_score_request(body);
    } catch (const ApiException& exception) { return exception.error().param; }
    return {};
}

Json raw() {
    return Json{{"prompt", "The capital of France is"}, {"apply_chat_template", false},
                {"continuation", " Paris"}, {"top_k", 20}};
}

Json templated() {
    return Json{{"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
                {"apply_chat_template", true},
                {"enable_thinking", false},
                {"continuation", "Hi"},
                {"top_k", 5}};
}

void test_parse() {
    const ScoreRequest a = parse_score_request(raw());
    check(!a.apply_chat_template && a.prompt == "The capital of France is", "raw prompt parsed");
    check(a.continuation == " Paris" && a.top_k == 20, "raw continuation and top_k parsed");
    check(a.chat.messages.empty(), "a raw prompt renders no turns");

    const ScoreRequest b = parse_score_request(templated());
    check(b.apply_chat_template && b.chat.messages.size() == 1, "messages parsed by the chat parser");
    check(b.chat.enable_thinking == std::optional<bool>(false), "enable_thinking reaches the template");

    Json c = raw();
    c["apply_chat_template"] = true;
    const ScoreRequest prompt_as_turn = parse_score_request(c);
    check(prompt_as_turn.chat.messages.size() == 1 &&
              prompt_as_turn.chat.messages.front().role == infernix::ChatRole::User,
          "a templated prompt is one user turn");

    Json d = raw();
    d.erase("top_k");
    check(parse_score_request(d).top_k == 0, "top_k defaults to 0");
    d["top_k"] = infernix::kMaximumScoreTopTokens;
    check(parse_score_request(d).top_k == infernix::kMaximumScoreTopTokens, "top_k at the bound");

    Json e = templated();
    check(!parse_score_request(e).hidden_states, "hidden_states defaults to off");
    e["hidden_states"] = true;
    check(parse_score_request(e).hidden_states, "hidden_states asked for");
}

void test_refusals() {
    check(api_code(Json::array()) == "<no code>", "a non-object body is refused");
    Json body = raw();
    body.erase("apply_chat_template");
    check(api_code(body) == "apply_chat_template_required", "the template choice has no default");
    body = raw();
    body["apply_chat_template"] = "yes";
    check(api_code(body) == "apply_chat_template_required", "the template choice is a boolean");

    body             = raw();
    body["messages"] = templated()["messages"];
    check(api_code(body) == "prompt_or_messages", "prompt and messages together are refused");
    body = raw();
    body.erase("prompt");
    check(api_code(body) == "prompt_or_messages", "a body with neither is refused");
    body           = raw();
    body["prompt"] = "";
    check(api_param(body) == "prompt", "an empty prompt is refused");

    body                        = templated();
    body["apply_chat_template"] = false;
    body.erase("enable_thinking");
    check(api_code(body) == "messages_need_template", "messages without the template are refused");
    body                    = raw();
    body["enable_thinking"] = false;
    check(api_code(body) == "template_option_without_template",
          "a template option without the template is refused");

    body          = raw();
    body["top_k"] = infernix::kMaximumScoreTopTokens + 1;
    check(api_code(body) == "top_k_out_of_range", "an over-large top_k is refused, not truncated");
    check(api_param(body) == "top_k", "the top_k refusal names its param");
    body["top_k"] = -1;
    check(api_code(body) == "top_k_out_of_range", "a negative top_k is refused");
    body["top_k"] = 1.5;
    check(api_code(body) == "invalid_top_k", "a fractional top_k is refused");

    body = raw();
    body.erase("continuation");
    check(api_code(body) == "continuation_required", "a scored position needs a continuation");
    body["continuation"] = "";
    check(api_code(body) == "continuation_required", "an empty continuation is refused");

    body             = raw();
    body["logprobs"] = true;
    check(api_code(body) == "unknown_field", "an unknown field is refused, not ignored");
    check(api_param(body) == "logprobs", "the unknown field is named");

    body                  = templated();
    body["hidden_states"] = "last";
    check(api_code(body) == "invalid_hidden_states" && api_param(body) == "hidden_states",
          "hidden_states takes a boolean");

    body             = templated();
    body["messages"] = Json::array({Json{
        {"role", "user"},
        {"content", Json::array({Json{{"type", "image_url"},
                                      {"image_url", Json{{"url", "data:image/png;base64,AA=="}}}}})}}});
    check(api_code(body) == "score_text_only", "media is refused");
}

void test_response() {
    ScoreOutcome outcome;
    outcome.chat_template_applied = false;
    outcome.top_k                 = 2;
    outcome.prompt_tokens         = 7;
    outcome.positions.push_back(ScoredPosition{
        .id = 11, .text = " Paris", .logprob = -0.25F,
        .top = {ScoredAlternative{.id = 11, .text = " Paris", .logprob = -0.25F},
                ScoredAlternative{.id = 12, .text = " Lyon", .logprob = -3.5F}}});
    const Json body = make_score_response("qwen", outcome);
    check(body["object"] == "score" && body["model"] == "qwen", "response identity");
    check(body["chat_template_applied"] == false, "the template choice is stated");
    check(body["enable_thinking"].is_null(), "no thinking switch without a template");
    check(body["top_k"] == 2 && body["max_top_k"] == infernix::kMaximumScoreTopTokens, "top_k and its bound");
    check(body["positions"].size() == 1, "one scored position per continuation token");
    const Json& position = body["positions"][0];
    check(position["token_id"] == 11 && position["token"] == " Paris", "the scored token");
    check(position["logprob"].get<double>() == -0.25, "the scored token's logprob");
    check(position["top_logprobs"].size() == 2, "top_logprobs width is top_k");
    check(position["top_logprobs"][0]["logprob"].get<double>() >=
              position["top_logprobs"][1]["logprob"].get<double>(),
          "top_logprobs descend");
    check(position["top_logprobs"][1]["token"] == " Lyon", "alternatives are detokenized");
    check(body["usage"]["prompt_tokens"] == 7 && body["usage"]["continuation_tokens"] == 1, "usage");
    check(!body.contains("hidden_states"), "no hidden states unless asked for");

    // Two layers of three values, layer-major as Engine::score_tokens returns them: one row per layer, each value as
    // the engine widened it from BF16.
    outcome.hidden_states       = {0.5F, -1.25F, 0.0078125F, 3.0F, -0.375F, 1024.0F};
    outcome.hidden_state_layers = 2;
    const Json hidden           = make_score_response("qwen", outcome)["hidden_states"];
    check(hidden.size() == 2 && hidden[0].size() == 3 && hidden[1].size() == 3, "one row per layer");
    check(hidden[0][2].get<double>() == 0.0078125 && hidden[1][0].get<double>() == 3.0 &&
              hidden[1][2].get<double>() == 1024.0,
          "rows in order, values exact");
    outcome.hidden_states.clear();
    outcome.hidden_state_layers = 0;

    outcome.chat_template_applied = true;
    outcome.enable_thinking       = false;
    const Json templated_body     = make_score_response("qwen", outcome);
    check(templated_body["chat_template_applied"] == true && templated_body["enable_thinking"] == false,
          "a templated score states its thinking switch");
}

} // namespace

int main() {
    test_parse();
    test_refusals();
    test_response();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "score schema: all checks passed\n";
    return 0;
}
