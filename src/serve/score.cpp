#include "serve/score.h"

#include "serve/openai_chat.h"
#include "serve/request_validation.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace infernix::serve {
namespace {

using Json = RequestJson;

// Template options that only mean something when the template renders the prompt.
constexpr std::array<std::string_view, 4> kTemplateFields{"enable_thinking", "preserve_thinking",
                                                          "chat_template_kwargs", "reasoning_effort"};
constexpr std::array<std::string_view, 7> kScoreFields{"model",        "prompt", "messages",
                                                       "apply_chat_template", "continuation",
                                                       "top_k", "hidden_states"};

bool known_field(std::string_view key) {
    for (const std::string_view field : kScoreFields) {
        if (key == field) { return true; }
    }
    for (const std::string_view field : kTemplateFields) {
        if (key == field) { return true; }
    }
    return false;
}

bool present(const Json& body, const char* key) { return body.contains(key) && !body.at(key).is_null(); }

std::uint32_t parse_top_k(const Json& body) {
    if (!present(body, "top_k")) { return 0; }
    const Json& value = body.at("top_k");
    if (!value.is_number_integer()) { bad_request("top_k must be an integer", "top_k", "invalid_top_k"); }
    const auto top_k = value.get<std::int64_t>();
    if (top_k < 0 || top_k > static_cast<std::int64_t>(infernix::kMaximumScoreTopTokens)) {
        // Refused, not narrowed: a top-N quietly cut to the bound changes what a KL over it means.
        bad_request("top_k must be in [0," + std::to_string(infernix::kMaximumScoreTopTokens) + "]",
                    "top_k", "top_k_out_of_range");
    }
    return static_cast<std::uint32_t>(top_k);
}

} // namespace

ScoreRequest parse_score_request(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    for (const auto& [key, value] : body.items()) {
        if (!known_field(key)) {
            bad_request("unknown field: " + ascii_preview(key), key, "unknown_field");
        }
    }
    ScoreRequest request;
    if (present(body, "model")) {
        if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
        request.model = body.at("model").get<std::string>();
    }
    if (!present(body, "apply_chat_template") || !body.at("apply_chat_template").is_boolean()) {
        // No default: whether the template ran decides which distribution is measured.
        bad_request("apply_chat_template must be given as true or false", "apply_chat_template",
                    "apply_chat_template_required");
    }
    request.apply_chat_template = body.at("apply_chat_template").get<bool>();

    const bool has_prompt   = present(body, "prompt");
    const bool has_messages = present(body, "messages");
    if (has_prompt == has_messages) {
        bad_request("exactly one of prompt or messages is required", has_prompt ? "messages" : "prompt",
                    "prompt_or_messages");
    }
    if (has_prompt) {
        if (!body.at("prompt").is_string() || body.at("prompt").get_ref<const std::string&>().empty()) {
            bad_request("prompt must be a non-empty string", "prompt");
        }
        request.prompt = body.at("prompt").get<std::string>();
    }
    if (has_messages && !request.apply_chat_template) {
        bad_request("messages are rendered by the chat template: apply_chat_template must be true",
                    "messages", "messages_need_template");
    }
    if (!request.apply_chat_template) {
        for (const std::string_view field : kTemplateFields) {
            if (present(body, std::string(field).c_str())) {
                bad_request(std::string(field) + " is a chat template option: apply_chat_template is false",
                            std::string(field), "template_option_without_template");
            }
        }
    } else {
        // The Chat Completions parser, so the turns and template options mean exactly what they
        // mean on /v1/chat/completions.
        Json chat = Json::object();
        chat["model"] = "score";
        if (has_messages) {
            chat["messages"] = body.at("messages");
        } else {
            chat["messages"] = Json::array({Json{{"role", "user"}, {"content", request.prompt}}});
        }
        for (const std::string_view field : kTemplateFields) {
            const std::string key(field);
            if (body.contains(key)) { chat[key] = body.at(key); }
        }
        request.chat = parse_chat_completion_request(chat, RequestLimits{}).generation;
        if (request.chat.media_item_count() != 0) {
            bad_request("scoring takes text only", "messages", "score_text_only");
        }
    }

    if (!present(body, "continuation") || !body.at("continuation").is_string() ||
        body.at("continuation").get_ref<const std::string&>().empty()) {
        // A scored position needs a target token; the top_k at position 0 is the prompt's
        // next-token distribution whatever that token is.
        bad_request("continuation must be a non-empty string", "continuation", "continuation_required");
    }
    request.continuation = body.at("continuation").get<std::string>();
    request.top_k        = parse_top_k(body);
    if (present(body, "hidden_states")) {
        if (!body.at("hidden_states").is_boolean()) {
            bad_request("hidden_states must be true or false", "hidden_states", "invalid_hidden_states");
        }
        request.hidden_states = body.at("hidden_states").get<bool>();
    }
    return request;
}

RequestJson make_score_response(const std::string& model, const ScoreOutcome& outcome) {
    Json positions = Json::array();
    for (const ScoredPosition& position : outcome.positions) {
        Json top = Json::array();
        for (const ScoredAlternative& alternative : position.top) {
            top.push_back(Json{{"token_id", alternative.id},
                               {"token", alternative.text},
                               {"logprob", alternative.logprob}});
        }
        positions.push_back(Json{{"token_id", position.id},
                                 {"token", position.text},
                                 {"logprob", position.logprob},
                                 {"top_logprobs", std::move(top)}});
    }
    Json body;
    body["object"]                = "score";
    body["model"]                 = model;
    body["chat_template_applied"] = outcome.chat_template_applied;
    body["enable_thinking"] = outcome.enable_thinking ? Json(*outcome.enable_thinking) : Json(nullptr);
    body["top_k"]           = outcome.top_k;
    body["max_top_k"]       = infernix::kMaximumScoreTopTokens;
    body["positions"]       = std::move(positions);
    if (outcome.hidden_state_layers != 0) {
        // One row per layer (hidden_size values each), as Engine::score_tokens returns them.
        const std::size_t width = outcome.hidden_states.size() / outcome.hidden_state_layers;
        Json rows               = Json::array();
        for (std::size_t layer = 0; layer < outcome.hidden_state_layers; ++layer) {
            const auto first = outcome.hidden_states.begin() + static_cast<std::ptrdiff_t>(layer * width);
            rows.push_back(Json(std::vector<float>(first, first + static_cast<std::ptrdiff_t>(width))));
        }
        body["hidden_states"] = std::move(rows);
    }
    body["usage"]["prompt_tokens"]       = outcome.prompt_tokens;
    body["usage"]["continuation_tokens"] = outcome.positions.size();
    return body;
}

} // namespace infernix::serve
