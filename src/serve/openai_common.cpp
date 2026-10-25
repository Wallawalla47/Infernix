#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <string_view>
#include <utility>

namespace infernix::serve {

namespace {

std::string chat_identifier(std::string_view prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx",
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + buffer.data();
}

std::string responses_identifier(std::string_view prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 48> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx%016llx",
                  static_cast<unsigned long long>(distribution(random)),
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + "_" + buffer.data();
}

} // namespace

using Json = nlohmann::json;

bool parse_openai_prompt_cache_breakpoint(const RequestJson& value, std::string_view param) {
    if (!value.contains("prompt_cache_breakpoint") ||
        value.at("prompt_cache_breakpoint").is_null()) {
        return false;
    }
    const RequestJson& breakpoint = value.at("prompt_cache_breakpoint");
    if (!breakpoint.is_object() || !breakpoint.contains("mode") ||
        !breakpoint.at("mode").is_string() ||
        breakpoint.at("mode").get<std::string>() != "explicit") {
        bad_request("prompt_cache_breakpoint must be {mode:'explicit'}", std::string(param),
                    "invalid_cache_breakpoint");
    }
    return true;
}

OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body) {
    auto require_string_hint = [&](const char* field, std::optional<std::size_t> maximum = {}) {
        if (!body.contains(field) || body.at(field).is_null()) { return; }
        if (!body.at(field).is_string()) {
            bad_request(std::string(field) + " must be a string", field);
        }
        if (maximum && body.at(field).get_ref<const std::string&>().size() > *maximum) {
            bad_request(std::string(field) + " must be at most " + std::to_string(*maximum) +
                            " characters",
                        field);
        }
    };
    require_string_hint("prompt_cache_key", 64U);
    require_string_hint("safety_identifier", 64U);
    require_string_hint("user");

    if (body.contains("prompt_cache_retention") && !body.at("prompt_cache_retention").is_null()) {
        if (!body.at("prompt_cache_retention").is_string()) {
            bad_request("prompt_cache_retention must be a string", "prompt_cache_retention");
        }
        const std::string value = body.at("prompt_cache_retention").get<std::string>();
        if (value != "in_memory" && value != "24h") {
            bad_request("prompt_cache_retention must be 'in_memory' or '24h'",
                        "prompt_cache_retention");
        }
    }

    OpenAIPromptCachePolicy policy;
    if (!body.contains("prompt_cache_options") || body.at("prompt_cache_options").is_null()) {
        return policy;
    }
    const RequestJson& options = body.at("prompt_cache_options");
    if (!options.is_object()) {
        bad_request("prompt_cache_options must be an object", "prompt_cache_options");
    }
    policy.automatic = OpenAIPromptCacheAutomatic::Requested;
    if (options.contains("mode") && !options.at("mode").is_null()) {
        if (!options.at("mode").is_string()) {
            bad_request("prompt_cache_options.mode must be a string", "prompt_cache_options");
        }
        const std::string mode = options.at("mode").get<std::string>();
        if (mode == "explicit") {
            policy.automatic = OpenAIPromptCacheAutomatic::Disabled;
        } else if (mode != "implicit") {
            bad_request("prompt_cache_options.mode must be 'implicit' or 'explicit'",
                        "prompt_cache_options");
        }
    }
    if (options.contains("ttl") && !options.at("ttl").is_null() &&
        (!options.at("ttl").is_string() || options.at("ttl").get<std::string>() != "30m")) {
        bad_request("prompt_cache_options.ttl must be '30m'", "prompt_cache_options");
    }
    return policy;
}

void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy) {
    std::vector<std::optional<CacheBoundary>*> explicit_boundaries;
    for (ToolDefinition& tool : request.tools) {
        if (tool.cache_boundary_after) {
            explicit_boundaries.push_back(&tool.cache_boundary_after);
        }
    }
    for (ChatTurn& turn : request.messages) {
        for (ContentPart& part : turn.content) {
            if (part.cache_boundary_after) {
                explicit_boundaries.push_back(&part.cache_boundary_after);
            }
        }
        if (turn.cache_boundary_after) {
            explicit_boundaries.push_back(&turn.cache_boundary_after);
        }
    }

    std::optional<CacheBoundary>* automatic_target = nullptr;
    for (auto turn = request.messages.rbegin(); turn != request.messages.rend(); ++turn) {
        // Calls and loaded tool definitions render after the turn's content parts.
        if (!turn->tool_calls.empty() || !turn->loaded_tools.empty()) {
            automatic_target = &turn->cache_boundary_after;
            break;
        }
        if (!turn->content.empty()) {
            automatic_target = &turn->content.back().cache_boundary_after;
            break;
        }
    }
    if (automatic_target == nullptr) {
        if (ToolDefinition* listed = request.last_listed_tool()) {
            automatic_target = &listed->cache_boundary_after;
        }
    }

    const bool automatic_enabled =
        policy.automatic != OpenAIPromptCacheAutomatic::Disabled && automatic_target != nullptr &&
        (automatic_target->has_value() ||
         explicit_boundaries.size() < kMaximumExplicitPromptCacheMarkers);
    const bool automatic_merges_explicit   = automatic_enabled && automatic_target->has_value();
    const std::size_t explicit_write_slots = !automatic_enabled || automatic_merges_explicit
                                                 ? kMaximumExplicitPromptCacheMarkers
                                                 : kMaximumExplicitPromptCacheMarkers - 1U;
    const std::size_t first_selected       = explicit_boundaries.size() > explicit_write_slots
                                                 ? explicit_boundaries.size() - explicit_write_slots
                                                 : 0U;
    for (std::size_t index = 0; index < explicit_boundaries.size(); ++index) {
        if (index < first_selected) {
            explicit_boundaries[index]->reset();
        } else {
            explicit_boundaries[index]->value().kind =
                infernix::PromptCacheMarkerKind::SharedStablePrefix;
            explicit_boundaries[index]->value().evidence =
                infernix::SharedCandidateEvidence::ExplicitBoundary;
        }
    }

    if (automatic_enabled) {
        const infernix::SharedCandidateEvidence evidence =
            policy.automatic == OpenAIPromptCacheAutomatic::Default
                ? infernix::SharedCandidateEvidence::DefaultAutomatic
                : infernix::SharedCandidateEvidence::RequestedAutomatic;
        if (*automatic_target) {
            automatic_target->value().evidence |= evidence;
        } else {
            *automatic_target = CacheBoundary{.evidence = evidence};
        }
    }
    // OpenAI already defines the automatic/explicit write policy for every request. Existing
    // exact shared residents are still considered by the Engine independently of this switch.
    request.allow_engine_automatic_shared_prefixes = false;
}

namespace {

// OpenAI model object carrying the llama.cpp-compatible `meta` and `chat_template_caps` fields.
// `id` is the public OpenAI alias; the facts describe the loaded model. n_ctx is this instance's
// configured per-request context ceiling (equal to max_model_len), not the model's native context;
// n_ctx_train is the model's native training context. ftype is the loaded weights profile (the
// quantization name).
Json make_model_object_json(const std::string& model_id, std::int64_t created,
                            std::uint32_t max_model_len, const ModelMetadata& metadata,
                            const ChatTemplateCaps& caps) {
    return Json{{"id", model_id},
                {"object", "model"},
                {"created", created},
                {"owned_by", "infernix"},
                {"max_model_len", max_model_len},
                {"meta",
                 Json{{"n_vocab", metadata.vocab_size},
                      {"n_ctx", max_model_len},
                      {"n_ctx_train", metadata.native_context},
                      {"n_embd", metadata.embedding_size},
                      {"n_params", metadata.parameters},
                      {"size", metadata.weight_bytes},
                      {"ftype", metadata.weights_id}}},
                {"chat_template_caps", Json{{"supports_tools", caps.supports_tools}}}};
}

// A one-message conversation, declaring one tool named `tool_name` unless it is empty.
GenerationRequest template_probe_request(std::string tool_name) {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.text     = "hi";
    content.type_raw = "text";
    turn.content.push_back(std::move(content));
    request.messages.push_back(std::move(turn));
    request.max_tokens = 1;
    if (!tool_name.empty()) {
        ToolDefinition tool;
        tool.name              = std::move(tool_name);
        tool.description       = "Probe tool.";
        tool.input_schema_json = R"({"type":"object","properties":{}})";
        tool.schema_param      = "tools/0/input_schema";
        request.tools.push_back(std::move(tool));
    }
    return request;
}

} // namespace

ChatTemplateCaps
probe_chat_template_caps(const std::function<int(const GenerationRequest&)>& count_prompt_tokens) {
    // The tool-free prompt renders and fits; any error here is not about tools.
    (void)count_prompt_tokens(template_probe_request({}));
    try {
        const int short_name = count_prompt_tokens(template_probe_request("probe"));
        const int long_name  = count_prompt_tokens(
            template_probe_request("probe_tool_declaration_reaches_the_rendered_prompt"));
        return ChatTemplateCaps{.supports_tools = long_name > short_name};
    } catch (const ApiException& exception) {
        if (exception.error().code == "context_length_exceeded") {
            return ChatTemplateCaps{.supports_tools = true};
        }
        if (exception.error().status != 400) { throw; }
        return ChatTemplateCaps{};
    }
}

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len, const ModelMetadata& metadata,
                             const ChatTemplateCaps& caps) {
    // vLLM/llama.cpp-compatible discovery metadata for the configured per-request context limit.
    const Json payload = {
        {"object", "list"},
        {"data", Json::array({make_model_object_json(model_id, created, max_model_len, metadata,
                                                     caps)})}};
    return payload.dump();
}

std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len, const ModelMetadata& metadata,
                              const ChatTemplateCaps& caps) {
    return make_model_object_json(model_id, created, max_model_len, metadata, caps).dump();
}

std::string make_error_body(const ApiError& error) {
    Json rendered     = {{"message", error.message}, {"type", error.type}};
    rendered["param"] = error.param.empty() ? Json(nullptr) : Json(error.param);
    rendered["code"]  = error.code.empty() ? Json(nullptr) : Json(error.code);
    return Json{{"error", std::move(rendered)}}.dump();
}

std::int64_t unix_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string resolve_openai_model(std::string_view requested, std::string_view available, bool any_name) {
    if (requested == available || (any_name && !requested.empty())) { return std::string(available); }
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "model";
    error.code    = "model_not_found";
    error.message = "model '" + std::string(requested) + "' not found";
    throw ApiException(std::move(error));
}

std::string new_openai_chat_completion_id() { return chat_identifier("chatcmpl-"); }

std::string new_openai_chat_tool_call_id() { return chat_identifier("call_"); }

std::string new_openai_request_id() { return responses_identifier("req"); }

std::string new_openai_response_id() { return responses_identifier("resp"); }

std::string new_openai_response_item_id(std::string_view prefix) {
    return responses_identifier(prefix);
}

} // namespace infernix::serve
