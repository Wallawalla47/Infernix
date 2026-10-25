#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace infernix::serve {

enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,
    Requested,
    Disabled,
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
};

[[nodiscard]] bool parse_openai_prompt_cache_breakpoint(const RequestJson& value,
                                                        std::string_view param);
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body);
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

// The loaded chat template's capabilities, reported as `chat_template_caps` on each /v1/models
// object (the object llama.cpp reports for its own template).
struct ChatTemplateCaps {
    bool supports_tools = false; // a declared tool reaches the rendered prompt
};

// Probes the chat template through the serving prompt path: `count_prompt_tokens` counts a short
// conversation declaring one probe tool, once under a short name and once under a longer one. The
// template renders tool declarations when the longer name lengthens the prompt; a template that
// ignores tools, or renders a fixed tool preamble without the declaration, counts both alike. A
// template that refuses a declared tool (HTTP 400) does not support tools, and a tool-bearing
// prompt that overflows the context while the tool-free one fits shows that the declaration
// rendered. Other errors propagate.
[[nodiscard]] ChatTemplateCaps
probe_chat_template_caps(const std::function<int(const GenerationRequest&)>& count_prompt_tokens);

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len, const ModelMetadata& metadata,
                             const ChatTemplateCaps& caps);
std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len, const ModelMetadata& metadata,
                              const ChatTemplateCaps& caps);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

// The model an OpenAI request runs on: `available` (the public model ID) when `requested` names it,
// or names anything non-empty and `any_name` (--any-model-name) is set; otherwise throws 404
// model_not_found. Responses and request logs report the returned name.
[[nodiscard]] std::string resolve_openai_model(std::string_view requested, std::string_view available,
                                               bool any_name);

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

} // namespace infernix::serve
