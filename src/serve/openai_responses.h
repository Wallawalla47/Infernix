#pragma once

// OpenAI Responses protocol adapter. Request parsing, state normalization, response encoding and
// HTTP transport live in separate translation units; only wire-independent GenerationRequest is
// passed to GenerationService.

#include "serve/openai_common.h"
#include "serve/openai_responses_store.h"
#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernix::serve {

struct GenerationOutcome;

struct OpenAIResponsesFunctionIdentity {
    std::string name;
    std::optional<std::string> wire_namespace;
    // A free-form `custom` tool: served to the model as a function with one string parameter,
    // `input`, and returned on the wire as custom_tool_call Items carrying that text.
    bool custom = false;
    // The `tool_search` tool: served to the model as a function; its calls are tool_search_call
    // Items.
    bool tool_search = false;

    bool operator==(const OpenAIResponsesFunctionIdentity&) const = default;
};

// The Engine-side parameter that carries a custom tool's free-form input.
inline constexpr const char* kCustomToolInputParameter = "input";
// The Engine-side function name of the `tool_search` tool.
inline constexpr const char* kToolSearchFunctionName = "tool_search";

// One thing a hosted tool search loads: the deferred functions of a namespace, or one top-level
// deferred function or custom tool.
struct OpenAIResponsesSearchPath {
    std::string path;
    nlohmann::json wire; // its tool_search_output `tools` entry (deferred members only)
};

// The request's `tool_search` tool. A hosted search (execution "server", the default) runs in
// Infernix: the model names paths from the catalog in the tool's description, the loaded
// definitions follow in a tool_search_output Item, and the response continues. A client search
// ends the response with a tool_search_call Item; the client answers with a tool_search_output.
struct OpenAIResponsesToolSearch {
    bool server = true;
    std::vector<OpenAIResponsesSearchPath> paths;
};

struct OpenAIResponsesPromptRequest {
    std::string model;
    GenerationRequest generation;
    OpenAIPromptCachePolicy cache_policy;
    std::vector<ChatTurn> input_turns;
    std::vector<nlohmann::json> input_items;
    std::optional<std::string> instructions;
    std::optional<std::string> previous_response_id;
    std::optional<std::string> reasoning_summary;
    std::optional<OpenAIResponsesToolSearch> tool_search;
};

struct OpenAIResponsesCreateRequest {
    OpenAIResponsesPromptRequest prompt;
    nlohmann::json text_format = {{"type", "text"}};
    nlohmann::json metadata    = nlohmann::json::object();
    nlohmann::json tools       = nlohmann::json::array();
    nlohmann::json tool_choice = "auto";
    // Responses beta namespace tools are flattened for the Engine and restored only at the wire
    // boundary. Never infer a namespace by splitting an Engine function name.
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity> tool_identities;
    std::optional<int> requested_max_output_tokens;
    std::optional<int> max_tool_calls;
    bool parallel_tool_calls                 = true;
    bool store                               = true;
    bool stream                              = false;
    bool include_reasoning_encrypted_content = false;
};

struct OpenAIResponsesResolvedPrompt {
    GenerationRequest generation;
    OpenAIResponseContext parent;
    bool preserve_thinking_semantic_change = false;
};

struct OpenAIResponsesRuntimeValues {
    float temperature       = 1.0F;
    float top_p             = 1.0F;
    int cached_input_tokens = 0;
};

struct BuiltOpenAIResponse {
    nlohmann::json body;
    std::vector<nlohmann::json> output_items;
    std::vector<ChatTurn> output_history;
};

OpenAIResponsesCreateRequest parse_openai_responses_create_request(const RequestJson& body,
                                                                   const RequestLimits& limits);

// The result turn of a tool_search_output: the `tools` it carries are loaded (rendered after the
// turn's content) and parsed as request tools would be, registering their identities. History
// replay and a hosted search build it the same way, so both render the same tokens.
ChatTurn lower_openai_tool_search_output(
    const RequestJson& tools, std::string call_id,
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities);

// A hosted tool search the server ran for one model call. The result turn is lowered from `tools`
// exactly as the response serializes it, so a client that replays the Item renders the same tokens.
struct OpenAIToolSearchExecution {
    nlohmann::json arguments; // the call's arguments, as its tool_search_call Item shows them
    nlohmann::json tools;     // the tool_search_output `tools`
    ChatTurn turn;            // the result turn the model reads
};

// Loads the catalog paths a hosted `tool_search` call names (`paths`: namespace or deferred
// function names); unknown names load nothing.
OpenAIToolSearchExecution execute_openai_tool_search(const OpenAIResponsesCreateRequest& request,
                                                     const ToolCall& call);

OpenAIResponsesPromptRequest
parse_openai_responses_input_tokens_request(const RequestJson& body, const RequestLimits& limits);

OpenAIResponsesResolvedPrompt
resolve_openai_responses_prompt(const OpenAIResponsesPromptRequest& request,
                                OpenAIResponsesStore& store);

BuiltOpenAIResponse make_openai_response_object(const std::string& id, std::int64_t created_at,
                                                const OpenAIResponsesCreateRequest& request,
                                                const OpenAIResponsesRuntimeValues& runtime,
                                                const GenerationOutcome& outcome);

// One generation of a Responses response. A response whose model calls a hosted tool_search has
// several: each but the last ended in hosted searches, and the server continued after their
// tool_search_output Items.
struct OpenAIResponsesRound {
    const GenerationOutcome* outcome = nullptr;
    // Per call: the call identifier the server used (a hosted search's is its tool_search_call
    // Item id); empty lets the response assign one.
    std::vector<std::string> call_ids;
    // Per call: the hosted search the server ran for it.
    std::vector<std::optional<OpenAIToolSearchExecution>> searches;
};

// A response of several generations; `totals` carries the summed usage (combine_round_outcomes).
// `finish` replaces the final generation's finish reason for the status: the output budget or the
// context ran out after a hosted search.
BuiltOpenAIResponse make_openai_response_object(const std::string& id, std::int64_t created_at,
                                                const OpenAIResponsesCreateRequest& request,
                                                const OpenAIResponsesRuntimeValues& runtime,
                                                std::span<const OpenAIResponsesRound> rounds,
                                                const GenerationOutcome& totals,
                                                std::optional<infernix::FinishReason> finish);

std::string make_openai_response_input_tokens_body(int input_tokens);

struct OpenAIResponsesStreamFinish {
    BuiltOpenAIResponse response;
    std::vector<std::string> events_before_terminal;
};

class OpenAIResponsesEventStream {
public:
    OpenAIResponsesEventStream(std::string response_id, std::int64_t created_at,
                               OpenAIResponsesCreateRequest request,
                               OpenAIResponsesRuntimeValues runtime);
    ~OpenAIResponsesEventStream();
    OpenAIResponsesEventStream(OpenAIResponsesEventStream&&) noexcept;
    OpenAIResponsesEventStream& operator=(OpenAIResponsesEventStream&&) noexcept;

    OpenAIResponsesEventStream(const OpenAIResponsesEventStream&)            = delete;
    OpenAIResponsesEventStream& operator=(const OpenAIResponsesEventStream&) = delete;

    std::vector<std::string> start();
    std::vector<std::string> reasoning_delta(const std::string& text);
    std::vector<std::string> content_delta(const std::string& text);
    // A constrained call's step: its first adds the call's Item; a function call streams its
    // arguments and completes with its last step, other calls complete when their round ends.
    std::vector<std::string> tool_call_delta(const infernix::ToolCallDelta& delta);
    // The identifier of the current generation's call `index` when it is a hosted search: the Item id
    // it streamed with, or a new one.
    [[nodiscard]] std::string hosted_search_id(std::size_t index) const;
    // Ends a generation the server continues after its hosted searches: closes its Items and
    // emits the rest of its calls and its tool_search_output Items. The next deltas open new Items.
    std::vector<std::string> round(const OpenAIResponsesRound& round);
    OpenAIResponsesStreamFinish finish(const GenerationOutcome& outcome);
    // `rounds` holds every generation, the final one last; earlier ones were passed to round().
    OpenAIResponsesStreamFinish finish(std::span<const OpenAIResponsesRound> rounds,
                                       const GenerationOutcome& totals,
                                       std::optional<infernix::FinishReason> finish_reason);
    std::string terminal(const BuiltOpenAIResponse& response);
    std::string failed(const ApiError& error);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infernix::serve
