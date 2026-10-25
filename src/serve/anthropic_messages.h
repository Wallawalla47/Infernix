#pragma once

// Anthropic Messages wire adapter. Request parsing lowers executable semantics into the common
// GenerationRequest; response construction owns Anthropic aggregate, SSE, usage, and error shapes.
// Visible Thinking text owns prompt semantics. With `thinking.display:"summarized"` its wire
// signature is response metadata and never enters common request or Engine state. With
// `display:"omitted"` the block's `thinking` is empty and the signature carries the reasoning
// itself, so request lowering restores it from the returned block. A signature without the Infernix
// marker stays non-semantic transport metadata.

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

struct GenerationOutcome;
class ToolSearchIndex;

enum class AnthropicToolSearchVariant : std::uint8_t {
    Regex, // tool_search_tool_regex_20251119: `pattern`, a Python regular expression
    Bm25,  // tool_search_tool_bm25_20251119: `query`, natural language
};

// A declared server-side tool search tool. The model sees it as an ordinary function; Infernix
// runs the search when the model calls it and continues the response after the result.
struct AnthropicToolSearch {
    std::string name;
    AnthropicToolSearchVariant variant = AnthropicToolSearchVariant::Regex;
};

struct AnthropicMessagesRequest {
    std::string model;
    GenerationRequest generation;
    bool stream                 = false;
    bool output_tokens_explicit = false;
    // `thinking.display:"omitted"`: reasoning travels only inside the block signature.
    bool hide_thinking = false;
    std::vector<AnthropicToolSearch> tool_searches;
};

// Hidden-reasoning signature: a marker followed by the Base64 reasoning text.
std::string encode_thinking_signature(std::string_view reasoning);
// The reasoning carried by a signature, or nullopt when the signature is not an Infernix one.
// Throws std::invalid_argument when the marker is present but the payload is malformed.
std::optional<std::string> decode_thinking_signature(std::string_view signature);

struct AnthropicCountTokensRequest {
    std::string model;
    GenerationRequest generation;
};

// limits.claude_code_prompt selects the Claude Code System rewrite (--claude-code-prompt): with
// Local, a System array whose consumed first block is Claude Code's attribution header and whose
// next block opens with its identity line is rewritten for a local model; anything else is lowered
// as sent. Count Tokens takes the same setting, so it counts the prompt Messages would run.
AnthropicMessagesRequest parse_anthropic_messages_request(const RequestJson& body,
                                                          const RequestLimits& limits);
AnthropicCountTokensRequest parse_anthropic_count_tokens_request(const RequestJson& body,
                                                                 const RequestLimits& limits = {});

// Tool search (deferred tools). The model reads a search result as a tool result: no text when it
// loads tools (the loaded definitions follow, ChatTurn::loaded_tools), a fixed sentence when it
// matched nothing, and the error code and message when it failed. Lowering depends on the wire
// content and the request's tool definitions alone, so the continuation Infernix runs after a
// search and a later request that replays the response as history render the same tokens.
//
// Lowers one `tool_search_tool_result` content object; throws ApiException for a malformed object
// or a tool_reference to a tool the request does not declare.
ChatTurn lower_anthropic_tool_search_result(const RequestJson& content, std::string tool_use_id,
                                            const GenerationRequest& request);

// One executed search: the `tool_search_tool_result` content and the result turn the model reads.
struct AnthropicToolSearchResult {
    RequestJson content;
    ChatTurn turn;
};

// Runs the model's calls of the request's tool search tools over its deferred tools. The search
// index is built on the first call. The request must outlive the executor.
class AnthropicToolSearchExecutor {
public:
    explicit AnthropicToolSearchExecutor(const AnthropicMessagesRequest& request);
    ~AnthropicToolSearchExecutor();
    AnthropicToolSearchExecutor(const AnthropicToolSearchExecutor&)            = delete;
    AnthropicToolSearchExecutor& operator=(const AnthropicToolSearchExecutor&) = delete;

    // The declared search tool called `name`, or nullptr for a client-executed tool.
    [[nodiscard]] const AnthropicToolSearch* search_tool(std::string_view name) const;
    // Runs `call` of a search tool. Invalid input and an exhausted search budget become error
    // results the model reads, as Anthropic returns them with HTTP 200.
    [[nodiscard]] AnthropicToolSearchResult execute(const ToolCall& call);
    // The search tools' names, for lifting a tool choice that a search satisfied.
    [[nodiscard]] const std::vector<std::string>& names() const noexcept { return names_; }

private:
    const AnthropicMessagesRequest& request_;
    std::vector<std::string> names_;
    std::vector<ToolDefinition> deferred_;
    std::unique_ptr<ToolSearchIndex> index_;
};

struct AnthropicResponseIdentity {
    std::string request_id;
    std::string message_id;
    std::string model;
};

std::string new_anthropic_request_id();
AnthropicResponseIdentity make_anthropic_response_identity(std::string request_id,
                                                           std::string model);

// A generation's calls with wire identifiers: `srvtoolu_` for the request's search tools, which
// Infernix executes, and `toolu_` for client tools.
std::vector<ToolCall> identify_anthropic_tool_calls(const GenerationOutcome& outcome,
                                                    std::span<const AnthropicToolSearch> searches);

// One generation of a response. A response that ran tool searches has several: each but the last
// ended in searches the server executed, and the next continued after their results.
struct AnthropicResponseSegment {
    const GenerationOutcome* outcome = nullptr;
    std::vector<ToolCall> calls; // the generation's calls with wire identifiers, in call order
    // Per call: the executed search's `tool_search_tool_result` content; nullopt for client tools.
    std::vector<std::optional<RequestJson>> search_results;
};

ApiError normalize_anthropic_error(ApiError error);
std::string make_anthropic_error_body(const ApiError& error, const std::string& request_id);
std::string make_anthropic_sse_error(const ApiError& error, const std::string& request_id);

std::string make_anthropic_messages_response(const AnthropicResponseIdentity& identity,
                                             const GenerationOutcome& outcome,
                                             bool hide_thinking = false);
// A response of several generations. `totals` carries the summed usage and timings (see
// combine_round_outcomes); `stop_reason` replaces the final generation's (`pause_turn`, or
// `max_tokens` when the budget ran out after a search).
std::string make_anthropic_messages_response(const AnthropicResponseIdentity& identity,
                                             std::span<const AnthropicResponseSegment> segments,
                                             const GenerationOutcome& totals,
                                             std::optional<std::string_view> stop_reason,
                                             bool hide_thinking = false);
std::string make_anthropic_count_tokens_response(int input_tokens);

class AnthropicMessagesStream {
public:
    // `searches` are the request's search tools: their calls stream as `server_tool_use` blocks.
    AnthropicMessagesStream(AnthropicResponseIdentity identity, int input_tokens,
                            bool hide_thinking = false, std::vector<AnthropicToolSearch> searches = {});

    // The Engine start event is exact for normal streams. The no-argument form is reserved for an
    // error raised before admission, so an Anthropic error event still has a valid stream prefix.
    std::string start();
    std::string start(const infernix::GenerationStart& generation);

    [[nodiscard]] bool started() const noexcept { return started_; }

    std::vector<std::string> reasoning_delta(const std::string& text);
    std::vector<std::string> content_delta(const std::string& text);
    // A constrained call's step: its first opens the call's block, every one extends its input, the
    // complete one closes the block.
    std::vector<std::string> tool_call_delta(const infernix::ToolCallDelta& delta);
    // The current generation's calls with wire identifiers: those it streamed keep theirs. A
    // segment of the generation carries them.
    [[nodiscard]] std::vector<ToolCall> identify_calls(const GenerationOutcome& outcome) const;
    // Ends a generation that the server continues: closes its blocks, then emits the calls it did
    // not stream and its search results. The next generation's deltas open new blocks in the same
    // message.
    std::vector<std::string> segment(const AnthropicResponseSegment& segment);
    std::vector<std::string> finish(const GenerationOutcome& outcome);
    std::vector<std::string> finish(const AnthropicResponseSegment& final_segment,
                                    const GenerationOutcome& totals,
                                    std::optional<std::string_view> stop_reason);
    std::string error(const ApiError& api_error) const;

private:
    std::string start_with_cache(std::optional<int> cache_read_input_tokens);
    std::vector<std::string> close_thinking();
    std::vector<std::string> close_text();
    std::vector<std::string> close_generation(const GenerationOutcome& outcome);
    std::vector<std::string> call_blocks(const AnthropicResponseSegment& segment);

    struct StreamedCall {
        ToolCall call;
        int index     = 0;
        bool complete = false;
    };

    AnthropicResponseIdentity identity_;
    std::vector<AnthropicToolSearch> searches_;
    std::string reasoning_;
    std::string content_;
    std::vector<StreamedCall> calls_; // the current generation's streamed calls
    int input_tokens_   = 0;
    bool hide_thinking_ = false;
    int next_index_     = 0;
    int thinking_index_ = -1;
    int text_index_     = -1;
    bool started_       = false;
    bool finished_      = false;
    bool thinking_open_ = false;
    bool text_open_     = false;
};

} // namespace infernix::serve
