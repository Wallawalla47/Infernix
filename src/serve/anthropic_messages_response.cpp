#include "serve/anthropic_messages.h"
#include "product/constraint_observation.h"

#include "serve/completion_timings.h"
#include "serve/generation_service.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace infernix::serve {
namespace {

using Json        = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

std::string random_identifier(const char* prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx",
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + buffer.data();
}

constexpr std::string_view kThinkingSignatureMarker = "ninfer-reasoning.v1:";
constexpr std::string_view kBase64Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string event(const char* type, Json payload) {
    return std::string("event: ") + type + "\ndata: " + payload.dump() + "\n\n";
}

OrderedJson parse_tool_input(const ToolCall& call) {
    OrderedJson input = OrderedJson::parse(call.arguments_json, nullptr, false);
    if (input.is_discarded() || !input.is_object()) {
        throw std::logic_error("Engine produced non-object Anthropic tool input");
    }
    return input;
}

// A single-generation response: every call is a client call.
AnthropicResponseSegment client_segment(const GenerationOutcome& outcome) {
    AnthropicResponseSegment segment;
    segment.outcome = &outcome;
    segment.calls   = identify_anthropic_tool_calls(outcome, {});
    segment.search_results.resize(segment.calls.size());
    return segment;
}

void validate_segment(const AnthropicResponseSegment& segment) {
    if (segment.outcome == nullptr || segment.calls.size() != segment.outcome->tool_calls.size() ||
        segment.search_results.size() != segment.calls.size()) {
        throw std::logic_error("Anthropic response segment does not match its generation");
    }
}

// A generation's calls in call order, then the results of the searches among them: the order in
// which the server ran them, and the one history lowering reads back as one generation.
std::vector<OrderedJson> call_blocks_json(const AnthropicResponseSegment& segment) {
    std::vector<OrderedJson> blocks;
    for (std::size_t index = 0; index < segment.calls.size(); ++index) {
        const ToolCall& call = segment.calls[index];
        blocks.push_back(OrderedJson{{"type", segment.search_results[index] ? "server_tool_use" : "tool_use"},
                                     {"id", call.id},
                                     {"name", call.name},
                                     {"input", parse_tool_input(call)}});
    }
    for (std::size_t index = 0; index < segment.calls.size(); ++index) {
        if (const auto& result = segment.search_results[index]) {
            blocks.push_back(OrderedJson{{"type", "tool_search_tool_result"},
                                         {"tool_use_id", segment.calls[index].id},
                                         {"content", *result}});
        }
    }
    return blocks;
}

struct StopPresentation {
    std::string reason = "end_turn";
    Json sequence      = nullptr;
};

StopPresentation stop_presentation(const GenerationOutcome& outcome);

// A response that ran searches may end for a reason of its own: `pause_turn` when it reached the
// generation limit, `max_tokens` when the output budget ran out after a search.
StopPresentation stop_presentation(const GenerationOutcome& outcome,
                                   std::optional<std::string_view> stop_reason) {
    if (stop_reason) { return StopPresentation{.reason = std::string(*stop_reason)}; }
    return stop_presentation(outcome);
}

StopPresentation stop_presentation(const GenerationOutcome& outcome) {
    if (!outcome.tool_calls.empty() && (outcome.finish_reason == infernix::FinishReason::StopToken ||
                                        outcome.finish_reason == infernix::FinishReason::None))
        return StopPresentation{.reason = "tool_use"};
    switch (outcome.finish_reason) {
    case infernix::FinishReason::OutputLimit:
        return StopPresentation{.reason = "max_tokens"};
    case infernix::FinishReason::ContextCapacity:
        return StopPresentation{.reason = "model_context_window_exceeded"};
    case infernix::FinishReason::StopString:
        if (!outcome.matched_stop_string) {
            throw std::logic_error("stop-string terminal result has no matched declaration");
        }
        return StopPresentation{.reason   = "stop_sequence",
                                .sequence = *outcome.matched_stop_string};
    case infernix::FinishReason::StopToken:
    case infernix::FinishReason::None:
        return StopPresentation{.reason = "end_turn"};
    case infernix::FinishReason::Cancelled:
        throw std::logic_error("cancelled generation cannot be serialized as an Anthropic message");
    }
    throw std::logic_error("unknown Engine finish reason");
}

Json final_usage(const GenerationOutcome& outcome) {
    const int prompt = std::max(0, outcome.prompt_tokens);
    const int cached = static_cast<int>(
        std::min<std::uint64_t>(outcome.reused_prompt_tokens, static_cast<std::uint64_t>(prompt)));
    // As in the Anthropic API, input_tokens counts the prompt tokens neither read from the cache nor
    // written to it; creation is unknown (null) without the prefix cache.
    std::optional<int> created;
    if (const auto& written = outcome.cache_created_prompt_tokens) {
        created = static_cast<int>(std::min<std::uint64_t>(*written, static_cast<std::uint64_t>(prompt - cached)));
    }
    return Json{
        {"input_tokens", prompt - cached - created.value_or(0)},
        {"cache_creation_input_tokens", created ? Json(*created) : Json(nullptr)},
        {"cache_read_input_tokens", cached},
        {"cache_creation", nullptr},
        {"output_tokens", std::max(0, outcome.completion_tokens)},
        {"output_tokens_details", Json{{"thinking_tokens", std::max(0, outcome.reasoning_tokens)}}},
        {"server_tool_use", Json{{"web_search_requests", 0}, {"web_fetch_requests", 0}}},
        {"service_tier", nullptr},
        {"inference_geo", nullptr}};
}

Json streaming_start_usage(int input_tokens, std::optional<int> cache_read_input_tokens) {
    const int prompt = std::max(0, input_tokens);
    std::optional<int> cached;
    if (cache_read_input_tokens) { cached = std::clamp(*cache_read_input_tokens, 0, prompt); }
    return Json{{"input_tokens", cached ? prompt - *cached : prompt},
                {"cache_creation_input_tokens", nullptr},
                {"cache_read_input_tokens", cached ? Json(*cached) : Json(nullptr)},
                {"cache_creation", nullptr},
                {"output_tokens", 0},
                {"server_tool_use", Json{{"web_search_requests", 0}, {"web_fetch_requests", 0}}},
                {"service_tier", nullptr},
                {"inference_geo", nullptr}};
}

void append(std::vector<std::string>& target, std::vector<std::string> source) {
    target.insert(target.end(), std::make_move_iterator(source.begin()),
                  std::make_move_iterator(source.end()));
}

void require_prefix(std::string_view complete, std::string_view streamed, const char* channel) {
    if (!complete.starts_with(streamed)) {
        throw std::logic_error(std::string("streamed Anthropic ") + channel +
                               " does not match terminal output");
    }
}

bool is_search_tool(std::span<const AnthropicToolSearch> searches, std::string_view name) {
    return std::any_of(searches.begin(), searches.end(),
                       [&](const AnthropicToolSearch& search) { return search.name == name; });
}

// `srvtoolu_` for the request's search tools, which Infernix executes, and `toolu_` for client tools.
std::string anthropic_call_id(std::span<const AnthropicToolSearch> searches, std::string_view name) {
    return random_identifier(is_search_tool(searches, name) ? "srvtoolu_" : "toolu_");
}

} // namespace

std::string new_anthropic_request_id() { return random_identifier("req_"); }

AnthropicResponseIdentity make_anthropic_response_identity(std::string request_id,
                                                           std::string model) {
    return AnthropicResponseIdentity{.request_id = std::move(request_id),
                                     .message_id = random_identifier("msg_"),
                                     .model      = std::move(model)};
}

ApiError normalize_anthropic_error(ApiError error) {
    if (error.param == "reasoning_effort") { error.param = "output_config.effort"; }
    if (const auto& details = error.context_length) {
        // The Anthropic API's wording, which Claude Code matches to compact the conversation.
        error.param   = {};
        error.message = "prompt is too long: " + std::string(details->exact ? "" : "more than ") +
                        std::to_string(details->prompt_tokens) + " tokens > " + std::to_string(details->max_context) +
                        " maximum" + vision_overflow_note(*details);
    }
    if (error.code == "server_overloaded" || error.status == 429) {
        error.status = 529;
        error.type   = "overloaded_error";
        return error;
    }
    if (error.code == "request_queue_timeout" || error.code == "media_fetch_timeout" ||
        error.status == 504) {
        error.status = 504;
        error.type   = "timeout_error";
        return error;
    }
    switch (error.status) {
    case 400:
        error.type = "invalid_request_error";
        break;
    case 401:
        error.type = "authentication_error";
        break;
    case 403:
        error.type = "permission_error";
        break;
    case 404:
        error.type = "not_found_error";
        break;
    case 413:
        error.type = "request_too_large";
        break;
    default:
        if (error.status >= 500) { error.type = "api_error"; }
        break;
    }
    return error;
}

std::string make_anthropic_error_body(const ApiError& api_error, const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    const std::string message =
        error.param.empty() ? error.message : error.param + ": " + error.message;
    return Json{{"type", "error"},
                {"error", Json{{"type", error.type}, {"message", message}}},
                {"request_id", request_id}}
        .dump();
}

std::string make_anthropic_sse_error(const ApiError& error, const std::string& request_id) {
    return std::string("event: error\ndata: ") + make_anthropic_error_body(error, request_id) +
           "\n\n";
}

std::string encode_thinking_signature(std::string_view reasoning) {
    std::string out(kThinkingSignatureMarker);
    out.reserve(out.size() + (reasoning.size() + 2) / 3 * 4);
    for (std::size_t i = 0; i < reasoning.size(); i += 3) {
        const std::size_t left = reasoning.size() - i;
        std::uint32_t group    = static_cast<std::uint32_t>(static_cast<std::uint8_t>(reasoning[i]))
                                 << 16;
        if (left > 1) {
            group |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(reasoning[i + 1])) << 8;
        }
        if (left > 2) { group |= static_cast<std::uint8_t>(reasoning[i + 2]); }
        out += kBase64Alphabet[group >> 18 & 63];
        out += kBase64Alphabet[group >> 12 & 63];
        out += left > 1 ? kBase64Alphabet[group >> 6 & 63] : '=';
        out += left > 2 ? kBase64Alphabet[group & 63] : '=';
    }
    return out;
}

std::optional<std::string> decode_thinking_signature(std::string_view signature) {
    if (!signature.starts_with(kThinkingSignatureMarker)) { return std::nullopt; }
    const std::string_view payload = signature.substr(kThinkingSignatureMarker.size());
    const auto malformed           = [] {
        return std::invalid_argument("malformed hidden-reasoning signature");
    };
    if (payload.size() % 4 != 0) { throw malformed(); }
    std::string out;
    out.reserve(payload.size() / 4 * 3);
    for (std::size_t i = 0; i < payload.size(); i += 4) {
        std::uint32_t group = 0;
        int data_chars      = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            const char c = payload[i + j];
            if (c == '=') {
                // Padding is valid only in the last two positions of the last group.
                if (i + 4 != payload.size() || j < 2) { throw malformed(); }
                group <<= 6;
                continue;
            }
            const std::size_t value = kBase64Alphabet.find(c);
            if (value == std::string_view::npos || data_chars != static_cast<int>(j)) {
                throw malformed();
            }
            group = group << 6 | static_cast<std::uint32_t>(value);
            ++data_chars;
        }
        out += static_cast<char>(group >> 16 & 255);
        if (data_chars > 2) { out += static_cast<char>(group >> 8 & 255); }
        if (data_chars > 3) { out += static_cast<char>(group & 255); }
    }
    return out;
}

std::vector<ToolCall> identify_anthropic_tool_calls(const GenerationOutcome& outcome,
                                                    std::span<const AnthropicToolSearch> searches) {
    std::vector<ToolCall> result;
    result.reserve(outcome.tool_calls.size());
    for (const infernix::GeneratedToolCall& call : outcome.tool_calls) {
        result.push_back(ToolCall{.id             = anthropic_call_id(searches, call.name),
                                  .name           = call.name,
                                  .arguments_json = call.arguments_json});
    }
    return result;
}

std::string make_anthropic_messages_response(const AnthropicResponseIdentity& identity,
                                             const GenerationOutcome& outcome, bool hide_thinking) {
    const AnthropicResponseSegment segment = client_segment(outcome);
    return make_anthropic_messages_response(identity, std::span(&segment, 1), outcome,
                                            std::nullopt, hide_thinking);
}

std::string make_anthropic_messages_response(const AnthropicResponseIdentity& identity,
                                             std::span<const AnthropicResponseSegment> segments,
                                             const GenerationOutcome& totals,
                                             std::optional<std::string_view> stop_reason,
                                             bool hide_thinking) {
    if (segments.empty()) { throw std::logic_error("Anthropic response has no generation"); }
    OrderedJson content = OrderedJson::array();
    for (const AnthropicResponseSegment& segment : segments) {
        validate_segment(segment);
        const GenerationOutcome& outcome = *segment.outcome;
        if (!outcome.reasoning.empty()) {
            // Claude clients expect a non-empty opaque value. Summarized display uses the response
            // identity, so Thinking introduces no independent state. Omitted display makes the
            // signature the only carrier of the reasoning, restored when the client returns the
            // block.
            content.push_back(OrderedJson{
                {"type", "thinking"},
                {"thinking", hide_thinking ? std::string() : outcome.reasoning},
                {"signature", hide_thinking ? encode_thinking_signature(outcome.reasoning)
                                            : identity.message_id}});
        }
        if (!outcome.text.empty()) {
            content.push_back(OrderedJson{{"type", "text"}, {"text", outcome.text}});
        }
        for (OrderedJson& block : call_blocks_json(segment)) { content.push_back(std::move(block)); }
    }
    const GenerationOutcome& final_outcome = *segments.back().outcome;
    const StopPresentation stop            = stop_presentation(final_outcome, stop_reason);
    OrderedJson response{{"id", identity.message_id},
                         {"type", "message"},
                         {"role", "assistant"},
                         {"model", identity.model},
                         {"content", std::move(content)},
                         {"stop_reason", stop.reason},
                         {"stop_sequence", stop.sequence},
                         {"usage", final_usage(totals)}};
    if (final_outcome.constraint)
        response["constraint"] = product::constraint_observation_json(final_outcome.constraint);
    response["timings"] = timings_json(outcome_timings(totals));
    return response.dump();
}

std::string make_anthropic_count_tokens_response(int input_tokens) {
    return Json{{"input_tokens", input_tokens}}.dump();
}

AnthropicMessagesStream::AnthropicMessagesStream(AnthropicResponseIdentity identity,
                                                 int input_tokens, bool hide_thinking,
                                                 std::vector<AnthropicToolSearch> searches)
    : identity_(std::move(identity)), searches_(std::move(searches)), input_tokens_(input_tokens),
      hide_thinking_(hide_thinking) {}

std::string AnthropicMessagesStream::start() { return start_with_cache(std::nullopt); }

std::string AnthropicMessagesStream::start(const infernix::GenerationStart& generation) {
    if (generation.prompt.prompt_tokens != static_cast<std::uint32_t>(input_tokens_)) {
        throw std::logic_error("Anthropic stream prompt count differs from Engine start");
    }
    return start_with_cache(static_cast<int>(generation.reused_prompt_tokens));
}

std::string AnthropicMessagesStream::start_with_cache(std::optional<int> cache_read_input_tokens) {
    if (started_ || finished_) { throw std::logic_error("Anthropic stream already started"); }
    started_ = true;
    const Json message{{"id", identity_.message_id},
                       {"type", "message"},
                       {"role", "assistant"},
                       {"model", identity_.model},
                       {"content", Json::array()},
                       {"stop_reason", nullptr},
                       {"stop_sequence", nullptr},
                       {"usage", streaming_start_usage(input_tokens_, cache_read_input_tokens)}};
    return event("message_start", Json{{"type", "message_start"}, {"message", message}});
}

std::vector<std::string> AnthropicMessagesStream::reasoning_delta(const std::string& text) {
    if (!started_ || finished_ || text_open_) {
        throw std::logic_error("invalid Anthropic reasoning delta state");
    }
    std::vector<std::string> events;
    if (!thinking_open_) {
        thinking_index_ = next_index_++;
        thinking_open_  = true;
        events.push_back(
            event("content_block_start",
                  Json{{"type", "content_block_start"},
                       {"index", thinking_index_},
                       {"content_block",
                        Json{{"type", "thinking"}, {"thinking", ""}, {"signature", ""}}}}));
    }
    reasoning_ += text;
    if (!text.empty() && !hide_thinking_) {
        events.push_back(
            event("content_block_delta",
                  Json{{"type", "content_block_delta"},
                       {"index", thinking_index_},
                       {"delta", Json{{"type", "thinking_delta"}, {"thinking", text}}}}));
    }
    return events;
}

std::vector<std::string> AnthropicMessagesStream::close_thinking() {
    std::vector<std::string> events;
    if (!thinking_open_) { return events; }
    const std::string signature =
        hide_thinking_ ? encode_thinking_signature(reasoning_) : identity_.message_id;
    events.push_back(event("content_block_delta", Json{{"type", "content_block_delta"},
                                                       {"index", thinking_index_},
                                                       {"delta", Json{{"type", "signature_delta"},
                                                                      {"signature", signature}}}}));
    events.push_back(event("content_block_stop",
                           Json{{"type", "content_block_stop"}, {"index", thinking_index_}}));
    thinking_open_ = false;
    return events;
}

std::vector<std::string> AnthropicMessagesStream::content_delta(const std::string& text) {
    if (!started_ || finished_) { throw std::logic_error("invalid Anthropic text delta state"); }
    std::vector<std::string> events = close_thinking();
    if (!text_open_) {
        text_index_ = next_index_++;
        text_open_  = true;
        events.push_back(event("content_block_start",
                               Json{{"type", "content_block_start"},
                                    {"index", text_index_},
                                    {"content_block", Json{{"type", "text"}, {"text", ""}}}}));
    }
    content_ += text;
    if (!text.empty()) {
        events.push_back(event("content_block_delta",
                               Json{{"type", "content_block_delta"},
                                    {"index", text_index_},
                                    {"delta", Json{{"type", "text_delta"}, {"text", text}}}}));
    }
    return events;
}

std::vector<std::string> AnthropicMessagesStream::close_text() {
    if (!text_open_) { return {}; }
    text_open_ = false;
    return {
        event("content_block_stop", Json{{"type", "content_block_stop"}, {"index", text_index_}})};
}

// Streams what the deltas have not shown of a generation's output and closes its blocks. The next
// generation's deltas open new blocks.
std::vector<std::string> AnthropicMessagesStream::close_generation(const GenerationOutcome& outcome) {
    require_prefix(outcome.reasoning, reasoning_, "reasoning");
    require_prefix(outcome.text, content_, "content");

    std::vector<std::string> events;
    const std::string reasoning_suffix = outcome.reasoning.substr(reasoning_.size());
    if (!reasoning_suffix.empty()) {
        if (text_open_) {
            throw std::logic_error("terminal Anthropic reasoning appeared after streamed text");
        }
        append(events, reasoning_delta(reasoning_suffix));
    }
    const std::string content_suffix = outcome.text.substr(content_.size());
    if (!content_suffix.empty()) { append(events, content_delta(content_suffix)); }
    append(events, close_thinking());
    append(events, close_text());
    reasoning_.clear();
    content_.clear();
    return events;
}

std::vector<std::string> AnthropicMessagesStream::tool_call_delta(const infernix::ToolCallDelta& delta) {
    const bool opens = delta.index == calls_.size();
    if (!started_ || finished_ || (opens && delta.name.empty()) ||
        (!opens && (delta.index + 1U != calls_.size() || calls_.back().complete || !delta.name.empty()))) {
        throw std::logic_error("invalid Anthropic tool-call delta state");
    }
    std::vector<std::string> events;
    if (opens) {
        append(events, close_thinking());
        append(events, close_text());
        calls_.push_back(StreamedCall{
            .call  = ToolCall{.id = anthropic_call_id(searches_, delta.name), .name = delta.name},
            .index = next_index_++});
        const char* type = is_search_tool(searches_, delta.name) ? "server_tool_use" : "tool_use";
        events.push_back(event("content_block_start",
                               Json{{"type", "content_block_start"},
                                    {"index", calls_.back().index},
                                    {"content_block", Json{{"type", type},
                                                           {"id", calls_.back().call.id},
                                                           {"name", delta.name},
                                                           {"input", Json::object()}}}}));
    }
    StreamedCall& call = calls_.back();
    call.call.arguments_json += delta.arguments;
    if (!delta.arguments.empty()) {
        events.push_back(event("content_block_delta",
                               Json{{"type", "content_block_delta"},
                                    {"index", call.index},
                                    {"delta", Json{{"type", "input_json_delta"},
                                                   {"partial_json", delta.arguments}}}}));
    }
    if (delta.complete) {
        call.complete = true;
        events.push_back(event("content_block_stop",
                               Json{{"type", "content_block_stop"}, {"index", call.index}}));
    }
    return events;
}

std::vector<ToolCall> AnthropicMessagesStream::identify_calls(const GenerationOutcome& outcome) const {
    std::vector<ToolCall> calls = identify_anthropic_tool_calls(outcome, searches_);
    for (std::size_t index = 0; index < calls.size() && index < calls_.size(); ++index) {
        calls[index].id = calls_[index].call.id;
    }
    return calls;
}

std::vector<std::string> AnthropicMessagesStream::call_blocks(const AnthropicResponseSegment& segment) {
    validate_segment(segment);
    std::vector<std::string> events;
    if (!calls_.empty()) {
        // Streamed calls are the segment's, and at most one more that the turn's end cut: its block
        // closes now.
        bool matches = calls_.size() == segment.calls.size() ||
                       (calls_.size() == segment.calls.size() + 1U && !calls_.back().complete);
        for (std::size_t position = 0; matches && position < segment.calls.size(); ++position) {
            const ToolCall& call = segment.calls[position];
            matches = calls_[position].complete && calls_[position].call.id == call.id &&
                      calls_[position].call.name == call.name &&
                      calls_[position].call.arguments_json == call.arguments_json;
        }
        if (!matches) { throw std::logic_error("streamed Anthropic tool calls do not match terminal output"); }
        if (!calls_.back().complete) {
            events.push_back(event("content_block_stop",
                                   Json{{"type", "content_block_stop"}, {"index", calls_.back().index}}));
        }
    }
    // Free tool output's calls arrive whole.
    for (std::size_t position = calls_.empty() ? 0 : segment.calls.size(); position < segment.calls.size();
         ++position) {
        const ToolCall& call = segment.calls[position];
        (void)parse_tool_input(call);
        const int index = next_index_++;
        const char* type = segment.search_results[position] ? "server_tool_use" : "tool_use";
        events.push_back(
            event("content_block_start", Json{{"type", "content_block_start"},
                                              {"index", index},
                                              {"content_block", Json{{"type", type},
                                                                     {"id", call.id},
                                                                     {"name", call.name},
                                                                     {"input", Json::object()}}}}));
        events.push_back(event("content_block_delta",
                               Json{{"type", "content_block_delta"},
                                    {"index", index},
                                    {"delta", Json{{"type", "input_json_delta"},
                                                   {"partial_json", call.arguments_json}}}}));
        events.push_back(
            event("content_block_stop", Json{{"type", "content_block_stop"}, {"index", index}}));
    }
    calls_.clear();
    // A search result arrives whole, as Anthropic streams it: one start event carrying the content.
    for (std::size_t position = 0; position < segment.calls.size(); ++position) {
        const auto& result = segment.search_results[position];
        if (!result) { continue; }
        const int index = next_index_++;
        events.push_back(event("content_block_start",
                               Json{{"type", "content_block_start"},
                                    {"index", index},
                                    {"content_block",
                                     Json{{"type", "tool_search_tool_result"},
                                          {"tool_use_id", segment.calls[position].id},
                                          {"content", Json::parse(result->dump())}}}}));
        events.push_back(
            event("content_block_stop", Json{{"type", "content_block_stop"}, {"index", index}}));
    }
    return events;
}

std::vector<std::string> AnthropicMessagesStream::segment(const AnthropicResponseSegment& segment) {
    if (!started_ || finished_) { throw std::logic_error("invalid Anthropic stream segment state"); }
    validate_segment(segment);
    std::vector<std::string> events = close_generation(*segment.outcome);
    append(events, call_blocks(segment));
    return events;
}

std::vector<std::string> AnthropicMessagesStream::finish(const GenerationOutcome& outcome) {
    AnthropicResponseSegment segment{.outcome = &outcome, .calls = identify_calls(outcome), .search_results = {}};
    segment.search_results.resize(segment.calls.size());
    return finish(segment, outcome, std::nullopt);
}

std::vector<std::string> AnthropicMessagesStream::finish(const AnthropicResponseSegment& final_segment,
                                                         const GenerationOutcome& totals,
                                                         std::optional<std::string_view> stop_reason) {
    if (!started_ || finished_) { throw std::logic_error("invalid Anthropic stream finish state"); }
    validate_segment(final_segment);
    const GenerationOutcome& outcome = *final_segment.outcome;
    std::vector<std::string> events  = close_generation(outcome);
    append(events, call_blocks(final_segment));

    const StopPresentation stop = stop_presentation(outcome, stop_reason);
    Json terminal{{"type", "message_delta"},
                  {"delta", Json{{"stop_reason", stop.reason}, {"stop_sequence", stop.sequence}}},
                  {"usage", final_usage(totals)}};
    if (outcome.constraint)
        terminal["constraint"] = product::constraint_observation_json(outcome.constraint);
    terminal["timings"] = timings_json(outcome_timings(totals));
    events.push_back(event("message_delta", std::move(terminal)));
    events.push_back(event("message_stop", Json{{"type", "message_stop"}}));
    finished_ = true;
    return events;
}

std::string AnthropicMessagesStream::error(const ApiError& api_error) const {
    return make_anthropic_sse_error(api_error, identity_.request_id);
}

} // namespace infernix::serve
