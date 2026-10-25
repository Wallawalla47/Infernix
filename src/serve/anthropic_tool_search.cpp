#include "serve/anthropic_messages.h"

#include "serve/request_validation.h"
#include "serve/tool_search.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

using Json = RequestJson;

constexpr std::string_view kNoMatch = "No deferred tools matched the search.";

ContentPart text_content(std::string text) {
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "text";
    part.text     = std::move(text);
    return part;
}

const ToolDefinition& referenced_tool(const GenerationRequest& request, const std::string& name) {
    for (const ToolDefinition& tool : request.tools) {
        if (tool.name == name) { return tool; }
    }
    bad_request("Tool reference '" + name + "' not found in available tools", "messages",
                "tool_reference_not_found");
}

std::string require_member_string(const Json& object, const char* key, const char* block) {
    if (!object.contains(key) || !object.at(key).is_string()) {
        bad_request(std::string(block) + " must contain a string '" + key + "'", "messages");
    }
    return object.at(key).get<std::string>();
}

// The search input the model gave; ToolSearchInputError for anything the API calls invalid.
std::size_t search_limit(const Json& input) {
    if (!input.contains("limit") || input.at("limit").is_null()) { return kToolSearchDefaultLimit; }
    const Json& value = input.at("limit");
    if (!value.is_number_integer()) {
        throw ToolSearchInputError("limit must be an integer from 1 to " +
                                   std::to_string(kToolSearchMaximumLimit));
    }
    const bool in_range = value.is_number_unsigned()
                              ? value.get<std::uint64_t>() >= 1U &&
                                    value.get<std::uint64_t>() <= kToolSearchMaximumLimit
                              : value.get<std::int64_t>() >= 1 &&
                                    value.get<std::int64_t>() <=
                                        static_cast<std::int64_t>(kToolSearchMaximumLimit);
    if (!in_range) {
        throw ToolSearchInputError("limit must be an integer from 1 to " +
                                   std::to_string(kToolSearchMaximumLimit));
    }
    return value.is_number_unsigned() ? static_cast<std::size_t>(value.get<std::uint64_t>())
                                      : static_cast<std::size_t>(value.get<std::int64_t>());
}

Json search_error(const char* code, const std::string& message) {
    return Json{{"type", "tool_search_tool_result_error"},
                {"error_code", code},
                {"error_message", message}};
}

} // namespace

ChatTurn lower_anthropic_tool_search_result(const RequestJson& content, std::string tool_use_id,
                                            const GenerationRequest& request) {
    ChatTurn turn;
    turn.role         = ChatRole::Tool;
    turn.tool_call_id = std::move(tool_use_id);
    const std::string type =
        require_member_string(content, "type", "tool_search_tool_result content");
    if (type == "tool_search_tool_search_result") {
        if (!content.contains("tool_references") || !content.at("tool_references").is_array()) {
            bad_request("tool_search_tool_search_result must contain a tool_references array",
                        "messages");
        }
        for (const Json& reference : content.at("tool_references")) {
            if (!reference.is_object() ||
                (reference.contains("type") && reference.at("type") != "tool_reference")) {
                bad_request("tool_references entries must be tool_reference blocks", "messages");
            }
            const std::string name = require_member_string(reference, "tool_name", "tool_reference");
            const ToolDefinition& tool = referenced_tool(request, name);
            if (std::none_of(turn.loaded_tools.begin(), turn.loaded_tools.end(),
                             [&](const ToolDefinition& loaded) { return loaded.name == name; })) {
                ToolDefinition copy = tool;
                copy.cache_boundary_after.reset();
                turn.loaded_tools.push_back(std::move(copy));
            }
        }
        if (turn.loaded_tools.empty()) { turn.content.push_back(text_content(std::string(kNoMatch))); }
        return turn;
    }
    if (type == "tool_search_tool_result_error") {
        const std::string code = require_member_string(content, "error_code", "tool search error");
        std::string text       = "Tool search failed (" + code + ")";
        if (content.contains("error_message") && content.at("error_message").is_string()) {
            text += ": " + content.at("error_message").get<std::string>();
        }
        turn.tool_result_is_error = true;
        turn.content.push_back(text_content(std::move(text)));
        return turn;
    }
    bad_request("unsupported tool_search_tool_result content type: " + type, "messages",
                "content_block_not_supported");
}

AnthropicToolSearchExecutor::AnthropicToolSearchExecutor(const AnthropicMessagesRequest& request)
    : request_(request) {
    for (const AnthropicToolSearch& search : request.tool_searches) {
        names_.push_back(search.name);
    }
    for (const ToolDefinition& tool : request.generation.tools) {
        if (tool.deferred) { deferred_.push_back(tool); }
    }
}

AnthropicToolSearchExecutor::~AnthropicToolSearchExecutor() = default;

const AnthropicToolSearch* AnthropicToolSearchExecutor::search_tool(std::string_view name) const {
    for (const AnthropicToolSearch& search : request_.tool_searches) {
        if (search.name == name) { return &search; }
    }
    return nullptr;
}

AnthropicToolSearchResult AnthropicToolSearchExecutor::execute(const ToolCall& call) {
    const AnthropicToolSearch* search = search_tool(call.name);
    if (search == nullptr) { throw std::logic_error("tool search executor received a client call"); }
    const bool regex = search->variant == AnthropicToolSearchVariant::Regex;
    Json content;
    try {
        const Json input = Json::parse(call.arguments_json);
        if (!input.is_object()) { throw ToolSearchInputError("the search input must be an object"); }
        const char* key = regex ? "pattern" : "query";
        if (!input.contains(key) || !input.at(key).is_string()) {
            throw ToolSearchInputError(std::string("the search input must contain a string '") +
                                       key + "'");
        }
        const std::size_t limit = search_limit(input);
        const std::string& text = input.at(key).get_ref<const std::string&>();
        if (!index_) { index_ = std::make_unique<ToolSearchIndex>(deferred_); }
        const std::vector<std::size_t> found =
            regex ? index_->regex(text, limit) : index_->bm25(text, limit);
        Json references = Json::array();
        for (const std::size_t index : found) {
            references.push_back(
                Json{{"type", "tool_reference"}, {"tool_name", deferred_[index].name}});
        }
        content = Json{{"type", "tool_search_tool_search_result"},
                       {"tool_references", std::move(references)}};
    } catch (const ToolSearchInputError& error) {
        content = search_error("invalid_tool_input", error.what());
    } catch (const ToolSearchBudgetExceeded& error) {
        content = search_error("execution_time_exceeded", error.what());
    } catch (const Json::exception& error) {
        content = search_error("invalid_tool_input", error.what());
    }
    ChatTurn turn = lower_anthropic_tool_search_result(content, call.id, request_.generation);
    return AnthropicToolSearchResult{.content = std::move(content), .turn = std::move(turn)};
}

} // namespace infernix::serve
