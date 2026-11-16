#pragma once

#include "product/media_acquire/source.h"

#include <infernix/types.h>

// Internal, wire-format-independent representation of a generation request.
//
// OpenAI and Anthropic schemas map into this wire-independent value.
// translate.cpp then produces the public PromptInput and RequestOptions consumed
// by Engine; media sources remain unresolved until the product service acquires
// owning bytes.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

// A structured API error mapped onto an error object + HTTP status. Wire-format
// independent: each protocol layer renders it into its own error body shape.
struct ApiError {
    int status       = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param; // optional
    std::string code;  // optional
    // A context overflow's counts, so each protocol can word it as its clients detect it.
    std::optional<ContextLengthDetails> context_length;
    // A queue-pressure answer's Retry-After (seconds; 0 sends none): the server's admission window.
    std::uint32_t retry_after_seconds = 0;
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

// --claude-code-prompt: what Anthropic Messages lowering does with Claude Code's system prompt.
enum class ClaudeCodePrompt : std::uint8_t {
    AsSent, // the client's System text, unchanged
    Local,  // Claude-specific sections replaced by a short local-agent preamble
};

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    int default_max_tokens              = 8192;
    ClaudeCodePrompt claude_code_prompt = ClaudeCodePrompt::AsSent; // Anthropic Messages only
};

enum class ContentKind {
    Text,
    Image,
    Video,
    Document, // a document or search result, resolved before translation (documents.h)
};

struct CacheBoundary {
    enum class Ttl : std::uint8_t {
        Default,
        FiveMinutes,
        OneHour,
    };

    infernix::PromptCacheMarkerKind kind       = infernix::PromptCacheMarkerKind::SharedStablePrefix;
    infernix::SharedCandidateEvidence evidence = infernix::SharedCandidateEvidence::ExplicitBoundary;
    Ttl ttl                                  = Ttl::Default;

    [[nodiscard]] friend constexpr bool operator==(CacheBoundary, CacheBoundary) noexcept = default;
};

// A document or search result as the client sent it. resolve_documents (documents.h) decodes, chunks and renders
// it after parsing, so parsing stays free of I/O and Count Tokens resolves exactly what Messages runs.
struct DocumentSource {
    enum class Kind : std::uint8_t {
        Document,     // Anthropic `document`; OpenAI file inputs
        SearchResult, // Anthropic `search_result`
    };
    enum class Format : std::uint8_t {
        Text,    // `text`: plain text, chunked into sentences
        Content, // `blocks`: custom content (a document's content source, a search result's content), a chunk each
        Pdf,     // `pdf`: a PDF file, each page's text chunked into sentences
    };
    Kind kind     = Kind::Document;
    Format format = Format::Text;
    std::string title;   // optional for a document, required for a search result
    std::string context; // a document's context: shown to the model, never cited
    std::string source;  // a search result's source (its URL or name)
    std::string text;
    std::vector<std::string> blocks;
    infernix::product::media_acquire::Source pdf;
    std::string filename; // an OpenAI file input's name, the title when it has none
    bool citations = false;
};

// A citation an assistant history text block carries (the location fields of an Anthropic citation), so the
// block renders back as the cite tag the model wrote (documents.h).
struct CitationLocation {
    enum class Kind : std::uint8_t {
        Char,         // char_location: [start, end) in code points
        Page,         // page_location: [start, end) page numbers, 1-based
        ContentBlock, // content_block_location: [start, end) content blocks
        SearchResult, // search_result_location: [start, end) content blocks of a search result
    };
    Kind kind                = Kind::Char;
    std::size_t source_index = 0; // document_index, or search_result_index
    std::uint64_t start      = 0;
    std::uint64_t end        = 0;
    std::string cited_text;

    [[nodiscard]] friend bool operator==(const CitationLocation&, const CitationLocation&) = default;
};

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;     // populated for Text
    std::string type_raw; // original wire "type" string for diagnostics
    infernix::product::media_acquire::Source source;
    infernix::ImageResizePolicy image_resize_policy = infernix::ImageResizePolicy::Downsize;
    std::shared_ptr<const DocumentSource> document; // populated for Document
    std::vector<CitationLocation> citations;        // an assistant text block's citations
    // A PDF page image resolve_documents rendered: the vision budget bounds it, so the request's media byte budget,
    // already charged with the PDF, does not count it again.
    bool derived = false;
    std::optional<CacheBoundary> cache_boundary_after;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string input_schema_json;
    std::string schema_param; // Original protocol path; never included in the model prompt.
    bool strict = false;
    // A deferred tool is declared for parsing and constraints but left out of the prompt's tool
    // list; the model sees its definition only where a later turn loads it (ChatTurn::loaded_tools).
    bool deferred = false;
    // Declared by an Anthropic mid-conversation tool_addition without defer_loading: the model saw it where it was
    // added (`deferred` is set as well, so the tool section never lists it), so a server tool search does not offer
    // it again.
    bool shown_in_messages = false;
    std::optional<std::string> input_examples_json;
    std::optional<CacheBoundary> cache_boundary_after;
};

// A declared tool an Anthropic conversation withdrew (tool_removal) or redefined (a tool_addition of another
// definition), as at the end of its history: `definition` is the newer one, nullopt a withdrawal.
struct ToolContractChange {
    std::string name;
    std::optional<ToolDefinition> definition;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

using ToolChoiceMode = infernix::ToolChoiceMode;
using ToolChoice     = infernix::ToolChoice;

struct ChatTurn {
    ChatRole role = ChatRole::User;
    std::vector<ContentPart> content; // ordered parts; may be empty when wire content is empty
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id; // populated for role=tool
    // Optional protocol assertion for a tool result. Call-graph normalization verifies it against
    // the function identified by tool_call_id before the Engine sees the history.
    std::optional<std::string> tool_result_name;
    bool tool_result_is_error = false;
    std::string reasoning_content; // assistant thinking carried across turns (round-tripped to the
                                   // template)
    // Tool definitions this turn loads (a tool-search result, or a block of additional tools).
    // They are rendered after the turn's content, so everything before the turn keeps its tokens.
    std::vector<ToolDefinition> loaded_tools;
    std::optional<CacheBoundary> cache_boundary_after;
};

// Sampling overrides that have an executable Engine meaning. Protocol-only
// fields are normalized or rejected before this value is constructed.
struct SamplingParams {
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<double> min_p;
    std::optional<int> top_k;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Protocol-level effort vocabulary. Each wire adapter accepts the values from
// its external contract; translation passes explicit values to the selected template.
enum class RequestedReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::optional<RequestedReasoningEffort>
parse_requested_reasoning_effort(std::string_view value) noexcept {
    if (value == "none") { return RequestedReasoningEffort::None; }
    if (value == "minimal") { return RequestedReasoningEffort::Minimal; }
    if (value == "low") { return RequestedReasoningEffort::Low; }
    if (value == "medium") { return RequestedReasoningEffort::Medium; }
    if (value == "high") { return RequestedReasoningEffort::High; }
    if (value == "xhigh") { return RequestedReasoningEffort::XHigh; }
    if (value == "max") { return RequestedReasoningEffort::Max; }
    return std::nullopt;
}

[[nodiscard]] constexpr std::string_view
requested_reasoning_effort_name(RequestedReasoningEffort effort) noexcept {
    switch (effort) {
    case RequestedReasoningEffort::None:
        return "none";
    case RequestedReasoningEffort::Minimal:
        return "minimal";
    case RequestedReasoningEffort::Low:
        return "low";
    case RequestedReasoningEffort::Medium:
        return "medium";
    case RequestedReasoningEffort::High:
        return "high";
    case RequestedReasoningEffort::XHigh:
        return "xhigh";
    case RequestedReasoningEffort::Max:
        return "max";
    }
    return {};
}

// Maximum tool-name length accepted at every protocol boundary and enforced on
// model-generated tool-call names. The OpenAI schema allows 64 bytes, but agent
// hosts synthesize longer names: VS Code Copilot wraps MCP tools as
// "activate_fallback_mcp_<server>_<tool>" (67 bytes observed), and the Anthropic
// adapter already accepted 128. 256 keeps real client names valid while still
// bounding prompt rendering and the streaming parser.
inline constexpr std::size_t kMaximumToolNameLength = 256;

struct GenerationRequest {
    std::optional<OutputConstraint> constraint;
    std::string constraint_param;
    NgramSessionHints ngram_session;
    std::vector<ChatTurn> messages;
    std::vector<ToolDefinition> tools;
    // Declared tools the conversation withdrew or redefined after declaring them (Anthropic mid-conversation
    // system messages). The prompt keeps every declaration where the conversation saw it; the output contract
    // follows these (translate.cpp contract_declarations).
    std::vector<ToolContractChange> tool_contract_changes;
    // The client knows tools it does not declare (Anthropic Messages with a deferred tool and no
    // server tool search: Claude Code's MCP tools, listed by name only): infernix::PromptOptions::open_tool_names.
    bool open_tool_names = false;
    std::size_t tool_name_max_length = kMaximumToolNameLength;
    ToolChoice tool_choice;
    std::vector<std::string> stop_strings;
    // Benchmark/serving extension shared with vLLM, SGLang and llama.cpp: suppress the
    // checkpoint's default stop tokens so generation runs to the requested token budget.
    // Caller-supplied stop tokens and stop strings still apply.
    bool ignore_eos = false;
    int max_tokens                       = 0; // resolved budget; zero means immediate output limit
    std::optional<bool> enable_thinking;      // unset => use the server default
    std::optional<std::uint32_t> thinking_budget;
    std::optional<RequestedReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
    infernix::PromptContinuationMode continuation = infernix::PromptContinuationMode::NewAssistantTurn;
    bool allow_engine_automatic_shared_prefixes = true;
    SamplingParams sampling;
    // Tokens whose log-probabilities at the prompt's next position the outcome reports
    // (/v1/decide). Empty for every ordinary generation.
    std::vector<infernix::TokenId> readout_tokens;
    // Restricts each output token to its step's set (/v1/decide's generated primitives).
    infernix::TokenConstraint token_constraint;
    // Tokens that end generation once committed (a scalar's closing brace).
    std::vector<infernix::TokenId> stop_token_ids;

    [[nodiscard]] bool uses_tools() const noexcept {
        return !tools.empty() && tool_choice.mode != ToolChoiceMode::None &&
               (!tool_choice.allowed_names || !tool_choice.allowed_names->empty());
    }

    [[nodiscard]] bool constrains_tools() const noexcept {
        if (tools.empty() || constraint) return false;
        if (!uses_tools()) return true;
        if (tool_choice.mode == ToolChoiceMode::Required || !tool_choice.parallel ||
            tool_choice.allowed_names || tool_choice.constraints == ToolConstraintMode::Basic)
            return true;
        return std::any_of(tools.begin(), tools.end(),
                           [](const auto& tool) { return tool.strict; });
    }

    [[nodiscard]] std::size_t media_item_count() const noexcept {
        std::size_t count = 0;
        for (const ChatTurn& message : messages) {
            for (const ContentPart& part : message.content) {
                if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { ++count; }
            }
        }
        return count;
    }

    [[nodiscard]] bool has_tool_history() const noexcept {
        for (const ChatTurn& message : messages) {
            if (!message.tool_calls.empty() || message.role == ChatRole::Tool) { return true; }
        }
        return false;
    }

    // The last tool the prompt lists, or nullptr: a deferred declaration has no prompt position.
    [[nodiscard]] ToolDefinition* last_listed_tool() noexcept {
        for (auto tool = tools.rbegin(); tool != tools.rend(); ++tool) {
            if (!tool->deferred) { return &*tool; }
        }
        return nullptr;
    }
};

} // namespace infernix::serve
