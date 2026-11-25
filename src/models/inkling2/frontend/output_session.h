#pragma once

// Inkling 2's output session: the generated tokens of one turn split into its channels by the checkpoint's
// response_template (tokenizer_config.json), published as the Engine commits them.
//
// A turn writes its own message headers: the prompt ends with a closed message, and every block of the reply is
// `<|message_model|>NAME?<|content_K|>BODY<|end_message|>`, K thinking (the reasoning channel), text, tool_error or
// invoke_tool_text (the content channel, blocks joined without a separator) or invoke_tool_json (a tool call: BODY
// is {"name": ..., "args": {...}}). Text outside a block is dropped, as the template's parser drops it. The markers
// count by token identity: their spellings in ordinary text tokens are the block's text. The end-of-turn token
// `<|content_model_end_sampling|>` is the model's default stop token. A thinking budget closes the open thinking with
// `<|end_message|>` (after the configured message, when there is one); the model then writes its answer's header
// itself. A constrained turn's matcher (constraints.h) reads the same tokens in the same transactions: a round's
// preview advances it, its commit confirms it and its discard rolls it back.

#include "infernix/types.h"
#include "runtime/contract/request.h"

#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace infernix::text {
class Tokenizer;
class GrammarSession;
}

namespace infernix::models::inkling2 {

// The added tokens a turn's framing uses; -1 where the vocabulary lacks one (the frontend refuses such a
// tokenizer).
struct OutputMarkers {
    TokenId message_model  = -1; // <|message_model|>: a block's header
    TokenId thinking       = -1; // <|content_thinking|>
    std::array<TokenId, 3> content{-1, -1, -1}; // <|content_text|>, <|content_tool_error|>, <|content_invoke_tool_text|>
    TokenId tool_call      = -1; // <|content_invoke_tool_json|>
    TokenId end_message    = -1; // <|end_message|>
    TokenId end_of_turn    = -1; // <|content_model_end_sampling|>
};

// One declared tool: its name, its declaration's parameter schema (JSON text; `{}` when it declares none), whether its
// calls' arguments must satisfy that schema (strict), and its index among the declarations (the deferred ones after the
// listed ones).
struct DeclaredTool {
    std::string name;
    std::string parameters = "{}";
    // The properties the schema's root requires (a published call that leaves one out is counted, not refused).
    std::vector<std::string> required;
    bool strict            = false;
    std::size_t declaration_index = 0;
};

// The tools a call may name: the declared ones, or any name when the client keeps its own (PromptOptions::
// open_tool_names). Without one, a call is text.
struct ToolNames {
    std::unordered_set<std::string> declared;
    bool open = false;
    std::vector<DeclaredTool> tools; // in declaration order, the withdrawn ones left out
};

class PublishedOutput {
public:
    using iterator       = std::vector<OutputDelta>::iterator;
    using const_iterator = std::vector<OutputDelta>::const_iterator;

    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }
    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }
    [[nodiscard]] iterator end() noexcept { return values_.end(); }
    [[nodiscard]] const_iterator end() const noexcept { return values_.end(); }
    [[nodiscard]] OutputDelta& back() noexcept { return values_.back(); }
    [[nodiscard]] const OutputDelta& back() const noexcept { return values_.back(); }
    // Constrained calls' steps; Inkling's calls arrive whole (OutputSession::take_tool_calls), so this stays empty.
    [[nodiscard]] std::vector<ToolCallDelta>& tool_calls() noexcept { return tool_calls_; }
    [[nodiscard]] const std::vector<ToolCallDelta>& tool_calls() const noexcept { return tool_calls_; }

    void clear() noexcept {
        values_.clear();
        tool_calls_.clear();
    }
    // Appends `text` to the channel's last delta when it is the last one, else as a new delta.
    void append(OutputChannel channel, std::string_view text);

private:
    std::vector<OutputDelta> values_;
    std::vector<ToolCallDelta> tool_calls_;
};

class OutputSession {
public:
    struct Settings {
        std::shared_ptr<const text::Tokenizer> tokenizer;
        OutputMarkers markers;
        StopPolicy policy; // merged with the model's defaults
        OutputOptions output;
        ThinkingControlOptions thinking;
        // The tokens a reached thinking budget (or the loop guard's Conclude) injects: the configured message, then
        // <|end_message|>.
        std::shared_ptr<const std::vector<TokenId>> thinking_control;
        std::shared_ptr<const ToolNames> tools; // null: no tools (calls are text)
        // The turn's matcher (constraints.h), or null: the turn is free. It reads every token the session commits,
        // the thinking control's included.
        std::shared_ptr<text::GrammarSession> grammar;
        // The prompt ended with the answer's opener (PromptOptions::open_answer): the first token is the answer's.
        bool answer_opened = false;
        // The tools a call written in an XML notation the model was not given may name (candidates JC-01): a
        // <|content_text|> block that begins with `<tool_call>`, `<function_calls>` or `<invoke ` and holds nothing but
        // closed `<invoke name="NAME">` calls of these tools, followed by a call block or the end of the turn, is those
        // calls. Empty: such a block is text. The frontend gives the turn's callable non-strict tools (a strict tool's
        // arguments must be its schema's, which only its call block's grammar guarantees).
        std::vector<DeclaredTool> notation_tools;
        bool parallel_calls = true; // ToolChoice::parallel: false, a turn of more than one call keeps such a block text
    };

    OutputSession() noexcept;
    explicit OutputSession(Settings settings);
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;
    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview_model(std::span<const TokenId> tokens,
                                                        std::uint32_t total_budget_remaining,
                                                        FinishReason limit_reason);
    [[nodiscard]] std::uint32_t model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept;
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId> tokens,
                                                          std::uint32_t total_budget_remaining);
    void validate_generation_capacity(std::uint32_t effective_output_tokens) const;
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview();
    void discard_preview();

    // Whether the turn runs under a grammar (Settings::grammar).
    [[nodiscard]] bool constrained() const noexcept;
    // While the committed output's thinking is open: <|end_message|> closes it, and the end-of-turn token (and the
    // tool-call opener, when the turn declares tools) take no probability before it. Once a thinking budget is used up
    // (its control applied, or a block ended at it): <|content_thinking|> takes no probability for the rest of the turn.
    [[nodiscard]] runtime::ReasoningBan reasoning_ban() const noexcept;
    void observe_constraint(bool timings, double prepare_seconds) noexcept;
    void constraint_uploaded(std::size_t bytes) noexcept;
    [[nodiscard]] std::optional<ConstraintObservation> constraint_observation() const;
    // The legal tokens of the committed position and of each draft's after it (text::GrammarSession::masks); waits
    // for a compile still running beside the prefill.
    [[nodiscard]] std::uint64_t grammar_masks(std::span<const TokenId> drafts, std::span<std::uint32_t> words);
    [[nodiscard]] bool constraint_pending() const noexcept;
    [[nodiscard]] std::exception_ptr constraint_failure() const noexcept;

    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] bool has_tool_calls() const noexcept;
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infernix::models::inkling2
