#pragma once

#include "models/qwen3_5/frontend/tool_contract.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::models::qwen3_5::frontend {

// The call opener the model writes as its own added token.
inline constexpr std::string_view kToolCallOpen = "<tool_call>";

// Generated text and the byte offsets, ascending, at which the model's `<tool_call>` token starts
// in it. Only those openers start a call: the same spelling from ordinary text tokens is text.
struct GeneratedText {
    std::string_view text;
    std::span<const std::size_t> tool_call_tokens;
};

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

// The Markdown code state of answer text read so far: an open ``` or ~~~ fence, and the
// backticks of the current paragraph. A tool-call marker of free output inside a fence or inline
// code is a quoted example and stays text. Only the text before a marker decides, so streamed
// and whole outputs agree.
class MarkdownCodeTracker {
public:
    void feed(std::string_view text);
    // Whether the next byte lies inside a code fence or inline code.
    [[nodiscard]] bool in_code() const noexcept;

private:
    void end_line();

    std::string line_;            // the current, unfinished line
    char fence_          = '\0'; // the open fence's character, '\0' when none is open
    std::uint32_t ticks_ = 0;     // backticks of the current paragraph's finished lines
};

// Parse Qwen's XML-like tool-call format. In tolerant mode, a complete function call is
// recovered even when the model adds malformed wrapper markup, a trailing suffix, or stops at
// its output budget before the closing tags; the strict parser keeps its all-or-nothing
// behavior. A `<tool_call>` wrapper opens a call only where the model wrote its token. Free output
// also reads the wrapper-less agent-harness openers (`<function=...>`, `<invoke ...>`,
// `<function_calls>`), which have no token of their own, by their spelling; a marker inside
// Markdown code is text there, and `code` is the code state of the text before `output`.
[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(GeneratedText output, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract, bool tolerant = false,
                            MarkdownCodeTracker code = {});

class ConstrainedToolStream;

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix. Constrained output streams its calls as they are generated (take_call_deltas) and at
// terminal time retains the completed ones across interruptions; free output publishes its calls at
// terminal time and restores malformed regions verbatim.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length, bool tolerant = false);
    ~ToolCallOutputDecoder();
    ToolCallOutputDecoder(ToolCallOutputDecoder&&) noexcept;
    ToolCallOutputDecoder& operator=(ToolCallOutputDecoder&&) noexcept;

    [[nodiscard]] bool in_tool_region() const noexcept {
        return saw_tool_marker_ || !pending_tag_.empty();
    }
    // Feeds the next bytes of the generated content stream; `tool_call_tokens` are relative to
    // this piece. A token's bytes may continue in the next piece.
    [[nodiscard]] std::string feed(GeneratedText content);
    // The constrained calls' steps read since the last take, in generation order.
    [[nodiscard]] std::vector<ToolCallDelta> take_call_deltas();
    // The assistant prefix a request continues is the client's text: each `<tool_call>` spelled
    // in it is the opener of the call it continues.
    void initialize_continuation(std::string_view prefix);
    // `open_reasoning` is the thinking of a turn that ended without closing it. For free tool
    // output in tolerant mode, when the turn ended on a stop token and no content was fed,
    // complete calls stranded at the end of that thinking become the structured turn.
    [[nodiscard]] Terminal finish(FinishReason reason          = FinishReason::StopToken,
                                  GeneratedText open_reasoning = {});

private:
    void append_region(GeneratedText content, std::size_t from);

    std::shared_ptr<const ToolCallOutputContract> contract_;
    // The constrained route's calls, read as they arrive from the region's first token.
    std::unique_ptr<ConstrainedToolStream> stream_;
    std::string trailing_whitespace_;
    std::string tool_region_;
    // Offsets in tool_region_ at which a `<tool_call>` token starts.
    std::vector<std::size_t> region_tool_calls_;
    std::string pending_tag_;
    // The last published bytes, enough to see a `<tool_call>` spelled with ordinary tokens.
    std::string published_tail_;
    // The code state of the text fed before the next byte, and of the text before the tool
    // region (the state its parse starts from).
    MarkdownCodeTracker code_;
    MarkdownCodeTracker region_code_;
    std::size_t max_tool_name_length_        = 0;
    bool tolerant_                           = false;
    bool saw_tool_marker_                    = false;
    bool fed_content_                        = false;
    bool finished_                           = false;
    std::size_t continuation_withheld_bytes_ = 0;
};

} // namespace infernix::models::qwen3_5::frontend
