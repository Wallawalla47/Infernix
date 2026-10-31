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

// The Markdown code state of answer text read so far, by CommonMark's code rules without container
// blocks (lists, block quotes): a ``` or ~~~ fence, open from its opening line (up to three columns
// of indentation, a run of at least three, and no backtick in a backtick fence's info string) until
// a line of the same character, at least as long, with nothing after it; an indented code block
// (four or more columns, after a blank line, a heading or the start of the text); and a code span of
// the current paragraph, closed only by a backtick run of its opening length (a backslash outside a
// span makes the next backtick text). A tool-call marker inside code is a quoted example and stays
// text. Only the text before a marker decides, so streamed and whole outputs agree; the state is
// fixed-size, so a copy costs nothing.
class MarkdownCodeTracker {
public:
    void feed(std::string_view text);
    // Whether the next byte lies inside a fence (its opening line included), an indented code block
    // or a code span.
    [[nodiscard]] bool in_code() const noexcept;

private:
    // Where the current line is: its indentation, then outside a fence an indented code line, the
    // opening run of a possible fence (and its info string), the hashes of a possible heading (and
    // its text), or paragraph text; inside a fence a possible closing run (and the blanks after it),
    // or code.
    enum class Line : std::uint8_t { Indent, Indented, Run, Info, Hashes, Heading, Text, Closer, CloserTail, Code };

    void byte(char c);
    void end_line();
    void text_byte(char c);
    // The open span once the backtick run in progress ends (0: none).
    [[nodiscard]] std::uint32_t open_span() const noexcept;

    char fence_              = '\0';  // the open fence's character, '\0' when none is open
    std::uint32_t fence_run_ = 0;     // its opening run
    bool paragraph_          = false; // the last non-blank line was paragraph text (it takes no indented code)
    std::uint32_t span_      = 0;     // the paragraph's open code span: its backtick run (0: none)
    Line line_               = Line::Indent;
    std::uint32_t indent_    = 0;     // the line's columns of leading spaces and tabs
    char run_char_           = '\0';  // the character of the line's first run (Run, Info)
    std::uint32_t run_       = 0;     // the length of that run, or of the hashes, or of a closing run
    bool info_backtick_      = false; // a backtick in the info string (Info): no fence then
    std::uint32_t ticks_     = 0;     // the backtick run in progress
    bool escaped_            = false; // the previous byte was a backslash outside a span
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
    // The constrained calls' steps read since the last take, in generation order. A call whose
    // closing lines end the output completes in finish(), so take its last step after that.
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
