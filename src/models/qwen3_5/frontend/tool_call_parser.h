#pragma once

#include "models/qwen3_5/frontend/tool_contract.h"

#include <array>
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

// The Markdown code state of answer text read so far, by CommonMark's code rules: a ``` or ~~~ fence,
// open from its opening line (up to three columns of indentation, a run of at least three, and no
// backtick in a backtick fence's info string) until a line of the same character, at least as long,
// with nothing after it; an indented code block (four or more columns, after a blank line, a heading
// or the start of the text); and a code span of the current paragraph, closed only by a backtick run
// of its opening length (a backslash outside a span makes the next backtick text). Block quotes (`>`,
// any depth) and list items (`-`, `+`, `*`, `1.`, `1)`) are containers, by CommonMark's rules for
// them: their lines' columns count from the container's content (a fence or indented code inside a
// quote or an item is code there), a line that does not repeat a quote's `>` or a list item's
// indentation ends it unless it continues a paragraph (laziness), and a fence ends with its container.
// Up to eight nested containers are read (consecutive quotes count as one); deeper ones are text. A
// tool-call marker inside code is a quoted example and stays text; a marker at the start of a line or
// of a container's content (after at most three columns of indentation) is outside any span opened on
// an earlier line, whose run may never close. Only the text before a marker decides, so streamed and
// whole outputs agree; the state is fixed-size, so a copy costs nothing.
class MarkdownCodeTracker {
public:
    void feed(std::string_view text);
    // Whether the next byte lies inside a fence (its opening line included), an indented code block
    // or a code span.
    [[nodiscard]] bool in_code() const noexcept;

private:
    // Where the current line is: the open containers' prefixes (a quote's `>`, an item's indentation),
    // a possible list marker and the blanks after it, the indentation of its content, then outside a
    // fence an indented code line, the opening run of a possible fence (and its info string), the
    // hashes of a possible heading (and its text), or paragraph text; inside a fence a possible closing
    // run (and the blanks after it), or code.
    enum class Line : std::uint8_t {
        Prefix, Marker, Gap, Indent, Indented, Run, Info, Hashes, Heading, Text, Closer, CloserTail, Code
    };
    // An open container: `quotes` nested block quotes, or (`quotes` 0) a list item whose content starts
    // `offset` columns after its parent's.
    struct Container {
        std::uint32_t quotes = 0;
        std::uint32_t offset = 0;
    };
    static constexpr std::size_t kMaxContainers = 8;

    void byte(char c);
    // True when `c` was container syntax (a prefix, a marker or its blanks); otherwise the line takes it
    // as content.
    bool structure(char c);
    // The line's first content byte `c` does not repeat every open container.
    void begin_content(char c);
    bool open_quote();
    bool open_item(bool empty);
    void refute_marker();
    void close_containers();
    void next_container();
    void leaf(char c);
    [[nodiscard]] bool leaf_in_code() const noexcept;
    [[nodiscard]] std::uint32_t levels(std::size_t containers) const noexcept;
    [[nodiscard]] std::uint32_t indent() const noexcept { return column_ > base_ ? column_ - base_ : 0; }
    // A thematic break (`- - -`, `* * *`) is no list item: the line's content bytes are noted.
    void note(char c);
    void end_line();
    void text_byte(char c);
    // The open span once the backtick run in progress ends (0: none).
    [[nodiscard]] std::uint32_t open_span() const noexcept;

    std::array<Container, kMaxContainers> containers_{};
    std::uint8_t depth_         = 0;     // open containers
    char fence_                 = '\0';  // the open fence's character, '\0' when none is open
    std::uint32_t fence_run_    = 0;     // its opening run
    std::uint32_t fence_levels_ = 0;     // the container levels it is inside
    bool paragraph_             = false; // the last non-blank line was paragraph text (it takes no indented code)
    std::uint32_t span_         = 0;     // the paragraph's open code span: its backtick run (0: none)
    // The current line.
    Line line_                = Line::Indent;
    std::uint8_t entry_       = 0;     // the containers it repeats so far, and `level_` quotes of the next
    std::uint32_t level_      = 0;
    std::uint32_t column_     = 0;     // the next byte's column
    std::uint32_t base_       = 0;     // the column the innermost repeated container's content starts at
    bool marker_blank_        = false; // a `>` was read: one blank after it belongs to it
    bool lazy_pending_        = false; // a list marker decides whether the containers it did not repeat end
    char marker_              = '\0';  // the list marker read (Marker, Gap): its bullet, or '0' when ordered
    bool marker_done_         = false; // the marker is complete (an ordered one has its `.` or `)`)
    bool marker_one_          = false; // an ordered marker's number is 1
    std::uint8_t marker_digits_  = 0;
    std::uint32_t marker_column_ = 0;
    std::uint32_t marker_width_  = 0;
    std::uint8_t items_opened_   = 0;     // list items the line opened, and the paragraph state before them
    bool saved_paragraph_        = false;
    std::uint32_t saved_span_    = 0;
    char break_char_             = '\0';  // the content's character so far ('!': not a thematic break)
    std::uint32_t break_count_   = 0;
    char run_char_               = '\0';  // the character of the line's first run (Run, Info)
    std::uint32_t run_           = 0;     // the length of that run, or of the hashes, or of a closing run
    bool info_backtick_          = false; // a backtick in the info string (Info): no fence then
    std::uint32_t ticks_         = 0;     // the backtick run in progress
    bool escaped_                = false; // the previous byte was a backslash outside a span
};

// Parse Qwen's XML-like tool-call format. In tolerant mode, a complete function call is
// recovered even when the model adds malformed wrapper markup, a trailing suffix, or stops at
// its output budget before the closing tags; the strict parser keeps its all-or-nothing
// behavior. A `<tool_call>` wrapper opens a call only where the model wrote its token. Free output
// also reads the wrapper-less agent-harness openers (`<function=...>`, `<invoke ...>`,
// `<function_calls>`), which have no token of their own, by their spelling; a marker inside
// Markdown code is text there, and `code` is the code state of the text before `output`.
// `cut_short`: the generation was cut short (output or context budget, cancellation) rather than
// ended by the model. Tolerant mode keeps a final call the end left inside its arguments only
// then; otherwise such a call is returned as text with a truncated-tail diagnostic.
[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(GeneratedText output, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract, bool tolerant = false,
                            MarkdownCodeTracker code = {}, bool cut_short = false);

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
