// The tool-call parser's Markdown code tracker (MarkdownCodeTracker, tool_call_parser.h): whether the next byte of
// a model's output lies inside a fence, an indented code block or a code span, under CommonMark's container rules.

#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace infernix::models::qwen3_5::frontend {
namespace {

constexpr bool is_ascii_digit(char byte) { return byte >= '0' && byte <= '9'; }
constexpr bool is_blank_space(char c) { return c == ' ' || c == '\t'; }
// Bytes that leave a line blank and take no column.
constexpr bool is_zero_width_blank(char c) { return c == '\r' || c == '\f' || c == '\v'; }
constexpr std::uint32_t next_column(std::uint32_t column, char c) { return c == '\t' ? column / 4 * 4 + 4 : column + 1; }

} // namespace

void MarkdownCodeTracker::feed(std::string_view text) {
    for (const char c : text) { byte(c); }
}

void MarkdownCodeTracker::byte(char c) {
    if (c == '\n') {
        end_line();
        return;
    }
    if (!structure(c)) { leaf(c); }
}

std::uint32_t MarkdownCodeTracker::levels(std::size_t containers) const noexcept {
    std::uint32_t total = 0;
    for (std::size_t i = 0; i < containers; ++i) {
        const std::uint32_t add = containers_[i].quotes != 0 ? containers_[i].quotes : 1;
        total                   = total > UINT32_MAX - add ? UINT32_MAX : total + add;
    }
    return total;
}

void MarkdownCodeTracker::next_container() {
    ++entry_;
    level_ = 0;
    if (entry_ == depth_) { line_ = Line::Indent; }
}

bool MarkdownCodeTracker::structure(char c) {
    if (marker_blank_) {
        // One blank after a `>` belongs to it (a tab's first column).
        marker_blank_ = false;
        if (is_blank_space(c)) {
            ++base_;
            column_ = next_column(column_, c);
            return true;
        }
    }
    switch (line_) {
    case Line::Prefix: {
        const Container& open = containers_[entry_];
        if (is_blank_space(c)) {
            column_ = next_column(column_, c);
            // A list item continues on a line indented to its content (the rest is the content's).
            if (open.quotes == 0 && indent() >= open.offset) {
                base_ += open.offset;
                next_container();
            }
            return true;
        }
        if (is_zero_width_blank(c)) { return true; }
        if (open.quotes != 0 && c == '>' && indent() <= 3) {
            column_       = base_ = column_ + 1;
            marker_blank_ = true;
            if (++level_ == open.quotes) { next_container(); }
            return true;
        }
        begin_content(c);
        return structure(c);
    }
    case Line::Indent:
        // The content's first byte may open a block quote or a list item.
        if (is_blank_space(c) || is_zero_width_blank(c) || fence_ != '\0' || indent() > 3) { return false; }
        if (entry_ != depth_ && !lazy_pending_) { return false; } // a lazy paragraph line
        if (c == '>' && !lazy_pending_) {
            note(c);
            return open_quote();
        }
        if (c != '-' && c != '+' && c != '*' && !is_ascii_digit(c)) { return false; }
        marker_        = is_ascii_digit(c) ? '0' : c;
        marker_done_   = !is_ascii_digit(c);
        marker_one_    = c == '1';
        marker_digits_ = 1;
        marker_column_ = column_;
        marker_width_  = 1;
        ++column_;
        line_ = Line::Marker;
        note(c);
        return true;
    case Line::Marker:
        if (!marker_done_ && is_ascii_digit(c) && marker_digits_ < 9) {
            ++marker_digits_;
            ++marker_width_;
            ++column_;
            marker_one_ = false;
            note(c);
            return true;
        }
        if (!marker_done_ && marker_digits_ != 0 && (c == '.' || c == ')')) {
            marker_done_ = true;
            ++marker_width_;
            ++column_;
            note(c);
            return true;
        }
        if (marker_done_ && (is_blank_space(c) || is_zero_width_blank(c))) {
            if (is_blank_space(c)) { column_ = next_column(column_, c); }
            line_ = Line::Gap;
            return true;
        }
        refute_marker();
        return false;
    case Line::Gap:
        if (is_blank_space(c)) {
            column_ = next_column(column_, c);
            return true;
        }
        if (is_zero_width_blank(c)) { return true; }
        if (!open_item(false)) {
            refute_marker();
            return false;
        }
        return structure(c); // the item's content may open a container itself
    default: return false;
    }
}

void MarkdownCodeTracker::begin_content(char c) {
    line_ = Line::Indent;
    // A fence inside a container the line does not repeat ends with it.
    if (fence_ != '\0' && fence_levels_ > levels(entry_) + level_) {
        close_containers();
        return;
    }
    // A line that would continue the paragraph keeps the containers (laziness), unless it starts a block
    // that interrupts the paragraph; a list marker decides that itself.
    const bool block  = indent() <= 3 && (c == '`' || c == '~' || c == '#' || c == '>');
    const bool marker = indent() <= 3 && (c == '-' || c == '+' || c == '*' || is_ascii_digit(c));
    if (paragraph_ && !block) {
        lazy_pending_ = marker;
        return;
    }
    close_containers();
}

void MarkdownCodeTracker::close_containers() {
    if (level_ != 0) {
        containers_[entry_].quotes = level_;
        depth_                     = static_cast<std::uint8_t>(entry_ + 1);
    } else {
        depth_ = entry_;
    }
    entry_        = depth_;
    level_        = 0;
    lazy_pending_ = false;
    if (fence_ != '\0' && fence_levels_ > levels(depth_)) {
        fence_     = '\0';
        fence_run_ = 0;
    }
    paragraph_ = false;
    span_      = 0;
}

bool MarkdownCodeTracker::open_quote() {
    if (depth_ != 0 && containers_[depth_ - 1].quotes != 0) {
        std::uint32_t& quotes = containers_[depth_ - 1].quotes;
        if (quotes != UINT32_MAX) { ++quotes; }
    } else if (depth_ == kMaxContainers) {
        return false;
    } else {
        containers_[depth_++] = Container{.quotes = 1, .offset = 0};
    }
    entry_        = depth_;
    level_        = 0;
    column_       = base_ = column_ + 1;
    marker_blank_ = true;
    paragraph_    = false;
    span_         = 0;
    return true;
}

// A list item interrupts a paragraph (a line that would continue one) only when it has content and,
// ordered, starts at 1. `empty`: the marker ends its line.
bool MarkdownCodeTracker::open_item(bool empty) {
    if (paragraph_ && (empty || (marker_ == '0' && !marker_one_))) { return false; }
    const std::size_t kept = lazy_pending_ ? entry_ + (level_ != 0 ? 1U : 0U) : depth_;
    if (kept >= kMaxContainers) { return false; }
    if (lazy_pending_) { close_containers(); }
    const std::uint32_t gap  = column_ - (marker_column_ + marker_width_);
    // Five or more blanks after the marker: one belongs to it, the rest start indented code.
    const std::uint32_t used = empty || gap == 0 || gap > 4 ? 1 : gap;
    if (items_opened_++ == 0) {
        saved_paragraph_ = paragraph_;
        saved_span_      = span_;
    }
    const std::uint32_t content = marker_column_ + marker_width_ + used;
    containers_[depth_++]       = Container{.quotes = 0, .offset = content - base_};
    entry_                      = depth_;
    level_                      = 0;
    base_                       = content;
    paragraph_                  = false;
    span_                       = 0;
    line_                       = Line::Indent;
    return true;
}

// The marker read is paragraph text; a lazy line keeps the containers it did not repeat.
void MarkdownCodeTracker::refute_marker() {
    lazy_pending_ = false;
    line_         = Line::Text;
}

void MarkdownCodeTracker::note(char c) {
    if (is_blank_space(c) || is_zero_width_blank(c)) { return; }
    if (break_char_ == '\0' && (c == '-' || c == '*' || c == '_')) {
        break_char_ = c;
    } else if (c != break_char_) {
        break_char_ = '!';
    }
    ++break_count_;
}

void MarkdownCodeTracker::leaf(char c) {
    const auto blank = [](char b) { return b == ' ' || b == '\t' || b == '\r'; };
    note(c);
    switch (line_) {
    case Line::Indent:
        if (is_blank_space(c)) {
            column_ = next_column(column_, c);
        } else if (is_zero_width_blank(c)) {
            // still a blank line
        } else if (fence_ != '\0') {
            line_ = indent() <= 3 && c == fence_ ? Line::Closer : Line::Code;
            run_  = 1;
        } else if (indent() >= 4 && !paragraph_) {
            line_ = Line::Indented;
        } else {
            line_     = indent() <= 3 && (c == '`' || c == '~') ? Line::Run
                        : indent() <= 3 && c == '#'           ? Line::Hashes
                                                              : Line::Text;
            run_char_ = c;
            run_      = 1;
            text_byte(c);
        }
        return;
    case Line::Closer:
        if (c == fence_) {
            ++run_;
        } else {
            line_ = run_ >= fence_run_ && blank(c) ? Line::CloserTail : Line::Code;
        }
        return;
    case Line::CloserTail:
        if (!blank(c)) { line_ = Line::Code; }
        return;
    case Line::Code:
    case Line::Indented: return;
    case Line::Run:
        if (c == run_char_) {
            ++run_;
        } else {
            line_          = run_ >= 3 ? Line::Info : Line::Text;
            info_backtick_ = c == '`';
        }
        break;
    case Line::Info: info_backtick_ = info_backtick_ || c == '`'; break;
    case Line::Hashes:
        if (c == '#') {
            ++run_;
        } else if (run_ <= 6 && blank(c)) {
            line_ = Line::Heading;
            span_ = 0; // a heading ends the paragraph before it, and any span left open there
        } else {
            line_ = Line::Text;
        }
        break;
    case Line::Prefix: // structure() leaves these before content
    case Line::Marker:
    case Line::Gap:
    case Line::Heading:
    case Line::Text: break;
    }
    text_byte(c);
}

// Code spans: a run opens one, and only a run of the same length closes it.
void MarkdownCodeTracker::text_byte(char c) {
    if (c == '`') {
        if (escaped_ && ticks_ == 0) {
            escaped_ = false; // \` is a backtick of text
        } else {
            ++ticks_;
        }
        return;
    }
    span_    = open_span();
    ticks_   = 0;
    escaped_ = span_ == 0 && c == '\\' && !escaped_;
}

std::uint32_t MarkdownCodeTracker::open_span() const noexcept {
    if (ticks_ == 0) { return span_; }
    if (span_ == 0) { return ticks_; }
    return ticks_ == span_ ? 0 : span_;
}

void MarkdownCodeTracker::end_line() {
    switch (line_) {
    case Line::Prefix:
        // A blank line after the prefixes it repeats: list items go on over it, a block quote it does not
        // mark ends.
        while (entry_ < depth_ && level_ == 0 && containers_[entry_].quotes == 0) { ++entry_; }
        if (entry_ < depth_) { close_containers(); }
        line_ = Line::Indent;
        break;
    case Line::Marker:
        if (!marker_done_ || !open_item(true)) { refute_marker(); }
        break;
    case Line::Gap:
        if (!open_item(true)) { refute_marker(); }
        break;
    default: break;
    }
    // A thematic break opens no list item: the line is the text it was before containers were read.
    if (items_opened_ != 0 && break_char_ != '!' && break_char_ != '\0' && break_count_ >= 3) {
        depth_     = static_cast<std::uint8_t>(depth_ - items_opened_);
        paragraph_ = saved_paragraph_;
        span_      = saved_span_;
        line_      = Line::Text;
    }
    const bool opener = (line_ == Line::Run && run_ >= 3) ||
                        (line_ == Line::Info && (run_char_ == '~' || !info_backtick_));
    if (fence_ != '\0') {
        if ((line_ == Line::Closer && run_ >= fence_run_) || line_ == Line::CloserTail) {
            fence_     = '\0';
            fence_run_ = 0;
        }
    } else if (opener) {
        fence_        = run_char_;
        fence_run_    = run_;
        fence_levels_ = levels(depth_);
        paragraph_    = false;
        span_         = 0;
    } else if (line_ == Line::Text || line_ == Line::Run || line_ == Line::Info ||
               (line_ == Line::Hashes && run_ > 6)) {
        span_      = open_span(); // a paragraph's span continues on its next line
        paragraph_ = true;
    } else {
        // A blank line, an indented code line or a heading: no paragraph continues past it.
        paragraph_ = false;
        span_      = 0;
    }
    line_          = depth_ == 0 ? Line::Indent : Line::Prefix;
    entry_         = 0;
    level_         = 0;
    column_        = 0;
    base_          = 0;
    marker_blank_  = false;
    lazy_pending_  = false;
    items_opened_  = 0;
    break_char_    = '\0';
    break_count_   = 0;
    run_char_      = '\0';
    run_           = 0;
    info_backtick_ = false;
    ticks_         = 0;
    escaped_       = false;
}

bool MarkdownCodeTracker::in_code() const noexcept {
    // The container structure the next byte settles first: a marker's `<`, which is no container syntax.
    MarkdownCodeTracker next = *this;
    (void)next.structure('<');
    return next.leaf_in_code();
}

bool MarkdownCodeTracker::leaf_in_code() const noexcept {
    if (fence_ != '\0') { return true; } // a marker after a closing run makes it no closer
    switch (line_) {
    // A span left open on an earlier line does not reach a line's start: a streamed reader cannot know
    // that its run is never closed (CommonMark then makes it text), and a stray backtick in a preamble
    // must not turn the call on the next line into text (Strata #4923).
    case Line::Indent: return indent() >= 4 && !paragraph_;
    case Line::Indented: return true;
    case Line::Run: return run_ >= 3 || open_span() != 0;
    case Line::Info: return run_char_ == '~' || !info_backtick_ || open_span() != 0;
    default: return open_span() != 0;
    }
}

} // namespace infernix::models::qwen3_5::frontend
