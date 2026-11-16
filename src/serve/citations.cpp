#include "serve/citations.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace infernix::serve {
namespace {

bool is_continuation(unsigned char byte) noexcept { return (byte & 0xC0U) == 0x80U; }

std::size_t sequence_length(std::string_view text, std::size_t at) noexcept {
    const auto lead = static_cast<unsigned char>(text[at]);
    std::size_t length = 1;
    if ((lead & 0xE0U) == 0xC0U) {
        length = 2;
    } else if ((lead & 0xF0U) == 0xE0U) {
        length = 3;
    } else if ((lead & 0xF8U) == 0xF0U) {
        length = 4;
    }
    length = std::min(length, text.size() - at);
    for (std::size_t i = 1; i < length; ++i) {
        if (!is_continuation(static_cast<unsigned char>(text[at + i]))) { return i; }
    }
    return length;
}

bool at(std::string_view text, std::size_t position, std::string_view token) noexcept {
    return text.compare(position, token.size(), token) == 0;
}

// The full-width sentence marks: 。！？
constexpr std::array<std::string_view, 3> kWideTerminals{"\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F"};
// Closing quotes and brackets that may follow a terminal: ” ’ » 」 』 ）
constexpr std::array<std::string_view, 6> kWideClosers{"\xE2\x80\x9D", "\xE2\x80\x99", "\xC2\xBB",
                                                       "\xE3\x80\x8D", "\xE3\x80\x8F", "\xEF\xBC\x89"};
constexpr std::string_view kIdeographicSpace = "\xE3\x80\x80";

// The byte length of a terminal at `position`, 0 when there is none; `wide` reports a full-width mark.
std::size_t terminal_at(std::string_view text, std::size_t position, bool& wide) noexcept {
    const char c = text[position];
    if (c == '.' || c == '!' || c == '?') { return 1; }
    for (const std::string_view mark : kWideTerminals) {
        if (at(text, position, mark)) {
            wide = true;
            return mark.size();
        }
    }
    return 0;
}

std::size_t closer_at(std::string_view text, std::size_t position) noexcept {
    const char c = text[position];
    if (c == '"' || c == '\'' || c == ')' || c == ']' || c == '}') { return 1; }
    for (const std::string_view mark : kWideClosers) {
        if (at(text, position, mark)) { return mark.size(); }
    }
    return 0;
}

std::size_t space_at(std::string_view text, std::size_t position) noexcept {
    const char c = text[position];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') { return 1; }
    return at(text, position, kIdeographicSpace) ? kIdeographicSpace.size() : 0;
}

struct Cursor {
    std::size_t byte    = 0;
    std::uint64_t point = 0;
};

// Advances over a run of whitespace.
Cursor skip_spaces(std::string_view text, Cursor cursor) noexcept {
    while (cursor.byte < text.size()) {
        const std::size_t length = space_at(text, cursor.byte);
        if (length == 0) { break; }
        cursor.byte += length;
        ++cursor.point;
    }
    return cursor;
}

bool is_tag_space(char c) noexcept { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

// Abbreviations a number follows ("Fig. 3", "p. 12"), compared without case.
constexpr std::array<std::string_view, 16> kNumberAbbreviations{
    "fig", "figs", "eq", "eqs", "no", "nos", "sec", "secs", "ch", "vol", "p", "pp", "ref", "refs", "tab", "art"};

// Whether the ASCII terminal at `mark`, followed by whitespace and then `following`, ends no sentence:
// - before a lowercase word ("e.g. the", "approx. ten");
// - inside or after a contents line's dot leader ("Intro . . . . 174": a mark after whitespace or after a mark,
//   and a mark before a mark), so a contents page is not cut at every dot;
// - after an abbreviation a number follows ("Fig. 3");
// - after a line's leading enumerator ("1. Install", "6.1.1. Compilation"), which numbers what follows it.
bool no_sentence_end(std::string_view text, std::size_t mark, bool full_stop, char following) noexcept {
    if (following == '.' || following == '!' || following == '?') { return true; }
    if (!full_stop) { return false; }
    if (following >= 'a' && following <= 'z') { return true; }
    const char before = mark == 0 ? '\n' : text[mark - 1];
    const bool leader = before == ' ' || before == '\t' || before == '.';
    if (leader && is_digit(following)) { return true; }
    // The word the stop ends, back to the whitespace before it.
    std::size_t start = mark;
    while (start > 0 && text[start - 1] != ' ' && text[start - 1] != '\t' && text[start - 1] != '\n' &&
           text[start - 1] != '\r') {
        --start;
    }
    const std::string_view word = text.substr(start, mark - start);
    if (word.empty()) { return false; }
    if (is_digit(following) && word.size() <= 4) {
        for (const std::string_view abbreviation : kNumberAbbreviations) {
            if (abbreviation.size() == word.size() &&
                std::equal(word.begin(), word.end(), abbreviation.begin(), [](char a, char b) {
                    return (a >= 'A' && a <= 'Z' ? static_cast<char>(a - 'A' + 'a') : a) == b;
                })) {
                return true;
            }
        }
    }
    // A leading enumerator: digits with inner dots, first on its line.
    if (!is_digit(word.front()) || word.back() == '.') { return false; }
    for (const char c : word) {
        if (!is_digit(c) && c != '.') { return false; }
    }
    std::size_t line = start;
    while (line > 0 && (text[line - 1] == ' ' || text[line - 1] == '\t')) { --line; }
    return line == 0 || text[line - 1] == '\n';
}

} // namespace

std::uint64_t code_point_count(std::string_view text) noexcept {
    std::uint64_t count = 0;
    for (const char c : text) {
        if (!is_continuation(static_cast<unsigned char>(c))) { ++count; }
    }
    return count;
}

std::vector<SentenceSpan> split_sentences(std::string_view text) {
    std::vector<SentenceSpan> spans;
    Cursor begin;
    Cursor cursor;
    bool content = false; // the open sentence holds more than whitespace
    const auto close_at = [&](Cursor end) {
        spans.push_back(SentenceSpan{.byte_begin  = begin.byte,
                                     .byte_end    = end.byte,
                                     .point_begin = begin.point,
                                     .point_end   = end.point});
        begin   = end;
        cursor  = end;
        content = false;
    };
    while (cursor.byte < text.size()) {
        bool wide                    = false;
        const std::size_t terminal = terminal_at(text, cursor.byte, wide);
        if (terminal != 0) {
            const bool full_stop = text[cursor.byte] == '.';
            Cursor after{cursor.byte + terminal, cursor.point + 1};
            // Further marks and closers ("?!", "...", ".)") stay with the sentence.
            while (after.byte < text.size()) {
                std::size_t length = terminal_at(text, after.byte, wide);
                if (length == 0) { length = closer_at(text, after.byte); }
                if (length == 0) { break; }
                after.byte += length;
                ++after.point;
            }
            const Cursor next = skip_spaces(text, after);
            content           = true;
            bool boundary     = wide || next.byte == text.size() || next.byte > after.byte;
            if (boundary && !wide && next.byte < text.size() &&
                no_sentence_end(text, cursor.byte, full_stop, text[next.byte])) {
                boundary = false;
            }
            if (boundary) {
                close_at(next);
            } else {
                cursor = after;
            }
            continue;
        }
        const std::size_t space = space_at(text, cursor.byte);
        if (space != 0) {
            const Cursor next = skip_spaces(text, cursor);
            if (content) {
                const auto newlines = std::count(text.begin() + static_cast<std::ptrdiff_t>(cursor.byte),
                                                 text.begin() + static_cast<std::ptrdiff_t>(next.byte), '\n');
                if (newlines >= 2 || cursor.point - begin.point >= kMaximumSentenceCodePoints) {
                    close_at(next);
                    continue;
                }
            }
            cursor = next;
            continue;
        }
        content = true;
        cursor.byte += sequence_length(text, cursor.byte);
        ++cursor.point;
    }
    if (begin.byte < text.size()) {
        if (content || spans.empty()) {
            close_at(cursor);
        } else {
            spans.back().byte_end  = cursor.byte;
            spans.back().point_end = cursor.point;
        }
    }
    return spans;
}


std::string ResolvedCitation::cited_text() const {
    std::string text;
    for (std::size_t chunk = first; chunk <= last; ++chunk) { text += source->chunks[chunk].text; }
    return text;
}

std::size_t CitationRegistry::add(CitableSource source) {
    if (sources_.size() >= std::numeric_limits<std::uint32_t>::max() ||
        source.chunks.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("citable sources hold too many chunks");
    }
    sources_.push_back(std::move(source));
    return sources_.size() - 1U;
}

std::optional<CitationId> CitationRegistry::parse_id(std::string_view spelling) const {
    const std::size_t dot = spelling.find('.');
    if (dot == std::string_view::npos || dot == 0 || dot + 1U == spelling.size()) { return std::nullopt; }
    const auto number = [](std::string_view digits) -> std::optional<std::uint64_t> {
        if (digits.empty() || digits.size() > 9) { return std::nullopt; }
        std::uint64_t value = 0;
        for (const char c : digits) {
            if (c < '0' || c > '9') { return std::nullopt; }
            value = value * 10U + static_cast<std::uint64_t>(c - '0');
        }
        return value;
    };
    const std::optional<std::uint64_t> source = number(spelling.substr(0, dot));
    const std::optional<std::uint64_t> chunk  = number(spelling.substr(dot + 1U));
    if (!source || !chunk || *source >= sources_.size() || *chunk >= sources_[*source].chunks.size()) {
        return std::nullopt;
    }
    return CitationId{.source = static_cast<std::uint32_t>(*source), .chunk = static_cast<std::uint32_t>(*chunk)};
}

std::vector<CitationId> CitationRegistry::ids_for(const CitationLocation& location) const {
    const DocumentSource::Kind kind = location.kind == CitationLocation::Kind::SearchResult
                                          ? DocumentSource::Kind::SearchResult
                                          : DocumentSource::Kind::Document;
    const auto source = std::find_if(sources_.begin(), sources_.end(), [&](const CitableSource& item) {
        return item.kind == kind && item.index == location.source_index;
    });
    if (source == sources_.end()) { return {}; }
    const auto source_position = static_cast<std::uint32_t>(source - sources_.begin());
    std::vector<CitationId> ids;
    const auto add_range = [&](std::size_t first, std::size_t last) {
        for (std::size_t chunk = first; chunk <= last; ++chunk) {
            ids.push_back(CitationId{.source = source_position, .chunk = static_cast<std::uint32_t>(chunk)});
        }
    };
    if (location.kind == CitationLocation::Kind::Page) {
        if (source->format != DocumentSource::Format::Pdf) { return {}; }
        std::vector<std::size_t> pages;
        for (std::size_t chunk = 0; chunk < source->chunks.size(); ++chunk) {
            const std::uint32_t page = source->chunks[chunk].page;
            if (page >= location.start && page < location.end) { pages.push_back(chunk); }
        }
        if (pages.empty()) { return {}; }
        // The run of consecutive chunks whose text is the cited text.
        for (std::size_t first = 0; !location.cited_text.empty() && first < pages.size(); ++first) {
            std::string text;
            for (std::size_t last = first; last < pages.size(); ++last) {
                if (last > first && pages[last] != pages[last - 1U] + 1U) { break; }
                text += source->chunks[pages[last]].text;
                if (text.size() > location.cited_text.size() || !location.cited_text.starts_with(text)) { break; }
                if (text.size() == location.cited_text.size()) {
                    add_range(pages[first], pages[last]);
                    return ids;
                }
            }
        }
        add_range(pages.front(), pages.back());
        return ids;
    }
    const DocumentSource::Format format =
        location.kind == CitationLocation::Kind::Char ? DocumentSource::Format::Text : DocumentSource::Format::Content;
    if (source->format != format) { return {}; }
    for (std::size_t chunk = 0; chunk < source->chunks.size(); ++chunk) {
        const CitationChunk& item = source->chunks[chunk];
        if (item.start >= location.start && item.end <= location.end) {
            ids.push_back(CitationId{.source = source_position, .chunk = static_cast<std::uint32_t>(chunk)});
        }
    }
    return ids;
}

std::vector<ResolvedCitation> CitationRegistry::resolve(const std::vector<CitationId>& ids) const {
    std::vector<ResolvedCitation> citations;
    for (const CitationId id : ids) {
        if (id.source >= sources_.size() || id.chunk >= sources_[id.source].chunks.size()) { continue; }
        const CitableSource* source = &sources_[id.source];
        if (!citations.empty() && citations.back().source == source && citations.back().last + 1U == id.chunk) {
            citations.back().last = id.chunk;
            continue;
        }
        citations.push_back(ResolvedCitation{.source = source, .first = id.chunk, .last = id.chunk});
    }
    return citations;
}

std::string chunk_marker(std::size_t source, std::size_t chunk) {
    return "[" + std::to_string(source) + "." + std::to_string(chunk) + "] ";
}

std::string cite_open_tag(const std::vector<CitationId>& ids) {
    std::string tag = "<cite ids=\"";
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) { tag += ','; }
        tag += std::to_string(ids[index].source) + "." + std::to_string(ids[index].chunk);
    }
    return tag + "\">";
}

std::string citation_instruction() {
    return "When your answer uses these sources, put the words of your answer that a passage supports inside a cite "
           "tag naming the bracketed IDs of the passages that support them, for example: Sales rose <cite "
           "ids=\"0.4\">by 12 percent in 2023</cite>, mostly <cite ids=\"0.5,0.6\">in Germany and France</cite>. "
           "The cited words are part of your answer: do not repeat them outside the tag. Use only IDs shown in "
           "square brackets, and do not cite inside your reasoning.";
}

void CiteParser::emit_text(std::vector<Event>& events, std::string_view text) {
    if (text.empty()) { return; }
    std::string opened;
    if (state_ == State::Cited && !span_open_) {
        // The span opens at its first letter or digit (any non-ASCII character counts as one); until then its
        // text waits, so a tag around punctuation alone can give its citations to the text before it.
        const auto word = std::find_if(text.begin(), text.end(), [](char c) {
            const auto byte = static_cast<unsigned char>(c);
            return byte >= 0x80U || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
        });
        if (word == text.end()) {
            pending_text_ += text;
            return;
        }
        events.push_back(Event{.kind = Event::Kind::Open, .text = {}, .citations = std::move(pending_)});
        pending_.clear();
        span_open_ = true;
        opened = std::move(pending_text_);
        pending_text_.clear();
        opened += text;
        text = opened;
    }
    if (!events.empty() && events.back().kind == Event::Kind::Text) {
        events.back().text += text;
        return;
    }
    events.push_back(Event{.kind = Event::Kind::Text, .text = std::string(text), .citations = {}});
}

void CiteParser::close_span(std::vector<Event>& events) {
    if (span_open_) {
        events.push_back(Event{.kind = Event::Kind::Close, .text = {}, .citations = {}});
    } else if (state_ == State::Cited && !pending_text_.empty()) {
        events.push_back(Event{.kind = Event::Kind::Attach, .text = std::move(pending_text_), .citations = std::move(pending_)});
    }
    span_open_ = false;
    pending_.clear();
    pending_text_.clear();
    state_ = State::Plain;
}

std::optional<std::vector<CitationId>> CiteParser::parse_open_tag(std::string_view tag) {
    // tag = "<cite" attributes ">"
    std::string_view rest = tag.substr(5, tag.size() - 6);
    const auto skip       = [&] {
        while (!rest.empty() && is_tag_space(rest.front())) { rest.remove_prefix(1); }
    };
    skip();
    if (!rest.starts_with("ids")) { return std::nullopt; }
    rest.remove_prefix(3);
    skip();
    if (rest.empty() || rest.front() != '=') { return std::nullopt; }
    rest.remove_prefix(1);
    skip();
    if (rest.empty() || (rest.front() != '"' && rest.front() != '\'')) { return std::nullopt; }
    const char quote = rest.front();
    rest.remove_prefix(1);
    const std::size_t close = rest.find(quote);
    if (close == std::string_view::npos) { return std::nullopt; }
    const std::string_view value = rest.substr(0, close);
    rest.remove_prefix(close + 1U);
    skip();
    if (!rest.empty()) { return std::nullopt; }

    const auto separator = [](char c) { return c == ',' || c == '[' || c == ']' || is_tag_space(c); };
    std::vector<CitationId> ids;
    std::size_t position = 0;
    while (position < value.size()) {
        while (position < value.size() && separator(value[position])) { ++position; }
        const std::size_t start = position;
        while (position < value.size() && !separator(value[position])) { ++position; }
        if (start == position) { continue; }
        const std::optional<CitationId> id = registry_->parse_id(value.substr(start, position - start));
        if (id && std::find(ids.begin(), ids.end(), *id) == ids.end()) { ids.push_back(*id); }
    }
    return ids;
}

std::vector<CiteParser::Event> CiteParser::feed(std::string_view text) {
    held_ += text;
    std::vector<Event> events;
    std::size_t position = 0;
    const std::string_view buffer(held_);
    while (position < buffer.size()) {
        const std::size_t open = buffer.find('<', position);
        if (open == std::string_view::npos) {
            emit_text(events, buffer.substr(position));
            position = buffer.size();
            break;
        }
        emit_text(events, buffer.substr(position, open - position));
        const std::string_view rest = buffer.substr(open);
        // Decide what starts at `open`: a closing tag, an opening tag, something else, or not yet known.
        enum class Match : std::uint8_t { Incomplete, None, Open, Close };
        Match match        = Match::None;
        std::size_t length = 0;
        constexpr std::string_view kClose = "</cite";
        constexpr std::string_view kOpen  = "<cite";
        const auto prefix_of              = [&](std::string_view token) {
            return rest.size() < token.size() && token.starts_with(rest);
        };
        if (prefix_of(kClose) || prefix_of(kOpen)) {
            match = Match::Incomplete;
        } else if (rest.starts_with(kClose)) {
            std::size_t i = kClose.size();
            while (i < rest.size() && is_tag_space(rest[i])) { ++i; }
            if (i == rest.size()) {
                match = Match::Incomplete;
            } else if (rest[i] == '>') {
                match  = Match::Close;
                length = i + 1U;
            }
        } else if (rest.starts_with(kOpen)) {
            if (rest.size() == kOpen.size()) {
                match = Match::Incomplete;
            } else if (rest[kOpen.size()] == '>' || is_tag_space(rest[kOpen.size()])) {
                const std::size_t end = rest.find('>');
                if (end == std::string_view::npos) {
                    match = Match::Incomplete;
                } else if (end < kMaximumCiteTagBytes) {
                    match  = Match::Open;
                    length = end + 1U;
                }
            }
        }
        if (match == Match::Incomplete && rest.size() >= kMaximumCiteTagBytes) { match = Match::None; }
        if (match == Match::Incomplete) {
            position = open;
            break;
        }
        if (match == Match::None) {
            emit_text(events, "<");
            position = open + 1U;
            continue;
        }
        const std::string_view tag = rest.substr(0, length);
        position                   = open + length;
        if (match == Match::Close) {
            if (state_ != State::Plain) {
                close_span(events);
            } else if (orphan_closes_ != 0) {
                --orphan_closes_;
            } else {
                emit_text(events, tag);
            }
            continue;
        }
        std::optional<std::vector<CitationId>> ids = parse_open_tag(tag);
        if (!ids) {
            emit_text(events, tag);
            continue;
        }
        if (state_ != State::Plain) {
            // A tag inside an open tag closes it; the outer closing tag will then close nothing.
            close_span(events);
            ++orphan_closes_;
        }
        if (ids->empty()) {
            ++invalid_tags_;
            state_ = State::Dropped;
            continue;
        }
        pending_ = registry_->resolve(*ids);
        state_   = State::Cited;
    }
    held_.erase(0, position);
    return events;
}

std::vector<CiteParser::Event> CiteParser::finish() {
    std::vector<Event> events;
    // Held text may begin a tag: one the end cut after its name is dropped, a shorter prefix ("<", "<ci") is text.
    if (!held_.starts_with("<cite") && !held_.starts_with("</cite")) { emit_text(events, held_); }
    held_.clear();
    close_span(events);
    orphan_closes_ = 0;
    return events;
}

} // namespace infernix::serve
