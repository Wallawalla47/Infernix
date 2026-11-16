#pragma once

// Citations over a request's citable sources (Anthropic `document` and `search_result` blocks with citations
// enabled). Each source is cut into chunks: the sentences of plain text and of a PDF's pages, or the blocks of
// custom content and of a search result. The model sees each chunk after its ID, `[S.C]`: S counts the request's
// citable sources from 0 in request order, C the source's chunks from 0, so an earlier source keeps its IDs when a
// later turn adds another (the prompt prefix stays cached), and a document's own bracketed numbers (a paper's
// "[12]") are never valid IDs. The instruction after the first citable source asks the model to wrap each
// supported part of its answer in a cite tag; CiteParser turns the generated text back into plain and cited spans.
// The prompt rendering and the history's cite tags live in documents.h.

#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

// One sentence of a text: [begin, end) in bytes and in Unicode code points. A chunk keeps the whitespace after its
// sentence, so a text's chunks concatenate to the text.
struct SentenceSpan {
    std::size_t byte_begin    = 0;
    std::size_t byte_end      = 0;
    std::uint64_t point_begin = 0;
    std::uint64_t point_end   = 0;
};

// Cuts `text` into sentences that partition it. A sentence ends after terminal punctuation (. ! ? and the
// full-width 。！？, with any closing quotes or brackets after it) followed by whitespace, or the full-width marks
// alone, or after a blank line; the whitespace that follows belongs to it. These do not end one: a full stop before
// a lowercase ASCII letter ("e.g. the"), after an abbreviation a number follows ("Fig. 3"), or after a line's
// leading enumerator ("1. Install", "6.1.1. Compilation"); and a contents line's dot leader with its page number
// (". . . 174"). A sentence that has run past
// kMaximumSentenceCodePoints ends at its next whitespace, so a table or a list without punctuation still gives
// citable pieces. Whitespace before the first sentence belongs to it, and whitespace after the last one too.
inline constexpr std::uint64_t kMaximumSentenceCodePoints = 1'000;
[[nodiscard]] std::vector<SentenceSpan> split_sentences(std::string_view text);

// The number of Unicode code points in UTF-8 text (continuation bytes are not counted).
[[nodiscard]] std::uint64_t code_point_count(std::string_view text) noexcept;

// A chunk's position in the registry: sources().at(source).chunks.at(chunk).
struct CitationId {
    std::uint32_t source = 0;
    std::uint32_t chunk  = 0;

    [[nodiscard]] friend bool operator==(CitationId, CitationId) noexcept = default;
};

struct CitationChunk {
    std::string text;        // verbatim, as cited_text reports it
    std::uint64_t start = 0; // code points (Text), block index (Content, search results), or 0 (PDF)
    std::uint64_t end   = 0; // exclusive
    std::uint32_t page  = 0; // PDF: the chunk's page number, 1-based
};

struct CitableSource {
    DocumentSource::Kind kind     = DocumentSource::Kind::Document;
    DocumentSource::Format format = DocumentSource::Format::Text;
    std::size_t index             = 0; // document_index, or search_result_index
    std::optional<std::string> title;  // document_title (null without one); a search result's title
    std::string source;                // a search result's source
    std::vector<CitationChunk> chunks;
};

// A cited run of one source's chunks, [first, last] inclusive.
struct ResolvedCitation {
    const CitableSource* source = nullptr;
    std::size_t first           = 0;
    std::size_t last            = 0;

    [[nodiscard]] std::string cited_text() const;
};

// The citable sources of one request. Built once by resolve_documents and read-only after, so the response
// encoders of every generation the request runs share it.
class CitationRegistry {
public:
    // Appends a source; its position is the S of its chunk IDs.
    std::size_t add(CitableSource source);

    [[nodiscard]] bool empty() const noexcept { return sources_.empty(); }
    [[nodiscard]] const std::vector<CitableSource>& sources() const noexcept { return sources_; }

    // The chunk an ID spelling ("S.C") names, or nullopt for one the request never showed.
    [[nodiscard]] std::optional<CitationId> parse_id(std::string_view spelling) const;

    // The IDs a history citation names, in chunk order; empty when the request does not hold its source or the
    // location covers none of its chunks. A PDF's page location narrows to the run of chunks on those pages whose
    // text is the citation's cited_text, the sentences the model cited.
    [[nodiscard]] std::vector<CitationId> ids_for(const CitationLocation& location) const;

    // A cite tag's IDs as citations: one per run of consecutive chunks of one source, in the order the tag listed
    // them.
    [[nodiscard]] std::vector<ResolvedCitation> resolve(const std::vector<CitationId>& ids) const;

private:
    std::vector<CitableSource> sources_;
};

// The marker the prompt puts before a chunk: "[S.C] ".
[[nodiscard]] std::string chunk_marker(std::size_t source, std::size_t chunk);
// `<cite ids="S.C,S.C">`, as the model writes it.
[[nodiscard]] std::string cite_open_tag(const std::vector<CitationId>& ids);
inline constexpr std::string_view kCiteCloseTag = "</cite>";

// The instruction the model reads once, after the request's first citable source.
[[nodiscard]] std::string citation_instruction();

// Splits generated text into plain and cited spans, as it streams. A tag is `<cite ids="0.1,0.2">` (single quotes,
// spaces around `=`, IDs separated by commas or spaces, and brackets around them accepted) up to its `</cite>`.
// - A tag whose IDs name no chunk of the request is dropped with its closing tag, and its text stays plain; IDs
//   it names among valid ones are dropped (invalid_tags() counts the dropped tags).
// - A `<cite>` without IDs, and a `</cite>` that closes no tag, are text.
// - A tag opened inside an open tag closes it first; the outer closing tag is then dropped.
// - A cited span begins (Event::Kind::Open) only with its first letter or digit, so an empty cite tag gives
//   nothing, and a tag around punctuation alone (a model that writes `claim <cite ids="0.1">.</cite>`) gives its
//   citations to the text before it (Event::Kind::Attach) rather than a cited span of ".".
// - finish() closes a tag still open (the output limit or a tool call ended the text) and drops a tag that the end
//   cut part-way; other held text is released.
// Text that may begin a tag is held until it is decided, at most kMaximumCiteTagBytes.
class CiteParser {
public:
    inline static constexpr std::size_t kMaximumCiteTagBytes = 512;

    struct Event {
        enum class Kind : std::uint8_t {
            Text,   // `text` continues the current span
            Open,   // a cited span begins: `citations`
            Close,  // the cited span ends
            Attach, // `citations` belong to the current span, which `text` (punctuation) continues
        };
        Kind kind = Kind::Text;
        std::string text;
        std::vector<ResolvedCitation> citations;
    };

    explicit CiteParser(const CitationRegistry& registry) : registry_(&registry) {}

    [[nodiscard]] std::vector<Event> feed(std::string_view text);
    [[nodiscard]] std::vector<Event> finish();

    [[nodiscard]] std::size_t invalid_tags() const noexcept { return invalid_tags_; }

private:
    enum class State : std::uint8_t {
        Plain,
        Cited,   // inside a valid tag
        Dropped, // inside a tag whose IDs named nothing: its text is plain, its closing tag is dropped
    };
    void emit_text(std::vector<Event>& events, std::string_view text);
    void close_span(std::vector<Event>& events);
    // Parses a complete open tag `tag`; nullopt when it carries no `ids` attribute.
    [[nodiscard]] std::optional<std::vector<CitationId>> parse_open_tag(std::string_view tag);

    const CitationRegistry* registry_;
    std::string held_;
    State state_ = State::Plain;
    // The open tag's citations, and its text, until a letter or digit opens the span; false once Open was sent.
    std::vector<ResolvedCitation> pending_;
    std::string pending_text_;
    bool span_open_ = false;
    // Closing tags of tags that a nested tag closed early, dropped when they come.
    std::size_t orphan_closes_ = 0;
    std::size_t invalid_tags_  = 0;
};

} // namespace infernix::serve
