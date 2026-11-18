// Citations and PDF documents (serve/citations.h, serve/documents.h, media/pdf/pdf.h): the sentence chunker, chunk
// IDs, the cite-tag parser at every split of its input, document rendering and history re-rendering, the Anthropic
// response blocks (aggregate and streamed alike), and PDFs read through PDFium (skipped in a build without it).

#include "media/decode/decode.h"
#include "media/pdf/pdf.h"
#include "serve/anthropic_messages.h"
#include "serve/citations.h"
#include "serve/documents.h"
#include "serve/generation_service.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace infernix::serve;
using Json        = nlohmann::json;
using RequestBody = infernix::serve::RequestJson;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

std::string api_code(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& error) { return error.error().code.empty() ? "<" + error.error().param + ">" : error.error().code; }
    catch (...) { return "wrong_exception"; }
    return {};
}

std::vector<std::string> pieces(std::string_view text) {
    std::vector<std::string> out;
    for (const SentenceSpan& span : split_sentences(text)) {
        out.emplace_back(text.substr(span.byte_begin, span.byte_end - span.byte_begin));
    }
    return out;
}

int test_sentences() {
    int failures = 0;
    const std::string text = "The grass is green. The sky is blue.";
    const std::vector<SentenceSpan> spans = split_sentences(text);
    failures += check(spans.size() == 2 && spans[0].point_begin == 0 && spans[0].point_end == 20 &&
                          spans[1].point_begin == 20 && spans[1].point_end == 36,
                      "Anthropic's example sentences are [0,20) and [20,36)");
    failures += check(pieces("See e.g. the notes. Next one.") ==
                          std::vector<std::string>{"See e.g. the notes. ", "Next one."},
                      "an abbreviation before a lowercase word ended a sentence");
    failures += check(pieces("It weighs 3.5 kg. Really?! Yes.") ==
                          std::vector<std::string>{"It weighs 3.5 kg. ", "Really?! ", "Yes."},
                      "a decimal point or repeated marks split wrongly");
    failures += check(pieces("He said \"stop.\" Then left.") ==
                          std::vector<std::string>{"He said \"stop.\" ", "Then left."},
                      "a closing quote did not stay with its sentence");
    failures += check(pieces("Title\n\nBody text here.") == std::vector<std::string>{"Title\n\n", "Body text here."},
                      "a blank line did not end a chunk");
    // A contents page (as PDFium extracts the CUDA guide's): dot leaders and page numbers end no chunk.
    const std::string contents = "10.8.1.11 tex2DLod() . . . . . . . . . . . . . . 174\n"
                                 "10.8.1.12 tex2DLod() for sparse CUDA arrays . . . . . . 174\n"
                                 "10.8.1.13 tex3D() . . . . . . . . . . . . . . . . . 175\n";
    failures += check(split_sentences(contents).size() == 1, "a contents page's dot leaders were cut into " +
                                                                  std::to_string(split_sentences(contents).size()) +
                                                                  " chunks");
    failures += check(pieces("See Fig. 3 below. Then stop.") ==
                          std::vector<std::string>{"See Fig. 3 below. ", "Then stop."},
                      "an abbreviation before a number ended a sentence");
    // A numbered list and a numbered heading (the CUDA guide's page 40): an item's number starts its chunk.
    failures += check(pieces("Steps:\n1. Install it.\n2. Configure it.") ==
                          std::vector<std::string>{"Steps:\n1. Install it.\n", "2. Configure it."},
                      "a numbered list's chunks begin at the wrong place");
    failures += check(pieces("See the nvcc user manual.\n6.1.1. Compilation Workflow\nSource files compile.") ==
                          std::vector<std::string>{"See the nvcc user manual.\n",
                                                   "6.1.1. Compilation Workflow\nSource files compile."},
                      "a numbered heading's chunk begins at the wrong place");
    failures += check(pieces("It opened in 2019. 2020 was better.") ==
                          std::vector<std::string>{"It opened in 2019. ", "2020 was better."},
                      "a sentence ending in a number did not end");
    failures += check(pieces("  \n\nHello. World.") == std::vector<std::string>{"  \n\nHello. ", "World."},
                      "leading whitespace made a chunk of its own");
    const std::string wide = "\xE4\xBB\x8A\xE6\x97\xA5\xE3\x80\x82\xE6\x98\x8E\xE6\x97\xA5\xE3\x80\x82"; // 今日。明日。
    const std::vector<SentenceSpan> cjk = split_sentences(wide);
    failures += check(cjk.size() == 2 && cjk[0].point_end == 3 && cjk[1].point_end == 6,
                      "full-width sentence marks did not split or count code points");
    std::string table;
    for (int i = 0; i < 600; ++i) { table += "cell "; }
    std::size_t covered = 0;
    bool bounded        = true;
    for (const SentenceSpan& span : split_sentences(table)) {
        bounded = bounded && span.point_end - span.point_begin <= kMaximumSentenceCodePoints + 5U;
        covered += span.byte_end - span.byte_begin;
    }
    failures += check(bounded && covered == table.size(), "a long unpunctuated text gave an unbounded chunk");
    for (const std::string_view sample : {std::string_view("a. b. c."), std::string_view("x"),
                                          std::string_view("  "), std::string_view("Q? A! Done.\n\n\nEnd")}) {
        std::string joined;
        for (const std::string& piece : pieces(sample)) { joined += piece; }
        failures += check(joined == sample, "sentences do not concatenate to the text: " + std::string(sample));
    }
    failures += check(split_sentences("").empty(), "an empty text has a chunk");
    return failures;
}

// Two text documents' sources: "The grass is green. The sky is blue." and "Water is wet."
CitationRegistry sample_registry() {
    CitationRegistry registry;
    CitableSource first;
    first.kind   = DocumentSource::Kind::Document;
    first.format = DocumentSource::Format::Text;
    first.index  = 0;
    first.title  = "Notes";
    first.chunks = {CitationChunk{.text = "The grass is green. ", .start = 0, .end = 20, .page = 0},
                    CitationChunk{.text = "The sky is blue.", .start = 20, .end = 36, .page = 0}};
    registry.add(first);
    CitableSource second;
    second.index  = 1;
    second.chunks = {CitationChunk{.text = "Water is wet.", .start = 0, .end = 13, .page = 0}};
    registry.add(second);
    return registry;
}

int test_registry() {
    int failures                    = 0;
    const CitationRegistry registry = sample_registry();
    failures += check(registry.parse_id("0.1") == CitationId{0, 1} && registry.parse_id("1.0") == CitationId{1, 0},
                      "valid IDs were not found");
    for (const std::string_view bad : {"2.0", "0.2", "1", "a.b", ".1", "0.", "0.1.2", "-1.0"}) {
        failures += check(!registry.parse_id(bad), "an invalid ID was accepted: " + std::string(bad));
    }
    const std::vector<ResolvedCitation> citations =
        registry.resolve({CitationId{0, 0}, CitationId{0, 1}, CitationId{1, 0}, CitationId{0, 0}});
    failures += check(citations.size() == 3 && citations[0].first == 0 && citations[0].last == 1 &&
                          citations[0].cited_text() == "The grass is green. The sky is blue." &&
                          citations[1].source == &registry.sources()[1] && citations[2].first == 0,
                      "consecutive IDs of one source did not merge into one citation");
    CitationLocation location{.kind = CitationLocation::Kind::Char, .source_index = 0, .start = 20, .end = 36,
                              .cited_text = "The sky is blue."};
    failures += check(registry.ids_for(location) == std::vector<CitationId>{CitationId{0, 1}},
                      "a char location did not map back to its chunk");
    location.source_index = 5;
    failures += check(registry.ids_for(location).empty(), "a citation of a missing document mapped to IDs");
    return failures;
}

struct Parsed {
    std::string text; // plain and cited text with spans marked as {ids|text}
    std::size_t invalid_tags = 0;
};

std::string describe(const std::vector<ResolvedCitation>& citations) {
    std::string out;
    for (const ResolvedCitation& citation : citations) {
        if (!out.empty()) { out += ','; }
        out += std::to_string(citation.source->index) + ":" + std::to_string(citation.first) + "-" +
               std::to_string(citation.last);
    }
    return out;
}

Parsed run_parser(const CitationRegistry& registry, const std::vector<std::string_view>& feeds) {
    CiteParser parser(registry);
    std::vector<CiteParser::Event> events;
    for (const std::string_view feed : feeds) {
        for (CiteParser::Event& event : parser.feed(feed)) { events.push_back(std::move(event)); }
    }
    for (CiteParser::Event& event : parser.finish()) { events.push_back(std::move(event)); }
    Parsed out;
    for (const CiteParser::Event& event : events) {
        switch (event.kind) {
        case CiteParser::Event::Kind::Text:
            out.text += event.text;
            break;
        case CiteParser::Event::Kind::Open:
            out.text += "{" + describe(event.citations) + "|";
            break;
        case CiteParser::Event::Kind::Close:
            out.text += "}";
            break;
        case CiteParser::Event::Kind::Attach:
            out.text += "{^" + describe(event.citations) + "|" + event.text + "}";
            break;
        }
    }
    out.invalid_tags = parser.invalid_tags();
    return out;
}

int test_cite_parser() {
    int failures                    = 0;
    const CitationRegistry registry = sample_registry();
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"A <cite ids=\"0.1\">claim</cite> end.", "A {0:1-1|claim} end."},
        {"<cite ids='0.0, 0.1'>both</cite>", "{0:0-1|both}"},
        {"<cite ids=\"[0.1]\">bracketed</cite>", "{0:1-1|bracketed}"},
        {"x <cite ids=\"9.9\">dropped tag</cite> y", "x dropped tag y"},
        {"<cite ids=\"0.1,9.9\">partly valid</cite>", "{0:1-1|partly valid}"},
        {"<cite ids=\"3\">bare number</cite>", "bare number"},
        {"<cite>Book Title</cite> stays", "<cite>Book Title</cite> stays"},
        {"stray </cite> close", "stray </cite> close"},
        {"<cite ids=\"0.0\">a <cite ids=\"0.1\">b</cite> c</cite>.", "{0:0-0|a }{0:1-1|b} c."},
        {"<cite ids=\"0.0\"></cite>empty", "empty"},
        {"Sales rose <cite ids=\"0.1\">.</cite> Next", "Sales rose {^0:1-1|.} Next"},
        {"<cite ids=\"0.1\"> (</cite>x", "{^0:1-1| (}x"},
        {"<cite ids=\"0.1\">, \xC3\xA9t\xC3\xA9</cite>", "{0:1-1|, \xC3\xA9t\xC3\xA9}"},
        {"cut <cite ids=\"0.0\">mid", "cut {0:0-0|mid}"},
        {"cut tag <cite ids=\"0.", "cut tag "},
        {"a <", "a <"},
        {"x < y <b>bold</b> <ci", "x < y <b>bold</b> <ci"},
        {"<cite ids=\"1.0\">w</cite><cite ids=\"0.0\">g</cite>", "{1:0-0|w}{0:0-0|g}"},
    };
    for (const auto& [input, expected] : cases) {
        const Parsed whole = run_parser(registry, {input});
        failures += check(whole.text == expected, "cite parser on '" + input + "' gave '" + whole.text + "'");
        // Every split into two feeds, and one byte at a time, gives the same spans.
        for (std::size_t split = 0; split <= input.size(); ++split) {
            const std::string_view view(input);
            const Parsed parts = run_parser(registry, {view.substr(0, split), view.substr(split)});
            if (parts.text != whole.text) {
                failures += check(false, "cite parser split at " + std::to_string(split) + " of '" + input + "'");
                break;
            }
        }
        std::vector<std::string_view> bytes;
        for (std::size_t i = 0; i < input.size(); ++i) { bytes.push_back(std::string_view(input).substr(i, 1)); }
        failures += check(run_parser(registry, bytes).text == whole.text, "cite parser fed bytewise on '" + input + "'");
    }
    failures += check(run_parser(registry, {"<cite ids=\"9.9\">x</cite>"}).invalid_tags == 1,
                      "a tag naming nothing was not counted");
    std::string long_tag = "<cite ids=\"";
    long_tag += std::string(CiteParser::kMaximumCiteTagBytes, '0');
    failures += check(run_parser(registry, {long_tag, " rest"}).text == long_tag + " rest",
                      "an overlong tag start was held instead of released as text");
    return failures;
}

RequestBody text_document(std::string text, bool citations, std::string title = "Notes") {
    return RequestBody{{"type", "document"},
                       {"source", RequestBody{{"type", "text"}, {"media_type", "text/plain"}, {"data", std::move(text)}}},
                       {"title", std::move(title)},
                       {"citations", RequestBody{{"enabled", citations}}}};
}

RequestBody messages_body(RequestBody messages) {
    return RequestBody{{"model", "local"}, {"max_tokens", 256}, {"messages", std::move(messages)}};
}

GenerationRequest resolved(const GenerationRequest& request, DocumentResolution* resolution = nullptr) {
    DocumentResolution result = resolve_documents(request, {});
    GenerationRequest out     = result.request ? *result.request : request;
    if (resolution != nullptr) { *resolution = std::move(result); }
    return out;
}

std::string turn_text(const ChatTurn& turn) {
    std::string text;
    for (const ContentPart& part : turn.content) { text += part.text; }
    return text;
}

int test_document_rendering() {
    int failures = 0;
    RequestBody user{{"role", "user"},
                     {"content", RequestBody::array({text_document("The grass is green. The sky is blue.", true),
                                                     text_document("Water is wet.", true, "Second"),
                                                     RequestBody{{"type", "text"}, {"text", "Colours?"}}})}};
    const AnthropicMessagesRequest parsed =
        parse_anthropic_messages_request(messages_body(RequestBody::array({user})), RequestLimits{});
    DocumentResolution resolution;
    const GenerationRequest request = resolved(parsed.generation, &resolution);
    const std::string text          = turn_text(request.messages[0]);
    const std::string instruction   = citation_instruction();
    failures += check(text == "<document>\n<title>Notes</title>\n[0.0] The grass is green. [0.1] The sky is blue.\n"
                              "</document>\n" +
                                  instruction + "\n<document>\n<title>Second</title>\n[1.0] Water is wet.\n</document>\nColours?",
                      "citable documents rendered as:\n" + text);
    failures += check(resolution.citations != nullptr && resolution.citations->sources().size() == 2 &&
                          resolution.citations->sources()[1].chunks[0].text == "Water is wet.",
                      "the registry does not hold both documents");

    RequestBody mixed = user;
    mixed["content"][1]["citations"]["enabled"] = false;
    failures += check(api_code([&] {
                          (void)parse_anthropic_messages_request(messages_body(RequestBody::array({mixed})), {});
                      }) == "<messages>",
                      "citations on some documents only were accepted");
    RequestBody formatted              = messages_body(RequestBody::array({user}));
    formatted["output_config"]         = RequestBody{{"format", RequestBody{{"type", "json_schema"},
                                                                  {"schema", RequestBody{{"type", "object"}}}}}};
    failures += check(api_code([&] { (void)parse_anthropic_messages_request(formatted, {}); }) ==
                          "<output_config.format>",
                      "citations with a structured output format were accepted");

    RequestBody search{{"role", "user"},
                       {"content", RequestBody::array({RequestBody{
                                       {"type", "search_result"},
                                       {"source", "https://example.com/a"},
                                       {"title", "A"},
                                       {"citations", RequestBody{{"enabled", true}}},
                                       {"content", RequestBody::array({RequestBody{{"type", "text"}, {"text", "One."}},
                                                                       RequestBody{{"type", "text"}, {"text", "Two."}}})}}})}};
    const GenerationRequest searched = resolved(
        parse_anthropic_messages_request(messages_body(RequestBody::array({search})), {}).generation);
    failures += check(turn_text(searched.messages[0]) ==
                          "<search_result>\n<source>https://example.com/a</source>\n<title>A</title>\n[0.0] One.\n\n"
                          "[0.1] Two.\n</search_result>\n" +
                              instruction + "\n",
                      "a citable search result rendered as:\n" + turn_text(searched.messages[0]));
    return failures;
}

infernix::serve::GenerationOutcome outcome_with(std::string text) {
    GenerationOutcome outcome;
    outcome.text          = std::move(text);
    outcome.finish_reason = infernix::FinishReason::StopToken;
    return outcome;
}

// The blocks of an SSE stream, rebuilt as an SDK accumulates them: text and citations per text block.
Json stream_blocks(const std::vector<std::string>& events) {
    Json blocks = Json::array();
    for (const std::string& event : events) {
        const std::size_t data = event.find("data: ");
        if (data == std::string::npos) { continue; }
        const Json payload = Json::parse(event.substr(data + 6));
        const std::string type = payload.value("type", "");
        if (type == "content_block_start" && payload["content_block"]["type"] == "text") {
            Json block{{"type", "text"}, {"text", ""}};
            if (payload["content_block"].contains("citations")) { block["citations"] = Json::array(); }
            blocks.push_back(block);
        } else if (type == "content_block_delta") {
            const Json& delta = payload["delta"];
            if (delta["type"] == "text_delta") {
                blocks.back()["text"] = blocks.back()["text"].get<std::string>() + delta["text"].get<std::string>();
            } else if (delta["type"] == "citations_delta") {
                blocks.back()["citations"].push_back(delta["citation"]);
            }
        }
    }
    Json kept = Json::array();
    for (const Json& block : blocks) {
        if (!block["text"].get<std::string>().empty()) { kept.push_back(block); }
    }
    return kept;
}

int test_response_blocks() {
    int failures = 0;
    auto registry = std::make_shared<const CitationRegistry>(sample_registry());
    const std::string raw =
        "The sky <cite ids=\"0.1\">is blue</cite>, and <cite ids=\"0.0,1.0\">grass is green</cite> <ci";
    const GenerationOutcome outcome = outcome_with(raw);
    const AnthropicResponseIdentity identity{.request_id = "req_1", .message_id = "msg_1", .model = "local"};
    const Json response = Json::parse(make_anthropic_messages_response(identity, outcome, false, registry.get()));
    const Json& content = response["content"];
    failures += check(content.size() == 5 && content[0]["text"] == "The sky " && content[1]["text"] == "is blue" &&
                          content[1]["citations"].size() == 1 &&
                          content[1]["citations"][0] == Json{{"type", "char_location"},
                                                             {"cited_text", "The sky is blue."},
                                                             {"document_index", 0},
                                                             {"document_title", "Notes"},
                                                             {"start_char_index", 20},
                                                             {"end_char_index", 36}} &&
                          content[3]["citations"].size() == 2 && content[3]["citations"][1]["document_title"].is_null() &&
                          content[4]["text"] == " <ci" && !content[0].contains("citations"),
                      "aggregate cited blocks: " + content.dump());
    const Json aggregate = [&] {
        Json blocks = Json::array();
        for (const Json& block : content) {
            Json copy{{"type", "text"}, {"text", block["text"]}};
            if (block.contains("citations")) { copy["citations"] = block["citations"]; }
            blocks.push_back(copy);
        }
        return blocks;
    }();
    // The stream gives the same blocks at every split of the text.
    for (std::size_t split = 0; split <= raw.size(); ++split) {
        AnthropicMessagesStream stream(identity, 0, false, {}, false, registry);
        std::vector<std::string> events{stream.start()};
        for (std::string& event : stream.content_delta(raw.substr(0, split))) { events.push_back(std::move(event)); }
        for (std::string& event : stream.content_delta(raw.substr(split))) { events.push_back(std::move(event)); }
        for (std::string& event : stream.finish(outcome)) { events.push_back(std::move(event)); }
        if (stream_blocks(events) != aggregate) {
            failures += check(false, "streamed blocks at split " + std::to_string(split) + ": " +
                                         stream_blocks(events).dump());
            break;
        }
    }
    // A tag around punctuation alone cites the text before it, aggregate and streamed alike.
    const std::string trailing = "Grass is green <cite ids=\"0.0\">.</cite> And <cite ids=\"0.1\">.</cite>";
    const Json attached =
        Json::parse(make_anthropic_messages_response(identity, outcome_with(trailing), false, registry.get()))["content"];
    failures += check(attached.size() == 2 && attached[0]["text"] == "Grass is green ." &&
                          attached[0]["citations"][0]["cited_text"] == "The grass is green. " &&
                          attached[1]["text"] == " And ." && attached[1]["citations"].size() == 1,
                      "a tag around punctuation did not cite the text before it: " + attached.dump());
    for (std::size_t split = 0; split <= trailing.size(); ++split) {
        AnthropicMessagesStream stream(identity, 0, false, {}, false, registry);
        std::vector<std::string> events{stream.start()};
        for (std::string& event : stream.content_delta(trailing.substr(0, split))) { events.push_back(std::move(event)); }
        for (std::string& event : stream.content_delta(trailing.substr(split))) { events.push_back(std::move(event)); }
        for (std::string& event : stream.finish(outcome_with(trailing))) { events.push_back(std::move(event)); }
        if (stream_blocks(events) != attached) {
            failures += check(false, "streamed attached citation at split " + std::to_string(split) + ": " +
                                         stream_blocks(events).dump());
            break;
        }
    }
    // Without citations the text stays one block, cite tags and all.
    const Json plain = Json::parse(make_anthropic_messages_response(identity, outcome, false, nullptr));
    failures += check(plain["content"].size() == 1 && plain["content"][0]["text"] == raw,
                      "a response without citations was split");
    return failures;
}

int test_history_identity() {
    int failures = 0;
    RequestBody user{{"role", "user"},
                     {"content", RequestBody::array({text_document("The grass is green. The sky is blue.", true),
                                                     RequestBody{{"type", "text"}, {"text", "Colours?"}}})}};
    DocumentResolution first;
    (void)resolved(parse_anthropic_messages_request(messages_body(RequestBody::array({user})), {}).generation, &first);
    const std::string raw = "Both: <cite ids=\"0.1\">the sky is blue</cite> and <cite ids=\"0.0,0.1\">grass is "
                            "green and sky blue</cite>. <cite ids=\"0.1,0.0\">reversed</cite>";
    const AnthropicResponseIdentity identity{.request_id = "req_1", .message_id = "msg_1", .model = "local"};
    const Json response =
        Json::parse(make_anthropic_messages_response(identity, outcome_with(raw), false, first.citations.get()));
    RequestBody assistant{{"role", "assistant"}, {"content", RequestBody::parse(response["content"].dump())}};
    RequestBody next{{"role", "user"}, {"content", "More?"}};
    const GenerationRequest history = resolved(
        parse_anthropic_messages_request(messages_body(RequestBody::array({user, assistant, next})), {}).generation);
    failures += check(turn_text(history.messages[1]) == raw,
                      "an echoed cited turn did not render as the model wrote it: " + turn_text(history.messages[1]));
    // The prompt the Engine gets holds the same text.
    ServeOptions options;
    const infernix::PromptInput input =
        to_prompt_input(history, resolve_prompt_semantics(history, options), [](const ContentPart&) {
            return infernix::OwnedMedia{};
        });
    std::string prompt_text;
    for (const infernix::MessagePart& part : input.messages[1].parts) { prompt_text += part.text; }
    failures += check(prompt_text == raw, "the echoed cited turn's prompt parts differ from the model's text");
    return failures;
}

// A minimal PDF: one US Letter page per entry, its text in 12-point Helvetica at the top left (an empty entry draws
// a red box over x 100-400, y 400-600 points, origin bottom left, and no text, as a scanned page would have).
std::vector<std::uint8_t> make_pdf(const std::vector<std::string>& pages) {
    std::string pdf = "%PDF-1.4\n";
    std::vector<std::size_t> offsets;
    const auto object = [&](const std::string& body) {
        offsets.push_back(pdf.size());
        pdf += std::to_string(offsets.size()) + " 0 obj\n" + body + "\nendobj\n";
    };
    object("<< /Type /Catalog /Pages 2 0 R >>");
    std::string kids;
    for (std::size_t page = 0; page < pages.size(); ++page) { kids += std::to_string(4 + 2 * page) + " 0 R "; }
    object("<< /Type /Pages /Kids [" + kids + "] /Count " + std::to_string(pages.size()) + " >>");
    object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    for (std::size_t page = 0; page < pages.size(); ++page) {
        object("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
               "/Contents " +
               std::to_string(5 + 2 * page) + " 0 R >>");
        const std::string stream = pages[page].empty() ? "1 0 0 rg 100 400 300 200 re f"
                                                       : "BT /F1 12 Tf 72 720 Td (" + pages[page] + ") Tj ET";
        object("<< /Length " + std::to_string(stream.size()) + " >>\nstream\n" + stream + "\nendstream");
    }
    const std::size_t xref = pdf.size();
    pdf += "xref\n0 " + std::to_string(offsets.size() + 1) + "\n0000000000 65535 f \n";
    for (const std::size_t offset : offsets) {
        char entry[32];
        std::snprintf(entry, sizeof(entry), "%010zu 00000 n \n", offset);
        pdf += entry;
    }
    pdf += "trailer\n<< /Size " + std::to_string(offsets.size() + 1) + " /Root 1 0 R >>\nstartxref\n" +
           std::to_string(xref) + "\n%%EOF\n";
    return std::vector<std::uint8_t>(pdf.begin(), pdf.end());
}

GenerationRequest pdf_request(const std::vector<std::vector<std::uint8_t>>& pdfs, bool citations) {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = infernix::ChatRole::User;
    for (const std::vector<std::uint8_t>& bytes : pdfs) {
        auto document        = std::make_shared<DocumentSource>();
        document->format     = DocumentSource::Format::Pdf;
        document->title      = "Report";
        document->citations  = citations;
        document->pdf.kind   = infernix::product::media_acquire::SourceKind::Bytes;
        document->pdf.bytes  = bytes;
        ContentPart part;
        part.kind     = ContentKind::Document;
        part.document = std::move(document);
        turn.content.push_back(std::move(part));
    }
    request.messages.push_back(std::move(turn));
    return request;
}

DocumentResolveOptions pdf_options(std::uint64_t vision_tokens, std::uint64_t max_context) {
    DocumentResolveOptions options;
    if (vision_tokens != 0) {
        options.vision = infernix::VisionImageBudget{
            .prompt_tokens = vision_tokens, .token_edge = 32, .min_pixels = 65'536, .max_pixels = 16'777'216};
    }
    options.max_context = max_context;
    options.acquire     = [](const infernix::product::media_acquire::Source& source) { return source.bytes; };
    return options;
}

std::vector<const ContentPart*> images_of(const GenerationRequest& request) {
    std::vector<const ContentPart*> images;
    for (const ContentPart& part : request.messages[0].content) {
        if (part.kind == ContentKind::Image) { images.push_back(&part); }
    }
    return images;
}

// Two requests rendering the same page at once (audit 2026-11-18 F-16): the page is cached and counted once, whichever
// stores it first (the second appended a duplicate and counted its bytes again, so the cache evicted early). Each
// render waits in its first checkpoint until both have missed the cache.
int test_pdf_cache_concurrent_render() {
    if (!infernix::media::pdf::available()) { return 0; }
    infernix::serve::PdfCache cache(64ULL << 20);
    const std::vector<std::uint8_t> pdf = make_pdf({"One page."});
    (void)cache.text(pdf);
    const std::size_t text_bytes = cache.retained_bytes();
    const std::vector<infernix::media::pdf::RenderRequest> page{{.number = 1, .size = {.width = 64, .height = 64}}};
    std::atomic<int> arrived{0};
    const std::function<void()> checkpoint = [&] {
        thread_local bool counted = false;
        if (counted) { return; }
        counted = true;
        ++arrived;
        const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (arrived.load() < 2 && std::chrono::steady_clock::now() < limit) { std::this_thread::yield(); }
    };
    std::array<std::vector<infernix::serve::PdfPageImage>, 2> images;
    std::thread first([&] { images[0] = cache.images(pdf, page, checkpoint); });
    std::thread second([&] { images[1] = cache.images(pdf, page, checkpoint); });
    first.join();
    second.join();
    const std::size_t image_bytes = images[0].empty() || !images[0][0] ? 0 : images[0][0]->size();
    return check(arrived.load() == 2 && image_bytes != 0 && cache.retained_bytes() == text_bytes + image_bytes,
                 "a page two requests rendered at once was cached twice (retained " +
                     std::to_string(cache.retained_bytes() - text_bytes) + " bytes for a " +
                     std::to_string(image_bytes) + "-byte image)");
}

int test_pdf() {
    if (!infernix::media::pdf::available()) {
        std::cout << "SKIP: built without PDFium\n";
        return check(api_code([] { (void)resolve_documents(pdf_request({make_pdf({"x"})}, false), pdf_options(0, 0)); }) ==
                         "documents_not_supported",
                     "a PDF was accepted by a build without PDFium");
    }
    int failures = 0;
    const std::vector<std::uint8_t> pdf = make_pdf({"First page. It has two sentences.", "Second page here."});
    const std::vector<infernix::media::pdf::PageText> text = infernix::media::pdf::read_text(pdf, 100);
    failures += check(text.size() == 2 && text[0].text.find("First page. It has two sentences.") != std::string::npos &&
                          text[1].number == 2 && text[0].width_points == 612.0 && text[0].height_points == 792.0,
                      "PDF page text or size was not read");
    bool too_many = false;
    try {
        (void)infernix::media::pdf::read_text(pdf, 1);
    } catch (const infernix::media::pdf::Error& error) {
        too_many = error.kind() == infernix::media::pdf::ErrorKind::TooManyPages;
    }
    failures += check(too_many, "a PDF over the page limit was read");
    const infernix::media::pdf::RenderSize size = infernix::media::pdf::render_size(
        612, 792, {.max_pixels = 1'048'576, .min_pixels = 65'536, .edge_multiple = 32, .max_points_scale = 2.0});
    failures += check(size.width == 896 && size.height == 1152, "a Letter page at 1M pixels did not size to 896 x 1152");
    // At the processor's minimum the grid sizes round below it; the nearest size within the bounds is kept.
    const infernix::media::pdf::RenderSize smallest = infernix::media::pdf::render_size(
        612, 792, {.max_pixels = 65'536, .min_pixels = 65'536, .edge_multiple = 32, .max_points_scale = 2.0});
    failures += check(smallest.width == 256 && smallest.height == 256,
                      "a Letter page at the minimum sized to " + std::to_string(smallest.width) + " x " +
                          std::to_string(smallest.height));
    const infernix::media::pdf::RenderSize near_min = infernix::media::pdf::render_size(
        612, 792, {.max_pixels = 90'000, .min_pixels = 65'536, .edge_multiple = 32, .max_points_scale = 2.0});
    failures += check(near_min.width % 32 == 0 && near_min.height % 32 == 0 &&
                          near_min.width * near_min.height >= 65'536 && near_min.width * near_min.height <= 90'000,
                      "a page just above the minimum left the bounds");
    const infernix::media::pdf::RenderRequest render_request{.number = 1, .size = size};
    const std::vector<infernix::media::pdf::RenderedPage> rendered =
        infernix::media::pdf::render(pdf, std::span(&render_request, 1));
    bool white_corner = rendered.size() == 1 && rendered[0].rgb.size() == 896U * 1152U * 3U &&
                        rendered[0].rgb[0] == 255 && rendered[0].rgb[1] == 255 && rendered[0].rgb[2] == 255;
    // The text (baseline 72 points below the top, 896 / 612 pixels a point) inks only the page's top rows.
    std::size_t ink_top = 0;
    std::size_t ink_rest = 0;
    for (std::size_t y = 0; white_corner && y < 1152U; ++y) {
        for (std::size_t x = 0; x < 896U; ++x) {
            if (rendered[0].rgb[(y * 896U + x) * 3U] < 128) { (y < 160U ? ink_top : ink_rest) += 1; }
        }
    }
    failures += check(white_corner && ink_top > 0 && ink_rest == 0, "a rendered page's text is not at its top on white");
    // The textless page's red box (rows 192-392 points from the top, columns 100-400) comes out red, not blue:
    // PDFium's BGRx is converted to RGB.
    const std::vector<std::uint8_t> box_pdf = make_pdf({""});
    const std::vector<infernix::media::pdf::RenderedPage> box =
        infernix::media::pdf::render(box_pdf, std::span(&render_request, 1));
    const auto pixel = [&](std::size_t x, std::size_t y) { return &box[0].rgb[(y * 896U + x) * 3U]; };
    const std::uint8_t* inside  = pixel(366, 427); // the box's centre: 250 x 292 points from the top left
    const std::uint8_t* outside = pixel(366, 100);
    failures += check(inside[0] > 200 && inside[1] < 60 && inside[2] < 60 && outside[0] == 255 && outside[2] == 255,
                      "the red box rendered as (" + std::to_string(inside[0]) + "," + std::to_string(inside[1]) + "," +
                          std::to_string(inside[2]) + ")");
    const std::vector<std::uint8_t> encoded =
        infernix::media::decode::encode_raw_rgb(rendered[0].width, rendered[0].height, rendered[0].rgb);
    const infernix::media::decode::Image decoded = infernix::media::decode::decode_image(encoded, {});
    failures += check(decoded.width == 896 && decoded.height == 1152 && decoded.rgb == rendered[0].rgb,
                      "raw RGB did not round-trip through decode_image");

    // Two requests read and render two PDFs at once: PDFium's lock is taken per page, and each result equals its
    // read alone.
    {
        std::vector<std::string> first_pages;
        std::vector<std::string> second_pages;
        for (int page = 0; page < 60; ++page) {
            first_pages.push_back("First document page " + std::to_string(page) + ".");
            second_pages.push_back(page % 7 == 0 ? std::string() : "Second document page " + std::to_string(page) + ".");
        }
        const std::vector<std::uint8_t> first  = make_pdf(first_pages);
        const std::vector<std::uint8_t> second = make_pdf(second_pages);
        const auto read = [](const std::vector<std::uint8_t>& bytes) {
            std::string text;
            for (const auto& page : infernix::media::pdf::read_text(bytes, 100)) { text += page.text + "\f"; }
            std::vector<infernix::media::pdf::RenderRequest> requests;
            for (std::uint32_t page = 1; page <= 60; page += 6) { requests.push_back({.number = page, .size = {256, 320}}); }
            for (const auto& image : infernix::media::pdf::render(bytes, requests)) {
                text.append(image.rgb.begin(), image.rgb.end());
            }
            return text;
        };
        const std::string alone_first  = read(first);
        const std::string alone_second = read(second);
        bool same = true;
        for (int round = 0; round < 4 && same; ++round) {
            std::string together_first;
            std::string together_second;
            std::thread a([&] { together_first = read(first); });
            std::thread b([&] { together_second = read(second); });
            a.join();
            b.join();
            same = together_first == alone_first && together_second == alone_second;
        }
        failures += check(same, "PDFs read at once differ from their reads alone");
    }

    // Text only without vision; with vision every page also as an image, from the budget.
    const GenerationRequest text_only = *resolve_documents(pdf_request({pdf}, false), pdf_options(0, 0)).request;
    failures += check(images_of(text_only).empty() && turn_text(text_only.messages[0]).find("<page number=\"2\">\nSecond page here.") != std::string::npos,
                      "a PDF without vision was not text only:\n" + turn_text(text_only.messages[0]));
    const GenerationRequest with_images = *resolve_documents(pdf_request({pdf}, false), pdf_options(32'768, 131'072)).request;
    failures += check(images_of(with_images).size() == 2 &&
                          images_of(with_images)[0]->source.media_type == kPdfPageImageMediaType,
                      "a PDF with vision did not get its page images");
    // A later PDF does not change an earlier one's pages (the budget is spent in request order).
    const GenerationRequest two =
        *resolve_documents(pdf_request({pdf, make_pdf({"Other.", "More.", "Most."})}, false), pdf_options(2'048, 0)).request;
    const GenerationRequest one = *resolve_documents(pdf_request({pdf}, false), pdf_options(2'048, 0)).request;
    failures += check(images_of(two).size() >= 2 && images_of(one).size() == 2 &&
                          images_of(two)[0]->source.bytes == images_of(one)[0]->source.bytes,
                      "a later PDF resized an earlier PDF's pages");
    // Too little budget for every page: the page without a text layer gets an image, the others none.
    const GenerationRequest scarce = *resolve_documents(pdf_request({make_pdf({"Text one.", "Text two.", ""})}, false),
                                                        pdf_options(128, 0))
                                          .request;
    failures += check(images_of(scarce).size() == 1 && turn_text(scarce.messages[0]).find("<page number=\"3\">\n</page>") != std::string::npos,
                      "a scarce budget did not go to the page without text");
    failures += check(api_code([] { (void)resolve_documents(pdf_request({make_pdf({""})}, false), pdf_options(0, 0)); }) ==
                          "pdf_no_text",
                      "a PDF without a text layer was accepted without vision");
    // A 300-page scan at --max-context 65536: the budget (16,384 tokens) gives the smallest image (64 tokens) to
    // its first 256 pages; the other 44 say they were not read.
    {
        const GenerationRequest scan =
            *resolve_documents(pdf_request({make_pdf(std::vector<std::string>(300))}, false), pdf_options(32'768, 65'536))
                 .request;
        const std::string text = turn_text(scan.messages[0]);
        std::size_t notes      = 0;
        for (std::size_t at = text.find(kPdfPageUnreadOverBudget); at != std::string::npos;
             at = text.find(kPdfPageUnreadOverBudget, at + 1)) {
            ++notes;
        }
        failures += check(images_of(scan).size() == 256 && notes == 44 &&
                              text.find(std::string("<page number=\"300\">\n") + std::string(kPdfPageUnreadOverBudget)) !=
                                  std::string::npos,
                          "a 300-page scan gave " + std::to_string(images_of(scan).size()) + " images and " +
                              std::to_string(notes) + " notes");
    }
    // 1,000 pages a request: one PDF over it, and two that only together are.
    failures += check(api_code([] {
                          (void)resolve_documents(pdf_request({make_pdf(std::vector<std::string>(1'001, "Page."))}, false),
                                                  pdf_options(0, 0));
                      }) == "pdf_too_many_pages",
                      "a 1,001-page PDF was accepted");
    {
        const std::vector<std::uint8_t> half = make_pdf(std::vector<std::string>(600, "Page."));
        failures += check(api_code([&] { (void)resolve_documents(pdf_request({half, half}, false), pdf_options(0, 0)); }) ==
                              "pdf_too_many_pages",
                          "two PDFs of 1,200 pages together were accepted");
        failures += check(resolve_documents(pdf_request({half}, false), pdf_options(0, 0)).request.has_value(),
                          "a 600-page PDF was refused");
    }
    failures += check(api_code([] {
                          (void)resolve_documents(pdf_request({std::vector<std::uint8_t>{'n', 'o', 't'}}, false),
                                                  pdf_options(0, 0));
                      }) == "invalid_pdf",
                      "bytes that are not a PDF were accepted");

    // Page citations: the sentences a tag names, located by page.
    DocumentResolution cited = resolve_documents(pdf_request({pdf}, true), pdf_options(0, 0));
    const AnthropicResponseIdentity identity{.request_id = "req_1", .message_id = "msg_1", .model = "local"};
    const Json response = Json::parse(make_anthropic_messages_response(
        identity, outcome_with("<cite ids=\"0.1,0.2\">two sentences</cite>"), false, cited.citations.get()));
    const Json& citation = response["content"][0]["citations"][0];
    failures += check(response["content"][0]["citations"].size() == 1 && citation["type"] == "page_location" &&
                          citation["start_page_number"] == 1 && citation["end_page_number"] == 3 &&
                          citation["cited_text"].get<std::string>().find("It has two sentences.") != std::string::npos,
                      "a PDF citation: " + response["content"].dump());
    CitationLocation location{.kind         = CitationLocation::Kind::Page,
                              .source_index = 0,
                              .start        = 1,
                              .end          = 3,
                              .cited_text   = citation["cited_text"].get<std::string>()};
    failures += check(cited.citations->ids_for(location) == std::vector<CitationId>{CitationId{0, 1}, CitationId{0, 2}},
                      "a page citation did not map back to the sentences it cited");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_sentences();
    failures += test_registry();
    failures += test_cite_parser();
    failures += test_document_rendering();
    failures += test_response_blocks();
    failures += test_history_identity();
    failures += test_pdf();
    failures += test_pdf_cache_concurrent_render();
    if (failures != 0) {
        std::cerr << failures << " citation check(s) failed\n";
        return 1;
    }
    std::cout << "citations: all checks passed\n";
    return 0;
}
