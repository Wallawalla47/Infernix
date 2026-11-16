#include "serve/documents.h"

#include "core/content_hash.h"
#include "media/decode/decode.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace infernix::serve {
namespace {

[[noreturn]] void document_error(std::string message, std::string code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = std::move(code);
    error.message = std::move(message);
    throw ApiException(std::move(error));
}

[[noreturn]] void pdf_error(const media::pdf::Error& error) {
    switch (error.kind()) {
    case media::pdf::ErrorKind::Unavailable:
        document_error(error.what(), "documents_not_supported");
    case media::pdf::ErrorKind::Invalid:
        document_error(std::string("invalid PDF document: ") + error.what(), "invalid_pdf");
    case media::pdf::ErrorKind::Encrypted:
        document_error(std::string(error.what()) + "; encrypted PDFs are not supported", "pdf_encrypted");
    case media::pdf::ErrorKind::TooManyPages:
        document_error(error.what(), "pdf_too_many_pages");
    }
    document_error(error.what(), "invalid_pdf");
}

std::uint64_t hash_bytes(std::span<const std::uint8_t> bytes) noexcept {
    return content_hash(reinterpret_cast<const std::byte*>(bytes.data()), bytes.size());
}

bool is_blank(std::string_view text) noexcept {
    return std::all_of(text.begin(), text.end(), [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    });
}

bool same_request(const media::pdf::RenderRequest& a, const media::pdf::RenderRequest& b) noexcept {
    return a.number == b.number && a.size == b.size;
}

ContentPart text_part(std::string text) {
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "text";
    part.text     = std::move(text);
    return part;
}

// One document's parts: text accumulates until a page image interrupts it.
struct PartWriter {
    std::vector<ContentPart>& parts;
    std::string text;

    void image(const std::vector<std::uint8_t>& bytes) {
        flush();
        ContentPart part;
        part.kind              = ContentKind::Image;
        part.type_raw          = "document";
        part.source.kind       = product::media_acquire::SourceKind::Bytes;
        part.source.media_type = std::string(kPdfPageImageMediaType);
        part.source.bytes      = bytes;
        part.derived           = true;
        parts.push_back(std::move(part));
    }

    void flush() {
        if (text.empty()) { return; }
        parts.push_back(text_part(std::move(text)));
        text.clear();
    }
};

// A PDF of the request: its bytes, text layer and the images its pages get.
struct PdfInput {
    std::vector<std::uint8_t> bytes;
    std::shared_ptr<const std::vector<media::pdf::PageText>> text;
    std::vector<PdfPageImage> images; // per page, null for a page without an image
};

void append_header(std::string& text, const DocumentSource& document, std::string_view title) {
    if (document.kind == DocumentSource::Kind::SearchResult) {
        text += "<search_result>\n<source>" + document.source + "</source>\n<title>" + document.title + "</title>\n";
        return;
    }
    text += "<document>\n";
    if (!title.empty()) { text += "<title>" + std::string(title) + "</title>\n"; }
    if (!document.context.empty()) { text += "<context>" + document.context + "</context>\n"; }
}

// Sentences of `text` after their IDs (with citations, as source `source_position`) or `text` as it is, adding the
// chunks to `citable`. `page` is a PDF page's number; 0 for plain text, whose chunks record code-point offsets.
void append_sentences(std::string& out, std::string_view text, CitableSource* citable, std::size_t source_position,
                      std::uint32_t page) {
    if (citable == nullptr) {
        out += text;
        return;
    }
    for (const SentenceSpan& span : split_sentences(text)) {
        CitationChunk chunk;
        chunk.text  = std::string(text.substr(span.byte_begin, span.byte_end - span.byte_begin));
        chunk.start = page == 0 ? span.point_begin : 0;
        chunk.end   = page == 0 ? span.point_end : 0;
        chunk.page  = page;
        out += chunk_marker(source_position, citable->chunks.size());
        out += chunk.text;
        citable->chunks.push_back(std::move(chunk));
    }
}

void append_blocks(std::string& out, const std::vector<std::string>& blocks, CitableSource* citable,
                   std::size_t source_position) {
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        if (index != 0) { out += "\n\n"; }
        if (citable != nullptr) {
            out += chunk_marker(source_position, index);
            citable->chunks.push_back(CitationChunk{.text = blocks[index], .start = index, .end = index + 1U, .page = 0});
        }
        out += blocks[index];
    }
}

// The page-image budget, spent in request order (documents.h).
class PageImageBudget {
public:
    explicit PageImageBudget(const DocumentResolveOptions& options) : vision_(options.vision) {
        if (vision_.prompt_tokens == 0 || vision_.token_edge == 0) { return; }
        token_pixels_ = std::uint64_t{vision_.token_edge} * vision_.token_edge;
        min_tokens_   = std::max<std::uint64_t>(1, (vision_.min_pixels + token_pixels_ - 1U) / token_pixels_);
        remaining_    = vision_.prompt_tokens;
        if (options.max_context != 0) { remaining_ = std::min(remaining_, options.max_context / 4U); }
    }

    // An image or video before the next PDF: its typical share.
    void other_media() noexcept { remaining_ -= std::min(remaining_, kPdfPageVisionTokens); }

    // Each page's render size (0 x 0: none), spending the budget.
    std::vector<media::pdf::RenderSize> allocate(const std::vector<media::pdf::PageText>& pages) {
        std::vector<media::pdf::RenderSize> sizes(pages.size());
        if (token_pixels_ == 0 || pages.empty()) { return sizes; }
        std::vector<std::size_t> chosen(pages.size());
        for (std::size_t page = 0; page < pages.size(); ++page) { chosen[page] = page; }
        std::uint64_t tokens = std::min(kPdfPageVisionTokens, remaining_ / pages.size());
        if (tokens < min_tokens_) {
            // Too little for every page: the pages without a text layer first, the others none.
            chosen.erase(std::remove_if(chosen.begin(), chosen.end(),
                                        [&](std::size_t page) { return !is_blank(pages[page].text); }),
                         chosen.end());
            if (chosen.empty()) { return sizes; }
            tokens = std::min(kPdfPageVisionTokens, remaining_ / chosen.size());
            if (tokens < min_tokens_) {
                chosen.resize(std::min<std::size_t>(chosen.size(), remaining_ / min_tokens_));
                tokens = min_tokens_;
            }
        }
        if (vision_.max_pixels != 0) { tokens = std::min(tokens, vision_.max_pixels / token_pixels_); }
        if (tokens < min_tokens_) { return sizes; }
        const media::pdf::RenderPolicy policy{.max_pixels       = tokens * token_pixels_,
                                              .min_pixels       = vision_.min_pixels,
                                              .edge_multiple    = vision_.token_edge,
                                              .max_points_scale = 2.0};
        for (const std::size_t page : chosen) {
            const media::pdf::RenderSize size =
                media::pdf::render_size(pages[page].width_points, pages[page].height_points, policy);
            const std::uint64_t used =
                (static_cast<std::uint64_t>(size.width) * static_cast<std::uint64_t>(size.height)) / token_pixels_;
            if (size.width == 0 || used > remaining_) { continue; }
            sizes[page] = size;
            remaining_ -= used;
        }
        return sizes;
    }

private:
    VisionImageBudget vision_;
    std::uint64_t token_pixels_ = 0;
    std::uint64_t min_tokens_   = 0;
    std::uint64_t remaining_    = 0;
};

} // namespace

PdfCache::Entry* PdfCache::find(std::uint64_t hash, std::span<const std::uint8_t> bytes) {
    for (auto entry = entries_.begin(); entry != entries_.end(); ++entry) {
        if (entry->hash == hash && entry->bytes.size() == bytes.size() &&
            std::equal(bytes.begin(), bytes.end(), entry->bytes.begin())) {
            entries_.splice(entries_.begin(), entries_, entry);
            return &entries_.front();
        }
    }
    return nullptr;
}

void PdfCache::trim() {
    while (bytes_ > capacity_bytes_ && !entries_.empty()) {
        bytes_ -= entries_.back().cost;
        entries_.pop_back();
    }
}

std::shared_ptr<const std::vector<media::pdf::PageText>>
PdfCache::text(std::span<const std::uint8_t> bytes, const std::function<void()>& checkpoint) {
    const std::uint64_t hash = hash_bytes(bytes);
    {
        const std::lock_guard lock(mutex_);
        if (const Entry* entry = find(hash, bytes)) { return entry->text; }
    }
    auto text = std::make_shared<const std::vector<media::pdf::PageText>>(
        media::pdf::read_text(bytes, kPdfMaximumPages, checkpoint));
    std::size_t cost = bytes.size();
    for (const media::pdf::PageText& page : *text) { cost += page.text.size() + sizeof(page); }
    if (cost > capacity_bytes_ / 2U) { return text; }
    const std::lock_guard lock(mutex_);
    if (const Entry* entry = find(hash, bytes)) { return entry->text; } // another request read it meanwhile
    entries_.push_front(Entry{.hash   = hash,
                              .bytes  = std::vector<std::uint8_t>(bytes.begin(), bytes.end()),
                              .text   = text,
                              .images = {},
                              .cost   = cost});
    bytes_ += cost;
    trim();
    return text;
}

std::vector<PdfPageImage> PdfCache::images(std::span<const std::uint8_t> bytes,
                                           std::span<const media::pdf::RenderRequest> pages,
                                           const std::function<void()>& checkpoint) {
    const std::uint64_t hash = hash_bytes(bytes);
    std::vector<PdfPageImage> out(pages.size());
    std::vector<media::pdf::RenderRequest> missing;
    std::vector<std::size_t> missing_positions;
    {
        const std::lock_guard lock(mutex_);
        const Entry* entry = find(hash, bytes);
        for (std::size_t position = 0; position < pages.size(); ++position) {
            if (entry != nullptr) {
                const auto cached = std::find_if(entry->images.begin(), entry->images.end(), [&](const auto& image) {
                    return same_request(image.first, pages[position]);
                });
                if (cached != entry->images.end()) {
                    out[position] = cached->second;
                    continue;
                }
            }
            missing.push_back(pages[position]);
            missing_positions.push_back(position);
        }
    }
    if (missing.empty()) { return out; }
    std::vector<media::pdf::RenderedPage> rendered = media::pdf::render(bytes, missing, checkpoint);
    std::size_t added = 0;
    for (std::size_t index = 0; index < rendered.size(); ++index) {
        media::pdf::RenderedPage& page = rendered[index];
        auto image = std::make_shared<const std::vector<std::uint8_t>>(
            media::decode::encode_raw_rgb(page.width, page.height, page.rgb));
        added += image->size();
        out[missing_positions[index]] = std::move(image);
    }
    const std::lock_guard lock(mutex_);
    Entry* entry = find(hash, bytes);
    if (entry == nullptr || entry->cost + added > capacity_bytes_ / 2U) { return out; }
    for (std::size_t index = 0; index < missing.size(); ++index) {
        entry->images.emplace_back(missing[index], out[missing_positions[index]]);
    }
    entry->cost += added;
    bytes_ += added;
    trim();
    return out;
}

bool has_documents(const GenerationRequest& request) noexcept {
    for (const ChatTurn& turn : request.messages) {
        for (const ContentPart& part : turn.content) {
            if (part.kind == ContentKind::Document || !part.citations.empty()) { return true; }
        }
    }
    return false;
}

DocumentResolution resolve_documents(const GenerationRequest& request, const DocumentResolveOptions& options) {
    DocumentResolution resolution;
    if (!has_documents(request)) { return resolution; }
    PdfCache uncached(0);
    PdfCache& cache = options.cache != nullptr ? *options.cache : uncached;

    // The PDFs in request order: their bytes and text, then their page images from the budget as it stands there.
    std::unordered_map<const DocumentSource*, PdfInput> pdfs;
    PageImageBudget budget(options);
    std::uint64_t total_pages = 0;
    for (const ChatTurn& turn : request.messages) {
        for (const ContentPart& part : turn.content) {
            if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { budget.other_media(); }
            if (part.kind != ContentKind::Document || part.document->format != DocumentSource::Format::Pdf ||
                pdfs.contains(part.document.get())) {
                continue;
            }
            if (!media::pdf::available()) {
                document_error("PDF documents need PDFium, and this build of Infernix has none",
                               "documents_not_supported");
            }
            PdfInput input;
            input.bytes = options.acquire(part.document->pdf);
            try {
                input.text = cache.text(input.bytes, options.checkpoint);
                total_pages += input.text->size();
                if (total_pages > kPdfMaximumPages) {
                    document_error("the request's PDFs hold more than " + std::to_string(kPdfMaximumPages) +
                                       " pages together (" + std::to_string(total_pages) + " so far)",
                                   "pdf_too_many_pages");
                }
                const std::vector<media::pdf::RenderSize> sizes = budget.allocate(*input.text);
                std::vector<media::pdf::RenderRequest> requests;
                for (std::size_t page = 0; page < sizes.size(); ++page) {
                    if (sizes[page].width != 0) {
                        requests.push_back(media::pdf::RenderRequest{.number = (*input.text)[page].number,
                                                                     .size   = sizes[page]});
                    }
                }
                const std::vector<PdfPageImage> images = cache.images(input.bytes, requests, options.checkpoint);
                input.images.resize(sizes.size());
                for (std::size_t index = 0; index < requests.size(); ++index) {
                    input.images[requests[index].number - 1U] = images[index];
                }
            } catch (const media::pdf::Error& error) { pdf_error(error); }
            const bool has_text = std::any_of(input.text->begin(), input.text->end(),
                                              [](const media::pdf::PageText& page) { return !is_blank(page.text); });
            const bool has_images = std::any_of(input.images.begin(), input.images.end(),
                                                [](const PdfPageImage& image) { return image != nullptr; });
            if (!has_text && !has_images) {
                document_error(options.vision.prompt_tokens == 0
                                   ? "the PDF has no text layer (a scanned document?), and this server has no vision "
                                     "input to read its pages as images (start it with --vision)"
                                   : "the PDF has no text layer (a scanned document?), and no room is left in the "
                                     "request's vision budget for its pages as images",
                               "pdf_no_text");
            }
            pdfs.emplace(part.document.get(), std::move(input));
        }
    }

    auto registry = std::make_shared<CitationRegistry>();
    GenerationRequest out = request;
    std::size_t document_index = 0;
    std::size_t search_index   = 0;
    bool instructed            = false;
    for (ChatTurn& turn : out.messages) {
        std::vector<ContentPart> parts;
        parts.reserve(turn.content.size());
        for (ContentPart& part : turn.content) {
            if (part.kind != ContentKind::Document) {
                if (!part.citations.empty()) {
                    // A cited block of the history renders as the tag the model wrote.
                    std::vector<CitationId> ids;
                    for (const CitationLocation& location : part.citations) {
                        for (const CitationId id : registry->ids_for(location)) {
                            if (std::find(ids.begin(), ids.end(), id) == ids.end()) { ids.push_back(id); }
                        }
                    }
                    if (!ids.empty()) { part.text = cite_open_tag(ids) + part.text + std::string(kCiteCloseTag); }
                    part.citations.clear();
                }
                parts.push_back(std::move(part));
                continue;
            }
            const DocumentSource& document = *part.document;
            const bool search_result       = document.kind == DocumentSource::Kind::SearchResult;
            const std::size_t index        = search_result ? search_index++ : document_index++;
            const std::string title =
                document.format == DocumentSource::Format::Pdf && document.title.empty() ? document.filename
                                                                                          : document.title;
            std::optional<CitableSource> citable;
            if (document.citations) {
                citable.emplace();
                citable->kind   = document.kind;
                citable->format = document.format;
                citable->index  = index;
                if (search_result || !title.empty()) { citable->title = title; }
                citable->source = document.source;
            }
            CitableSource* chunks          = citable ? &*citable : nullptr;
            const std::size_t position     = registry->sources().size();
            const std::size_t first_part   = parts.size();
            PartWriter writer{parts, {}};
            append_header(writer.text, document, title);
            switch (document.format) {
            case DocumentSource::Format::Text:
                append_sentences(writer.text, document.text, chunks, position, 0);
                writer.text += "\n</document>\n";
                break;
            case DocumentSource::Format::Content:
                append_blocks(writer.text, document.blocks, chunks, position);
                writer.text += search_result ? "\n</search_result>\n" : "\n</document>\n";
                break;
            case DocumentSource::Format::Pdf: {
                const PdfInput& pdf = pdfs.at(&document);
                for (std::size_t page = 0; page < pdf.text->size(); ++page) {
                    const media::pdf::PageText& text = (*pdf.text)[page];
                    writer.text += "<page number=\"" + std::to_string(text.number) + "\">\n";
                    if (pdf.images[page] != nullptr) {
                        writer.image(*pdf.images[page]);
                    } else if (is_blank(text.text)) {
                        // The model would otherwise see an empty page and not know it went unread.
                        writer.text += options.vision.prompt_tokens == 0 ? kPdfPageUnreadWithoutVision
                                                                         : kPdfPageUnreadOverBudget;
                    }
                    append_sentences(writer.text, text.text, chunks, position, text.number);
                    if (!writer.text.empty() && writer.text.back() != '\n') { writer.text += '\n'; }
                    writer.text += "</page>\n";
                }
                writer.text += "</document>\n";
                break;
            }
            }
            if (citable) {
                registry->add(std::move(*citable));
                if (!instructed) {
                    writer.text += citation_instruction() + "\n";
                    instructed = true;
                }
            }
            writer.flush();
            if (part.cache_boundary_after && parts.size() > first_part) {
                parts.back().cache_boundary_after = part.cache_boundary_after;
            }
        }
        turn.content = std::move(parts);
    }
    resolution.request = std::move(out);
    if (!registry->empty()) { resolution.citations = std::move(registry); }
    return resolution;
}

} // namespace infernix::serve
