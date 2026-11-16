#pragma once

// Resolution of a request's documents (ContentKind::Document parts) into the text and page images the model reads,
// and of its history's cited text blocks into the cite tags the model wrote. GenerationService runs it before
// translation, for generation and Count Tokens alike, so both see the same prompt.
//
// The prompt form depends on the request's own content alone, so a conversation that sends the same documents again
// renders the same tokens and keeps its prefix cache:
// - a text or custom-content document: `<document>` with its title and context, then its text;
// - a search result: `<search_result>` with its source and title, then its content blocks;
// - a PDF: `<document>` with its title and context, then per page `<page number="N">`, the page's image (with
//   vision) and its text;
// - with citations, a chunk's ID before each chunk (citations.h), and the instruction once, right after the
//   request's first citable source.
//
// Page images (with vision) spend a budget in request order, so a page's size depends only on what precedes it:
// min(the prompt's vision tokens, a quarter of the context), less kPdfPageVisionTokens for each image or video
// before it. A PDF's pages each get min(kPdfPageVisionTokens, what remains / its pages); when that falls below the
// processor's smallest image, its pages without a text layer get the remainder first and the others none. A page
// left with neither text nor an image says so in a fixed note, so the model knows the page is there unread.
//
// What bounds a PDF is the context: its pages' text (and IDs, and images) must fit the prompt, which the Engine
// checks exactly. Bytes are bounded by the request body and media budgets. kPdfMaximumPages bounds only the PDFium
// work a request can ask for before that check (about 1 ms a page for text).

#include "infernix/types.h"
#include "media/pdf/pdf.h"
#include "product/media_acquire/source.h"
#include "serve/citations.h"
#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace infernix::serve {

// A PDF page's image tokens at most (about a million pixels: 10-point text stays legible).
inline constexpr std::uint64_t kPdfPageVisionTokens = 1'024;
// The pages a request's PDFs may hold together.
inline constexpr std::uint32_t kPdfMaximumPages = 1'000;
// The server's cache of read PDFs (their bytes, page text and page images).
inline constexpr std::size_t kPdfCacheBytes = 256ULL << 20;
// The media type of a rendered page image (media::decode's raw RGB container).
inline constexpr std::string_view kPdfPageImageMediaType = "image/x-infernix-rgb";
// The note a page with neither text nor an image carries in the prompt.
inline constexpr std::string_view kPdfPageUnreadWithoutVision =
    "[This page has no text layer, and this server has no vision input to read it as an image.]\n";
inline constexpr std::string_view kPdfPageUnreadOverBudget =
    "[This page has no text layer, and the request's vision budget left no room for its image.]\n";

using PdfPageImage = std::shared_ptr<const std::vector<std::uint8_t>>;

// Recently read PDFs by their bytes: each one's page text, and its page images by page and size, so an agent
// conversation that sends the same PDF on every turn (and a Count Tokens before Messages) reads and renders it
// once. Thread-safe; least recently used PDFs leave first, and one larger than half the capacity is not kept.
class PdfCache {
public:
    explicit PdfCache(std::size_t capacity_bytes) : capacity_bytes_(capacity_bytes) {}

    // Every page's text and size. Throws media::pdf::Error.
    [[nodiscard]] std::shared_ptr<const std::vector<media::pdf::PageText>>
    text(std::span<const std::uint8_t> bytes, const std::function<void()>& checkpoint = {});

    // Each requested page's image as encoded media, rendering the ones not cached in one pass. Throws
    // media::pdf::Error.
    [[nodiscard]] std::vector<PdfPageImage> images(std::span<const std::uint8_t> bytes,
                                                   std::span<const media::pdf::RenderRequest> pages,
                                                   const std::function<void()>& checkpoint = {});

private:
    struct Entry {
        std::uint64_t hash = 0;
        std::vector<std::uint8_t> bytes;
        std::shared_ptr<const std::vector<media::pdf::PageText>> text;
        std::vector<std::pair<media::pdf::RenderRequest, PdfPageImage>> images;
        std::size_t cost = 0;
    };
    // Under mutex_: the entry for these bytes, made the most recent; nullptr when none.
    [[nodiscard]] Entry* find(std::uint64_t hash, std::span<const std::uint8_t> bytes);
    void trim(); // under mutex_

    std::size_t capacity_bytes_;
    std::mutex mutex_;
    std::list<Entry> entries_; // most recent first
    std::size_t bytes_ = 0;
};

struct DocumentResolveOptions {
    // The Engine's image sizing; prompt_tokens == 0 (no vision) reads PDFs as text only.
    infernix::VisionImageBudget vision;
    // The Engine's context, which bounds the page images.
    std::uint64_t max_context = 0;
    // A PDF source's bytes (inline data or a URL), under the request's media budget, deadline and cancellation.
    // Throws ApiException.
    std::function<std::vector<std::uint8_t>(const infernix::product::media_acquire::Source&)> acquire;
    // Called between PDF pages: throws to stop the request.
    std::function<void()> checkpoint;
    PdfCache* cache = nullptr; // optional
};

struct DocumentResolution {
    // The request with every document resolved and every cited history block rendered as its cite tag; nullopt
    // when it holds neither (the request is used as it is).
    std::optional<GenerationRequest> request;
    // The request's citable sources, or null when it has none.
    std::shared_ptr<const CitationRegistry> citations;
};

// Throws ApiException for a PDF that cannot be read (invalid, encrypted, too many pages, a build without PDFium),
// and for a PDF with neither a text layer nor room for its page images.
[[nodiscard]] DocumentResolution resolve_documents(const GenerationRequest& request,
                                                   const DocumentResolveOptions& options);

// Whether the request holds a document part or a cited history block, so resolve_documents has work to do.
[[nodiscard]] bool has_documents(const GenerationRequest& request) noexcept;

} // namespace infernix::serve
