#pragma once

// PDF documents through PDFium (bblanchon/pdfium-binaries, the build without V8 and XFA): each page's text, and
// the pages a caller asks for rendered. PDFium's state is process-wide and not thread-safe, so its calls hold one
// process-wide lock, taken per page and served in arrival order (concurrent documents interleave by page, and a
// caller's checkpoint runs between pages, unlocked); a build without PDFium (INFERNIX_HAVE_PDFIUM unset) refuses
// every document.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::media::pdf {

enum class ErrorKind : std::uint8_t {
    Unavailable,  // this build has no PDFium
    Invalid,      // not a PDF PDFium can open
    Encrypted,    // needs a password, or an unsupported security handler
    TooManyPages, // more pages than the caller allows
};

class Error final : public std::runtime_error {
public:
    Error(ErrorKind kind, std::string message) : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }

private:
    ErrorKind kind_;
};

// Whether this build reads PDFs.
[[nodiscard]] bool available() noexcept;

struct PageText {
    std::uint32_t number = 0; // 1-based
    std::string text;         // UTF-8; line breaks as "\n", control characters removed
    double width_points  = 0.0;
    double height_points = 0.0;
};

// Every page's text and size. Throws Error; `checkpoint` runs between pages and throws to stop.
[[nodiscard]] std::vector<PageText> read_text(std::span<const std::uint8_t> bytes, std::uint32_t max_pages,
                                              const std::function<void()>& checkpoint = {});

struct RenderPolicy {
    // A page renders as the largest image of its aspect ratio with both sides multiples of `edge_multiple` and at
    // most `max_pixels` pixels, at no more than `max_points_scale` pixels per PDF point (a small page is not blown
    // up), and at least `min_pixels` pixels.
    std::uint64_t max_pixels    = 0;
    std::uint64_t min_pixels    = 0;
    std::uint32_t edge_multiple = 1;
    double max_points_scale     = 2.0;
};

struct RenderSize {
    int width  = 0;
    int height = 0;

    [[nodiscard]] friend bool operator==(RenderSize, RenderSize) noexcept = default;
};

// The size a page of `width_points` x `height_points` renders at under `policy`; 0 x 0 when it renders none (no
// pixels allowed, an aspect ratio past 100 that the Vision processor would refuse after rounding, or no size on
// the grid within [min_pixels, max_pixels]).
[[nodiscard]] RenderSize render_size(double width_points, double height_points, const RenderPolicy& policy) noexcept;

struct RenderRequest {
    std::uint32_t number = 0; // 1-based
    RenderSize size;
};

struct RenderedPage {
    std::uint32_t number = 0;
    int width            = 0;
    int height           = 0;
    std::vector<std::uint8_t> rgb; // width * height * 3, row-major, on white
};

// The requested pages rendered, in request order, in one pass over the document. Throws Error.
[[nodiscard]] std::vector<RenderedPage> render(std::span<const std::uint8_t> bytes,
                                               std::span<const RenderRequest> pages,
                                               const std::function<void()>& checkpoint = {});

} // namespace infernix::media::pdf
