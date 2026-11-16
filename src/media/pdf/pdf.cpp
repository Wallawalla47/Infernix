#include "media/pdf/pdf.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>

#ifdef INFERNIX_HAVE_PDFIUM
#include <fpdf_text.h>
#include <fpdfview.h>
#endif

namespace infernix::media::pdf {

RenderSize render_size(double width_points, double height_points, const RenderPolicy& policy) noexcept {
    if (policy.max_pixels == 0 || !(width_points > 0.0) || !(height_points > 0.0)) { return {}; }
    // The Vision processor refuses an aspect ratio past 200, and rounding to the edge multiple widens a narrow
    // page's: one past 100 stays text only.
    if (std::max(width_points, height_points) / std::min(width_points, height_points) > 100.0) { return {}; }
    const double multiple = std::max<std::uint32_t>(policy.edge_multiple, 1U);
    const double area     = width_points * height_points;
    double scale          = std::min(std::sqrt(static_cast<double>(policy.max_pixels) / area), policy.max_points_scale);
    if (policy.min_pixels != 0) { scale = std::max(scale, std::sqrt(static_cast<double>(policy.min_pixels) / area)); }
    const auto fit = [&](double points) {
        return std::max(multiple, std::floor(points * scale / multiple) * multiple);
    };
    const auto max_pixels = static_cast<double>(policy.max_pixels);
    const auto min_pixels = static_cast<double>(policy.min_pixels);
    double width          = fit(width_points);
    double height         = fit(height_points);
    if (width * height < min_pixels) {
        // Rounding down left the page below the minimum (a page budget near it): the grid size within the bounds
        // whose aspect ratio is nearest the page's, the larger on a tie.
        const double aspect = width_points / height_points;
        double best         = -1.0;
        for (double w = multiple; w * multiple <= max_pixels; w += multiple) {
            const double h = std::floor(max_pixels / w / multiple) * multiple;
            if (h < multiple || w * h < min_pixels) { continue; }
            const double error = std::abs(std::log((w / h) / aspect));
            if (best < 0.0 || error < best - 1e-12 || (std::abs(error - best) <= 1e-12 && w * h > width * height)) {
                best   = error;
                width  = w;
                height = h;
            }
        }
        if (best < 0.0) { return {}; }
    }
    // Outside the bounds (the processor would resize it, and the budget would count the wrong size): no image.
    if (width * height > max_pixels || width * height < min_pixels) { return {}; }
    return RenderSize{.width = static_cast<int>(width), .height = static_cast<int>(height)};
}

#ifdef INFERNIX_HAVE_PDFIUM
namespace {

// PDFium allows one call at a time from any thread. Its lock is taken per page and served in arrival order, so a
// long document's pages interleave with other requests' instead of holding them for the whole document, and each
// request's checkpoint (deadline, cancellation) runs between its pages, unlocked.
class LibraryLock {
public:
    void lock() {
        std::unique_lock guard(mutex_);
        const std::uint64_t ticket = next_++;
        turn_.wait(guard, [&] { return serving_ == ticket; });
    }
    void unlock() {
        {
            const std::lock_guard guard(mutex_);
            ++serving_;
        }
        turn_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable turn_;
    std::uint64_t next_    = 0;
    std::uint64_t serving_ = 0;
};

LibraryLock& library_mutex() {
    static LibraryLock lock;
    return lock;
}

// Runs the checkpoint between pages with the library unlocked; the lock is held again when it returns or throws, so
// the document handles still close under it.
void between_pages(std::unique_lock<LibraryLock>& lock, const std::function<void()>& checkpoint) {
    lock.unlock();
    try {
        if (checkpoint) { checkpoint(); }
    } catch (...) {
        lock.lock();
        throw;
    }
    lock.lock();
}

// Under library_mutex(). PDFium stays initialized for the process's life.
void initialize_library() {
    static bool initialized = false;
    if (initialized) { return; }
    FPDF_LIBRARY_CONFIG config{};
    config.version = 2;
    FPDF_InitLibraryWithConfig(&config);
    initialized = true;
}

struct DocumentHandle {
    FPDF_DOCUMENT document = nullptr;
    DocumentHandle()                                 = default;
    DocumentHandle(const DocumentHandle&)            = delete;
    DocumentHandle& operator=(const DocumentHandle&) = delete;
    ~DocumentHandle() {
        if (document != nullptr) { FPDF_CloseDocument(document); }
    }
};

struct PageHandle {
    FPDF_PAGE page = nullptr;
    explicit PageHandle(FPDF_PAGE value) : page(value) {}
    PageHandle(const PageHandle&)            = delete;
    PageHandle& operator=(const PageHandle&) = delete;
    ~PageHandle() {
        if (page != nullptr) { FPDF_ClosePage(page); }
    }
};

struct TextPageHandle {
    FPDF_TEXTPAGE page = nullptr;
    explicit TextPageHandle(FPDF_TEXTPAGE value) : page(value) {}
    TextPageHandle(const TextPageHandle&)            = delete;
    TextPageHandle& operator=(const TextPageHandle&) = delete;
    ~TextPageHandle() {
        if (page != nullptr) { FPDFText_ClosePage(page); }
    }
};

struct BitmapHandle {
    FPDF_BITMAP bitmap = nullptr;
    explicit BitmapHandle(FPDF_BITMAP value) : bitmap(value) {}
    BitmapHandle(const BitmapHandle&)            = delete;
    BitmapHandle& operator=(const BitmapHandle&) = delete;
    ~BitmapHandle() {
        if (bitmap != nullptr) { FPDFBitmap_Destroy(bitmap); }
    }
};

// Under library_mutex().
void open(DocumentHandle& handle, std::span<const std::uint8_t> bytes) {
    initialize_library();
    if (bytes.empty()) { throw Error(ErrorKind::Invalid, "the PDF is empty"); }
    handle.document = FPDF_LoadMemDocument64(bytes.data(), bytes.size(), nullptr);
    if (handle.document != nullptr) { return; }
    switch (FPDF_GetLastError()) {
    case FPDF_ERR_PASSWORD:
        throw Error(ErrorKind::Encrypted, "the PDF is password-protected");
    case FPDF_ERR_SECURITY:
        throw Error(ErrorKind::Encrypted, "the PDF uses an unsupported security handler");
    default:
        throw Error(ErrorKind::Invalid, "the file is not a PDF that can be read");
    }
}

void append_utf8(std::string& out, char32_t point) {
    if (point < 0x80) {
        out += static_cast<char>(point);
    } else if (point < 0x800) {
        out += static_cast<char>(0xC0 | (point >> 6));
        out += static_cast<char>(0x80 | (point & 0x3F));
    } else if (point < 0x10000) {
        out += static_cast<char>(0xE0 | (point >> 12));
        out += static_cast<char>(0x80 | ((point >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (point & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (point >> 18));
        out += static_cast<char>(0x80 | ((point >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((point >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (point & 0x3F));
    }
}

// PDFium's UTF-16 text as UTF-8: CR LF and lone CR become LF, control characters other than LF and tab, and
// noncharacters, are dropped, and an unpaired surrogate becomes U+FFFD.
std::string page_text(FPDF_PAGE page) {
    const TextPageHandle text(FPDFText_LoadPage(page));
    if (text.page == nullptr) { return {}; }
    const int count = FPDFText_CountChars(text.page);
    if (count <= 0) { return {}; }
    std::vector<unsigned short> units(static_cast<std::size_t>(count) + 1U);
    const int written = FPDFText_GetText(text.page, 0, count, units.data());
    const std::size_t length = written > 0 ? static_cast<std::size_t>(written) - 1U : 0U; // without the NUL
    std::string out;
    out.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        char32_t point = units[i];
        if (point >= 0xD800 && point <= 0xDBFF && i + 1U < length && units[i + 1U] >= 0xDC00 && units[i + 1U] <= 0xDFFF) {
            point = 0x10000 + ((point - 0xD800) << 10) + (units[i + 1U] - 0xDC00);
            ++i;
        } else if (point >= 0xD800 && point <= 0xDFFF) {
            point = 0xFFFD;
        }
        if (point == '\r') {
            if (i + 1U < length && units[i + 1U] == '\n') { continue; }
            point = '\n';
        }
        if ((point < 0x20 && point != '\n' && point != '\t') || point == 0x7F || point == 0xFEFF || point == 0xFFFE ||
            point == 0xFFFF) {
            continue;
        }
        append_utf8(out, point);
    }
    return out;
}

void render_page(FPDF_PAGE page, const RenderSize& size, RenderedPage& out) {
    const BitmapHandle bitmap(FPDFBitmap_Create(size.width, size.height, 0));
    if (bitmap.bitmap == nullptr) { throw Error(ErrorKind::Invalid, "a PDF page could not be rendered"); }
    FPDFBitmap_FillRect(bitmap.bitmap, 0, 0, size.width, size.height, 0xFFFFFFFFU);
    FPDF_RenderPageBitmap(bitmap.bitmap, page, 0, 0, size.width, size.height, 0, FPDF_ANNOT);
    const auto* pixels = static_cast<const std::uint8_t*>(FPDFBitmap_GetBuffer(bitmap.bitmap));
    const int stride   = FPDFBitmap_GetStride(bitmap.bitmap);
    out.width          = size.width;
    out.height         = size.height;
    out.rgb.resize(static_cast<std::size_t>(size.width) * static_cast<std::size_t>(size.height) * 3U);
    std::uint8_t* target = out.rgb.data();
    for (int y = 0; y < size.height; ++y) {
        const std::uint8_t* row = pixels + static_cast<std::ptrdiff_t>(y) * stride;
        for (int x = 0; x < size.width; ++x) {
            // BGRx
            target[0] = row[4 * x + 2];
            target[1] = row[4 * x + 1];
            target[2] = row[4 * x + 0];
            target += 3;
        }
    }
}

int checked_page_count(const DocumentHandle& document, std::uint32_t max_pages) {
    const int count = FPDF_GetPageCount(document.document);
    if (count < 0 || static_cast<std::uint32_t>(count) > max_pages) {
        throw Error(ErrorKind::TooManyPages, "the PDF has " + std::to_string(count) + " pages, more than the " +
                                                 std::to_string(max_pages) + " a document may hold");
    }
    return count;
}

} // namespace

bool available() noexcept { return true; }

std::vector<PageText> read_text(std::span<const std::uint8_t> bytes, std::uint32_t max_pages,
                                const std::function<void()>& checkpoint) {
    std::unique_lock lock(library_mutex());
    DocumentHandle document;
    open(document, bytes);
    const int count = checked_page_count(document, max_pages);
    std::vector<PageText> pages(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        between_pages(lock, checkpoint);
        const PageHandle handle(FPDF_LoadPage(document.document, index));
        if (handle.page == nullptr) {
            throw Error(ErrorKind::Invalid, "page " + std::to_string(index + 1) + " of the PDF could not be read");
        }
        PageText& page     = pages[static_cast<std::size_t>(index)];
        page.number        = static_cast<std::uint32_t>(index) + 1U;
        page.text          = page_text(handle.page);
        page.width_points  = FPDF_GetPageWidthF(handle.page);
        page.height_points = FPDF_GetPageHeightF(handle.page);
    }
    return pages;
}

std::vector<RenderedPage> render(std::span<const std::uint8_t> bytes, std::span<const RenderRequest> pages,
                                 const std::function<void()>& checkpoint) {
    std::vector<RenderedPage> out;
    if (pages.empty()) { return out; }
    std::unique_lock lock(library_mutex());
    DocumentHandle document;
    open(document, bytes);
    const int count = FPDF_GetPageCount(document.document);
    out.reserve(pages.size());
    for (const RenderRequest& request : pages) {
        between_pages(lock, checkpoint);
        if (request.number == 0 || request.number > static_cast<std::uint32_t>(std::max(count, 0)) ||
            request.size.width <= 0 || request.size.height <= 0) {
            throw std::invalid_argument("PDF render request names no page of the document");
        }
        const PageHandle handle(FPDF_LoadPage(document.document, static_cast<int>(request.number) - 1));
        if (handle.page == nullptr) {
            throw Error(ErrorKind::Invalid, "page " + std::to_string(request.number) + " of the PDF could not be read");
        }
        RenderedPage page;
        page.number = request.number;
        render_page(handle.page, request.size, page);
        out.push_back(std::move(page));
    }
    return out;
}

#else

bool available() noexcept { return false; }

std::vector<PageText> read_text(std::span<const std::uint8_t>, std::uint32_t, const std::function<void()>&) {
    throw Error(ErrorKind::Unavailable, "this build of Infernix has no PDF reader (built without PDFium)");
}

std::vector<RenderedPage> render(std::span<const std::uint8_t>, std::span<const RenderRequest>,
                                 const std::function<void()>&) {
    throw Error(ErrorKind::Unavailable, "this build of Infernix has no PDF reader (built without PDFium)");
}

#endif

} // namespace infernix::media::pdf
