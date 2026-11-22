#include "media/decode/png_decode.h"

extern "C" {
#include <zlib.h>
}

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::media::decode {
namespace {

constexpr std::array<std::uint8_t, 8> kSignature = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};

[[nodiscard]] std::uint32_t be32(const std::uint8_t* p) noexcept {
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
}

struct Header {
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    int color_type = 0;
};

struct Palette {
    std::vector<std::array<std::uint8_t, 3>> rgb;
};

// tRNS: a palette's per-entry alpha (entries past it opaque), or the one transparent gray or RGB sample value.
struct Transparency {
    bool present = false;
    std::vector<std::uint8_t> palette_alpha;
    std::array<std::uint32_t, 3> key{};
};

struct Parse {
    Header header;
    std::vector<std::uint8_t> idat;
    Palette palette;
    Transparency transparency;
    bool have_ihdr = false;
};

// Channels per pixel for each color type (0 for an unsupported type).
[[nodiscard]] int samples_per_pixel(int color_type) {
    switch (color_type) {
        case 0: return 1; // gray
        case 2: return 3; // rgb
        case 3: return 1; // palette
        case 4: return 2; // gray + alpha
        case 6: return 4; // rgba
        default: return 0;
    }
}

// Walk the chunk stream, collecting IHDR, PLTE, tRNS and the concatenated IDAT. The decode reads transparency (tRNS,
// and the alpha samples of color types 4 and 6) only when asked to keep alpha.
void parse_chunks(std::span<const std::uint8_t> bytes, Parse& out) {
    std::size_t offset = 8;
    while (offset + 8 <= bytes.size()) {
        const std::uint32_t length = be32(bytes.data() + offset);
        offset += 4;
        if (offset + 4 + static_cast<std::size_t>(length) + 4 > bytes.size()) {
            throw std::invalid_argument("png chunk extends past end of stream");
        }
        const std::uint8_t* tag  = bytes.data() + offset;
        offset                  += 4;
        const std::uint8_t* data = bytes.data() + offset;
        offset += length + 4; // payload + CRC
        if (std::memcmp(tag, "IHDR", 4) == 0) {
            if (out.have_ihdr || length < 13) {
                throw std::invalid_argument("png has a malformed IHDR chunk");
            }
            out.have_ihdr         = true;
            out.header.width      = static_cast<int>(be32(data));
            out.header.height     = static_cast<int>(be32(data + 4));
            out.header.bit_depth  = data[8];
            out.header.color_type = data[9];
            // compression=0, filter=0, interlace=0 are the only supported methods.
            if (data[10] != 0 || data[11] != 0 || data[12] != 0) {
                throw std::invalid_argument(
                    "png uses an unsupported compression, filter, or interlace method");
            }
            if (out.header.width <= 0 || out.header.height <= 0) {
                throw std::invalid_argument("png dimensions are invalid");
            }
            // The legal (colour type, bit depth) pairs (PNG specification, 11.2.2): every other depth would
            // reach the sample reader's shifts and to8's divisor outside their domain.
            const int depth   = out.header.bit_depth;
            const bool legal  = [&] {
                switch (out.header.color_type) {
                case 0: return depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16;
                case 3: return depth == 1 || depth == 2 || depth == 4 || depth == 8;
                case 2:
                case 4:
                case 6: return depth == 8 || depth == 16;
                default: return false;
                }
            }();
            if (!legal) {
                throw std::invalid_argument("png has an invalid colour type and bit depth (" +
                                            std::to_string(out.header.color_type) + ", " + std::to_string(depth) +
                                            ")");
            }
        } else if (std::memcmp(tag, "PLTE", 4) == 0) {
            if (length % 3 != 0 || length == 0) {
                throw std::invalid_argument("png has a malformed PLTE chunk");
            }
            const std::size_t entries = length / 3;
            out.palette.rgb.resize(entries);
            for (std::size_t i = 0; i < entries; ++i) {
                out.palette.rgb[i] = {data[3 * i], data[3 * i + 1], data[3 * i + 2]};
            }
        } else if (std::memcmp(tag, "IDAT", 4) == 0) {
            out.idat.insert(out.idat.end(), data, data + length);
        } else if (std::memcmp(tag, "tRNS", 4) == 0 && out.have_ihdr) {
            Transparency& t = out.transparency;
            const auto be16 = [&](std::size_t at) { return (std::uint32_t{data[at]} << 8) | data[at + 1]; };
            if (out.header.color_type == 3) {
                t.present = true;
                t.palette_alpha.assign(data, data + length);
            } else if (out.header.color_type == 0 && length >= 2) {
                t.present = true;
                t.key     = {be16(0), 0, 0};
            } else if (out.header.color_type == 2 && length >= 6) {
                t.present = true;
                t.key     = {be16(0), be16(2), be16(4)};
            }
        }
    }
    if (!out.have_ihdr) { throw std::invalid_argument("png is missing its IHDR chunk"); }
}

// The IDAT stream inflated a scanline at a time: the decode holds the compressed stream, two scanlines and the RGB
// output, never the whole inflated image (an 8192 x 8192 RGBA-16 image inflates to 537 MB, and its unfiltered copy as
// much again). The stream must hold exactly the image's scanlines: a shorter one, a longer one (more output after the
// last scanline) and a failing one are refused, as when it was inflated whole.
class ScanlineInflater {
public:
    explicit ScanlineInflater(std::span<const std::uint8_t> data) {
        if (data.size() > std::numeric_limits<uInt>::max()) {
            throw std::invalid_argument("png IDAT stream exceeds the zlib input limit");
        }
        if (inflateInit2(&strm_, 15 + 32) != Z_OK) {
            throw std::invalid_argument("png: failed to initialize decompressor");
        }
        strm_.next_in  = const_cast<std::uint8_t*>(data.data());
        strm_.avail_in = static_cast<uInt>(data.size());
    }
    ~ScanlineInflater() { inflateEnd(&strm_); }
    ScanlineInflater(const ScanlineInflater&)            = delete;
    ScanlineInflater& operator=(const ScanlineInflater&) = delete;

    // Fills `out` (one filter byte and a scanline; at most 4 MiB + 1 by the row-width guard).
    void read(std::uint8_t* out, std::size_t bytes) {
        strm_.next_out  = out;
        strm_.avail_out = static_cast<uInt>(bytes);
        while (strm_.avail_out > 0) {
            const int status = inflate(&strm_, Z_NO_FLUSH);
            if (status == Z_STREAM_END) {
                ended_ = true;
                if (strm_.avail_out > 0) { throw std::invalid_argument("png: IDAT stream length does not match the image"); }
                return;
            }
            if (status != Z_OK) { throw std::invalid_argument("png: IDAT stream failed to decompress"); }
        }
    }

    // After the last scanline: nothing more may come out of the stream. (A stream that is exactly the image still
    // has zlib's end marker to process, which produces no byte.)
    void finish() {
        while (!ended_) {
            std::uint8_t extra = 0;
            strm_.next_out     = &extra;
            strm_.avail_out    = 1;
            const int status   = inflate(&strm_, Z_NO_FLUSH);
            if (strm_.avail_out == 0) { throw std::invalid_argument("png: IDAT stream is longer than the image"); }
            if (status == Z_STREAM_END) { return; }
            if (status != Z_OK) { throw std::invalid_argument("png: IDAT stream failed to decompress"); }
        }
    }

private:
    z_stream strm_{};
    bool ended_ = false;
};

// Reconstructs one raw (unfiltered) scanline. The encoder stores F = raw - predictor (mod 256), so reconstruction ADDS
// the predictor back: raw = F + predictor (wrapping). Predictors are pixel-aligned (libpng applies the filter at sample
// granularity): left = recon(x-bpp), up = prev(x), up-left = prev(x-bpp), where bpp is the number of bytes per pixel
// (bytes_per_sample * samples_per_pixel); the first scanline's `prev` is zeros.
void unfilter_row(std::uint8_t filter, const std::uint8_t* filtered, const std::uint8_t* up, std::uint8_t* cur,
                  int row_bytes, int bpp) {
    for (int x = 0; x < row_bytes; ++x) {
        const std::uint8_t left   = x >= bpp ? cur[x - bpp] : 0;
        const std::uint8_t upb    = up[x];
        const std::uint8_t upleft = x >= bpp ? up[x - bpp] : 0;
        std::uint8_t v            = filtered[x];
        switch (filter) {
            case 0: break;
            case 1: v = static_cast<std::uint8_t>(v + left); break;
            case 2: v = static_cast<std::uint8_t>(v + upb); break;
            case 3: v = static_cast<std::uint8_t>(v + ((left + upb) >> 1)); break;
            case 4: {
                const int pa = std::abs(static_cast<int>(upb) - static_cast<int>(upleft));
                const int pb = std::abs(static_cast<int>(left) - static_cast<int>(upleft));
                const int pc = std::abs(static_cast<int>(left) + static_cast<int>(upb) - 2 * static_cast<int>(upleft));
                const std::uint8_t c = (pa <= pb && pa <= pc) ? left : (pb <= pc ? upb : upleft);
                v = static_cast<std::uint8_t>(v + c);
                break;
            }
            default: throw std::invalid_argument("png: unknown scanline filter type");
        }
        cur[x] = v;
    }
}

// Reduce a full-precision sample to 8 bits.
[[nodiscard]] std::uint8_t to8(std::uint32_t v, int bit_depth) {
    if (bit_depth == 16) { return static_cast<std::uint8_t>(v >> 8); }
    if (bit_depth == 8) { return static_cast<std::uint8_t>(v); }
    const int maxv = (1 << bit_depth) - 1;
    return static_cast<std::uint8_t>((static_cast<int>(v) * 255 + maxv / 2) / maxv);
}

} // namespace

bool is_png(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < kSignature.size()) { return false; }
    return std::memcmp(bytes.data(), kSignature.data(), kSignature.size()) == 0;
}

ImageInfo png_image_info(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels) {
    if (!is_png(bytes)) { throw std::invalid_argument("not a png image"); }
    Parse parse;
    parse_chunks(bytes, parse);
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(parse.header.width) * static_cast<std::uint64_t>(parse.header.height);
    if (pixels > max_decoded_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
    }
    return ImageInfo{.width = parse.header.width, .height = parse.header.height};
}

Image decode_png(std::span<const std::uint8_t> bytes, std::uint64_t max_decoded_pixels, bool keep_alpha) {
    if (!is_png(bytes)) { throw std::invalid_argument("not a png image"); }
    Parse parse;
    parse_chunks(bytes, parse);
    const int width      = parse.header.width;
    const int height     = parse.header.height;
    const int bit_depth  = parse.header.bit_depth;
    const int color_type = parse.header.color_type;
    const int samples    = samples_per_pixel(color_type);
    if (samples == 0) { throw std::invalid_argument("png has an unsupported color type"); }
    if (color_type == 3 && parse.palette.rgb.empty()) {
        throw std::invalid_argument("png palette image is missing its PLTE chunk");
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    if (pixels > max_decoded_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
    }
    const int bytes_per_sample = bit_depth >= 8 ? bit_depth / 8 : 1;
    // Row geometry feeds 32-bit `int` offsets below, so bound the row width in 64-bit first.
    // A crafted 33,554,432x2 RGBA-16 image passes the 64-bit pixel budget with exactly this
    // product at 2^31 bits and would otherwise wrap `row_bytes` negative.
    const std::uint64_t row_bits = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(samples) *
                                   static_cast<std::uint64_t>(bit_depth);
    if (row_bits == 0) { throw std::invalid_argument("png row width is invalid"); }
    if (row_bits > 32ULL * 1024ULL * 1024ULL) {
        throw Error(ErrorKind::BudgetExceeded, "png row width exceeds the decode buffer limit");
    }
    const int row_bytes = static_cast<int>((row_bits + 7) / 8);
    const int bpp       = std::max(1, samples * bytes_per_sample);

    // Full-precision sample (0 .. (1<<bit_depth)-1) of the reconstructed scanline `row` at pixel x, channel s.
    auto raw_value = [&](const std::uint8_t* row, int x, int s) -> std::uint32_t {
        if (bit_depth >= 8) {
            const std::uint8_t* base = row + static_cast<std::size_t>(x) * static_cast<std::size_t>(samples) *
                                                 bytes_per_sample +
                                       static_cast<std::size_t>(s) * bytes_per_sample;
            if (bit_depth == 16) {
                return (static_cast<std::uint32_t>(base[0]) << 8) | static_cast<std::uint32_t>(base[1]);
            }
            return base[0];
        }
        const int bitpos     = (x * samples + s) * bit_depth;
        const int byte       = bitpos >> 3;
        const int shift      = 7 - (bitpos & 7) - (bit_depth - 1);
        const std::uint8_t b = row[byte];
        return (static_cast<int>(b) >> shift) & ((1 << bit_depth) - 1);
    };

    Image out;
    out.width  = width;
    out.height = height;
    out.rgb.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    std::uint8_t* px = out.rgb.data();
    // The alpha plane (keep_alpha and a transparent source): colour types 4 and 6 their alpha sample (8 bits as the
    // colour samples), tRNS a palette entry's alpha or 0 where the full-precision sample equals the key.
    const bool alpha = keep_alpha && (color_type == 4 || color_type == 6 || parse.transparency.present);
    if (alpha) { out.alpha.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 255); }
    std::uint8_t* ap = out.alpha.data();
    // The previous scanline (zeros above the first) and this one, and this one's filter byte and filtered bytes.
    std::vector<std::uint8_t> previous(static_cast<std::size_t>(row_bytes), 0), current(static_cast<std::size_t>(row_bytes));
    std::vector<std::uint8_t> filtered(static_cast<std::size_t>(row_bytes) + 1);
    ScanlineInflater stream(parse.idat);
    for (int y = 0; y < height; ++y) {
        stream.read(filtered.data(), filtered.size());
        unfilter_row(filtered[0], filtered.data() + 1, previous.data(), current.data(), row_bytes, bpp);
        const std::uint8_t* row = current.data();
        for (int x = 0; x < width; ++x) {
            std::uint8_t r = 0, g = 0, b = 0;
            switch (color_type) {
                case 0: {
                    const std::uint8_t v = to8(raw_value(row, x, 0), bit_depth);
                    r = g = b = v;
                    break;
                }
                case 2:
                    r = to8(raw_value(row, x, 0), bit_depth);
                    g = to8(raw_value(row, x, 1), bit_depth);
                    b = to8(raw_value(row, x, 2), bit_depth);
                    break;
                case 3: {
                    const std::uint32_t idx = raw_value(row, x, 0);
                    if (idx >= parse.palette.rgb.size()) {
                        throw std::invalid_argument("png palette index is out of range");
                    }
                    r = parse.palette.rgb[idx][0];
                    g = parse.palette.rgb[idx][1];
                    b = parse.palette.rgb[idx][2];
                    break;
                }
                case 4: { // gray + alpha: the gray sample
                    const std::uint8_t v = to8(raw_value(row, x, 0), bit_depth);
                    r = g = b = v;
                    break;
                }
                case 6: // RGBA: the color samples
                    r = to8(raw_value(row, x, 0), bit_depth);
                    g = to8(raw_value(row, x, 1), bit_depth);
                    b = to8(raw_value(row, x, 2), bit_depth);
                    break;
            }
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px += 3;
            if (alpha) {
                const Transparency& t = parse.transparency;
                switch (color_type) {
                case 0:
                    *ap = raw_value(row, x, 0) == t.key[0] ? 0 : 255;
                    break;
                case 2:
                    *ap = raw_value(row, x, 0) == t.key[0] && raw_value(row, x, 1) == t.key[1] &&
                                  raw_value(row, x, 2) == t.key[2]
                              ? 0
                              : 255;
                    break;
                case 3: {
                    const std::uint32_t idx = raw_value(row, x, 0);
                    *ap = idx < t.palette_alpha.size() ? t.palette_alpha[idx] : 255;
                    break;
                }
                case 4:
                    *ap = to8(raw_value(row, x, 1), bit_depth);
                    break;
                case 6:
                    *ap = to8(raw_value(row, x, 3), bit_depth);
                    break;
                }
                ++ap;
            }
        }
        previous.swap(current);
    }
    stream.finish();
    return out;
}

} // namespace infernix::media::decode