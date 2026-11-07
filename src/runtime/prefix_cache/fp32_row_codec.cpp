#include "runtime/prefix_cache/fp32_row_codec.h"

#include <array>
#include <cstring>

namespace infernix::runtime::prefix_cache {
namespace {

constexpr std::size_t kWords        = kCodecRowBytes / 4U; // 128
constexpr std::uint8_t kRowPacked   = 0;
constexpr std::uint8_t kRowRaw      = 1;
constexpr std::uint32_t kEscape     = 15;
constexpr std::size_t kPackedFixed  = 3U + kWords / 2U + kWords * 3U; // 451
constexpr std::uint32_t kRawSlabBit = 0x80000000U;

std::uint32_t load_u32(const std::byte* p) noexcept {
    std::uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

void store_u32(std::byte* p, std::uint32_t v) noexcept { std::memcpy(p, &v, sizeof(v)); }

// One row of 128 words into `out`; returns the bytes written (452..513).
std::size_t encode_row(const std::byte* in, std::byte* out) noexcept {
    std::array<std::uint32_t, kWords> words;
    std::memcpy(words.data(), in, kCodecRowBytes);
    std::array<std::uint8_t, kWords> exponent;
    std::uint32_t top = 0;
    for (std::size_t i = 0; i < kWords; ++i) {
        exponent[i] = static_cast<std::uint8_t>((words[i] >> 23U) & 0xFFU);
        top         = exponent[i] > top ? exponent[i] : top;
    }
    std::size_t escapes = 0;
    for (std::size_t i = 0; i < kWords; ++i) { escapes += top - exponent[i] >= kEscape ? 1U : 0U; }
    if (kPackedFixed + escapes >= 1U + kCodecRowBytes) {
        out[0] = static_cast<std::byte>(kRowRaw);
        std::memcpy(out + 1, in, kCodecRowBytes);
        return 1U + kCodecRowBytes;
    }
    out[0]                 = static_cast<std::byte>(kRowPacked);
    out[1]                 = static_cast<std::byte>(top);
    out[2]                 = static_cast<std::byte>(escapes);
    std::byte* nibbles     = out + 3;
    std::byte* escaped     = nibbles + kWords / 2U;
    std::byte* low         = escaped + escapes;
    std::byte* high        = low + kWords * 2U;
    std::size_t next_escape = 0;
    for (std::size_t i = 0; i < kWords; i += 2) {
        const std::uint32_t a = top - exponent[i], b = top - exponent[i + 1U];
        const std::uint32_t ca = a >= kEscape ? kEscape : a, cb = b >= kEscape ? kEscape : b;
        nibbles[i / 2U] = static_cast<std::byte>(ca | (cb << 4U));
        if (ca == kEscape) { escaped[next_escape++] = static_cast<std::byte>(exponent[i]); }
        if (cb == kEscape) { escaped[next_escape++] = static_cast<std::byte>(exponent[i + 1U]); }
    }
    for (std::size_t i = 0; i < kWords; ++i) {
        const auto lo = static_cast<std::uint16_t>(words[i] & 0xFFFFU);
        std::memcpy(low + 2U * i, &lo, sizeof(lo));
        high[i] = static_cast<std::byte>(((words[i] >> 24U) & 0x80U) | ((words[i] >> 16U) & 0x7FU));
    }
    return kPackedFixed + escapes;
}

// One row from `in` (at most `available` bytes); returns the bytes read, 0 when damaged.
std::size_t decode_row(const std::byte* in, std::size_t available, std::byte* out) noexcept {
    if (available < 1U) { return 0; }
    const auto kind = static_cast<std::uint8_t>(in[0]);
    if (kind == kRowRaw) {
        if (available < 1U + kCodecRowBytes) { return 0; }
        std::memcpy(out, in + 1, kCodecRowBytes);
        return 1U + kCodecRowBytes;
    }
    if (kind != kRowPacked || available < kPackedFixed) { return 0; }
    const auto top        = static_cast<std::uint32_t>(in[1]);
    const auto escapes    = static_cast<std::size_t>(in[2]);
    if (available < kPackedFixed + escapes) { return 0; }
    const std::byte* nibbles = in + 3;
    const std::byte* escaped = nibbles + kWords / 2U;
    const std::byte* low     = escaped + escapes;
    const std::byte* high    = low + kWords * 2U;
    std::array<std::uint32_t, kWords> words;
    for (std::size_t i = 0; i < kWords; ++i) {
        const std::uint32_t code = (static_cast<std::uint32_t>(nibbles[i >> 1U]) >> ((i & 1U) * 4U)) & 0xFU;
        std::uint16_t lo;
        std::memcpy(&lo, low + 2U * i, sizeof(lo));
        const auto hi = static_cast<std::uint32_t>(high[i]);
        words[i]      = ((hi & 0x80U) << 24U) | (((top - code) & 0xFFU) << 23U) | ((hi & 0x7FU) << 16U) | lo;
    }
    if (escapes != 0) {
        std::size_t next = 0;
        for (std::size_t i = 0; i < kWords; ++i) {
            const std::uint32_t code = (static_cast<std::uint32_t>(nibbles[i >> 1U]) >> ((i & 1U) * 4U)) & 0xFU;
            if (code != kEscape) { continue; }
            if (next == escapes) { return 0; }
            words[i] = (words[i] & ~(0xFFU << 23U)) | (static_cast<std::uint32_t>(escaped[next++]) << 23U);
        }
        if (next != escapes) { return 0; }
    }
    std::memcpy(out, words.data(), kCodecRowBytes);
    return kPackedFixed + escapes;
}

} // namespace

std::size_t encode_slab(const std::byte* in, std::size_t bytes, std::byte* out) noexcept {
    std::byte* body  = out + 4;
    std::size_t used = 0;
    for (std::size_t offset = 0; offset < bytes; offset += kCodecRowBytes) {
        // A row may take 513 bytes: one that might not fit leaves the slab raw (it is not shrinking).
        if (used + 1U + kCodecRowBytes > bytes) {
            used = bytes;
            break;
        }
        used += encode_row(in + offset, body + used);
    }
    if (used >= bytes) {
        // Rows that would not shrink the slab: it is stored as it is.
        std::memcpy(body, in, bytes);
        store_u32(out, static_cast<std::uint32_t>(bytes) | kRawSlabBit);
        return 4U + bytes;
    }
    store_u32(out, static_cast<std::uint32_t>(used));
    return 4U + used;
}

std::size_t payload_size(const std::byte* payload) noexcept { return 4U + (load_u32(payload) & ~kRawSlabBit); }

bool decode_slab(const std::byte* payload, std::size_t available, std::byte* out, std::size_t bytes) noexcept {
    if (available < 4U || bytes % kCodecRowBytes != 0) { return false; }
    const std::uint32_t header = load_u32(payload);
    const std::size_t body     = header & ~kRawSlabBit;
    if (available < 4U + body) { return false; }
    const std::byte* in = payload + 4;
    if ((header & kRawSlabBit) != 0) {
        if (body != bytes) { return false; }
        std::memcpy(out, in, bytes);
        return true;
    }
    std::size_t read = 0;
    for (std::size_t offset = 0; offset < bytes; offset += kCodecRowBytes) {
        const std::size_t row = decode_row(in + read, body - read, out + offset);
        if (row == 0) { return false; }
        read += row;
    }
    return read == body;
}

} // namespace infernix::runtime::prefix_cache
