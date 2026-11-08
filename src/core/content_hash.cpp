#include "core/content_hash.h"

#include <cstring>

namespace infernix {
namespace {

constexpr std::uint64_t kPrime1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t kPrime2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t kPrime3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t kPrime4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t kPrime5 = 0x27D4EB2F165667C5ULL;

constexpr std::uint64_t rotate(std::uint64_t value, int bits) noexcept {
    return (value << bits) | (value >> (64 - bits));
}

std::uint64_t word(const std::byte* p) noexcept {
    std::uint64_t out = 0;
    std::memcpy(&out, p, sizeof(out)); // every supported target is little-endian
    return out;
}

constexpr std::uint64_t round(std::uint64_t acc, std::uint64_t input) noexcept {
    return rotate(acc + input * kPrime2, 31) * kPrime1;
}

constexpr std::uint64_t merge(std::uint64_t acc, std::uint64_t lane) noexcept {
    return (acc ^ round(0, lane)) * kPrime1 + kPrime4;
}

} // namespace

std::uint64_t content_hash(const std::byte* data, std::size_t bytes) noexcept {
    const std::byte* p   = data;
    const std::byte* end = data + bytes;
    std::uint64_t hash   = 0;
    if (bytes >= 32) {
        std::uint64_t v1 = kPrime1 + kPrime2, v2 = kPrime2, v3 = 0, v4 = 0ULL - kPrime1;
        for (; p + 32 <= end; p += 32) {
            v1 = round(v1, word(p));
            v2 = round(v2, word(p + 8));
            v3 = round(v3, word(p + 16));
            v4 = round(v4, word(p + 24));
        }
        hash = rotate(v1, 1) + rotate(v2, 7) + rotate(v3, 12) + rotate(v4, 18);
        hash = merge(hash, v1);
        hash = merge(hash, v2);
        hash = merge(hash, v3);
        hash = merge(hash, v4);
    } else {
        hash = kPrime5;
    }
    hash += static_cast<std::uint64_t>(bytes);
    for (; p + 8 <= end; p += 8) { hash = rotate(hash ^ round(0, word(p)), 27) * kPrime1 + kPrime4; }
    for (; p < end; ++p) {
        hash = rotate(hash ^ (static_cast<std::uint64_t>(static_cast<unsigned char>(*p)) * kPrime5), 11) * kPrime1;
    }
    hash ^= hash >> 33;
    hash *= kPrime2;
    hash ^= hash >> 29;
    hash *= kPrime3;
    hash ^= hash >> 32;
    return hash;
}

} // namespace infernix
