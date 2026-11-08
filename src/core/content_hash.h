#pragma once

#include <cstddef>
#include <cstdint>

namespace infernix {

// A fast 64-bit content hash of a byte range (four independent multiply-rotate lanes over 8-byte words,
// then an avalanche; the XXH64 construction). It detects changed or foreign bytes, such as an SSD slab
// that a crashed process overwrote after the prefix cache file that names it was saved; it is not a
// cryptographic hash. The same bytes give the same value on every platform (little-endian words).
[[nodiscard]] std::uint64_t content_hash(const std::byte* data, std::size_t bytes) noexcept;

} // namespace infernix
