#pragma once

#include <cstddef>
#include <cstdint>

namespace infernix::runtime::prefix_cache {

// The SSD tier's lossless codec for state images (docs/maintainer/hybrid-prefix-cache-spec.md §5.7).
// A state image is mostly FP32 recurrent state, whose exponents span a few binades within a row while
// sign and mantissa are incompressible (measured on Flash-Next FP4-Local snapshots: 3.4 bits of
// exponent entropy, 2.9 given the row's largest exponent, 7.97 bits in the top mantissa byte). The
// codec reads 32-bit words in rows of 128 (512 bytes) and stores each row either raw or as its largest
// exponent and a 4-bit offset per word (15 escapes to a raw exponent byte), with the low 16 mantissa
// bits and the sign with the top 7 mantissa bits as plain planes: 451 bytes plus one per escape, about
// 12 % less on that state. Other bytes (16-bit state, page records) mostly stay raw row by row. Exact
// for every bit pattern (NaN, infinity, subnormal and zero included).
//
// A payload encodes one slab: a 4-byte header (the body's length; the top bit marks a slab stored
// raw), then the body. A slab is stored raw when its rows would not shrink it.
inline constexpr std::size_t kCodecRowBytes = 512;

// Upper bound of a payload for a slab of `bytes` (a multiple of kCodecRowBytes).
[[nodiscard]] constexpr std::size_t codec_payload_bound(std::size_t bytes) noexcept { return 4U + bytes; }

// Encodes `bytes` (a positive multiple of kCodecRowBytes) of `in` into `out` (codec_payload_bound
// bytes); returns the payload's length.
[[nodiscard]] std::size_t encode_slab(const std::byte* in, std::size_t bytes, std::byte* out) noexcept;

// The length of the payload starting at `payload` (its header and body).
[[nodiscard]] std::size_t payload_size(const std::byte* payload) noexcept;

// Decodes the payload at `payload` into the `bytes` it encodes; false when it is not a payload of a slab
// of `bytes` (a damaged record), leaving `out` unspecified.
[[nodiscard]] bool decode_slab(const std::byte* payload, std::size_t available, std::byte* out,
                               std::size_t bytes) noexcept;

} // namespace infernix::runtime::prefix_cache
