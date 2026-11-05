#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

// Byte j of a register value, read by shift rather than through a byte pointer into the local value: an optimised
// CUDA 13.2 sm_120 build returned wrong bytes for that read (Basalt #121; candidates WB-03). j is a constant after
// unrolling. The signed form returns std::int8_t, not int: converted to float it is then one byte-select I2F.S8, as
// the pointer read was; an int would be sign-extended by two shifts first.
__device__ __forceinline__ std::uint32_t byte_of(std::uint32_t word, int j) { return (word >> (8 * j)) & 0xFFU; }
__device__ __forceinline__ std::int8_t signed_byte_of(std::uint32_t word, int j) {
    return static_cast<std::int8_t>(static_cast<std::uint8_t>(word >> (8 * j)));
}
__device__ __forceinline__ std::int8_t signed_byte_of(uint2 value, int j) {
    return signed_byte_of(j < 4 ? value.x : value.y, j & 3);
}
__device__ __forceinline__ std::int8_t signed_byte_of(int2 value, int j) {
    return signed_byte_of(static_cast<std::uint32_t>(j < 4 ? value.x : value.y), j & 3);
}
__device__ __forceinline__ std::uint32_t word_of(uint4 value, int j) {
    return j < 4 ? value.x : j < 8 ? value.y : j < 12 ? value.z : value.w;
}
__device__ __forceinline__ std::uint32_t byte_of(uint4 value, int j) { return byte_of(word_of(value, j), j & 3); }
__device__ __forceinline__ std::int8_t signed_byte_of(uint4 value, int j) {
    return signed_byte_of(word_of(value, j), j & 3);
}

} // namespace infernix::ops
