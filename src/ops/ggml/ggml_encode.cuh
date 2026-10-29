#pragma once

// One column's A16 encoding by a whole CTA (ops/ggml/ggml_dp4a.cuh: the column's exponent, then per
// 32-element slice its three limb planes and its X sum, with the optional input block permutation):
// ggml_linear's encode_kernel, and the producers that write ggml_linear's input already encoded
// (include/infernix/ops/ggml_encoded.h), so both give the same words.

#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_dp4a.cuh"

#include <cstdint>

namespace infernix::ops::ggml_encode {

namespace fast = ggml::fast;

// The column exponent from the largest |bits| (sign cleared) of its BF16 values.
__device__ __forceinline__ int column_exponent(unsigned key) {
    return key >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(key >> 7), 1);
}

// The largest |bits| of eight BF16 values.
__device__ __forceinline__ unsigned max8(const uint4& u) {
    const unsigned m0 = max(u.x & 0x7FFFU, (u.x >> 16) & 0x7FFFU), m1 = max(u.y & 0x7FFFU, (u.y >> 16) & 0x7FFFU);
    const unsigned m2 = max(u.z & 0x7FFFU, (u.z >> 16) & 0x7FFFU), m3 = max(u.w & 0x7FFFU, (u.w >> 16) & 0x7FFFU);
    return max(max(m0, m1), max(m2, m3));
}

// Chunk c (elements 8c .. 8c + 7, values u) of a column of exponent e: its groups 2c and 2c + 1 into `column`;
// returns their sum.
__device__ __forceinline__ int encode_chunk(std::int32_t* __restrict__ column, int c, int e, const uint4& u) {
    const auto value = [&](unsigned bits) {
        return e == canon::kA16NonFinite ? 0 : canon::a16_value(static_cast<std::uint16_t>(bits), e);
    };
    int sum = fast::store_group(column, 2 * c, value(u.x & 0xFFFFU), value(u.x >> 16), value(u.y & 0xFFFFU),
                                value(u.y >> 16));
    sum += fast::store_group(column, 2 * c + 1, value(u.z & 0xFFFFU), value(u.z >> 16), value(u.w & 0xFFFFU),
                             value(u.w >> 16));
    return sum;
}

// A slice's sum over the four chunk lanes that hold it (every lane of the warp calls it; chunk c in lane c mod 32).
__device__ __forceinline__ void finish_chunk(std::int32_t* __restrict__ column, int slices, int c, bool valid, int sum) {
    sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
    sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2);
    if (valid && c % 4 == 0) { column[slices * fast::kSliceWords + c / 4] = sum; }
}

// Encodes one column with the whole CTA (Threads threads): its exponent (returned to every thread),
// then its limbs and slice sums into `column`. `warp_key` and `broadcast` are shared scratch.
template <int Threads>
__device__ int encode_column(const std::uint16_t* __restrict__ v, int k, const std::int32_t* __restrict__ perm,
                             int perm_block, std::int32_t* __restrict__ column, unsigned* warp_key, int* broadcast) {
    constexpr int kEncodeWarps = Threads / 32;
    const auto source = [&](int j) { return perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block; };
    unsigned key = 0;
    for (int j = threadIdx.x; j < k; j += Threads) { key = max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (threadIdx.x % 32 == 0) { warp_key[threadIdx.x / 32] = key; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned m = 0;
        for (int w = 0; w < kEncodeWarps; ++w) { m = max(m, warp_key[w]); }
        *broadcast = column_exponent(m);
    }
    __syncthreads();
    const int e = *broadcast;
    const int groups = k / 4, slices = k / 32;
    const auto value = [&](int j) { return e == canon::kA16NonFinite ? 0 : canon::a16_value(v[source(j)], e); };
    // Groups are visited so that the eight groups of a slice sit in eight consecutive lanes.
    for (int base = 0; base < groups; base += Threads) {
        const int g = base + static_cast<int>(threadIdx.x);
        int sum     = 0;
        if (g < groups) { sum = fast::store_group(column, g, value(4 * g), value(4 * g + 1), value(4 * g + 2), value(4 * g + 3)); }
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2);
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 4);
        if (g < groups && g % 8 == 0) { column[slices * fast::kSliceWords + g / 8] = sum; }
    }
    return e;
}

// encode_column with 16-byte loads, for every column it can read (end to end 0.7 % faster, two
// rounds, 2026-10-16). Thread i takes the column's 8-element chunks i, i + Threads, ... (chunk c holds
// groups 2c and 2c + 1, read from the permuted source in one load: needs perm_block % 8 == 0 and a
// 16-byte aligned column), keeps the first kKeep of them in registers from the exponent's pass to the
// limbs' pass, and adds a slice's sum over its four chunks' lanes. Writes the same words as
// encode_column.
template <int Threads>
__device__ int encode_column_vec(const std::uint16_t* __restrict__ v, int k, const std::int32_t* __restrict__ perm,
                                 int perm_block, std::int32_t* __restrict__ column, unsigned* warp_key, int* broadcast) {
    constexpr int kEncodeWarps = Threads / 32;
    constexpr int kKeep        = 4;
    const auto source = [&](int j) { return perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block; };
    const int chunks  = k / 8;
    const auto load   = [&](int c) { return *reinterpret_cast<const uint4*>(v + source(8 * c)); };
    uint4 raw[kKeep];
    unsigned key = 0;
#pragma unroll
    for (int m = 0; m < kKeep; ++m) {
        const int c = m * Threads + static_cast<int>(threadIdx.x);
        raw[m]      = c < chunks ? load(c) : make_uint4(0, 0, 0, 0);
        key         = max(key, max8(raw[m]));
    }
    for (int c = kKeep * Threads + static_cast<int>(threadIdx.x); c < chunks; c += Threads) { key = max(key, max8(load(c))); }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (threadIdx.x % 32 == 0) { warp_key[threadIdx.x / 32] = key; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned m = 0;
        for (int w = 0; w < kEncodeWarps; ++w) { m = max(m, warp_key[w]); }
        *broadcast = column_exponent(m);
    }
    __syncthreads();
    const int e = *broadcast;
    const int slices = k / 32;
#pragma unroll
    for (int m = 0; m < kKeep; ++m) {
        if (m * Threads >= chunks) { break; } // uniform
        const int c = m * Threads + static_cast<int>(threadIdx.x);
        finish_chunk(column, slices, c, c < chunks, c < chunks ? encode_chunk(column, c, e, raw[m]) : 0);
    }
    for (int base = kKeep * Threads; base < chunks; base += Threads) {
        const int c = base + static_cast<int>(threadIdx.x);
        finish_chunk(column, slices, c, c < chunks, c < chunks ? encode_chunk(column, c, e, load(c)) : 0);
    }
    return e;
}

} // namespace infernix::ops::ggml_encode
