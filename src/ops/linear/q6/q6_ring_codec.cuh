#pragma once

// Q6 row-split weights for the staging ring (ops/common/rowsplit_ring.cuh): a chunk is half a 64-value group, 16
// code bytes and 8 bytes of the two-bit plane under the group's FP16 scale, decoded exactly as
// Q6MmaDecodeAtom::decode_pair decodes a pair.

#include "ops/common/memory.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops::detail {

// RowMap: the weight row of a tile's local row, or a negative value for a row without weights. `groups` bounds
// the groups that hold weights (a K tile at or past it decodes to zero).
template <class RowMap>
struct Q6RingCodec {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    int groups_per_row;
    int groups;
    RowMap row_of;

    struct Raw {
        uint4 codes;
        uint2 high; // bits 2 i, 2 i + 1: the fifth and sixth bits of the chunk's K value i
        float scale;
    };

    __device__ __forceinline__ Raw load(int row, int k_tile, int half) const {
        Raw raw{make_uint4(0U, 0U, 0U, 0U), make_uint2(0U, 0U), 0.0f};
        const std::int64_t global = row_of(row);
        if (global >= 0 && k_tile < groups) {
            const std::int64_t group = global * groups_per_row + k_tile;
            raw.codes = load_vec<uint4>(codes + group * Q6RowSplitStorage::kCodeBytesPerGroup + half * 16);
            raw.high  = load_vec<uint2>(high + group * Q6RowSplitStorage::kHighBytesPerGroup + half * 8);
            raw.scale = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(scales + group * Q6RowSplitStorage::kScaleBytesPerGroup)));
        }
        return raw;
    }

    __device__ __forceinline__ void decode(const Raw& raw, unsigned (&pairs)[16]) const {
        const unsigned words[4] = {raw.codes.x, raw.codes.y, raw.codes.z, raw.codes.w};
        const unsigned planes[2] = {raw.high.x, raw.high.y};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned packed = (words[i >> 2] >> (8 * (i & 3))) & 0xffU;
            // Pair i's four plane bits: two for each of its K values.
            const unsigned plane = (planes[i >> 3] >> (4 * (i & 7))) & 0xfU;
            const int q0 = ((static_cast<int>(packed & 0x0fU) | static_cast<int>((plane & 3U) << 4)) ^ 0x20) - 0x20;
            const int q1 = ((static_cast<int>(packed >> 4) | static_cast<int>((plane >> 2) << 4)) ^ 0x20) - 0x20;
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * raw.scale,
                                                               static_cast<float>(q1) * raw.scale);
            pairs[i]                   = *reinterpret_cast<const unsigned*>(&value);
        }
    }
};

// The format of the ring linear (ops/linear/common/rowsplit_ring_linear.cuh).
struct Q6RingFormat {
    struct Weights {
        const std::uint8_t* codes;
        const std::uint8_t* high;
        const std::uint8_t* scales;
    };

    template <class RowMap>
    __device__ static __forceinline__ Q6RingCodec<RowMap> codec(const Weights& weights, int k, int padded_k,
                                                                RowMap row_of) {
        return {weights.codes,
                weights.high,
                weights.scales,
                padded_k / Q6RowSplitStorage::kGroupK,
                (k + Q6RowSplitStorage::kGroupK - 1) / Q6RowSplitStorage::kGroupK,
                row_of};
    }
};

} // namespace infernix::ops::detail
