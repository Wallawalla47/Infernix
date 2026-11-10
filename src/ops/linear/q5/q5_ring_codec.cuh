#pragma once

// Q5 row-split weights for the staging ring (ops/common/rowsplit_ring.cuh): a chunk is half a 64-value group, 16
// code bytes and 4 bytes of the high-bit plane under the group's FP16 scale, decoded exactly as
// Q5MmaDecodeAtom::decode_pair decodes a pair.

#include "ops/common/memory.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops::detail {

// RowMap: the weight row of a tile's local row, or a negative value for a row without weights. `groups` bounds
// the groups that hold weights (a K tile at or past it decodes to zero).
template <class RowMap>
struct Q5RingCodec {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    int groups_per_row;
    int groups;
    RowMap row_of;

    struct Raw {
        uint4 codes;
        unsigned high; // bit i: the fifth bit of the chunk's K value i
        float scale;
    };

    __device__ __forceinline__ Raw load(int row, int k_tile, int half) const {
        Raw raw{make_uint4(0U, 0U, 0U, 0U), 0U, 0.0f};
        const std::int64_t global = row_of(row);
        if (global >= 0 && k_tile < groups) {
            const std::int64_t group = global * groups_per_row + k_tile;
            raw.codes = load_vec<uint4>(codes + group * Q5RowSplitStorage::kCodeBytesPerGroup + half * 16);
            raw.high  = *reinterpret_cast<const std::uint32_t*>(high + group * Q5RowSplitStorage::kHighBytesPerGroup +
                                                                half * 4);
            raw.scale = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(scales + group * Q5RowSplitStorage::kScaleBytesPerGroup)));
        }
        return raw;
    }

    __device__ __forceinline__ void decode(const Raw& raw, unsigned (&pairs)[16]) const {
        const unsigned words[4] = {raw.codes.x, raw.codes.y, raw.codes.z, raw.codes.w};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned packed = (words[i >> 2] >> (8 * (i & 3))) & 0xffU;
            const int q0 =
                ((static_cast<int>(packed & 0x0fU) | static_cast<int>(((raw.high >> (2 * i)) & 1U) << 4)) ^ 0x10) - 0x10;
            const int q1 =
                ((static_cast<int>(packed >> 4) | static_cast<int>(((raw.high >> (2 * i + 1)) & 1U) << 4)) ^ 0x10) - 0x10;
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * raw.scale,
                                                               static_cast<float>(q1) * raw.scale);
            pairs[i]                   = *reinterpret_cast<const unsigned*>(&value);
        }
    }
};

// The format of the ring linear (ops/linear/common/rowsplit_ring_linear.cuh).
struct Q5RingFormat {
    struct Weights {
        const std::uint8_t* codes;
        const std::uint8_t* high;
        const std::uint8_t* scales;
    };

    template <class RowMap>
    __device__ static __forceinline__ Q5RingCodec<RowMap> codec(const Weights& weights, int k, int padded_k,
                                                                RowMap row_of) {
        return {weights.codes,
                weights.high,
                weights.scales,
                padded_k / Q5RowSplitStorage::kGroupK,
                (k + Q5RowSplitStorage::kGroupK - 1) / Q5RowSplitStorage::kGroupK,
                row_of};
    }
};

} // namespace infernix::ops::detail
