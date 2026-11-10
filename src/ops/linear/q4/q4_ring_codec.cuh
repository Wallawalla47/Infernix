#pragma once

// Q4 row-split weights for the staging ring (ops/common/rowsplit_ring.cuh): a chunk is half a 64-value group, 16
// code bytes under the group's FP16 scale, decoded exactly as Q4MmaDecodeAtom::decode_pair decodes a pair.

#include "ops/common/memory.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops::detail {

// RowMap: the weight row (in the codes' row order) of a tile's local row, or a negative value for a row without
// weights (a tile past the matrix).
template <class RowMap>
struct Q4RingCodec {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int groups_per_row;
    RowMap row_of;

    struct Raw {
        uint4 codes;
        float scale;
    };

    __device__ __forceinline__ Raw load(int row, int k_tile, int half) const {
        Raw raw{make_uint4(0U, 0U, 0U, 0U), 0.0f};
        const std::int64_t global = row_of(row);
        if (global >= 0) {
            const std::int64_t group = global * groups_per_row + k_tile;
            raw.codes = load_vec<uint4>(codes + group * Q4RowSplitStorage::kCodeBytesPerGroup + half * 16);
            raw.scale = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(scales + group * Q4RowSplitStorage::kScaleBytesPerGroup)));
        }
        return raw;
    }

    __device__ __forceinline__ void decode(const Raw& raw, unsigned (&pairs)[16]) const {
        const unsigned words[4] = {raw.codes.x, raw.codes.y, raw.codes.z, raw.codes.w};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned packed      = (words[i >> 2] >> (8 * (i & 3))) & 0xffU;
            const int q0               = (static_cast<int>(packed & 0x0fU) ^ 0x08) - 0x08;
            const int q1               = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * raw.scale,
                                                               static_cast<float>(q1) * raw.scale);
            pairs[i]                   = *reinterpret_cast<const unsigned*>(&value);
        }
    }
};

// The format of the ring linear (ops/linear/common/rowsplit_ring_linear.cuh).
struct Q4RingFormat {
    struct Weights {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
    };

    template <class RowMap>
    __device__ static __forceinline__ Q4RingCodec<RowMap> codec(const Weights& weights, int, int padded_k,
                                                                RowMap row_of) {
        return {weights.codes, weights.scales, padded_k / Q4RowSplitStorage::kGroupK, row_of};
    }
};

} // namespace infernix::ops::detail
