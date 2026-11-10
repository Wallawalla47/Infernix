#pragma once

// Q8 row-split weights for the staging ring (ops/common/rowsplit_ring.cuh): a chunk is one 32-value group, 32
// signed code bytes under the group's FP16 scale, decoded as q8_a16_mma_kernel's dequant pass decodes them
// (bf16(code * scale), a pair at a time).

#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops::detail {

// RowMap: the weight row of a tile's local row, or a negative value for a row without weights. `groups` bounds
// the groups that hold weights (one past K's last group decodes to zero; within the last group the activation is
// zero past K).
template <class RowMap>
struct Q8RingCodec {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int padded_k;
    int groups;
    RowMap row_of;

    struct Raw {
        uint4 codes[2];
        float scale;
    };

    __device__ __forceinline__ Raw load(int row, int k_tile, int half) const {
        Raw raw{{make_uint4(0U, 0U, 0U, 0U), make_uint4(0U, 0U, 0U, 0U)}, 0.0f};
        const std::int64_t global = row_of(row);
        const int group           = k_tile * 2 + half;
        if (global >= 0 && group < groups) {
            const std::uint8_t* chunk = codes + global * padded_k + group * Q8RowSplitStorage::kGroupK;
            raw.codes[0]              = load_vec<uint4>(chunk);
            raw.codes[1]              = load_vec<uint4>(chunk + 16);
            raw.scale                 = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                scales + (global * (padded_k / Q8RowSplitStorage::kGroupK) + group) *
                             Q8RowSplitStorage::kScaleBytesPerGroup)));
        }
        return raw;
    }

    __device__ __forceinline__ void decode(const Raw& raw, unsigned (&pairs)[16]) const {
        const unsigned words[8] = {raw.codes[0].x, raw.codes[0].y, raw.codes[0].z, raw.codes[0].w,
                                   raw.codes[1].x, raw.codes[1].y, raw.codes[1].z, raw.codes[1].w};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned word = words[i >> 1] >> ((i & 1) * 16);
            const int q0        = static_cast<int>(static_cast<std::int8_t>(word & 0xffU));
            const int q1        = static_cast<int>(static_cast<std::int8_t>((word >> 8) & 0xffU));
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * raw.scale,
                                                               static_cast<float>(q1) * raw.scale);
            pairs[i]                   = *reinterpret_cast<const unsigned*>(&value);
        }
    }
};

// The format of the ring linear (ops/linear/common/rowsplit_ring_linear.cuh).
struct Q8RingFormat {
    struct Weights {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
    };

    template <class RowMap>
    __device__ static __forceinline__ Q8RingCodec<RowMap> codec(const Weights& weights, int k, int padded_k,
                                                                RowMap row_of) {
        return {weights.codes, weights.scales, padded_k,
                (k + Q8RowSplitStorage::kGroupK - 1) / Q8RowSplitStorage::kGroupK, row_of};
    }
};

} // namespace infernix::ops::detail
