#pragma once
#include "ops/common/math.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace infernix::ops::detail {
template <int RowsPerBranch>
struct Nvfp4SwiGluRows {
    static constexpr bool kPaired = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int rows) const {
        return begin + row % RowsPerBranch + (row >= RowsPerBranch ? rows / 2 : 0);
    }
};

// A16 MMA rows: every 16-row MMA tile holds 8 gate rows over their 8 up rows, so the thread that
// owns C-fragment row g also owns g+8 and finishes an output row without crossing lanes. A CTA of
// BM weight rows covers BM/2 consecutive output rows.
struct Nvfp4SwiGluMmaRows {
    static constexpr bool kPaired       = true;
    static constexpr bool kThreadPaired = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int rows) const {
        return begin + (row / 16) * 8 + row % 8 + (row % 16 >= 8 ? rows / 2 : 0);
    }

    __device__ __forceinline__ int output_row(int begin, int row) const {
        return begin + (row / 16) * 8 + row % 8;
    }
};

struct Nvfp4SwiGluEpilogue {
    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, silu(gate) * up);
    }
};
} // namespace infernix::ops::detail
