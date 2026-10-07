#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_shared.cuh"

namespace infernix::ops::detail {

// Register-streamed Q8 decode GEMM (kernel K1 of the Qwen3.8-Flash-Next dense-Q8 plan): every code
// and scale a warp's rows need is loaded into registers before the first FMA, so a warp keeps its
// whole share of the weights in flight at once, and the rows of a warp share each x load.
//
// The per-(row, column) arithmetic is that of the SIMT route (q8_a16_simt.cuh) and the GEMV route:
// lane l owns the 8-code chunks at k = 8l + 256m of its warp's K range, m increasing, each decoded
// as float(code) * scale and accumulated with fmaf in k order; lane partials reduce with the same
// shuffle tree; warps that split a row's K hold contiguous scale-pair ranges and their partials are
// added in warp order. A column's result therefore does not depend on how many columns share the
// call, and matches the SIMT route bit for bit for 1..8 columns.
//
// Columns are taken in passes of kColsPerPass over the register-resident weights, so one launch
// covers up to kMaxCols columns per CTA (grid.y slices beyond that).
template <class S, bool FullRows, class Output, class Epilogue>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void q8_a16_stream_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, Epilogue epilogue, int rows,
    int padded_k, int tokens, int token_begin) {
    constexpr int K      = S::kK;
    constexpr int R      = S::kRowsPerWarp;
    constexpr int KS     = S::kKSplit;
    constexpr int C      = S::kChunks;
    constexpr int CP     = S::kColsPerPass;
    constexpr int WarpK  = S::kWarpK;
    constexpr bool Whole = WarpK % 256 == 0; // every lane owns every chunk

    unsigned char* storage = q8_shared_storage<S::kSharedBytes>();
    auto* x_tile           = reinterpret_cast<__nv_bfloat16*>(storage);                        // [CP][K]
    auto* partial          = reinterpret_cast<float*>(storage + S::kXTileBytes);                // [groups][KS][R][CP]

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int group   = warp / KS;
    const int split   = warp % KS;
    const int row0    = static_cast<int>(blockIdx.x) * S::kBlockRows + group * R;
    const int k_begin = split * WarpK;
    const int col0    = static_cast<int>(blockIdx.y) * S::kMaxCols;
    const int live    = min(S::kMaxCols, tokens - col0);
    const auto* x_cta = x + static_cast<std::int64_t>(col0) * K;

    // 1. Weights first: every code and scale of the warp's rows, before any x or FMA.
    uint2 code[R][C];
    std::uint32_t scale_bits[R][(C + 1) / 2];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row             = row0 + r;
        const std::int64_t source = FullRows || row < rows ? row : 0;
        const auto* code_row      = codes + source * padded_k + k_begin;
        const auto* scale_row     = scales + source * (padded_k / 32) * 2 + (k_begin / 32) * 2;
#pragma unroll
        for (int m = 0; m < C; ++m) {
            const int kk = m * 256 + lane * 8;
            std::uint32_t bits = 0;
            if (Whole || kk < WarpK) {
                code[r][m] = ld_nc_na<uint2>(code_row + kk);
                bits       = ld_nc_na<std::uint16_t>(scale_row + (kk / 32) * 2);
            } else {
                code[r][m] = uint2{0u, 0u};
            }
            if (m % 2 == 0) {
                scale_bits[r][m / 2] = bits;
            } else {
                scale_bits[r][m / 2] |= bits << 16;
            }
        }
    }

#pragma unroll 1
    for (int pass = 0; pass < S::kPasses; ++pass) {
        const int pass_col = pass * CP;
        if (pass_col >= live) break;
        const int pass_live = min(CP, live - pass_col);

        // 2. Stage this pass's x columns (or read them through L1).
        if constexpr (S::kXStage == Q8StreamX::Shared) {
            if (pass > 0) __syncthreads(); // every warp is done with the previous pass's tile
            constexpr int kVecsPerCol = K / 8;
            const int vecs            = pass_live * kVecsPerCol;
            for (int v = static_cast<int>(threadIdx.x); v < vecs; v += S::kThreads) {
                cp_async<16, Cache::cg>(x_tile + static_cast<std::int64_t>(v) * 8,
                                        x_cta + static_cast<std::int64_t>(pass_col) * K +
                                            static_cast<std::int64_t>(v) * 8);
            }
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }

        // 3. Stream: for each chunk decode the rows' weights once, reuse them for every column.
        float acc[R][CP];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int c = 0; c < CP; ++c) acc[r][c] = 0.0f;
        }
#pragma unroll
        for (int m = 0; m < C; ++m) {
            const int kk = m * 256 + lane * 8;
            if (Whole || kk < WarpK) {
                float w[R][8];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const unsigned half_bits = (scale_bits[r][m / 2] >> ((m % 2) * 16)) & 0xFFFFu;
                    const float scale        = __half2float(__ushort_as_half(static_cast<unsigned short>(half_bits)));
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        w[r][i] = static_cast<float>(static_cast<std::int8_t>(
                                      (i < 4 ? code[r][m].x : code[r][m].y) >> ((i & 3) * 8))) *
                                  scale;
                    }
                }
#pragma unroll
                for (int c = 0; c < CP; ++c) {
                    if (c < pass_live) {
                        uint4 values;
                        if constexpr (S::kXStage == Q8StreamX::Shared) {
                            values = load_vec<uint4>(x_tile + static_cast<std::int64_t>(c) * K + k_begin + kk);
                        } else {
                            values = load_ldg<uint4>(x_cta + static_cast<std::int64_t>(pass_col + c) * K +
                                                     k_begin + kk);
                        }
                        const float2 x0 = bf16x2_bits_to_float2(values.x);
                        const float2 x1 = bf16x2_bits_to_float2(values.y);
                        const float2 x2 = bf16x2_bits_to_float2(values.z);
                        const float2 x3 = bf16x2_bits_to_float2(values.w);
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            acc[r][c] = fmaf(w[r][0], x0.x, acc[r][c]);
                            acc[r][c] = fmaf(w[r][1], x0.y, acc[r][c]);
                            acc[r][c] = fmaf(w[r][2], x1.x, acc[r][c]);
                            acc[r][c] = fmaf(w[r][3], x1.y, acc[r][c]);
                            acc[r][c] = fmaf(w[r][4], x2.x, acc[r][c]);
                            acc[r][c] = fmaf(w[r][5], x2.y, acc[r][c]);
                            acc[r][c] = fmaf(w[r][6], x3.x, acc[r][c]);
                            acc[r][c] = fmaf(w[r][7], x3.y, acc[r][c]);
                        }
                    }
                }
            }
        }

        // 4. Reduce lanes (the SIMT route's shuffle tree), then K warps in warp order.
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int c = 0; c < CP; ++c) acc[r][c] = warp_reduce_sum(acc[r][c]);
        }
        if constexpr (KS > 1) {
            if (lane == 0) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int c = 0; c < CP; ++c) partial[((group * KS + split) * R + r) * CP + c] = acc[r][c];
                }
            }
            __syncthreads();
            if (split == 0 && lane == 0) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int c = 0; c < CP; ++c) {
                        float sum = partial[((group * KS) * R + r) * CP + c];
#pragma unroll
                        for (int part = 1; part < KS; ++part) sum += partial[((group * KS + part) * R + r) * CP + c];
                        acc[r][c] = sum;
                    }
                }
            }
            __syncthreads(); // the partials are rewritten by the next pass
        }

        // 5. Epilogue: lane 0 of the row group's first warp finishes its rows.
        if (split == 0 && lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                if (FullRows || row < rows) {
                    linear_finish_row(output, epilogue, row, token_begin + col0 + pass_col, acc[r], pass_live);
                }
            }
        }
    }
}

} // namespace infernix::ops::detail
