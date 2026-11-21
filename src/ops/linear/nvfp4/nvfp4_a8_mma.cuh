#pragma once

// NVFP4 weights with 8-bit activations (LinearPolicy::A8Only: `--a8-activations` on the NVFP4 Qwen3.8-27B artifacts; LOSSY,
// opt-in, never on the reproducible or lossless route). The activations are ops/ggml/ggml_a8.h's codec, the one
// 8-bit activation codec of the flag: each column quantized per 32 values of K to int8 codes q with a binary32
// scale d_s (ops::a8_encode_columns: a column's K codes, then its K / 32 scales). One NVFP4 block b (16 values of
// K, E4M3 scale word w_b) of row r against column t:
//
//   P_b  = sum_{j<16} c2(code[r, 16b + j]) * q[t, 16b + j]      c2 the doubled E2M1 value (canon::e2m1_x2_quad):
//                                                                |P_b| <= 16 * 12 * 127 < 2^15, exact in int32
//   acc  = fma(P_b * S_b, d_s, acc)                               S_b = the E4M3 value of w_b; P_b * S_b is exact
//                                                                (< 2^19 significant), one m16n8k16 s8 MMA a block
//                                                                from zero (::infernix::ggml::a8::mma_k16), P_b
//                                                                converted with one exact I2F
//   y    = epilogue(acc * alpha)                                  alpha = nvfp4_a8_alpha: half the matrix
//                                                                coefficient (the doubled codes), exact
//
// in K order per thread, then the routes' K-warp partials (the sliced-K tiles) in the A16 tiles' order. The FP32
// accumulator takes one rounding a block, so a column's output is deterministic and depends on its own codes and the
// call's tile shape only. The two kernels keep their A16 counterparts' structure (nvfp4_a16_sliced_k_mma.cuh,
// nvfp4_a16_mma.cuh), output and row policies (SwiGLU pairs, residual add, the GDN split outputs):
//
// - sliced-K (decode and verification widths): 16 rows a CTA, each of KWarps warps one 64-value K slice of a
//   KWarps * 64 group, its columns' codes (64 bytes, 80-byte rows: conflict-free B loads) and two slice scales
//   staged per warp; the A fragment decoded in registers from the swizzled codes, one MMA per block and 8-column
//   tile, the K-warp partials summed as the A16 tiles sum theirs.
// - MMA (the 5,120-row projections' 65-256 columns): BlockRows x BlockTokens tiles of BlockK = 128, the codes
//   widened once per K tile to int8 in a padded tile (each fragment then one 32-bit load) between two CTA barriers,
//   the activation codes and scales in a two-stage ring.
// - Wider: the 128 x 128 tile on a decode-warp ring (nvfp4_a8_ring.cuh), whose producer warps widen the codes.

#include "ops/common/canonical_math.h"
#include "ops/common/dynamic_shared.cuh"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/token_slices.h"
#include "ops/ggml/ggml_a8_mma.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"
#include "ops/lora/lora_device.cuh"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/linear/nvfp4/nvfp4_a8_ring.h"
#include "core/weight.h"
#include "core/arena.h"
#include "ops/ggml/ggml_linear_a8.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {

// The 8-bit operands: `x` the encoded columns (a8_encode_columns), `x_stride` bytes apart.
struct Nvfp4A8Operands {
    const std::uint8_t* x;
    int x_stride;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int rows, k, tokens;
    float alpha;
};

inline Nvfp4A8Operands nvfp4_a8_operands(const void* encoded, int stride_bytes, const Weight& w, int tokens) {
    return {static_cast<const std::uint8_t*>(encoded),
            stride_bytes,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales),
            w.n,
            w.k,
            tokens,
            nvfp4_a8_alpha(w)};
}

inline void validate_nvfp4_a8_operands(const Nvfp4A8Operands& p) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    if (p.rows <= 0 || p.rows % 128 || p.k <= 0 || p.k % 128 || p.tokens <= 0 || !aligned(p.x, 16) ||
        p.x_stride % 16 || p.x_stride < p.k + p.k / 8 || !aligned(p.codes, 16) || !aligned(p.scales, 16) ||
        !std::isfinite(p.alpha) || p.alpha <= 0)
        throw std::invalid_argument("NVFP4 A8 requires complete M128/K128 tiles and aligned operands");
}

// The value S of every E4M3 scale word, the table the kernels read a block's scale from.
__device__ __forceinline__ void nvfp4_a8_fill_scale_table(float* table, int tid, int threads) {
    for (int word = tid; word < 256; word += threads) {
        table[word] = decode_nvfp4_e4m3(static_cast<std::uint8_t>(word));
    }
}

// The doubled E2M1 values of four consecutive elements (2 bytes: element 2j in byte j's low nibble, 2j + 1 in its
// high one) as signed bytes in K order: one s8 A-fragment register.
__device__ __forceinline__ unsigned nvfp4_a8_quad(std::uint32_t pair) {
    const std::uint32_t spread =
        (pair & 0xFU) | ((pair & 0xF0U) << 4) | ((pair & 0xF00U) << 8) | ((pair & 0xF000U) << 12);
    return canon::e2m1_x2_quad(spread);
}

// P * S (both conversion and product exact) accumulated with the column's slice scale: acc = fma(P S, d, acc).
__device__ __forceinline__ float nvfp4_a8_term(float acc, int p, float scale, float d) {
    return __fmaf_rn(__fmul_rn(__int2float_rn(p), scale), d, acc);
}

// One NVFP4 block (one m16n8k16 s8 MMA from zero) of a fragment's rows a (scale sa) and b (sb) against two columns.
__device__ __forceinline__ void nvfp4_a8_block(float (&acc)[4], unsigned a0, unsigned a1, unsigned b0, float sa,
                                               float sb, float d0, float d1) {
    int r[4];
    ::infernix::ggml::a8::mma_k16(r, a0, a1, b0);
    acc[0] = nvfp4_a8_term(acc[0], r[0], sa, d0);
    acc[1] = nvfp4_a8_term(acc[1], r[1], sa, d1);
    acc[2] = nvfp4_a8_term(acc[2], r[2], sb, d0);
    acc[3] = nvfp4_a8_term(acc[3], r[3], sb, d1);
}

// ---------------------------------------------------------------------------------------------------- sliced-K

template <int KWarps, int TileTokens, int Stages, int MinBlocksPerSm = 1>
struct Nvfp4A8SlicedSchedule {
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens > 0 && TileTokens % 8 == 0);
    static_assert(Stages == 1 || Stages == 2);
    static constexpr int kKWarps            = KWarps;
    static constexpr int kBlockTokens       = TileTokens;
    static constexpr int kStages            = Stages;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kBlockK            = KWarps * kTileKPerWarp;
    static constexpr int kBlockRows         = 16;
    static constexpr int kRowsPerLoaderWarp = 16 / KWarps;
    static constexpr int kXRowBytes         = 80; // a warp's 64 codes of one column, padded
    static constexpr int kCodeBytes         = 16 * (kBlockK / 2);
    static constexpr int kScaleBytes        = 16 * (kBlockK / 16);
    static constexpr int kXBytes            = KWarps * TileTokens * kXRowBytes;
    static constexpr int kXScaleBytes       = KWarps * TileTokens * 2 * 4;
    static constexpr int kStageBytes        = kCodeBytes + kScaleBytes + kXBytes + kXScaleBytes;
    static constexpr int kStagingBytes      = Stages * kStageBytes;
    static constexpr int kPartialBytes      = KWarps * (TileTokens / 8) * 32 * 4 * 4;
    static constexpr int kTableBytes        = 256 * 4;
    static constexpr int kSharedBytes =
        kTableBytes + (kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes);
    static_assert(kStageBytes % 16 == 0 && kCodeBytes % 16 == 0 && kScaleBytes % 16 == 0 && kXBytes % 16 == 0);
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a8_sliced_k_mma_kernel(
    Nvfp4A8Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy, int token_offset) {
    constexpr int kTileK       = Schedule::kTileKPerWarp;
    constexpr int kWarps       = Schedule::kKWarps;
    constexpr int kBlockK      = Schedule::kBlockK;
    constexpr int kBlockTokens = Schedule::kBlockTokens;
    constexpr int kTokenMmas   = kBlockTokens / 8;
    constexpr int kXRow        = Schedule::kXRowBytes;
    const int kHidden          = operands.k;
    const int kGroups          = kHidden / kBlockK;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* table      = reinterpret_cast<float*>(shared_raw);
    unsigned char* s = shared_raw + Schedule::kTableBytes;
    const auto codes_at  = [&](int stage) { return s + stage * Schedule::kStageBytes; };
    const auto scales_at = [&](int stage) { return codes_at(stage) + Schedule::kCodeBytes; };
    const auto x_at      = [&](int stage) { return scales_at(stage) + Schedule::kScaleBytes; };
    const auto xs_at     = [&](int stage) { return reinterpret_cast<float*>(x_at(stage) + Schedule::kXBytes); };
    auto* partial        = reinterpret_cast<float*>(s);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * (Schedule::kBlockRows / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin  = token_offset + static_cast<int>(blockIdx.y) * kBlockTokens;
    const int live_columns = min(kBlockTokens, operands.tokens - token_begin);

    nvfp4_a8_fill_scale_table(table, tid, Schedule::kThreads);

    // A warp's K slice of each column: 64 codes (four 16-byte copies) and two slice scales (one 8-byte copy).
    const auto stage_activation = [&](int stage, int group_k0) {
        const int k0 = group_k0 + warp * kTileK;
        for (int item = lane; item < kBlockTokens * 5; item += 32) {
            const int token  = item / 5;
            const int part   = item - token * 5;
            const bool live  = token < live_columns;
            const auto* col  = operands.x + static_cast<std::int64_t>(token_begin + (live ? token : 0)) * operands.x_stride;
            if (part < 4) {
                cp_async_zfill<16>(x_at(stage) + (warp * kBlockTokens + token) * kXRow + part * 16, col + k0 + part * 16,
                                   live ? 16 : 0);
            } else {
                cp_async_zfill<8>(xs_at(stage) + (warp * kBlockTokens + token) * 2, col + kHidden + (k0 / 32) * 4,
                                  live ? 8 : 0);
            }
        }
    };

    constexpr int kCodeChunks  = kBlockK / 32;
    constexpr int kCodeSwizzle = kCodeChunks < 8 ? kCodeChunks - 1 : 7;
    const auto stage_codes     = [&](int stage, int group_k0) {
#pragma unroll
        for (int ri = 0; ri < Schedule::kRowsPerLoaderWarp; ++ri) {
            const int row    = warp * Schedule::kRowsPerLoaderWarp + ri;
            const int parent = row_policy.weight_row(row0, row, operands.rows);
            for (int chunk = lane; chunk < kCodeChunks; chunk += 32) {
                cp_async<16>(codes_at(stage) + row * (kBlockK / 2) + (chunk ^ (row & kCodeSwizzle)) * 16,
                             operands.codes + static_cast<std::int64_t>(parent) * (kHidden / 2) + group_k0 / 2 +
                                 chunk * 16);
            }
            for (int tile = lane; tile < kBlockK / 64; tile += 32) {
                cp_async<4>(scales_at(stage) + row * (kBlockK / 16) + tile * 4,
                            operands.scales + nvfp4_scale_byte_offset(parent, group_k0 / 16 + tile * 4, kHidden));
            }
        }
    };

    const int warp_k0                 = warp * kTileK;
    float accumulators[kTokenMmas][4] = {};

    // The first stages' weights (immutable) while the encode kernel still runs, where this grid is its programmatic
    // dependent (launch_nvfp4_a8_sliced_k_mma: one-column calls under CUDA Graph capture); the encoded columns only after
    // the wait, which returns at once for a serialized launch.
#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kGroups) { stage_codes(stage, stage * kBlockK); }
    }
    pdl::wait_for_dependencies();
#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kGroups) {
            stage_activation(stage, stage * kBlockK);
            cp_commit();
        }
    }
#pragma unroll 1
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int stage = group_index % Schedule::kStages;
        if (group_index + Schedule::kStages - 1 < kGroups)
            cp_wait<Schedule::kStages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
        const unsigned char* codes = codes_at(stage);
        const unsigned char* sc    = scales_at(stage);
        const unsigned char* xw    = x_at(stage) + warp * kBlockTokens * kXRow;
        const float* xsw           = xs_at(stage) + warp * kBlockTokens * 2;
#pragma unroll
        for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
            const auto load_quad = [&](int row) {
                const int byte   = (warp_k0 + k_step * 16) / 2 + lid * 2;
                const int offset = ((byte / 16) ^ (row & kCodeSwizzle)) * 16 + (byte & 15);
                return nvfp4_a8_quad(*reinterpret_cast<const std::uint16_t*>(codes + row * (kBlockK / 2) + offset));
            };
            const unsigned a0    = load_quad(gid);
            const unsigned a1    = load_quad(gid + 8);
            const int block      = (warp_k0 + k_step * 16) / 16;
            const float scale_a  = table[sc[gid * (kBlockK / 16) + block]];
            const float scale_b  = table[sc[(gid + 8) * (kBlockK / 16) + block]];
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const unsigned b0 =
                    *reinterpret_cast<const unsigned*>(xw + (token_mma * 8 + gid) * kXRow + k_step * 16 + lid * 4);
                const float d0 = xsw[(token_mma * 8 + 2 * lid) * 2 + (k_step >> 1)];
                const float d1 = xsw[(token_mma * 8 + 2 * lid + 1) * 2 + (k_step >> 1)];
                nvfp4_a8_block(accumulators[token_mma], a0, a1, b0, scale_a, scale_b, d0, d1);
            }
        }

        __syncthreads();
        const int next = group_index + Schedule::kStages;
        if (next < kGroups) {
            stage_codes(stage, next * kBlockK);
            stage_activation(stage, next * kBlockK);
            cp_commit();
        }
    }
    // The weights are read: the next kernel may launch beside the K-warp sum and the epilogue.
    pdl::trigger_dependents();

    __syncthreads();
    if ((warp & 1) != 0) {
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            store_vec(partial + ((warp * kTokenMmas + token_mma) * 32 + lane) * 4,
                      make_float4(accumulators[token_mma][0], accumulators[token_mma][1], accumulators[token_mma][2],
                                  accumulators[token_mma][3]));
        }
    }
    __syncthreads();
    if ((warp & 1) == 0) {
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            const float4 partner = load_vec<float4>(partial + (((warp + 1) * kTokenMmas + token_mma) * 32 + lane) * 4);
            accumulators[token_mma][0] += partner.x;
            accumulators[token_mma][1] += partner.y;
            accumulators[token_mma][2] += partner.z;
            accumulators[token_mma][3] += partner.w;
            if (warp != 0) {
                store_vec(partial + ((warp * kTokenMmas + token_mma) * 32 + lane) * 4,
                          make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                      accumulators[token_mma][2], accumulators[token_mma][3]));
            }
        }
    }
    __syncthreads();

    if constexpr (kLoraEpilogue<Epilogue>) {
        if (warp == 0) {
            // The reduced sums; gate and up are scaled (one rounded multiply, the SwiGLU's own product) and take the
            // term before the SwiGLU, a single value takes it on the epilogue's result of the kernel's own
            // sum * alpha (the residual's add may fuse the product).
            const auto destination =
                linear_output_tile<Schedule::kBlockRows / (RowPolicy::kPaired ? 2 : 1)>(output, row0);
            const float alpha = operands.alpha;
            float values[1][kTokenMmas][4];
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                float4 sum = make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                         accumulators[token_mma][2], accumulators[token_mma][3]);
#pragma unroll
                for (int split = 2; split < kWarps; split += 2) {
                    const float4 value =
                        load_vec<float4>(partial + ((split * kTokenMmas + token_mma) * 32 + lane) * 4);
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                values[0][token_mma][0] = sum.x;
                values[0][token_mma][1] = sum.y;
                values[0][token_mma][2] = sum.z;
                values[0][token_mma][3] = sum.w;
            }
            const auto row_of   = [&](int, int i) { return row_policy.weight_row(row0, i, operands.rows); };
            const auto token_of = [&](int mn, int j) {
                return mn * 8 + j < live_columns ? token_begin + mn * 8 + j : -1;
            };
            if constexpr (RowPolicy::kPaired) {
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma)
#pragma unroll
                    for (int e = 0; e < 4; ++e) values[0][token_mma][e] = __fmul_rn(values[0][token_mma][e], alpha);
                lora_add_rows_tile<1, kTokenMmas>(epilogue.lora, values, row_of, token_of);
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    const float* v        = values[0][token_mma];
                    const int local_token = token_mma * 8 + 2 * lid;
                    const int token0      = token_begin + local_token;
                    const int row_a       = row_policy.weight_row(row0, gid, operands.rows);
                    if (local_token < live_columns) epilogue.apply_pair(destination, row_a, token0, v[0], v[2]);
                    if (local_token + 1 < live_columns)
                        epilogue.apply_pair(destination, row_a, token0 + 1, v[1], v[3]);
                }
            } else {
                lora_finish_rows_tile<1, kTokenMmas>(epilogue, values, row_of, token_of,
                                                     [&](int, int, float v) { return v * alpha; });
                lora_store_rows_tile<1, kTokenMmas>(destination, values, row_of, token_of);
            }
        }
        return;
    }
    if (warp == 0) {
        const auto destination = linear_output_tile<Schedule::kBlockRows / (RowPolicy::kPaired ? 2 : 1)>(output, row0);
        const float alpha      = operands.alpha;
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            float4 sum = make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                     accumulators[token_mma][2], accumulators[token_mma][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value = load_vec<float4>(partial + ((split * kTokenMmas + token_mma) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int local_token = token_mma * 8 + 2 * lid;
            const int token0      = token_begin + local_token;
            const int row_a       = row_policy.weight_row(row0, gid, operands.rows);
            const int row_b       = row_policy.weight_row(row0, gid + 8, operands.rows);
            if constexpr (RowPolicy::kPaired) {
                if (local_token < live_columns)
                    epilogue.apply_pair(destination, row_a, token0, sum.x * alpha, sum.z * alpha);
                if (local_token + 1 < live_columns)
                    epilogue.apply_pair(destination, row_a, token0 + 1, sum.y * alpha, sum.w * alpha);
            } else {
                if (local_token < live_columns) {
                    destination.store(row_a, token0, epilogue.apply(row_a, token0, sum.x * alpha));
                    destination.store(row_b, token0, epilogue.apply(row_b, token0, sum.z * alpha));
                }
                if (local_token + 1 < live_columns) {
                    destination.store(row_a, token0 + 1, epilogue.apply(row_a, token0 + 1, sum.y * alpha));
                    destination.store(row_b, token0 + 1, epilogue.apply(row_b, token0 + 1, sum.w * alpha));
                }
            }
        }
    }
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a8_sliced_k_mma(const Nvfp4A8Operands& p, Output output, Epilogue epilogue, cudaStream_t stream,
                                  Rows rows = {}) {
    validate_nvfp4_a8_operands(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("NVFP4 A8 sliced-K requires complete row/K tiles");
    static constexpr auto kernel = nvfp4_a8_sliced_k_mma_kernel<Schedule, Output, Epilogue, Rows>;
    const int bytes              = prepare_dynamic_shared<Schedule::kSharedBytes, kernel, true>();
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        // A one-column call (plain decode) launches as the encode's programmatic dependent: its weight prefetch
        // overlaps the encode (RTX 5090, 2026-11-19, the NVIDIA NVFP4-QAD 27B under --a8-activations, ABBA: plain decode
        // +0.9 %, 85.13 against 84.39-84.41 tok/s); wider calls launch serialized, which MTP's verification rounds
        // measured faster (266.9 against 256.4 tok/s, -3.9 % as dependents).
        if (p.tokens == 1) {
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(Schedule::kThreads), static_cast<std::size_t>(bytes), stream},
                                            kernel, p, output, epilogue, rows, offset));
        } else {
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p, output, epilogue, rows, offset);
            CUDA_CHECK(cudaGetLastError());
        }
    });
}

// ---------------------------------------------------------------------------------------------------- MMA

// BlockRows x BlockTokens tiles of 128 values of K: WarpsRows x WarpsTokens warps of (BlockRows / WarpsRows) x
// (BlockTokens / WarpsTokens) outputs. The codes of a K tile are widened to int8 in a padded tile once, the
// activations' codes and scales staged in a two-stage ring.
template <int BlockRows, int BlockTokens, int WarpsRows, int WarpsTokens, int MinBlocksPerSm = 1>
struct Nvfp4A8MmaSchedule {
    static constexpr int kBlockRows      = BlockRows;
    static constexpr int kBlockTokens    = BlockTokens;
    static constexpr int kBlockK         = 128;
    static constexpr int kWarpsRows      = WarpsRows;
    static constexpr int kWarpsTokens    = WarpsTokens;
    static constexpr int kWarps          = WarpsRows * WarpsTokens;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kWarpRows       = BlockRows / WarpsRows;
    static constexpr int kWarpTokens     = BlockTokens / WarpsTokens;
    static constexpr int kMmaRows        = kWarpRows / 16;
    static constexpr int kMmaTokens      = kWarpTokens / 8;
    static constexpr int kRowBytes       = kBlockK + 16; // a widened weight row or an activation row, padded
    static constexpr int kWeightBytes    = BlockRows * kRowBytes;
    static constexpr int kCodeBytes      = BlockRows * (kBlockK / 2);
    static constexpr int kScaleBytes     = BlockRows * (kBlockK / 16);
    static constexpr int kXBytes         = BlockTokens * kRowBytes;
    static constexpr int kXScaleBytes    = BlockTokens * (kBlockK / 32) * 4;
    static constexpr int kStageBytes     = kCodeBytes + kScaleBytes + kXBytes + kXScaleBytes;
    static constexpr int kTableBytes     = 256 * 4;
    static constexpr int kSharedBytes    = kTableBytes + kWeightBytes + 2 * kStageBytes;
    static_assert(BlockRows % (16 * WarpsRows) == 0 && BlockTokens % (8 * WarpsTokens) == 0);
    static_assert(kStageBytes % 16 == 0 && kWeightBytes % 16 == 0);
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class Schedule, bool FullTokens, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a8_mma_kernel(
    Nvfp4A8Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy, int token_offset, int count) {
    constexpr int BM      = Schedule::kBlockRows;
    constexpr int BN      = Schedule::kBlockTokens;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int WM      = Schedule::kWarpRows;
    constexpr int WN      = Schedule::kWarpTokens;
    constexpr int MT      = Schedule::kMmaRows;
    constexpr int NT      = Schedule::kMmaTokens;
    constexpr int THREADS = Schedule::kThreads;
    constexpr int ROW     = Schedule::kRowBytes;
    const int M           = operands.rows;
    const int K           = operands.k;
    const int tokens      = token_offset + count;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* table           = reinterpret_cast<float*>(shared_raw);
    unsigned char* weight = shared_raw + Schedule::kTableBytes;
    unsigned char* ring   = weight + Schedule::kWeightBytes;
    const auto codes_at   = [&](int stage) { return ring + stage * Schedule::kStageBytes; };
    const auto scales_at  = [&](int stage) { return codes_at(stage) + Schedule::kCodeBytes; };
    const auto x_at       = [&](int stage) { return scales_at(stage) + Schedule::kScaleBytes; };
    const auto xs_at      = [&](int stage) { return reinterpret_cast<float*>(x_at(stage) + Schedule::kXBytes); };

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Schedule::kWarpsTokens;
    const int wn   = warp - wm * Schedule::kWarpsTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int row_begin   = static_cast<int>(blockIdx.x) * (BM / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + static_cast<int>(blockIdx.y) * BN;

    nvfp4_a8_fill_scale_table(table, tid, THREADS);

    const auto stage = [&](int s, int k_tile) {
        const int k0 = k_tile * BK;
        for (int item = tid; item < BM * (BK / 32); item += THREADS) {
            const int row = item / (BK / 32), chunk = item % (BK / 32);
            const int parent = row_policy.weight_row(row_begin, row, M);
            cp_async<16>(codes_at(s) + row * (BK / 2) + chunk * 16,
                         operands.codes + static_cast<std::int64_t>(parent) * (K / 2) + k0 / 2 + chunk * 16);
        }
        for (int item = tid; item < BM * (BK / 64); item += THREADS) {
            const int row = item / (BK / 64), tile = item % (BK / 64);
            const int parent = row_policy.weight_row(row_begin, row, M);
            cp_async<4>(scales_at(s) + row * (BK / 16) + tile * 4,
                        operands.scales + nvfp4_scale_byte_offset(parent, k0 / 16 + tile * 4, K));
        }
        // A column's 128 codes (eight 16-byte copies) and four slice scales (one 16-byte copy).
        for (int item = tid; item < BN * 9; item += THREADS) {
            const int local = item / 9, part = item - local * 9;
            const int token = token_begin + local;
            const bool live = FullTokens || token < tokens;
            const auto* col = operands.x + static_cast<std::int64_t>(live ? token : 0) * operands.x_stride;
            if (part < 8) {
                cp_async_zfill<16>(x_at(s) + local * ROW + part * 16, col + k0 + part * 16, live ? 16 : 0);
            } else {
                cp_async_zfill<16>(xs_at(s) + local * (BK / 32), col + K + (k0 / 32) * 4, live ? 16 : 0);
            }
        }
    };
    // The K tile's codes as int8 doubled values, one quad (four elements) an item.
    const auto widen = [&](int s) {
        for (int item = tid; item < BM * (BK / 4); item += THREADS) {
            const int row = item / (BK / 4), quad = item % (BK / 4);
            const auto pair =
                *reinterpret_cast<const std::uint16_t*>(codes_at(s) + row * (BK / 2) + quad * 2);
            *reinterpret_cast<unsigned*>(weight + row * ROW + quad * 4) = nvfp4_a8_quad(pair);
        }
    };

    float accumulators[MT][NT][4] = {};
    const int kTiles              = K / BK;
    stage(0, 0);
    cp_commit();
#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int s = k_tile & 1;
        cp_wait<0>();
        __syncthreads();
        widen(s);
        if (k_tile + 1 < kTiles) {
            stage(s ^ 1, k_tile + 1);
            cp_commit();
        }
        __syncthreads();
        const unsigned char* sc = scales_at(s);
        const unsigned char* xr = x_at(s);
        const float* xs         = xs_at(s);
#pragma unroll
        for (int k_step = 0; k_step < BK / 16; ++k_step) {
            unsigned a[MT][2];
            float sa[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + gid;
                a[mi][0]      = *reinterpret_cast<const unsigned*>(weight + row * ROW + k_step * 16 + lid * 4);
                a[mi][1]      = *reinterpret_cast<const unsigned*>(weight + (row + 8) * ROW + k_step * 16 + lid * 4);
                sa[mi][0]     = table[sc[row * (BK / 16) + k_step]];
                sa[mi][1]     = table[sc[(row + 8) * (BK / 16) + k_step]];
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int local   = wn * WN + ni * 8;
                const unsigned b0 = *reinterpret_cast<const unsigned*>(xr + (local + gid) * ROW + k_step * 16 + lid * 4);
                const float d0    = xs[(local + 2 * lid) * (BK / 32) + (k_step >> 1)];
                const float d1    = xs[(local + 2 * lid + 1) * (BK / 32) + (k_step >> 1)];
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    nvfp4_a8_block(accumulators[mi][ni], a[mi][0], a[mi][1], b0, sa[mi][0], sa[mi][1], d0, d1);
                }
            }
        }
    }

    const auto destination = linear_output_tile<BM / (RowPolicy::kPaired ? 2 : 1)>(output, row_begin);
    const float alpha      = operands.alpha;
    if constexpr (kLoraEpilogue<Epilogue>) {
        const auto row_of = [&](int mi, int i) { return row_policy.weight_row(row_begin, wm * WM + mi * 16 + i, M); };
        const auto token_of = [&](int ni, int j) {
            const int token = token_begin + wn * WN + ni * 8 + j;
            return token < tokens ? token : -1;
        };
        if constexpr (requires { RowPolicy::kThreadPaired; }) {
            // Gate and up: the scaled values (one rounded multiply, the SwiGLU's own product), the term, the SwiGLU.
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int ni = 0; ni < NT; ++ni)
#pragma unroll
                    for (int e = 0; e < 4; ++e) accumulators[mi][ni][e] = __fmul_rn(accumulators[mi][ni][e], alpha);
            lora_add_rows_tile<MT, NT>(epilogue.lora, accumulators, row_of, token_of);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int out_row = row_policy.output_row(row_begin, wm * WM + mi * 16 + gid);
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int token0 = token_begin + wn * WN + ni * 8 + 2 * lid;
                    const auto& v    = accumulators[mi][ni];
                    if (FullTokens || token0 < tokens) epilogue.apply_pair(destination, out_row, token0, v[0], v[2]);
                    if (FullTokens || token0 + 1 < tokens)
                        epilogue.apply_pair(destination, out_row, token0 + 1, v[1], v[3]);
                }
            }
        } else {
            static_assert(!RowPolicy::kPaired, "paired rows require the thread-paired MMA rows");
            // The epilogue on the kernel's own v * alpha (its residual add may fuse the product), then the term.
            lora_finish_rows_tile<MT, NT>(epilogue, accumulators, row_of, token_of,
                                          [&](int, int, float v) { return v * alpha; });
            lora_store_rows_tile<MT, NT>(destination, accumulators, row_of, token_of);
        }
        return;
    }
    if constexpr (requires { RowPolicy::kThreadPaired; }) {
        // Each m16 tile pairs C-fragment rows g (gate) and g+8 (up) of one output row.
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int out_row = row_policy.output_row(row_begin, wm * WM + mi * 16 + gid);
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int token0 = token_begin + wn * WN + ni * 8 + 2 * lid;
                const auto& v    = accumulators[mi][ni];
                if (FullTokens || token0 < tokens) {
                    epilogue.apply_pair(destination, out_row, token0, v[0] * alpha, v[2] * alpha);
                }
                if (FullTokens || token0 + 1 < tokens) {
                    epilogue.apply_pair(destination, out_row, token0 + 1, v[1] * alpha, v[3] * alpha);
                }
            }
        }
    } else {
        static_assert(!RowPolicy::kPaired, "paired rows require the thread-paired MMA rows");
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row0 = row_policy.weight_row(row_begin, wm * WM + mi * 16 + gid, M);
            const int row1 = row_policy.weight_row(row_begin, wm * WM + mi * 16 + gid + 8, M);
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int token0 = token_begin + wn * WN + ni * 8 + 2 * lid;
                const auto& v    = accumulators[mi][ni];
                if (FullTokens || token0 < tokens) {
                    destination.store(row0, token0, epilogue.apply(row0, token0, v[0] * alpha));
                    destination.store(row1, token0, epilogue.apply(row1, token0, v[2] * alpha));
                }
                if (FullTokens || token0 + 1 < tokens) {
                    destination.store(row0, token0 + 1, epilogue.apply(row0, token0 + 1, v[1] * alpha));
                    destination.store(row1, token0 + 1, epilogue.apply(row1, token0 + 1, v[3] * alpha));
                }
            }
        }
    }
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a8_mma(const Nvfp4A8Operands& p, Output output, Epilogue epilogue, cudaStream_t stream,
                         Rows rows = {}) {
    validate_nvfp4_a8_operands(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("NVFP4 A8 MMA requires complete row/K tiles");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        const auto launch = [&]<bool Full>() {
            static constexpr auto kernel = nvfp4_a8_mma_kernel<Schedule, Full, Output, Epilogue, Rows>;
            const int bytes              = prepare_dynamic_shared<Schedule::kSharedBytes, kernel, true>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p, output, epilogue, rows, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

// ---------------------------------------------------------------------------------------------------- routes

// x's encoded columns in `workspace` (a8_encode_columns) and the operands over them; the caller's workspace
// scope holds them until the projection's launch.
inline Nvfp4A8Operands nvfp4_a8_encode(const Tensor& x, const Weight& w, WorkspaceArena& workspace,
                                       cudaStream_t stream) {
    const std::int32_t words = a8_linear_column_words(x.ne[0], false);
    Tensor encoded           = workspace.alloc(DType::I32, {words, x.ne[1]}, 256);
    a8_encode_columns(x, encoded, stream);
    return nvfp4_a8_operands(encoded.data, words * 4, w, x.ne[1]);
}

// Every NVFP4 projection's 8-bit tiles by width (the families pass their output, epilogue and row policies: the
// sliced-K tiles' and the MMA's, as their A16 tiles take them): the sliced-K tiles through 64 columns (8 K-warps
// through 16, 4 above), the 128 x 128 tiles on the decode-warp ring wider (nvfp4_a8_ring.cuh, through each family's
// launch_nvfp4_a8_ring; the 5,120-row projections: 64-row MMA tiles through 256 columns).
// `NarrowOutput`: a 5,120-row projection (linear_add: the MLP down, attention and GDN outputs), whose 40 row tiles of
// 128 leave most SMs idle through 256 columns: there 64-row tiles. RTX 5090, 2026-11-19, [5120,17408] and [5120,6144],
// two rounds: 65-128 columns 64 x 64 took 0.46-0.54 of the 128 x 128 tiles' time, 192-256 columns 64 x 128 (2 x 4
// warps) 0.58-0.61 (1 x 4 warps 0.62-0.65); from 320 columns the 128 x 128 tiles took 0.80-0.89 of 64 x 128's.
template <class Output, class Epilogue, class SlicedRows = Nvfp4IdentityRows, class MmaRows = Nvfp4IdentityRows,
          bool NarrowOutput = false>
void launch_nvfp4_a8(const Nvfp4A8Operands& p, Output output, Epilogue epilogue, cudaStream_t stream,
                     SlicedRows sliced_rows = {}, MmaRows mma_rows = {}) {
    const int tokens = p.tokens;
    if (tokens <= 8)
        return launch_nvfp4_a8_sliced_k_mma<Nvfp4A8SlicedSchedule<8, 8, 2>>(p, output, epilogue, stream, sliced_rows);
    if (tokens <= 16)
        return launch_nvfp4_a8_sliced_k_mma<Nvfp4A8SlicedSchedule<8, 16, 2>>(p, output, epilogue, stream, sliced_rows);
    if (tokens <= 32)
        return launch_nvfp4_a8_sliced_k_mma<Nvfp4A8SlicedSchedule<4, 32, 2>>(p, output, epilogue, stream, sliced_rows);
    if (tokens <= 64)
        return launch_nvfp4_a8_sliced_k_mma<Nvfp4A8SlicedSchedule<4, 64, 1>>(p, output, epilogue, stream, sliced_rows);
    if constexpr (NarrowOutput) {
        if (tokens <= 128)
            return launch_nvfp4_a8_mma<Nvfp4A8MmaSchedule<64, 64, 2, 2>>(p, output, epilogue, stream, mma_rows);
        if (tokens <= 256)
            return launch_nvfp4_a8_mma<Nvfp4A8MmaSchedule<64, 128, 2, 4>>(p, output, epilogue, stream, mma_rows);
    }
    launch_nvfp4_a8_ring(p, output, epilogue, mma_rows, stream);
}

} // namespace infernix::ops::detail
