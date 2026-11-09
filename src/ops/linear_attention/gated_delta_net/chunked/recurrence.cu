#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"

#include <cuda_fp16.h>

#include <type_traits>

namespace infernix::ops::detail::gated_delta_net::chunked {
namespace {

template <int DV>
inline constexpr int kThreads = DV >= 32 ? 256 : 128;

template <int Width>
__device__ __forceinline__ int value_index(int row, int col) {
    if constexpr (Width == 8) return row * Width + col;
    if constexpr (Width == 16) return row * Width + (col ^ (((row >> 1) & 1) * 8));
    return row * Width + (col ^ ((row & 3) * 8));
}

__device__ __forceinline__ void load_vector_tf32_a(const __nv_bfloat16* p, int k, int lane,
                                                   unsigned (&a)[4]) {
    unsigned low, high;
    ldmatrix_x2(low, high, smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k)));
    unpack_bf16x2_to_fp32_bits(low, a[0], a[2]);
    unpack_bf16x2_to_fp32_bits(high, a[1], a[3]);
}

// Q/K retain their native BF16 coordinates. TF32 A expands adjacent pairs;
// its FP32 B loads follow the corresponding even/odd inner-coordinate order.
__device__ __forceinline__ void load_vector_bf16_a(const __nv_bfloat16* p, int k, int lane,
                                                   unsigned (&a)[4]) {
    ldmatrix_x4(
        a[0], a[1], a[2], a[3],
        smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k + (lane >> 4) * 8)));
}

__device__ __forceinline__ void load_transposed_bf16_a(const __nv_bfloat16* p, int key, int lane,
                                                       unsigned (&a)[4]) {
    ldmatrix_x4_t(
        a[0], a[1], a[2], a[3],
        smem_addr(p + vector_index((lane & 7) + (lane >> 4) * 8, key + ((lane >> 3) & 1) * 8)));
}

template <int DV>
__device__ __forceinline__ void load_bf16_b(const float* p, int k, int n, int lane, unsigned& b0,
                                            unsigned& b1) {
    const int t = lane & 3, col = n * 8 + lane / 4;
    b0 = pack_bf16x2(p[value_index<DV>(k + 2 * t, col)], p[value_index<DV>(k + 2 * t + 1, col)]);
    b1 =
        pack_bf16x2(p[value_index<DV>(k + 2 * t + 8, col)], p[value_index<DV>(k + 2 * t + 9, col)]);
}

// GdnChunkArithmetic::Compensated's helpers (after ninferpro dbf36e6): an FP32 operand as a high/low pair whose
// products sum to it at about twice the high part's precision.

// TF32: the high part keeps the bits TF32 MMA reads; the low part is the exact remainder (read as TF32).
__device__ __forceinline__ void split_tf32(float x, unsigned& high, unsigned& low) {
    high = __float_as_uint(x) & 0xffffe000U;
    low  = __float_as_uint(x - __uint_as_float(high));
}

// BF16 pairs of two adjacent elements (a in the low half): round to nearest, then the remainder.
__device__ __forceinline__ void split_bf16x2(float a, float b, unsigned& high, unsigned& low) {
    high = pack_bf16x2(a, b);
    low  = pack_bf16x2(a - __uint_as_float(high << 16), b - __uint_as_float(high & 0xffff0000U));
}

// An exact A (raw BF16 Q/K) against a B pair: A*B_lo first, then A*B_hi.
__device__ __forceinline__ void mma_bf16_pair_b(float (&acc)[4], const unsigned (&a)[4], const unsigned (&high)[2],
                                                const unsigned (&low)[2]) {
    mma_bf16(acc[0], acc[1], acc[2], acc[3], a[0], a[1], a[2], a[3], low[0], low[1]);
    mma_bf16(acc[0], acc[1], acc[2], acc[3], a[0], a[1], a[2], a[3], high[0], high[1]);
}

__device__ __forceinline__ void mma_tf32_pair_b(float (&acc)[4], const unsigned (&a)[4], float b0, float b1) {
    unsigned h0, l0, h1, l1;
    split_tf32(b0, h0, l0);
    split_tf32(b1, h1, l1);
    mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], a[0], a[1], a[2], a[3], l0, l1);
    mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], a[0], a[1], a[2], a[3], h0, h1);
}

// Both operands as TF32 pairs: hi*hi + hi*lo + lo*hi, the small products first.
__device__ __forceinline__ void mma_tf32_compensated(float (&acc)[4], const unsigned (&a)[4], float b0, float b1) {
    unsigned ah[4], al[4], h0, l0, h1, l1;
#pragma unroll
    for (int i = 0; i < 4; ++i) split_tf32(__uint_as_float(a[i]), ah[i], al[i]);
    split_tf32(b0, h0, l0);
    split_tf32(b1, h1, l1);
    mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], al[0], al[1], al[2], al[3], h0, h1);
    mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], ah[0], ah[1], ah[2], ah[3], l0, l1);
    mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], ah[0], ah[1], ah[2], ah[3], h0, h1);
}

// GdnChunkArithmetic::Fp16Accumulate's helpers. Its packet holds FP16 Q/K; the 16-bit ldmatrix
// loads above are type-agnostic.

// The FP32 state and delta operands as FP16, saturating: in range the bits of pack_f16x2 (round to
// nearest even), beyond +-65504 the largest finite value instead of an infinity, which would turn
// the products to NaN and persist in the FP32 state for the rest of the sequence.
__device__ __forceinline__ std::uint32_t pack_f16x2_sat(float lo, float hi) {
    std::uint32_t out;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;\n" : "=r"(out) : "f"(hi), "f"(lo));
    return out;
}

template <int DV>
__device__ __forceinline__ void load_f16_b(const float* p, int k, int n, int lane, unsigned& b0,
                                           unsigned& b1) {
    const int t = lane & 3, col = n * 8 + lane / 4;
    b0 = pack_f16x2_sat(p[value_index<DV>(k + 2 * t, col)], p[value_index<DV>(k + 2 * t + 1, col)]);
    b1 =
        pack_f16x2_sat(p[value_index<DV>(k + 2 * t + 8, col)], p[value_index<DV>(k + 2 * t + 9, col)]);
}

// The k16 products an FP16 partial sum spans before its FP32 carry (measured 2026-10-16 against
// one: 1.205x the exact route's op time at the 27B's shapes for both, and the more accurate state).
inline constexpr int kFp16Carry = 2;

// One k16 step of an FP16 partial sum; every kFp16Carry steps the partial (row g's columns
// {2t, 2t+1} in part[0], row g+8's in part[1]) is added to the FP32 accumulator in its C-fragment
// order, and the next step starts from zero. The FP16 rounding spans at most 16 * kFp16Carry
// products.
__device__ __forceinline__ void mma_f16_partial(float (&acc)[4], unsigned (&part)[2], int step,
                                                const unsigned (&a)[4], unsigned b0, unsigned b1) {
    if (step % kFp16Carry == 0)
        mma_f16_f16acc_zero(part[0], part[1], a[0], a[1], a[2], a[3], b0, b1);
    else
        mma_f16_f16acc(part[0], part[1], a[0], a[1], a[2], a[3], b0, b1);
    if (step % kFp16Carry == kFp16Carry - 1) {
        const float2 low  = __half22float2(half2_from_bits(part[0]));
        const float2 high = __half22float2(half2_from_bits(part[1]));
        acc[0] += low.x;
        acc[1] += low.y;
        acc[2] += high.x;
        acc[3] += high.y;
    }
}

template <int DV>
struct RecurrenceShared {
    struct alignas(256) Stage {
        QkChunk qk;
        ControlChunk control;
        __nv_bfloat16 value[kChunkSize * DV];
    } stage[2];

    float snapshot[kStateDim * DV];
    float delta[kChunkSize * DV];
};

template <int DV>
__device__ __forceinline__ void load_stage(typename RecurrenceShared<DV>::Stage& stage,
                                           const Arguments& args, const QkChunk* qk_workspace,
                                           const ControlChunk* control_workspace, int head,
                                           int value_start, int chunk, int tid) {
    const int qh       = head / (args.value_heads / args.qk_heads);
    const int chunks   = chunk_count(args.tokens);
    const auto* qk_src = reinterpret_cast<const uint4*>(
        qk_workspace + static_cast<std::int64_t>(qh) * chunks + chunk);
    auto* qk_dst = reinterpret_cast<uint4*>(&stage.qk);
#pragma unroll
    for (int i = tid; i < sizeof(QkChunk) / 16; i += kThreads<DV>)
        cp_async<16, Cache::cg>(qk_dst + i, qk_src + i);
    const auto* ctrl_src = reinterpret_cast<const uint4*>(
        control_workspace + static_cast<std::int64_t>(head) * chunks + chunk);
    auto* ctrl_dst = reinterpret_cast<uint4*>(&stage.control);
#pragma unroll
    for (int i = tid; i < sizeof(ControlChunk) / 16; i += kThreads<DV>)
        cp_async<16, Cache::cg>(ctrl_dst + i, ctrl_src + i);
    constexpr int vectors = kChunkSize * DV / 8;
#pragma unroll
    for (int i = tid; i < vectors; i += kThreads<DV>) {
        const int row = i / (DV / 8), col = i % (DV / 8) * 8;
        const int token       = chunk * kChunkSize + row;
        const int valid_token = token < args.tokens ? token : 0;
        const auto off =
            (static_cast<std::int64_t>(valid_token) * args.value_heads + head) * kStateDim +
            value_start + col;
        cp_async_zfill<16, Cache::cg>(stage.value + row * DV + col, args.v + off,
                                      token < args.tokens ? 16 : 0);
    }
    cp_commit();
}

// `key_norm`: each token's inverse K norm (Compensated's raw packet), null elsewhere.
template <int DV>
__device__ __forceinline__ void store_residual(float* delta, const __nv_bfloat16* value,
                                               const float (&prediction)[4], const float* prefix,
                                               int n, int lane, const float* key_norm = nullptr) {
    const int row = lane / 4, col = n * 8 + (lane & 3) * 2;
    const float s0 = key_norm != nullptr ? prefix[row] * key_norm[row] : prefix[row];
    const float s1 = key_norm != nullptr ? prefix[row + 8] * key_norm[row + 8] : prefix[row + 8];
    const float2 v0 = __bfloat1622float2(load_vec<__nv_bfloat162>(value + row * DV + col));
    const float2 v1 = __bfloat1622float2(load_vec<__nv_bfloat162>(value + (row + 8) * DV + col));
    store_vec(delta + value_index<DV>(row, col),
              make_float2(v0.x - s0 * prediction[0], v0.y - s0 * prediction[1]));
    store_vec(delta + value_index<DV>(row + 8, col),
              make_float2(v1.x - s1 * prediction[2], v1.y - s1 * prediction[3]));
}

template <int DV>
__device__ __forceinline__ void matrix_delta(float (&acc)[4], const float* matrix,
                                             const float* delta, int n, int lane, bool compensated = false) {
    const int g = lane / 4, t = lane & 3;
#pragma unroll
    for (int k = 0; k < kChunkSize; k += 8) {
        unsigned af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3],
                    smem_addr(matrix + square_index((lane & 7) + ((lane >> 3) & 1) * 8,
                                                    k + (lane >> 4) * 4)));
        if (compensated) {
            mma_tf32_compensated(acc, af, delta[value_index<DV>(k + t, n * 8 + g)],
                                 delta[value_index<DV>(k + t + 4, n * 8 + g)]);
        } else {
            mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3],
                          __float_as_uint(delta[value_index<DV>(k + t, n * 8 + g)]),
                          __float_as_uint(delta[value_index<DV>(k + t + 4, n * 8 + g)]));
        }
    }
}

__device__ __forceinline__ void store_output(const Arguments& args, float (&acc)[4], int chunk,
                                             int head, int value_start, int n, int lane) {
    const int row = lane / 4, col = value_start + n * 8 + (lane & 3) * 2;
    const int t0  = chunk * kChunkSize + row;
    const auto o0 = (static_cast<std::int64_t>(t0) * args.value_heads + head) * kStateDim + col;
    if (t0 < args.tokens) {
        store_vec(args.out + o0, __floats2bfloat162_rn(acc[0] * args.scale, acc[1] * args.scale));
    }
    if (t0 + 8 < args.tokens) {
        store_vec(args.out + o0 + static_cast<std::int64_t>(8) * args.value_heads * kStateDim,
                  __floats2bfloat162_rn(acc[2] * args.scale, acc[3] * args.scale));
    }
}

// fp16-gdn-state: the recurrent kernels' clamp (gated_delta_net/common.cuh), so a stored state
// saturates at FP16's largest finite magnitude instead of overflowing.
__device__ __forceinline__ float saturate_fp16_state(float value) {
    constexpr float kMaximum = 65504.0f;
    return value == value ? fminf(fmaxf(value, -kMaximum), kMaximum) : value;
}

__device__ __forceinline__ float stored_state(float value) { return value; }
__device__ __forceinline__ float stored_state(__half value) { return __half2float(value); }

template <class State>
__device__ __forceinline__ State state_to_store(float value) {
    if constexpr (std::is_same_v<State, float>) {
        return value;
    } else {
        return __float2half_rn(saturate_fp16_state(value));
    }
}

// Fp16 selects GdnChunkArithmetic::Fp16Accumulate (launch.h); without it GdnChunkArithmetic::Compensated. State is
// the stored state's element type (float, or __half under fp16-gdn-state).
template <int DV, bool Fp16 = false, class State = float>
__global__ __launch_bounds__(kThreads<DV>, (DV >= 32 ? 2 : 4)) void chunk_recurrence_kernel(
    Arguments args, const QkChunk* __restrict__ qk_workspace,
    const ControlChunk* __restrict__ control_workspace) {
    // GdnChunkArithmetic::Compensated (without Fp16): the raw packet with its norms, and the prediction, control-matrix
    // and update operand pairs (launch.h).
    [[maybe_unused]] constexpr bool raw_packet = !Fp16;
    const State* __restrict__ state_in = static_cast<const State*>(args.state_in);
    State* __restrict__ state_out      = static_cast<State*>(args.state_out);
    constexpr int kWarps = kThreads<DV> / 32;
    constexpr int NT     = DV / 8;
    constexpr int MW     = NT < kWarps ? kWarps / NT : 1;
    constexpr int MPW    = (kStateDim / 16) / MW;
    constexpr int NPW    = NT > kWarps ? NT / kWarps : 1;
    extern __shared__ __align__(256) unsigned char backing[];
    auto& sm      = *reinterpret_cast<RecurrenceShared<DV>*>(backing);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int g = lane / 4, t = lane & 3;
    const int head        = blockIdx.x / (kStateDim / DV);
    const int value_start = blockIdx.x % (kStateDim / DV) * DV;
    const int n_start     = warp % NT;
    const int m_start     = NT < kWarps ? warp / NT : 0;
    const int chunks      = chunk_count(args.tokens);
    const auto state_base = static_cast<std::int64_t>(head) * kStateDim * kStateDim;

    // MMA accumulators describe S^T[key,value]. External state remains [value,key].
    float state[NPW][MPW][4];
#pragma unroll
    for (int ni = 0; ni < NPW; ++ni) {
        const int n = n_start + ni * kWarps;
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key    = (m_start + mi * MW) * 16 + g;
            const int value  = value_start + n * 8 + t * 2;
            state[ni][mi][0] = stored_state(state_in[state_base + value * kStateDim + key]);
            state[ni][mi][1] = stored_state(state_in[state_base + (value + 1) * kStateDim + key]);
            state[ni][mi][2] = stored_state(state_in[state_base + value * kStateDim + key + 8]);
            state[ni][mi][3] = stored_state(state_in[state_base + (value + 1) * kStateDim + key + 8]);
        }
    }
    load_stage<DV>(sm.stage[0], args, qk_workspace, control_workspace, head, value_start, 0, tid);
    if (chunks > 1)
        load_stage<DV>(sm.stage[1], args, qk_workspace, control_workspace, head, value_start, 1,
                       tid);

    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (chunk + 1 < chunks)
            cp_wait<1>();
        else
            cp_wait<0>();
#pragma unroll
        for (int ni = 0; ni < NPW; ++ni) {
            const int n = n_start + ni * kWarps;
#pragma unroll
            for (int mi = 0; mi < MPW; ++mi) {
                const int key = (m_start + mi * MW) * 16 + g;
                const int col = n * 8 + t * 2;
                store_vec(sm.snapshot + value_index<DV>(key, col),
                          make_float2(state[ni][mi][0], state[ni][mi][1]));
                store_vec(sm.snapshot + value_index<DV>(key + 8, col),
                          make_float2(state[ni][mi][2], state[ni][mi][3]));
            }
        }
        __syncthreads();
        const auto& stage     = sm.stage[chunk & 1];
        const ControlChunk& p = stage.control;
        const QkChunk& qk     = stage.qk;
        float history[NPW][4]{};

        if constexpr (NT >= kWarps) {
            float prediction[NPW][4]{};
            if constexpr (!Fp16) {
#pragma unroll
                for (int k = 0; k < kStateDim; k += 16) {
                    unsigned ka0[4], ka1[4], qa[4];
                    load_vector_tf32_a(qk.k, k, lane, ka0);
                    load_vector_tf32_a(qk.k, k + 8, lane, ka1);
                    load_vector_bf16_a(qk.q, k, lane, qa);
#pragma unroll
                    for (int ni = 0; ni < NPW; ++ni) {
                        const int n    = warp + ni * kWarps;
                        const float b0 = sm.snapshot[value_index<DV>(k + 2 * t, n * 8 + g)];
                        const float b1 = sm.snapshot[value_index<DV>(k + 2 * t + 1, n * 8 + g)];
                        const float b2 = sm.snapshot[value_index<DV>(k + 2 * t + 8, n * 8 + g)];
                        const float b3 = sm.snapshot[value_index<DV>(k + 2 * t + 9, n * 8 + g)];
                        // The prediction takes S as a TF32 pair, the readout S rounded to BF16.
                        mma_tf32_pair_b(prediction[ni], ka0, b0, b1);
                        mma_tf32_pair_b(prediction[ni], ka1, b2, b3);
                        mma_bf16(history[ni][0], history[ni][1], history[ni][2], history[ni][3], qa[0], qa[1],
                                 qa[2], qa[3], pack_bf16x2(b0, b1), pack_bf16x2(b2, b3));
                    }
                }
            } else {
                // S is rounded to FP16 once per element for both products.
                unsigned prediction_part[NPW][2]{}, history_part[NPW][2]{};
#pragma unroll
                for (int k = 0; k < kStateDim; k += 16) {
                    unsigned ka[4], qa[4];
                    load_vector_bf16_a(qk.k, k, lane, ka);
                    load_vector_bf16_a(qk.q, k, lane, qa);
#pragma unroll
                    for (int ni = 0; ni < NPW; ++ni) {
                        const int n    = warp + ni * kWarps;
                        const float b0 = sm.snapshot[value_index<DV>(k + 2 * t, n * 8 + g)];
                        const float b1 = sm.snapshot[value_index<DV>(k + 2 * t + 1, n * 8 + g)];
                        const float b2 = sm.snapshot[value_index<DV>(k + 2 * t + 8, n * 8 + g)];
                        const float b3 = sm.snapshot[value_index<DV>(k + 2 * t + 9, n * 8 + g)];
                        const unsigned s0 = pack_f16x2_sat(b0, b1), s1 = pack_f16x2_sat(b2, b3);
                        mma_f16_partial(prediction[ni], prediction_part[ni], k / 16, ka, s0, s1);
                        mma_f16_partial(history[ni], history_part[ni], k / 16, qa, s0, s1);
                    }
                }
            }
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                store_residual<DV>(sm.delta, stage.value, prediction[ni], p.prefix,
                                   warp + ni * kWarps, lane, raw_packet ? p.qk_norm + kChunkSize : nullptr);
            }
        } else {
            if (warp < 2 * NT) {
                const bool query = warp >= NT;
                const int n      = warp % NT;
                if (query) {
                    if constexpr (!Fp16) {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 16) {
                            unsigned af[4];
                            load_vector_bf16_a(qk.q, k, lane, af);
                            unsigned b0, b1;
                            load_bf16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                            mma_bf16(history[0][0], history[0][1], history[0][2], history[0][3], af[0], af[1],
                                     af[2], af[3], b0, b1);
                        }
                    } else {
                        unsigned part[2]{};
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 16) {
                            unsigned af[4], b0, b1;
                            load_vector_bf16_a(qk.q, k, lane, af);
                            load_f16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                            mma_f16_partial(history[0], part, k / 16, af, b0, b1);
                        }
                    }
                } else {
                    if constexpr (Fp16) {
                        unsigned part[2]{};
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 16) {
                            unsigned af[4], b0, b1;
                            load_vector_bf16_a(qk.k, k, lane, af);
                            load_f16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                            mma_f16_partial(history[0], part, k / 16, af, b0, b1);
                        }
                    } else {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 8) {
                            unsigned af[4];
                            load_vector_tf32_a(qk.k, k, lane, af);
                            const float s0 = sm.snapshot[value_index<DV>(k + 2 * t, n * 8 + g)];
                            const float s1 = sm.snapshot[value_index<DV>(k + 2 * t + 1, n * 8 + g)];
                            mma_tf32_pair_b(history[0], af, s0, s1);
                        }
                    }
                    store_residual<DV>(sm.delta, stage.value, history[0], p.prefix, n, lane,
                                       raw_packet ? p.qk_norm + kChunkSize : nullptr);
                }
            }
        }
        __syncthreads();

        // Each value tile reads only its own R columns, so R can become Delta in place.
#pragma unroll
        for (int n = warp; n < NT; n += kWarps) {
            float delta[4]{};
            matrix_delta<DV>(delta, p.solve, sm.delta, n, lane, !Fp16);
            const int col = n * 8 + t * 2;
            __syncwarp();
            store_vec(sm.delta + value_index<DV>(g, col), make_float2(delta[0], delta[1]));
            store_vec(sm.delta + value_index<DV>(g + 8, col), make_float2(delta[2], delta[3]));
        }
        __syncthreads();

        if constexpr (NT >= kWarps) {
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int n = warp + ni * kWarps;
                // A raw packet's readout rows take their inverse Q norms.
                const float r0 = raw_packet ? p.prefix[g] * p.qk_norm[g] : p.prefix[g];
                const float r1 = raw_packet ? p.prefix[g + 8] * p.qk_norm[g + 8] : p.prefix[g + 8];
                history[ni][0] *= r0;
                history[ni][1] *= r0;
                history[ni][2] *= r1;
                history[ni][3] *= r1;
                matrix_delta<DV>(history[ni], p.mqk, sm.delta, n, lane, !Fp16);
                store_output(args, history[ni], chunk, head, value_start, n, lane);
            }
        } else if (warp >= NT && warp < 2 * NT) {
            const int n = warp - NT;
            const float r0 = raw_packet ? p.prefix[g] * p.qk_norm[g] : p.prefix[g];
            const float r1 = raw_packet ? p.prefix[g + 8] * p.qk_norm[g + 8] : p.prefix[g + 8];
            history[0][0] *= r0;
            history[0][1] *= r0;
            history[0][2] *= r1;
            history[0][3] *= r1;
            matrix_delta<DV>(history[0], p.mqk, sm.delta, n, lane, !Fp16);
            store_output(args, history[0], chunk, head, value_start, n, lane);
        }

        if constexpr (!Fp16) {
            // S^T = gamma S^T + K^T diag(suffix) Delta. The master state stays FP32.
            // Delta, scaled by the suffix and each token's inverse K norm (the raw packet), as a BF16 pair for this
            // MMA (two against the exact raw K); its fragments serve every key tile. The output branch above
            // consumes the FP32 Delta through the TF32 pairs.
            const auto scale = [&](int i) { return p.suffix[i] * p.qk_norm[kChunkSize + i]; };
            unsigned delta_b[NPW][2], delta_l[NPW][2];
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int col = (n_start + ni * kWarps) * 8 + g;
                const float d0 = scale(2 * t) * sm.delta[value_index<DV>(2 * t, col)];
                const float d1 = scale(2 * t + 1) * sm.delta[value_index<DV>(2 * t + 1, col)];
                const float d2 = scale(2 * t + 8) * sm.delta[value_index<DV>(2 * t + 8, col)];
                const float d3 = scale(2 * t + 9) * sm.delta[value_index<DV>(2 * t + 9, col)];
                split_bf16x2(d0, d1, delta_b[ni][0], delta_l[ni][0]);
                split_bf16x2(d2, d3, delta_b[ni][1], delta_l[ni][1]);
            }
#pragma unroll
            for (int mi = 0; mi < MPW; ++mi) {
                const int key     = (m_start + mi * MW) * 16 + g;
                const float decay = p.prefix[kChunkSize - 1];
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    state[ni][mi][0] *= decay;
                    state[ni][mi][1] *= decay;
                    state[ni][mi][2] *= decay;
                    state[ni][mi][3] *= decay;
                }
                unsigned af[4];
                load_transposed_bf16_a(qk.k, key - g, lane, af);
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) { mma_bf16_pair_b(state[ni][mi], af, delta_b[ni], delta_l[ni]); }
            }
        } else {
            // The same update with FP16 operands. Each tile's 16-token product is rounded to FP16
            // once, from zero, and carried into the FP32 state together with the decay.
            unsigned delta_b[NPW][2];
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int col = (n_start + ni * kWarps) * 8 + g;
                delta_b[ni][0] =
                    pack_f16x2_sat(p.suffix[2 * t] * sm.delta[value_index<DV>(2 * t, col)],
                               p.suffix[2 * t + 1] * sm.delta[value_index<DV>(2 * t + 1, col)]);
                delta_b[ni][1] =
                    pack_f16x2_sat(p.suffix[2 * t + 8] * sm.delta[value_index<DV>(2 * t + 8, col)],
                               p.suffix[2 * t + 9] * sm.delta[value_index<DV>(2 * t + 9, col)]);
            }
#pragma unroll
            for (int mi = 0; mi < MPW; ++mi) {
                const int key     = (m_start + mi * MW) * 16 + g;
                const float decay = p.prefix[kChunkSize - 1];
                unsigned af[4];
                load_transposed_bf16_a(qk.k, key - g, lane, af);
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    unsigned part[2];
                    mma_f16_f16acc_zero(part[0], part[1], af[0], af[1], af[2], af[3],
                                        delta_b[ni][0], delta_b[ni][1]);
                    const float2 low  = __half22float2(half2_from_bits(part[0]));
                    const float2 high = __half22float2(half2_from_bits(part[1]));
                    state[ni][mi][0]  = fmaf(state[ni][mi][0], decay, low.x);
                    state[ni][mi][1]  = fmaf(state[ni][mi][1], decay, low.y);
                    state[ni][mi][2]  = fmaf(state[ni][mi][2], decay, high.x);
                    state[ni][mi][3]  = fmaf(state[ni][mi][3], decay, high.y);
                }
            }
        }
        __syncthreads();
        if (chunk + 2 < chunks) {
            load_stage<DV>(sm.stage[chunk & 1], args, qk_workspace, control_workspace, head,
                           value_start, chunk + 2, tid);
        }
    }
#pragma unroll
    for (int ni = 0; ni < NPW; ++ni) {
        const int n = n_start + ni * kWarps;
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key                                        = (m_start + mi * MW) * 16 + g;
            const int value                                      = value_start + n * 8 + t * 2;
            state_out[state_base + value * kStateDim + key]           = state_to_store<State>(state[ni][mi][0]);
            state_out[state_base + (value + 1) * kStateDim + key]     = state_to_store<State>(state[ni][mi][1]);
            state_out[state_base + value * kStateDim + key + 8]       = state_to_store<State>(state[ni][mi][2]);
            state_out[state_base + (value + 1) * kStateDim + key + 8] = state_to_store<State>(state[ni][mi][3]);
        }
    }
}

template <int DV, bool Fp16, class State>
void launch(const Arguments& args, const QkChunk* qk, const ControlChunk* control,
            cudaStream_t stream) {
    constexpr int shared_bytes = sizeof(RecurrenceShared<DV>);
    CUDA_CHECK(cudaFuncSetAttribute(chunk_recurrence_kernel<DV, Fp16, State>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    chunk_recurrence_kernel<DV, Fp16, State>
        <<<args.value_heads*(kStateDim / DV), kThreads<DV>, shared_bytes, stream>>>(args, qk,
                                                                                    control);
    CUDA_CHECK(cudaGetLastError());
}

template <int DV, class State>
void dispatch_state(const Arguments& args, const QkChunk* qk, const ControlChunk* control,
                    cudaStream_t stream, GdnChunkArithmetic arithmetic) {
    if (arithmetic == GdnChunkArithmetic::Fp16Accumulate) return launch<DV, true, State>(args, qk, control, stream);
    return launch<DV, false, State>(args, qk, control, stream);
}

template <int DV>
void dispatch(const Arguments& args, const QkChunk* qk, const ControlChunk* control,
              cudaStream_t stream, GdnChunkArithmetic arithmetic) {
    if (args.fp16_state) return dispatch_state<DV, __half>(args, qk, control, stream, arithmetic);
    return dispatch_state<DV, float>(args, qk, control, stream, arithmetic);
}

} // namespace

int value_tile(int value_heads, int multiprocessor_count) {
    // Small grids need the narrowest stripe. Otherwise use fixed relative tile costs
    // to weight the SM-wave count. These are launch-heuristic weights, not measured latency.
    // Wider stripes reduce replicated packet reads but retain more state per CTA.
    if (static_cast<std::int64_t>(value_heads) * 16 <= multiprocessor_count) return 8;
    int best               = 64;
    std::int64_t best_work = INT64_MAX;

    struct Choice {
        int tile, cost;
    };

    for (const auto [tile, cost] : {Choice{64, 34}, Choice{32, 20}, Choice{16, 13}}) {
        const auto blocks = static_cast<std::int64_t>(value_heads) * (kStateDim / tile);
        const auto waves  = (blocks + multiprocessor_count - 1) / multiprocessor_count;
        const auto work   = waves * cost;
        if (work < best_work) {
            best      = tile;
            best_work = work;
        }
    }
    return best;
}

void launch_recurrence(const Arguments& args, const QkChunk* qk, const ControlChunk* control,
                       DeviceExecutionView execution, GdnChunkArithmetic arithmetic) {
    switch (value_tile(args.value_heads, execution.multiprocessor_count)) {
    case 64:
        return dispatch<64>(args, qk, control, execution.stream, arithmetic);
    case 32:
        return dispatch<32>(args, qk, control, execution.stream, arithmetic);
    case 16:
        return dispatch<16>(args, qk, control, execution.stream, arithmetic);
    default:
        return dispatch<8>(args, qk, control, execution.stream, arithmetic);
    }
}

} // namespace infernix::ops::detail::gated_delta_net::chunked
