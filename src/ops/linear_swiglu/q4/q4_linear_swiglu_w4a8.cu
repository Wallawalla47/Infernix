// The q4-w4a8 route of the Q4 LinearSwiGLU (ops::LinearRoutes::Q4W4A8), after giveen/ninfer-ext
// e99059abb and b8a4ec826.
//
// 1. Each token's BF16 activation becomes int8 codes q = round(x * 127 / max|x|) with one FP32
//    scale max|x| / 127 per token (the only change to the arithmetic: the exact route keeps the
//    BF16 activation).
// 2. The stored Q4 codes stay packed in global memory (half the bytes of int8 weights, so the
//    active weight slice stays L2-resident) and are sign-extended to int8 while staging a group.
// 3. Per 64-element group, two m16n8k32 s8 x s8 products give the exact integer dot product of
//    the codes; the group's FP16 weight scale is folded into an FP32 accumulator with one FMA, and
//    the per-token scale multiplies the total in the epilogue, which pairs silu(gate) * up.
//
// A CTA computes 128 tokens x 64 gate rows and their 64 matching up rows. The token block is the
// fast grid axis, so the CTAs resident together read the same weight rows.

#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_w4a8.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kInputRows       = 5120; // K of the registered problem
constexpr int kQuantizeThreads = 128;

constexpr int kTileTokens = 128;
constexpr int kTileRows   = 64; // gate rows per CTA; the CTA also computes their up rows
constexpr int kGroupK     = 64;
constexpr int kThreads    = 512;
constexpr int kStages     = 2;
constexpr int kWarps      = kThreads / kWarpSize;    // 16
constexpr int kTokenWarps = kTileTokens / 16;        // 8 warps along tokens (m16)
constexpr int kRowWarps   = kWarps / kTokenWarps;    // 2 warps along rows
constexpr int kWarpRows   = kTileRows / kRowWarps;   // 32 gate rows per warp
constexpr int kRowTiles   = kWarpRows / 8;           // 4 n8 tiles per warp
constexpr int kChunks     = kGroupK / 16;            // 16-byte int8 chunks per tile row
constexpr int kCodeBytes  = 32;                      // packed Q4 bytes per (row, group)

static_assert(kWarps == kTokenWarps * kRowWarps, "warps must tile the CTA");
static_assert(kTileTokens * kChunks == kThreads, "one activation chunk per thread per group");
static_assert(2 * kTileRows * kChunks == kThreads, "one 16-code weight chunk per thread per group");
static_assert(2 * kTileRows <= kThreads, "one weight scale per thread per group");
static_assert(kGroupK == 64, "one Q4 scale group per stage");

// ---------------------------------------------------------------------------------------------
// Per-token int8 activation quantization.

__device__ __forceinline__ std::uint32_t pack_int8x4(int q0, int q1, int q2, int q3) {
    return (static_cast<std::uint32_t>(q0) & 0xffU) | ((static_cast<std::uint32_t>(q1) & 0xffU) << 8) |
           ((static_cast<std::uint32_t>(q2) & 0xffU) << 16) |
           ((static_cast<std::uint32_t>(q3) & 0xffU) << 24);
}

__device__ __forceinline__ int quantize_int8(float value, float inverse) {
    return min(127, max(-127, __float2int_rn(value * inverse)));
}

// One CTA per token. The symmetric range maps the token's largest magnitude to +-127. A token with
// a non-finite element gets a NaN scale, so every output it feeds is non-finite, as on the exact
// route.
template <int InputRows, int Threads>
__global__ __launch_bounds__(Threads) void q4_w4a8_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::int8_t* __restrict__ codes,
    float* __restrict__ scales) {
    static_assert(InputRows % (Threads * 8) == 0, "each thread converts whole 8-element vectors");
    static_assert(Threads % kWarpSize == 0, "whole warps");
    constexpr int kVectors = InputRows / (Threads * 8);
    constexpr int kQuantizeWarps = Threads / kWarpSize;
    __shared__ float warp_maxima[kQuantizeWarps];
    __shared__ int warp_nonfinite[kQuantizeWarps];

    const std::int64_t token = static_cast<std::int64_t>(blockIdx.x);
    const int tid            = static_cast<int>(threadIdx.x);
    const int lane           = tid & (kWarpSize - 1);
    const int warp           = tid / kWarpSize;
    const auto* source = reinterpret_cast<const uint4*>(input + token * InputRows);
    auto* destination  = reinterpret_cast<uint2*>(codes + token * InputRows);

    float values[kVectors][8];
    float maximum  = 0.0F;
    bool nonfinite = false;
#pragma unroll
    for (int v = 0; v < kVectors; ++v) {
        const uint4 bits                   = source[tid + v * Threads];
        const std::uint32_t words[4]       = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 pair    = bf16x2_bits_to_float2(words[w]);
            values[v][2 * w]     = pair.x;
            values[v][2 * w + 1] = pair.y;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float magnitude = fabsf(values[v][e]);
            maximum               = fmaxf(maximum, magnitude);
            // True for NaN as well as infinity (fmaxf above ignores NaN).
            nonfinite = nonfinite || !(magnitude <= 3.402823466e38F);
        }
    }
    maximum             = warp_max(maximum);
    const bool warp_bad = __any_sync(kFullWarpMask, nonfinite) != 0;
    if (lane == 0) {
        warp_maxima[warp]    = maximum;
        warp_nonfinite[warp] = warp_bad ? 1 : 0;
    }
    __syncthreads();
    maximum  = 0.0F;
    bool bad = false;
#pragma unroll
    for (int w = 0; w < kQuantizeWarps; ++w) {
        maximum = fmaxf(maximum, warp_maxima[w]);
        bad     = bad || warp_nonfinite[w] != 0;
    }

    const float inverse = (!bad && maximum > 0.0F) ? 127.0F / maximum : 0.0F;
#pragma unroll
    for (int v = 0; v < kVectors; ++v) {
        const float* x = values[v];
        destination[tid + v * Threads] = make_uint2(
            pack_int8x4(quantize_int8(x[0], inverse), quantize_int8(x[1], inverse),
                        quantize_int8(x[2], inverse), quantize_int8(x[3], inverse)),
            pack_int8x4(quantize_int8(x[4], inverse), quantize_int8(x[5], inverse),
                        quantize_int8(x[6], inverse), quantize_int8(x[7], inverse)));
    }
    if (tid == 0) { scales[token] = bad ? __int_as_float(0x7fffffff) : maximum / 127.0F; }
}

// ---------------------------------------------------------------------------------------------
// Q4 code expansion.

// Each byte holds one nibble n; the result byte is (n ^ 8) - 8, the signed value Q4_G64_FP16
// stores, in two's complement. Setting every byte's high bit before the subtraction keeps each
// borrow inside its byte; clearing it afterwards restores the sign.
__device__ __forceinline__ std::uint32_t sign_extend_nibbles(std::uint32_t nibbles) {
    return (((nibbles ^ 0x08080808U) | 0x80808080U) - 0x08080808U) ^ 0x80808080U;
}

// Four packed bytes (eight weights, the even weight in each low nibble) become eight int8 weights
// in weight order: `first` holds weights 0-3, `second` weights 4-7.
__device__ __forceinline__ void expand_q4_codes(std::uint32_t word, std::uint32_t& first,
                                                std::uint32_t& second) {
    const std::uint32_t low  = word & 0x0f0f0f0fU;
    const std::uint32_t high = (word >> 4) & 0x0f0f0f0fU;
    first                    = sign_extend_nibbles(__byte_perm(low, high, 0x5140));
    second                   = sign_extend_nibbles(__byte_perm(low, high, 0x7362));
}

// Byte offset of (row, byte) in a 64-byte-row int8 tile. Rows 2i and 2i+1 share one 128-byte bank
// line, so the 16-byte chunk index is XORed with (row / 2) % 4: the same logical chunk of eight
// consecutive rows lands on eight distinct 16-byte bank slots (conflict-free ldmatrix).
__device__ __forceinline__ int tile_offset(int row, int byte) {
    return row * kGroupK + ((((byte >> 4) ^ (row >> 1)) & 3) << 4) + (byte & 15);
}

// ---------------------------------------------------------------------------------------------
// The fused gate/up W4A8 GEMM with the SwiGLU epilogue.

__global__ __launch_bounds__(kThreads, 2) void q4_linear_swiglu_w4a8_kernel(
    const std::int8_t* __restrict__ xq, const float* __restrict__ xs,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t intermediate, std::int32_t k, std::int32_t t) {
    __shared__ __align__(16) std::int8_t a_tile[kStages][kTileTokens * kGroupK];
    __shared__ __align__(16) std::int8_t b_tile[kStages][2][kTileRows * kGroupK];
    __shared__ __align__(16) float w_scale[kStages][2][kTileRows];

    const int tid        = static_cast<int>(threadIdx.x);
    const int lane       = tid & (kWarpSize - 1);
    const int warp       = tid / kWarpSize;
    const int gid        = lane >> 2;
    const int tig        = lane & 3;
    const int warp_token = warp % kTokenWarps;
    const int warp_row   = warp / kTokenWarps;
    const int tok0       = static_cast<int>(blockIdx.x) * kTileTokens;
    const int row0       = static_cast<int>(blockIdx.y) * kTileRows;
    const int groups     = k / kGroupK;

    // This thread's 16-byte activation chunk of every group.
    const int a_load_row     = tid / kChunks;
    const int a_load_chunk   = tid % kChunks;
    const bool a_load_valid  = tok0 + a_load_row < t;
    const std::int8_t* a_src = xq +
                               static_cast<std::int64_t>(a_load_valid ? tok0 + a_load_row : 0) * k +
                               a_load_chunk * 16;
    const int a_dst = tile_offset(a_load_row, a_load_chunk * 16);

    // This thread's 16 weight codes (8 packed bytes) of every group: gate rows, then up rows.
    const int b_half   = tid / (kTileRows * kChunks);
    const int b_row    = (tid / kChunks) % kTileRows;
    const int b_chunk  = tid % kChunks;
    const std::uint8_t* b_src =
        codes +
        (static_cast<std::int64_t>(row0 + b_row) + (b_half != 0 ? intermediate : 0)) * groups *
            kCodeBytes +
        b_chunk * 8;
    const int b_dst = tile_offset(b_row, b_chunk * 16);

    // Threads [0, 128) each stage one FP16 group scale: gate rows, then up rows.
    const bool loads_scale = tid < 2 * kTileRows;
    const int s_index      = loads_scale ? tid : 0;
    const int s_half       = s_index / kTileRows;
    const int s_row        = s_index % kTileRows;
    const std::uint8_t* s_src =
        scales +
        (static_cast<std::int64_t>(row0 + s_row) + (s_half != 0 ? intermediate : 0)) * groups * 2;

    auto stage_activation = [&](int stage, int g) {
        std::int8_t* dst = a_tile[stage] + a_dst;
        if (a_load_valid) {
            cp_async<16, Cache::cg>(dst, a_src + g * kGroupK);
        } else {
            store_vec(dst, make_int4(0, 0, 0, 0));
        }
    };
    auto load_codes = [&](int g) {
        return load_ldg<uint2>(b_src + static_cast<std::int64_t>(g) * kCodeBytes);
    };
    auto load_scale = [&](int g) {
        return load_ldg<std::uint16_t>(s_src + static_cast<std::int64_t>(g) * 2);
    };
    auto store_codes = [&](int stage, uint2 raw) {
        uint4 expanded;
        expand_q4_codes(raw.x, expanded.x, expanded.y);
        expand_q4_codes(raw.y, expanded.z, expanded.w);
        store_vec(b_tile[stage][b_half] + b_dst, expanded);
    };
    auto store_scale = [&](int stage, std::uint16_t bits) {
        w_scale[stage][s_half][s_row] = __half2float(__ushort_as_half(bits));
    };

#pragma unroll
    for (int s = 0; s < kStages; ++s) {
        if (s < groups) {
            stage_activation(s, s);
            store_codes(s, load_codes(s));
            if (loads_scale) { store_scale(s, load_scale(s)); }
        }
        cp_commit();
    }

    float gate_acc[kRowTiles][4];
    float up_acc[kRowTiles][4];
#pragma unroll
    for (int mr = 0; mr < kRowTiles; ++mr) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            gate_acc[mr][c] = 0.0F;
            up_acc[mr][c]   = 0.0F;
        }
    }

    // ldmatrix x4 addresses. A: matrices (tokens 0-7, k 0-15), (8-15, 0-15), (0-7, 16-31),
    // (8-15, 16-31) of the warp's 16 tokens give a0-a3 of m16n8k32. B: lanes 0-15 address the gate
    // tile and lanes 16-31 the up tile, each at k 0-15 and 16-31, giving gate b0/b1, up b0/b1.
    const int a_row        = warp_token * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_col        = (lane >> 4) * 16;
    const int b_row_base   = warp_row * kWarpRows + (lane & 7);
    const int b_col        = ((lane >> 3) & 1) * 16;
    const int b_frag_half  = lane >> 4;

#pragma unroll 1
    for (int g = 0; g < groups; ++g) {
        const int stage = g % kStages;
        cp_wait<kStages - 1>();
        __syncthreads();

        // Fetch the codes and scale of the group this stage holds next, so their global latency
        // overlaps this group's products.
        const int next      = g + kStages;
        const bool has_next = next < groups;
        uint2 next_codes    = make_uint2(0U, 0U);
        std::uint16_t next_scale = 0;
        if (has_next) {
            next_codes = load_codes(next);
            if (loads_scale) { next_scale = load_scale(next); }
        }

        unsigned a[2][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            ldmatrix_x4(a[ks][0], a[ks][1], a[ks][2], a[ks][3],
                        smem_addr(a_tile[stage] + tile_offset(a_row, ks * 32 + a_col)));
        }
        const std::int8_t* b_frag = b_tile[stage][b_frag_half];
        const float* gate_scale   = w_scale[stage][0];
        const float* up_scale     = w_scale[stage][1];
#pragma unroll
        for (int mr = 0; mr < kRowTiles; ++mr) {
            const int brow = b_row_base + mr * 8;
            unsigned g0, g1, u0, u1;
            int gate[4];
            int up[4];
            ldmatrix_x4(g0, g1, u0, u1, smem_addr(b_frag + tile_offset(brow, b_col)));
            mma_s8s8_k32_zero(gate[0], gate[1], gate[2], gate[3], a[0][0], a[0][1], a[0][2],
                              a[0][3], g0, g1);
            mma_s8s8_k32_zero(up[0], up[1], up[2], up[3], a[0][0], a[0][1], a[0][2], a[0][3], u0,
                              u1);
            ldmatrix_x4(g0, g1, u0, u1, smem_addr(b_frag + tile_offset(brow, 32 + b_col)));
            mma_s8(gate[0], gate[1], gate[2], gate[3], a[1][0], a[1][1], a[1][2], a[1][3], g0, g1);
            mma_s8(up[0], up[1], up[2], up[3], a[1][0], a[1][1], a[1][2], a[1][3], u0, u1);

            // The group's integer dot products are exact (|sum| <= 64 * 127 * 8); one FMA per
            // output folds in the weight scale of its row (c0/c2: row 2*tig, c1/c3: 2*tig + 1).
            const int r     = warp_row * kWarpRows + mr * 8 + 2 * tig;
            const float2 gs = load_vec<float2>(gate_scale + r);
            const float2 us = load_vec<float2>(up_scale + r);
            gate_acc[mr][0] = fmaf(static_cast<float>(gate[0]), gs.x, gate_acc[mr][0]);
            gate_acc[mr][1] = fmaf(static_cast<float>(gate[1]), gs.y, gate_acc[mr][1]);
            gate_acc[mr][2] = fmaf(static_cast<float>(gate[2]), gs.x, gate_acc[mr][2]);
            gate_acc[mr][3] = fmaf(static_cast<float>(gate[3]), gs.y, gate_acc[mr][3]);
            up_acc[mr][0]   = fmaf(static_cast<float>(up[0]), us.x, up_acc[mr][0]);
            up_acc[mr][1]   = fmaf(static_cast<float>(up[1]), us.y, up_acc[mr][1]);
            up_acc[mr][2]   = fmaf(static_cast<float>(up[2]), us.x, up_acc[mr][2]);
            up_acc[mr][3]   = fmaf(static_cast<float>(up[3]), us.y, up_acc[mr][3]);
        }

        __syncthreads();
        if (has_next) {
            stage_activation(stage, next);
            store_codes(stage, next_codes);
            if (loads_scale) { store_scale(stage, next_scale); }
        }
        cp_commit();
    }

    // c0/c1 belong to token gid, c2/c3 to token gid + 8; each pair is two adjacent output rows.
    const int token_lo   = tok0 + warp_token * 16 + gid;
    const int token_hi   = token_lo + 8;
    const float scale_lo = token_lo < t ? xs[token_lo] : 0.0F;
    const float scale_hi = token_hi < t ? xs[token_hi] : 0.0F;
#pragma unroll
    for (int mr = 0; mr < kRowTiles; ++mr) {
        const int row = row0 + warp_row * kWarpRows + mr * 8 + 2 * tig;
        if (token_lo < t) {
            const float g0 = gate_acc[mr][0] * scale_lo;
            const float g1 = gate_acc[mr][1] * scale_lo;
            const float u0 = up_acc[mr][0] * scale_lo;
            const float u1 = up_acc[mr][1] * scale_lo;
            store_vec(out + static_cast<std::int64_t>(token_lo) * intermediate + row,
                      pack_bf16x2(silu(g0) * u0, silu(g1) * u1));
        }
        if (token_hi < t) {
            const float g0 = gate_acc[mr][2] * scale_hi;
            const float g1 = gate_acc[mr][3] * scale_hi;
            const float u0 = up_acc[mr][2] * scale_hi;
            const float u1 = up_acc[mr][3] * scale_hi;
            store_vec(out + static_cast<std::int64_t>(token_hi) * intermediate + row,
                      pack_bf16x2(silu(g0) * u0, silu(g1) * u1));
        }
    }
}

} // namespace

void q4_linear_swiglu_w4a8_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    const std::int32_t k            = x.ne[0];
    const std::int32_t t            = x.ne[1];
    const std::int32_t intermediate = out.ne[0];
    if (k != kInputRows || w.padded_shape[1] != k || w.n != 2 * intermediate ||
        intermediate % kTileRows != 0 || t <= 0 || out.ne[1] != t) {
        throw std::invalid_argument("q4 w4a8 linear_swiglu: unsupported problem");
    }

    auto scope                    = ws.scope();
    const Q4W4A8Workspace scratch = allocate_q4_w4a8_workspace(ws, t, k);
    q4_w4a8_quantize_kernel<kInputRows, kQuantizeThreads>
        <<<static_cast<unsigned>(t), kQuantizeThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), scratch.codes, scratch.scales);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid(static_cast<unsigned>(div_up(t, kTileTokens)),
                    static_cast<unsigned>(intermediate / kTileRows));
    q4_linear_swiglu_w4a8_kernel<<<grid, kThreads, 0, stream>>>(
        scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
        intermediate, k, t);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
