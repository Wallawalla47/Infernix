// GGUF Q6_K quantization of BF16 rows by ggml's reference quantizer (include/infernix/ops/ggml_quantize.h).

#include "infernix/ops/ggml_quantize.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr int kBlockElems  = 256;
constexpr int kBlockBytes  = 210;
constexpr int kCtaBlocks   = 8;               // Q6_K blocks a CTA
constexpr int kThreads     = kCtaBlocks * 16; // a 16-element sub-block a thread
constexpr float kGroupEps  = 1e-15F;          // ggml's GROUP_MAX_EPS

__device__ __forceinline__ int clamp_code(int l) { return l < -32 ? -32 : (l > 31 ? 31 : l); }

// make_qx_quants(16, 32, x, L, 1, nullptr): the sub-block's scale and its codes L + 32 in [0, 63].
__device__ float sub_scale(const float (&x)[16], int (&L)[16]) {
    float max = 0.0F, amax = 0.0F;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max  = x[i];
        }
    }
    if (amax < kGroupEps) {
#pragma unroll
        for (int i = 0; i < 16; ++i) { L[i] = 0; }
        return 0.0F;
    }
    float iscale = __fdiv_rn(-32.0F, max);
    float sumlx = 0.0F, suml2 = 0.0F;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int l = clamp_code(__float2int_rn(__fmul_rn(iscale, x[i])));
        L[i]        = l + 32;
        const float w = __fmul_rn(x[i], x[i]), fl = static_cast<float>(l);
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), fl));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, fl), fl));
    }
    float scale = suml2 != 0.0F ? __fdiv_rn(sumlx, suml2) : 0.0F;
    float best  = __fmul_rn(scale, sumlx);
#pragma unroll 1
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) { continue; }
        iscale = __fdiv_rn(-__fadd_rn(32.0F, __fmul_rn(0.1F, static_cast<float>(is))), max);
        sumlx = suml2 = 0.0F;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float fl = static_cast<float>(clamp_code(__float2int_rn(__fmul_rn(iscale, x[i]))));
            const float w  = __fmul_rn(x[i], x[i]);
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), fl));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, fl), fl));
        }
        if (suml2 > 0.0F && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
#pragma unroll
            for (int i = 0; i < 16; ++i) { L[i] = 32 + clamp_code(__float2int_rn(__fmul_rn(iscale, x[i]))); }
            scale = __fdiv_rn(sumlx, suml2);
            best  = __fmul_rn(scale, sumlx);
        }
    }
    return scale;
}

// Thread 16 j + ib takes sub-block ib of the CTA's block j (a half-warp a block).
__global__ void __launch_bounds__(kThreads)
    quantize_q6_k_kernel(const __nv_bfloat16* __restrict__ w, int k, int blocks, int tiled, std::uint8_t* __restrict__ out) {
    __shared__ std::uint8_t codes[kCtaBlocks][kBlockElems];
    const int tid = static_cast<int>(threadIdx.x), j = tid / 16, ib = tid % 16;
    const int block = static_cast<int>(blockIdx.x) * kCtaBlocks + j;
    const bool live = block < blocks;
    const int per_row = k / kBlockElems;
    const int r = live ? block / per_row : 0, b = live ? block % per_row : 0;
    float x[16];
    const __nv_bfloat16* src = w + static_cast<std::size_t>(r) * k + b * kBlockElems + ib * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) { x[i] = live ? __bfloat162float(src[i]) : 0.0F; }
    int L[16];
    const float scale = sub_scale(x, L);
    // The block's scale of largest magnitude, the first one on ties (ggml's strict > in sub-block order).
    float best_abs = fabsf(scale), best = scale;
    int best_ib = ib;
#pragma unroll
    for (int o = 8; o >= 1; o >>= 1) {
        const float other_abs = __shfl_xor_sync(0xFFFFFFFFU, best_abs, o);
        const float other     = __shfl_xor_sync(0xFFFFFFFFU, best, o);
        const int other_ib    = __shfl_xor_sync(0xFFFFFFFFU, best_ib, o);
        if (other_abs > best_abs || (other_abs == best_abs && other_ib < best_ib)) {
            best_abs = other_abs;
            best     = other;
            best_ib  = other_ib;
        }
    }
    const bool zero = best_abs < kGroupEps;
    std::uint16_t d_bits = 0;
    int sub              = 0;
    if (!zero) {
        const float iscale = __fdiv_rn(-128.0F, best);
        const __half d     = __float2half_rn(__fdiv_rn(1.0F, iscale));
        d_bits             = __half_as_ushort(d);
        sub                = min(127, __float2int_rn(__fmul_rn(iscale, scale)));
        const float dl     = __fmul_rn(__half2float(d), static_cast<float>(sub));
        if (dl != 0.0F) {
#pragma unroll
            for (int i = 0; i < 16; ++i) { L[i] = 32 + clamp_code(__float2int_rn(__fdiv_rn(x[i], dl))); }
        }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) { codes[j][ib * 16 + i] = zero ? 0 : static_cast<std::uint8_t>(L[i]); }
    __syncwarp();
    if (!live) { return; }
    const std::size_t index = tiled != 0 ? (static_cast<std::size_t>(r / 16) * per_row + b) * 16 + r % 16
                                         : static_cast<std::size_t>(r) * per_row + b;
    std::uint8_t* y = out + index * kBlockBytes;
    // Thread ib packs positions l = 4 (ib % 8) .. + 3 of half h = ib / 8 (elements 128 h + l + {0, 32, 64, 96}).
    const int h = ib / 8;
    const std::uint8_t* q = codes[j] + 128 * h;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int l = 4 * (ib % 8) + e;
        y[64 * h + l]      = static_cast<std::uint8_t>((q[l] & 0xF) | ((q[l + 64] & 0xF) << 4));
        y[64 * h + 32 + l] = static_cast<std::uint8_t>((q[l + 32] & 0xF) | ((q[l + 96] & 0xF) << 4));
        y[128 + 32 * h + l] = static_cast<std::uint8_t>((q[l] >> 4) | ((q[l + 32] >> 4) << 2) | ((q[l + 64] >> 4) << 4) |
                                                        ((q[l + 96] >> 4) << 6));
    }
    y[192 + ib] = static_cast<std::uint8_t>(static_cast<std::int8_t>(sub));
    if (ib == 0) {
        y[208] = static_cast<std::uint8_t>(d_bits & 0xFF);
        y[209] = static_cast<std::uint8_t>(d_bits >> 8);
    }
}

} // namespace

void ggml_quantize_q6_k(const Tensor& w, GgmlLayout layout, std::uint8_t* out, cudaStream_t stream) {
    const auto fail = [](const char* message) { throw std::invalid_argument(std::string("ggml_quantize_q6_k: ") + message); };
    if (w.dtype != DType::BF16 || !w.is_contiguous() || w.ne[2] != 1 || w.ne[3] != 1) { fail("w must be contiguous BF16 [k, n]"); }
    const std::int64_t k = w.ne[0], n = w.ne[1];
    if (k <= 0 || k % kBlockElems != 0 || n <= 0) { fail("k must be a positive multiple of 256 and n positive"); }
    if (layout == GgmlLayout::Tiles16 && n % 16 != 0) { fail("a Tiles16 matrix needs a multiple of 16 rows"); }
    if (out == nullptr) { fail("out is null"); }
    const std::int64_t blocks = n * (k / kBlockElems);
    if (blocks > (std::int64_t{1} << 30)) { fail("too many blocks"); }
    const unsigned grid = static_cast<unsigned>((blocks + kCtaBlocks - 1) / kCtaBlocks);
    quantize_q6_k_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(w.data), static_cast<int>(k),
                                                         static_cast<int>(blocks), layout == GgmlLayout::Tiles16 ? 1 : 0,
                                                         out);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ggml_quantize_q6_k: launch: ") + cudaGetErrorString(error));
    }
}

} // namespace infernix::ops
