// Register-streamed Q8 LinearSwiGLU for small T (Qwen3.8-Flash-Next's shared expert, gate/up
// [1280, 2560] -> [640]; design §19.3.3 Phase 1b). A warp owns output row i: it loads every code and
// scale of gate row i and up row M + i into registers before reading x, then computes both dot
// products with the K1 lane-chunk order (lane l owns codes k = 8l + 256m, fmaf in k order, a warp
// butterfly) and writes bf16(SiLU(gate) * up) with gate and up in FP32. A column's result does not
// depend on how many columns share the call.

#include "core/weight.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace infernix::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kK          = 2560;
constexpr int kChunks     = kK / 256;
constexpr int kWarps      = 4; // output rows per CTA
constexpr int kPassCols   = 8;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    return value;
}

__device__ __forceinline__ void decode8(uint2 code, float scale, float (&w)[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        w[i] = static_cast<float>(static_cast<std::int8_t>((i < 4 ? code.x : code.y) >> ((i & 3) * 8))) * scale;
    }
}

__global__ __launch_bounds__(kWarps * 32) void q8_linear_swiglu_stream_pair_kernel(
    const bf16* __restrict__ x, const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    int m_rows, int padded_k, int columns, bf16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int i    = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (i >= m_rows) { return; }

    // Weights first: gate row i and up row M + i.
    uint2 code[2][kChunks];
    float scale[2][kChunks];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const std::int64_t row        = static_cast<std::int64_t>(h) * m_rows + i;
        const std::uint8_t* code_row  = codes + row * padded_k;
        const std::uint8_t* scale_row = scales + row * (padded_k / 32) * 2;
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            const int kk = m * 256 + lane * 8;
            code[h][m]   = ld_nc_na<uint2>(code_row + kk);
            scale[h][m]  = __half2float(__ushort_as_half(ld_nc_na<std::uint16_t>(scale_row + (kk / 32) * 2)));
        }
    }

#pragma unroll 1
    for (int c0 = 0; c0 < columns; c0 += kPassCols) {
        const int pass = min(kPassCols, columns - c0);
        float gate[kPassCols], up[kPassCols];
#pragma unroll
        for (int c = 0; c < kPassCols; ++c) {
            gate[c] = 0.0F;
            up[c]   = 0.0F;
        }
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            const int kk = m * 256 + lane * 8;
            float wg[8], wu[8];
            decode8(code[0][m], scale[0][m], wg);
            decode8(code[1][m], scale[1][m], wu);
#pragma unroll
            for (int c = 0; c < kPassCols; ++c) {
                if (c < pass) {
                    const uint4 values = load_ldg<uint4>(x + static_cast<std::int64_t>(c0 + c) * kK + kk);
                    const float2 x0 = bf16x2_bits_to_float2(values.x), x1 = bf16x2_bits_to_float2(values.y);
                    const float2 x2 = bf16x2_bits_to_float2(values.z), x3 = bf16x2_bits_to_float2(values.w);
                    const float xv[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        gate[c] = fmaf(wg[j], xv[j], gate[c]);
                        up[c]   = fmaf(wu[j], xv[j], up[c]);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < kPassCols; ++c) {
            if (c < pass) {
                const float g = warp_sum(gate[c]);
                const float u = warp_sum(up[c]);
                if (lane == 0) {
                    out[static_cast<std::int64_t>(c0 + c) * m_rows + i] = __float2bfloat16_rn(g / (1.0F + expf(-g)) * u);
                }
            }
        }
    }
}

} // namespace

void q8_linear_swiglu_stream_pair_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int m_rows = out.ne[0], columns = x.ne[1];
    if (x.ne[0] != kK || w.k != kK || w.n != 2 * m_rows || columns <= 0) {
        throw std::invalid_argument("Q8 LinearSwiGLU stream pair: unsupported problem");
    }
    q8_linear_swiglu_stream_pair_kernel<<<(m_rows + kWarps - 1) / kWarps, kWarps * 32, 0, stream>>>(
        static_cast<const bf16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), m_rows, w.padded_shape[1], columns, static_cast<bf16*>(out.data));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("Q8 LinearSwiGLU stream pair: ") + cudaGetErrorString(error));
    }
}

} // namespace infernix::ops::detail
