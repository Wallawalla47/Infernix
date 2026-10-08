// Register-streamed BF16 LinearSwiGLU for small T: Qwen3.8-Flash-Next's shared expert in the bit-exact
// artifact (NVIDIA stores it BF16), gate/up [1280, 2560] -> [640]. The BF16 twin of the Q8 stream-pair
// route: a warp owns output row i, loads every chunk of gate row i and up row M + i into registers
// before reading x, computes both dot products in the lane-chunk order (lane l owns k = 8l + 256m,
// fmaf in k order, a warp butterfly) and writes bf16(SiLU(gate) * up) with gate and up in FP32. A
// column's result does not depend on how many columns share the call.

#include "core/weight.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear_swiglu/bf16/bf16_linear_swiglu_kernels.h"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace infernix::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kK        = 2560;
constexpr int kChunks   = kK / 256;
constexpr int kWarps    = 4; // output rows per CTA
constexpr int kPassCols = 8;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    return value;
}

__device__ __forceinline__ void widen8(uint4 bits, float (&w)[8]) {
    const float2 a = bf16x2_bits_to_float2(bits.x), b = bf16x2_bits_to_float2(bits.y);
    const float2 c = bf16x2_bits_to_float2(bits.z), d = bf16x2_bits_to_float2(bits.w);
    w[0] = a.x, w[1] = a.y, w[2] = b.x, w[3] = b.y, w[4] = c.x, w[5] = c.y, w[6] = d.x, w[7] = d.y;
}

__global__ __launch_bounds__(kWarps * 32) void bf16_linear_swiglu_stream_pair_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ weight, int m_rows, int stride, int columns,
    bf16* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int i    = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (i >= m_rows) { return; }

    // Weights first: gate row i and up row M + i.
    uint4 w[2][kChunks];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const bf16* row = weight + (static_cast<std::int64_t>(h) * m_rows + i) * stride;
#pragma unroll
        for (int m = 0; m < kChunks; ++m) { w[h][m] = ld_nc_na<uint4>(row + m * 256 + lane * 8); }
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
            widen8(w[0][m], wg);
            widen8(w[1][m], wu);
#pragma unroll
            for (int c = 0; c < kPassCols; ++c) {
                if (c < pass) {
                    float xv[8];
                    widen8(load_ldg<uint4>(x + static_cast<std::int64_t>(c0 + c) * kK + kk), xv);
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

bool bf16_linear_swiglu_stream_pair_supported(const Tensor& x, const Weight& w, const Tensor& out) noexcept {
    const int stride = w.padded_shape[1] >= w.k ? w.padded_shape[1] : w.k;
    return w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.qdata != nullptr &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0 && stride % 8 == 0 && x.ne[0] == kK && w.k == kK &&
           w.n == 2 * out.ne[0] && x.ne[1] >= 1 && x.ne[1] <= kBf16LinearSwiGluStreamMaxColumns;
}

void bf16_linear_swiglu_stream_pair_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (!bf16_linear_swiglu_stream_pair_supported(x, w, out)) {
        throw std::invalid_argument("BF16 LinearSwiGLU stream pair: unsupported problem");
    }
    const int m_rows = out.ne[0], columns = x.ne[1];
    const int stride = w.padded_shape[1] >= w.k ? w.padded_shape[1] : w.k;
    bf16_linear_swiglu_stream_pair_kernel<<<(m_rows + kWarps - 1) / kWarps, kWarps * 32, 0, stream>>>(
        static_cast<const bf16*>(x.data), static_cast<const bf16*>(w.qdata), m_rows, stride, columns,
        static_cast<bf16*>(out.data));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("BF16 LinearSwiGLU stream pair: ") + cudaGetErrorString(error));
    }
}

} // namespace infernix::ops::detail
