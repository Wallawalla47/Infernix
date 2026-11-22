// Inkling 2's vision-tower elementwise Ops (include/infernix/ops/inkling2_vision.h): the fused QKV bias with the axial
// rotary on interleaved pairs, and the layer-scaled residual branch. Each thread owns one feature pair of one row, so a
// row's values depend on that row alone.

#include "infernix/ops/inkling2_vision.h"

#include "core/device.h"
#include "ops/common/canonical_math.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::inkling2 {
namespace {

namespace canon = infernix::ops::canon;

constexpr int kThreads = 256;

__device__ __forceinline__ float2 load_pair(const std::uint16_t* p) {
    const std::uint32_t w = *reinterpret_cast<const std::uint32_t*>(p);
    return {canon::f32_from_bits(w << 16), canon::f32_from_bits(w & 0xFFFF0000U)};
}

__device__ __forceinline__ void store_pair(std::uint16_t* p, float a, float b) {
    const std::uint32_t w = std::uint32_t{canon::f32_to_bf16_rn(a)} | (std::uint32_t{canon::f32_to_bf16_rn(b)} << 16);
    *reinterpret_cast<std::uint32_t*>(p) = w;
}

// The inverse frequencies of one axis: inv[i] = theta^(-i / R), i < R = head_dim / 4, FP32 of the FP64 value (a
// kernel parameter, so the call allocates nothing and stays graph-capturable).
constexpr int kMaxAxisPairs = 32; // head_dim <= 128
struct InverseFrequencies {
    float inv[kMaxAxisPairs];
};

__global__ void vision_qkv_rotary_kernel(std::uint16_t* __restrict__ qkv, const std::uint16_t* __restrict__ bias,
                                         const std::int32_t* __restrict__ positions, InverseFrequencies inverse,
                                         std::int32_t width, std::int32_t head_dim) {
    const int token  = static_cast<int>(blockIdx.y);
    const int pair   = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    const int pairs  = 3 * width / 2;
    if (pair >= pairs) { return; }
    const int f      = 2 * pair;
    std::uint16_t* row = qkv + static_cast<std::int64_t>(token) * 3 * width;
    const float2 z0  = load_pair(row + f);
    const float2 b   = load_pair(bias + f);
    float x0 = z0.x + b.x;
    float x1 = z0.y + b.y;
    if (f < 2 * width) { // q or k: rotate the pair
        const int j    = (f % head_dim) / 2;
        const int R    = head_dim / 4;
        const int axis = j < R ? 1 : 0; // the first R pairs by column, the rest by row
        const float angle = static_cast<float>(positions[2 * token + axis]) * inverse.inv[j % R];
        float s, c;
        sincosf(angle, &s, &c);
        const float r0 = fmaf(x0, c, -x1 * s);
        const float r1 = fmaf(x0, s, x1 * c);
        x0 = r0;
        x1 = r1;
    }
    store_pair(row + f, x0, x1);
}

__global__ void vision_scaled_residual_kernel(std::uint16_t* __restrict__ x, const std::uint16_t* __restrict__ y,
                                              const std::uint16_t* __restrict__ bias,
                                              const std::uint16_t* __restrict__ scale, std::int32_t width) {
    const int token = static_cast<int>(blockIdx.y);
    const int pair  = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (2 * pair >= width) { return; }
    const int f              = 2 * pair;
    const std::int64_t at    = static_cast<std::int64_t>(token) * width + f;
    const float2 xv = load_pair(x + at), yv = load_pair(y + at), bv = load_pair(bias + f), sv = load_pair(scale + f);
    store_pair(x + at, fmaf(sv.x, yv.x + bv.x, xv.x), fmaf(sv.y, yv.y + bv.y, xv.y));
}

} // namespace

void vision_qkv_rotary(const VisionQkvArgs& a, cudaStream_t stream) {
    if (a.qkv == nullptr || a.bias == nullptr || a.positions == nullptr || a.tokens <= 0 || a.heads <= 0 ||
        a.head_dim <= 0 || a.head_dim % 4 != 0 || a.head_dim / 4 > kMaxAxisPairs || !(a.theta > 0.0F) ||
        !std::isfinite(a.theta)) {
        throw std::invalid_argument("vision_qkv_rotary: invalid arguments");
    }
    const int width = a.heads * a.head_dim;
    const int R     = a.head_dim / 4;
    InverseFrequencies inverse{};
    for (int i = 0; i < R; ++i) {
        inverse.inv[i] = static_cast<float>(std::pow(static_cast<double>(a.theta), -static_cast<double>(i) / R));
    }
    const int pairs = 3 * width / 2;
    const dim3 grid((pairs + kThreads - 1) / kThreads, a.tokens);
    vision_qkv_rotary_kernel<<<grid, kThreads, 0, stream>>>(a.qkv, a.bias, a.positions, inverse, width, a.head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void vision_scaled_residual(const VisionScaledResidualArgs& a, cudaStream_t stream) {
    if (a.x == nullptr || a.y == nullptr || a.bias == nullptr || a.scale == nullptr || a.tokens <= 0 ||
        a.width <= 0 || a.width % 2 != 0) {
        throw std::invalid_argument("vision_scaled_residual: invalid arguments");
    }
    const dim3 grid((a.width / 2 + kThreads - 1) / kThreads, a.tokens);
    vision_scaled_residual_kernel<<<grid, kThreads, 0, stream>>>(a.x, a.y, a.bias, a.scale, a.width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::inkling2
