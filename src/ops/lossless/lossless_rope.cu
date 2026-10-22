#include "infernix/ops/lossless_ops.h"

#include "core/device.h"
#include "ops/lossless/cr_math.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

// The --lossless route's rotary embedding (include/infernix/ops/lossless_ops.h): Hugging Face's
// Qwen3.5 rotary operations, each correctly rounded. inv_freq = fp32(1 / fp32(theta^(2i / R))); per column
// and pair, the angle fp32(position inv_freq), its cos and sin correctly rounded to binary32 and then to
// BF16 (the module's .to(dtype)); per head, x_i = bf16(bf16(x_i cos) + bf16(-x_(i+R/2) sin)) and
// x_(i+R/2) = bf16(bf16(x_(i+R/2) cos) + bf16(x_i sin)) (apply_rotary_pos_emb's BF16 tensor products and
// sum; a BF16 product is exact in binary32 and a sum of two BF16 values rounds correctly through
// binary64).

namespace infernix::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless_rope: ") + message); }
}

__global__ void rope_frequencies_kernel(float theta, int rotary_dim, float* inv_freq) {
    const int i = static_cast<int>(threadIdx.x);
    if (i >= rotary_dim / 2) { return; }
    inv_freq[i] = __fdiv_rn(1.0F, lossless::cr_pow_ratio(theta, 2 * i, rotary_dim));
}

// table: cos [T][P] then sin [T][P] (P = R / 2 pairs), BF16.
__global__ void rope_table_kernel(const std::int32_t* positions, int axes, int columns, const float* inv_freq, int pairs,
                                  std::uint16_t* table) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= columns * pairs) { return; }
    const int t = index / pairs, i = index % pairs;
    const int axis = axes == 3 ? i % 3 : 0;
    const float position = static_cast<float>(positions[static_cast<std::int64_t>(axis) * columns + t]);
    const float angle = __fmul_rn(position, inv_freq[i]);
    float c, s;
    lossless::cr_cos_sin(angle, c, s);
    table[index]                                          = __bfloat16_as_ushort(__float2bfloat16_rn(c));
    table[static_cast<std::size_t>(columns) * pairs + index] = __bfloat16_as_ushort(__float2bfloat16_rn(s));
}

__device__ __forceinline__ float bf16f(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16); }

// One thread per (pair, head, column), in place.
__global__ void rope_apply_kernel(const std::uint16_t* table, int pairs, std::uint16_t* x, int head_dim, int heads,
                                  int columns) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<long long>(pairs) * heads * columns) { return; }
    const int i = static_cast<int>(index % pairs);
    const long long rest = index / pairs;
    const int h = static_cast<int>(rest % heads), t = static_cast<int>(rest / heads);
    const float c = bf16f(table[static_cast<std::size_t>(t) * pairs + i]);
    const float s = bf16f(table[static_cast<std::size_t>(columns) * pairs + static_cast<std::size_t>(t) * pairs + i]);
    std::uint16_t* head = x + (static_cast<std::size_t>(t) * heads + h) * head_dim;
    const float a = bf16f(head[i]), b = bf16f(head[i + pairs]);
    // Products of two BF16 values are exact in binary32: one rounding to BF16 each.
    const float ac = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a, c)));
    const float bs = __bfloat162float(__float2bfloat16_rn(__fmul_rn(b, s)));
    const float bc = __bfloat162float(__float2bfloat16_rn(__fmul_rn(b, c)));
    const float as = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a, s)));
    head[i]         = lossless::bf16_of_double(static_cast<double>(ac) - static_cast<double>(bs));
    head[i + pairs] = lossless::bf16_of_double(static_cast<double>(bc) + static_cast<double>(as));
}

} // namespace

void lossless_rope_frequencies(float theta, int rotary_dim, Tensor& inv_freq, cudaStream_t stream) {
    require(theta > 1.0F && rotary_dim > 0 && rotary_dim % 2 == 0 && rotary_dim <= 256, "theta > 1, even rotary_dim");
    require(inv_freq.dtype == DType::FP32 && inv_freq.is_contiguous() && inv_freq.numel() == rotary_dim / 2,
            "inv_freq is FP32 [rotary_dim / 2]");
    rope_frequencies_kernel<<<1, 128, 0, stream>>>(theta, rotary_dim, static_cast<float*>(inv_freq.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_rope_table(const Tensor& positions, const Tensor& inv_freq, Tensor& table, cudaStream_t stream) {
    require(positions.dtype == DType::I32 && positions.is_contiguous() && positions.ne[2] == 1 && positions.ne[3] == 1 &&
                (positions.ne[1] == 1 || positions.ne[1] == 3),
            "positions are I32 [T] or [T, 3]");
    const int columns = positions.ne[0], pairs = static_cast<int>(inv_freq.numel());
    require(inv_freq.dtype == DType::FP32 && inv_freq.is_contiguous(), "inv_freq is FP32");
    require(table.dtype == DType::BF16 && table.is_contiguous() && table.ne[0] == pairs && table.ne[1] == columns &&
                table.ne[2] == 2,
            "table is BF16 [pairs, T, 2]");
    const int total = columns * pairs;
    rope_table_kernel<<<(total + 255) / 256, 256, 0, stream>>>(static_cast<const std::int32_t*>(positions.data),
                                                               positions.ne[1], columns,
                                                               static_cast<const float*>(inv_freq.data), pairs,
                                                               static_cast<std::uint16_t*>(table.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_rope_apply(const Tensor& table, Tensor& x, cudaStream_t stream) {
    require(x.dtype == DType::BF16 && x.is_contiguous() && x.ne[3] == 1, "x is contiguous BF16 [D, H, T]");
    const int pairs = table.ne[0], columns = table.ne[1];
    require(table.dtype == DType::BF16 && table.is_contiguous() && table.ne[2] == 2 && x.ne[2] == columns &&
                2 * pairs <= x.ne[0],
            "table [pairs, T, 2] for x [D >= 2 pairs, H, T]");
    const long long total = static_cast<long long>(pairs) * x.ne[1] * columns;
    rope_apply_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(table.data), pairs, static_cast<std::uint16_t*>(x.data), x.ne[0], x.ne[1],
        columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops

// This unit's copy of the noise twin's seed (ops/lossless/noise.cuh).
INFERNIX_LOSSLESS_NOISE_UNIT(rope)
