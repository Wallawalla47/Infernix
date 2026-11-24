// The --lossless route's LoRA adapter (include/infernix/ops/lossless_lora.h; docs/maintainer/lossless-route.md §2):
// u = fp32(A x), d = fp32(B u), y = bf16(fp32(y + fp32(d fp32(s)))). Both sums are lossless_linear's exact sums over
// BF16 operands: an FP32 value v is exactly v0 + v1 + v2 with each part a BF16 value (its significand's next eight
// bits; exact unless v is a nonzero value below 2^-110, whose last part falls under BF16's smallest subnormal), so
// A x is the exact sum over A's three planes against x stacked three times, and B u the exact sum over the nine
// products of B's and u's parts. u's parts are taken from u scaled by a power of two per column (its largest value
// near 2^120), which lossless_linear's column scale undoes exactly.

#include "infernix/ops/lossless_lora.h"

#include "core/device.h"
#include "core/weight.h"
#include "infernix/ops/lossless_linear.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace infernix::ops {
namespace {

constexpr std::int32_t kChunk = 512; // lossless_linear's column chunk

std::int32_t nine_k(std::int32_t rank) { return (9 * rank + 63) / 64 * 64; }

std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// A contiguous BF16 [n, k] matrix as lossless_linear's BF16 weight.
Weight bf16_weight(const void* data, std::int32_t n, std::int32_t k) {
    Weight w;
    w.payload       = data;
    w.payload_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k) * 2;
    w.qtype         = QType::BF16;
    w.layout        = QuantLayout::Contiguous;
    w.ndim          = 2;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.qdata                        = data;
    w.n                            = n;
    w.k                            = k;
    w.scale_dtype                  = DType::BF16;
    return w;
}

__host__ __device__ inline std::uint32_t float_bits(float f) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
#endif
}

__host__ __device__ inline float bits_float(std::uint32_t b) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(b);
#else
    float f;
    std::memcpy(&f, &b, 4);
    return f;
#endif
}

// v = p0 + p1 + p2, each a BF16 value (truncations of the FP32 pattern); false when p2 is not exactly one.
__host__ __device__ inline bool split3_value(float v, std::uint16_t (&parts)[3]) {
    const float p0 = bits_float(float_bits(v) & 0xFFFF0000U);
    const float r1 = v - p0; // exact: v's low sixteen bits
    const float p1 = bits_float(float_bits(r1) & 0xFFFF0000U);
    const float p2 = r1 - p1; // exact: r1's low bits
    parts[0]       = static_cast<std::uint16_t>(float_bits(p0) >> 16);
    parts[1]       = static_cast<std::uint16_t>(float_bits(p1) >> 16);
    parts[2]       = static_cast<std::uint16_t>(float_bits(p2) >> 16);
    return (float_bits(p2) & 0xFFFFU) == 0U && (float_bits(v) & 0x7F800000U) != 0x7F800000U;
}

// u's parts for the nine-product sum: x9[9 j + 3 a + b, t] = part b of u[u_row + j, t] 2^e(t) (paired with B's part
// a), zero past 9 rank; scales[t] = 2^-e(t), or NaN when a part is not exact. One block a column.
__global__ void lossless_lora_split_u_kernel(const float* __restrict__ u, std::int32_t u_rows, std::int32_t u_row,
                                             std::int32_t rank, std::int32_t k9, std::int32_t column0,
                                             __nv_bfloat16* __restrict__ x9, float* __restrict__ scales) {
    const int t      = static_cast<int>(blockIdx.x);
    const float* col = u + static_cast<std::int64_t>(column0 + t) * u_rows + u_row;
    __shared__ int max_exponent;
    __shared__ int inexact;
    if (threadIdx.x == 0) {
        max_exponent = -1000;
        inexact      = 0;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < rank; j += blockDim.x) {
        const float v = col[j];
        if (v != 0.0F) { atomicMax(&max_exponent, ilogbf(v)); }
    }
    __syncthreads();
    // The largest value near 2^120; the scale 2^-e stays a normal binary32.
    const int e = max_exponent == -1000 ? 0 : max(-126, min(126, 120 - max_exponent));
    auto* out   = reinterpret_cast<std::uint16_t*>(x9) + static_cast<std::int64_t>(t) * k9;
    for (int j = threadIdx.x; j < rank; j += blockDim.x) {
        std::uint16_t parts[3];
        if (!split3_value(ldexpf(col[j], e), parts)) { inexact = 1; }
#pragma unroll
        for (int a = 0; a < 3; ++a) {
#pragma unroll
            for (int b = 0; b < 3; ++b) { out[9 * j + 3 * a + b] = parts[b]; }
        }
    }
    for (int k = 9 * rank + threadIdx.x; k < k9; k += blockDim.x) { out[k] = 0; }
    __syncthreads();
    if (threadIdx.x == 0) { scales[t] = inexact != 0 ? __int_as_float(0x7FC00000) : ldexpf(1.0F, -e); }
}

// B's parts: w9[n, 9 j + 3 a + b] = part a of B[n, j], zero past 9 rank (B's values split exactly: checked at load).
__global__ void lossless_lora_expand_b_kernel(const float* __restrict__ b, std::int32_t rows, std::int32_t rank,
                                              std::int32_t k9, __nv_bfloat16* __restrict__ w9) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * k9;
    auto* out                = reinterpret_cast<std::uint16_t*>(w9);
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t n = i / k9;
        const int k          = static_cast<int>(i % k9);
        std::uint16_t value  = 0;
        if (k < 9 * rank) {
            const int j = k / 9, a = (k % 9) / 3;
            std::uint16_t parts[3];
            (void)split3_value(b[n * rank + j], parts);
            value = a == 0 ? parts[0] : a == 1 ? parts[1] : parts[2]; // selection: a computed index spills the array
        }
        out[i] = value;
    }
}

// y[y_row + n, column0 + t] = bf16(fp32(y + fp32(d fp32(s)))).
__global__ void lossless_lora_apply_kernel(__nv_bfloat16* __restrict__ y, std::int32_t y_rows, std::int32_t y_row,
                                           const float* __restrict__ d, std::int32_t rows, std::int32_t columns,
                                           std::int32_t column0, float scale) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * columns;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t t = i / rows;
        const std::int64_t n = i % rows;
        __nv_bfloat16& out   = y[(column0 + t) * y_rows + y_row + n];
        const float term     = __fmul_rn(d[t * rows + n], scale);
        out                  = __float2bfloat16_rn(__fadd_rn(__bfloat162float(out), term));
    }
}

// The FP32 output's form (Inkling 2's message gates, whose forget and message projections are FP32 modules):
// y[y_row + n, column0 + t] = fp32(y + fp32(d fp32(s))). A chunk's rows x columns fits 31 bits (lossless_lora_add), so
// the element index divides in 32 bits.
__global__ void lossless_lora_apply_fp32_kernel(float* __restrict__ y, std::int32_t y_rows, std::int32_t y_row,
                                                const float* __restrict__ d, std::int32_t rows, std::int32_t columns,
                                                std::int32_t column0, float scale) {
    const auto urows  = static_cast<std::uint32_t>(rows);
    const std::uint32_t total = urows * static_cast<std::uint32_t>(columns);
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const std::uint32_t t = i / urows;
        const std::uint32_t n = i % urows;
        float& out = y[(static_cast<std::int64_t>(column0) + t) * y_rows + y_row + n];
        out        = __fadd_rn(out, __fmul_rn(d[i], scale));
    }
}

unsigned blocks_for(std::int64_t total) {
    return static_cast<unsigned>(std::min<std::int64_t>((total + 255) / 256, 4096));
}

} // namespace

bool lossless_lora_split3(const float* values, std::int64_t rows, std::int64_t cols, std::vector<std::uint16_t>& out) {
    out.assign(static_cast<std::size_t>(rows * cols * 3), 0);
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            std::uint16_t parts[3];
            if (!split3_value(values[r * cols + c], parts)) { return false; }
            for (int a = 0; a < 3; ++a) { out[static_cast<std::size_t>((r * 3 + a) * cols + c)] = parts[a]; }
        }
    }
    return true;
}

std::size_t lossless_lora_shrink_workspace_bytes(std::int32_t rows, std::int32_t k, std::int32_t columns) {
    const std::int32_t chunk = std::min(columns, kChunk);
    const Weight a3          = bf16_weight(nullptr, rows, 3 * k);
    return aligned(static_cast<std::size_t>(3) * k * chunk * 2) + lossless_linear_workspace_bytes(a3, rows, chunk);
}

void lossless_lora_shrink(const Tensor& x, const void* a3, std::int32_t rows, Tensor& u, WorkspaceArena& workspace,
                          cudaStream_t stream) {
    const std::int32_t k = x.ne[0], columns = x.ne[1];
    if (x.dtype != DType::BF16 || !x.is_contiguous() || k % 64 != 0 || u.dtype != DType::FP32 || !u.is_contiguous() ||
        u.ne[0] != rows || u.ne[1] != columns || a3 == nullptr) {
        throw std::invalid_argument("lossless_lora_shrink: x BF16 [K, T] (K a multiple of 64), u FP32 [rows, T]");
    }
    const Weight weight = bf16_weight(a3, rows, 3 * k);
    for (std::int32_t c0 = 0; c0 < columns; c0 += kChunk) {
        const std::int32_t n = std::min(kChunk, columns - c0);
        auto scope           = workspace.scope();
        Tensor x3            = workspace.alloc(DType::BF16, {3 * k, n});
        // x stacked three times: column t's K values at 0, K and 2 K.
        for (int a = 0; a < 3; ++a) {
            CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::uint16_t*>(x3.data) + static_cast<std::int64_t>(a) * k,
                                         static_cast<std::size_t>(3) * k * 2,
                                         static_cast<const std::uint16_t*>(x.data) + static_cast<std::int64_t>(c0) * k,
                                         static_cast<std::size_t>(k) * 2, static_cast<std::size_t>(k) * 2, n,
                                         cudaMemcpyDeviceToDevice, stream));
        }
        Tensor uc = u.slice(1, c0, n);
        lossless_linear(x3, weight, 0, rows, uc, 0, workspace, stream);
    }
}

std::size_t lossless_lora_add_workspace_bytes(std::int32_t rows, std::int32_t rank, std::int32_t columns) {
    const std::int32_t chunk = std::min(columns, kChunk), k9 = nine_k(rank);
    const Weight w9          = bf16_weight(nullptr, rows, k9);
    return aligned(static_cast<std::size_t>(rows) * k9 * 2) + aligned(static_cast<std::size_t>(k9) * chunk * 2) +
           aligned(static_cast<std::size_t>(chunk) * 4) + aligned(static_cast<std::size_t>(rows) * chunk * 4) +
           lossless_linear_workspace_bytes(w9, rows, chunk);
}

void lossless_lora_add(Tensor& y, std::int32_t y_row, const float* b, std::int32_t rows, std::int32_t rank,
                       const Tensor& u, std::int32_t u_row, float scale, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    const std::int32_t columns = y.ne[1];
    if ((y.dtype != DType::BF16 && y.dtype != DType::FP32) || !y.is_contiguous() || y_row < 0 || y_row + rows > y.ne[0] ||
        u.dtype != DType::FP32 || !u.is_contiguous() || u.ne[1] != columns || u_row < 0 || u_row + rank > u.ne[0] ||
        rank <= 0 || b == nullptr || (y.dtype == DType::FP32 && rows > INT32_MAX / kChunk)) {
        throw std::invalid_argument("lossless_lora_add: y BF16 or FP32 [*, T], u FP32 [*, T], B FP32 [rows, rank]");
    }
    const std::int32_t k9 = nine_k(rank);
    auto scope            = workspace.scope();
    Tensor w9             = workspace.alloc(DType::BF16, {k9, rows});
    lossless_lora_expand_b_kernel<<<blocks_for(static_cast<std::int64_t>(rows) * k9), 256, 0, stream>>>(
        b, rows, rank, k9, static_cast<__nv_bfloat16*>(w9.data));
    CUDA_CHECK(cudaGetLastError());
    const Weight weight = bf16_weight(w9.data, rows, k9);
    for (std::int32_t c0 = 0; c0 < columns; c0 += kChunk) {
        const std::int32_t n = std::min(kChunk, columns - c0);
        auto chunk_scope     = workspace.scope();
        Tensor x9            = workspace.alloc(DType::BF16, {k9, n});
        Tensor scales        = workspace.alloc(DType::FP32, {n});
        Tensor d             = workspace.alloc(DType::FP32, {rows, n});
        lossless_lora_split_u_kernel<<<static_cast<unsigned>(n), 64, 0, stream>>>(
            static_cast<const float*>(u.data), u.ne[0], u_row, rank, k9, c0, static_cast<__nv_bfloat16*>(x9.data),
            static_cast<float*>(scales.data));
        CUDA_CHECK(cudaGetLastError());
        LosslessActivationScale column_scale;
        column_scale.column_scales = static_cast<const float*>(scales.data);
        lossless_linear(x9, weight, 0, rows, d, 0, workspace, stream, column_scale);
        if (y.dtype == DType::FP32) {
            lossless_lora_apply_fp32_kernel<<<blocks_for(static_cast<std::int64_t>(rows) * n), 256, 0, stream>>>(
                static_cast<float*>(y.data), y.ne[0], y_row, static_cast<const float*>(d.data), rows, n, c0, scale);
        } else {
            lossless_lora_apply_kernel<<<blocks_for(static_cast<std::int64_t>(rows) * n), 256, 0, stream>>>(
                static_cast<__nv_bfloat16*>(y.data), y.ne[0], y_row, static_cast<const float*>(d.data), rows, n, c0,
                scale);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace infernix::ops
