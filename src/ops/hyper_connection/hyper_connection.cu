#include "ninfer/ops/hyper_connection.h"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("hyper_connection: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection ") + what + ": " +
                                 cudaGetErrorString(error));
    }
}

bool contiguous_2d(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < blockDim.x / 32; ++w) { total += scratch[w]; }
    __syncthreads();
    return total;
}

// One CTA per (stream, column).
__global__ void __launch_bounds__(kThreads) norm_kernel(const bf16* __restrict__ residual,
                                                      const bf16* __restrict__ weight, int hidden,
                                                      int width, float eps, bf16* __restrict__ out) {
    __shared__ float scratch[kThreads / 32];
    const int s = blockIdx.x, t = blockIdx.y;
    const bf16* x = residual + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    bf16* y       = out + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    const bf16* w = weight + static_cast<std::size_t>(s) * hidden;
    float sum = 0.0F;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const float v = __bfloat162float(x[d]);
        sum += v * v;
    }
    const float inv = rsqrtf(block_sum(sum, scratch) / static_cast<float>(hidden) + eps);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        y[d] = __float2bfloat16_rn(__bfloat162float(x[d]) * inv * (1.0F + __bfloat162float(w[d])));
    }
}

__global__ void gates_kernel(const bf16* __restrict__ z, int rows, int rank, int streams, int columns,
                             bf16* __restrict__ mix, float* __restrict__ inject) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= rows || t >= columns) { return; }
    const float scale = 1.0F / static_cast<float>(streams);
    const float v     = __bfloat162float(z[static_cast<std::size_t>(t) * rows + i]) * scale;
    if (i < rank) {
        mix[static_cast<std::size_t>(t) * rank + i] = __float2bfloat16_rn(v * sigmoidf(v));
    } else if (inject != nullptr) {
        inject[static_cast<std::size_t>(t) * streams + (i - rank)] = 2.0F * sigmoidf(v);
    }
}

__global__ void collapse_kernel(const bf16* __restrict__ logits, const bf16* __restrict__ normalized,
                                int hidden, int streams, int columns, bf16* __restrict__ out) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    float sum = 0.0F;
    for (int s = 0; s < streams; ++s) {
        const std::size_t i = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d;
        sum += sigmoidf(__bfloat162float(logits[i])) * __bfloat162float(normalized[i]);
    }
    out[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(sum / static_cast<float>(streams));
}

__global__ void inject_kernel(const bf16* __restrict__ y, const float* __restrict__ inject, int hidden,
                              int streams, int columns, bf16* __restrict__ residual) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    const float v           = __bfloat162float(y[static_cast<std::size_t>(t) * hidden + d]);
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int s = 0; s < streams; ++s) {
        bf16& r = residual[static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d];
        r       = __float2bfloat16_rn(__bfloat162float(r) + inject[static_cast<std::size_t>(t) * streams + s] * v);
    }
}

__global__ void expand_kernel(const bf16* __restrict__ x, int hidden, int streams, int columns,
                              bf16* __restrict__ residual) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    const bf16 v            = x[static_cast<std::size_t>(t) * hidden + d];
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int s = 0; s < streams; ++s) {
        residual[static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d] = v;
    }
}

} // namespace

void hyper_connection_norm(const Tensor& residual, const Tensor& weight, std::int32_t streams,
                           float eps, Tensor& out, cudaStream_t stream) {
    require(streams > 0 && contiguous_2d(residual, DType::BF16) && contiguous_2d(out, DType::BF16) &&
                contiguous_2d(weight, DType::BF16),
            "norm requires contiguous BF16 tensors");
    const int width = residual.ne[0], columns = residual.ne[1];
    require(width % streams == 0 && (width / streams) % 8 == 0 && weight.ne[0] == width &&
                weight.ne[1] == 1 && out.ne[0] == width && out.ne[1] == columns && columns > 0,
            "norm shapes disagree");
    require(eps > 0, "norm epsilon must be positive");
    norm_kernel<<<dim3(streams, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(residual.data), static_cast<const bf16*>(weight.data),
        width / streams, width, eps, static_cast<bf16*>(out.data));
    check_launch("norm");
}

void hyper_connection_gates(const Tensor& z, std::int32_t rank, std::int32_t streams, Tensor& mix,
                            Tensor* inject, cudaStream_t stream) {
    require(contiguous_2d(z, DType::BF16) && contiguous_2d(mix, DType::BF16) && rank > 0 && streams > 0,
            "gates require contiguous BF16 tensors");
    const int columns = z.ne[1];
    const int rows    = rank + (inject != nullptr ? streams : 0);
    require(z.ne[0] == rows && mix.ne[0] == rank && mix.ne[1] == columns && columns > 0,
            "gates shapes disagree");
    if (inject != nullptr) {
        require(contiguous_2d(*inject, DType::FP32) && inject->ne[0] == streams &&
                    inject->ne[1] == columns,
                "gates inject output must be FP32 [S, T]");
    }
    gates_kernel<<<dim3((rows + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(z.data), rows, rank, streams, columns, static_cast<bf16*>(mix.data),
        inject != nullptr ? static_cast<float*>(inject->data) : nullptr);
    check_launch("gates");
}

void hyper_connection_collapse(const Tensor& mix_logits, const Tensor& normalized,
                               std::int32_t streams, Tensor& out, cudaStream_t stream) {
    require(contiguous_2d(mix_logits, DType::BF16) && contiguous_2d(normalized, DType::BF16) &&
                contiguous_2d(out, DType::BF16) && streams > 0,
            "collapse requires contiguous BF16 tensors");
    const int width = mix_logits.ne[0], columns = mix_logits.ne[1], hidden = out.ne[0];
    require(width == streams * hidden && normalized.ne[0] == width && normalized.ne[1] == columns &&
                out.ne[1] == columns && columns > 0,
            "collapse shapes disagree");
    collapse_kernel<<<dim3((hidden + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(mix_logits.data), static_cast<const bf16*>(normalized.data), hidden,
        streams, columns, static_cast<bf16*>(out.data));
    check_launch("collapse");
}

void hyper_connection_inject(const Tensor& y, const Tensor& inject, Tensor& residual,
                             cudaStream_t stream) {
    require(contiguous_2d(y, DType::BF16) && contiguous_2d(inject, DType::FP32) &&
                contiguous_2d(residual, DType::BF16),
            "inject requires contiguous tensors");
    const int hidden = y.ne[0], columns = y.ne[1], streams = inject.ne[0];
    require(inject.ne[1] == columns && residual.ne[0] == streams * hidden && residual.ne[1] == columns &&
                columns > 0,
            "inject shapes disagree");
    inject_kernel<<<dim3((hidden + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(y.data), static_cast<const float*>(inject.data), hidden, streams,
        columns, static_cast<bf16*>(residual.data));
    check_launch("inject");
}

void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream) {
    require(contiguous_2d(x, DType::BF16) && contiguous_2d(residual, DType::BF16) && streams > 0,
            "expand requires contiguous BF16 tensors");
    const int hidden = x.ne[0], columns = x.ne[1];
    require(residual.ne[0] == streams * hidden && residual.ne[1] == columns && columns > 0,
            "expand shapes disagree");
    expand_kernel<<<dim3((hidden + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x.data), hidden, streams, columns, static_cast<bf16*>(residual.data));
    check_launch("expand");
}

} // namespace ninfer::ops
