#include "infernix/ops/ple.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ple: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ple ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous();
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

__global__ void embed_kernel(const std::uint8_t* __restrict__ rows, const bf16* __restrict__ scale,
                             int elements, bf16* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= elements) { return; }
    __nv_fp8_e4m3 code;
    code.__x      = rows[static_cast<std::size_t>(t) * elements + i];
    const float v = static_cast<float>(code) * __bfloat162float(scale[0]);
    out[static_cast<std::size_t>(t) * elements + i] = __float2bfloat16_rn(v);
}

// bf16_rn(p * m) of the exact product: hi = fl32(p * m) and its error lo (an FMA residual, exact)
// round together, so lo decides only a hi that lies exactly halfway between two BF16 values.
__device__ __forceinline__ std::uint16_t bf16_rn_product(float p, float m) {
    const float hi         = p * m;
    const float lo         = fmaf(p, m, -hi);
    const std::uint32_t b  = __float_as_uint(hi);
    const std::uint32_t lo16 = b & 0xFFFFU;
    std::uint32_t up;
    if (lo16 == 0x8000U && lo != 0.0F) {
        up = (lo > 0.0F) == (hi > 0.0F) ? 1U : 0U; // the exact magnitude lies above or below the tie
    } else {
        up = lo16 > 0x8000U || (lo16 == 0x8000U && (b & 0x10000U) != 0);
    }
    return static_cast<std::uint16_t>((b >> 16) + up);
}

// One CTA per column; each task decodes one 16-element block of one head's row.
__global__ void __launch_bounds__(kThreads)
    embed_nvfp4_kernel(const std::uint8_t* __restrict__ rows, const float* __restrict__ multiplier, int row_bytes,
                       int heads, int width, std::uint16_t* __restrict__ out) {
    constexpr float kE2m1[16] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
                                 -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
    const int t      = blockIdx.x;
    const int blocks = width / 16;
    const float m    = multiplier[0];
    for (int task = threadIdx.x; task < heads * blocks; task += blockDim.x) {
        const int h = task / blocks, b = task % blocks;
        const std::uint8_t* row = rows + (static_cast<std::size_t>(t) * heads + h) * row_bytes;
        __nv_fp8_e4m3 code;
        code.__x       = row[width / 2 + b];
        const float s  = static_cast<float>(code);
        std::uint16_t* o = out + static_cast<std::size_t>(t) * heads * width + static_cast<std::size_t>(h) * width + 16 * b;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint8_t pair = row[8 * b + i];
            o[2 * i]     = bf16_rn_product(kE2m1[pair & 15U] * s, m); // E2M1 times E4M3: exact in FP32
            o[2 * i + 1] = bf16_rn_product(kE2m1[pair >> 4U] * s, m);
        }
    }
}

__device__ __forceinline__ float norm_value(bf16 w) { return __bfloat162float(w); }
__device__ __forceinline__ float norm_value(float w) { return w; }

// One CTA per (stream, column). The norm weights are BF16, or FP32 (NVIDIA FP4-Local's).
template <class Norm>
__global__ void __launch_bounds__(kThreads)
    gate_kernel(const bf16* __restrict__ key, const bf16* __restrict__ value,
                const bf16* __restrict__ residual, const Norm* __restrict__ key_norm,
                const Norm* __restrict__ query_norm, const Norm* __restrict__ conv_norm, int hidden,
                int width, float eps, bf16* __restrict__ gated, bf16* __restrict__ normalized) {
    __shared__ float scratch[kThreads / 32];
    const int s = blockIdx.x, t = blockIdx.y;
    const std::size_t base = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
    float key_sq = 0.0F, query_sq = 0.0F;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const float k = __bfloat162float(key[base + d]);
        const float q = __bfloat162float(residual[base + d]);
        key_sq += k * k;
        query_sq += q * q;
    }
    const float key_inv   = rsqrtf(block_sum(key_sq, scratch) / static_cast<float>(hidden) + eps);
    const float query_inv = rsqrtf(block_sum(query_sq, scratch) / static_cast<float>(hidden) + eps);
    float dot = 0.0F;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
        const float k = __bfloat162float(key[base + d]) * key_inv * (1.0F + norm_value(key_norm[c]));
        const float q = __bfloat162float(residual[base + d]) * query_inv * (1.0F + norm_value(query_norm[c]));
        dot += k * q;
    }
    float g = block_sum(dot, scratch) / sqrtf(static_cast<float>(hidden));
    g       = copysignf(sqrtf(fmaxf(fabsf(g), 1.0e-6F)), g);
    const float gate = sigmoidf(g);
    float gated_sq   = 0.0F;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const float v = gate * __bfloat162float(value[static_cast<std::size_t>(t) * hidden + d]);
        gated[base + d] = __float2bfloat16_rn(v);
        gated_sq += v * v;
    }
    const float gated_inv = rsqrtf(block_sum(gated_sq, scratch) / static_cast<float>(hidden) + eps);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
        const float v = gate * __bfloat162float(value[static_cast<std::size_t>(t) * hidden + d]);
        normalized[base + d] = __float2bfloat16_rn(v * gated_inv * (1.0F + norm_value(conv_norm[c])));
    }
}

// One thread per (channel, column): reads the pre-update history.
__global__ void conv_kernel(const bf16* __restrict__ gated, const bf16* __restrict__ normalized,
                            const bf16* __restrict__ weight, int taps, int dilation, int span,
                            int channels, int width, const bf16* __restrict__ states,
                            const std::int32_t* __restrict__ source_slots, int columns,
                            bf16* __restrict__ residual) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= channels || t >= columns) { return; }
    const int sequence = t / width, local = t % width;
    const int slot     = source_slots[sequence];
    float acc          = 0.0F;
    for (int j = 0; j < taps; ++j) {
        const int source = local - (taps - 1 - j) * dilation;
        float x;
        if (source >= 0) {
            x = __bfloat162float(normalized[static_cast<std::size_t>(sequence * width + source) * channels + c]);
        } else {
            x = __bfloat162float(states[(static_cast<std::size_t>(slot) * span + (span + source)) * channels + c]);
        }
        acc += __bfloat162float(weight[static_cast<std::size_t>(j) * channels + c]) * x;
    }
    const std::size_t i = static_cast<std::size_t>(t) * channels + c;
    const float out     = __bfloat162float(gated[i]) + acc * sigmoidf(acc);
    residual[i]         = __float2bfloat16_rn(__bfloat162float(residual[i]) + out);
}

// One thread per (channel, sequence) shifts the history in increasing order, so an in-place
// update (source == destination slot) never reads an element it already overwrote.
__global__ void state_kernel(const bf16* __restrict__ normalized, int span, int channels, int width,
                             int sequences, bf16* __restrict__ states,
                             const std::int32_t* __restrict__ source_slots,
                             const std::int32_t* __restrict__ destination_slots) {
    const int c        = blockIdx.x * blockDim.x + threadIdx.x;
    const int sequence = blockIdx.y;
    if (c >= channels || sequence >= sequences) { return; }
    const int src = source_slots[sequence], dst = destination_slots[sequence];
    for (int k = 0; k < span; ++k) {
        const int index = width - span + k; // column relative to the chunk start
        bf16 v;
        if (index >= 0) {
            v = normalized[static_cast<std::size_t>(sequence * width + index) * channels + c];
        } else {
            v = states[(static_cast<std::size_t>(src) * span + (span + index)) * channels + c];
        }
        states[(static_cast<std::size_t>(dst) * span + k) * channels + c] = v;
    }
}

// ple_conv_commit: the state_kernel shift with a per-row column count, reading row b's columns
// at b * stride.
__global__ void commit_kernel(const bf16* __restrict__ normalized, int span, int channels, int stride,
                              const std::int32_t* __restrict__ commit_columns, bf16* __restrict__ states,
                              const std::int32_t* __restrict__ slots) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels) { return; }
    const int n = commit_columns[b];
    if (n <= 0) { return; }
    const int slot = slots[b];
    for (int k = 0; k < span; ++k) {
        const int index = n - span + k;
        bf16 v;
        if (index >= 0) {
            v = normalized[static_cast<std::size_t>(b * stride + index) * channels + c];
        } else {
            v = states[(static_cast<std::size_t>(slot) * span + (span + index)) * channels + c];
        }
        states[(static_cast<std::size_t>(slot) * span + k) * channels + c] = v;
    }
}

} // namespace

void ple_embed(const Tensor& rows, const Tensor& scale, Tensor& out, cudaStream_t stream) {
    require(contiguous(rows, DType::U8) && contiguous(scale, DType::BF16) && contiguous(out, DType::BF16),
            "embed requires contiguous U8 rows, BF16 scale and BF16 output");
    const int elements = rows.ne[0] * rows.ne[1];
    const int columns  = rows.ne[2];
    require(scale.numel() == 1 && out.ne[0] == elements && out.ne[1] == columns && columns > 0,
            "embed shapes disagree");
    embed_kernel<<<dim3((elements + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(rows.data), static_cast<const bf16*>(scale.data), elements,
        static_cast<bf16*>(out.data));
    check_launch("embed");
}

void ple_embed_nvfp4(const Tensor& rows, const Tensor& multiplier, Tensor& out, cudaStream_t stream) {
    require(contiguous(rows, DType::U8) && contiguous(multiplier, DType::FP32) && contiguous(out, DType::BF16),
            "NVFP4 embed requires contiguous U8 rows, an FP32 multiplier and BF16 output");
    const int row_bytes = rows.ne[0], heads = rows.ne[1], columns = rows.ne[2];
    require(row_bytes % 9 == 0 && heads > 0 && columns > 0 && multiplier.numel() == 1,
            "NVFP4 rows must be whole 16-element blocks (8 code bytes and a scale each)");
    const int width = row_bytes / 9 * 16;
    require(out.ne[0] == heads * width && out.ne[1] == columns, "NVFP4 embed shapes disagree");
    embed_nvfp4_kernel<<<columns, kThreads, 0, stream>>>(static_cast<const std::uint8_t*>(rows.data),
                                                         static_cast<const float*>(multiplier.data), row_bytes, heads,
                                                         width, static_cast<std::uint16_t*>(out.data));
    check_launch("embed nvfp4");
}

void ple_gate(const Tensor& key, const Tensor& value, const Tensor& residual,
              const Tensor& key_norm, const Tensor& query_norm, const Tensor& conv_norm,
              std::int32_t streams, float eps, Tensor& gated, Tensor& normalized,
              cudaStream_t stream) {
    for (const Tensor* t : {&key, &value, &residual, static_cast<const Tensor*>(&gated),
                            static_cast<const Tensor*>(&normalized)}) {
        require(contiguous(*t, DType::BF16), "gate requires contiguous BF16 tensors");
    }
    const DType norm = key_norm.dtype;
    require((norm == DType::BF16 || norm == DType::FP32) && contiguous(key_norm, norm) &&
                contiguous(query_norm, norm) && contiguous(conv_norm, norm),
            "gate norms must be contiguous, all BF16 or all FP32");
    const int width = key.ne[0], columns = key.ne[1];
    require(streams > 0 && width % streams == 0, "gate width must hold whole streams");
    const int hidden = width / streams;
    require(value.ne[0] == hidden && value.ne[1] == columns && residual.ne[0] == width &&
                residual.ne[1] == columns && key_norm.numel() == width && query_norm.numel() == width &&
                conv_norm.numel() == width && gated.ne[0] == width && gated.ne[1] == columns &&
                normalized.ne[0] == width && normalized.ne[1] == columns && columns > 0 && eps > 0,
            "gate shapes disagree");
    const auto launch = [&]<class Norm>() {
        gate_kernel<Norm><<<dim3(streams, columns), kThreads, 0, stream>>>(
            static_cast<const bf16*>(key.data), static_cast<const bf16*>(value.data),
            static_cast<const bf16*>(residual.data), static_cast<const Norm*>(key_norm.data),
            static_cast<const Norm*>(query_norm.data), static_cast<const Norm*>(conv_norm.data), hidden, width, eps,
            static_cast<bf16*>(gated.data), static_cast<bf16*>(normalized.data));
    };
    if (norm == DType::FP32) {
        launch.template operator()<float>();
    } else {
        launch.template operator()<bf16>();
    }
    check_launch("gate");
}

void ple_conv_inject(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                     std::int32_t dilation, Tensor& states, const Tensor& source_slots,
                     const Tensor& destination_slots, Tensor& residual, cudaStream_t stream) {
    require(contiguous(gated, DType::BF16) && contiguous(normalized, DType::BF16) &&
                contiguous(weight, DType::BF16) && contiguous(states, DType::BF16) &&
                contiguous(residual, DType::BF16) && contiguous(source_slots, DType::I32) &&
                (destination_slots.data == nullptr || contiguous(destination_slots, DType::I32)),
            "conv requires contiguous tensors");
    const int channels = gated.ne[0], columns = gated.ne[1], taps = weight.ne[1];
    const int sequences = source_slots.ne[0];
    const bool update   = destination_slots.data != nullptr;
    require(dilation > 0 && taps > 1 && weight.ne[0] == channels && normalized.ne[0] == channels &&
                normalized.ne[1] == columns && residual.ne[0] == channels && residual.ne[1] == columns &&
                sequences > 0 && (!update || destination_slots.ne[0] == sequences) && columns % sequences == 0,
            "conv shapes disagree");
    const int span = (taps - 1) * dilation;
    require(states.ne[0] == channels && states.ne[1] == span, "conv state must be [C, span, slots]");
    const int width = columns / sequences;
    conv_kernel<<<dim3((channels + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(gated.data), static_cast<const bf16*>(normalized.data),
        static_cast<const bf16*>(weight.data), taps, dilation, span, channels, width,
        static_cast<const bf16*>(states.data), static_cast<const std::int32_t*>(source_slots.data),
        columns, static_cast<bf16*>(residual.data));
    check_launch("conv");
    if (!update) { return; }
    state_kernel<<<dim3((channels + kThreads - 1) / kThreads, sequences), kThreads, 0, stream>>>(
        static_cast<const bf16*>(normalized.data), span, channels, width, sequences,
        static_cast<bf16*>(states.data), static_cast<const std::int32_t*>(source_slots.data),
        static_cast<const std::int32_t*>(destination_slots.data));
    check_launch("state");
}

void ple_conv_commit(const Tensor& normalized, const Tensor& commit_columns, Tensor& states,
                     const Tensor& slots, cudaStream_t stream) {
    require(contiguous(normalized, DType::BF16) && contiguous(commit_columns, DType::I32) &&
                contiguous(states, DType::BF16) && contiguous(slots, DType::I32),
            "commit requires contiguous tensors");
    const int channels = normalized.ne[0], width = normalized.ne[1], rows = normalized.ne[2];
    require(rows > 0 && width > 0 && commit_columns.ne[0] == rows && slots.ne[0] == rows &&
                states.ne[0] == channels && states.ne[1] > 0,
            "commit shapes disagree");
    commit_kernel<<<dim3((channels + kThreads - 1) / kThreads, rows), kThreads, 0, stream>>>(
        static_cast<const bf16*>(normalized.data), states.ne[1], channels, width,
        static_cast<const std::int32_t*>(commit_columns.data), static_cast<bf16*>(states.data),
        static_cast<const std::int32_t*>(slots.data));
    check_launch("commit");
}

} // namespace infernix::ops
