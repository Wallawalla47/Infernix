// The fused route of hyper_connection_mix: two kernels instead of norm, the down projection,
// gates, the up projection and collapse (docs/maintainer/qwen3_8-flash-next-design.md §8.3 K1a/K1b,
// §19.3.3 Phase 1b), for Q8_G32_FP16 RowSplit weights (Dense8) and BF16 contiguous weights (the
// bit-exact artifact). The weight codec only changes how a lane's 8 weights are read; the order of
// every product and sum is the codec-independent schedule below.
//
// K1a hc_down_partials: one CTA per (stream s, 8 rows of W_down), a row per warp. The CTA stages its
// stream's slice of R for up to 8 columns, reduces each column's sum of squares in a fixed order
// (per-thread sums over its elements in index order, a warp butterfly, warps in order), forms
// Rn = R * inv * (1 + w) in FP32 shared memory and computes the slice's dot products in the
// lane-chunk order (lane l owns weights k = 8l + 256m, fmaf in k order, a warp butterfly). Outputs:
// FP32 partial[s][row][t]; the row-group-0 CTA of each stream writes inv[s][t].
//
// K1b hc_up_collapse: one CTA per 16 features d, a warp per d. Every CTA sums the partials in the
// fixed order s = 0..S-1 into z, forms m = SiLU(z / S) in FP32 shared memory (CTA 0 also writes
// inject = 2 sigmoid(z[rank + s] / S)), computes the S rows (s*H + d) of W_up against m in the same
// lane-chunk order, and writes x[d] = bf16((1/S) * sum_s sigmoid(u_s) * Rn_s) with Rn_s formed in
// FP32 from R, inv and w.
//
// No intermediate is rounded to BF16 (op development §6.1): Rn, z, m and u stay FP32. Each column is
// computed independently in a fixed order, so a column's result does not depend on how many
// columns share the call.

#include "ops/hyper_connection/hyper_connection_mix_fused.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace infernix::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kStreams      = 4;
constexpr int kHidden       = 2560;
constexpr int kChunks       = kHidden / 256; // a lane's 8-weight chunks per stream slice
constexpr int kMaxRank      = 512;
constexpr int kDownRows     = 8; // rows of W_down per CTA (a row per warp)
constexpr int kDownThreads  = kDownRows * 32;
constexpr int kPassColumns  = 8;
constexpr int kUpFeatures   = 16; // features d per CTA (a d per warp)
constexpr int kUpThreads    = kUpFeatures * 32;
constexpr std::size_t kDownSharedBytes = static_cast<std::size_t>(kPassColumns) * kHidden * sizeof(float);

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection_mix ") + what + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    return value;
}

// Fixed-order block sum: warp butterflies, then the warps' sums in warp order.
template <int Threads>
__device__ __forceinline__ float block_sum(float value, float* scratch) {
    value = warp_sum(value);
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < Threads / 32; ++w) { total += scratch[w]; }
    __syncthreads();
    return total;
}

__device__ __forceinline__ float half_bits_to_float(std::uint16_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// q8_g32_fp16 in row_split_k128_v1: int8 codes [rows, padded_k], FP16 scales per 32 codes.
struct Q8Codec {
    struct Params {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
        int padded_k;
    };
    struct Chunk {
        uint2 code;
        float scale;
    };
    __device__ static Chunk load(const Params& p, std::int64_t row, int k) {
        return {ld_nc_na<uint2>(p.codes + row * p.padded_k + k),
                half_bits_to_float(ld_nc_na<std::uint16_t>(p.scales + row * (p.padded_k / 32) * 2 + (k / 32) * 2))};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[i] = static_cast<float>(static_cast<std::int8_t>((i < 4 ? c.code.x : c.code.y) >> ((i & 3) * 8))) *
                   c.scale;
        }
    }
};

// BF16 contiguous: rows of `stride` elements.
struct Bf16Codec {
    struct Params {
        const bf16* data;
        int stride;
    };
    struct Chunk {
        uint4 bits;
    };
    __device__ static Chunk load(const Params& p, std::int64_t row, int k) {
        return {ld_nc_na<uint4>(p.data + row * p.stride + k)};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
        const float2 a = bf16x2_bits_to_float2(c.bits.x), b = bf16x2_bits_to_float2(c.bits.y);
        const float2 d = bf16x2_bits_to_float2(c.bits.z), e = bf16x2_bits_to_float2(c.bits.w);
        w[0] = a.x, w[1] = a.y, w[2] = b.x, w[3] = b.y, w[4] = d.x, w[5] = d.y, w[6] = e.x, w[7] = e.y;
    }
};

template <class Codec>
__global__ __launch_bounds__(kDownThreads, 1) void hc_down_partials_kernel(
    const bf16* __restrict__ residual, const bf16* __restrict__ norm_weight, typename Codec::Params down, int rows,
    int columns, float eps, float* __restrict__ partial, float* __restrict__ inv_out) {
    extern __shared__ __align__(16) float rn[]; // [kPassColumns][kHidden]
    __shared__ float scratch[kDownThreads / 32];
    __shared__ float inv[kPassColumns];
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int s    = static_cast<int>(blockIdx.y);
    const int row  = static_cast<int>(blockIdx.x) * kDownRows + warp;
    const bool live = row < rows;
    constexpr int width = kStreams * kHidden;

    // 1. Weights first: this warp's row of the stream's K slice, before any activation.
    const std::int64_t source = live ? row : 0;
    typename Codec::Chunk chunk[kChunks];
#pragma unroll
    for (int m = 0; m < kChunks; ++m) { chunk[m] = Codec::load(down, source, s * kHidden + m * 256 + lane * 8); }
    const bf16* w = norm_weight + static_cast<std::int64_t>(s) * kHidden;

#pragma unroll 1
    for (int c0 = 0; c0 < columns; c0 += kPassColumns) {
        const int pass = min(kPassColumns, columns - c0);
        // 2. Stage the slice in FP32 and reduce each column's sum of squares in a fixed order.
#pragma unroll 1
        for (int c = 0; c < pass; ++c) {
            const bf16* r = residual + static_cast<std::int64_t>(c0 + c) * width + static_cast<std::int64_t>(s) * kHidden;
            float sum     = 0.0F;
            for (int i = static_cast<int>(threadIdx.x) * 8; i < kHidden; i += kDownThreads * 8) {
                const uint4 values = load_ldg<uint4>(r + i);
                const float2 v0 = bf16x2_bits_to_float2(values.x), v1 = bf16x2_bits_to_float2(values.y);
                const float2 v2 = bf16x2_bits_to_float2(values.z), v3 = bf16x2_bits_to_float2(values.w);
                const float v[8] = {v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};
                float* out = rn + c * kHidden + i;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    out[j] = v[j];
                    sum += v[j] * v[j];
                }
            }
            const float total = block_sum<kDownThreads>(sum, scratch);
            if (threadIdx.x == 0) { inv[c] = rsqrtf(total / static_cast<float>(kHidden) + eps); }
        }
        __syncthreads();
        // Rn = R * inv * (1 + w), FP32.
        for (int i = static_cast<int>(threadIdx.x); i < pass * kHidden; i += kDownThreads) {
            const int c = i / kHidden, d = i - c * kHidden;
            rn[i]       = rn[i] * inv[c] * (1.0F + __bfloat162float(w[d]));
        }
        if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < pass) {
            inv_out[static_cast<std::int64_t>(s) * columns + c0 + static_cast<int>(threadIdx.x)] = inv[threadIdx.x];
        }
        __syncthreads();
        // 3. The row's dot products: the lane-chunk order, then a warp butterfly.
        float acc[kPassColumns];
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) { acc[c] = 0.0F; }
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            const int kk = m * 256 + lane * 8;
            float wv[8];
            Codec::decode(chunk[m], wv);
#pragma unroll
            for (int c = 0; c < kPassColumns; ++c) {
                if (c < pass) {
                    const float4 a = *reinterpret_cast<const float4*>(rn + c * kHidden + kk);
                    const float4 b = *reinterpret_cast<const float4*>(rn + c * kHidden + kk + 4);
                    acc[c] = fmaf(wv[0], a.x, acc[c]);
                    acc[c] = fmaf(wv[1], a.y, acc[c]);
                    acc[c] = fmaf(wv[2], a.z, acc[c]);
                    acc[c] = fmaf(wv[3], a.w, acc[c]);
                    acc[c] = fmaf(wv[4], b.x, acc[c]);
                    acc[c] = fmaf(wv[5], b.y, acc[c]);
                    acc[c] = fmaf(wv[6], b.z, acc[c]);
                    acc[c] = fmaf(wv[7], b.w, acc[c]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            if (c < pass) {
                const float total = warp_sum(acc[c]);
                if (lane == 0 && live) {
                    partial[(static_cast<std::int64_t>(s) * rows + row) * columns + c0 + c] = total;
                }
            }
        }
        __syncthreads(); // the next pass overwrites rn
    }
}

template <class Codec>
__global__ __launch_bounds__(kUpThreads, 1) void hc_up_collapse_kernel(
    const float* __restrict__ partial, const float* __restrict__ inv_in, const bf16* __restrict__ residual,
    const bf16* __restrict__ norm_weight, typename Codec::Params up, int rank, int down_rows, int columns,
    float* __restrict__ inject, bf16* __restrict__ x) {
    __shared__ __align__(16) float m[kHcMixFusedMaxColumns * kMaxRank]; // [T][rank]
    __shared__ float inv[kStreams * kHcMixFusedMaxColumns];             // [S][T]
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int d    = static_cast<int>(blockIdx.x) * kUpFeatures + warp;
    const bool live = d < kHidden;
    constexpr int width = kStreams * kHidden;
    const int chunks    = rank / 8;

    // 1. Weights first: the S rows (s*H + d) of W_up, chunks lane and lane + 32.
    typename Codec::Chunk chunk[kStreams][2];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        const std::int64_t row = static_cast<std::int64_t>(s) * kHidden + (live ? d : 0);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = lane + 32 * h;
            chunk[s][h] = c < chunks ? Codec::load(up, row, c * 8) : typename Codec::Chunk{};
        }
    }

    // 2. z = sum_s partial[s] in order s = 0..S-1; m = SiLU(z / S); CTA 0 writes the injects.
    const float inv_streams = 1.0F / static_cast<float>(kStreams);
    for (int i = static_cast<int>(threadIdx.x); i < down_rows * columns; i += kUpThreads) {
        const int k = i / columns, t = i - k * columns;
        float z     = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) { z += partial[(static_cast<std::int64_t>(s) * down_rows + k) * columns + t]; }
        const float v = z * inv_streams;
        if (k < rank) {
            m[t * rank + k] = v * sigmoidf(v);
        } else if (inject != nullptr && blockIdx.x == 0) {
            inject[static_cast<std::int64_t>(t) * kStreams + (k - rank)] = 2.0F * sigmoidf(v);
        }
    }
    for (int i = static_cast<int>(threadIdx.x); i < kStreams * columns; i += kUpThreads) { inv[i] = inv_in[i]; }
    __syncthreads();
    if (!live) { return; }

    // 3. u_s = W_up[s*H + d] . m in the lane-chunk order, then the collapse in lane 0.
    float w_bias[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        w_bias[s] = 1.0F + __bfloat162float(norm_weight[static_cast<std::int64_t>(s) * kHidden + d]);
    }
#pragma unroll 1
    for (int t = 0; t < columns; ++t) {
        const float* mt = m + t * rank;
        float acc[kStreams];
#pragma unroll
        for (int s = 0; s < kStreams; ++s) { acc[s] = 0.0F; }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = lane + 32 * h;
            if (c < chunks) {
                const float4 a = *reinterpret_cast<const float4*>(mt + c * 8);
                const float4 b = *reinterpret_cast<const float4*>(mt + c * 8 + 4);
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    float wv[8];
                    Codec::decode(chunk[s][h], wv);
                    acc[s] = fmaf(wv[0], a.x, acc[s]);
                    acc[s] = fmaf(wv[1], a.y, acc[s]);
                    acc[s] = fmaf(wv[2], a.z, acc[s]);
                    acc[s] = fmaf(wv[3], a.w, acc[s]);
                    acc[s] = fmaf(wv[4], b.x, acc[s]);
                    acc[s] = fmaf(wv[5], b.y, acc[s]);
                    acc[s] = fmaf(wv[6], b.z, acc[s]);
                    acc[s] = fmaf(wv[7], b.w, acc[s]);
                }
            }
        }
#pragma unroll
        for (int s = 0; s < kStreams; ++s) { acc[s] = warp_sum(acc[s]); }
        if (lane == 0) {
            float sum = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const float r  = __bfloat162float(residual[static_cast<std::int64_t>(t) * width +
                                                           static_cast<std::int64_t>(s) * kHidden + d]);
                const float rn = r * inv[s * columns + t] * w_bias[s];
                sum += sigmoidf(acc[s]) * rn;
            }
            x[static_cast<std::int64_t>(t) * kHidden + d] = __float2bfloat16_rn(sum * inv_streams);
        }
    }
}

bool q8_rowsplit(const Weight& w) noexcept {
    return w.qtype == QType::Q8_G32_FP16 && w.layout == QuantLayout::RowSplit && w.qdata != nullptr &&
           w.scales != nullptr && w.padded_shape[1] % 128 == 0 && w.padded_shape[1] >= w.k;
}

int bf16_stride(const Weight& w) noexcept { return w.padded_shape[1] >= w.k ? w.padded_shape[1] : w.k; }

bool bf16_contiguous(const Weight& w) noexcept {
    return w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.qdata != nullptr &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0 && bf16_stride(w) % 8 == 0;
}

Q8Codec::Params q8_params(const Weight& w) {
    return {static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales), w.padded_shape[1]};
}

Bf16Codec::Params bf16_params(const Weight& w) { return {static_cast<const bf16*>(w.qdata), bf16_stride(w)}; }

template <class Codec>
void launch(const Tensor& residual, const Tensor& norm_weight, typename Codec::Params down,
            typename Codec::Params up, int down_rows, int rank, int streams, float eps, Tensor& x, Tensor* inject,
            void* partials, cudaStream_t stream) {
    static const bool configured = [] {
        const cudaError_t error = cudaFuncSetAttribute(hc_down_partials_kernel<Codec>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       static_cast<int>(kDownSharedBytes));
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("hyper_connection_mix: shared memory: ") + cudaGetErrorString(error));
        }
        return true;
    }();
    (void)configured;
    const int columns = residual.ne[1];
    auto* partial     = static_cast<float*>(partials);
    const std::size_t partial_bytes =
        (static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float) + 255) / 256 * 256;
    auto* inv = reinterpret_cast<float*>(static_cast<std::byte*>(partials) + partial_bytes);
    hc_down_partials_kernel<Codec><<<dim3((down_rows + kDownRows - 1) / kDownRows, streams), kDownThreads,
                                     kDownSharedBytes, stream>>>(
        static_cast<const bf16*>(residual.data), static_cast<const bf16*>(norm_weight.data), down, down_rows,
        columns, eps, partial, inv);
    check_launch("down partials");
    hc_up_collapse_kernel<Codec><<<(kHidden + kUpFeatures - 1) / kUpFeatures, kUpThreads, 0, stream>>>(
        partial, inv, static_cast<const bf16*>(residual.data), static_cast<const bf16*>(norm_weight.data), up, rank,
        down_rows, columns, inject != nullptr ? static_cast<float*>(inject->data) : nullptr,
        static_cast<bf16*>(x.data));
    check_launch("up collapse");
}

} // namespace

bool hc_mix_fused_supported(const Weight& down, const Weight& up, std::int32_t hidden, std::int32_t streams,
                            std::int32_t rank, std::int32_t columns) noexcept {
    const bool codec = (q8_rowsplit(down) && q8_rowsplit(up)) || (bf16_contiguous(down) && bf16_contiguous(up));
    return codec && columns >= 1 && columns <= kHcMixFusedMaxColumns && hidden == kHidden && streams == kStreams &&
           rank > 0 && rank % 8 == 0 && rank <= kMaxRank && down.k == kStreams * kHidden &&
           (down.n == rank || down.n == rank + kStreams) && up.n == kStreams * kHidden && up.k == rank;
}

std::size_t hc_mix_fused_workspace_bytes(std::int32_t down_rows, std::int32_t streams, std::int32_t columns) noexcept {
    const auto align = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    return align(static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float)) +
           align(static_cast<std::size_t>(streams) * columns * sizeof(float));
}

void hc_mix_fused(const Tensor& residual, const Tensor& norm_weight, const Weight& down, const Weight& up,
                  std::int32_t streams, std::int32_t rank, float eps, Tensor& x, Tensor* inject, void* partials,
                  cudaStream_t stream) {
    if (!hc_mix_fused_supported(down, up, x.ne[0], streams, rank, residual.ne[1])) {
        throw std::invalid_argument("hyper_connection_mix: the fused route does not serve this problem");
    }
    if (inject != nullptr && down.n != rank + streams) {
        throw std::invalid_argument("hyper_connection_mix: injects need the down projection's injection rows");
    }
    if (down.qtype == QType::Q8_G32_FP16) {
        launch<Q8Codec>(residual, norm_weight, q8_params(down), q8_params(up), down.n, rank, streams, eps, x, inject,
                        partials, stream);
    } else {
        launch<Bf16Codec>(residual, norm_weight, bf16_params(down), bf16_params(up), down.n, rank, streams, eps, x,
                          inject, partials, stream);
    }
}

} // namespace infernix::ops::detail
