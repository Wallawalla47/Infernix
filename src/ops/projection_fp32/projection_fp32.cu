#include "ninfer/ops/projection_fp32.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads     = 256;
constexpr int kWarps       = kThreads / 32;
constexpr int kRowsPerWarp = 4;
constexpr int kRowsPerCta  = kWarps * kRowsPerWarp;
constexpr int kColumns     = 8; // columns staged per CTA
constexpr int kMaxSegments = 4;
constexpr int kMaxK        = 3072;
constexpr int kMaxLaneChunks = kMaxK / 8 / 32; // 8-element chunks one lane reads per row

struct Segments {
    const bf16* data[kMaxSegments];
    int begin[kMaxSegments + 1]; // first output row of each segment, then the total
    int count;
};

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("projection_fp32: ") + message); }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// Warp w of a CTA owns rows first + w, first + w + Warps, ...; lane l accumulates the 8-element
// chunks l, l + 32, ... in order and the warp reduces by a fixed butterfly, so an output's bits
// depend only on K. A lane issues all of a row's chunk loads before its FMAs. Decode-width calls
// give each warp one row (Threads = 128, RowsPerWarp = 1: 129 CTAs for a 513-row router instead of
// 17), so a router's latency is one DRAM round trip instead of ~40; wide calls keep four rows per
// warp, which stages x in fewer CTAs. Both mappings produce the same bits.
template <int Threads, int RowsPerWarp>
__global__ void __launch_bounds__(Threads)
    projection_kernel(const bf16* __restrict__ x, int k, int columns, Segments segments, float* __restrict__ out) {
    constexpr int Warps = Threads / 32;
    extern __shared__ uint4 staged[];
    const int first_column = blockIdx.y * kColumns;
    const int count        = min(kColumns, columns - first_column);
    const int chunks       = k / 8;
    for (int i = threadIdx.x; i < count * chunks; i += blockDim.x) {
        const int column = i / chunks, chunk = i % chunks;
        staged[column * chunks + chunk] =
            reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column + column) * k)[chunk];
    }
    __syncthreads();
    const int rows = segments.begin[segments.count];
    const int lane = threadIdx.x % 32;
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = blockIdx.x * (Warps * RowsPerWarp) + r * Warps + threadIdx.x / 32;
        if (row >= rows) { return; }
        int segment = 0;
        while (row >= segments.begin[segment + 1]) { ++segment; }
        const auto* w = reinterpret_cast<const uint4*>(segments.data[segment] +
                                                       static_cast<std::size_t>(row - segments.begin[segment]) * k);
        uint4 loaded[kMaxLaneChunks];
#pragma unroll
        for (int i = 0; i < kMaxLaneChunks; ++i) {
            const int chunk = lane + 32 * i;
            if (chunk < chunks) { loaded[i] = w[chunk]; }
        }
        float sums[kColumns] = {};
#pragma unroll
        for (int i = 0; i < kMaxLaneChunks; ++i) {
            const int chunk = lane + 32 * i;
            if (chunk >= chunks) { break; }
            const uint4 packed = loaded[i];
            const bf16* wv     = reinterpret_cast<const bf16*>(&packed);
            float wf[8];
            for (int i = 0; i < 8; ++i) { wf[i] = __bfloat162float(wv[i]); }
            for (int column = 0; column < kColumns; ++column) {
                if (column < count) {
                    const uint4 xp = staged[column * chunks + chunk];
                    const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                    for (int i = 0; i < 8; ++i) { sums[column] = fmaf(wf[i], __bfloat162float(xv[i]), sums[column]); }
                }
            }
        }
        for (int column = 0; column < kColumns; ++column) {
            float v = sums[column];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && column < count) {
                out[static_cast<std::size_t>(first_column + column) * rows + row] = v;
            }
        }
    }
}

// The q8_g32_fp16 form: row r's codes start at r * padded_k, its FP16 scales at r * padded_k / 32.
// Lane l reads the same 8-element chunks as the BF16 form, decodes each weight exactly and
// accumulates in the same order.
__global__ void projection_q8_kernel(const bf16* __restrict__ x, int k, int padded_k, int columns, int rows,
                                     const std::int8_t* __restrict__ codes, const __half* __restrict__ scales,
                                     float* __restrict__ out) {
    extern __shared__ uint4 staged[];
    const int first_column = blockIdx.y * kColumns;
    const int count        = min(kColumns, columns - first_column);
    const int chunks       = k / 8;
    for (int i = threadIdx.x; i < count * chunks; i += blockDim.x) {
        const int column = i / chunks, chunk = i % chunks;
        staged[column * chunks + chunk] =
            reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column + column) * k)[chunk];
    }
    __syncthreads();
    const int lane = threadIdx.x % 32;
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = blockIdx.x * kRowsPerCta + r * kWarps + threadIdx.x / 32;
        if (row >= rows) { return; }
        const auto* w      = reinterpret_cast<const uint2*>(codes + static_cast<std::size_t>(row) * padded_k);
        const __half* srow = scales + static_cast<std::size_t>(row) * (padded_k / 32);
        float sums[kColumns] = {};
        for (int chunk = lane; chunk < chunks; chunk += 32) {
            const uint2 packed = w[chunk];
            const float scale  = __half2float(srow[chunk / 4]);
            const auto* wv     = reinterpret_cast<const std::int8_t*>(&packed);
            float wf[8];
            for (int i = 0; i < 8; ++i) { wf[i] = static_cast<float>(wv[i]) * scale; }
            for (int column = 0; column < kColumns; ++column) {
                if (column < count) {
                    const uint4 xp = staged[column * chunks + chunk];
                    const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                    for (int i = 0; i < 8; ++i) { sums[column] = fmaf(wf[i], __bfloat162float(xv[i]), sums[column]); }
                }
            }
        }
        for (int column = 0; column < kColumns; ++column) {
            float v = sums[column];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && column < count) {
                out[static_cast<std::size_t>(first_column + column) * rows + row] = v;
            }
        }
    }
}

// The q4_g64_fp16 form: row r's two's-complement nibbles start at r * padded_k / 2 (low nibble
// first), its FP16 scales at r * padded_k / 64. Same chunks and order as the other forms.
__global__ void projection_q4_kernel(const bf16* __restrict__ x, int k, int padded_k, int columns, int rows,
                                     const std::uint8_t* __restrict__ codes, const __half* __restrict__ scales,
                                     float* __restrict__ out) {
    extern __shared__ uint4 staged[];
    const int first_column = blockIdx.y * kColumns;
    const int count        = min(kColumns, columns - first_column);
    const int chunks       = k / 8;
    for (int i = threadIdx.x; i < count * chunks; i += blockDim.x) {
        const int column = i / chunks, chunk = i % chunks;
        staged[column * chunks + chunk] =
            reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column + column) * k)[chunk];
    }
    __syncthreads();
    const int lane = threadIdx.x % 32;
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = blockIdx.x * kRowsPerCta + r * kWarps + threadIdx.x / 32;
        if (row >= rows) { return; }
        const auto* w      = reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::size_t>(row) * (padded_k / 2));
        const __half* srow = scales + static_cast<std::size_t>(row) * (padded_k / 64);
        float sums[kColumns] = {};
        for (int chunk = lane; chunk < chunks; chunk += 32) {
            const std::uint32_t packed = w[chunk];
            const float scale          = __half2float(srow[chunk / 8]);
            float wf[8];
            for (int i = 0; i < 8; ++i) {
                const int u = static_cast<int>((packed >> (4 * i)) & 0xFU);
                wf[i]       = static_cast<float>(u >= 8 ? u - 16 : u) * scale;
            }
            for (int column = 0; column < kColumns; ++column) {
                if (column < count) {
                    const uint4 xp = staged[column * chunks + chunk];
                    const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                    for (int i = 0; i < 8; ++i) { sums[column] = fmaf(wf[i], __bfloat162float(xv[i]), sums[column]); }
                }
            }
        }
        for (int column = 0; column < kColumns; ++column) {
            float v = sums[column];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && column < count) {
                out[static_cast<std::size_t>(first_column + column) * rows + row] = v;
            }
        }
    }
}

} // namespace

void projection_fp32(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 && x.ne[3] == 1,
            "x must be contiguous BF16 [K, T]");
    require(out.data != nullptr && out.dtype == DType::FP32 && out.is_contiguous(), "out must be contiguous FP32");
    const int k = x.ne[0], columns = x.ne[1], rows = weight.n, padded_k = weight.padded_shape[1];
    require(k > 0 && k % 8 == 0 && k <= kMaxK && columns > 0, "K must be a multiple of 8 up to 3072");
    const bool q4 = weight.qtype == QType::Q4_G64_FP16;
    require((weight.qtype == QType::Q8_G32_FP16 || q4) && weight.layout == QuantLayout::RowSplit &&
                weight.qhigh == nullptr && weight.k == k && rows > 0 && padded_k >= k &&
                padded_k % (q4 ? 64 : 32) == 0 && weight.qdata != nullptr && weight.scales != nullptr &&
                aligned16(weight.qdata) && aligned16(x.data),
            "weight must be an aligned row-split q8_g32_fp16 or q4_g64_fp16 [N, K]");
    require(out.ne[0] == rows && out.ne[1] == columns && out.ne[2] == 1 && out.ne[3] == 1,
            "out must be FP32 [N, T]");
    const dim3 grid((rows + kRowsPerCta - 1) / kRowsPerCta, (columns + kColumns - 1) / kColumns);
    const std::size_t staged = static_cast<std::size_t>(std::min(kColumns, columns)) * k * sizeof(bf16);
    if (q4) {
        projection_q4_kernel<<<grid, kThreads, staged, stream>>>(
            static_cast<const bf16*>(x.data), k, padded_k, columns, rows,
            static_cast<const std::uint8_t*>(weight.qdata), static_cast<const __half*>(weight.scales),
            static_cast<float*>(out.data));
    } else {
        projection_q8_kernel<<<grid, kThreads, staged, stream>>>(
            static_cast<const bf16*>(x.data), k, padded_k, columns, rows,
            static_cast<const std::int8_t*>(weight.qdata), static_cast<const __half*>(weight.scales),
            static_cast<float*>(out.data));
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("projection_fp32 quantized: ") + cudaGetErrorString(error));
    }
}

void projection_fp32(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                     cudaStream_t stream) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 && x.ne[3] == 1,
            "x must be contiguous BF16 [K, T]");
    require(out.data != nullptr && out.dtype == DType::FP32 && out.is_contiguous(), "out must be contiguous FP32");
    const int k = x.ne[0], columns = x.ne[1];
    require(k > 0 && k % 8 == 0 && k <= kMaxK && columns > 0, "K must be a multiple of 8 up to 3072");
    require(!weights.empty() && weights.size() <= kMaxSegments, "between one and four weights");
    require(aligned16(x.data), "x must be 16-byte aligned");
    Segments segments{};
    segments.count = static_cast<int>(weights.size());
    int rows       = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const Tensor& w = *weights[i];
        require(w.data != nullptr && w.dtype == DType::BF16 && w.is_contiguous() && w.ne[0] == k && w.ne[1] > 0 &&
                    w.ne[2] == 1 && w.ne[3] == 1 && aligned16(w.data),
                "each weight must be contiguous, 16-byte aligned BF16 [K, N]");
        segments.data[i]  = static_cast<const bf16*>(w.data);
        segments.begin[i] = rows;
        rows += w.ne[1];
    }
    segments.begin[segments.count] = rows;
    require(out.ne[0] == rows && out.ne[1] == columns && out.ne[2] == 1 && out.ne[3] == 1,
            "out must be FP32 [sum N, T]");
    const std::size_t staged = static_cast<std::size_t>(std::min(kColumns, columns)) * k * sizeof(bf16);
    const dim3 column_groups(1, (columns + kColumns - 1) / kColumns);
    if (columns <= kColumns) {
        constexpr int threads = 128, rows_per_warp = 1, rows_per_cta = threads / 32 * rows_per_warp;
        projection_kernel<threads, rows_per_warp><<<dim3((rows + rows_per_cta - 1) / rows_per_cta, column_groups.y),
                                                    threads, staged, stream>>>(
            static_cast<const bf16*>(x.data), k, columns, segments, static_cast<float*>(out.data));
    } else {
        projection_kernel<kThreads, kRowsPerWarp><<<dim3((rows + kRowsPerCta - 1) / kRowsPerCta, column_groups.y),
                                                    kThreads, staged, stream>>>(
            static_cast<const bf16*>(x.data), k, columns, segments, static_cast<float*>(out.data));
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("projection_fp32: ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops
