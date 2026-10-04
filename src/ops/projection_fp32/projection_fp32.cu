#include "ninfer/ops/projection_fp32.h"

#include <cuda_bf16.h>

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

struct Segments {
    const bf16* data[kMaxSegments];
    int begin[kMaxSegments + 1]; // first output row of each segment, then the total
    int count;
};

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("projection_fp32: ") + message); }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// Warp w of a CTA owns rows first + w, first + w + 8, ...; lane l accumulates the 8-element
// chunks l, l + 32, ... in order and the warp reduces by a fixed butterfly, so an output's bits
// depend only on K.
__global__ void projection_kernel(const bf16* __restrict__ x, int k, int columns, Segments segments,
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
    const int rows = segments.begin[segments.count];
    const int lane = threadIdx.x % 32;
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = blockIdx.x * kRowsPerCta + r * kWarps + threadIdx.x / 32;
        if (row >= rows) { return; }
        int segment = 0;
        while (row >= segments.begin[segment + 1]) { ++segment; }
        const auto* w = reinterpret_cast<const uint4*>(segments.data[segment] +
                                                       static_cast<std::size_t>(row - segments.begin[segment]) * k);
        float sums[kColumns] = {};
        for (int chunk = lane; chunk < chunks; chunk += 32) {
            const uint4 packed = w[chunk];
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

} // namespace

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
    const dim3 grid((rows + kRowsPerCta - 1) / kRowsPerCta, (columns + kColumns - 1) / kColumns);
    const std::size_t staged = static_cast<std::size_t>(kColumns) * k * sizeof(bf16);
    projection_kernel<<<grid, kThreads, staged, stream>>>(static_cast<const bf16*>(x.data), k, columns, segments,
                                                          static_cast<float*>(out.data));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("projection_fp32: ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops
