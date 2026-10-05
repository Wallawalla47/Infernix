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

// The total row count. Segment fields are read at compile-time indices only: a runtime index into
// the kernel parameter would copy it to local memory and put a local load before the weight loads.
__device__ __forceinline__ int total_rows(const Segments& segments) {
    int rows = segments.begin[1];
#pragma unroll
    for (int s = 2; s <= kMaxSegments; ++s) {
        if (s == segments.count) { rows = segments.begin[s]; }
    }
    return rows;
}

// Issues lane `lane`'s chunk loads of `row` (chunks lane, lane + 32, ...) into `loaded`.
__device__ __forceinline__ void load_row(const Segments& segments, int row, int k, int chunks, int lane,
                                         uint4 (&loaded)[kMaxLaneChunks]) {
    const bf16* data = segments.data[0];
    int begin        = 0;
#pragma unroll
    for (int s = 1; s < kMaxSegments; ++s) {
        if (s < segments.count && row >= segments.begin[s]) {
            data  = segments.data[s];
            begin = segments.begin[s];
        }
    }
    const auto* w = reinterpret_cast<const uint4*>(data + static_cast<std::size_t>(row - begin) * k);
#pragma unroll
    for (int i = 0; i < kMaxLaneChunks; ++i) {
        const int chunk = lane + 32 * i;
        if (chunk < chunks) { loaded[i] = w[chunk]; }
    }
}

// Warp w of a CTA owns rows first + w, first + w + Warps, ...; lane l accumulates the 8-element
// chunks l, l + 32, ... in order and the warp reduces by a fixed butterfly, so an output's bits
// depend only on K. A lane issues all of a row's chunk loads before its FMAs, and the first row's
// loads before x is staged, so the weights' DRAM trip overlaps the staging instead of following
// it. Decode-width calls give each warp one row (Threads = 128, RowsPerWarp = 1: 129 CTAs for a
// 513-row router instead of 17), so a router's latency is one DRAM round trip instead of ~40; wide
// calls keep four rows per warp, which stages x in fewer CTAs. Both mappings produce the same bits.
template <int Threads, int RowsPerWarp>
__global__ void __launch_bounds__(Threads)
    projection_kernel(const bf16* __restrict__ x, int k, int columns, Segments segments, float* __restrict__ out) {
    constexpr int Warps = Threads / 32;
    // 16-byte x chunks of the CTA's columns, as one unrolled batch of at most this many per thread.
    constexpr int kStagePerThread = (kColumns * kMaxK / 8 + Threads - 1) / Threads;
    extern __shared__ uint4 staged[];
    const int first_column = blockIdx.y * kColumns;
    const int count        = min(kColumns, columns - first_column);
    const int chunks       = k / 8;
    const int rows         = total_rows(segments);
    const int lane         = threadIdx.x % 32;
    const int first_row    = blockIdx.x * (Warps * RowsPerWarp) + threadIdx.x / 32;
    uint4 loaded[kMaxLaneChunks];
    if (first_row < rows) { load_row(segments, first_row, k, chunks, lane, loaded); }
    // The CTA's columns of x are contiguous, so chunk i of the staging is chunk i of x from there.
    const auto* xs = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column) * k);
#pragma unroll
    for (int u = 0; u < kStagePerThread; ++u) {
        const int i = static_cast<int>(threadIdx.x) + u * Threads;
        if (i < count * chunks) {
            const auto dst = static_cast<unsigned>(__cvta_generic_to_shared(staged + i));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(xs + i) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = first_row + r * Warps;
        if (row >= rows) { return; }
        if (r > 0) { load_row(segments, row, k, chunks, lane, loaded); }
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

// Wide calls (prefill chunks). The narrow mapping re-reads a CTA's weight rows for every eight
// columns and reads nine 16-byte words per 64 FMAs: a 4K-token router call takes ~0.75 ms at
// ~14 TFLOP/s. Here a CTA owns a kWideRows x kWideColumns output tile and walks K in slices of 256
// elements, which is exactly one 8-element chunk per lane: in slice s lane l handles chunk
// 32 s + l, the narrow kernels' l-th lane partial in the same order. Each slice's weight rows and
// x columns are staged with cp.async (double-buffered); a warp owns a 4-row x 8-column sub-tile, so
// a lane loads 12 16-byte words per 256 FMAs. After the last slice each warp reduces its partials
// by the same butterfly, so the bits equal the narrow kernels' (an output's summation still
// depends only on K).
// 512 threads: a lane's 32 accumulators and 32 widened weights need more than the 64 registers a
// 1,024-thread CTA allows (that version spilled and ran at 1.6 ms).
constexpr int kWideRows        = 32; // output rows per CTA
constexpr int kWideColumns     = 16; // output columns per CTA
constexpr int kWideTileRows    = 4;  // per warp
constexpr int kWideTileColumns = 8;
constexpr int kWideRowWarps    = kWideRows / kWideTileRows;       // 8
constexpr int kWideColumnWarps = kWideColumns / kWideTileColumns; // 2
constexpr int kWideThreads     = kWideRowWarps * kWideColumnWarps * 32;
constexpr int kSliceChunks     = 32; // chunks per K slice: one per lane
constexpr int kStageWords      = (kWideRows + kWideColumns) * kSliceChunks; // 16-byte words per stage
constexpr int kWideSharedBytes = 2 * kStageWords * 16;
constexpr int kWideMinColumns  = 64; // narrower calls keep the narrow mapping (same bits)
static_assert(kWideThreads == 512 && kStageWords % kWideThreads == 0);

// The address of chunk `chunk` (8 elements) of output row `row`.
__device__ __forceinline__ const uint4* row_chunk(const Segments& segments, int row, int k, int chunk) {
    const bf16* data = segments.data[0];
    int begin        = 0;
#pragma unroll
    for (int s = 1; s < kMaxSegments; ++s) {
        if (s < segments.count && row >= segments.begin[s]) {
            data  = segments.data[s];
            begin = segments.begin[s];
        }
    }
    return reinterpret_cast<const uint4*>(data + static_cast<std::size_t>(row - begin) * k) + chunk;
}

__device__ __forceinline__ void widen8(const uint4& packed, float (&out)[8]) {
    const auto* v = reinterpret_cast<const bf16*>(&packed);
#pragma unroll
    for (int i = 0; i < 8; ++i) { out[i] = __bfloat162float(v[i]); }
}

// 16 bytes from `src` to shared `dst`, or zeros when `valid` is false (src is then not read).
__device__ __forceinline__ void stage16(uint4* dst, const void* src, bool valid) {
    const auto d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

__global__ void __launch_bounds__(kWideThreads, 1)
    projection_wide_kernel(const bf16* __restrict__ x, int k, int columns, Segments segments, float* __restrict__ out) {
    extern __shared__ uint4 stage[]; // [2][rows then columns][kSliceChunks]
    const int chunks       = k / 8;
    const int slices       = (chunks + kSliceChunks - 1) / kSliceChunks;
    const int rows         = total_rows(segments);
    const int first_row    = static_cast<int>(blockIdx.x) * kWideRows;
    const int first_column = static_cast<int>(blockIdx.y) * kWideColumns;
    const auto issue = [&](int slice, int buffer) {
        uint4* base = stage + buffer * kStageWords;
#pragma unroll
        for (int u = 0; u < kStageWords / kWideThreads; ++u) {
            const int i     = static_cast<int>(threadIdx.x) + u * kWideThreads;
            const int line  = i / kSliceChunks;
            const int chunk = slice * kSliceChunks + i % kSliceChunks;
            if (line < kWideRows) {
                const int row    = first_row + line;
                const bool valid = row < rows && chunk < chunks;
                stage16(base + i, valid ? static_cast<const void*>(row_chunk(segments, row, k, chunk)) : x, valid);
            } else {
                const int column = first_column + line - kWideRows;
                const bool valid = column < columns && chunk < chunks;
                stage16(base + i, valid ? static_cast<const void*>(x + static_cast<std::size_t>(column) * k + 8 * chunk) : x,
                        valid);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int r0 = (warp % kWideRowWarps) * kWideTileRows;
    const int c0 = (warp / kWideRowWarps) * kWideTileColumns;
    float sums[kWideTileRows][kWideTileColumns] = {};
    issue(0, 0);
    for (int s = 0; s < slices; ++s) {
        if (s + 1 < slices) {
            issue(s + 1, (s + 1) & 1);
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();
        if (s * kSliceChunks + lane < chunks) { // the narrow kernels' loop ends at the last chunk too
            const uint4* base = stage + (s & 1) * kStageWords;
            float wf[kWideTileRows][8];
#pragma unroll
            for (int r = 0; r < kWideTileRows; ++r) { widen8(base[(r0 + r) * kSliceChunks + lane], wf[r]); }
#pragma unroll
            for (int c = 0; c < kWideTileColumns; ++c) {
                float xf[8];
                widen8(base[(kWideRows + c0 + c) * kSliceChunks + lane], xf);
#pragma unroll
                for (int r = 0; r < kWideTileRows; ++r) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) { sums[r][c] = fmaf(wf[r][i], xf[i], sums[r][c]); }
                }
            }
        }
        __syncthreads(); // the next iteration's issue overwrites this buffer
    }
#pragma unroll
    for (int r = 0; r < kWideTileRows; ++r) {
        const int row = first_row + r0 + r;
#pragma unroll
        for (int c = 0; c < kWideTileColumns; ++c) {
            const int column = first_column + c0 + c;
            float v          = sums[r][c];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && row < rows && column < columns) { out[static_cast<std::size_t>(column) * rows + row] = v; }
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
    if (columns >= kWideMinColumns) {
        static const bool configured = [] {
            return cudaFuncSetAttribute(projection_wide_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kWideSharedBytes) == cudaSuccess;
        }();
        require(configured, "the wide kernel's shared memory could not be configured");
        projection_wide_kernel<<<dim3((rows + kWideRows - 1) / kWideRows, (columns + kWideColumns - 1) / kWideColumns),
                                 kWideThreads, kWideSharedBytes, stream>>>(
            static_cast<const bf16*>(x.data), k, columns, segments, static_cast<float*>(out.data));
    } else if (columns <= kColumns) {
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
