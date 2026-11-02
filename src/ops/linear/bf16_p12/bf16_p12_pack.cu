// infernix::ops - bf16_p12 packing (include/infernix/ops/bf16_p12.h): load-time work, two passes on the device.
#include "infernix/ops/bf16_p12.h"

#include "core/device.h" // CUDA_CHECK
#include "core/weight_view.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops {
namespace {

constexpr int kRowWarps   = 4;
constexpr int kScanThreads = 1024;

// Scratch: per row its escape count << 8 | its base, then the rows' escape starts (n + 1; the last the total),
// then a fill counter per row.
struct Scratch {
    unsigned* info;
    unsigned* start;
    unsigned* fill;
};

Scratch scratch_planes(void* scratch, std::int32_t n) {
    auto* p = static_cast<unsigned*>(scratch);
    return {p, p + n, p + 2 * static_cast<std::int64_t>(n) + 1};
}

__device__ __forceinline__ bool p12_escaped(unsigned bits, int base) {
    const int e = static_cast<int>((bits >> 7) & 0xFFU);
    return e == 0 || e < base || e > base + 14;
}

// One warp a row: the base (the 15-exponent window holding most nonzero weights, ties to the lower base)
// and the escape count.
__global__ void __launch_bounds__(kRowWarps * 32)
    p12_row_info(const std::uint16_t* __restrict__ w, std::int32_t n, std::int32_t k, unsigned* __restrict__ info) {
    __shared__ unsigned hist[kRowWarps][256];
    const int lane = static_cast<int>(threadIdx.x % 32), warp = static_cast<int>(threadIdx.x / 32);
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x) * kRowWarps + warp;
    for (int i = lane; i < 256; i += 32) { hist[warp][i] = 0; }
    __syncwarp();
    const std::uint16_t* r = w + static_cast<std::int64_t>(min(row, n - 1)) * k;
    if (row < n) {
        for (std::int32_t i = lane; i < k; i += 32) { atomicAdd(&hist[warp][(r[i] >> 7) & 0xFFU], 1U); }
    }
    __syncwarp();
    unsigned best = 0, best_base = 1;
    for (int b = 1 + lane; b + 14 < 255; b += 32) {
        unsigned s = 0;
        for (int e = b; e < b + 15; ++e) { s += hist[warp][e]; }
        if (s > best) {
            best      = s;
            best_base = static_cast<unsigned>(b);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned other      = __shfl_xor_sync(0xFFFFFFFFU, best, offset);
        const unsigned other_base = __shfl_xor_sync(0xFFFFFFFFU, best_base, offset);
        if (other > best || (other == best && other_base < best_base)) {
            best      = other;
            best_base = other_base;
        }
    }
    unsigned escapes = 0;
    if (row < n) {
        for (std::int32_t i = lane; i < k; i += 32) { escapes += p12_escaped(r[i], static_cast<int>(best_base)); }
    }
    for (int offset = 16; offset > 0; offset >>= 1) { escapes += __shfl_xor_sync(0xFFFFFFFFU, escapes, offset); }
    if (lane == 0 && row < n) { info[row] = escapes << 8 | best_base; }
}

// One block: the rows' escape starts (chunked serial prefix; load-time only).
__global__ void __launch_bounds__(kScanThreads)
    p12_escape_starts(const unsigned* __restrict__ info, std::int32_t n, unsigned* __restrict__ start) {
    __shared__ unsigned partial[kScanThreads];
    const std::int32_t per = (n + kScanThreads - 1) / kScanThreads;
    const std::int32_t lo = static_cast<std::int32_t>(threadIdx.x) * per, hi = min(n, lo + per);
    unsigned sum = 0;
    for (std::int32_t i = lo; i < hi; ++i) { sum += info[i] >> 8; }
    partial[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned run = 0;
        for (int i = 0; i < kScanThreads; ++i) {
            const unsigned v = partial[i];
            partial[i]       = run;
            run += v;
        }
        start[n] = run;
    }
    __syncthreads();
    unsigned run = partial[threadIdx.x];
    for (std::int32_t i = lo; i < hi; ++i) {
        start[i] = run;
        run += info[i] >> 8;
    }
}

__global__ void p12_row_words(const unsigned* __restrict__ info, const unsigned* __restrict__ start, std::int32_t n,
                              unsigned* __restrict__ words, unsigned* __restrict__ fill) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < n) {
        words[row] = start[row] << 8 | (info[row] & 0xFFU);
        fill[row]  = 0;
    }
    if (row == n) { words[n] = start[n] << 8; }
}

// One thread a chunk of 8 weights: its code bytes and nibbles, and its escapes into the row's list (unordered).
__global__ void p12_pack_chunks(const uint4* __restrict__ w, std::int64_t chunks, std::int32_t chunks_per_row,
                                const unsigned* __restrict__ info, const unsigned* __restrict__ start,
                                unsigned* __restrict__ fill, uint2* __restrict__ codes, unsigned* __restrict__ nibbles,
                                unsigned* __restrict__ escapes) {
    const std::int64_t c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= chunks) { return; }
    const std::int64_t row  = c / chunks_per_row;
    const int base          = static_cast<int>(info[row] & 0xFFU);
    const uint4 v           = w[c];
    const unsigned words[4] = {v.x, v.y, v.z, v.w};
    unsigned code[2] = {0, 0}, nib = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned bits = (words[i / 2] >> (16 * (i % 2))) & 0xFFFFU;
        const bool escaped  = p12_escaped(bits, base);
        const unsigned c4   = escaped ? 15U : (((bits >> 7) & 0xFFU) - static_cast<unsigned>(base));
        code[i / 4] |= (((bits >> 8) & 0x80U) | (bits & 0x7FU)) << (8 * (i % 4));
        nib |= c4 << (4 * i);
        if (escaped) {
            const unsigned column = static_cast<unsigned>((c - row * chunks_per_row) * 8 + i);
            escapes[start[row] + atomicAdd(&fill[row], 1U)] = column << 16 | bits;
        }
    }
    codes[c]   = make_uint2(code[0], code[1]);
    nibbles[c] = nib;
}

// One thread a row: its escapes in column order (insertion sort; a row holds few).
__global__ void p12_sort_rows(const unsigned* __restrict__ start, std::int32_t n, unsigned* __restrict__ escapes) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n) { return; }
    const unsigned lo = start[row], hi = start[row + 1];
    for (unsigned i = lo + 1; i < hi; ++i) {
        const unsigned v = escapes[i];
        unsigned j       = i;
        while (j > lo && escapes[j - 1] > v) {
            escapes[j] = escapes[j - 1];
            --j;
        }
        escapes[j] = v;
    }
}

void check_matrix(const void* bf16, std::int32_t n, std::int32_t k, const void* scratch) {
    if (bf16 == nullptr || scratch == nullptr || n <= 0 || k <= 0 || k % 8 != 0 ||
        reinterpret_cast<std::uintptr_t>(bf16) % 16 != 0) {
        throw std::invalid_argument("bf16_p12: a contiguous, 16-byte aligned BF16 [n, k] with k a multiple of 8");
    }
    if (k >= (1 << 16)) { throw std::invalid_argument("bf16_p12: k must be below 65536 (16-bit escape columns)"); }
}

} // namespace

std::size_t bf16_p12_pack_scratch_bytes(std::int32_t n) {
    if (n <= 0) { throw std::invalid_argument("bf16_p12: n must be positive"); }
    return (3 * static_cast<std::size_t>(n) + 1) * sizeof(unsigned);
}

void bf16_p12_count(const void* bf16, std::int32_t n, std::int32_t k, void* scratch, std::uint32_t* escapes,
                    cudaStream_t stream) {
    check_matrix(bf16, n, k, scratch);
    if (escapes == nullptr) { throw std::invalid_argument("bf16_p12: the escape count needs a device word"); }
    const Scratch s = scratch_planes(scratch, n);
    p12_row_info<<<(n + kRowWarps - 1) / kRowWarps, kRowWarps * 32, 0, stream>>>(
        static_cast<const std::uint16_t*>(bf16), n, k, s.info);
    p12_escape_starts<<<1, kScanThreads, 0, stream>>>(s.info, n, s.start);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(escapes, s.start + n, sizeof(std::uint32_t), cudaMemcpyDeviceToDevice, stream));
}

void bf16_p12_pack(const void* bf16, std::int32_t n, std::int32_t k, std::uint64_t escapes, void* scratch, void* dst,
                   cudaStream_t stream) {
    check_matrix(bf16, n, k, scratch);
    if (dst == nullptr || reinterpret_cast<std::uintptr_t>(dst) % 256 != 0) {
        throw std::invalid_argument("bf16_p12: the destination must be 256-byte aligned");
    }
    const WeightGeometry g = bf16_p12_geometry(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k), escapes);
    auto* out              = static_cast<std::byte*>(dst);
    const Scratch s        = scratch_planes(scratch, n);
    auto* words            = reinterpret_cast<unsigned*>(out + g.scale_offset);
    p12_row_words<<<(n + 1 + 255) / 256, 256, 0, stream>>>(s.info, s.start, n, words, s.fill);
    const std::int64_t chunks = static_cast<std::int64_t>(n) * (k / 8);
    p12_pack_chunks<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint4*>(bf16), chunks, k / 8, s.info, s.start, s.fill, reinterpret_cast<uint2*>(out),
        reinterpret_cast<unsigned*>(out + g.high_offset), reinterpret_cast<unsigned*>(out + g.escape_offset));
    p12_sort_rows<<<(n + 255) / 256, 256, 0, stream>>>(s.start, n, reinterpret_cast<unsigned*>(out + g.escape_offset));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops
