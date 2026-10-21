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
// columns share the call. Calls of two or more columns run the same arithmetic with Strata v4's read
// scheduling (hc_down_partials_v4 / hc_up_collapse_v4 below).

#include "ops/hyper_connection/hyper_connection_mix_fused.h"

#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
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

void check_launch(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection_mix ") + what + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

// The norm weight w is BF16, or FP32 (NVIDIA FP4-Local's trained norms); every kernel forms 1 + w in
// FP32 from the represented value.
__device__ __forceinline__ float norm_value(bf16 w) { return __bfloat162float(w); }
__device__ __forceinline__ float norm_value(float w) { return w; }

__device__ __forceinline__ void bf16x8_bits_to_floats(const uint4& bits, float (&v)[8]) {
    const float2 a = bf16x2_bits_to_float2(bits.x), b = bf16x2_bits_to_float2(bits.y);
    const float2 c = bf16x2_bits_to_float2(bits.z), d = bf16x2_bits_to_float2(bits.w);
    v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y, v[4] = c.x, v[5] = c.y, v[6] = d.x, v[7] = d.y;
}

// Eight consecutive norm weights as the v4 kernels stage them: one 16-byte vector of BF16 words, or
// two of FP32 words.
template <class Norm>
struct NormGroup;

template <>
struct NormGroup<bf16> {
    uint4 bits;
    __device__ static NormGroup load(const bf16* p) { return {load_ldg<uint4>(p)}; }
    __device__ static NormGroup zero() { return {make_uint4(0U, 0U, 0U, 0U)}; }
    __device__ void floats(float (&w)[8]) const { bf16x8_bits_to_floats(bits, w); }
};

template <>
struct NormGroup<float> {
    float4 low, high;
    __device__ static NormGroup load(const float* p) { return {load_ldg<float4>(p), load_ldg<float4>(p + 4)}; }
    __device__ static NormGroup zero() { return {make_float4(0.F, 0.F, 0.F, 0.F), make_float4(0.F, 0.F, 0.F, 0.F)}; }
    __device__ void floats(float (&w)[8]) const {
        w[0] = low.x, w[1] = low.y, w[2] = low.z, w[3] = low.w, w[4] = high.x, w[5] = high.y, w[6] = high.z,
        w[7] = high.w;
    }
};

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    return value;
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

template <class Codec, class Norm>
__global__ __launch_bounds__(kDownThreads, 1) void hc_down_partials_kernel(
    const bf16* __restrict__ residual, const Norm* __restrict__ norm_weight, typename Codec::Params down, int rows,
    int columns, float eps, float* __restrict__ partial, float* __restrict__ inv_out) {
    extern __shared__ __align__(16) float rn[]; // [kPassColumns][kHidden]
    __shared__ float scratch[kDownThreads / 32][kPassColumns];
    __shared__ float inv[kPassColumns];
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int s    = static_cast<int>(blockIdx.y);
    const int row  = static_cast<int>(blockIdx.x) * kDownRows + warp;
    const bool live = row < rows;
    constexpr int width = kStreams * kHidden;
    constexpr int kOwned = kHidden / kDownThreads; // features d = threadIdx.x + kDownThreads * j a thread scales
    static_assert(kHidden % kDownThreads == 0);

    // 1. Weights first: this warp's row of the stream's K slice and the thread's norm weights (1 + w),
    // before any activation.
    const std::int64_t source = live ? row : 0;
    typename Codec::Chunk chunk[kChunks];
#pragma unroll
    for (int m = 0; m < kChunks; ++m) { chunk[m] = Codec::load(down, source, s * kHidden + m * 256 + lane * 8); }
    const Norm* w = norm_weight + static_cast<std::int64_t>(s) * kHidden;
    float w_bias[kOwned];
#pragma unroll
    for (int j = 0; j < kOwned; ++j) { w_bias[j] = 1.0F + norm_value(w[threadIdx.x + kDownThreads * j]); }
    // Launched as a programmatic dependent: the weights load while the producer finishes. The up
    // collapse may launch beside this grid at once and fetch its own weights meanwhile; it reads the
    // partials only after this grid completes.
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    // Lanes 4-7 of each quarter warp read a chunk's two float4 halves of Rn in the other order, so a
    // quarter warp's eight 16-byte reads cover all 32 banks.
    const int first_half = (lane >> 2) & 1;

#pragma unroll 1
    for (int c0 = 0; c0 < columns; c0 += kPassColumns) {
        const int pass = min(kPassColumns, columns - c0);
        // 2. Stage the slice in FP32 with each column's per-thread sum of squares (its elements in
        // index order), then every column's warp butterfly and its warps' sums in warp order.
        // Every column's loads are issued before any is used (one CTA per SM: registers are free).
        constexpr int kSlices = (kHidden + kDownThreads * 8 - 1) / (kDownThreads * 8);
        uint4 values[kPassColumns][kSlices];
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            const bf16* r = residual + static_cast<std::int64_t>(c0 + c) * width + static_cast<std::int64_t>(s) * kHidden;
#pragma unroll
            for (int h = 0; h < kSlices; ++h) {
                const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                if (c < pass && i < kHidden) { values[c][h] = load_ldg<uint4>(r + i); }
            }
        }
        float sums[kPassColumns];
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            sums[c] = 0.0F;
#pragma unroll
            for (int h = 0; h < kSlices; ++h) {
                const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                if (c < pass && i < kHidden) {
                    const uint4 u   = values[c][h];
                    const float2 v0 = bf16x2_bits_to_float2(u.x), v1 = bf16x2_bits_to_float2(u.y);
                    const float2 v2 = bf16x2_bits_to_float2(u.z), v3 = bf16x2_bits_to_float2(u.w);
                    const float v[8] = {v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};
                    float4* out = reinterpret_cast<float4*>(rn + c * kHidden + i);
                    out[0] = make_float4(v[0], v[1], v[2], v[3]);
                    out[1] = make_float4(v[4], v[5], v[6], v[7]);
#pragma unroll
                    for (int j = 0; j < 8; ++j) { sums[c] += v[j] * v[j]; }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            if (c < pass) {
                const float total = warp_sum(sums[c]);
                if (lane == 0) { scratch[warp][c] = total; }
            }
        }
        __syncthreads();
        if (static_cast<int>(threadIdx.x) < pass) {
            float total = 0.0F;
#pragma unroll
            for (int wp = 0; wp < kDownThreads / 32; ++wp) { total += scratch[wp][threadIdx.x]; }
            inv[threadIdx.x] = rsqrtf(total / static_cast<float>(kHidden) + eps);
        }
        __syncthreads();
        // Rn = R * inv * (1 + w), FP32.
        for (int c = 0; c < pass; ++c) {
#pragma unroll
            for (int j = 0; j < kOwned; ++j) {
                const int i = c * kHidden + static_cast<int>(threadIdx.x) + kDownThreads * j;
                rn[i]       = rn[i] * inv[c] * w_bias[j];
            }
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
                    const float4* half = reinterpret_cast<const float4*>(rn + c * kHidden + kk);
                    const float4 p = half[first_half], q = half[first_half ^ 1];
                    const float4 a = first_half ? q : p, b = first_half ? p : q;
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

template <class Codec, class Norm>
__global__ __launch_bounds__(kUpThreads, 1) void hc_up_collapse_kernel(
    const float* __restrict__ partial, const float* __restrict__ inv_in, const bf16* __restrict__ residual,
    const Norm* __restrict__ norm_weight, typename Codec::Params up, int rank, int down_rows, int columns,
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
    // The norm weights (1 + w) of the warp's S rows, and the residual R[s*H + d, t] for every (t, s):
    // entry e = t*S + s is held by lane e % 32 in r_held[e / 32], for lane 0's collapse.
    static_assert(kStreams * kHcMixFusedMaxColumns <= 64);
    const int dl = live ? d : 0;
    float w_bias[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        w_bias[s] = 1.0F + norm_value(norm_weight[static_cast<std::int64_t>(s) * kHidden + dl]);
    }
    // Launched as a programmatic dependent: the weights load while the down partials finish. The next
    // projection may launch beside this grid at once and fetch its weights meanwhile.
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    float r_held[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int e = lane + 32 * h, t = e / kStreams, s = e - t * kStreams;
        r_held[h]   = t < columns ? __bfloat162float(residual[static_cast<std::int64_t>(t) * width +
                                                              static_cast<std::int64_t>(s) * kHidden + dl])
                                  : 0.0F;
    }
    // Lanes 4-7 of each quarter warp read a chunk's two float4 halves of m in the other order, so a
    // quarter warp's eight 16-byte reads cover all 32 banks.
    const int first_half = (lane >> 2) & 1;

    // 2. z = sum_s partial[s] in order s = 0..S-1; m = SiLU(z / S); CTA 0 writes the injects.
    const float inv_streams = 1.0F / static_cast<float>(kStreams);
#pragma unroll 4
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
                const float4* half = reinterpret_cast<const float4*>(mt + c * 8);
                const float4 p = half[first_half], q = half[first_half ^ 1];
                const float4 a = first_half ? q : p, b = first_half ? p : q;
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
        float r[kStreams];
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            const int e    = t * kStreams + s;
            const float lo = __shfl_sync(0xFFFFFFFFU, r_held[0], e & 31);
            const float hi = __shfl_sync(0xFFFFFFFFU, r_held[1], e & 31);
            r[s]           = e < 32 ? lo : hi;
        }
        if (lane == 0) {
            float sum = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const float rn = r[s] * inv[s * columns + t] * w_bias[s];
                sum += sigmoidf(acc[s]) * rn;
            }
            x[static_cast<std::int64_t>(t) * kHidden + d] = __float2bfloat16_rn(sum * inv_streams);
        }
    }
}

// Strata v4's hyper-connection read scheduling (strata-nvfp4 src/kernels/cuda/fused_gr.cu,
// gr_down_v4_kernel and gr_up_v4_kernel) for calls of two or more columns. Every product and sum
// below is the one K1a/K1b compute, in the same order, so the outputs are bit-identical. RTX 5090,
// Q8 mixer, 2026-10-16: per call -7.6 % (T 2) to -17.8 % (T 5), +2.2 % at T 1 (which keeps
// K1a/K1b); Flash-Next Dense8 MTP 4 decode +0.9 %, plain decode +0.6 %. Two rows per warp in K1a
// (v4's other change) spilled 120-160 bytes of stack and was slower (-0 to +20 % per call).
//
// K1a/K1b already had v4's weights-first loads (every weight chunk in registers before the PDL
// wait), the residual read through __ldg, its lane-chunk K mapping (k = 8 lane + 256 m) and, in
// K1b, tokens outer with each token's m chunk read once for all of a warp's rows. What these
// kernels add:
//   K1a v4: Rn is staged once, straight from the registers that summed its squares, in two planes
//     (floats 0-3 of every 8 in plane 0, floats 4-7 in plane 1). The 16-byte stores of a quarter
//     warp and the dot product's reads are then conflict-free, and the raw store plus in-place
//     rescale pass of K1a (a two-way conflicted store, then a read and a write) is gone. The CTA
//     keeps 8 warps of one row each, so the norm's per-thread sums and warp order are unchanged.
//   K1b v4: 8 warps of 8 rows (2 features x 4 streams, v4's row map) instead of 16 warps of 4; a
//     warp's 8 row sums of a token reduce by one transposed butterfly of 9 shuffles, so a CTA issues
//     8 x 9 = 72 shuffles per token instead of 16 x (4 warp_sums' 20 + 8 for lane 0's residual
//     gather) = 448, and each m read feeds 8 rows instead of 4; m is staged in two planes; partials,
//     inv and R are read through __ldg. The collapse runs after the token loop, one thread per
//     (token, feature), from the row sums in shared memory, with K1b's own expressions in its order.
// Not adopted: v4's deferred RMS scale (Strata multiplies unscaled partials by inv in up, another
// rounding) and its collapse (it rounds each sigmoid(u) Rn product before the stream sum, where
// K1b's chain may fuse each product into the sum).
//
// Why the transposed butterfly is exact. warp_sum adds own + partner at xor offsets 16, 8, 4, 2,
// 1: level j turns lane l's value into v_l + v_(l ^ o_j). IEEE-754 addition is commutative (signed
// zeros included; the GPU returns the canonical NaN for any NaN operand), so lanes l and l ^ o_j
// hold identical bits after every level, and every lane ends with the same tree: the pairs added
// at each level are fixed by the xor offsets alone. warp_sum8_transposed reduces eight values per
// lane at once. At offset 16 lane l keeps the four values of its half (values 4 b4 .. 4 b4 + 3,
// b4 = bit 4 of l) and adds the partner's same four: for each such value r it computes
// v_l[r] + v_(l^16)[r], exactly warp_sum's first level for value r in lane l. At offset 8 the
// partner l ^ 8 has the same bit 4, so it holds the same four values' level-1 sums; lane l keeps
// two (by bit 3) and adds the partner's: warp_sum's second level. Offset 4 does the same for one
// value (by bit 2), and offsets 2 and 1 are ordinary butterfly levels between lanes holding the same
// value. Lane l ends with value l >> 2, every level formed from warp_sum's own pairs: bit-identical
// to warp_sum of that value, in every lane of the group 4 (l >> 2) .. 4 (l >> 2) + 3.

// Eight per-lane values reduced across the warp: lane l returns the warp_sum of v[l >> 2].
__device__ __forceinline__ float warp_sum8_transposed(const float (&v)[8], int lane) {
    const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
    float h4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float keep = b4 ? v[4 + i] : v[i];
        const float send = b4 ? v[i] : v[4 + i];
        h4[i]            = keep + __shfl_xor_sync(0xFFFFFFFFU, send, 16);
    }
    float h2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float keep = b3 ? h4[2 + i] : h4[i];
        const float send = b3 ? h4[i] : h4[2 + i];
        h2[i]            = keep + __shfl_xor_sync(0xFFFFFFFFU, send, 8);
    }
    const float keep = b2 ? h2[1] : h2[0];
    const float send = b2 ? h2[0] : h2[1];
    float total      = keep + __shfl_xor_sync(0xFFFFFFFFU, send, 4);
    total += __shfl_xor_sync(0xFFFFFFFFU, total, 2);
    total += __shfl_xor_sync(0xFFFFFFFFU, total, 1);
    return total;
}

// K1a with v4's staging (see above): one row of W_down per warp, 8 warps per CTA. Shared Rn:
// [pass column][plane][kHidden / 2].
template <class Codec, class Norm>
__global__ __launch_bounds__(kDownThreads, 1) void hc_down_partials_v4_kernel(
    const bf16* __restrict__ residual, const Norm* __restrict__ norm_weight, typename Codec::Params down, int rows,
    int columns, float eps, float* __restrict__ partial, float* __restrict__ inv_out) {
    constexpr int RowsPerWarp = 1;
    extern __shared__ __align__(16) float rn[]; // [kPassColumns][2][kHidden / 2]
    __shared__ float scratch[kDownThreads / 32][kPassColumns];
    __shared__ float inv[kPassColumns];
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int s    = static_cast<int>(blockIdx.y);
    const int row0 = (static_cast<int>(blockIdx.x) * (kDownThreads / 32) + warp) * RowsPerWarp;
    constexpr int width   = kStreams * kHidden;
    constexpr int kSlices = (kHidden + kDownThreads * 8 - 1) / (kDownThreads * 8);
    constexpr int kPlane  = kHidden / 8; // float4s per plane of one column
    static_assert(kHidden % 8 == 0);

    // 1. Weights first: the warp's rows of the stream's K slice, and the norm weights of the 8-element
    // groups this thread stages (i = 8 tid + 8 kDownThreads h, the same elements K1a's thread sums).
    typename Codec::Chunk chunk[RowsPerWarp][kChunks];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const std::int64_t source = row0 + r < rows ? row0 + r : 0;
#pragma unroll
        for (int m = 0; m < kChunks; ++m) { chunk[r][m] = Codec::load(down, source, s * kHidden + m * 256 + lane * 8); }
    }
    const Norm* w = norm_weight + static_cast<std::int64_t>(s) * kHidden;
    NormGroup<Norm> w_bits[kSlices];
#pragma unroll
    for (int h = 0; h < kSlices; ++h) {
        const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
        w_bits[h]   = i < kHidden ? NormGroup<Norm>::load(w + i) : NormGroup<Norm>::zero();
    }
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();

#pragma unroll 1
    for (int c0 = 0; c0 < columns; c0 += kPassColumns) {
        const int pass = min(kPassColumns, columns - c0);
        // 2. K1a's loads and per-thread sums of squares (its elements in index order), its warp
        // butterflies and its warps' sums in warp order.
        uint4 values[kPassColumns][kSlices];
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            const bf16* r = residual + static_cast<std::int64_t>(c0 + c) * width + static_cast<std::int64_t>(s) * kHidden;
#pragma unroll
            for (int h = 0; h < kSlices; ++h) {
                const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                if (c < pass && i < kHidden) { values[c][h] = load_ldg<uint4>(r + i); }
            }
        }
        float sums[kPassColumns];
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            sums[c] = 0.0F;
#pragma unroll
            for (int h = 0; h < kSlices; ++h) {
                const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                if (c < pass && i < kHidden) {
                    float v[8];
                    bf16x8_bits_to_floats(values[c][h], v);
#pragma unroll
                    for (int j = 0; j < 8; ++j) { sums[c] += v[j] * v[j]; }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            if (c < pass) {
                const float total = warp_sum(sums[c]);
                if (lane == 0) { scratch[warp][c] = total; }
            }
        }
        __syncthreads();
        if (static_cast<int>(threadIdx.x) < pass) {
            float total = 0.0F;
#pragma unroll
            for (int wp = 0; wp < kDownThreads / 32; ++wp) { total += scratch[wp][threadIdx.x]; }
            inv[threadIdx.x] = rsqrtf(total / static_cast<float>(kHidden) + eps);
        }
        __syncthreads();
        // Rn = R * inv * (1 + w), FP32, K1a's expression, stored once from the registers: group g of 8
        // elements goes to float4 g of plane 0 (elements 0-3) and of plane 1 (elements 4-7).
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            if (c < pass) {
                const float ic = inv[c];
                float4* plane  = reinterpret_cast<float4*>(rn) + c * 2 * kPlane;
#pragma unroll
                for (int h = 0; h < kSlices; ++h) {
                    const int i = static_cast<int>(threadIdx.x) * 8 + kDownThreads * 8 * h;
                    if (i < kHidden) {
                        float v[8], b[8];
                        bf16x8_bits_to_floats(values[c][h], v);
                        w_bits[h].floats(b);
#pragma unroll
                        for (int j = 0; j < 8; ++j) { b[j] = 1.0F + b[j]; }
                        const int g       = static_cast<int>(threadIdx.x) + kDownThreads * h;
                        plane[g] = make_float4(v[0] * ic * b[0], v[1] * ic * b[1], v[2] * ic * b[2],
                                               v[3] * ic * b[3]);
                        plane[kPlane + g] = make_float4(v[4] * ic * b[4], v[5] * ic * b[5], v[6] * ic * b[6],
                                                        v[7] * ic * b[7]);
                    }
                }
            }
        }
        if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < pass) {
            inv_out[static_cast<std::int64_t>(s) * columns + c0 + static_cast<int>(threadIdx.x)] = inv[threadIdx.x];
        }
        __syncthreads();
        // 3. Each row's dot products: the lane-chunk order, then a warp butterfly per (row, column).
        float acc[RowsPerWarp][kPassColumns];
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
            for (int c = 0; c < kPassColumns; ++c) { acc[r][c] = 0.0F; }
        }
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            const int g = m * 32 + lane;
            float wv[RowsPerWarp][8];
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) { Codec::decode(chunk[r][m], wv[r]); }
#pragma unroll
            for (int c = 0; c < kPassColumns; ++c) {
                if (c < pass) {
                    const float4* plane = reinterpret_cast<const float4*>(rn) + c * 2 * kPlane;
                    const float4 a = plane[g], b = plane[kPlane + g];
#pragma unroll
                    for (int r = 0; r < RowsPerWarp; ++r) {
                        acc[r][c] = fmaf(wv[r][0], a.x, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][1], a.y, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][2], a.z, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][3], a.w, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][4], b.x, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][5], b.y, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][6], b.z, acc[r][c]);
                        acc[r][c] = fmaf(wv[r][7], b.w, acc[r][c]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
            for (int c = 0; c < kPassColumns; ++c) {
                if (c < pass) {
                    const float total = warp_sum(acc[r][c]);
                    if (lane == 0 && row0 + r < rows) {
                        partial[(static_cast<std::int64_t>(s) * rows + row0 + r) * columns + c0 + c] = total;
                    }
                }
            }
        }
        __syncthreads(); // the next pass overwrites rn
    }
}

// K1b with v4's row map and transposed butterfly (see above).
constexpr int kUpV4Warps   = 8;
constexpr int kUpV4Threads = kUpV4Warps * 32;
constexpr int kUpV4Rows    = 8; // a warp's rows: features warp and warp + 8, 4 streams each

template <class Codec, class Norm>
__global__ __launch_bounds__(kUpV4Threads, 1) void hc_up_collapse_v4_kernel(
    const float* __restrict__ partial, const float* __restrict__ inv_in, const bf16* __restrict__ residual,
    const Norm* __restrict__ norm_weight, typename Codec::Params up, int rank, int down_rows, int columns,
    float* __restrict__ inject, bf16* __restrict__ x) {
    static_assert(kUpFeatures == 2 * kUpV4Warps && kUpV4Rows == 2 * kStreams);
    static_assert(kUpFeatures * kHcMixFusedMaxColumns <= kUpV4Threads);
    __shared__ __align__(16) float m[kHcMixFusedMaxColumns * kMaxRank];    // [T][2][rank / 2]
    __shared__ float u[kHcMixFusedMaxColumns * kStreams * kUpFeatures];   // [T][S][feature]
    __shared__ float inv[kStreams * kHcMixFusedMaxColumns];               // [S][T]
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31, warp = tid >> 5;
    const int d0   = static_cast<int>(blockIdx.x) * kUpFeatures;
    constexpr int width = kStreams * kHidden;
    const int chunks    = rank / 8;

    // 1. Weights first: row q of the warp is stream q / 2 of feature d0 + warp + 8 (q % 2), chunks lane
    // and lane + 32.
    typename Codec::Chunk chunk[kUpV4Rows][2];
#pragma unroll
    for (int q = 0; q < kUpV4Rows; ++q) {
        const int d            = d0 + warp + kUpV4Warps * (q & 1);
        const std::int64_t row = static_cast<std::int64_t>(q >> 1) * kHidden + (d < kHidden ? d : 0);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = lane + 32 * h;
            chunk[q][h] = c < chunks ? Codec::load(up, row, c * 8) : typename Codec::Chunk{};
        }
    }
    // The collapse thread of (token te, feature de) and its norm weights (1 + w).
    const int fe = tid % kUpFeatures, te = tid / kUpFeatures;
    const int de = d0 + fe;
    const int dl = de < kHidden ? de : 0;
    float w_bias[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        w_bias[s] = 1.0F + norm_value(norm_weight[static_cast<std::int64_t>(s) * kHidden + dl]);
    }
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    const bool collapses = de < kHidden && te < columns;
    float r[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        r[s] = collapses ? __bfloat162float(__ldg(residual + static_cast<std::int64_t>(te) * width +
                                                  static_cast<std::int64_t>(s) * kHidden + de))
                         : 0.0F;
    }

    // 2. z = sum_s partial[s] in order s = 0..S-1; m = SiLU(z / S) into plane (k / 4) % 2 at
    // 4 (k / 8) + k % 4; CTA 0 writes the injects.
    const float inv_streams = 1.0F / static_cast<float>(kStreams);
    const int half          = rank / 2;
#pragma unroll 4
    for (int i = tid; i < down_rows * columns; i += kUpV4Threads) {
        const int k = i / columns, t = i - k * columns;
        float z     = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            z += __ldg(partial + (static_cast<std::int64_t>(s) * down_rows + k) * columns + t);
        }
        const float v = z * inv_streams;
        if (k < rank) {
            m[t * rank + ((k >> 2) & 1) * half + (k >> 3) * 4 + (k & 3)] = v * sigmoidf(v);
        } else if (inject != nullptr && blockIdx.x == 0) {
            inject[static_cast<std::int64_t>(t) * kStreams + (k - rank)] = 2.0F * sigmoidf(v);
        }
    }
    for (int i = tid; i < kStreams * columns; i += kUpV4Threads) { inv[i] = __ldg(inv_in + i); }
    __syncthreads();

    // 3. Per token, the warp's 8 rows in the lane-chunk order, one transposed butterfly, and lane
    // 4q stores row q's sum.
#pragma unroll 1
    for (int t = 0; t < columns; ++t) {
        const float4* mt = reinterpret_cast<const float4*>(m + t * rank);
        float acc[kUpV4Rows];
#pragma unroll
        for (int q = 0; q < kUpV4Rows; ++q) { acc[q] = 0.0F; }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = lane + 32 * h;
            if (c < chunks) {
                const float4 a = mt[c], b = mt[chunks + c];
#pragma unroll
                for (int q = 0; q < kUpV4Rows; ++q) {
                    float wv[8];
                    Codec::decode(chunk[q][h], wv);
                    acc[q] = fmaf(wv[0], a.x, acc[q]);
                    acc[q] = fmaf(wv[1], a.y, acc[q]);
                    acc[q] = fmaf(wv[2], a.z, acc[q]);
                    acc[q] = fmaf(wv[3], a.w, acc[q]);
                    acc[q] = fmaf(wv[4], b.x, acc[q]);
                    acc[q] = fmaf(wv[5], b.y, acc[q]);
                    acc[q] = fmaf(wv[6], b.z, acc[q]);
                    acc[q] = fmaf(wv[7], b.w, acc[q]);
                }
            }
        }
        const float total = warp_sum8_transposed(acc, lane);
        if ((lane & 3) == 0) {
            const int q = lane >> 2;
            u[(t * kStreams + (q >> 1)) * kUpFeatures + warp + kUpV4Warps * (q & 1)] = total;
        }
    }
    __syncthreads();

    // 4. The collapse, K1b's expressions in its order.
    if (collapses) {
        float sum = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            const float rn = r[s] * inv[s * columns + te] * w_bias[s];
            sum += sigmoidf(u[(te * kStreams + s) * kUpFeatures + fe]) * rn;
        }
        x[static_cast<std::int64_t>(te) * kHidden + de] = __float2bfloat16_rn(sum * inv_streams);
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

// Launches hc_down_partials_v4_kernel in K1a's place.
template <class Codec, class Norm>
void launch_down_v4(const Tensor& residual, const Tensor& norm_weight, typename Codec::Params down, int down_rows,
                    int streams, float eps, float* partial, float* inv, std::size_t rn_bytes, cudaStream_t stream) {
    static const bool configured = [] {
        const cudaError_t error = cudaFuncSetAttribute(hc_down_partials_v4_kernel<Codec, Norm>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       static_cast<int>(kDownSharedBytes));
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("hyper_connection_mix: shared memory: ") + cudaGetErrorString(error));
        }
        return true;
    }();
    (void)configured;
    check_launch(pdl::launch_consumer({dim3((down_rows + kDownRows - 1) / kDownRows, streams), dim3(kDownThreads),
                                       rn_bytes, stream},
                                      hc_down_partials_v4_kernel<Codec, Norm>,
                                      static_cast<const bf16*>(residual.data), static_cast<const Norm*>(norm_weight.data),
                                      down, down_rows, residual.ne[1], eps, partial, inv),
                 "down partials v4");
}

template <class Codec, class Norm>
void launch(const Tensor& residual, const Tensor& norm_weight, typename Codec::Params down,
            typename Codec::Params up, int down_rows, int rank, int streams, float eps, Tensor& x, Tensor* inject,
            void* partials, cudaStream_t stream) {
    static const bool configured = [] {
        const cudaError_t error = cudaFuncSetAttribute(hc_down_partials_kernel<Codec, Norm>,
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
    // Both kernels read their weights before pdl::enter(), so as programmatic dependents in a captured
    // graph they fetch them while the kernel before them finishes.
    // Rn for the columns of one pass only, so the up collapse fits beside a down CTA on its SM.
    const std::size_t rn_bytes =
        static_cast<std::size_t>(std::min(columns, kPassColumns)) * kHidden * sizeof(float);
    // Two or more columns take v4's read scheduling (above); the v4 K1a reads the norm weights as
    // 16-byte vectors.
    const bool v4 = columns >= 2 && reinterpret_cast<std::uintptr_t>(norm_weight.data) % 16 == 0;
    if (v4) {
        launch_down_v4<Codec, Norm>(residual, norm_weight, down, down_rows, streams, eps, partial, inv, rn_bytes,
                                    stream);
    } else {
        check_launch(pdl::launch_consumer({dim3((down_rows + kDownRows - 1) / kDownRows, streams), dim3(kDownThreads),
                                           rn_bytes, stream},
                                          hc_down_partials_kernel<Codec, Norm>, static_cast<const bf16*>(residual.data),
                                          static_cast<const Norm*>(norm_weight.data), down, down_rows, columns, eps,
                                          partial, inv),
                     "down partials");
    }
    if (v4) {
        check_launch(pdl::launch_consumer({dim3((kHidden + kUpFeatures - 1) / kUpFeatures), dim3(kUpV4Threads), 0, stream},
                                          hc_up_collapse_v4_kernel<Codec, Norm>, static_cast<const float*>(partial),
                                          static_cast<const float*>(inv), static_cast<const bf16*>(residual.data),
                                          static_cast<const Norm*>(norm_weight.data), up, rank, down_rows, columns,
                                          inject != nullptr ? static_cast<float*>(inject->data) : nullptr,
                                          static_cast<bf16*>(x.data)),
                     "up collapse v4");
        return;
    }
    check_launch(pdl::launch_consumer({dim3((kHidden + kUpFeatures - 1) / kUpFeatures), dim3(kUpThreads), 0, stream},
                                      hc_up_collapse_kernel<Codec, Norm>, static_cast<const float*>(partial),
                                      static_cast<const float*>(inv), static_cast<const bf16*>(residual.data),
                                      static_cast<const Norm*>(norm_weight.data), up, rank, down_rows, columns,
                                      inject != nullptr ? static_cast<float*>(inject->data) : nullptr,
                                      static_cast<bf16*>(x.data)),
                 "up collapse");
}

template <class Codec>
void launch_norm(const Tensor& residual, const Tensor& norm_weight, typename Codec::Params down,
                 typename Codec::Params up, int down_rows, int rank, int streams, float eps, Tensor& x, Tensor* inject,
                 void* partials, cudaStream_t stream) {
    if (norm_weight.dtype == DType::FP32) {
        launch<Codec, float>(residual, norm_weight, down, up, down_rows, rank, streams, eps, x, inject, partials,
                             stream);
    } else {
        launch<Codec, bf16>(residual, norm_weight, down, up, down_rows, rank, streams, eps, x, inject, partials,
                            stream);
    }
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
        launch_norm<Q8Codec>(residual, norm_weight, q8_params(down), q8_params(up), down.n, rank, streams, eps, x,
                             inject, partials, stream);
    } else {
        launch_norm<Bf16Codec>(residual, norm_weight, bf16_params(down), bf16_params(up), down.n, rank, streams, eps,
                               x, inject, partials, stream);
    }
}

} // namespace infernix::ops::detail
