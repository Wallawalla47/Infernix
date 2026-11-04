// The fused route of hyper_connection_mix: two kernels instead of norm, the down projection,
// gates, the up projection and collapse (docs/maintainer/qwen3_8-flash-next-design.md §8.3 K1a/K1b,
// §19.3.3 Phase 1b), for Q8_G32_FP16 RowSplit weights (Dense8), BF16 contiguous weights (the
// bit-exact artifact) and bf16_p12 weights (those BF16 weights packed at load, which decode to the same
// values). The weight codec only changes how a lane's 8 weights are read; the order of every product and
// sum is the codec-independent schedule below.
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
#include "ops/ggml/ggml_a8.h"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace infernix::ops::detail {
namespace {

namespace cg = cooperative_groups;
using bf16   = __nv_bfloat16;

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

// A codec reads a row's 8-weight chunk (load: the chunk at row `row`, columns k..k+7) with the row's state (row(), loaded
// once a row) and widens it to 8 floats (widen()). A decoding codec (its Ready differs from its Chunk: kDecodesEarly)
// turns each chunk into the form widen() takes (ready(), once a chunk).

// q8_g32_fp16 in row_split_k128_v1: int8 codes [rows, padded_k], FP16 scales per 32 codes.
struct Q8Codec {
    struct Params {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
        int padded_k;
    };
    struct Row {};
    struct Chunk {
        uint2 code;
        float scale;
    };
    using Ready = Chunk;
    __device__ static Row row(const Params&, std::int64_t) { return {}; }
    __device__ static Chunk load(const Params& p, std::int64_t row, int k) {
        return {ld_nc_na<uint2>(p.codes + row * p.padded_k + k),
                half_bits_to_float(ld_nc_na<std::uint16_t>(p.scales + row * (p.padded_k / 32) * 2 + (k / 32) * 2))};
    }
    __device__ static void widen(const Ready& c, float (&w)[8]) {
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
    struct Row {};
    struct Chunk {
        uint4 bits;
    };
    using Ready = Chunk;
    __device__ static Row row(const Params&, std::int64_t) { return {}; }
    __device__ static Chunk load(const Params& p, std::int64_t row, int k) {
        return {ld_nc_na<uint4>(p.data + row * p.stride + k)};
    }
    __device__ static void widen(const Ready& c, float (&w)[8]) {
        const float2 a = bf16x2_bits_to_float2(c.bits.x), b = bf16x2_bits_to_float2(c.bits.y);
        const float2 d = bf16x2_bits_to_float2(c.bits.z), e = bf16x2_bits_to_float2(c.bits.w);
        w[0] = a.x, w[1] = a.y, w[2] = b.x, w[3] = b.y, w[4] = d.x, w[5] = d.y, w[6] = e.x, w[7] = e.y;
    }
};

// bf16_p12 (core/weight.h P12Rows, packed at load): a chunk is its 8 code bytes and its nibble word (fewer registers
// than a BF16 chunk), a row's state its window base; a chunk with escapes reads its row's escape range when it is made
// ready. Every weight decodes to its BF16 bits, so the mixer's arithmetic is the BF16 codec's. (Staging each row's
// escapes in shared memory cost registers, which kept the next kernel off the up kernel's SMs, and gained nothing once
// the decode moved before the dependency wait.)
struct P12Codec {
    struct Params {
        const std::uint8_t* codes;
        const std::uint8_t* nibbles;
        const unsigned* words;
        const unsigned* escapes;
        int k;
    };
    struct Row {
        unsigned base2;
    };
    struct Chunk {
        uint2 code;
        unsigned nib;
    };
    using Ready = Bf16Codec::Chunk;
    __device__ static Row row(const Params& p, std::int64_t row) { return {(__ldg(p.words + row) & 0xFFU) * 0x10001U}; }
    __device__ static Chunk load(const Params& p, std::int64_t row, int k) {
        return {ld_nc_na<uint2>(p.codes + row * p.k + k), ld_nc_na<unsigned>(p.nibbles + row * (p.k / 2) + k / 2)};
    }
    __device__ static Ready ready(const Params& p, const Row& r, const Chunk& c, std::int64_t row, int k) {
        P12Row state{r.base2, 0, 0};
        if (p12_has_escape(c.nib)) { state = p12_row(p.words, row); }
        return {p12_decode_chunk(c.code, c.nib, state, k / 8, p.escapes)};
    }
    __device__ static void widen(const Ready& c, float (&w)[8]) { Bf16Codec::widen(c, w); }
};

// A codec whose chunks need decoding (bf16_p12): the kernels make every chunk ready once, after the weight loads and
// before the programmatic-dependency wait (weights do not depend on the producer, whose tail hides the decode), and
// their dot-product loops then widen ready BF16 words exactly as the BF16 codec's do.
template <class Codec>
inline constexpr bool kDecodesEarly = !std::is_same_v<typename Codec::Ready, typename Codec::Chunk>;

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
    typename Codec::Row row_state = Codec::row(down, source);
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
    [[maybe_unused]] typename Codec::Ready weights[kChunks];
    if constexpr (kDecodesEarly<Codec>) {
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            weights[m] = Codec::ready(down, row_state, chunk[m], source, s * kHidden + m * 256 + lane * 8);
        }
    }
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
            if constexpr (kDecodesEarly<Codec>) {
                Codec::widen(weights[m], wv);
            } else {
                Codec::widen(chunk[m], wv);
            }
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
    typename Codec::Row row_state[kStreams];
    typename Codec::Chunk chunk[kStreams][2];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        const std::int64_t row = static_cast<std::int64_t>(s) * kHidden + (live ? d : 0);
        row_state[s]           = Codec::row(up, row);
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
    [[maybe_unused]] typename Codec::Ready weights[kStreams][2];
    if constexpr (kDecodesEarly<Codec>) {
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                weights[s][h] = Codec::ready(up, row_state[s], chunk[s][h], static_cast<std::int64_t>(s) * kHidden + dl,
                                             (lane + 32 * h) * 8);
            }
        }
    }
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
                    if constexpr (kDecodesEarly<Codec>) {
                        Codec::widen(weights[s][h], wv);
                    } else {
                        Codec::widen(chunk[s][h], wv);
                    }
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
//     keeps 8 warps of one row each, so the norm's per-thread sums and warp order are unchanged. A
//     pass's 8 column sums (the squares, and each row's dot products) reduce by one transposed
//     butterfly each, 9 shuffles for 8 x 5, with warp_sum's bits (candidates SZ-03, 2026-11-04).
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
    typename Codec::Row row_state[RowsPerWarp];
    typename Codec::Chunk chunk[RowsPerWarp][kChunks];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const std::int64_t source = row0 + r < rows ? row0 + r : 0;
        row_state[r]              = Codec::row(down, source);
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
    [[maybe_unused]] typename Codec::Ready weights[RowsPerWarp][kChunks];
    if constexpr (kDecodesEarly<Codec>) {
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            const std::int64_t source = row0 + r < rows ? row0 + r : 0;
#pragma unroll
            for (int m = 0; m < kChunks; ++m) {
                weights[r][m] = Codec::ready(down, row_state[r], chunk[r][m], source, s * kHidden + m * 256 + lane * 8);
            }
        }
    }
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
        // The pass's 8 column sums by one transposed butterfly (9 shuffles for 8 x 5): lane 4c holds column c's
        // warp_sum, bit for bit (see warp_sum8_transposed); a column past the pass sums zeros and is not stored.
        static_assert(kPassColumns == 8, "the transposed butterfly reduces eight values");
        {
            const float total = warp_sum8_transposed(sums, lane);
            if ((lane & 3) == 0 && (lane >> 2) < pass) { scratch[warp][lane >> 2] = total; }
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
            for (int r = 0; r < RowsPerWarp; ++r) {
                if constexpr (kDecodesEarly<Codec>) {
                    Codec::widen(weights[r][m], wv[r]);
                } else {
                    Codec::widen(chunk[r][m], wv[r]);
                }
            }
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
        // Each row's 8 column sums by one transposed butterfly: lane 4c stores column c.
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            const float total = warp_sum8_transposed(acc[r], lane);
            const int c       = lane >> 2;
            if ((lane & 3) == 0 && c < pass && row0 + r < rows) {
                partial[(static_cast<std::int64_t>(s) * rows + row0 + r) * columns + c0 + c] = total;
            }
        }
        __syncthreads(); // the next pass overwrites rn
    }
}

// K1b with v4's row map and transposed butterfly (see above). With kA8 (--a8-activations: x's 8-bit copy for the
// next GGUF projection), CTAs 2c and 2c + 1 form a cluster holding the two halves of 32-element slice c of every
// column, and quantize it with ops/ggml/ggml_a8.h's codec, exchanging the halves' maxima and code sums through
// distributed shared memory. (One CTA a slice, 32 features, measured slower: 80 CTAs at H 2,560 against 160, +2-3 us a
// call at 2-4 columns, more than the encoding launch it saves.)
constexpr int kUpV4Warps   = 8;
constexpr int kUpV4Threads = kUpV4Warps * 32;
constexpr int kUpV4Rows    = 8; // a warp's rows: features warp and warp + 8, 4 streams each

// x's 8-bit copy (null: none): column t's words at words + t * stride (ggml_linear_a8's layout; slice sums with
// `sums`).
struct UpA8Columns {
    std::int32_t* words = nullptr;
    int stride          = 0;
    int sums            = 0;
};

// The v4 collapse's 16-byte norm-weight loads, and with them the v4 kernels, need this.
bool v4_norm(const Tensor& norm_weight) noexcept { return reinterpret_cast<std::uintptr_t>(norm_weight.data) % 16 == 0; }

template <class Codec, class Norm, bool kA8>
__global__ __launch_bounds__(kUpV4Threads, 1) void hc_up_collapse_v4_kernel(
    const float* __restrict__ partial, const float* __restrict__ inv_in, const bf16* __restrict__ residual,
    const Norm* __restrict__ norm_weight, typename Codec::Params up, int rank, int down_rows, int columns,
    float* __restrict__ inject, bf16* __restrict__ x, UpA8Columns a8_columns) {
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
    typename Codec::Row row_state[kUpV4Rows];
    typename Codec::Chunk chunk[kUpV4Rows][2];
#pragma unroll
    for (int q = 0; q < kUpV4Rows; ++q) {
        const int d            = d0 + warp + kUpV4Warps * (q & 1);
        const std::int64_t row = static_cast<std::int64_t>(q >> 1) * kHidden + (d < kHidden ? d : 0);
        row_state[q]           = Codec::row(up, row);
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
    [[maybe_unused]] typename Codec::Ready weights[kUpV4Rows][2];
    if constexpr (kDecodesEarly<Codec>) {
#pragma unroll
        for (int q = 0; q < kUpV4Rows; ++q) {
            const int d            = d0 + warp + kUpV4Warps * (q & 1);
            const std::int64_t row = static_cast<std::int64_t>(q >> 1) * kHidden + (d < kHidden ? d : 0);
#pragma unroll
            for (int h = 0; h < 2; ++h) { weights[q][h] = Codec::ready(up, row_state[q], chunk[q][h], row, (lane + 32 * h) * 8); }
        }
    }
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
    // 4q stores row q's sum. Tokens go in groups of G (2, the last one alone for an odd count): each weight chunk
    // is widened once a group, and every (row, token) sum keeps its fmaf order (candidates NQ-02).
    const auto group = [&]<int G>(int t0) {
        float acc[G][kUpV4Rows];
#pragma unroll
        for (int g = 0; g < G; ++g) {
#pragma unroll
            for (int q = 0; q < kUpV4Rows; ++q) { acc[g][q] = 0.0F; }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = lane + 32 * h;
            if (c < chunks) {
                float4 a[G], b[G];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float4* mt = reinterpret_cast<const float4*>(m + (t0 + g) * rank);
                    a[g] = mt[c];
                    b[g] = mt[chunks + c];
                }
#pragma unroll
                for (int q = 0; q < kUpV4Rows; ++q) {
                    float wv[8];
                    if constexpr (kDecodesEarly<Codec>) {
                        Codec::widen(weights[q][h], wv);
                    } else {
                        Codec::widen(chunk[q][h], wv);
                    }
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        acc[g][q] = fmaf(wv[0], a[g].x, acc[g][q]);
                        acc[g][q] = fmaf(wv[1], a[g].y, acc[g][q]);
                        acc[g][q] = fmaf(wv[2], a[g].z, acc[g][q]);
                        acc[g][q] = fmaf(wv[3], a[g].w, acc[g][q]);
                        acc[g][q] = fmaf(wv[4], b[g].x, acc[g][q]);
                        acc[g][q] = fmaf(wv[5], b[g].y, acc[g][q]);
                        acc[g][q] = fmaf(wv[6], b[g].z, acc[g][q]);
                        acc[g][q] = fmaf(wv[7], b[g].w, acc[g][q]);
                    }
                }
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float total = warp_sum8_transposed(acc[g], lane);
            if ((lane & 3) == 0) {
                const int q = lane >> 2;
                u[((t0 + g) * kStreams + (q >> 1)) * kUpFeatures + warp + kUpV4Warps * (q & 1)] = total;
            }
        }
    };
    constexpr int kUpTokenGroup = 2;
#pragma unroll 1
    for (int t = 0; t + kUpTokenGroup <= columns; t += kUpTokenGroup) { group.template operator()<kUpTokenGroup>(t); }
    if (columns % kUpTokenGroup != 0) { group.template operator()<1>(columns - 1); }
    __syncthreads();

    // 4. The collapse, K1b's expressions in its order.
    bf16 out = __float2bfloat16_rn(0.0F);
    if (collapses) {
        float sum = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            const float rn = r[s] * inv[s * columns + te] * w_bias[s];
            sum += sigmoidf(u[(te * kStreams + s) * kUpFeatures + fe]) * rn;
        }
        out = __float2bfloat16_rn(sum * inv_streams);
        x[static_cast<std::int64_t>(te) * kHidden + de] = out;
    }
    // 5. The 8-bit copy: the 16 threads of column te (a half-warp) hold half `half_of` of slice blockIdx.x / 2,
    // quantized as ggml_linear_a8's encode_kernel quantizes the slice (the same key, scale, codes and code sum, so the
    // same words). Every thread passes the shuffles and both cluster barriers; each half-warp reduces within itself.
    if constexpr (kA8) {
        namespace a8 = ggml::a8;
        __shared__ unsigned partner_key[kHcMixFusedMaxColumns]; // written by the other CTA
        __shared__ int partner_sum[kHcMixFusedMaxColumns];      // rank 0's: written by rank 1
        cg::cluster_group cluster = cg::this_cluster();
        const unsigned half_of    = cluster.block_rank();
        const bool live           = te < columns;
        const auto bits           = __bfloat16_as_ushort(out);
        unsigned key              = bits & 0x7FFFU;
#pragma unroll
        for (int o = 8; o >= 1; o >>= 1) { key = max(key, __shfl_xor_sync(0xFFFFFFFFU, key, o)); }
        if (live && fe == 0) { *cluster.map_shared_rank(&partner_key[te], half_of ^ 1U) = key; }
        cluster.sync();
        if (live) { key = max(key, partner_key[te]); }
        const a8::SliceScale sc = a8::slice_scale(key);
        const int code          = live ? a8::code(bits, sc.inv) : 0;
        const int c1 = __shfl_down_sync(0xFFFFFFFFU, code, 1), c2 = __shfl_down_sync(0xFFFFFFFFU, code, 2),
                  c3 = __shfl_down_sync(0xFFFFFFFFU, code, 3);
        int sum = code;
#pragma unroll
        for (int o = 8; o >= 1; o >>= 1) { sum += __shfl_xor_sync(0xFFFFFFFFU, sum, o); }
        const int slice      = static_cast<int>(blockIdx.x) / 2;
        std::int32_t* column = a8_columns.words + static_cast<std::int64_t>(te) * a8_columns.stride;
        if (live && (fe & 3) == 0) {
            column[slice * a8::kSliceWords + static_cast<int>(half_of) * 4 + fe / 4] = ggml::fast::pack4(code, c1, c2, c3);
        }
        if (live && fe == 0 && half_of == 1) { *cluster.map_shared_rank(&partner_sum[te], 0U) = sum; }
        cluster.sync(); // no access to the other CTA's shared memory follows
        if (live && fe == 0 && half_of == 0) {
            reinterpret_cast<float*>(column + a8::scales_offset(kHidden))[slice] = sc.d;
            if (a8_columns.sums != 0) { column[a8::sums_offset(kHidden) + slice] = sum + partner_sum[te]; }
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

bool bf16_p12(const Weight& w) noexcept {
    return w.qtype == QType::BF16_P12 && w.layout == QuantLayout::P12Rows && w.qdata != nullptr && w.qhigh != nullptr &&
           w.scales != nullptr && w.escapes != nullptr && w.k % 8 == 0;
}

P12Codec::Params p12_params(const Weight& w) {
    return {static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
            static_cast<const unsigned*>(w.scales), static_cast<const unsigned*>(w.escapes), w.k};
}

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
    // One CTA per row group and stream, looping over the columns. A grid of one CTA per column at 2-8 columns
    // (candidates §2.3 ST-03) was slower: Dense8 MTP-3 0.996x, IQ3_S DFlash2 0.983x (2026-10-28, removed).
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
            void* partials, cudaStream_t stream, UpA8Columns a8) {
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
    const bool v4 = columns >= 2 && v4_norm(norm_weight);
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
        const dim3 grid((kHidden + kUpFeatures - 1) / kUpFeatures);
        auto* const inject_data = inject != nullptr ? static_cast<float*>(inject->data) : nullptr;
        if (a8.words == nullptr) {
            check_launch(pdl::launch_consumer({grid, dim3(kUpV4Threads), 0, stream},
                                              hc_up_collapse_v4_kernel<Codec, Norm, false>,
                                              static_cast<const float*>(partial), static_cast<const float*>(inv),
                                              static_cast<const bf16*>(residual.data),
                                              static_cast<const Norm*>(norm_weight.data), up, rank, down_rows, columns,
                                              inject_data, static_cast<bf16*>(x.data), a8),
                         "up collapse v4");
            return;
        }
        // x's 8-bit copy: clusters of two CTAs (a slice's halves), launched as launch_consumer launches.
        static_assert(kHidden % (2 * kUpFeatures) == 0);
        cudaLaunchAttribute attributes[2]{};
        attributes[0].id               = cudaLaunchAttributeClusterDimension;
        attributes[0].val.clusterDim.x = 2;
        attributes[0].val.clusterDim.y = 1;
        attributes[0].val.clusterDim.z = 1;
        unsigned count                 = 1;
        if (pdl::graph_dependent(stream, hc_up_collapse_v4_kernel<Codec, Norm, true>)) {
            attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[1].val.programmaticStreamSerializationAllowed = 1;
            count                                                    = 2;
        }
        cudaLaunchConfig_t config{};
        config.gridDim  = grid;
        config.blockDim = dim3(kUpV4Threads);
        config.stream   = stream;
        config.attrs    = attributes;
        config.numAttrs = count;
        check_launch(cudaLaunchKernelEx(&config, hc_up_collapse_v4_kernel<Codec, Norm, true>,
                                        static_cast<const float*>(partial), static_cast<const float*>(inv),
                                        static_cast<const bf16*>(residual.data),
                                        static_cast<const Norm*>(norm_weight.data), up, rank, down_rows, columns,
                                        inject_data, static_cast<bf16*>(x.data), a8),
                     "up collapse v4 (8-bit copy)");
        return;
    }
    if (a8.words != nullptr) { throw std::logic_error("hyper_connection_mix: an 8-bit copy needs the v4 kernels"); }
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
                 void* partials, cudaStream_t stream, UpA8Columns a8) {
    if (norm_weight.dtype == DType::FP32) {
        launch<Codec, float>(residual, norm_weight, down, up, down_rows, rank, streams, eps, x, inject, partials,
                             stream, a8);
    } else {
        launch<Codec, bf16>(residual, norm_weight, down, up, down_rows, rank, streams, eps, x, inject, partials,
                            stream, a8);
    }
}

} // namespace

bool hc_mix_fused_supported(const Weight& down, const Weight& up, std::int32_t hidden, std::int32_t streams,
                            std::int32_t rank, std::int32_t columns) noexcept {
    const bool codec = (q8_rowsplit(down) && q8_rowsplit(up)) || (bf16_contiguous(down) && bf16_contiguous(up)) ||
                       (bf16_p12(down) && bf16_p12(up));
    return codec && columns >= 1 && columns <= kHcMixFusedMaxColumns && hidden == kHidden && streams == kStreams &&
           rank > 0 && rank % 8 == 0 && rank <= kMaxRank && down.k == kStreams * kHidden &&
           (down.n == rank || down.n == rank + kStreams) && up.n == kStreams * kHidden && up.k == rank;
}

std::size_t hc_mix_fused_workspace_bytes(std::int32_t down_rows, std::int32_t streams, std::int32_t columns) noexcept {
    const auto align = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    return align(static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float)) +
           align(static_cast<std::size_t>(streams) * columns * sizeof(float));
}

bool hc_mix_fused_a8_supported(const Tensor& norm_weight, std::int32_t columns) noexcept {
    return columns >= 2 && columns <= kHcMixFusedMaxColumns && v4_norm(norm_weight);
}

void hc_mix_fused(const Tensor& residual, const Tensor& norm_weight, const Weight& down, const Weight& up,
                  std::int32_t streams, std::int32_t rank, float eps, Tensor& x, Tensor* inject, void* partials,
                  cudaStream_t stream, HcMixA8Columns a8) {
    if (!hc_mix_fused_supported(down, up, x.ne[0], streams, rank, residual.ne[1])) {
        throw std::invalid_argument("hyper_connection_mix: the fused route does not serve this problem");
    }
    if (inject != nullptr && down.n != rank + streams) {
        throw std::invalid_argument("hyper_connection_mix: injects need the down projection's injection rows");
    }
    if (a8.words != nullptr && !hc_mix_fused_a8_supported(norm_weight, residual.ne[1])) {
        throw std::invalid_argument("hyper_connection_mix: an 8-bit copy of x needs 2-16 columns and aligned norm weights");
    }
    const UpA8Columns columns{a8.words, a8.stride, a8.sums ? 1 : 0};
    if (down.qtype == QType::Q8_G32_FP16) {
        launch_norm<Q8Codec>(residual, norm_weight, q8_params(down), q8_params(up), down.n, rank, streams, eps, x,
                             inject, partials, stream, columns);
    } else if (down.qtype == QType::BF16_P12) {
        launch_norm<P12Codec>(residual, norm_weight, p12_params(down), p12_params(up), down.n, rank, streams, eps, x,
                              inject, partials, stream, columns);
    } else {
        launch_norm<Bf16Codec>(residual, norm_weight, bf16_params(down), bf16_params(up), down.n, rank, streams, eps,
                               x, inject, partials, stream, columns);
    }
}

} // namespace infernix::ops::detail
