// The canonical W4A16 dense projection of an nvfp4_rg16_v1 matrix (include/infernix/ops/rg16_linear.h)
// at its registered shapes: Qwen3.5-122B-A10B's output head [248320, 3072] (429 MB read once per
// call), and Qwen3.8-Flash-Next FP4-Local's vision MLP (fc1 [4304, 1152], fc2 [1152, 4304]) and
// merger (fc1 [4608, 4608], fc2 [2560, 4608]), which encode every patch of an image: widths of
// hundreds to tens of thousands of columns.
//
// Through the narrow width a dp4a GEMV: x is encoded once per call (canon::a16 encoding, per column),
// every CTA keeps the encoded columns in shared memory, and each warp owns one 16-row group at a
// time, streaming its units (evict-first loads) with the narrow expert route's three dp4a per block
// and an exact int64 row sum. The narrow width is eight columns, or four where eight columns'
// encoded blocks would not fit in shared memory (K 4304 and 4608). Wider calls run the exact wide
// expert route's grouped integer tensor-core kernel (wide_expert_a16.cuh) on the matrix as one job,
// its tile columns the call's columns, in passes of at most kWideColumns columns that bound the
// encoded activations' workspace: 128-row blocks by 64-column tiles, K in 128-element stages. The vision
// matrices' 4304 rows (fc1) and 4304-element K (fc2) are not whole blocks and stages; the kernel runs
// partial last ones (its limb rows are zero past K, so a short stage's stale units add exact zeros,
// and the epilogue skips absent row groups). Both routes form the same exact S, so outputs depend
// on neither the route, nor T, nor the passes.
//
// The narrow widths' streamed GEMV (candidates §11 OR-10), for the head's decode and verification calls
// (the 122B head took 337 µs a token at 1.27 TB/s, against 260 µs at the DRAM roofline; with the stream the
// 122B's decode is +0.9 %, 5 of 6 ABBA blocks ahead and one tied, 2026-10-25). One 1,024-thread CTA per SM; each warp keeps the next kDepth units'
// words of its row-group sequence in registers (three words a block, issued a block ahead of their
// use across row-group boundaries), so a warp holds kDepth × 144 bytes in flight instead of an
// unrolled loop's few blocks; the CTA encodes the call's columns into shared memory itself (the
// encode kernel's arithmetic, so no separate launch), and the kernel is a PDL consumer that loads its
// first units before waiting for x. Same products and exact sums: the same bits.

#include "infernix/ops/rg16_linear.h"

#include "core/device.h"
#include "core/layout.h"
#include "core/pdl.cuh"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

namespace moe  = offloaded_moe;
namespace wide = offloaded_moe::wide;

constexpr int kWarps       = 16;
constexpr int kThreads     = kWarps * 32;
constexpr int kNarrowBytes = 99 * 1024; // the GEMV's encoded columns, in shared memory
// Columns per wide-route pass: the encoded activations (3 bytes per element) of at most this many
// columns live in the workspace at once. Wider calls (a large image's patches) take several passes;
// the vision matrices stay in L2 across them, and every pass fills the GPU (fc2 [1152, 4304]: 576
// tiles of 128 rows by 64 columns, 3.4 waves on 170 SMs).
constexpr int kWideColumns = 4096;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("rg16_linear: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("rg16_linear ") + what + ": " + cudaGetErrorString(error));
    }
}

// An encoded A16 block as the dp4a loop reads it: elements 4q..4q+3's bytes in word q (the narrow
// expert route's A16Act).
struct Act {
    std::uint32_t lo[4];
    std::uint32_t mid[4];
    std::int32_t hi[4];
};

__host__ __device__ constexpr int round_up32(int n) { return (n + 31) / 32 * 32; }

// A registered matrix [Rows, K]: its 16-row groups and 16-element blocks (one 144-byte unit each per
// row group), the wide route's 128-element stages (the last one partial when K is not a multiple of
// 128) and the bytes of an encoded limb row (whole stages, zero past K), and the GEMV's widest call.
template <int Rows, int K>
struct Dense {
    using S = wide::ScheduleA16;
    static_assert(Rows % 16 == 0 && K % 16 == 0, "nvfp4_rg16_v1 holds whole row groups and blocks");
    static constexpr int kRows     = Rows;
    static constexpr int kK        = K;
    static constexpr int kGroups   = Rows / 16;
    static constexpr int kBlocks   = K / 16;
    static constexpr int kStages   = (kBlocks + S::kUnitsPerKTile - 1) / S::kUnitsPerKTile;
    static constexpr int kPadBlocks = kStages * S::kUnitsPerKTile;
    static constexpr int kRowBytes = kStages * S::kStageRowBytes;
    static constexpr int kNarrow =
        static_cast<int>(sizeof(Act)) * kBlocks * moe::kMaxColumns <= kNarrowBytes ? moe::kMaxColumns : 4;
    static_assert(static_cast<int>(sizeof(Act)) * kBlocks * kNarrow <= kNarrowBytes);
};

using Head      = Dense<248320, 3072>; // Qwen3.5-122B-A10B's lm_head
using VisionFc1 = Dense<4304, 1152>;   // Qwen3.8-Flash-Next FP4-Local's vision MLP
using VisionFc2 = Dense<1152, 4304>;
using MergerFc1 = Dense<4608, 4608>;   // and its merger
using MergerFc2 = Dense<2560, 4608>;
static_assert(Head::kNarrow == 8 && VisionFc1::kNarrow == 8 && VisionFc2::kNarrow == 4 && MergerFc1::kNarrow == 4);
static_assert(Head::kRowBytes == wide::A16Planes<moe::Qwen3_5_122BExperts>::kXRowBytes);

// Calls f.template operator()<D>() with the registered matrix of [rows, k].
template <class F>
decltype(auto) with_dense(std::int32_t rows, std::int32_t k, F&& f) {
    if (rows == Head::kRows && k == Head::kK) { return f.template operator()<Head>(); }
    if (rows == VisionFc1::kRows && k == VisionFc1::kK) { return f.template operator()<VisionFc1>(); }
    if (rows == VisionFc2::kRows && k == VisionFc2::kK) { return f.template operator()<VisionFc2>(); }
    if (rows == MergerFc1::kRows && k == MergerFc1::kK) { return f.template operator()<MergerFc1>(); }
    if (rows == MergerFc2::kRows && k == MergerFc2::kK) { return f.template operator()<MergerFc2>(); }
    throw std::invalid_argument("rg16_linear: unregistered [N, K] (the 122B head [248320, 3072], the FP4-Local vision "
                                "MLP [4304, 1152], [1152, 4304] and merger [4608, 4608], [2560, 4608])");
}

// Signed bytes of a times unsigned bytes of b, plus c (PTX dp4a.s32.u32).
__device__ __forceinline__ int dp4a_su(int a, std::uint32_t b, int c) {
    int d;
    asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Block b of a column of K elements: its eight BF16 pairs.
template <int K>
__device__ __forceinline__ void load_block(const std::uint16_t* __restrict__ column, int b, unsigned (&words)[8]) {
    const auto* source = reinterpret_cast<const uint4*>(column + 16 * b);
    const uint4 v0 = source[0], v1 = source[1];
    words[0] = v0.x, words[1] = v0.y, words[2] = v0.z, words[3] = v0.w;
    words[4] = v1.x, words[5] = v1.y, words[6] = v1.z, words[7] = v1.w;
}

// A block's largest magnitude bits, the column exponent's input.
__device__ __forceinline__ unsigned block_key(const unsigned (&words)[8]) {
    unsigned key = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) { key = max(key, max(words[w] & 0x7FFFU, (words[w] >> 16) & 0x7FFFU)); }
    return key;
}

// The column exponent of its blocks' largest key.
__device__ __forceinline__ int column_exponent(unsigned all) {
    return all >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(all >> 7), 1);
}

// A block's A16 encoding under its column's exponent.
__device__ __forceinline__ Act encode_block(const unsigned (&words)[8], int e) {
    Act out{};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const auto h  = static_cast<std::uint16_t>(j % 2 == 0 ? words[j / 2] & 0xFFFFU : words[j / 2] >> 16);
        const auto xv = static_cast<std::uint32_t>(e == canon::kA16NonFinite ? 0 : canon::a16_value(h, e));
        const int q = j / 4, s = 8 * (j % 4);
        out.lo[q] |= (xv & 0xFFU) << s;
        out.mid[q] |= ((xv >> 8) & 0xFFU) << s;
        out.hi[q] = static_cast<std::int32_t>(static_cast<std::uint32_t>(out.hi[q]) | (((xv >> 16) & 0xFFU) << s));
    }
    return out;
}

// One CTA per column, one 16-element block per thread (the last warp's spare threads only join the
// exponent): the column exponent, then every block's encoding into acts[column][block].
template <class D>
__global__ void __launch_bounds__(round_up32(D::kBlocks))
    encode_kernel(const std::uint16_t* __restrict__ x, Act* __restrict__ acts, std::int32_t* __restrict__ emax) {
    constexpr int kEncodeThreads = round_up32(D::kBlocks);
    __shared__ unsigned keys[kEncodeThreads / 32];
    const int t = static_cast<int>(blockIdx.x), b = static_cast<int>(threadIdx.x);
    const bool block = b < D::kBlocks;
    unsigned words[8] = {};
    if (block) { load_block<D::kK>(x + static_cast<std::size_t>(t) * D::kK, b, words); }
    const unsigned key = __reduce_max_sync(0xFFFFFFFFU, block_key(words));
    if (b % 32 == 0) { keys[b / 32] = key; }
    __syncthreads();
    if (!block) { return; }
    unsigned all = 0;
#pragma unroll
    for (int w = 0; w < kEncodeThreads / 32; ++w) { all = max(all, keys[w]); }
    const int e = column_exponent(all);
    acts[static_cast<std::size_t>(t) * D::kBlocks + b] = encode_block(words, e);
    if (b == 0) { emax[t] = e; }
}

// One block's products for a lane (row r of its warp's row group at k = 8 half .. 8 half + 7) with
// every column's encoded block, the halves joined before the scale, into s (the narrow expert
// route's three dp4a per block and exact int64 row sum).
template <int Columns>
__device__ __forceinline__ void accumulate_block(const Act* __restrict__ shared, int blocks, int b, int half, int n,
                                                 unsigned c0, unsigned c1, unsigned sc, int shift, int scale_shift,
                                                 long long (&s)[Columns]) {
    const int w0 = static_cast<int>(canon::e2m1_x2_quad((c0 >> shift) & 0x0F0F0F0FU));
    const int w1 = static_cast<int>(canon::e2m1_x2_quad((c1 >> shift) & 0x0F0F0F0FU));
    const int sw = canon::e4m3_scaled((sc >> scale_shift) & 0xFFU);
#pragma unroll
    for (int c = 0; c < Columns; ++c) {
        if (c < n) { // uniform across the warp
            const Act& a  = shared[c * blocks + b];
            const int lo  = dp4a_su(w1, a.lo[2 * half + 1], dp4a_su(w0, a.lo[2 * half], 0));
            const int mid = dp4a_su(w1, a.mid[2 * half + 1], dp4a_su(w0, a.mid[2 * half], 0));
            const int hi  = __dp4a(w1, a.hi[2 * half + 1], __dp4a(w0, a.hi[2 * half], 0));
            unsigned p = static_cast<unsigned>(lo) + (static_cast<unsigned>(mid) << 8) + (static_cast<unsigned>(hi) << 16);
            p += static_cast<unsigned>(__shfl_xor_sync(0xFFFFFFFFU, static_cast<int>(p), 16));
            s[c] += static_cast<long long>(static_cast<int>(p)) * sw;
        }
    }
}

// Row r of row group g into every column's output (lanes 0-15).
template <class D, int Columns, bool F32>
__device__ __forceinline__ void store_rows(const long long (&s)[Columns], const std::int32_t* emax, int g, int r, int n,
                                           float multiplier, void* __restrict__ out) {
    const std::size_t row = static_cast<std::size_t>(g) * 16 + r;
#pragma unroll
    for (int c = 0; c < Columns; ++c) {
        if (c >= n) { continue; }
        const std::size_t at = static_cast<std::size_t>(c) * D::kRows + row;
        if constexpr (F32) {
            static_cast<float*>(out)[at] = canon::a16_row_output_f32(s[c], emax[c], multiplier);
        } else {
            static_cast<std::uint16_t*>(out)[at] = canon::a16_row_output(s[c], emax[c], multiplier);
        }
    }
}

// The wide route's A operand: each column's limb row (wide_expert_a16.cuh) and exponent, one CTA per
// column and one 16-element block per thread. The blocks of the last stage past K are zero.
template <class D>
__global__ void __launch_bounds__(round_up32(D::kPadBlocks))
    limbs_kernel(const std::uint16_t* __restrict__ x, std::uint8_t* __restrict__ limbs, std::int32_t* __restrict__ emax) {
    constexpr int kEncodeThreads = round_up32(D::kPadBlocks);
    __shared__ unsigned keys[kEncodeThreads / 32];
    const int t = static_cast<int>(blockIdx.x), b = static_cast<int>(threadIdx.x);
    std::uint16_t v[16] = {};
    if (b < D::kBlocks) {
        const auto* source = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * D::kK + 16 * b);
        wide::unpack16(source[0], source[1], v);
    }
    const unsigned key = __reduce_max_sync(0xFFFFFFFFU, wide::a16_key(v));
    if (b % 32 == 0) { keys[b / 32] = key; }
    __syncthreads();
    unsigned all = 0;
#pragma unroll
    for (int w = 0; w < kEncodeThreads / 32; ++w) { all = max(all, keys[w]); }
    const int e = wide::a16_emax(all);
    if (b < D::kPadBlocks) { wide::store_a16_block(limbs + static_cast<std::size_t>(t) * D::kRowBytes, b, v, e); }
    if (b == 0) { emax[t] = e; }
}

// The GEMV of Columns-wide calls (n <= Columns columns). Lane l = 16 half + r holds row r of the
// warp's row group at k = 8 half .. 8 half + 7 of each block; the halves' products are joined
// before the scale, as the narrow expert route does.
template <class D, int Columns, bool F32>
__global__ void __launch_bounds__(kThreads) narrow_kernel(const Act* __restrict__ acts,
                                                          const std::int32_t* __restrict__ emax,
                                                          const std::uint8_t* __restrict__ matrix, float multiplier,
                                                          int n, void* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* shared = reinterpret_cast<Act*>(smem_raw);
    for (int i = static_cast<int>(threadIdx.x); i < n * D::kBlocks; i += kThreads) { shared[i] = acts[i]; }
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int r = lane & 15, half = lane >> 4;
    const int shift = r < 8 ? 0 : 4;
    const int code_offset  = 64 * half + 4 * (r & 7);
    const int scale_offset = 128 + 4 * (r >> 2), scale_shift = 8 * (r & 3);
    int e[Columns];
#pragma unroll
    for (int c = 0; c < Columns; ++c) { e[c] = c < n ? emax[c] : 0; }
    for (int g = static_cast<int>(blockIdx.x) * kWarps + warp; g < D::kGroups; g += static_cast<int>(gridDim.x) * kWarps) {
        const std::uint8_t* units = matrix + static_cast<std::size_t>(g) * D::kBlocks * moe::kUnitBytes;
        long long s[Columns];
#pragma unroll
        for (int c = 0; c < Columns; ++c) { s[c] = 0; }
#pragma unroll 4
        for (int b = 0; b < D::kBlocks; ++b) {
            const std::uint8_t* unit = units + b * static_cast<int>(moe::kUnitBytes);
            const unsigned c0 = __ldcs(reinterpret_cast<const unsigned*>(unit + code_offset));
            const unsigned c1 = __ldcs(reinterpret_cast<const unsigned*>(unit + code_offset + 32));
            const unsigned sc = __ldcs(reinterpret_cast<const unsigned*>(unit + scale_offset));
            accumulate_block<Columns>(shared, D::kBlocks, b, half, n, c0, c1, sc, shift, scale_shift, s);
        }
        if (lane < 16) { store_rows<D, Columns, F32>(s, e, g, r, n, multiplier, out); }
    }
}

// The streamed GEMV: one CTA of kStreamWarps warps per SM, so the encoded
// columns are copied once per SM, and kStreamDepth units in flight per warp (three words a lane per
// unit, at most 64 registers a thread). A depth divides 24, so whole depths make every row group's
// blocks (kStreamed: matrices of whole 24-block multiples, the head's 192 among them).
constexpr int kStreamWarps   = 32;
constexpr int kStreamThreads = kStreamWarps * 32;

// 12 / 8 / 6 spilled 40-48 / 16 / 8 bytes under the 64-register cap (build of 2026-10-23); 8 / 6 / 4
// keep 32 warps x 8 x 144 = 36 KiB in flight per SM at one or two columns.
template <int Columns>
constexpr int kStreamDepth = Columns <= 2 ? 8 : Columns <= 4 ? 6 : 4;

template <class D, int Columns>
constexpr int kStreamBytes = static_cast<int>(sizeof(Act)) * D::kBlocks * Columns +
                             static_cast<int>(sizeof(std::int32_t)) * Columns +
                             static_cast<int>(sizeof(unsigned)) * kStreamWarps;

template <class D, int Columns>
constexpr bool kStreamed = D::kBlocks % 24 == 0 && kStreamBytes<D, Columns> <= kNarrowBytes;

template <class D, int Columns, bool F32>
__global__ void __launch_bounds__(kStreamThreads, 1)
    stream_kernel(const std::uint16_t* __restrict__ x, const std::uint8_t* __restrict__ matrix, float multiplier, int n,
                  void* __restrict__ out) {
    constexpr int kDepth = kStreamDepth<Columns>;
    static_assert(D::kBlocks % kDepth == 0);
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* shared = reinterpret_cast<Act*>(smem_raw);
    auto* emax   = reinterpret_cast<std::int32_t*>(smem_raw + sizeof(Act) * D::kBlocks * Columns);
    auto* keys   = reinterpret_cast<unsigned*>(emax + Columns);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int r = lane & 15, half = lane >> 4;
    const int shift = r < 8 ? 0 : 4;
    const int code_offset  = 64 * half + 4 * (r & 7);
    const int scale_offset = 128 + 4 * (r >> 2), scale_shift = 8 * (r & 3);

    // The warp's row groups first, first + stride, ...: block L of its sequence is block L % kBlocks
    // of its (L / kBlocks)th group; `next` walks the sequence kDepth blocks ahead of the products.
    const int stride = static_cast<int>(gridDim.x) * kStreamWarps;
    const int first  = static_cast<int>(blockIdx.x) * kStreamWarps + warp;
    const int total  = first < D::kGroups ? ((D::kGroups - 1 - first) / stride + 1) * D::kBlocks : 0;
    const std::uint8_t* next = matrix + static_cast<std::size_t>(first) * D::kBlocks * moe::kUnitBytes;
    int next_group = first, next_block = 0;
    unsigned c0[kDepth], c1[kDepth], sc[kDepth];
    // The immutable weights only, before the PDL wait.
#pragma unroll
    for (int i = 0; i < kDepth; ++i) {
        if (i < total) {
            c0[i] = __ldcs(reinterpret_cast<const unsigned*>(next + code_offset));
            c1[i] = __ldcs(reinterpret_cast<const unsigned*>(next + code_offset + 32));
            sc[i] = __ldcs(reinterpret_cast<const unsigned*>(next + scale_offset));
            next += moe::kUnitBytes;
            if (++next_block == D::kBlocks) {
                next_block = 0;
                next_group += stride;
                next = matrix + static_cast<std::size_t>(next_group) * D::kBlocks * moe::kUnitBytes;
            }
        }
    }
    pdl::enter_streaming();

    // The call's columns into shared memory, kPerPass at a time: a column's blocks on whole warps.
    constexpr int kWidth   = round_up32(D::kBlocks);
    constexpr int kPerPass = kStreamThreads / kWidth;
    static_assert(kPerPass >= 1);
    for (int c0_pass = 0; c0_pass < n; c0_pass += kPerPass) {
        const int slot = static_cast<int>(threadIdx.x) / kWidth, b = static_cast<int>(threadIdx.x) % kWidth;
        const int c    = c0_pass + slot;
        const bool block = slot < kPerPass && c < n && b < D::kBlocks;
        unsigned words[8] = {};
        if (block) { load_block<D::kK>(x + static_cast<std::size_t>(c) * D::kK, b, words); }
        const unsigned key = __reduce_max_sync(0xFFFFFFFFU, block_key(words));
        if (lane == 0) { keys[warp] = key; }
        __syncthreads();
        if (block) {
            unsigned all = 0;
#pragma unroll
            for (int w = 0; w < kWidth / 32; ++w) { all = max(all, keys[slot * (kWidth / 32) + w]); }
            const int e = column_exponent(all);
            shared[c * D::kBlocks + b] = encode_block(words, e);
            if (b == 0) { emax[c] = e; }
        }
        __syncthreads();
    }

    long long s[Columns];
#pragma unroll
    for (int c = 0; c < Columns; ++c) { s[c] = 0; }
    int group = first, block = 0;
    for (int at = 0; at < total; at += kDepth) {
#pragma unroll
        for (int i = 0; i < kDepth; ++i) {
            const unsigned w0 = c0[i], w1 = c1[i], ws = sc[i];
            if (at + i + kDepth < total) {
                c0[i] = __ldcs(reinterpret_cast<const unsigned*>(next + code_offset));
                c1[i] = __ldcs(reinterpret_cast<const unsigned*>(next + code_offset + 32));
                sc[i] = __ldcs(reinterpret_cast<const unsigned*>(next + scale_offset));
                next += moe::kUnitBytes;
                if (++next_block == D::kBlocks) {
                    next_block = 0;
                    next_group += stride;
                    next = matrix + static_cast<std::size_t>(next_group) * D::kBlocks * moe::kUnitBytes;
                }
            }
            accumulate_block<Columns>(shared, D::kBlocks, block + i, half, n, w0, w1, ws, shift, scale_shift, s);
        }
        block += kDepth;
        if (block == D::kBlocks) {
            if (lane < 16) { store_rows<D, Columns, F32>(s, emax, group, r, n, multiplier, out); }
#pragma unroll
            for (int c = 0; c < Columns; ++c) { s[c] = 0; }
            block = 0;
            group += stride;
        }
    }
    pdl::trigger_dependents();
}

// The wide route's matrix: the projection as one job's gate/up-like matrix, its rows by token.
template <class D>
struct DenseMatrix {
    static constexpr int kRowGroups      = D::kGroups;
    static constexpr int kBlocks         = D::kBlocks;
    static constexpr std::size_t kOffset = 0;
    static constexpr int kRowBytes       = D::kRowBytes;
    static constexpr bool kTokenRows     = true;
};

// The canonical output of each tile column into its output column (rows 2i, 2i + 1 in one thread);
// the row groups past the matrix in a partial last row block are skipped.
template <class D, bool F32>
struct DenseEpilogue {
    const std::int32_t* x_emax;
    void* out;
    float multiplier;

    __device__ __forceinline__ void operator()(wide::SumsA16& sums, const wide::TileContext& ctx) const {
        using S = wide::ScheduleA16;
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * S::kWarpTokens + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                const int t    = ctx.first + column; // entries are the identity
                const int emax = x_emax[t];
                const std::size_t base = static_cast<std::size_t>(t) * D::kRows + static_cast<std::size_t>(ctx.row_block) * S::kBlockRows;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    if constexpr (D::kGroups % S::kRowGroups != 0) {
                        if (ctx.row_block * S::kRowGroups + S::kGroupsPerWarp * ctx.warp_n + (r >> 1) >= D::kGroups) { continue; }
                    }
                    const std::size_t row = base + wide::a16_tile_row(ctx.warp_n, r, ctx.tig);
                    if constexpr (F32) {
                        *reinterpret_cast<float2*>(static_cast<float*>(out) + row) =
                            make_float2(canon::a16_row_output_f32(sums[m][r][2 * half], emax, multiplier),
                                        canon::a16_row_output_f32(sums[m][r][2 * half + 1], emax, multiplier));
                    } else {
                        const std::uint32_t y0 = canon::a16_row_output(sums[m][r][2 * half], emax, multiplier);
                        const std::uint32_t y1 = canon::a16_row_output(sums[m][r][2 * half + 1], emax, multiplier);
                        *reinterpret_cast<std::uint32_t*>(static_cast<std::uint16_t*>(out) + row) = y0 | (y1 << 16);
                    }
                }
            }
        }
    }
};

// The wide route's bookkeeping in the workspace: the matrix as the one job of a dispatch whose
// entries are a pass's columns, and its 64-column tiles in one pass.
struct WidePlan {
    std::int32_t* jobs;      // [1]
    std::int32_t* job_count; // [1]
    std::int32_t* offsets;   // [2]
    std::int32_t* entries;   // [columns]
    std::int32_t* tiles;     // [2 tiles]
    std::int32_t* pass_tiles; // [2]
    const std::uint8_t** job_records; // [1]
    std::uint8_t* x_limbs;   // [columns][D::kRowBytes]
    std::int32_t* x_emax;    // [columns]
};

__global__ void plan_kernel(WidePlan plan, const std::uint8_t* matrix, int columns, int tiles) {
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < columns;
         i += static_cast<int>(gridDim.x * blockDim.x)) {
        plan.entries[i] = i;
        if (i < tiles) {
            plan.tiles[2 * i]     = 0;
            plan.tiles[2 * i + 1] = i * wide::kTileColumns;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        plan.jobs[0]        = 0;
        *plan.job_count     = 1;
        plan.offsets[0]     = 0;
        plan.offsets[1]     = columns;
        plan.pass_tiles[0]  = 0;
        plan.pass_tiles[1]  = tiles;
        plan.job_records[0] = matrix;
    }
}

std::size_t round256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

struct NarrowPlan {
    Act* acts;
    std::int32_t* emax;
};

template <class D, class Arena>
NarrowPlan carve_narrow(Arena& arena, std::int32_t columns) {
    return {static_cast<Act*>(arena.alloc_bytes(round256(sizeof(Act) * D::kBlocks * columns), 256).data),
            static_cast<std::int32_t*>(arena.alloc_bytes(round256(sizeof(std::int32_t) * columns), 256).data)};
}

// `columns`: the widest pass (at most kWideColumns).
template <class D, class Arena>
WidePlan carve_wide(Arena& arena, std::int32_t columns) {
    const auto t     = static_cast<std::size_t>(columns);
    const auto tiles = static_cast<std::size_t>((columns + wide::kTileColumns - 1) / wide::kTileColumns);
    WidePlan plan{};
    const auto i32 = [&](std::size_t count) {
        return static_cast<std::int32_t*>(arena.alloc_bytes(round256(count * sizeof(std::int32_t)), 256).data);
    };
    plan.jobs        = i32(1);
    plan.job_count   = i32(1);
    plan.offsets     = i32(2);
    plan.entries     = i32(t);
    plan.tiles       = i32(2 * tiles);
    plan.pass_tiles  = i32(2);
    plan.job_records = static_cast<const std::uint8_t**>(arena.alloc_bytes(256, 256).data);
    plan.x_limbs     = static_cast<std::uint8_t*>(arena.alloc_bytes(round256(t * D::kRowBytes), 256).data);
    plan.x_emax      = i32(t);
    return plan;
}

template <class D>
void require_operands(const Tensor& x, const Weight& weight, const Tensor& out) {
    require(weight.qtype == QType::NVFP4_MUL && weight.layout == QuantLayout::Rg16 && weight.qdata != nullptr &&
                reinterpret_cast<std::uintptr_t>(weight.qdata) % 16 == 0 && std::isfinite(weight.weight_multiplier),
            "weight must be an nvfp4_rg16_v1 matrix with 16-byte aligned units and a finite multiplier");
    require(x.dtype == DType::BF16 && x.is_contiguous() && x.ne[0] == D::kK && x.ne[1] > 0 && x.ne[2] == 1 &&
                x.ne[3] == 1 && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0,
            "x must be contiguous, 16-byte aligned BF16 [K, T]");
    require((out.dtype == DType::FP32 || out.dtype == DType::BF16) && out.is_contiguous() && out.ne[0] == D::kRows &&
                out.ne[1] == x.ne[1] && out.ne[2] == 1 && out.ne[3] == 1 &&
                reinterpret_cast<std::uintptr_t>(out.data) % 16 == 0,
            "out must be contiguous FP32 or BF16 [N, T]");
}

template <class D, int Columns, bool F32>
void launch_narrow(const NarrowPlan& plan, const Weight& weight, int n, void* out, cudaStream_t stream) {
    constexpr int bytes = static_cast<int>(sizeof(Act)) * D::kBlocks * Columns;
    static_assert(bytes <= kNarrowBytes);
    static const int ctas = [] {
        CUDA_CHECK(cudaFuncSetAttribute(narrow_kernel<D, Columns, F32>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int device = 0, sms = 0, per_sm = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, narrow_kernel<D, Columns, F32>, kThreads, bytes));
        // A persistent grid: every resident CTA copies the encoded columns once.
        return std::min((D::kGroups + kWarps - 1) / kWarps, sms * std::max(per_sm, 1));
    }();
    narrow_kernel<D, Columns, F32><<<ctas, kThreads, bytes, stream>>>(plan.acts, plan.emax,
                                                                     static_cast<const std::uint8_t*>(weight.qdata),
                                                                     weight.weight_multiplier, n, out);
    check_launch("narrow");
}

template <class D, int Columns, bool F32>
void launch_stream(const Tensor& x, const Weight& weight, int n, void* out, cudaStream_t stream) {
    constexpr int bytes = kStreamBytes<D, Columns>;
    static const int ctas = [] {
        CUDA_CHECK(cudaFuncSetAttribute(stream_kernel<D, Columns, F32>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int device = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return std::min((D::kGroups + kStreamWarps - 1) / kStreamWarps, sms);
    }();
    CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(ctas)), dim3(kStreamThreads), bytes, stream},
                                    stream_kernel<D, Columns, F32>, static_cast<const std::uint16_t*>(x.data),
                                    static_cast<const std::uint8_t*>(weight.qdata), weight.weight_multiplier, n, out));
}

// The streamed GEMV at the narrow kernel's column templates where the matrix has it; false where
// it does not (the vision fc2's 269 blocks a row).
template <class D, bool F32>
bool dispatch_stream(const Tensor& x, const Weight& weight, int n, void* out, cudaStream_t stream) {
    const auto at = [&]<int Columns>() {
        if constexpr (kStreamed<D, Columns>) {
            launch_stream<D, Columns, F32>(x, weight, n, out, stream);
            return true;
        }
        return false;
    };
    if (n == 1) { return at.template operator()<1>(); }
    if (n == 2) { return at.template operator()<2>(); }
    if constexpr (D::kNarrow > 4) {
        if (n > 4) { return at.template operator()<D::kNarrow>(); }
    }
    return at.template operator()<4>();
}

template <class D, bool F32>
void dispatch_narrow(const NarrowPlan& plan, const Weight& weight, int n, void* out, cudaStream_t stream) {
    if (n == 1) { return launch_narrow<D, 1, F32>(plan, weight, n, out, stream); }
    if (n == 2) { return launch_narrow<D, 2, F32>(plan, weight, n, out, stream); }
    if constexpr (D::kNarrow > 4) {
        if (n > 4) { return launch_narrow<D, D::kNarrow, F32>(plan, weight, n, out, stream); }
    }
    launch_narrow<D, 4, F32>(plan, weight, n, out, stream);
}

// One wide pass over `columns` columns of x into the same columns of out.
template <class D, bool F32>
void wide_pass(const WidePlan& plan, const Weight& weight, const std::uint16_t* x, int columns, void* out,
               cudaStream_t stream) {
    const int tiles = (columns + wide::kTileColumns - 1) / wide::kTileColumns;
    plan_kernel<<<(columns + 255) / 256, 256, 0, stream>>>(plan, static_cast<const std::uint8_t*>(weight.qdata), columns,
                                                         tiles);
    check_launch("plan");
    limbs_kernel<D><<<columns, round_up32(D::kPadBlocks), 0, stream>>>(x, plan.x_limbs, plan.x_emax);
    check_launch("limbs");
    wide::Call call;
    call.dispatch.jobs      = plan.jobs;
    call.dispatch.job_count = plan.job_count;
    call.dispatch.offsets   = plan.offsets;
    call.dispatch.entries   = plan.entries;
    call.job_records        = plan.job_records;
    call.tiles              = plan.tiles;
    call.pass_tiles         = plan.pass_tiles;
    call.passes             = 1;
    call.a16                = true;
    call.x_limbs            = plan.x_limbs;
    call.x_emax             = plan.x_emax;
    call.top_k              = 1;
    wide::launch_a16<DenseMatrix<D>>(call, 0, DenseEpilogue<D, F32>{plan.x_emax, out, weight.weight_multiplier}, stream);
}

template <class D>
void run(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    require_operands<D>(x, weight, out);
    const int columns = x.ne[1];
    const bool f32    = out.dtype == DType::FP32;
    auto scope        = workspace.scope();
    if (columns <= D::kNarrow) {
        if (f32 ? dispatch_stream<D, true>(x, weight, columns, out.data, stream)
                : dispatch_stream<D, false>(x, weight, columns, out.data, stream)) {
            return;
        }
        const NarrowPlan plan = carve_narrow<D>(workspace, columns);
        encode_kernel<D><<<columns, round_up32(D::kBlocks), 0, stream>>>(static_cast<const std::uint16_t*>(x.data),
                                                                          plan.acts, plan.emax);
        check_launch("encode");
        if (f32) {
            dispatch_narrow<D, true>(plan, weight, columns, out.data, stream);
        } else {
            dispatch_narrow<D, false>(plan, weight, columns, out.data, stream);
        }
        return;
    }
    // Passes of at most kWideColumns columns, equal in whole tiles (4,097 columns: 2,112 and 1,985,
    // not 4,096 and a one-column pass), share one plan and limb buffer, reused in stream order.
    const int passes    = (columns + kWideColumns - 1) / kWideColumns;
    const int per_pass  = (columns + passes - 1) / passes;
    const int width     = std::min(columns, (per_pass + wide::kTileColumns - 1) / wide::kTileColumns * wide::kTileColumns);
    const WidePlan plan = carve_wide<D>(workspace, std::min(columns, kWideColumns));
    const auto* source  = static_cast<const std::uint16_t*>(x.data);
    auto* target        = static_cast<std::byte*>(out.data);
    const std::size_t element_bytes = f32 ? sizeof(float) : sizeof(std::uint16_t);
    for (int first = 0; first < columns; first += width) {
        const int n          = std::min(width, columns - first);
        const auto* x_pass   = source + static_cast<std::size_t>(first) * D::kK;
        void* out_pass       = target + static_cast<std::size_t>(first) * D::kRows * element_bytes;
        if (f32) {
            wide_pass<D, true>(plan, weight, x_pass, n, out_pass, stream);
        } else {
            wide_pass<D, false>(plan, weight, x_pass, n, out_pass, stream);
        }
    }
}

} // namespace

std::size_t rg16_linear_workspace_bytes(std::int32_t rows, std::int32_t input_rows, std::int32_t columns) {
    require(columns > 0, "empty call");
    return with_dense(rows, input_rows, [&]<class D>() {
        WorkspaceLayoutBuilder layout;
        if (columns <= D::kNarrow) {
            (void)carve_narrow<D>(layout, columns);
        } else {
            (void)carve_wide<D>(layout, std::min(columns, kWideColumns));
        }
        return layout.peak_bytes(1);
    });
}

void rg16_linear(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    with_dense(weight.n, weight.k, [&]<class D>() { run<D>(x, weight, out, workspace, stream); });
}

} // namespace infernix::ops
