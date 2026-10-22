// The wide route of offloaded_sparse_moe: work planning, A4 quantization of x for the wide
// entries, TMA descriptors and the per-pass GEMM launches (wide_expert.h, wide_expert.cuh), and the
// W4A16 route's limb encodings and exact integer GEMMs (wide_expert_a16.cuh).

#include "ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace infernix::ops::offloaded_moe::wide {
namespace {

constexpr int kPlanThreads     = 1024;
constexpr int kQuantizeThreads = 256;

std::size_t round256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("offloaded_sparse_moe wide ") + what + ": " + cudaGetErrorString(error));
    }
}

// One CTA: the call's wide work tiles, (job, first column) for each 64-column slice of every wide
// job in job order, and for each pass of pass_jobs jobs the index of its first tile.
__global__ void __launch_bounds__(kPlanThreads)
    plan_kernel(MoeDispatch dispatch, const ExpertScales* __restrict__ scales, int max_jobs, int pass_jobs, int passes,
                std::int32_t* __restrict__ tiles, std::int32_t* __restrict__ pass_tiles) {
    __shared__ int scan[kPlanThreads];
    __shared__ int base;
    const int tid  = static_cast<int>(threadIdx.x);
    const int jobs = min(*dispatch.job_count, max_jobs);
    if (tid == 0) { base = 0; }
    __syncthreads();
    for (int chunk = 0; chunk < passes * pass_jobs; chunk += kPlanThreads) {
        const int job = chunk + tid;
        int n         = 0;
        if (job < jobs) {
            const int expert  = dispatch.jobs[job];
            const int columns = dispatch.offsets[expert + 1] - dispatch.offsets[expert];
            if (wide_route(columns, scales[expert])) { n = (columns + kTileColumns - 1) / kTileColumns; }
        }
        scan[tid] = n;
        __syncthreads();
        for (int step = 1; step < kPlanThreads; step <<= 1) {
            const int add = tid >= step ? scan[tid - step] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const int first = base + scan[tid] - n;
        for (int i = 0; i < n; ++i) {
            tiles[2 * (first + i)]     = job;
            tiles[2 * (first + i) + 1] = i * kTileColumns;
        }
        if (job % pass_jobs == 0 && job / pass_jobs < passes) { pass_tiles[job / pass_jobs] = first; }
        __syncthreads();
        if (tid == kPlanThreads - 1) { base = first + n; }
        __syncthreads();
    }
    if (tid == 0) { pass_tiles[passes] = base; }
}

// A4(x) of every wide entry with its expert's gate/up input scale, in the plane layout of
// wide_expert.h: row p (the entry's dispatch index) holds 1,280 code bytes then 160 scale bytes.
__global__ void __launch_bounds__(kQuantizeThreads)
    quantize_kernel(const std::uint16_t* __restrict__ x, int top_k, MoeDispatch dispatch,
                    const ExpertScales* __restrict__ scales, const std::int32_t* __restrict__ tiles,
                    const std::int32_t* __restrict__ pass_tiles, int passes, std::uint8_t* __restrict__ plane) {
    const int total = pass_tiles[passes];
    for (int tile = static_cast<int>(blockIdx.x); tile < total; tile += static_cast<int>(gridDim.x)) {
        const int job = tiles[2 * tile], column = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int first  = dispatch.offsets[expert] + column;
        const int count  = min(kTileColumns, dispatch.offsets[expert + 1] - first);
        const float scale = scales[expert].input_gate;
        for (int task = static_cast<int>(threadIdx.x); task < count * kGateUpBlocks; task += kQuantizeThreads) {
            const int i = task / kGateUpBlocks, b = task % kGateUpBlocks;
            const int p = first + i;
            const int t = dispatch.entries[p] / top_k;
            const auto* source = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * kHidden + 16 * b);
            const uint4 v0 = source[0], v1 = source[1];
            std::uint16_t v[16];
            const unsigned words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                v[2 * w]     = static_cast<std::uint16_t>(words[w] & 0xFFFFU);
                v[2 * w + 1] = static_cast<std::uint16_t>(words[w] >> 16);
            }
            const canon::A4Codes q = canon::quantize_a4_codes(v, scale);
            unsigned low = 0, high = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                low |= (static_cast<unsigned>(q.code[j]) | (static_cast<unsigned>(q.code[4 + j]) << 4)) << (8 * j);
                high |= (static_cast<unsigned>(q.code[8 + j]) | (static_cast<unsigned>(q.code[12 + j]) << 4)) << (8 * j);
            }
            std::uint8_t* row  = plane + static_cast<std::size_t>(p) * kXRowBytes;
            const int position = plane_block(b);
            *reinterpret_cast<uint2*>(row + 8 * position) = make_uint2(low, high);
            row[kXCodeBytes + position]                   = q.scale_word;
        }
    }
}

Descriptors make_descriptors(const std::uint8_t* x_plane, const std::uint8_t* h_plane, int entries) {
    const auto rows = static_cast<std::uint64_t>(entries);
    auto* x = const_cast<std::uint8_t*>(x_plane);
    auto* h = const_cast<std::uint8_t*>(h_plane);
    Descriptors d{};
    d.x_codes  = detail::nvfp4_make_tma_2d(x, CU_TENSOR_MAP_DATA_TYPE_UINT8, kXCodeBytes, rows, kXRowBytes,
                                           Schedule::kCodeRowBytes, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_64B,
                                           "encode wide x codes TMA");
    d.x_scales = detail::nvfp4_make_tma_2d(x + kXCodeBytes, CU_TENSOR_MAP_DATA_TYPE_UINT8, kXScaleBytes, rows,
                                           kXRowBytes, 16, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_NONE,
                                           "encode wide x scales TMA");
    d.h_codes  = detail::nvfp4_make_tma_2d(h, CU_TENSOR_MAP_DATA_TYPE_UINT8, kHCodeBytes, rows, kHRowStride,
                                           Schedule::kCodeRowBytes, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_64B,
                                           "encode wide h codes TMA");
    // 40 scale bytes per row: the last K tile's 16-byte box also reads the 8 bytes after them
    // (inside the row), which no MMA step uses.
    d.h_scales = detail::nvfp4_make_tma_2d(h + kHCodeBytes, CU_TENSOR_MAP_DATA_TYPE_UINT8, kHScaleMapBytes, rows,
                                           kHRowStride, 16, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_NONE,
                                           "encode wide h scales TMA");
    return d;
}

#ifdef _WIN32
// The single staging of the wide route's descriptors (core/tma_descriptor_staging.cuh); every wide
// launch runs on the compute stream that stages them.
TmaDescriptorStaging<Descriptors>& descriptor_staging() {
    static TmaDescriptorStaging<Descriptors> staging;
    return staging;
}
#else
// Kernel parameters are copied at launch, so one host copy serves every call in turn.
Descriptors& host_descriptors() {
    static Descriptors descriptors;
    return descriptors;
}
#endif

} // namespace

std::int32_t max_tiles(std::int32_t max_jobs, std::int32_t entries) {
    return max_jobs + (entries + kTileColumns - 1) / kTileColumns;
}

ResidualTerms residual_terms(WideA16 route) {
    if (route == WideA16::kA4Residual) { return {2, 2}; }
    if (route == WideA16::kA4Residual3) { return {3, 3}; }
    return {};
}

namespace {

// Byte offsets of the workspace's regions; `end` is its size.
struct Offsets {
    std::size_t job_records, cpu_flags, cpu_call, tiles, pass_tiles;
    std::size_t h_blocks, x_plane;                      // W4A4
    std::size_t x_limbs, h_rows, x_emax, h_emax;        // W4A16
    std::size_t x_terms_plane, x_row_scale, h_terms_plane, h_row_scale; // W4A16 two-term A4
    std::size_t ggml_planes;                            // GGUF banks' kGgmlBf16
    bool has_ggml_planes;
    std::size_t lossless;                               // kLossless
    std::size_t x_acts, h_acts, ring_jobs, ring_counters; // the narrow ring (ring_call)
    bool has_ring;
    std::size_t end;
};

Offsets offsets(const ExpertShape& shape, std::int32_t max_jobs, std::int32_t entries, std::int32_t columns,
                ExpertActivation activation, WideA16 wide_a16) {
    if (max_jobs <= 0 || entries <= 0 || columns <= 0) {
        throw std::invalid_argument("offloaded_sparse_moe experts workspace: empty call");
    }
    // The bank kind (ggml or NVFP4) is checked by moe_experts; both size the same A16 planes.
    if (!shape_supported(shape, activation, wide_a16, false)) {
        throw std::invalid_argument("offloaded_sparse_moe experts workspace: unsupported expert geometry");
    }
    const auto x_row = static_cast<std::size_t>(a16_x_row_bytes(shape));
    const auto h_row = static_cast<std::size_t>(a16_h_row_bytes(shape));
    const auto e = static_cast<std::size_t>(entries), j = static_cast<std::size_t>(max_jobs);
    const auto t = static_cast<std::size_t>(columns);
    Offsets o{};
    std::size_t p = 0;
    const auto take = [&p](std::size_t bytes) {
        const std::size_t at = p;
        p += round256(bytes);
        return at;
    };
    o.job_records = take(j * sizeof(void*));
    o.cpu_flags   = take(j * sizeof(std::int32_t));
    o.cpu_call    = take(kCpuCallBytes);
    o.tiles       = take(static_cast<std::size_t>(max_tiles(max_jobs, entries)) * 2 * sizeof(std::int32_t));
    o.pass_tiles  = take((j + 2) * sizeof(std::int32_t));
    const ResidualTerms terms = activation == ExpertActivation::kA16 ? residual_terms(wide_a16) : ResidualTerms{};
    if (wide_a16 == WideA16::kLossless) {
        o.lossless = take(lossless_experts_bytes(shape, max_jobs, entries));
    } else if (terms.x != 0) {
        // The narrow route still needs its h rows (BF16 h of every entry) and exponents.
        o.h_rows      = take(e * h_row);
        o.h_emax      = take(e * sizeof(std::int32_t));
        o.x_terms_plane = take(e * x_terms_row_bytes(terms.x));
        o.x_row_scale = take(t * sizeof(float));
        o.h_terms_plane = take(e * h_terms_row_bytes(terms.h));
        o.h_row_scale = take(e * sizeof(float));
    } else if (activation == ExpertActivation::kA16) {
        o.x_limbs = take(t * x_row);
        o.h_rows  = take(e * h_row);
        o.x_emax  = take(t * sizeof(std::int32_t));
        o.h_emax  = take(e * sizeof(std::int32_t));
        if (wide_a16 == WideA16::kGgmlBf16 && columns >= kGgmlBf16MinColumns) {
            o.ggml_planes     = take(kGgmlBf16PlaneBytes);
            o.has_ggml_planes = true;
        }
    } else {
        o.h_blocks = take(e * kHBlocks * sizeof(canon::A4Block));
        o.x_plane  = take(e * kXRowBytes);
    }
    if (ring_call(shape, activation, wide_a16, columns)) {
        o.x_acts        = take(t * static_cast<std::size_t>(shape.gate_up_blocks()) * kA16ActBytes);
        o.h_acts        = take(e * static_cast<std::size_t>(shape.h_blocks()) * kA16ActBytes);
        o.ring_jobs     = take(2 * j * sizeof(std::uint32_t));
        o.ring_counters = take(2 * kRingCounterSlots * sizeof(std::uint32_t));
        o.has_ring      = true;
    }
    o.end = p;
    return o;
}

// One 16-element block b of a row into its limb-plane row (wide_expert_a16.cuh): the canonical X
// (canon::a16_value; zero in a non-finite column) as H1 = X >> 14, L1 = (X >> 3) & 255 and
// H0 | L0 << 4 with H0 = (X >> 11) & 7, L0 = X & 7, in K tile b / 4 at block offset 16 (b % 4).
template <class Byte>
__device__ __forceinline__ uint4 pack16(const Byte (&b)[16]) {
    unsigned w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i])) |
               (static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i + 1])) << 8) |
               (static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i + 2])) << 16) |
               (static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i + 3])) << 24);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ void store_a16_block(std::uint8_t* row, int b, const std::uint16_t (&v)[16], int emax) {
    std::uint8_t high[16], low[16], small[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const std::int32_t x = emax == canon::kA16NonFinite ? 0 : canon::a16_value(v[j], emax);
        high[j]  = static_cast<std::uint8_t>(x >> 14);
        low[j]   = static_cast<std::uint8_t>((x >> 3) & 255);
        small[j] = static_cast<std::uint8_t>(((x >> 11) & 7) | ((x & 7) << 4));
    }
    std::uint8_t* at                              = row + (b / 4) * kA16LimbTileBytes + 16 * (b % 4);
    *reinterpret_cast<uint4*>(at)                 = pack16(high);
    *reinterpret_cast<uint4*>(at + kA16KTile)     = pack16(low);
    *reinterpret_cast<uint4*>(at + 2 * kA16KTile) = pack16(small);
}

__device__ __forceinline__ void unpack16(const uint4 v0, const uint4 v1, std::uint16_t (&v)[16]) {
    const unsigned words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        v[2 * w]     = static_cast<std::uint16_t>(words[w] & 0xFFFFU);
        v[2 * w + 1] = static_cast<std::uint16_t>(words[w] >> 16);
    }
}

// The largest magnitude's bits of 16 BF16 values (their column-exponent key, canon::a16_column_exponent).
__device__ __forceinline__ unsigned a16_key(const std::uint16_t (&v)[16]) {
    unsigned key = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) { key = max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    return key;
}

__device__ __forceinline__ int a16_emax(unsigned key) {
    return key >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(key >> 7), 1);
}

constexpr int kEncodeXThreads = kHidden / 16; // one 16-element block per thread (Flash-Next's A4 terms)
constexpr int kEncodeWarps    = 8;

// Every token's x limbs and column exponent (W4A16 wide calls): one CTA per token, one 16-element
// block per thread.
template <class G>
__global__ void __launch_bounds__(G::kHidden / 16)
    encode_x_kernel(const std::uint16_t* __restrict__ x, std::uint8_t* __restrict__ limbs, std::int32_t* __restrict__ emax) {
    constexpr int kThreads = G::kHidden / 16;
    static_assert(kThreads % 32 == 0);
    __shared__ unsigned keys[kThreads / 32];
    const int t = static_cast<int>(blockIdx.x), b = static_cast<int>(threadIdx.x);
    const auto* source = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * G::kHidden + 16 * b);
    std::uint16_t v[16];
    unpack16(source[0], source[1], v);
    const unsigned key = __reduce_max_sync(0xFFFFFFFFU, a16_key(v));
    if (b % 32 == 0) { keys[b / 32] = key; }
    __syncthreads();
    unsigned all = 0;
#pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) { all = max(all, keys[w]); }
    const int e = a16_emax(all);
    store_a16_block(limbs + static_cast<std::size_t>(t) * A16Planes<G>::kXRowBytes, b, v, e);
    if (b == 0) { emax[t] = e; }
}

// The pass's wide entries' h rows rewritten in place from BF16 to limbs, with their exponents: one
// warp per entry, which holds the whole BF16 row in registers before it writes.
template <class G>
__global__ void __launch_bounds__(kEncodeWarps * 32)
    encode_h_kernel(MoeDispatch dispatch, const std::int32_t* __restrict__ tiles, const std::int32_t* __restrict__ pass_tiles,
                    int pass, std::uint8_t* __restrict__ h_rows, std::int32_t* __restrict__ h_emax) {
    constexpr int kBlocks = G::kIntermediate / 16; // 40 or 64: lane l holds blocks l and l + 32
    static_assert(kBlocks > 32 && kBlocks <= 64);
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp = static_cast<int>(blockIdx.x) * kEncodeWarps + static_cast<int>(threadIdx.x) / 32;
    const int warps = static_cast<int>(gridDim.x) * kEncodeWarps;
    const int first_tile = pass_tiles[pass], tile_count = pass_tiles[pass + 1] - first_tile;
    for (int task = warp; task < tile_count * kTileColumns; task += warps) {
        const int tile = first_tile + task / kTileColumns, c = task % kTileColumns;
        const int job = tiles[2 * tile], column = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int entry  = dispatch.offsets[expert] + column + c;
        if (entry >= dispatch.offsets[expert + 1]) { continue; } // warp-uniform
        std::uint8_t* row = h_rows + static_cast<std::size_t>(entry) * A16Planes<G>::kHRowBytes;
        const auto* source = reinterpret_cast<const uint4*>(row);
        std::uint16_t v0[16] = {}, v1[16] = {};
        unpack16(source[2 * lane], source[2 * lane + 1], v0);
        const bool second = lane + 32 < kBlocks;
        if (second) { unpack16(source[2 * (lane + 32)], source[2 * (lane + 32) + 1], v1); }
        const int e = a16_emax(__reduce_max_sync(0xFFFFFFFFU, max(a16_key(v0), a16_key(v1))));
        __syncwarp();
        store_a16_block(row, lane, v0, e);
        if (second) { store_a16_block(row, lane + 32, v1, e); }
        if (lane == 0) { h_emax[entry] = e; }
    }
}

// ---------------------------------------------------------- W4A16 two-term A4 (design §19.3.23)

// A row scale: amax / (6 * 448), so every block's first-term E4M3 scale stays finite. An all-zero row
// has scale 0 and zero terms; a non-finite row has scale NaN and zero terms, so the epilogues'
// products are NaN, as on the exact route.
__device__ __forceinline__ float residual_row_scale(float amax) {
    if (!isfinite(amax)) { return __int_as_float(0x7FC00000); }
    return amax > 0.0F ? amax / (6.0F * 448.0F) : 0.0F;
}

// The NVFP4 terms of one 16-element block of a BF16 row in the A4 plane layout: the A4 residual
// profile (ops/linear/nvfp4/nvfp4_codec.cuh, quantize_nvfp4_terms_k16) of the row divided by its row
// scale, with the codes re-paired from that codec's element order (byte i: elements 2i, 2i + 1) to
// the plane's (byte i: elements 8 (i / 4) + i % 4 and + 4, quantize_kernel's order).
template <int Terms>
struct TermBlock {
    unsigned low[Terms], high[Terms];
    std::uint8_t scale[Terms];
};

__device__ __forceinline__ void plane_order(std::uint32_t lo, std::uint32_t hi, unsigned& low, unsigned& high) {
    unsigned nibble[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) { nibble[k] = ((k < 8 ? lo : hi) >> (4 * (k % 8))) & 0xFU; }
    low = high = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        low |= (nibble[j] | (nibble[4 + j] << 4)) << (8 * j);
        high |= (nibble[8 + j] | (nibble[12 + j] << 4)) << (8 * j);
    }
}

// `block`: 16 BF16 values (16-byte aligned); `row_scale`: their row's (0 or NaN: zero terms).
template <int Terms>
__device__ __forceinline__ void term_block(const std::uint16_t* block, float row_scale, TermBlock<Terms>& out) {
    const float divisor = row_scale > 0.0F ? 1.0F / row_scale : 0.0F;
    ops::detail::Nvfp4QuantizedK16 terms[Terms];
    ops::detail::quantize_nvfp4_terms_k16<Terms>(reinterpret_cast<const __nv_bfloat16*>(block), divisor, terms);
#pragma unroll
    for (int term = 0; term < Terms; ++term) {
        plane_order(terms[term].codes_lo, terms[term].codes_hi, out.low[term], out.high[term]);
        out.scale[term] = terms[term].scale;
    }
}

// Writes block b's terms into a plane row: term t's A4 row at row + t * term_stride.
template <int Terms>
__device__ __forceinline__ void store_terms(std::uint8_t* row, int b, int code_bytes, int term_stride,
                                            const TermBlock<Terms>& q) {
    const int position = plane_block(b);
#pragma unroll
    for (int term = 0; term < Terms; ++term) {
        std::uint8_t* at = row + term * term_stride;
        *reinterpret_cast<uint2*>(at + 8 * position) = make_uint2(q.low[term], q.high[term]);
        at[code_bytes + position]                    = q.scale[term];
    }
}

// Every token's x row scale: one CTA per token.
__global__ void __launch_bounds__(kEncodeXThreads)
    x_row_scale_kernel(const std::uint16_t* __restrict__ x, float* __restrict__ row_scale) {
    __shared__ float maxima[kEncodeXThreads / 32];
    const int t = static_cast<int>(blockIdx.x), b = static_cast<int>(threadIdx.x);
    const auto* source = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * kHidden + 16 * b);
    std::uint16_t v[16];
    unpack16(source[0], source[1], v);
    float amax = 0.0F;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float a = fabsf(canon::bf16_to_f32(v[j]));
        amax          = a > amax || a != a ? a : amax; // NaN propagates (the row then has no terms)
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFFU, amax, offset);
        amax              = other > amax || other != other ? other : amax;
    }
    if (b % 32 == 0) { maxima[b / 32] = amax; }
    __syncthreads();
    if (b == 0) {
        float all = 0.0F;
#pragma unroll
        for (int w = 0; w < kEncodeXThreads / 32; ++w) { all = maxima[w] > all || maxima[w] != maxima[w] ? maxima[w] : all; }
        row_scale[t] = residual_row_scale(all);
    }
}

// The terms of every wide entry's x (its token's row), as quantize_kernel does for A4.
template <int Terms>
__global__ void __launch_bounds__(kQuantizeThreads)
    quantize_x_terms_kernel(const std::uint16_t* __restrict__ x, int top_k, MoeDispatch dispatch,
                       const std::int32_t* __restrict__ tiles, const std::int32_t* __restrict__ pass_tiles, int passes,
                       const float* __restrict__ row_scale, std::uint8_t* __restrict__ plane) {
    const int total = pass_tiles[passes];
    for (int tile = static_cast<int>(blockIdx.x); tile < total; tile += static_cast<int>(gridDim.x)) {
        const int job = tiles[2 * tile], column = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int first  = dispatch.offsets[expert] + column;
        const int count  = min(kTileColumns, dispatch.offsets[expert + 1] - first);
        for (int task = static_cast<int>(threadIdx.x); task < count * kGateUpBlocks; task += kQuantizeThreads) {
            const int i = task / kGateUpBlocks, b = task % kGateUpBlocks;
            const int p = first + i;
            const int t = dispatch.entries[p] / top_k;
            TermBlock<Terms> q;
            term_block(x + static_cast<std::size_t>(t) * kHidden + 16 * b, row_scale[t], q);
            store_terms(plane + static_cast<std::size_t>(p) * x_terms_row_bytes(Terms), b, kXCodeBytes, kXRowBytes, q);
        }
    }
}

// The pass's wide entries' h (BF16 rows) as terms with their row scales: one warp per entry.
template <int Terms>
__global__ void __launch_bounds__(kEncodeWarps * 32)
    quantize_h_terms_kernel(MoeDispatch dispatch, const std::int32_t* __restrict__ tiles,
                       const std::int32_t* __restrict__ pass_tiles, int pass, const std::uint8_t* __restrict__ h_rows,
                       std::uint8_t* __restrict__ plane, float* __restrict__ row_scale) {
    constexpr int kBlocks = kIntermediate / 16; // 40: lane l holds blocks l and l + 32
    const int lane  = static_cast<int>(threadIdx.x) % 32;
    const int warp  = static_cast<int>(blockIdx.x) * kEncodeWarps + static_cast<int>(threadIdx.x) / 32;
    const int warps = static_cast<int>(gridDim.x) * kEncodeWarps;
    const int first_tile = pass_tiles[pass], tile_count = pass_tiles[pass + 1] - first_tile;
    for (int task = warp; task < tile_count * kTileColumns; task += warps) {
        const int tile = first_tile + task / kTileColumns, c = task % kTileColumns;
        const int job = tiles[2 * tile], column = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int entry  = dispatch.offsets[expert] + column + c;
        if (entry >= dispatch.offsets[expert + 1]) { continue; } // warp-uniform
        const auto* source = reinterpret_cast<const uint4*>(h_rows + static_cast<std::size_t>(entry) * kA16HRowBytes);
        const bool second  = lane + 32 < kBlocks;
        std::uint16_t v0[16] = {}, v1[16] = {};
        unpack16(source[2 * lane], source[2 * lane + 1], v0);
        if (second) { unpack16(source[2 * (lane + 32)], source[2 * (lane + 32) + 1], v1); }
        float amax = 0.0F;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float a = fabsf(canon::bf16_to_f32(v0[j])), b = fabsf(canon::bf16_to_f32(v1[j]));
            amax = a > amax || a != a ? a : amax;
            amax = b > amax || b != b ? b : amax;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFFU, amax, offset);
            amax              = other > amax || other != other ? other : amax;
        }
        const float rs    = residual_row_scale(amax);
        const auto* h     = reinterpret_cast<const std::uint16_t*>(h_rows + static_cast<std::size_t>(entry) * kA16HRowBytes);
        std::uint8_t* row = plane + static_cast<std::size_t>(entry) * h_terms_row_bytes(Terms);
        TermBlock<Terms> q;
        term_block(h + 16 * lane, rs, q);
        store_terms(row, lane, kHCodeBytes, kHRowStride, q);
        if (second) {
            term_block(h + 16 * (lane + 32), rs, q);
            store_terms(row, lane + 32, kHCodeBytes, kHRowStride, q);
        }
        if (lane == 0) { row_scale[entry] = rs; }
    }
}

// Every term's TMA maps (unused third-term maps, at two terms, are encoded but never read).
Descriptors make_descriptors_terms(const std::uint8_t* x_terms_plane, const std::uint8_t* h_terms_plane, int entries,
                                   ResidualTerms terms) {
    const auto rows = static_cast<std::uint64_t>(entries);
    auto* x = const_cast<std::uint8_t*>(x_terms_plane);
    auto* h = const_cast<std::uint8_t*>(h_terms_plane);
    const auto x_stride = static_cast<std::uint64_t>(x_terms_row_bytes(terms.x));
    const auto h_stride = static_cast<std::uint64_t>(h_terms_row_bytes(terms.h));
    Descriptors d{};
    const auto x_maps = [&](std::uint8_t* base, CUtensorMap& codes, CUtensorMap& scales) {
        codes  = detail::nvfp4_make_tma_2d(base, CU_TENSOR_MAP_DATA_TYPE_UINT8, kXCodeBytes, rows, x_stride,
                                           Schedule::kCodeRowBytes, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_64B,
                                           "encode wide A4 residual x codes TMA");
        scales = detail::nvfp4_make_tma_2d(base + kXCodeBytes, CU_TENSOR_MAP_DATA_TYPE_UINT8, kXScaleBytes, rows,
                                           x_stride, 16, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_NONE,
                                           "encode wide A4 residual x scales TMA");
    };
    const auto h_maps = [&](std::uint8_t* base, CUtensorMap& codes, CUtensorMap& scales) {
        codes  = detail::nvfp4_make_tma_2d(base, CU_TENSOR_MAP_DATA_TYPE_UINT8, kHCodeBytes, rows, h_stride,
                                           Schedule::kCodeRowBytes, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_64B,
                                           "encode wide A4 residual h codes TMA");
        scales = detail::nvfp4_make_tma_2d(base + kHCodeBytes, CU_TENSOR_MAP_DATA_TYPE_UINT8, kHScaleMapBytes, rows,
                                           h_stride, 16, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_NONE,
                                           "encode wide A4 residual h scales TMA");
    };
    x_maps(x, d.x_codes, d.x_scales);
    x_maps(x + kXRowBytes, d.x2_codes, d.x2_scales);
    x_maps(x + 2 * kXRowBytes, d.x3_codes, d.x3_scales);
    h_maps(h, d.h_codes, d.h_scales);
    h_maps(h + kHRowStride, d.h2_codes, d.h2_scales);
    h_maps(h + 2 * kHRowStride, d.h3_codes, d.h3_scales);
    return d;
}

} // namespace

std::size_t experts_workspace_bytes(const ExpertShape& shape, std::int32_t max_jobs, std::int32_t entries,
                                    std::int32_t columns, ExpertActivation activation, WideA16 wide_a16) {
    return offsets(shape, max_jobs, entries, columns, activation, wide_a16).end;
}

ExpertsWorkspace carve_experts_workspace(void* base, const ExpertShape& shape, std::int32_t max_jobs,
                                         std::int32_t entries, std::int32_t columns, ExpertActivation activation,
                                         WideA16 wide_a16) {
    const Offsets o = offsets(shape, max_jobs, entries, columns, activation, wide_a16);
    auto* p         = static_cast<std::byte*>(base);
    ExpertsWorkspace w{};
    w.job_records = reinterpret_cast<const std::uint8_t**>(p + o.job_records);
    w.cpu_flags   = reinterpret_cast<std::int32_t*>(p + o.cpu_flags);
    w.cpu_call    = p + o.cpu_call;
    w.tiles       = reinterpret_cast<std::int32_t*>(p + o.tiles);
    w.pass_tiles  = reinterpret_cast<std::int32_t*>(p + o.pass_tiles);
    if (wide_a16 == WideA16::kLossless) {
        w.lossless = reinterpret_cast<std::uint8_t*>(p + o.lossless);
    } else if (activation == ExpertActivation::kA16 && residual_terms(wide_a16).x != 0) {
        w.h_rows      = reinterpret_cast<std::uint8_t*>(p + o.h_rows);
        w.h_emax      = reinterpret_cast<std::int32_t*>(p + o.h_emax);
        w.x_terms_plane = reinterpret_cast<std::uint8_t*>(p + o.x_terms_plane);
        w.x_row_scale = reinterpret_cast<float*>(p + o.x_row_scale);
        w.h_terms_plane = reinterpret_cast<std::uint8_t*>(p + o.h_terms_plane);
        w.h_row_scale = reinterpret_cast<float*>(p + o.h_row_scale);
    } else if (activation == ExpertActivation::kA16) {
        w.x_limbs = reinterpret_cast<std::uint8_t*>(p + o.x_limbs);
        w.h_rows  = reinterpret_cast<std::uint8_t*>(p + o.h_rows);
        w.x_emax  = reinterpret_cast<std::int32_t*>(p + o.x_emax);
        w.h_emax  = reinterpret_cast<std::int32_t*>(p + o.h_emax);
        w.ggml_planes = o.has_ggml_planes ? reinterpret_cast<std::uint8_t*>(p + o.ggml_planes) : nullptr;
    } else {
        w.h_blocks = reinterpret_cast<canon::A4Block*>(p + o.h_blocks);
        w.x_plane  = reinterpret_cast<std::uint8_t*>(p + o.x_plane);
    }
    if (o.has_ring) {
        w.x_acts        = reinterpret_cast<std::uint8_t*>(p + o.x_acts);
        w.h_acts        = reinterpret_cast<std::uint8_t*>(p + o.h_acts);
        w.ring_done     = reinterpret_cast<std::uint32_t*>(p + o.ring_jobs);
        w.ring_ready    = w.ring_done + max_jobs;
        w.ring_counters = reinterpret_cast<std::uint32_t*>(p + o.ring_counters);
    }
    return w;
}

int persistent_ctas() {
    static const int ctas = [] {
        int device = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return sms * Schedule::kMinBlocksPerSm;
    }();
    return ctas;
}

Call prepare(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source, std::int32_t top_k,
             std::int32_t max_jobs, std::int32_t pass_jobs, const ExpertsWorkspace& workspace, Tensor& outputs,
             cudaStream_t stream) {
    if (pass_jobs <= 0 || max_jobs <= 0) { throw std::invalid_argument("offloaded_sparse_moe wide: empty passes"); }
    const int entries = outputs.ne[1];
    Call call;
    call.dispatch    = dispatch;
    call.shape       = source.shape;
    call.scales      = source.scales;
    call.job_records = workspace.job_records;
    call.tiles       = workspace.tiles;
    call.pass_tiles  = workspace.pass_tiles;
    call.h_plane     = reinterpret_cast<std::uint8_t*>(workspace.h_blocks);
    call.outputs     = static_cast<std::uint16_t*>(outputs.data);
    call.passes      = (max_jobs + pass_jobs - 1) / pass_jobs;
    plan_kernel<<<1, kPlanThreads, 0, stream>>>(dispatch, source.scales, max_jobs, pass_jobs, call.passes,
                                                workspace.tiles, workspace.pass_tiles);
    check_launch("plan");
    const ResidualTerms terms =
        source.activation == ExpertActivation::kA16 ? residual_terms(source.wide_a16) : ResidualTerms{};
    if (terms.x != 0) {
        // W4A16 A4 residual: every token's row scale, then each wide entry's x terms; h's terms are
        // formed per pass from the BF16 h the gate/up epilogue writes.
        call.a4_residual   = true;
        call.terms         = terms;
        call.h_rows        = workspace.h_rows;
        call.top_k         = top_k;
        call.x_row_scale   = workspace.x_row_scale;
        call.h_terms_plane = workspace.h_terms_plane;
        call.h_row_scale   = workspace.h_row_scale;
        x_row_scale_kernel<<<x.ne[1], kEncodeXThreads, 0, stream>>>(static_cast<const std::uint16_t*>(x.data),
                                                                    workspace.x_row_scale);
        check_launch("x row scale");
        const auto quantize_x = [&]<int Terms>() {
            quantize_x_terms_kernel<Terms><<<persistent_ctas() * 2, kQuantizeThreads, 0, stream>>>(
                static_cast<const std::uint16_t*>(x.data), top_k, dispatch, workspace.tiles, workspace.pass_tiles,
                call.passes, workspace.x_row_scale, workspace.x_terms_plane);
        };
        terms.x == 3 ? quantize_x.template operator()<3>() : quantize_x.template operator()<2>();
        check_launch("quantize A4 residual x");
        const Descriptors descriptors =
            make_descriptors_terms(workspace.x_terms_plane, workspace.h_terms_plane, entries, terms);
#ifdef _WIN32
        call.descriptors = descriptor_staging().stage(descriptors, stream);
#else
        host_descriptors() = descriptors;
        call.descriptors   = &host_descriptors();
#endif
        return call;
    }
    if (source.activation == ExpertActivation::kA16) {
        // W4A16: every token's x limbs once (a token feeds up to top_k experts); h is encoded per pass.
        call.a16     = true;
        call.x_limbs = workspace.x_limbs;
        call.h_rows  = workspace.h_rows;
        call.x_emax  = workspace.x_emax;
        call.h_emax  = workspace.h_emax;
        call.top_k   = top_k;
        with_expert_geometry(source.shape, [&]<class G>() {
            encode_x_kernel<G><<<x.ne[1], G::kHidden / 16, 0, stream>>>(static_cast<const std::uint16_t*>(x.data),
                                                                       workspace.x_limbs, workspace.x_emax);
        });
        check_launch("encode x");
        return call;
    }
    quantize_kernel<<<persistent_ctas() * 2, kQuantizeThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), top_k, dispatch, source.scales, workspace.tiles,
        workspace.pass_tiles, call.passes, workspace.x_plane);
    check_launch("quantize");
    const Descriptors descriptors = make_descriptors(workspace.x_plane, call.h_plane, entries);
#ifdef _WIN32
    call.descriptors = descriptor_staging().stage(descriptors, stream);
#else
    host_descriptors() = descriptors;
    call.descriptors   = &host_descriptors();
#endif
    return call;
}

void encode_x_limbs(const ExpertShape& shape, const Tensor& x, std::uint8_t* limbs, std::int32_t* emax,
                    cudaStream_t stream) {
    if (!registered_shape(shape) || x.dtype != DType::BF16 || x.ne[0] != shape.hidden || x.ne[1] <= 0 ||
        !x.is_contiguous()) {
        throw std::invalid_argument("offloaded_sparse_moe: x limbs need BF16 x [H, T] of a registered geometry");
    }
    with_expert_geometry(shape, [&]<class G>() {
        encode_x_kernel<G><<<x.ne[1], G::kHidden / 16, 0, stream>>>(static_cast<const std::uint16_t*>(x.data), limbs,
                                                                   emax);
    });
    check_launch("encode x limbs");
}

void run_pass(const Call& call, std::int32_t pass, cudaStream_t stream) {
    if (pass < 0 || pass >= call.passes) { throw std::out_of_range("offloaded_sparse_moe wide: pass out of range"); }
    if (call.a4_residual) {
        const auto gate_up = [&]<int Terms>() {
            launch<Matrix::GateUp, ScheduleResidual<Terms>>(
                call, pass,
                GateUpEpilogueResidual{call.scales, call.dispatch.entries, call.x_row_scale, call.h_rows, call.top_k},
                stream);
        };
        const auto down = [&]<int Terms>() {
            quantize_h_terms_kernel<Terms><<<persistent_ctas(), kEncodeWarps * 32, 0, stream>>>(
                call.dispatch, call.tiles, call.pass_tiles, pass, call.h_rows, call.h_terms_plane, call.h_row_scale);
            check_launch("quantize A4 residual h");
            launch<Matrix::Down, ScheduleResidual<Terms>>(
                call, pass, DownEpilogueResidual{call.scales, call.dispatch.entries, call.h_row_scale, call.outputs},
                stream);
        };
        call.terms.x == 3 ? gate_up.template operator()<3>() : gate_up.template operator()<2>();
        call.terms.h == 3 ? down.template operator()<3>() : down.template operator()<2>();
        return;
    }
    if (call.a16) {
        with_expert_geometry(call.shape, [&]<class G>() {
            // A call with shared-expert jobs maps entries through entry_column; a routed-only call
            // (every Qwen3.8-Flash-Next call) keeps the plain token row.
            if (call.dispatch.shared_first != 0x7FFFFFFF) {
                launch_a16<A16Matrix<G, Matrix::GateUp, true>>(
                    call, pass,
                    SharedGateUpEpilogueA16<G>{call.scales, call.dispatch.entries, call.x_emax, call.h_rows,
                                               call.top_k, call.dispatch.shared_first},
                    stream);
            } else {
                launch_a16<A16Matrix<G, Matrix::GateUp>>(
                    call, pass,
                    GateUpEpilogueA16<G>{call.scales, call.dispatch.entries, call.x_emax, call.h_rows, call.top_k},
                    stream);
            }
            encode_h_kernel<G><<<persistent_ctas(), kEncodeWarps * 32, 0, stream>>>(
                call.dispatch, call.tiles, call.pass_tiles, pass, call.h_rows, call.h_emax);
            check_launch("encode h");
            launch_a16<A16Matrix<G, Matrix::Down>>(
                call, pass, DownEpilogueA16<G>{call.scales, call.dispatch.entries, call.h_emax, call.outputs}, stream);
        });
        return;
    }
    launch<Matrix::GateUp>(call, pass, GateUpEpilogue{call.scales, call.h_plane}, stream);
    launch<Matrix::Down>(call, pass, DownEpilogue{call.scales, call.dispatch.entries, call.outputs}, stream);
}

} // namespace infernix::ops::offloaded_moe::wide
