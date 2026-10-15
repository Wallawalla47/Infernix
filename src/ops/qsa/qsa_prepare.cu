// QSA prepare (include/infernix/ops/qsa.h qsa_prepare; the qsa-prepare-fused fast path): one kernel
// normalizes and rotates a call's queries, keys, index queries and pooled index keys and appends its
// K/V rows through kv_cache_append's cache codecs (ops/kv_cache), keeping FP32 from the input rows
// through the normalization, the rotation, the Hadamard preparation and the codec.
//
// One warp per task, four tasks per 128-thread CTA, the tasks of a column in consecutive CTAs: one
// per query head, index-query head and KV head, and one pool task. A lane holds dimensions
// lane + 32 j of a head (the layout normalized_hadamard_d256_inplace takes), so the rotary pair
// (lane, lane + 32) of the 64 rotated dimensions sits in one lane. Nothing a task computes depends on
// the other columns of the call, so a column's bits do not depend on the call's width or batch.

#include "infernix/ops/qsa.h"
#include "infernix/ops/rope.h"

#include "core/dtype.h"
#include "core/paged_kv_storage.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/qsa/page_spaces.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kHeadDim       = 256;
constexpr int kIndexDim      = 128;
constexpr int kRotary        = 64;
constexpr int kPairs         = kRotary / 2;
constexpr int kWarps         = 4;
constexpr int kThreads       = 32 * kWarps;
constexpr unsigned kFullMask = 0xFFFFFFFFU;

static_assert(kPairs == 32, "one rotary pair per lane: dimensions lane and lane + 32");

// Kernel arguments (one __grid_constant__ block: the YaRN frequencies are indexed by lane).
struct PrepareArgs {
    const bf16* q;
    const bf16* k;
    const bf16* v;
    const bf16* index_q;
    const bf16* index_k;
    const bf16* query_norm;
    const bf16* key_norm;
    const bf16* index_query_norm;
    const bf16* index_key_norm;
    bf16* q_out;
    bf16* index_q_out;
    const std::int32_t* positions;  // [columns]
    const std::int32_t* rope;       // [columns, 3] axis-major
    const std::int32_t* block_rope; // [batch, 3] axis-major
    const std::int32_t* tables;
    const std::int32_t* table_rows;
    const std::int32_t* tail_slots;
    bf16* tails;
    void* k_pages;
    void* v_pages;
    void* k_scales;
    void* v_scales;
    bf16* pooled;
    QsaPageSpaces spaces;
    std::int32_t table_stride;
    std::int32_t columns;
    std::int32_t batch;
    std::int32_t width;
    std::int32_t heads;
    std::int32_t index_heads;
    std::int32_t ratio;
    std::int32_t first_task; // 0, or the first index-free task of a K/V-only call
    std::int32_t groups;     // CTAs per column
    float theta;
    float eps;
    float attention_scale;   // YaRN's cosine/sine scale
    bool yarn;
    bool update_tails;
    double inverse[kPairs];  // YaRN inverse frequencies (prepare_rope)
};

// ------------------------------------------------------------------------------- rotation

// YaRN pair `pair` at coordinate `position`: the angle in FP64, reduced to one turn, FP32 sincos,
// both scaled by the attention scale (ops/kernel/rope.cuh PreparedRopeCoefficients::sincos_axis and
// qsa.cu's rope_angle, the same coefficients for the attention and the index rotation).
__device__ __forceinline__ void yarn_sincos(const PrepareArgs& a, int pair, int position, float& c, float& s) {
    constexpr double kInvTwoPi = 1.59154943091895336e-01;
    constexpr double kTwoPi    = 6.28318530717958648e+00;
    const double angle  = static_cast<double>(position) * a.inverse[pair];
    const float reduced = static_cast<float>(angle - nearbyint(angle * kInvTwoPi) * kTwoPi);
    sincosf(reduced, &s, &c);
    s *= a.attention_scale;
    c *= a.attention_scale;
}

// The attention rotation's coefficients: ops::rope's (the generic kernel's frequency theta^(-2i/R) in
// FP32 for the Qwen4Exp 24/2-head geometry), or YaRN's.
__device__ __forceinline__ void attention_sincos(const PrepareArgs& a, int pair, int position, float& c, float& s) {
    if (a.yarn) {
        yarn_sincos(a, pair, position, c, s);
        return;
    }
    const float frequency = powf(a.theta, -2.0F * static_cast<float>(pair) / static_cast<float>(kRotary));
    sincosf(static_cast<float>(position) * frequency, &s, &c);
}

// The index rotation's coefficients: qsa_index_query's and qsa_pool_keys's (qsa.cu rope_angle).
__device__ __forceinline__ void index_sincos(const PrepareArgs& a, int pair, int position, float& c, float& s) {
    if (a.yarn) {
        yarn_sincos(a, pair, position, c, s);
        return;
    }
    const float inverse = 1.0F / powf(a.theta, static_cast<float>(2 * pair) / static_cast<float>(kRotary));
    sincosf(static_cast<float>(position) * inverse, &s, &c);
}

// Axis lane % 3 (the axis of pair `lane`) of column `column`'s RoPE position.
__device__ __forceinline__ int column_coordinate(const PrepareArgs& a, int column, int lane) {
    return a.rope[(lane % 3) * a.columns + column];
}

// ------------------------------------------------------------------------------- rows

template <int N>
__device__ __forceinline__ void load_row(const bf16* row, float (&x)[N], int lane) {
#pragma unroll
    for (int j = 0; j < N; ++j) { x[j] = __bfloat162float(row[lane + 32 * j]); }
}

template <int N>
__device__ __forceinline__ void store_row(bf16* row, const float (&x)[N], int lane) {
#pragma unroll
    for (int j = 0; j < N; ++j) { row[lane + 32 * j] = __float2bfloat16_rn(x[j]); }
}

// RMSNorm of a 32 N-dimensional row with the unit-offset weight, in FP32.
template <int N>
__device__ __forceinline__ void normalize_row(float (&x)[N], const bf16* weight, float eps, int lane) {
    float squares = 0.0F;
#pragma unroll
    for (int j = 0; j < N; ++j) { squares += x[j] * x[j]; }
    squares             = warp_sum(squares);
    const float inverse = rsqrtf(squares / static_cast<float>(32 * N) + eps);
#pragma unroll
    for (int j = 0; j < N; ++j) { x[j] = x[j] * inverse * (1.0F + __bfloat162float(weight[lane + 32 * j])); }
}

// The half-split rotation of pair `lane`: dimensions lane (x[0]) and lane + 32 (x[1]).
template <int N>
__device__ __forceinline__ void rotate_pair(float (&x)[N], float c, float s) {
    const float first = x[0], second = x[1];
    x[0]              = first * c - second * s;
    x[1]              = second * c + first * s;
}

// ------------------------------------------------------------------------------- cache codecs
// Each writes one FP32 row in kv_cache_append's encoding (ops/kv_cache/append): the same codec
// functions in the same order, so a row's bytes are those kv_cache_append writes for that FP32 row.

template <class Geometry>
__device__ __forceinline__ void store_int8_row(const float (&x)[8], std::int8_t* codes, __half* scales, int page,
                                               int head, int slot, int lane) {
#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        const float x0   = x[2 * group];
        const float x1   = x[2 * group + 1];
        const auto quant = kv_cache_int8_quant_params(warp_max(fmaxf(fabsf(x0), fabsf(x1)), kFullMask));
        const std::int64_t base =
            kv_cache_int8_quant_code_index<Geometry>(page, head, group * kKVCacheInt8Group, slot);
        codes[base + lane]      = kv_cache_int8_quant_code(x0, quant.inverse_scale);
        codes[base + lane + 32] = kv_cache_int8_quant_code(x1, quant.inverse_scale);
        if (lane == 0) { scales[kv_cache_int8_quant_scale_index<Geometry>(page, head, group, slot)] = quant.scale; }
    }
}

template <class Geometry>
__device__ __forceinline__ void store_fp8_row(const float (&x)[8], std::uint8_t* codes, __half* scales, int page,
                                              int head, int slot, int lane) {
    float absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) { absmax = fmaxf(absmax, fabsf(x[r])); }
    const auto quant = kv_cache_fp8_quant_params(warp_max(absmax, kFullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        codes[kv_cache_fp8_code_index<Geometry>(page, head, lane + 32 * r, slot)] =
            kv_cache_fp8_quant_code(x[r], quant.inverse_scale);
    }
    if (lane == 0) { scales[kv_cache_fp8_scale_index<Geometry>(page, head, slot)] = quant.scale; }
}

// `scratch`: this warp's 256 FP32 values in shared memory.
template <class Geometry>
__device__ __forceinline__ void store_nvfp4_row(const float (&x)[8], std::uint8_t* codes, std::uint8_t* scales,
                                                int page, int head, int slot, int lane, float* scratch) {
#pragma unroll
    for (int r = 0; r < 8; ++r) { scratch[lane + 32 * r] = x[r]; }
    __syncwarp();
    const auto quantized = kv_cache_nvfp4_quantize_group16_warp(scratch, lane);
    if (lane < kKVCacheNvfp4Groups) {
        store_vec(codes + kv_cache_nvfp4_code_index<Geometry>(page, head, lane * kKVCacheNvfp4Group, slot),
                  make_uint2(quantized.codes_lo, quantized.codes_hi));
        scales[kv_cache_nvfp4_scale_index<Geometry>(page, head, lane, slot)] = quantized.scale;
    }
    __syncwarp(); // every lane has read the row before the warp reuses its scratch
}

// ------------------------------------------------------------------------------- tasks

// Query head `head` (index = false) or index-query head `head` (index = true) of `column`.
template <int N, bool Index>
__device__ __forceinline__ void prepare_query(const PrepareArgs& a, int column, int head, int lane) {
    constexpr int kDim     = 32 * N;
    const int heads        = Index ? a.index_heads : a.heads;
    const std::size_t row  = (static_cast<std::size_t>(column) * heads + head) * kDim;
    const bf16* in         = (Index ? a.index_q : a.q) + row;
    float x[N];
    load_row(in, x, lane);
    normalize_row(x, Index ? a.index_query_norm : a.query_norm, a.eps, lane);
    float c, s;
    if constexpr (Index) {
        index_sincos(a, lane, column_coordinate(a, column, lane), c, s);
    } else {
        attention_sincos(a, lane, column_coordinate(a, column, lane), c, s);
    }
    rotate_pair(x, c, s);
    store_row((Index ? a.index_q_out : a.q_out) + row, x, lane); // every load preceded the warp sum
}

// KV head `head` of `column`: the normalized, rotated FP32 key and the value into their codec.
template <KvCacheStorage Storage, class Geometry>
__device__ __forceinline__ void prepare_kv(const PrepareArgs& a, int column, int head, int lane, float* scratch) {
    const int sequence        = column / a.width;
    const int position        = a.positions[column];
    const std::int32_t* table = a.tables + static_cast<std::int64_t>(a.table_rows[sequence]) * a.table_stride;
    const int page            = paged_kv_physical_page(table, position);
    const int slot            = position & kPagedKVPageMask;
    const std::size_t row     = (static_cast<std::size_t>(column) * Geometry::KVHeads + head) * kHeadDim;
    float key[8];
    load_row(a.k + row, key, lane);
    normalize_row(key, a.key_norm, a.eps, lane);
    float c, s;
    attention_sincos(a, lane, column_coordinate(a, column, lane), c, s);
    rotate_pair(key, c, s);
    if constexpr (Storage == KvCacheStorage::BFloat16) {
        // K stored as BF16 (its one rounding); V as FP16_RNE(BF16 v), kv_cache_append's conversion.
        const std::int64_t base = paged_kv_element_offset<kHeadDim, Geometry::KVHeads>(page, head, slot, 0);
        store_row(static_cast<bf16*>(a.k_pages) + base, key, lane);
        const int4 value = load_vec<int4>(a.v + row + 8 * lane);
        store_vec(static_cast<__half*>(a.v_pages) + base + 8 * lane, bf16x8_bits_to_f16x8_bits(value));
    } else {
        // Keys of every quantized storage, and NVFP4 values, take the normalized D256 Hadamard
        // preparation; the codecs read FP32 rows.
        normalized_hadamard_d256_inplace(key, lane);
        float value[8];
        load_row(a.v + row, value, lane);
        if constexpr (Storage == KvCacheStorage::Int8Group64) {
            store_int8_row<Geometry>(key, static_cast<std::int8_t*>(a.k_pages), static_cast<__half*>(a.k_scales),
                                     page, head, slot, lane);
            store_int8_row<Geometry>(value, static_cast<std::int8_t*>(a.v_pages), static_cast<__half*>(a.v_scales),
                                     page, head, slot, lane);
        } else if constexpr (Storage == KvCacheStorage::Fp8E4M3Row256) {
            store_fp8_row<Geometry>(key, static_cast<std::uint8_t*>(a.k_pages), static_cast<__half*>(a.k_scales),
                                    page, head, slot, lane);
            store_fp8_row<Geometry>(value, static_cast<std::uint8_t*>(a.v_pages), static_cast<__half*>(a.v_scales),
                                    page, head, slot, lane);
        } else if constexpr (Storage == KvCacheStorage::Nvfp4Group16) {
            store_nvfp4_row<Geometry>(key, static_cast<std::uint8_t*>(a.k_pages),
                                      static_cast<std::uint8_t*>(a.k_scales), page, head, slot, lane, scratch);
            normalized_hadamard_d256_inplace(value, lane);
            store_nvfp4_row<Geometry>(value, static_cast<std::uint8_t*>(a.v_pages),
                                      static_cast<std::uint8_t*>(a.v_scales), page, head, slot, lane, scratch);
        } else {
            static_assert(Storage == KvCacheStorage::Fp8KeyNvfp4Value, "a storage without a fused prepare");
            store_fp8_row<Geometry>(key, static_cast<std::uint8_t*>(a.k_pages), static_cast<__half*>(a.k_scales),
                                    page, head, slot, lane);
            normalized_hadamard_d256_inplace(value, lane);
            store_nvfp4_row<Geometry>(value, static_cast<std::uint8_t*>(a.v_pages),
                                      static_cast<std::uint8_t*>(a.v_scales), page, head, slot, lane, scratch);
        }
    }
}

// The pooled key of sequence `sequence`'s block starting at `first` (start: the sequence's first
// column position): qsa_pool_keys's mean (raw keys before the call from the tails, in position
// order, rounded to BF16), then the FP32 normalization and rotation and one BF16 rounding.
__device__ __forceinline__ void pool_block(const PrepareArgs& a, int sequence, int start, int first, int lane) {
    const int ratio = a.ratio;
    const int tail  = a.tail_slots[sequence];
    float sum[4]    = {0.0F, 0.0F, 0.0F, 0.0F};
    for (int q = first; q < first + ratio; ++q) {
        const bf16* raw = q >= start
                              ? a.index_k + static_cast<std::size_t>(sequence * a.width + (q - start)) * kIndexDim
                              : a.tails + (static_cast<std::size_t>(tail) * (ratio - 1) + q % ratio) * kIndexDim;
#pragma unroll
        for (int j = 0; j < 4; ++j) { sum[j] += __bfloat162float(raw[lane + 32 * j]); }
    }
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        x[j] = __bfloat162float(__float2bfloat16_rn(sum[j] / static_cast<float>(ratio)));
    }
    normalize_row(x, a.index_key_norm, a.eps, lane);
    // The block's first token: a column of this call, or the sequence's block start before it.
    const int coordinate = first >= start ? column_coordinate(a, sequence * a.width + (first - start), lane)
                                          : a.block_rope[(lane % 3) * a.batch + sequence];
    float c, s;
    index_sincos(a, lane, coordinate, c, s);
    rotate_pair(x, c, s);
    const std::int32_t* table = a.tables + static_cast<std::int64_t>(a.table_rows[sequence]) * a.table_stride;
    const int slot_width      = kIndexDim / ratio;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d     = lane + 32 * j;
        const int token = first + d / slot_width;
        int page;
        bf16* plane = detail::qsa_pooled_plane(a.spaces, a.pooled, table[token >> kPagedKVPageShift], page);
        plane[static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
              static_cast<std::int64_t>(slot_width) * (token & kPagedKVPageMask) + d % slot_width] =
            __float2bfloat16_rn(x[j]);
    }
}

// qsa_pool_keys's tail update of sequence `sequence`: slot q % R holds the raw key of every position
// q of the latest block with q % R < R - 1 up to the call's last position.
__device__ __forceinline__ void write_tails(const PrepareArgs& a, int sequence, int start, int lane) {
    const int ratio = a.ratio;
    const int tail  = a.tail_slots[sequence];
    const int last  = a.positions[sequence * a.width + a.width - 1];
    const int base  = last - last % ratio;
    const int end   = min(last, base + ratio - 2);
    for (int q = max(base, start); q <= end; ++q) { // positions before the call are in the tail already
        const bf16* raw = a.index_k + static_cast<std::size_t>(sequence * a.width + (q - start)) * kIndexDim;
        bf16* out       = a.tails + (static_cast<std::size_t>(tail) * (ratio - 1) + q % ratio) * kIndexDim;
#pragma unroll
        for (int j = 0; j < 4; ++j) { out[lane + 32 * j] = raw[lane + 32 * j]; }
    }
}

// The pool task of `column`. A block whose positions all lie in the call is pooled by the column that
// completes it. The one block a sequence carries into the call (it starts before the call's first
// position and completes inside it) reads the tails, so the sequence's first column pools it and
// only then rewrites the tails: no other task reads them.
__device__ __forceinline__ void prepare_pool(const PrepareArgs& a, int column, int lane) {
    const int sequence = column / a.width;
    const int start    = a.positions[sequence * a.width];
    if (column % a.width == 0) {
        const int last  = a.positions[sequence * a.width + a.width - 1];
        const int first = start - start % a.ratio;
        if (first < start && first + a.ratio - 1 <= last) { pool_block(a, sequence, start, first, lane); }
        if (a.update_tails) {
            __syncwarp(); // the carried block's tail reads precede the tail writes
            write_tails(a, sequence, start, lane);
        }
        return;
    }
    const int position = a.positions[column];
    const int first    = position - a.ratio + 1;
    if ((position + 1) % a.ratio == 0 && first >= start) { pool_block(a, sequence, start, first, lane); }
}

template <KvCacheStorage Storage, class Geometry>
__global__ void __launch_bounds__(kThreads) qsa_prepare_kernel(const __grid_constant__ PrepareArgs a) {
    __shared__ float scratch[kWarps][kHeadDim]; // the NVFP4 codec's rows
    pdl::enter();                               // launched as a programmatic dependent
    const int warp   = static_cast<int>(threadIdx.x) / 32;
    const int lane   = static_cast<int>(threadIdx.x) % 32;
    const int column = static_cast<int>(blockIdx.x) / a.groups;
    const int task   = a.first_task + (static_cast<int>(blockIdx.x) % a.groups) * kWarps + warp;
    const int index_begin = a.heads;
    const int kv_begin    = index_begin + a.index_heads;
    const int pool_task   = kv_begin + Geometry::KVHeads;
    if (task < index_begin) {
        prepare_query<kHeadDim / 32, false>(a, column, task, lane);
    } else if (task < kv_begin) {
        prepare_query<kIndexDim / 32, true>(a, column, task - index_begin, lane);
    } else if (task < pool_task) {
        prepare_kv<Storage, Geometry>(a, column, task - kv_begin, lane, scratch[warp]);
    } else if (task == pool_task) {
        prepare_pool(a, column, lane);
    }
}

// ------------------------------------------------------------------------------- launcher

template <KvCacheStorage Storage, class Geometry>
void launch(const PrepareArgs& args, unsigned grid, cudaStream_t stream) {
    const cudaError_t error = pdl::launch_consumer({dim3(grid), dim3(kThreads), 0, stream},
                                                   qsa_prepare_kernel<Storage, Geometry>, args);
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("qsa_prepare launch: ") + cudaGetErrorString(error));
    }
}

template <KvCacheStorage Storage>
void launch_heads(const PrepareArgs& args, int kv_heads, unsigned grid, cudaStream_t stream) {
    if (kv_heads == KVCacheAppendD256Kv2::KVHeads) {
        launch<Storage, KVCacheAppendD256Kv2>(args, grid, stream);
    } else {
        launch<Storage, KVCacheAppendD256Kv4>(args, grid, stream);
    }
}

// ------------------------------------------------------------------------------- validation

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa_prepare: ") + message); }
}

// Contiguous, non-null, of `dtype` and exactly `shape` (trailing extents 1).
bool dense(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape) {
    if (t.data == nullptr || t.dtype != dtype || !t.is_contiguous()) { return false; }
    int d = 0;
    for (const std::int32_t n : shape) {
        if (t.ne[d++] != n) { return false; }
    }
    for (; d < 4; ++d) {
        if (t.ne[d] != 1) { return false; }
    }
    return true;
}

// A paged plane of `row_bytes` bytes per (token, head): [leading, 64, kv_heads, pages], contiguous.
bool plane(const Tensor& t, int row_bytes, int kv_heads, int pages) {
    return t.data != nullptr && t.is_contiguous() &&
           static_cast<std::size_t>(t.ne[0]) * dtype_size(t.dtype) == static_cast<std::size_t>(row_bytes) &&
           t.ne[1] == kPagedKVPageSize && t.ne[2] == kv_heads && t.ne[3] == pages &&
           reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0;
}

bool disjoint(const Tensor& a, const Tensor& b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.data), y = reinterpret_cast<std::uintptr_t>(b.data);
    return x + a.bytes() <= y || y + b.bytes() <= x;
}

bool same_view(const Tensor& a, const Tensor& b) {
    if (a.data != b.data || a.dtype != b.dtype) { return false; }
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d] || a.nb[d] != b.nb[d]) { return false; }
    }
    return true;
}

void require_geometry(const QsaGeometry& g) {
    require(g.head_dim == kHeadDim && g.index_head_dim == kIndexDim && g.rotary_dim == kRotary &&
                (g.kv_heads == 2 || g.kv_heads == 4) && g.heads > 0 && g.heads % g.kv_heads == 0 &&
                g.index_heads >= 1 && g.index_heads <= 4 && g.ratio > 1 && kIndexDim % g.ratio == 0 &&
                g.theta > 0.0F && std::isfinite(g.theta) && g.eps > 0.0F && std::isfinite(g.eps),
            "unsupported geometry (head_dim 256, Di 128, rotary_dim 64, 2 or 4 KV heads)");
    require(std::isfinite(g.yarn_factor) && g.yarn_factor >= 1.0F && g.yarn_factor <= 4.0F &&
                (g.yarn_factor == 1.0F || (g.original_positions > 0 && g.theta > 1.0F)),
            "YaRN needs a factor in [1,4], original positions and theta above 1");
}

} // namespace

bool qsa_prepare_supports(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16:
    case KvCacheStorage::Int8Group64:
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Nvfp4Group16:
    case KvCacheStorage::Fp8KeyNvfp4Value: return true;
    default: return false;
    }
}

void qsa_prepare(const QsaPrepareInputs& inputs, const QsaPrepareNorms& norms, Tensor& tails,
                 const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& geometry,
                 Tensor& q_out, Tensor& index_q_out, cudaStream_t stream) {
    require_geometry(geometry);
    const KvCacheStorage storage = layer.kv.storage;
    require(qsa_prepare_supports(storage),
            "the KV storage has no fused prepare (the vector-quantized storages append in qsa_attention)");
    const int heads = geometry.heads, kv_heads = geometry.kv_heads, index_heads = geometry.index_heads;
    const int columns = inputs.q.ne[2];
    require(batch.batch > 0 && batch.width > 0 && columns == batch.batch * batch.width, "T must be batch * width");

    // Inputs and norms.
    require(dense(inputs.q, DType::BF16, {kHeadDim, heads, columns}) &&
                dense(inputs.k, DType::BF16, {kHeadDim, kv_heads, columns}) &&
                dense(inputs.v, DType::BF16, {kHeadDim, kv_heads, columns}) &&
                dense(inputs.index_q, DType::BF16, {kIndexDim, index_heads, columns}) &&
                dense(inputs.index_k, DType::BF16, {kIndexDim, columns}),
            "inputs are BF16 q [256, heads, T], k/v [256, kv_heads, T], index_q [128, index_heads, T], "
            "index_k [128, T]");
    require(reinterpret_cast<std::uintptr_t>(inputs.v.data) % 16 == 0, "v must be 16-byte aligned");
    require(dense(norms.query, DType::BF16, {kHeadDim}) && dense(norms.key, DType::BF16, {kHeadDim}) &&
                dense(norms.index_query, DType::BF16, {kIndexDim}) &&
                dense(norms.index_key, DType::BF16, {kIndexDim}),
            "norm weights are BF16 [256] (query, key) and [128] (index query, index key)");

    // Outputs: both present (in place or disjoint from every input) or both empty (K/V only).
    const bool queries = q_out.data != nullptr;
    require(queries == (index_q_out.data != nullptr), "q_out and index_q_out are both present or both empty");
    if (queries) {
        require(dense(q_out, DType::BF16, {kHeadDim, heads, columns}) &&
                    dense(index_q_out, DType::BF16, {kIndexDim, index_heads, columns}),
                "q_out and index_q_out are shaped like q and index_q");
        require(disjoint(q_out, index_q_out), "q_out and index_q_out overlap");
        for (const Tensor* in : {&inputs.q, &inputs.k, &inputs.v, &inputs.index_q, &inputs.index_k}) {
            require((in == &inputs.q && same_view(q_out, *in)) || disjoint(q_out, *in),
                    "q_out must be q itself or not overlap the inputs");
            require((in == &inputs.index_q && same_view(index_q_out, *in)) || disjoint(index_q_out, *in),
                    "index_q_out must be index_q itself or not overlap the inputs");
        }
    }

    // The call's rows.
    require(dense(batch.positions, DType::I32, {columns}) && dense(batch.table_rows, DType::I32, {batch.batch}) &&
                dense(batch.tail_slots, DType::I32, {batch.batch}) &&
                batch.block_tables.data != nullptr && batch.block_tables.dtype == DType::I32 &&
                batch.block_tables.is_contiguous(),
            "batch description is invalid");
    require(dense(batch.rope_positions, DType::I32, {columns, 3}) &&
                dense(batch.block_start_rope, DType::I32, {batch.batch, 3}),
            "RoPE positions are [T, 3] and block starts [batch, 3]");
    require(tails.data != nullptr && tails.dtype == DType::BF16 && tails.is_contiguous() &&
                tails.ne[0] == kIndexDim && tails.ne[1] == geometry.ratio - 1,
            "tails are BF16 [128, R - 1, slots]");

    // The layer's planes (the storage's layout) and pooled keys.
    require(layer.kv.head_dim == kHeadDim && layer.kv.num_kv_heads == kv_heads, "KV geometry differs");
    const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kHeadDim);
    const int pages                   = layer.kv.k_pages.ne[3];
    const auto bytes                  = [](const PagedKVVectorLayout& v, bool scale) {
        return scale ? v.scale_leading_extent * static_cast<int>(dtype_size(v.scale_dtype))
                     : v.data_leading_extent * static_cast<int>(dtype_size(v.data_dtype));
    };
    require(pages > 0 && plane(layer.kv.k_pages, bytes(layout.key, false), kv_heads, pages) &&
                plane(layer.kv.v_pages, bytes(layout.value, false), kv_heads, pages),
            "K/V planes do not match the storage layout");
    require(layout.key.has_scale() ? plane(layer.kv.k_scale_pages, bytes(layout.key, true), kv_heads, pages)
                                   : layer.kv.k_scale_pages.data == nullptr,
            "K scale plane does not match the storage layout");
    require(layout.value.has_scale() ? plane(layer.kv.v_scale_pages, bytes(layout.value, true), kv_heads, pages)
                                     : layer.kv.v_scale_pages.data == nullptr,
            "V scale plane does not match the storage layout");
    require(layer.pooled_pages.data != nullptr && layer.pooled_pages.dtype == DType::BF16 &&
                layer.pooled_pages.is_contiguous() && layer.pooled_pages.ne[0] == kIndexDim / geometry.ratio &&
                layer.pooled_pages.ne[1] == kPagedKVPageSize,
            "pooled keys are BF16 [Di / R, 64, 1, pages]");

    PrepareArgs args{};
    args.q                = static_cast<const bf16*>(inputs.q.data);
    args.k                = static_cast<const bf16*>(inputs.k.data);
    args.v                = static_cast<const bf16*>(inputs.v.data);
    args.index_q          = static_cast<const bf16*>(inputs.index_q.data);
    args.index_k          = static_cast<const bf16*>(inputs.index_k.data);
    args.query_norm       = static_cast<const bf16*>(norms.query.data);
    args.key_norm         = static_cast<const bf16*>(norms.key.data);
    args.index_query_norm = static_cast<const bf16*>(norms.index_query.data);
    args.index_key_norm   = static_cast<const bf16*>(norms.index_key.data);
    args.q_out            = static_cast<bf16*>(q_out.data);
    args.index_q_out      = static_cast<bf16*>(index_q_out.data);
    args.positions        = static_cast<const std::int32_t*>(batch.positions.data);
    args.rope             = static_cast<const std::int32_t*>(batch.rope_positions.data);
    args.block_rope       = static_cast<const std::int32_t*>(batch.block_start_rope.data);
    args.tables           = static_cast<const std::int32_t*>(batch.block_tables.data);
    args.table_rows       = static_cast<const std::int32_t*>(batch.table_rows.data);
    args.tail_slots       = static_cast<const std::int32_t*>(batch.tail_slots.data);
    args.tails            = static_cast<bf16*>(tails.data);
    args.k_pages          = layer.kv.k_pages.data;
    args.v_pages          = layer.kv.v_pages.data;
    args.k_scales         = layer.kv.k_scale_pages.data;
    args.v_scales         = layer.kv.v_scale_pages.data;
    args.pooled           = static_cast<bf16*>(layer.pooled_pages.data);
    args.spaces           = layer.spaces;
    args.table_stride     = batch.block_tables.ne[0];
    args.columns          = columns;
    args.batch            = batch.batch;
    args.width            = batch.width;
    args.heads            = heads;
    args.index_heads      = index_heads;
    args.ratio            = geometry.ratio;
    args.theta            = geometry.theta;
    args.eps              = geometry.eps;
    args.update_tails     = batch.update_tails;
    args.yarn             = geometry.yarn_factor != 1.0F;
    if (args.yarn) {
        const PreparedRope prepared = prepare_rope(
            kRotary, geometry.theta, {geometry.yarn_factor, static_cast<std::uint32_t>(geometry.original_positions)});
        args.attention_scale = prepared.attention_scale;
        for (int i = 0; i < kPairs; ++i) { args.inverse[i] = prepared.inverse[i]; }
    } else {
        args.attention_scale = 1.0F;
    }
    // A K/V-only call skips the query and index-query tasks.
    const int tasks  = heads + index_heads + kv_heads + 1;
    args.first_task  = queries ? 0 : heads + index_heads;
    args.groups      = (tasks - args.first_task + kWarps - 1) / kWarps;
    const auto grid  = static_cast<unsigned>(static_cast<std::int64_t>(columns) * args.groups);
    require(static_cast<std::int64_t>(columns) * args.groups <= 0x7FFFFFFF, "too many columns");

    switch (storage) {
    case KvCacheStorage::BFloat16: launch_heads<KvCacheStorage::BFloat16>(args, kv_heads, grid, stream); break;
    case KvCacheStorage::Int8Group64: launch_heads<KvCacheStorage::Int8Group64>(args, kv_heads, grid, stream); break;
    case KvCacheStorage::Fp8E4M3Row256:
        launch_heads<KvCacheStorage::Fp8E4M3Row256>(args, kv_heads, grid, stream);
        break;
    case KvCacheStorage::Nvfp4Group16: launch_heads<KvCacheStorage::Nvfp4Group16>(args, kv_heads, grid, stream); break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        launch_heads<KvCacheStorage::Fp8KeyNvfp4Value>(args, kv_heads, grid, stream);
        break;
    default: require(false, "unsupported KV storage");
    }
}

} // namespace infernix::ops
