// QSA indexer, pooled keys, block selection and block-sparse attention
// (docs/maintainer/qwen3_8-flash-next-design.md §8.10). KV storages: bf16 and int8 (INT8-G64 with
// Hadamard-rotated keys, as written by kv_cache_append).

#include "ninfer/ops/qsa.h"

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kHeadDim      = 256;
constexpr int kIndexDim     = 128;
constexpr int kMaxGroup     = 16; // query heads per KV head
constexpr int kTokenTile    = 64;
constexpr int kSelectThreads = 512;
constexpr int kSelectGroup  = 32; // columns whose scores share one scratch pass

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("qsa ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous();
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, o); }
    return v;
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    value          = warp_sum(value);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < blockDim.x / 32; ++w) { total += scratch[w]; }
    __syncthreads();
    return total;
}

// Half-split RoPE angle of pair i (i < rotary/2) at `position`, in FP32 like the reference.
__device__ __forceinline__ void rope_angle(int position, int i, int rotary, float theta, float& c,
                                           float& s) {
    const float inv   = 1.0F / powf(theta, static_cast<float>(2 * i) / static_cast<float>(rotary));
    const float angle = static_cast<float>(position) * inv;
    sincosf(angle, &s, &c);
}

// ------------------------------------------------------------------------------- index query

// One CTA (128 threads) per (index head, column).
__global__ void index_query_kernel(bf16* __restrict__ q, const bf16* __restrict__ weight,
                                   const std::int32_t* __restrict__ positions, int heads, int rotary,
                                   float theta, float eps) {
    __shared__ float scratch[4];
    __shared__ float normed[kIndexDim];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    bf16* row     = q + (static_cast<std::size_t>(t) * heads + h) * kIndexDim;
    const float x = __bfloat162float(row[d]);
    const float inv = rsqrtf(block_sum(x * x, scratch) / kIndexDim + eps);
    normed[d]       = __bfloat162float(__float2bfloat16_rn(x * inv * (1.0F + __bfloat162float(weight[d]))));
    __syncthreads();
    float out = normed[d];
    if (d < rotary) {
        const int half = rotary / 2, i = d % half;
        float c, s;
        rope_angle(positions[t], i, rotary, theta, c, s);
        out = d < half ? normed[d] * c - normed[d + half] * s : normed[d] * c + normed[d - half] * s;
    }
    row[d] = __float2bfloat16_rn(out);
}

// --------------------------------------------------------------------------------- pooled keys

__device__ __forceinline__ std::int64_t pooled_offset(const std::int32_t* table, int position,
                                                      int slot_width, int lane) {
    const int page = table[position >> kPagedKVPageShift];
    return static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
           static_cast<std::int64_t>(slot_width) * (position & kPagedKVPageMask) + lane;
}

// One CTA (128 threads) per column; columns that complete a block pool it.
__global__ void pool_kernel(const bf16* __restrict__ raw, const bf16* __restrict__ weight,
                            const bf16* __restrict__ tails, const std::int32_t* __restrict__ tables,
                            int table_stride, const std::int32_t* __restrict__ table_rows,
                            const std::int32_t* __restrict__ positions,
                            const std::int32_t* __restrict__ tail_slots, int width, int ratio,
                            int rotary, float theta, float eps, bf16* __restrict__ pooled) {
    __shared__ float scratch[4];
    __shared__ float normed[kIndexDim];
    const int t = blockIdx.x, d = threadIdx.x;
    const int p = positions[t];
    if ((p + 1) % ratio != 0) { return; }
    const int sequence = t / width;
    const int start    = positions[sequence * width];
    const int first    = p - ratio + 1;
    float sum          = 0.0F;
    for (int q = first; q <= p; ++q) {
        const bf16 v = q >= start ? raw[static_cast<std::size_t>(t - (p - q)) * kIndexDim + d]
                                  : tails[(static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) +
                                           (q % ratio)) * kIndexDim + d];
        sum += __bfloat162float(v);
    }
    const float mean = __bfloat162float(__float2bfloat16_rn(sum / static_cast<float>(ratio)));
    const float inv  = rsqrtf(block_sum(mean * mean, scratch) / kIndexDim + eps);
    normed[d] = __bfloat162float(__float2bfloat16_rn(mean * inv * (1.0F + __bfloat162float(weight[d]))));
    __syncthreads();
    float out = normed[d];
    if (d < rotary) {
        const int half = rotary / 2, i = d % half;
        float c, s;
        rope_angle(first, i, rotary, theta, c, s);
        out = d < half ? normed[d] * c - normed[d + half] * s : normed[d] * c + normed[d - half] * s;
    }
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[sequence]) * table_stride;
    const int slot_width      = kIndexDim / ratio;
    const int token           = first + d / slot_width;
    pooled[pooled_offset(table, token, slot_width, d % slot_width)] = __float2bfloat16_rn(out);
}

// One CTA per sequence: the raw keys of the incomplete block after the call.
__global__ void tail_kernel(const bf16* __restrict__ raw, const std::int32_t* __restrict__ positions,
                            const std::int32_t* __restrict__ tail_slots, int width, int ratio,
                            bf16* __restrict__ tails) {
    const int sequence = blockIdx.x, d = threadIdx.x;
    const int start    = positions[sequence * width];
    const int last     = positions[sequence * width + width - 1];
    const int base     = (last + 1) - (last + 1) % ratio;
    for (int q = base; q <= last; ++q) {
        if (q < start) { continue; } // still in the tail from an earlier call
        const int column = sequence * width + (q - start);
        tails[(static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) + (q % ratio)) * kIndexDim + d] =
            raw[static_cast<std::size_t>(column) * kIndexDim + d];
    }
}

// ----------------------------------------------------------------------------------- selection

__device__ __forceinline__ std::uint32_t sortable(float x) {
    const std::uint32_t u = __float_as_uint(x);
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}

// One CTA per column of a group: block scores into scratch, then the exact top-k by radix select,
// written in ascending block order. Columns that select every block write count = -1 (dense).
__global__ void __launch_bounds__(kSelectThreads)
    select_kernel(const bf16* __restrict__ index_q, int index_heads, const bf16* __restrict__ pooled,
                  const std::int32_t* __restrict__ tables, int table_stride,
                  const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                  int width, int ratio, int top_blocks, int column_begin, float* __restrict__ scores,
                  int score_stride, std::int32_t* __restrict__ selected,
                  std::int32_t* __restrict__ counts) {
    const int t = column_begin + blockIdx.x;
    const int p = positions[t];
    const int blocks = (p + 1) / ratio;
    if (blocks <= top_blocks) {
        if (threadIdx.x == 0) { counts[t] = -1; }
        return;
    }
    __shared__ float q[4 * kIndexDim];
    __shared__ unsigned histogram[256];
    __shared__ std::uint32_t prefix_shared, mask_shared;
    __shared__ int remaining_shared, scan[kSelectThreads];
    for (int i = threadIdx.x; i < index_heads * kIndexDim; i += blockDim.x) {
        q[i] = __bfloat162float(index_q[static_cast<std::size_t>(t) * index_heads * kIndexDim + i]);
    }
    __syncthreads();
    const int sequence        = t / width;
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[sequence]) * table_stride;
    const int slot_width      = kIndexDim / ratio;
    float* column_scores      = scores + static_cast<std::size_t>(blockIdx.x) * score_stride;
    const float scale         = rsqrtf(static_cast<float>(kIndexDim));
    // Scores: one warp per block.
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    for (int b = warp; b < blocks; b += warps) {
        float key[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = lane + 32 * j;
            key[j] = __bfloat162float(pooled[pooled_offset(table, ratio * b + d / slot_width, slot_width,
                                                           d % slot_width)]);
        }
        float score = 0.0F;
        for (int h = 0; h < index_heads; ++h) {
            float dot = 0.0F;
#pragma unroll
            for (int j = 0; j < 4; ++j) { dot += q[h * kIndexDim + lane + 32 * j] * key[j]; }
            score += fmaxf(warp_sum(dot), 0.0F);
        }
        if (lane == 0) { column_scores[b] = score * scale; }
    }
    __syncthreads();
    // Radix select of the top_blocks-th largest key, eight bits at a time.
    if (threadIdx.x == 0) {
        prefix_shared    = 0;
        mask_shared      = 0;
        remaining_shared = top_blocks;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) { histogram[i] = 0; }
        __syncthreads();
        const std::uint32_t prefix = prefix_shared, mask = mask_shared;
        for (int b = threadIdx.x; b < blocks; b += blockDim.x) {
            const std::uint32_t key = sortable(column_scores[b]);
            if ((key & mask) == prefix) { atomicAdd(&histogram[(key >> shift) & 255U], 1U); }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int remaining = remaining_shared;
            for (int digit = 255; digit >= 0; --digit) {
                const int count = static_cast<int>(histogram[digit]);
                if (count >= remaining) {
                    prefix_shared = prefix | (static_cast<std::uint32_t>(digit) << shift);
                    mask_shared   = mask | (255U << shift);
                    break;
                }
                remaining -= count;
            }
            remaining_shared = remaining; // how many of the threshold value to take
        }
        __syncthreads();
    }
    const std::uint32_t threshold = prefix_shared;
    const int take_equal          = remaining_shared;
    // Ascending block order: everything above the threshold, and the first take_equal ties.
    int out = 0, equal_seen = 0;
    for (int begin = 0; begin < blocks; begin += blockDim.x) {
        const int b = begin + threadIdx.x;
        const std::uint32_t key = b < blocks ? sortable(column_scores[b]) : 0;
        const int equal = b < blocks && key == threshold ? 1 : 0;
        scan[threadIdx.x] = equal;
        __syncthreads();
        for (int step = 1; step < blockDim.x; step <<= 1) {
            const int v = threadIdx.x >= step ? scan[threadIdx.x - step] : 0;
            __syncthreads();
            scan[threadIdx.x] += v;
            __syncthreads();
        }
        const int equal_rank = equal_seen + scan[threadIdx.x] - equal;
        const int total_eq   = scan[blockDim.x - 1];
        __syncthreads();
        const int take = b < blocks && (key > threshold || (equal && equal_rank < take_equal)) ? 1 : 0;
        scan[threadIdx.x] = take;
        __syncthreads();
        for (int step = 1; step < blockDim.x; step <<= 1) {
            const int v = threadIdx.x >= step ? scan[threadIdx.x - step] : 0;
            __syncthreads();
            scan[threadIdx.x] += v;
            __syncthreads();
        }
        if (take) { selected[static_cast<std::size_t>(t) * top_blocks + out + scan[threadIdx.x] - 1] = b; }
        const int taken = scan[blockDim.x - 1];
        __syncthreads();
        out += taken;
        equal_seen += total_eq;
    }
    if (threadIdx.x == 0) { counts[t] = out; }
}

// ----------------------------------------------------------------------------------- attention

template <KvCacheStorage Storage>
struct KVReader;

template <>
struct KVReader<KvCacheStorage::BFloat16> {
    static constexpr bool kRotatedKeys = false;
    // Element d of the key / value of `token` for `head`.
    __device__ static float key(const PagedKVLayerView& kv, const std::int32_t* table, int head, int token, int d,
                                int kv_heads) {
        const int page = table[token >> kPagedKVPageShift];
        const std::int64_t i = static_cast<std::int64_t>(kHeadDim) * kPagedKVPageSize * (head + kv_heads * page) +
                               static_cast<std::int64_t>(kHeadDim) * (token & kPagedKVPageMask) + d;
        return __bfloat162float(static_cast<const bf16*>(kv.k_pages.data)[i]);
    }
    __device__ static float value(const PagedKVLayerView& kv, const std::int32_t* table, int head, int token,
                                  int d, int kv_heads) {
        const int page = table[token >> kPagedKVPageShift];
        const std::int64_t i = static_cast<std::int64_t>(kHeadDim) * kPagedKVPageSize * (head + kv_heads * page) +
                               static_cast<std::int64_t>(kHeadDim) * (token & kPagedKVPageMask) + d;
        return __half2float(static_cast<const __half*>(kv.v_pages.data)[i]);
    }
};

template <>
struct KVReader<KvCacheStorage::Int8Group64> {
    static constexpr bool kRotatedKeys = true;
    __device__ static float read(const void* codes, const void* scales, const std::int32_t* table, int head,
                                 int token, int d, int kv_heads) {
        const int page = table[token >> kPagedKVPageShift];
        const int off  = token & kPagedKVPageMask;
        const std::int64_t ci = static_cast<std::int64_t>(kHeadDim) * kPagedKVPageSize * (head + kv_heads * page) +
                                static_cast<std::int64_t>(kHeadDim) * off + d;
        const std::int64_t si = static_cast<std::int64_t>(4) * kPagedKVPageSize * (head + kv_heads * page) +
                                static_cast<std::int64_t>(4) * off + d / 64;
        return static_cast<float>(static_cast<const std::int8_t*>(codes)[ci]) *
               __half2float(static_cast<const __half*>(scales)[si]);
    }
    __device__ static float key(const PagedKVLayerView& kv, const std::int32_t* table, int head, int token, int d,
                                int kv_heads) {
        return read(kv.k_pages.data, kv.k_scale_pages.data, table, head, token, d, kv_heads);
    }
    __device__ static float value(const PagedKVLayerView& kv, const std::int32_t* table, int head, int token,
                                  int d, int kv_heads) {
        return read(kv.v_pages.data, kv.v_scale_pages.data, table, head, token, d, kv_heads);
    }
};

// The attended token at index i of column t's list.
__device__ __forceinline__ int attended_token(int i, int count, const std::int32_t* selected, int ratio, int p) {
    if (count < 0) { return i; } // dense: tokens 0..p
    const int block_tokens = count * ratio;
    if (i < block_tokens) { return selected[i / ratio] * ratio + i % ratio; }
    return (p + 1) / ratio * ratio + (i - block_tokens);
}

__device__ __forceinline__ int attended_count(int count, int ratio, int p) {
    return count < 0 ? p + 1 : count * ratio + (p + 1) % ratio;
}

// One CTA per (split, KV head, column): the column's query heads of this KV head over one slice of
// its attended tokens. Partials (max, sum, unnormalized output) go to workspace when split > 1.
template <KvCacheStorage Storage>
__global__ void __launch_bounds__(256)
    attention_kernel(const bf16* __restrict__ q, int heads, int kv_heads, PagedKVLayerView kv,
                     const std::int32_t* __restrict__ tables, int table_stride,
                     const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                     int width, int ratio, const std::int32_t* __restrict__ selected,
                     const std::int32_t* __restrict__ counts, int top_blocks, int splits, float scale,
                     float* __restrict__ partial, bf16* __restrict__ out) {
    using Reader = KVReader<Storage>;
    const int split = blockIdx.x, kv_head = blockIdx.y, t = blockIdx.z;
    const int group = heads / kv_heads;
    const int p = positions[t];
    const int count = counts[t];
    const int total = attended_count(count, ratio, p);
    const int per_split = (total + splits - 1) / splits;
    const int begin = split * per_split, end = min(total, begin + per_split);
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t / width]) * table_stride;
    const std::int32_t* list  = selected + static_cast<std::size_t>(t) * top_blocks;

    __shared__ float qs[kMaxGroup][kHeadDim];
    __shared__ float probs[kMaxGroup][kTokenTile];
    __shared__ float m_run[kMaxGroup], l_run[kMaxGroup], rescale[kMaxGroup];
    __shared__ int tokens[kTokenTile];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int i = threadIdx.x; i < group * kHeadDim; i += blockDim.x) {
        const int h = i / kHeadDim, d = i % kHeadDim;
        qs[h][d] = __bfloat162float(q[(static_cast<std::size_t>(t) * heads + kv_head * group + h) * kHeadDim + d]);
    }
    __syncthreads();
    if constexpr (Reader::kRotatedKeys) {
        // Keys are stored in the normalized Hadamard domain; rotate each query head the same way.
        if (warp < group) {
            float v[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { v[r] = qs[warp][lane + 32 * r]; }
            normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
            for (int r = 0; r < 8; ++r) { qs[warp][lane + 32 * r] = v[r]; }
        }
        for (int h = warp + 8; h < group; h += 8) {
            float v[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { v[r] = qs[h][lane + 32 * r]; }
            normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
            for (int r = 0; r < 8; ++r) { qs[h][lane + 32 * r] = v[r]; }
        }
        __syncthreads();
    }
    if (threadIdx.x < group) {
        m_run[threadIdx.x] = -INFINITY;
        l_run[threadIdx.x] = 0.0F;
    }
    float acc[kMaxGroup];
#pragma unroll
    for (int h = 0; h < kMaxGroup; ++h) { acc[h] = 0.0F; }
    const int d_own = threadIdx.x; // this thread's output dimension
    __syncthreads();
    for (int tile = begin; tile < end; tile += kTokenTile) {
        const int n = min(kTokenTile, end - tile);
        if (threadIdx.x < n) { tokens[threadIdx.x] = attended_token(tile + threadIdx.x, count, list, ratio, p); }
        __syncthreads();
        // Scores: one warp per token, all query heads of the group.
        for (int j = warp; j < n; j += 8) {
            const int token = tokens[j];
            float k[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { k[r] = Reader::key(kv, table, kv_head, token, lane + 32 * r, kv_heads); }
            for (int h = 0; h < group; ++h) {
                float dot = 0.0F;
#pragma unroll
                for (int r = 0; r < 8; ++r) { dot += qs[h][lane + 32 * r] * k[r]; }
                dot = warp_sum(dot);
                if (lane == 0) { probs[h][j] = dot * scale; }
            }
        }
        __syncthreads();
        // Online softmax per head.
        if (threadIdx.x < group) {
            const int h = threadIdx.x;
            float m = m_run[h];
            for (int j = 0; j < n; ++j) { m = fmaxf(m, probs[h][j]); }
            const float r = expf(m_run[h] - m);
            float l = l_run[h] * r;
            for (int j = 0; j < n; ++j) {
                const float e = expf(probs[h][j] - m);
                probs[h][j]   = e;
                l += e;
            }
            m_run[h]   = m;
            l_run[h]   = l;
            rescale[h] = r;
        }
        __syncthreads();
        for (int h = 0; h < group; ++h) { acc[h] *= rescale[h]; }
        for (int j = 0; j < n; ++j) {
            const float v = Reader::value(kv, table, kv_head, tokens[j], d_own, kv_heads);
            for (int h = 0; h < group; ++h) { acc[h] += probs[h][j] * v; }
        }
        __syncthreads();
    }
    for (int h = 0; h < group; ++h) {
        const int head = kv_head * group + h;
        if (splits == 1) {
            const float l = l_run[h];
            out[(static_cast<std::size_t>(t) * heads + head) * kHeadDim + d_own] =
                __float2bfloat16_rn(l > 0.0F ? acc[h] / l : 0.0F);
        } else {
            float* slot = partial + ((static_cast<std::size_t>(t) * splits + split) * heads + head) * (kHeadDim + 2);
            slot[d_own] = acc[h];
            if (d_own == 0) {
                slot[kHeadDim]     = m_run[h];
                slot[kHeadDim + 1] = l_run[h];
            }
        }
    }
}

__global__ void merge_kernel(const float* __restrict__ partial, int heads, int splits, bf16* __restrict__ out) {
    const int head = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        m = fmaxf(m, partial[((static_cast<std::size_t>(t) * splits + s) * heads + head) * (kHeadDim + 2) + kHeadDim]);
    }
    float l = 0.0F, acc = 0.0F;
    for (int s = 0; s < splits; ++s) {
        const float* slot = partial + ((static_cast<std::size_t>(t) * splits + s) * heads + head) * (kHeadDim + 2);
        const float ms = slot[kHeadDim];
        if (ms == -INFINITY) { continue; }
        const float w = expf(ms - m);
        l += w * slot[kHeadDim + 1];
        acc += w * slot[d];
    }
    out[(static_cast<std::size_t>(t) * heads + head) * kHeadDim + d] = __float2bfloat16_rn(l > 0.0F ? acc / l : 0.0F);
}

int attention_splits(int columns, int kv_heads) {
    return std::clamp(340 / std::max(1, columns * kv_heads), 1, 33);
}

struct Workspace {
    std::int32_t* selected;
    std::int32_t* counts;
    float* scores;
    float* partial;
};

Workspace carve(void* base, const QsaGeometry& g, int columns, int max_context, std::size_t& bytes) {
    auto align = [](std::size_t v) { return (v + 255) / 256 * 256; };
    const int top_blocks = g.budget / g.ratio;
    const std::size_t selected = align(sizeof(std::int32_t) * top_blocks * static_cast<std::size_t>(columns));
    const std::size_t counts   = align(sizeof(std::int32_t) * static_cast<std::size_t>(columns));
    const std::size_t scores   = align(sizeof(float) * kSelectGroup * static_cast<std::size_t>(max_context / g.ratio + 1));
    const int splits           = attention_splits(columns, g.kv_heads);
    const std::size_t partial  = splits > 1 ? align(sizeof(float) * static_cast<std::size_t>(columns) * splits *
                                                    g.heads * (kHeadDim + 2))
                                            : 0;
    bytes = selected + counts + scores + partial;
    auto* p = static_cast<unsigned char*>(base);
    Workspace out{};
    if (p != nullptr) {
        out.selected = reinterpret_cast<std::int32_t*>(p);
        out.counts   = reinterpret_cast<std::int32_t*>(p + selected);
        out.scores   = reinterpret_cast<float*>(p + selected + counts);
        out.partial  = partial ? reinterpret_cast<float*>(p + selected + counts + scores) : nullptr;
    }
    return out;
}

void require_geometry(const QsaGeometry& g) {
    require(g.head_dim == kHeadDim && g.index_head_dim == kIndexDim && g.kv_heads > 0 && g.heads % g.kv_heads == 0 &&
                g.heads / g.kv_heads <= kMaxGroup && g.index_heads > 0 && g.index_heads <= 4 && g.ratio > 1 &&
                kIndexDim % g.ratio == 0 && g.budget % g.ratio == 0 && g.rotary_dim > 0 &&
                g.rotary_dim <= kIndexDim && g.rotary_dim % 2 == 0 && g.theta > 0 && g.eps > 0,
            "unsupported geometry");
}

void require_batch(const QsaBatch& b, int columns) {
    require(b.batch > 0 && b.width > 0 && b.batch * b.width == columns && contiguous(b.positions, DType::I32) &&
                b.positions.numel() == columns && contiguous(b.table_rows, DType::I32) &&
                b.table_rows.numel() == b.batch && contiguous(b.block_tables, DType::I32) &&
                contiguous(b.tail_slots, DType::I32) && b.tail_slots.numel() == b.batch,
            "batch description is invalid");
}

} // namespace

void qsa_index_query(Tensor& q, const Tensor& norm_weight, const Tensor& positions,
                     const QsaGeometry& geometry, cudaStream_t stream) {
    require_geometry(geometry);
    require(contiguous(q, DType::BF16) && contiguous(norm_weight, DType::BF16) && contiguous(positions, DType::I32),
            "index query needs BF16 q, BF16 weight and I32 positions");
    const int columns = q.ne[2];
    require(q.ne[0] == kIndexDim && q.ne[1] == geometry.index_heads && positions.numel() == columns &&
                norm_weight.numel() == kIndexDim && columns > 0,
            "index query shapes disagree");
    index_query_kernel<<<dim3(geometry.index_heads, columns), kIndexDim, 0, stream>>>(
        static_cast<bf16*>(q.data), static_cast<const bf16*>(norm_weight.data),
        static_cast<const std::int32_t*>(positions.data), geometry.index_heads, geometry.rotary_dim, geometry.theta,
        geometry.eps);
    check_launch("index query");
}

void qsa_pool_keys(const Tensor& raw_keys, const Tensor& norm_weight, Tensor& tails,
                   const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& geometry,
                   cudaStream_t stream) {
    require_geometry(geometry);
    const int columns = raw_keys.ne[1];
    require_batch(batch, columns);
    require(contiguous(raw_keys, DType::BF16) && raw_keys.ne[0] == kIndexDim && contiguous(tails, DType::BF16) &&
                tails.ne[0] == kIndexDim && tails.ne[1] == geometry.ratio - 1 &&
                contiguous(layer.pooled_pages, DType::BF16) &&
                layer.pooled_pages.ne[0] == kIndexDim / geometry.ratio && contiguous(norm_weight, DType::BF16),
            "pool shapes disagree");
    pool_kernel<<<columns, kIndexDim, 0, stream>>>(
        static_cast<const bf16*>(raw_keys.data), static_cast<const bf16*>(norm_weight.data),
        static_cast<const bf16*>(tails.data), static_cast<const std::int32_t*>(batch.block_tables.data),
        batch.block_tables.ne[0], static_cast<const std::int32_t*>(batch.table_rows.data),
        static_cast<const std::int32_t*>(batch.positions.data), static_cast<const std::int32_t*>(batch.tail_slots.data),
        batch.width, geometry.ratio, geometry.rotary_dim, geometry.theta, geometry.eps,
        static_cast<bf16*>(layer.pooled_pages.data));
    check_launch("pool");
    tail_kernel<<<batch.batch, kIndexDim, 0, stream>>>(
        static_cast<const bf16*>(raw_keys.data), static_cast<const std::int32_t*>(batch.positions.data),
        static_cast<const std::int32_t*>(batch.tail_slots.data), batch.width, geometry.ratio,
        static_cast<bf16*>(tails.data));
    check_launch("tail");
}

std::size_t qsa_attention_workspace_bytes(const QsaGeometry& geometry, std::int32_t columns,
                                          std::int32_t max_context) {
    require_geometry(geometry);
    std::size_t bytes = 0;
    (void)carve(nullptr, geometry, columns, max_context, bytes);
    return bytes;
}

void qsa_attention(const Tensor& q, const Tensor& index_q, const QsaKVLayer& layer,
                   const QsaBatch& batch, const QsaGeometry& geometry, float scale,
                   std::int32_t max_context, void* workspace, std::size_t workspace_bytes,
                   Tensor& out, cudaStream_t stream) {
    require_geometry(geometry);
    const int columns = q.ne[2];
    require_batch(batch, columns);
    require(contiguous(q, DType::BF16) && q.ne[0] == kHeadDim && q.ne[1] == geometry.heads &&
                contiguous(index_q, DType::BF16) && index_q.ne[0] == kIndexDim &&
                index_q.ne[1] == geometry.index_heads && index_q.ne[2] == columns && contiguous(out, DType::BF16) &&
                out.ne[0] == kHeadDim && out.ne[1] == geometry.heads && out.ne[2] == columns,
            "attention shapes disagree");
    require(layer.kv.num_kv_heads == geometry.kv_heads && layer.kv.head_dim == kHeadDim, "KV geometry differs");
    std::size_t need = 0;
    Workspace ws     = carve(workspace, geometry, columns, max_context, need);
    require(workspace != nullptr && workspace_bytes >= need, "workspace is too small");
    const int top_blocks = geometry.budget / geometry.ratio;
    const auto* tables   = static_cast<const std::int32_t*>(batch.block_tables.data);
    const int stride     = batch.block_tables.ne[0];
    const auto* rows     = static_cast<const std::int32_t*>(batch.table_rows.data);
    const auto* pos      = static_cast<const std::int32_t*>(batch.positions.data);
    for (int begin = 0; begin < columns; begin += kSelectGroup) {
        const int n = std::min(kSelectGroup, columns - begin);
        select_kernel<<<n, kSelectThreads, 0, stream>>>(
            static_cast<const bf16*>(index_q.data), geometry.index_heads,
            static_cast<const bf16*>(layer.pooled_pages.data), tables, stride, rows, pos, batch.width,
            geometry.ratio, top_blocks, begin, ws.scores, max_context / geometry.ratio + 1, ws.selected, ws.counts);
        check_launch("select");
    }
    const int splits = attention_splits(columns, geometry.kv_heads);
    const dim3 grid(splits, geometry.kv_heads, columns);
    switch (layer.kv.storage) {
    case KvCacheStorage::BFloat16:
        attention_kernel<KvCacheStorage::BFloat16><<<grid, 256, 0, stream>>>(
            static_cast<const bf16*>(q.data), geometry.heads, geometry.kv_heads, layer.kv, tables, stride, rows, pos,
            batch.width, geometry.ratio, ws.selected, ws.counts, top_blocks, splits, scale, ws.partial,
            static_cast<bf16*>(out.data));
        break;
    case KvCacheStorage::Int8Group64:
        attention_kernel<KvCacheStorage::Int8Group64><<<grid, 256, 0, stream>>>(
            static_cast<const bf16*>(q.data), geometry.heads, geometry.kv_heads, layer.kv, tables, stride, rows, pos,
            batch.width, geometry.ratio, ws.selected, ws.counts, top_blocks, splits, scale, ws.partial,
            static_cast<bf16*>(out.data));
        break;
    default:
        throw std::invalid_argument("qsa: this KV storage is not implemented yet");
    }
    check_launch("attention");
    if (splits > 1) {
        merge_kernel<<<dim3(geometry.heads, columns), kHeadDim, 0, stream>>>(ws.partial, geometry.heads, splits,
                                                                            static_cast<bf16*>(out.data));
        check_launch("merge");
    }
}

} // namespace ninfer::ops
