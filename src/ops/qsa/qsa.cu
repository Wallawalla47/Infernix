// QSA indexer, pooled keys, block selection and block-sparse attention
// (docs/maintainer/qwen3_8-flash-next-design.md §8.10). KV storages: bf16 and int8 (INT8-G64 with
// Hadamard-rotated keys, as written by kv_cache_append).

#include "ninfer/ops/qsa.h"

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/qsa/qsa_select.h"

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

// Half-split RoPE angle of pair i (i < rotary/2) at the token whose axis a of its RoPE position is
// rope[a * stride], in FP32 like the reference. Interleaved M-RoPE: pair i takes axis i % 3, so a
// text token (three equal axes) rotates exactly as by its 1-D position.
__device__ __forceinline__ void rope_angle(const std::int32_t* rope, std::int64_t stride, int i, int rotary,
                                           float theta, float& c, float& s) {
    const float inv   = 1.0F / powf(theta, static_cast<float>(2 * i) / static_cast<float>(rotary));
    const float angle = static_cast<float>(rope[(i % 3) * stride]) * inv;
    sincosf(angle, &s, &c);
}

// ------------------------------------------------------------------------------- index query

// One CTA (128 threads) per (index head, column).
__global__ void index_query_kernel(bf16* __restrict__ q, const bf16* __restrict__ weight,
                                   const std::int32_t* __restrict__ rope, int heads, int rotary,
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
        rope_angle(rope + t, gridDim.y, i, rotary, theta, c, s);
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
                            const std::int32_t* __restrict__ tail_slots,
                            const std::int32_t* __restrict__ rope, const std::int32_t* __restrict__ block_rope,
                            int batch, int width, int ratio, int rotary, float theta, float eps,
                            bf16* __restrict__ pooled) {
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
        // The block's first token: a column of this call, or the sequence's block start before it.
        if (first >= start) {
            rope_angle(rope + (t - (p - first)), gridDim.x, i, rotary, theta, c, s);
        } else {
            rope_angle(block_rope + sequence, batch, i, rotary, theta, c, s);
        }
        out = d < half ? normed[d] * c - normed[d + half] * s : normed[d] * c + normed[d - half] * s;
    }
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[sequence]) * table_stride;
    const int slot_width      = kIndexDim / ratio;
    const int token           = first + d / slot_width;
    pooled[pooled_offset(table, token, slot_width, d % slot_width)] = __float2bfloat16_rn(out);
}

// One CTA per sequence: the raw keys of the latest block's positions q % R < R - 1, up to the
// call's last position. A block the call completes keeps them too: a later call that rewrites its
// last position alone (the MTP drafter's prepend and draft step 0) re-pools it from the tails.
__global__ void tail_kernel(const bf16* __restrict__ raw, const std::int32_t* __restrict__ positions,
                            const std::int32_t* __restrict__ tail_slots, int width, int ratio,
                            bf16* __restrict__ tails) {
    const int sequence = blockIdx.x, d = threadIdx.x;
    const int start    = positions[sequence * width];
    const int last     = positions[sequence * width + width - 1];
    const int base     = last - last % ratio;
    const int end      = min(last, base + ratio - 2);
    for (int q = max(base, start); q <= end; ++q) { // positions before the call are in the tail already
        const int column = sequence * width + (q - start);
        tails[(static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) + (q % ratio)) * kIndexDim + d] =
            raw[static_cast<std::size_t>(column) * kIndexDim + d];
    }
}

// One CTA per (sequence, layer): the tail after the first commit_columns[sequence] columns.
__global__ void commit_tail_kernel(const bf16* __restrict__ raw, const std::int32_t* __restrict__ positions,
                                   const std::int32_t* __restrict__ commit_columns,
                                   const std::int32_t* __restrict__ tail_slots, int width, int batch, int ratio,
                                   int slots, bf16* __restrict__ tails) {
    const int sequence = blockIdx.x, layer = blockIdx.y, d = threadIdx.x;
    const int n = commit_columns[sequence];
    if (n <= 0) { return; }
    const int start = positions[sequence * width];
    const int last  = start + n - 1;
    const int base  = last - last % ratio;
    const int end   = min(last, base + ratio - 2);
    const bf16* layer_raw = raw + static_cast<std::size_t>(layer) * batch * width * kIndexDim;
    bf16* layer_tails     = tails + static_cast<std::size_t>(layer) * slots * (ratio - 1) * kIndexDim;
    for (int q = max(base, start); q <= end; ++q) {
        const int column = sequence * width + (q - start);
        layer_tails[(static_cast<std::size_t>(tail_slots[sequence]) * (ratio - 1) + (q % ratio)) * kIndexDim + d] =
            layer_raw[static_cast<std::size_t>(column) * kIndexDim + d];
    }
}

// ----------------------------------------------------------------------------------- selection
// Block selection (scores, top-k, ascending ids) is qsa_select (qsa_select.cu).

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

// Score sums of one token for every query head of a group: part[h] is this lane's partial dot
// product for head h. A reduce-scatter in warp_sum's xor order (16, 8, then 4, 2, 1 over the
// remaining quarter): at each step a lane adds its partner's value of a head to its own, exactly
// as warp_sum's `v += shfl_xor(v, o)`, so head h's sum has warp_sum's bits; it ends in the lanes
// whose bits 4 and 3 select its quarter. Those lanes with bits 0-2 clear write probs[h][j].
template <int Group>
__device__ __forceinline__ void scatter_head_sums(const float (&part)[Group], int group, int lane, float scale,
                                                  float (*probs)[kTokenTile], int j) {
    static_assert(Group % 4 == 0, "the reduce-scatter halves the heads twice");
    constexpr int kHalf = Group / 2, kQuarter = Group / 4;
    const bool upper16 = (lane & 16) != 0, upper8 = (lane & 8) != 0;
    float half[kHalf];
#pragma unroll
    for (int i = 0; i < kHalf; ++i) {
        const float mine = upper16 ? part[i + kHalf] : part[i];
        const float send = upper16 ? part[i] : part[i + kHalf];
        half[i]          = mine + __shfl_xor_sync(0xFFFFFFFFU, send, 16);
    }
    float quarter[kQuarter];
#pragma unroll
    for (int i = 0; i < kQuarter; ++i) {
        const float mine = upper8 ? half[i + kQuarter] : half[i];
        const float send = upper8 ? half[i] : half[i + kQuarter];
        quarter[i]       = mine + __shfl_xor_sync(0xFFFFFFFFU, send, 8);
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
#pragma unroll
        for (int i = 0; i < kQuarter; ++i) { quarter[i] += __shfl_xor_sync(0xFFFFFFFFU, quarter[i], o); }
    }
    if ((lane & 7) == 0) {
        const int first = (upper16 ? kHalf : 0) + (upper8 ? kQuarter : 0);
#pragma unroll
        for (int i = 0; i < kQuarter; ++i) {
            if (first + i < group) { probs[first + i][j] = quarter[i] * scale; }
        }
    }
}

// Value cells whose loads one thread issues together before folding them in order.
constexpr int kValueBatch = 8;

// One CTA per (split, KV head, column): the column's query heads of this KV head over one slice of
// its attended tokens. Partials (max, sum, unnormalized output) go to workspace when split > 1.
// `Group` bounds the query heads per KV head (group = heads / kv_heads <= Group), so every head
// loop has a compile-time trip count and the accumulators stay in registers. Each head's
// arithmetic (the score's fma order and butterfly, the serial online softmax, the in-order value
// fold) does not depend on Group.
template <KvCacheStorage Storage, int Group>
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

    __shared__ float qs[Group][kHeadDim];
    __shared__ float probs[Group][kTokenTile];
    __shared__ float m_run[Group], l_run[Group], rescale[Group];
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
    float acc[Group];
#pragma unroll
    for (int h = 0; h < Group; ++h) { acc[h] = 0.0F; }
    const int d_own = threadIdx.x; // this thread's output dimension
    __syncthreads();
    for (int tile = begin; tile < end; tile += kTokenTile) {
        const int n = min(kTokenTile, end - tile);
        if (threadIdx.x < n) { tokens[threadIdx.x] = attended_token(tile + threadIdx.x, count, list, ratio, p); }
        __syncthreads();
        // Scores: one warp per token, all query heads of the group; the next token's key loads are
        // issued before this token's sums.
        float k[8];
        if (warp < n) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { k[r] = Reader::key(kv, table, kv_head, tokens[warp], lane + 32 * r, kv_heads); }
        }
        for (int j = warp; j < n; j += 8) {
            float next[8];
            if (j + 8 < n) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    next[r] = Reader::key(kv, table, kv_head, tokens[j + 8], lane + 32 * r, kv_heads);
                }
            }
            float part[Group];
#pragma unroll
            for (int h = 0; h < Group; ++h) {
                float dot = 0.0F;
                if (h < group) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) { dot += qs[h][lane + 32 * r] * k[r]; }
                }
                part[h] = dot;
            }
            scatter_head_sums<Group>(part, group, lane, scale, probs, j);
#pragma unroll
            for (int r = 0; r < 8; ++r) { k[r] = next[r]; }
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
#pragma unroll
        for (int h = 0; h < Group; ++h) {
            if (h < group) { acc[h] *= rescale[h]; }
        }
        // Values: a batch of cells' loads in flight together, then folded in token order.
        for (int j0 = 0; j0 < n; j0 += kValueBatch) {
            float v[kValueBatch];
#pragma unroll
            for (int u = 0; u < kValueBatch; ++u) {
                if (j0 + u < n) { v[u] = Reader::value(kv, table, kv_head, tokens[j0 + u], d_own, kv_heads); }
            }
#pragma unroll
            for (int u = 0; u < kValueBatch; ++u) {
                if (j0 + u < n) {
#pragma unroll
                    for (int h = 0; h < Group; ++h) {
                        if (h < group) { acc[h] += probs[h][j0 + u] * v[u]; }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = 0; h < Group; ++h) {
        if (h >= group) { continue; }
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
    void* select;             // qsa_select scratch
    std::size_t select_bytes;
    float* partial;
};

Workspace carve(void* base, const QsaGeometry& g, int columns, int max_context, std::size_t& bytes) {
    auto align = [](std::size_t v) { return (v + 255) / 256 * 256; };
    const int top_blocks = g.budget / g.ratio;
    const std::size_t selected = align(sizeof(std::int32_t) * top_blocks * static_cast<std::size_t>(columns));
    const std::size_t counts   = align(sizeof(std::int32_t) * static_cast<std::size_t>(columns));
    const std::size_t select   = align(detail::qsa_select_scratch_bytes(g, columns, max_context));
    const int splits           = attention_splits(columns, g.kv_heads);
    const std::size_t partial  = splits > 1 ? align(sizeof(float) * static_cast<std::size_t>(columns) * splits *
                                                    g.heads * (kHeadDim + 2))
                                            : 0;
    bytes = selected + counts + select + partial;
    auto* p = static_cast<unsigned char*>(base);
    Workspace out{};
    if (p != nullptr) {
        out.selected     = reinterpret_cast<std::int32_t*>(p);
        out.counts       = reinterpret_cast<std::int32_t*>(p + selected);
        out.select       = p + selected + counts;
        out.select_bytes = select;
        out.partial      = partial ? reinterpret_cast<float*>(p + selected + counts + select) : nullptr;
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

void qsa_index_query(Tensor& q, const Tensor& norm_weight, const Tensor& rope_positions,
                     const QsaGeometry& geometry, cudaStream_t stream) {
    require_geometry(geometry);
    require(contiguous(q, DType::BF16) && contiguous(norm_weight, DType::BF16) &&
                contiguous(rope_positions, DType::I32),
            "index query needs BF16 q, BF16 weight and I32 RoPE positions");
    const int columns = q.ne[2];
    require(q.ne[0] == kIndexDim && q.ne[1] == geometry.index_heads && rope_positions.ne[0] == columns &&
                rope_positions.numel() == 3 * static_cast<std::int64_t>(columns) &&
                norm_weight.numel() == kIndexDim && columns > 0,
            "index query shapes disagree (RoPE positions are [T, 3])");
    index_query_kernel<<<dim3(geometry.index_heads, columns), kIndexDim, 0, stream>>>(
        static_cast<bf16*>(q.data), static_cast<const bf16*>(norm_weight.data),
        static_cast<const std::int32_t*>(rope_positions.data), geometry.index_heads, geometry.rotary_dim,
        geometry.theta, geometry.eps);
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
    require(contiguous(batch.rope_positions, DType::I32) && batch.rope_positions.ne[0] == columns &&
                batch.rope_positions.numel() == 3 * static_cast<std::int64_t>(columns) &&
                contiguous(batch.block_start_rope, DType::I32) && batch.block_start_rope.ne[0] == batch.batch &&
                batch.block_start_rope.numel() == 3 * static_cast<std::int64_t>(batch.batch),
            "pool RoPE positions are [T, 3] and block starts [batch, 3]");
    pool_kernel<<<columns, kIndexDim, 0, stream>>>(
        static_cast<const bf16*>(raw_keys.data), static_cast<const bf16*>(norm_weight.data),
        static_cast<const bf16*>(tails.data), static_cast<const std::int32_t*>(batch.block_tables.data),
        batch.block_tables.ne[0], static_cast<const std::int32_t*>(batch.table_rows.data),
        static_cast<const std::int32_t*>(batch.positions.data), static_cast<const std::int32_t*>(batch.tail_slots.data),
        static_cast<const std::int32_t*>(batch.rope_positions.data),
        static_cast<const std::int32_t*>(batch.block_start_rope.data), batch.batch, batch.width, geometry.ratio,
        geometry.rotary_dim, geometry.theta, geometry.eps, static_cast<bf16*>(layer.pooled_pages.data));
    check_launch("pool");
    if (!batch.update_tails) { return; }
    tail_kernel<<<batch.batch, kIndexDim, 0, stream>>>(
        static_cast<const bf16*>(raw_keys.data), static_cast<const std::int32_t*>(batch.positions.data),
        static_cast<const std::int32_t*>(batch.tail_slots.data), batch.width, geometry.ratio,
        static_cast<bf16*>(tails.data));
    check_launch("tail");
}

void qsa_commit_tails(const Tensor& raw_keys, const Tensor& positions, const Tensor& commit_columns,
                      Tensor& tails, const Tensor& tail_slots, const QsaGeometry& geometry,
                      cudaStream_t stream) {
    require_geometry(geometry);
    const int width = raw_keys.ne[1], batch = raw_keys.ne[2], layers = raw_keys.ne[3];
    require(contiguous(raw_keys, DType::BF16) && raw_keys.ne[0] == kIndexDim && width > 0 && batch > 0 &&
                layers > 0 && contiguous(positions, DType::I32) && positions.numel() == std::int64_t(width) * batch &&
                contiguous(commit_columns, DType::I32) && commit_columns.ne[0] == batch &&
                contiguous(tail_slots, DType::I32) && tail_slots.ne[0] == batch && contiguous(tails, DType::BF16) &&
                tails.ne[0] == kIndexDim && tails.ne[1] == geometry.ratio - 1 && tails.ne[3] == layers,
            "commit tail shapes disagree");
    commit_tail_kernel<<<dim3(batch, layers), kIndexDim, 0, stream>>>(
        static_cast<const bf16*>(raw_keys.data), static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(commit_columns.data), static_cast<const std::int32_t*>(tail_slots.data),
        width, batch, geometry.ratio, tails.ne[2], static_cast<bf16*>(tails.data));
    check_launch("commit tails");
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
    detail::qsa_select(index_q, layer.pooled_pages, batch, geometry, max_context, ws.select, ws.select_bytes,
                       {ws.selected, ws.counts}, stream);
    const int splits = attention_splits(columns, geometry.kv_heads);
    const dim3 grid(splits, geometry.kv_heads, columns);
    const auto launch = [&](auto kernel) {
        kernel<<<grid, 256, 0, stream>>>(static_cast<const bf16*>(q.data), geometry.heads, geometry.kv_heads, layer.kv,
                                         tables, stride, rows, pos, batch.width, geometry.ratio, ws.selected, ws.counts,
                                         top_blocks, splits, scale, ws.partial, static_cast<bf16*>(out.data));
    };
    // Query heads per KV head, rounded up to a multiple of four (Qwen4Exp: 24 / 2 = 12).
    const auto launch_storage = [&]<KvCacheStorage Storage>() {
        const int group = geometry.heads / geometry.kv_heads;
        if (group <= 4) {
            launch(attention_kernel<Storage, 4>);
        } else if (group <= 8) {
            launch(attention_kernel<Storage, 8>);
        } else if (group <= 12) {
            launch(attention_kernel<Storage, 12>);
        } else {
            launch(attention_kernel<Storage, kMaxGroup>);
        }
    };
    switch (layer.kv.storage) {
    case KvCacheStorage::BFloat16:
        launch_storage.template operator()<KvCacheStorage::BFloat16>();
        break;
    case KvCacheStorage::Int8Group64:
        launch_storage.template operator()<KvCacheStorage::Int8Group64>();
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
