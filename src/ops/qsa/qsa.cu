// QSA indexer, pooled keys, block selection and block-sparse attention
// (docs/maintainer/qwen3_8-flash-next-design.md §8.10). KV storages: bf16 and int8 (INT8-G64 with
// Hadamard-rotated keys, as written by kv_cache_append).

#include "ninfer/ops/qsa.h"

#include "core/paged_kv_storage.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/launch.h"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/kv_window.cuh"
#include "ops/kv_cache/q4_lloyd_codec.cuh"
#include "ops/kv_cache/vq2_codec.cuh"
#include "ops/qsa/page_spaces.cuh"
#include "ops/qsa/qsa_prompt.h"
#include "ops/qsa/qsa_select.h"
#include "ops/qsa/vq_window.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

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

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFU, v, o)); }
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

// The pooled-key element of `position` (its page's space resolved: the pool's plane or the spaces').
__device__ __forceinline__ bf16* pooled_element(bf16* pool, const QsaPageSpaces& spaces, const std::int32_t* table,
                                                int position, int slot_width, int lane) {
    int page;
    bf16* plane = detail::qsa_pooled_plane(spaces, pool, table[position >> kPagedKVPageShift], page);
    return plane + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
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
                            bf16* __restrict__ pooled, QsaPageSpaces spaces) {
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
    *pooled_element(pooled, spaces, table, token, slot_width, d % slot_width) = __float2bfloat16_rn(out);
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

// One stage of attended tokens in shared memory, gathered by cp.async from each token's page space:
// planes [K rows | V rows | K scales | V scales] of kTokenTile tokens each, in the paged planes' own
// byte layout (paged_kv_storage_layout). A storage's key() and value() decode cell (j, d) to
// exactly its represented value in FP32: every code times its stored scale is exact in FP32.
template <int KeyBytes, int ValueBytes, int KeyScaleBytes, int ValueScaleBytes>
struct StagePlanes {
    static constexpr int kKeyBytes        = KeyBytes;
    static constexpr int kValueBytes      = ValueBytes;
    static constexpr int kKeyScaleBytes   = KeyScaleBytes;
    static constexpr int kValueScaleBytes = ValueScaleBytes;
    static constexpr bool kVq             = false;
    static constexpr int kValues          = kTokenTile * KeyBytes;
    static constexpr int kKeyScales       = kValues + kTokenTile * ValueBytes;
    static constexpr int kValueScales     = kKeyScales + kTokenTile * KeyScaleBytes;
    static constexpr int kStageBytes      = kValueScales + kTokenTile * ValueScaleBytes;
    static_assert(KeyBytes % 16 == 0 && ValueBytes % 16 == 0 && kKeyScales % 16 == 0 && kValueScales % 16 == 0);
    __device__ static const unsigned char* key_row(const unsigned char* stage, int j) { return stage + j * KeyBytes; }
    __device__ static const unsigned char* value_row(const unsigned char* stage, int j) {
        return stage + kValues + j * ValueBytes;
    }
    __device__ static const unsigned char* key_scales(const unsigned char* stage, int j) {
        return stage + kKeyScales + j * KeyScaleBytes;
    }
    __device__ static const unsigned char* value_scales(const unsigned char* stage, int j) {
        return stage + kValueScales + j * ValueScaleBytes;
    }
};

__device__ __forceinline__ float fp16_at(const unsigned char* p, int i) {
    return __half2float(reinterpret_cast<const __half*>(p)[i]);
}

__device__ __forceinline__ float e4m3_at(const unsigned char* p, int i) {
    __nv_fp8_e4m3 v;
    v.__x = p[i];
    return static_cast<float>(v);
}

// E2M1 element d of a packed row (dimension 2j in the low nibble of byte j).
__device__ __forceinline__ float e2m1_at(const unsigned char* p, int d) {
    const unsigned code = (p[d >> 1] >> (4 * (d & 1))) & 0xFU;
    const unsigned e = (code >> 1) & 3U, m = code & 1U;
    const float magnitude = e == 0 ? 0.5F * static_cast<float>(m) : __uint_as_float(((e + 126U) << 23) | (m << 22));
    return (code & 8U) != 0 ? -magnitude : magnitude;
}

// FP8 E4M3 rows with one FP16 scale; NVFP4 rows (E2M1 codes) with one E4M3 scale per 16.
__device__ __forceinline__ float fp8_row_cell(const unsigned char* row, const unsigned char* scale, int d) {
    return e4m3_at(row, d) * fp16_at(scale, 0);
}
__device__ __forceinline__ float nvfp4_cell(const unsigned char* row, const unsigned char* scales, int d) {
    return e2m1_at(row, d) * e4m3_at(scales, d / 16);
}

template <KvCacheStorage Storage>
struct KVTile;

// BF16 keys, FP16 values.
template <>
struct KVTile<KvCacheStorage::BFloat16> : StagePlanes<kHeadDim * 2, kHeadDim * 2, 0, 0> {
    static constexpr bool kRotatedKeys = false, kRotatedValues = false;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return __bfloat162float(reinterpret_cast<const bf16*>(key_row(stage, j))[d]);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) { return fp16_at(value_row(stage, j), d); }
};

// INT8 codes with one FP16 scale per 64; keys rotated.
template <>
struct KVTile<KvCacheStorage::Int8Group64> : StagePlanes<kHeadDim, kHeadDim, 8, 8> {
    static constexpr bool kRotatedKeys = true, kRotatedValues = false;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(key_row(stage, j))[d]) *
               fp16_at(key_scales(stage, j), d / 64);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(value_row(stage, j))[d]) *
               fp16_at(value_scales(stage, j), d / 64);
    }
};

// FP8 E4M3 rows with an FP16 row scale; keys rotated.
template <>
struct KVTile<KvCacheStorage::Fp8E4M3Row256> : StagePlanes<kHeadDim, kHeadDim, 2, 2> {
    static constexpr bool kRotatedKeys = true, kRotatedValues = false;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return fp8_row_cell(key_row(stage, j), key_scales(stage, j), d);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return fp8_row_cell(value_row(stage, j), value_scales(stage, j), d);
    }
};

// NVFP4 keys and values, both rotated.
template <>
struct KVTile<KvCacheStorage::Nvfp4Group16> : StagePlanes<kHeadDim / 2, kHeadDim / 2, 16, 16> {
    static constexpr bool kRotatedKeys = true, kRotatedValues = true;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return nvfp4_cell(key_row(stage, j), key_scales(stage, j), d);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return nvfp4_cell(value_row(stage, j), value_scales(stage, j), d);
    }
};

// FP8 keys, NVFP4 values, both rotated.
template <>
struct KVTile<KvCacheStorage::Fp8KeyNvfp4Value> : StagePlanes<kHeadDim, kHeadDim / 2, 2, 16> {
    static constexpr bool kRotatedKeys = true, kRotatedValues = true;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return fp8_row_cell(key_row(stage, j), key_scales(stage, j), d);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return nvfp4_cell(value_row(stage, j), value_scales(stage, j), d);
    }
};

// Vector-quantized storages (vq2, k4v2; ops/kv_cache/vq2_codec.cuh, q4_lloyd_codec.cuh): every tile
// token's K and V become INT8-G64 rows of the rotated vectors in the INT8 tile's planes, so scores and
// values then read exactly like INT8. A row is its exact window row (a window slot whose tag matches
// the stored codes, or the call's staged row) when the query reads the key exactly
// (vq_exact_key), and its codes decoded otherwise: codebook entry c (VQ2) or level c (Q4) with the
// row scale S as all four group scales, S * c exactly. Behind the INT8 planes each stage holds the
// paged code rows, their FP16 row scales, the window tags and one keep flag per row.
enum class VqKey : unsigned char { Vq2, Q4 };

template <VqKey Key>
struct VqTile : StagePlanes<kHeadDim, kHeadDim, 8, 8> {
    using Base                              = StagePlanes<kHeadDim, kHeadDim, 8, 8>;
    static constexpr bool kVq               = true;
    static constexpr bool kRotatedKeys      = true, kRotatedValues = true;
    static constexpr VqKey kKey             = Key;
    static constexpr int kKeyCodeBytes      = Key == VqKey::Vq2 ? kKVCacheVq2CodeBytes : kKVCacheQ4CodeBytes;
    static constexpr int kValueCodeBytes    = kKVCacheVq2CodeBytes;
    static constexpr int kKeyCodes          = Base::kStageBytes;
    static constexpr int kValueCodes        = kKeyCodes + kTokenTile * kKeyCodeBytes;
    static constexpr int kCodeScales        = kValueCodes + kTokenTile * kValueCodeBytes; // [token][role] FP16
    static constexpr int kTags              = kCodeScales + kTokenTile * 4;               // [token][role] I32
    static constexpr int kKeep              = kTags + kTokenTile * 8;                     // [token][role] u8
    static constexpr int kStageBytes        = kKeep + kTokenTile * 2;
    __device__ static float key(const unsigned char* stage, int j, int d) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(key_row(stage, j))[d]) *
               fp16_at(key_scales(stage, j), d / 64);
    }
    __device__ static float value(const unsigned char* stage, int j, int d) {
        return static_cast<float>(reinterpret_cast<const std::int8_t*>(value_row(stage, j))[d]) *
               fp16_at(value_scales(stage, j), d / 64);
    }
};

template <>
struct KVTile<KvCacheStorage::Vq2> : VqTile<VqKey::Vq2> {};
template <>
struct KVTile<KvCacheStorage::Q4KeyVq2Value> : VqTile<VqKey::Q4> {};

// Copies one scale row of `Bytes` (2: a plain load, which cp.async cannot do; 8 or 16: cp.async).
template <int Bytes>
__device__ __forceinline__ void copy_scales(unsigned char* dst, const unsigned char* src) {
    if constexpr (Bytes == 2) {
        *reinterpret_cast<__half*>(dst) = *reinterpret_cast<const __half*>(src);
    } else {
        cp_async<Bytes>(dst, src);
    }
}

// Dynamic shared memory of attention_kernel<Storage, Group, Stages>, in bytes and offsets: Stages
// tile stages, the rotated query heads, the tile's probabilities, the running statistics and one
// token/page-id slot per stage.
template <KvCacheStorage Storage, int Group, int Stages>
struct AttentionShared {
    using Tile                           = KVTile<Storage>;
    static constexpr std::size_t kStage  = Tile::kStageBytes;
    static constexpr std::size_t kQ      = Stages * kStage;
    static constexpr std::size_t kProbs  = kQ + sizeof(float) * Group * kHeadDim;
    static constexpr std::size_t kStats  = kProbs + sizeof(float) * Group * kTokenTile;
    static constexpr std::size_t kTokens = kStats + sizeof(float) * 4 * ((3 * Group + 3) / 4);
    static constexpr std::size_t kCodebook = kTokens + sizeof(std::int32_t) * 2 * Stages * kTokenTile;
    static constexpr std::size_t kBytes    = kCodebook + (Tile::kVq ? sizeof(kKVCacheVq2Codebook) : 0);
    static_assert(kStage % 16 == 0 && kBytes <= 99 * 1024, "one CTA's shared memory on sm_120");
};

// Threads of attention_kernel: one warp per token in the score phase, and two threads per output
// dimension in the value phase, each folding half of the group's heads.
constexpr int kAttentionThreads = 512;
constexpr int kAttentionWarps   = kAttentionThreads / 32;

// One CTA per (split, KV head, column): the column's query heads of this KV head over one slice of
// its attended tokens. Partials (max, sum, unnormalized output) go to workspace when split > 1.
// `Group` bounds the query heads per KV head (group = heads / kv_heads <= Group; `Exact` when they
// are equal, so head loops need no bounds), so every head loop has a compile-time trip count and
// the accumulators stay in registers.
//
// Each 64-token tile's K/V rows are gathered into shared memory by cp.async in one round. With two
// stages the next tile streams under this one; one stage serves calls whose splits hold one tile. A
// lane reads its eight query values of a head as two 16-byte words ([head][r / 4][lane][r % 4] for
// dimension lane + 32 r) and the probabilities four tokens at a time. A head's arithmetic is the
// reference order and does not depend on Group, the staging or the thread mapping: the score's fma
// order over r and the butterfly, the tile's online-softmax update (an exact maximum, then l summed
// serially in token order) and the in-order value fold, under the fixed split partition.
template <KvCacheStorage Storage, int Group, bool Exact, int Stages>
__global__ void __launch_bounds__(kAttentionThreads, 1)
    attention_kernel(const bf16* __restrict__ q, int heads, int kv_heads, PagedKVLayerView kv, QsaPageSpaces spaces,
                     const std::int32_t* __restrict__ tables, int table_stride,
                     const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                     int width, int ratio, const std::int32_t* __restrict__ selected,
                     const std::int32_t* __restrict__ counts, int top_blocks, int splits, float scale,
                     float* __restrict__ partial, bf16* __restrict__ out, detail::QsaVqWindow vq) {
    using Tile   = KVTile<Storage>;
    using Shared = AttentionShared<Storage, Group, Stages>;
    static_assert(Group % 2 == 0, "two threads per dimension split the heads");
    constexpr int Half = Group / 2;
    pdl::enter();
    const int split = blockIdx.x, kv_head = blockIdx.y, t = blockIdx.z;
    const int group = Exact ? Group : heads / kv_heads;
    const int p = positions[t];
    const int count = counts[t];
    const int total = attended_count(count, ratio, p);
    const int per_split = (total + splits - 1) / splits;
    const int begin = split * per_split, end = min(total, begin + per_split);
    const int tiles = begin < end ? (end - begin + kTokenTile - 1) / kTokenTile : 0;
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t / width]) * table_stride;
    const std::int32_t* list  = selected + static_cast<std::size_t>(t) * top_blocks;

    extern __shared__ __align__(16) unsigned char smem[];
    float* qs      = reinterpret_cast<float*>(smem + Shared::kQ); // [Group][2][32][4]
    auto* probs    = reinterpret_cast<float(*)[kTokenTile]>(smem + Shared::kProbs);
    float* m_run   = reinterpret_cast<float*>(smem + Shared::kStats);
    float* l_run   = m_run + Group;
    float* rescale = l_run + Group;
    auto* tokens   = reinterpret_cast<std::int32_t(*)[kTokenTile]>(smem + Shared::kTokens); // [Stages]
    auto* ids      = tokens + Stages;                                                       // [Stages]
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    // Query element (h, lane + 32 r) in the lane-contiguous layout.
    const auto q_at = [](int h, int lane_, int r) { return (h * 64 + (r >> 2) * 32 + lane_) * 4 + (r & 3); };

    // Tile i's tokens and their block-table page ids into slot `buf`.
    const auto prepare = [&](int i, int buf) {
        const int first = begin + i * kTokenTile;
        if (threadIdx.x < min(kTokenTile, end - first)) {
            const int token          = attended_token(first + threadIdx.x, count, list, ratio, p);
            tokens[buf][threadIdx.x] = token;
            ids[buf][threadIdx.x]    = table[token >> kPagedKVPageShift];
        }
    };
    // Vector-quantized storages: the paged code rows and their scales, and for every key the query
    // reads exactly its exact row (window slot with its tags, or the call's staged row).
    const int call_first = positions[(t / width) * width];
    const int window_row = vq.slots != nullptr ? vq.slots[t / width] : 0;
    const auto vq_issue  = [&](unsigned char* stage, int buf, int n) {
        if constexpr (Tile::kVq) {
            constexpr int KC = Tile::kKeyCodeBytes / 16, VC = Tile::kValueCodeBytes / 16;
            const auto row_at = [&](int j, int page) {
                return static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize +
                       (tokens[buf][j] & kPagedKVPageMask);
            };
            for (int c = threadIdx.x; c < n * (KC + VC); c += kAttentionThreads) {
                const int j = c / (KC + VC), r = c % (KC + VC);
                const bool is_key = r < KC;
                const int chunk   = is_key ? r : r - KC;
                int page;
                const void* base = is_key ? detail::qsa_space_plane(spaces, kv.k_pages.data, spaces.host_k,
                                                                    spaces.lent_k, ids[buf][j], page)
                                          : detail::qsa_space_plane(spaces, kv.v_pages.data, spaces.host_v,
                                                                    spaces.lent_v, ids[buf][j], page);
                const int bytes = is_key ? Tile::kKeyCodeBytes : Tile::kValueCodeBytes;
                cp_async<16, Cache::cg>(stage + (is_key ? Tile::kKeyCodes : Tile::kValueCodes) + j * bytes + 16 * chunk,
                                        static_cast<const unsigned char*>(base) + row_at(j, page) * bytes + 16 * chunk);
            }
            for (int c = threadIdx.x; c < n * 32; c += kAttentionThreads) {
                const int j = c / 32, role = (c / 16) % 2, chunk = c % 16;
                const int key  = tokens[buf][j];
                const int mode = detail::vq_read_mode(vq, key, p, call_first);
                if (mode == 0) { continue; }
                const std::int8_t* src =
                    mode == 2 ? (role ? vq.staged_v_codes : vq.staged_k_codes) +
                                    (static_cast<std::int64_t>(kv_head) * vq.staged_width + (key - call_first)) * kHeadDim
                              : (role ? vq.v_codes : vq.k_codes) +
                                    detail::vq_window_row(window_row, kv_head, kv_heads, kv_window_slot(key)) * kHeadDim;
                cp_async<16, Cache::cg>(stage + (role ? Tile::kValues : 0) + j * kHeadDim + 16 * chunk, src + 16 * chunk);
            }
            for (int c = threadIdx.x; c < n * 2; c += kAttentionThreads) {
                const int j = c / 2, role = c % 2;
                const int key = tokens[buf][j];
                int page;
                const void* base = role ? detail::qsa_space_plane(spaces, kv.v_scale_pages.data, spaces.host_v_scale,
                                                                  spaces.lent_v_scale, ids[buf][j], page)
                                        : detail::qsa_space_plane(spaces, kv.k_scale_pages.data, spaces.host_k_scale,
                                                                  spaces.lent_k_scale, ids[buf][j], page);
                copy_scales<2>(stage + Tile::kCodeScales + (2 * j + role) * 2,
                               static_cast<const unsigned char*>(base) + row_at(j, page) * 2);
                const int mode           = detail::vq_read_mode(vq, key, p, call_first);
                unsigned char* group_dst = stage + (role ? Tile::kValueScales : Tile::kKeyScales) + j * 8;
                if (mode == 2) {
                    const std::int64_t staged = static_cast<std::int64_t>(kv_head) * vq.staged_width + (key - call_first);
                    cp_async<8>(group_dst, (role ? vq.staged_v_scales : vq.staged_k_scales) + staged * kKVWindowGroups);
                } else if (mode == 1) {
                    const std::int64_t at = detail::vq_window_row(window_row, kv_head, kv_heads, kv_window_slot(key));
                    cp_async<8>(group_dst, (role ? vq.v_scales : vq.k_scales) + at * kKVWindowGroups);
                    cp_async<4>(stage + Tile::kTags + (2 * j + role) * 4, vq.tags + at * 2 + role);
                }
            }
        }
    };
    // VQ2 codebook for code decoding (in the first tile's copy group).
    const std::int8_t* codebook = reinterpret_cast<const std::int8_t*>(smem + Shared::kCodebook);
    if constexpr (Tile::kVq) {
        for (int i = threadIdx.x; i < static_cast<int>(sizeof(kKVCacheVq2Codebook)) / 16; i += kAttentionThreads) {
            cp_async<16>(smem + Shared::kCodebook + 16 * i, g_kv_cache_vq2_codebook + 16 * i);
        }
    }
    // Gathers the K/V rows and scales of tile i's tokens (slot `buf`) into stage s.
    const auto issue = [&](int i, int buf, int s) {
        unsigned char* stage = smem + s * Shared::kStage;
        const int n          = min(kTokenTile, end - (begin + i * kTokenTile));
        if constexpr (Tile::kVq) {
            vq_issue(stage, buf, n);
            cp_commit();
            return;
        }
        constexpr int KC = Tile::kKeyBytes / 16, VC = Tile::kValueBytes / 16;
        // Byte offset of token j's row of `bytes` in its page's plane.
        const auto offset = [&](int j, int page, int bytes) {
            return (static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize +
                    (tokens[buf][j] & kPagedKVPageMask)) *
                   bytes;
        };
        for (int c = threadIdx.x; c < n * (KC + VC); c += kAttentionThreads) {
            const int j = c / (KC + VC), r = c % (KC + VC);
            const bool is_key = r < KC;
            const int chunk   = is_key ? r : r - KC;
            int page;
            const void* base = is_key ? detail::qsa_space_plane(spaces, kv.k_pages.data, spaces.host_k, spaces.lent_k,
                                                                ids[buf][j], page)
                                      : detail::qsa_space_plane(spaces, kv.v_pages.data, spaces.host_v, spaces.lent_v,
                                                                ids[buf][j], page);
            unsigned char* dst = is_key ? stage + j * Tile::kKeyBytes : stage + Tile::kValues + j * Tile::kValueBytes;
            cp_async<16, Cache::cg>(dst + 16 * chunk, static_cast<const unsigned char*>(base) +
                                                          offset(j, page, is_key ? Tile::kKeyBytes : Tile::kValueBytes) +
                                                          16 * chunk);
        }
        if constexpr (Tile::kKeyScaleBytes > 0) {
            for (int j = threadIdx.x; j < n; j += kAttentionThreads) {
                int page;
                const void* base = detail::qsa_space_plane(spaces, kv.k_scale_pages.data, spaces.host_k_scale,
                                                           spaces.lent_k_scale, ids[buf][j], page);
                copy_scales<Tile::kKeyScaleBytes>(stage + Tile::kKeyScales + j * Tile::kKeyScaleBytes,
                                                  static_cast<const unsigned char*>(base) +
                                                      offset(j, page, Tile::kKeyScaleBytes));
            }
        }
        if constexpr (Tile::kValueScaleBytes > 0) {
            for (int j = threadIdx.x; j < n; j += kAttentionThreads) {
                int page;
                const void* base = detail::qsa_space_plane(spaces, kv.v_scale_pages.data, spaces.host_v_scale,
                                                           spaces.lent_v_scale, ids[buf][j], page);
                copy_scales<Tile::kValueScaleBytes>(stage + Tile::kValueScales + j * Tile::kValueScaleBytes,
                                                    static_cast<const unsigned char*>(base) +
                                                        offset(j, page, Tile::kValueScaleBytes));
            }
        }
        cp_commit();
    };

    // The first tile streams while the query heads load and rotate.
    if (tiles > 0) {
        prepare(0, 0);
        __syncthreads();
        issue(0, 0, 0);
    }
    // The query heads: one 16-byte load of eight elements per thread.
    for (int i = threadIdx.x; i < group * kHeadDim / 8; i += kAttentionThreads) {
        const int h = i / (kHeadDim / 8), d0 = 8 * (i % (kHeadDim / 8));
        const uint4 raw = *reinterpret_cast<const uint4*>(
            q + (static_cast<std::size_t>(t) * heads + kv_head * group + h) * kHeadDim + d0);
        const auto* values = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int d = d0 + e;
            qs[q_at(h, d & 31, d >> 5)] = __bfloat162float(values[e]);
        }
    }
    if (threadIdx.x < group) {
        m_run[threadIdx.x] = -INFINITY;
        l_run[threadIdx.x] = 0.0F;
    }
    __syncthreads();
    if constexpr (Tile::kRotatedKeys) {
        // Keys are stored in the normalized Hadamard domain; rotate each query head the same way.
        for (int h = warp; h < group; h += kAttentionWarps) {
            float v[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { v[r] = qs[q_at(h, lane, r)]; }
            normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
            for (int r = 0; r < 8; ++r) { qs[q_at(h, lane, r)] = v[r]; }
        }
        __syncthreads();
    }
    // Value phase: this thread's dimension and its half of the heads.
    const int d_own = threadIdx.x % kHeadDim, h_base = (threadIdx.x / kHeadDim) * Half;
    const auto mine = [&](int i) { return Exact || h_base + i < group; };
    float acc[Half];
#pragma unroll
    for (int i = 0; i < Half; ++i) { acc[i] = 0.0F; }
    for (int i = 0; i < tiles; ++i) {
        const int n = min(kTokenTile, end - (begin + i * kTokenTile));
        const int s = Stages == 2 ? i % 2 : 0;
        bool ahead  = false;
        if (Stages == 1 && i > 0) {
            prepare(i, 0);
            __syncthreads();
            issue(i, 0, 0);
        } else if (Stages == 2 && i + 1 < tiles) {
            prepare(i + 1, (i + 1) % 2);
            __syncthreads();
            issue(i + 1, (i + 1) % 2, 1 - s);
            ahead = true;
        }
        if (ahead) {
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        if constexpr (Tile::kVq) {
            // Keep a row's exact copy when its tag matches the stored codes (staged rows always), and
            // decode the codes of every other row into the INT8 planes, the row scale as each group's.
            unsigned char* st = smem + s * Shared::kStage;
            for (int c = threadIdx.x; c < n * 2; c += kAttentionThreads) {
                const int j = c / 2, role = c % 2;
                const int key  = tokens[s][j];
                const int mode = detail::vq_read_mode(vq, key, p, call_first);
                bool keep      = mode == 2;
                if (mode == 1) {
                    const std::uint32_t scale_bits =
                        reinterpret_cast<const std::uint16_t*>(st + Tile::kCodeScales)[2 * j + role];
                    std::uint32_t tag;
                    if (role == 0 && Tile::kKeyCodeBytes == 128) {
                        tag = kv_window_tag<32>(key, reinterpret_cast<const std::uint32_t*>(st + Tile::kKeyCodes + j * 128),
                                                scale_bits);
                    } else {
                        const unsigned char* codes =
                            role ? st + Tile::kValueCodes + j * 64 : st + Tile::kKeyCodes + j * Tile::kKeyCodeBytes;
                        tag = kv_window_tag<16>(key, reinterpret_cast<const std::uint32_t*>(codes), scale_bits);
                    }
                    keep = static_cast<std::uint32_t>(reinterpret_cast<const std::int32_t*>(st + Tile::kTags)[2 * j + role]) ==
                           tag;
                }
                st[Tile::kKeep + 2 * j + role] = keep ? 1 : 0;
            }
            __syncthreads();
            for (int c = threadIdx.x; c < n * 64; c += kAttentionThreads) {
                const int j = c / 64, role = (c / 32) % 2, w = c % 32;
                if (st[Tile::kKeep + 2 * j + role] != 0) { continue; }
                uint2 codes8;
                if (role == 0 && Tile::kKey == VqKey::Q4) {
                    codes8 = kv_cache_q4_decode_word(
                        reinterpret_cast<const std::uint32_t*>(st + Tile::kKeyCodes + j * Tile::kKeyCodeBytes)[w]);
                } else {
                    const unsigned char* codes =
                        role ? st + Tile::kValueCodes + j * 64 : st + Tile::kKeyCodes + j * Tile::kKeyCodeBytes;
                    codes8 = kv_cache_vq2_decode_word(reinterpret_cast<const std::uint16_t*>(codes)[w], codebook);
                }
                *reinterpret_cast<uint2*>(st + (role ? Tile::kValues : 0) + j * kHeadDim + 8 * w) = codes8;
                if (w == 0) {
                    const __half scale = reinterpret_cast<const __half*>(st + Tile::kCodeScales)[2 * j + role];
                    __half* groups = reinterpret_cast<__half*>(st + (role ? Tile::kValueScales : Tile::kKeyScales) + j * 8);
#pragma unroll
                    for (int g = 0; g < kKVWindowGroups; ++g) { groups[g] = scale; }
                }
            }
            __syncthreads();
        }
        const unsigned char* stage = smem + s * Shared::kStage;
        // Scores: one warp per token, all query heads of the group.
        for (int j = warp; j < n; j += kAttentionWarps) {
            float k[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { k[r] = Tile::key(stage, j, lane + 32 * r); }
            float part[Group];
#pragma unroll
            for (int h = 0; h < Group; ++h) {
                float dot = 0.0F;
                if (Exact || h < group) {
                    const float4 lo = reinterpret_cast<const float4*>(qs)[h * 64 + lane];
                    const float4 hi = reinterpret_cast<const float4*>(qs)[h * 64 + 32 + lane];
                    const float qv[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                    for (int r = 0; r < 8; ++r) { dot += qv[r] * k[r]; }
                }
                part[h] = dot;
            }
            scatter_head_sums<Group>(part, group, lane, scale, probs, j);
        }
        __syncthreads();
        // Online softmax per head: the tile's exact maximum, the exponentials in parallel, then l
        // in token order.
        for (int h = warp; h < group; h += kAttentionWarps) {
            const float a = lane < n ? probs[h][lane] : -INFINITY;
            const float b = lane + 32 < n ? probs[h][lane + 32] : -INFINITY;
            const float m = fmaxf(m_run[h], warp_max(fmaxf(a, b)));
            if (lane < n) { probs[h][lane] = expf(a - m); }
            if (lane + 32 < n) { probs[h][lane + 32] = expf(b - m); }
            __syncwarp();
            if (lane == 0) {
                rescale[h] = expf(m_run[h] - m);
                m_run[h]   = m;
            }
        }
        __syncthreads();
        if (threadIdx.x < group) {
            const int h = threadIdx.x;
            float l     = l_run[h] * rescale[h];
            for (int j = 0; j < n; ++j) { l += probs[h][j]; }
            l_run[h] = l;
        }
#pragma unroll
        for (int i = 0; i < Half; ++i) {
            if (mine(i)) { acc[i] *= rescale[h_base + i]; }
        }
        // Values, folded in token order, four tokens' probabilities a load.
        for (int j0 = 0; j0 < n; j0 += 4) {
            float v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) { v[u] = j0 + u < n ? Tile::value(stage, j0 + u, d_own) : 0.0F; }
#pragma unroll
            for (int i = 0; i < Half; ++i) {
                if (mine(i)) {
                    const float4 pr = *reinterpret_cast<const float4*>(&probs[h_base + i][j0]);
                    acc[i] += pr.x * v[0];
                    if (j0 + 1 < n) { acc[i] += pr.y * v[1]; }
                    if (j0 + 2 < n) { acc[i] += pr.z * v[2]; }
                    if (j0 + 3 < n) { acc[i] += pr.w * v[3]; }
                }
            }
        }
        __syncthreads();
    }
    if constexpr (Tile::kRotatedValues) {
        // Values are stored in the normalized Hadamard domain (its own inverse): rotate the
        // unnormalized output back, a linear map, so partials still merge by their weights.
        __syncthreads(); // the query heads are no longer read
        float* rows = qs; // [Group][256]
#pragma unroll
        for (int i = 0; i < Half; ++i) {
            if (mine(i)) { rows[(h_base + i) * kHeadDim + d_own] = acc[i]; }
        }
        __syncthreads();
        for (int h = warp; h < group; h += kAttentionWarps) {
            float v[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { v[r] = rows[h * kHeadDim + lane + 32 * r]; }
            normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
            for (int r = 0; r < 8; ++r) { rows[h * kHeadDim + lane + 32 * r] = v[r]; }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < Half; ++i) {
            if (mine(i)) { acc[i] = rows[(h_base + i) * kHeadDim + d_own]; }
        }
    }
#pragma unroll
    for (int i = 0; i < Half; ++i) {
        if (!mine(i)) { continue; }
        const int h = h_base + i, head = kv_head * group + h;
        if (splits == 1) {
            const float l = l_run[h];
            out[(static_cast<std::size_t>(t) * heads + head) * kHeadDim + d_own] =
                __float2bfloat16_rn(l > 0.0F ? acc[i] / l : 0.0F);
        } else {
            float* slot = partial + ((static_cast<std::size_t>(t) * splits + split) * heads + head) * (kHeadDim + 2);
            slot[d_own] = acc[i];
            if (d_own == 0) {
                slot[kHeadDim]     = m_run[h];
                slot[kHeadDim + 1] = l_run[h];
            }
        }
    }
    if constexpr (Tile::kVq) { cp_wait<0>(); } // the codebook copy of a split without tiles
}

__global__ void merge_kernel(const float* __restrict__ partial, int heads, int splits, bf16* __restrict__ out) {
    pdl::enter();
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

// Calls of at most this many columns: decode, one row's verification, up to eight plain lanes.
constexpr int kSmallCallColumns = 8;

// Splits per (column, KV head).
// - Calls of at most kSmallCallColumns columns: one 64-token tile per split of a column's attended
//   bound (33 at budget 2048, ratio 4), whatever the width. A column's bits then do not depend on how
//   many columns share the call, so concurrent sequences decode as they do alone (the preemption and
//   prefix-cache resumes rely on it; test_qsa checks it). Against one wave per call (the rule below)
//   this costs ~2 us at W = 5 and ~10 us at W = 8 per call, under 0.2 % of a round; W = 1-2 are equal.
// - Wider calls: one wave of the decode kernel's CTAs (one 512-thread CTA per SM), at most 48 and at
//   least 4 once split. A call that fills half the SMs unsplit takes the prompt route. Against the
//   earlier 340-CTA / 33-split rule: 42-73 % faster at 43-85 columns (design §19.3.15).
int attention_splits(int columns, const QsaGeometry& g) {
    const int attended = g.budget + g.ratio - 1;
    if (columns <= kSmallCallColumns) { return (attended + kTokenTile - 1) / kTokenTile; }
    static const int sms = [] {
        int device = 0, count = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || count <= 0) {
            throw std::runtime_error("qsa: cannot query the multiprocessor count");
        }
        return count;
    }();
    const int splits = std::clamp(sms / (columns * g.kv_heads), 1, 48);
    return splits == 1 ? 1 : std::max(splits, 4);
}

struct Workspace {
    std::int32_t* selected;
    std::int32_t* counts;
    void* select;             // qsa_select scratch
    std::size_t select_bytes;
    float* partial;
    detail::KVCacheVqStaging staging; // a wide vector-quantized call's window rows
};

bool vector_quantized(KvCacheStorage storage) {
    return storage == KvCacheStorage::Vq2 || storage == KvCacheStorage::Q4KeyVq2Value;
}

// Whether a call of `columns` columns stages its window rows: vector-quantized storage with a window,
// wider than the calls whose append may write their slots before attention.
bool staged_call(KvCacheStorage storage, bool window, int columns) {
    return vector_quantized(storage) && window && columns > kKVWindowInlineWidth;
}

Workspace carve(void* base, const QsaGeometry& g, int columns, int max_context, bool staging,
                std::size_t& bytes) {
    auto align = [](std::size_t v) { return (v + 255) / 256 * 256; };
    const int top_blocks = g.budget / g.ratio;
    const std::size_t selected = align(sizeof(std::int32_t) * top_blocks * static_cast<std::size_t>(columns));
    const std::size_t counts   = align(sizeof(std::int32_t) * static_cast<std::size_t>(columns));
    const std::size_t select   = align(detail::qsa_select_scratch_bytes(g, columns, max_context));
    const int splits           = attention_splits(columns, g);
    const std::size_t partial  = splits > 1 ? align(sizeof(float) * static_cast<std::size_t>(columns) * splits *
                                                    g.heads * (kHeadDim + 2))
                                            : 0;
    // Staging rows [256, columns, Hkv] per role, [4, ...] FP16 scales per role, [2, ...] tags.
    const std::size_t rows   = static_cast<std::size_t>(columns) * g.kv_heads;
    const std::size_t codes  = staging ? align(kHeadDim * rows) : 0;
    const std::size_t scales = staging ? align(sizeof(__half) * kKVWindowGroups * rows) : 0;
    const std::size_t tags   = staging ? align(sizeof(std::int32_t) * 2 * rows) : 0;
    bytes = selected + counts + select + partial + 2 * codes + 2 * scales + tags;
    auto* p = static_cast<unsigned char*>(base);
    Workspace out{};
    if (p != nullptr) {
        out.selected     = reinterpret_cast<std::int32_t*>(p);
        out.counts       = reinterpret_cast<std::int32_t*>(p + selected);
        out.select       = p + selected + counts;
        out.select_bytes = select;
        out.partial      = partial ? reinterpret_cast<float*>(p + selected + counts + select) : nullptr;
        if (staging) {
            unsigned char* at = p + selected + counts + select + partial;
            out.staging.k_codes  = reinterpret_cast<std::int8_t*>(at);
            out.staging.v_codes  = reinterpret_cast<std::int8_t*>(at + codes);
            out.staging.k_scales = reinterpret_cast<__half*>(at + 2 * codes);
            out.staging.v_scales = reinterpret_cast<__half*>(at + 2 * codes + scales);
            out.staging.tags     = reinterpret_cast<std::int32_t*>(at + 2 * codes + 2 * scales);
        }
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
        geometry.rotary_dim, geometry.theta, geometry.eps, static_cast<bf16*>(layer.pooled_pages.data), layer.spaces);
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
                                          std::int32_t max_context, KvCacheStorage storage) {
    require_geometry(geometry);
    // Bounded by the staging a windowed call of this width would need. Covers every narrower call
    // too: a call of kSmallCallColumns columns keeps more split partials than wider calls up to ~21
    // columns, and every other term grows with the width.
    std::size_t bytes = 0, small = 0;
    (void)carve(nullptr, geometry, columns, max_context, staged_call(storage, true, columns), bytes);
    const int narrow = std::min<int>(columns, kSmallCallColumns);
    (void)carve(nullptr, geometry, narrow, max_context, staged_call(storage, true, narrow), small);
    return std::max(bytes, small);
}

void qsa_attention(const Tensor& q, const Tensor& index_q, const QsaKVLayer& layer,
                   const QsaBatch& batch, const QsaGeometry& geometry, float scale,
                   std::int32_t max_context, void* workspace, std::size_t workspace_bytes,
                   Tensor& out, cudaStream_t stream, const QsaAppend* append) {
    require_geometry(geometry);
    const int columns = q.ne[2];
    require_batch(batch, columns);
    require(contiguous(q, DType::BF16) && q.ne[0] == kHeadDim && q.ne[1] == geometry.heads &&
                contiguous(index_q, DType::BF16) && index_q.ne[0] == kIndexDim &&
                index_q.ne[1] == geometry.index_heads && index_q.ne[2] == columns && contiguous(out, DType::BF16) &&
                out.ne[0] == kHeadDim && out.ne[1] == geometry.heads && out.ne[2] == columns,
            "attention shapes disagree");
    require(layer.kv.num_kv_heads == geometry.kv_heads && layer.kv.head_dim == kHeadDim, "KV geometry differs");
    const bool vq     = vector_quantized(layer.kv.storage);
    const bool window = vq && layer.kv.window.present();
    const bool staged = staged_call(layer.kv.storage, window, columns);
    require(vq == (append != nullptr), "the vector-quantized storages, and only they, take the call's K and V");
    require(!staged || batch.batch == 1, "a call wider than the inline window width takes one sequence");
    std::size_t need = 0;
    Workspace ws     = carve(workspace, geometry, columns, max_context, staged, need);
    require(workspace != nullptr && workspace_bytes >= need, "workspace is too small");
    // The vector-quantized append: paged codes, and the window slots of a narrow call or the staged
    // rows of a wide one (committed after attention).
    PagedKVBatchLayerView cache{.k_pages       = layer.kv.k_pages,
                                .v_pages       = layer.kv.v_pages,
                                .k_scale_pages = layer.kv.k_scale_pages,
                                .v_scale_pages = layer.kv.v_scale_pages,
                                .block_tables  = batch.block_tables,
                                .head_dim      = layer.kv.head_dim,
                                .num_kv_heads  = layer.kv.num_kv_heads,
                                .storage       = layer.kv.storage,
                                .window        = layer.kv.window};
    const Tensor positions2 = batch.positions.view({batch.width, batch.batch});
    if (vq) {
        require(contiguous(append->k, DType::BF16) && contiguous(append->v, DType::BF16) &&
                    append->k.ne[0] == kHeadDim && append->k.ne[1] == geometry.kv_heads &&
                    append->k.ne[2] == batch.width && append->k.ne[3] == batch.batch &&
                    std::equal(append->v.ne, append->v.ne + 4, append->k.ne),
                "append K/V are BF16 [head_dim, kv_heads, width, batch]");
        require(!window || (layer.kv.window.slots.data != nullptr && layer.kv.window.slots.dtype == DType::I32 &&
                            layer.kv.window.slots.numel() >= batch.batch),
                "the window's slots give each sequence's window row");
        detail::kv_cache_append_vq_batch_launch(append->k, append->v, positions2, Tensor{}, batch.table_rows, cache,
                                                staged ? &ws.staging : nullptr, stream);
    }
    detail::QsaVqWindow vq_args;
    if (window) {
        vq_args.k_codes  = static_cast<const std::int8_t*>(layer.kv.window.k_codes.data);
        vq_args.v_codes  = static_cast<const std::int8_t*>(layer.kv.window.v_codes.data);
        vq_args.k_scales = static_cast<const __half*>(layer.kv.window.k_scales.data);
        vq_args.v_scales = static_cast<const __half*>(layer.kv.window.v_scales.data);
        vq_args.tags     = static_cast<const std::int32_t*>(layer.kv.window.tags.data);
        vq_args.slots    = static_cast<const std::int32_t*>(layer.kv.window.slots.data);
        if (staged) {
            vq_args.staged_k_codes  = ws.staging.k_codes;
            vq_args.staged_v_codes  = ws.staging.v_codes;
            vq_args.staged_k_scales = ws.staging.k_scales;
            vq_args.staged_v_scales = ws.staging.v_scales;
            vq_args.staged_width    = columns;
        }
    }
    const auto commit = [&] {
        if (staged) {
            detail::kv_cache_vq_window_commit_launch(ws.staging, positions2, Tensor{}, cache, columns, stream);
        }
    };
    const int top_blocks = geometry.budget / geometry.ratio;
    const auto* tables   = static_cast<const std::int32_t*>(batch.block_tables.data);
    const int stride     = batch.block_tables.ne[0];
    const auto* rows     = static_cast<const std::int32_t*>(batch.table_rows.data);
    const auto* pos      = static_cast<const std::int32_t*>(batch.positions.data);
    detail::qsa_select(index_q, layer.pooled_pages, batch, geometry, max_context, ws.select, ws.select_bytes,
                       {ws.selected, ws.counts}, stream, layer.spaces);
    // Calls this kernel would run unsplit take the Tensor Core prompt route (qsa_prompt.h): the same
    // width-invariance class, so a column's bits still do not depend on its call's width there.
    const int splits = attention_splits(columns, geometry);
    if (splits == 1 && detail::qsa_prompt_supported(layer, geometry)) {
        detail::qsa_prompt_attention(q, layer, batch, geometry, scale, ws.selected, ws.counts, out, stream,
                                     vq_args);
        commit();
        return;
    }
    const dim3 grid(splits, geometry.kv_heads, columns);
    // A column attends to at most budget + ratio - 1 tokens (dense below the budget, selected blocks
    // and the open block above it), so this bounds every split's tiles.
    const int split_tokens = (geometry.budget + geometry.ratio - 1 + splits - 1) / splits;
    const auto launch = [&]<KvCacheStorage Storage, int Group, bool Exact, int Stages>() {
        using Shared                = AttentionShared<Storage, Group, Stages>;
        constexpr std::size_t bytes = Shared::kBytes;
        constexpr auto kernel       = attention_kernel<Storage, Group, Exact, Stages>;
        static const bool reserved  = [] {
            return cudaFuncSetAttribute(attention_kernel<Storage, Group, Exact, Stages>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(Shared::kBytes)) == cudaSuccess;
        }();
        require(reserved, "attention shared memory cannot be reserved");
        const cudaError_t error = pdl::launch_consumer(
            {grid, dim3(kAttentionThreads), bytes, stream}, kernel, static_cast<const bf16*>(q.data), geometry.heads,
            geometry.kv_heads, layer.kv, layer.spaces, tables, stride, rows, pos, batch.width, geometry.ratio,
            static_cast<const std::int32_t*>(ws.selected), static_cast<const std::int32_t*>(ws.counts), top_blocks,
            splits, scale, ws.partial, static_cast<bf16*>(out.data), vq_args);
        if (error != cudaSuccess) { throw std::runtime_error(std::string("qsa attention: ") + cudaGetErrorString(error)); }
    };
    // Two stages only where a split holds more than one tile and both fit; one stage otherwise
    // (BF16's 64 KiB stage, and every decode and verification call).
    const auto launch_stages = [&]<KvCacheStorage Storage, int Group, bool Exact>() {
        using One = AttentionShared<Storage, Group, 1>; // two stages add a stage and a token slot
        if constexpr (One::kBytes + One::kStage + sizeof(std::int32_t) * 2 * kTokenTile <= 99 * 1024) {
            if (split_tokens > kTokenTile) {
                launch.template operator()<Storage, Group, Exact, 2>();
                return;
            }
        }
        launch.template operator()<Storage, Group, Exact, 1>();
    };
    // Qwen4Exp's 24 / 2 = 12 query heads per KV head take the exact instance; any other group up to
    // 16 the bounded one.
    const auto launch_storage = [&]<KvCacheStorage Storage>() {
        if (geometry.heads / geometry.kv_heads == 12) {
            launch_stages.template operator()<Storage, 12, true>();
        } else {
            launch_stages.template operator()<Storage, kMaxGroup, false>();
        }
    };
    switch (layer.kv.storage) {
    case KvCacheStorage::BFloat16:
        launch_storage.template operator()<KvCacheStorage::BFloat16>();
        break;
    case KvCacheStorage::Int8Group64:
        launch_storage.template operator()<KvCacheStorage::Int8Group64>();
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        launch_storage.template operator()<KvCacheStorage::Fp8E4M3Row256>();
        break;
    case KvCacheStorage::Nvfp4Group16:
        launch_storage.template operator()<KvCacheStorage::Nvfp4Group16>();
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        launch_storage.template operator()<KvCacheStorage::Fp8KeyNvfp4Value>();
        break;
    case KvCacheStorage::Vq2:
        launch_storage.template operator()<KvCacheStorage::Vq2>();
        break;
    case KvCacheStorage::Q4KeyVq2Value:
        launch_storage.template operator()<KvCacheStorage::Q4KeyVq2Value>();
        break;
    default:
        throw std::invalid_argument("qsa: this KV storage is not implemented yet");
    }
    check_launch("attention");
    if (splits > 1) {
        const cudaError_t error =
            pdl::launch_consumer({dim3(geometry.heads, columns), dim3(kHeadDim), 0, stream}, merge_kernel,
                                 static_cast<const float*>(ws.partial), geometry.heads, splits,
                                 static_cast<bf16*>(out.data));
        if (error != cudaSuccess) { throw std::runtime_error(std::string("qsa merge: ") + cudaGetErrorString(error)); }
    }
    commit();
}

} // namespace ninfer::ops
