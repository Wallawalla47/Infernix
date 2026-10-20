// The reference route's causal attention (causal_softmax_attention_reference, softmax_attention.h): FP32
// scores and probabilities, FP32 P x V, over BF16 K and FP16 V in the paged cache.
#include "infernix/ops/kv_cache_append.h"
#include "infernix/ops/softmax_attention.h"

#include "core/device.h"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr int kHeadDim = 256;
constexpr int kWarps   = 8;
// Each warp owns four query columns of the CTA's head; a lane owns eight of a column's 256 features
// (its query values and output accumulators) and, in the softmax, one key of the tile.
constexpr int kQueriesPerWarp = 4;
constexpr int kColumns        = kWarps * kQueriesPerWarp;
constexpr int kTileKeys       = 32;
constexpr unsigned kFullMask  = 0xffffffffU;
static_assert(kPagedKVPageSize % kTileKeys == 0, "a key tile lies within one cache page");

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("causal_softmax_attention_reference: ") + message); }
}

// Lane r returns the sum over the warp's lanes of v[r] (a reduce-scatter in five exchange stages; the
// association is fixed, so the result is too).
__device__ __forceinline__ float reduce_scatter(float (&v)[32], int lane) {
#pragma unroll
    for (int width = 16; width >= 1; width >>= 1) {
        const bool upper = (lane & width) != 0;
#pragma unroll
        for (int i = 0; i < width; ++i) {
            const float send = upper ? v[i] : v[i + width];
            const float keep = upper ? v[i + width] : v[i];
            v[i]             = keep + __shfl_xor_sync(kFullMask, send, width);
        }
    }
    return v[0];
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) { x = fmaxf(x, __shfl_xor_sync(kFullMask, x, offset)); }
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) { x += __shfl_xor_sync(kFullMask, x, offset); }
    return x;
}

template <int KVHeads>
__global__ void __launch_bounds__(kWarps * 32)
    causal_attention_reference_kernel(const __nv_bfloat16* q, const __nv_bfloat16* keys, const __half* values,
                                      const std::int32_t* tables, const std::int32_t* table_rows, int table_stride,
                                      const std::int32_t* positions, int tokens, int query_heads, float scale,
                                      __nv_bfloat16* out) {
    __shared__ __align__(16) __nv_bfloat16 key_tile[kTileKeys][kHeadDim];
    __shared__ __align__(16) __half value_tile[kTileKeys][kHeadDim];
    __shared__ int block_max_position;

    const int head        = static_cast<int>(blockIdx.y);
    const int kv_head     = head / (query_heads / KVHeads);
    const int warp        = static_cast<int>(threadIdx.x) / 32;
    const int lane        = static_cast<int>(threadIdx.x) % 32;
    const int first       = static_cast<int>(blockIdx.x) * kColumns;
    const int* table      = tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    const int feature     = lane * 8;

    if (threadIdx.x == 0) { block_max_position = -1; }
    __syncthreads();
    float query[kQueriesPerWarp][8];
    float output[kQueriesPerWarp][8];
    float running_max[kQueriesPerWarp];
    float running_sum[kQueriesPerWarp];
    int query_position[kQueriesPerWarp];
#pragma unroll
    for (int i = 0; i < kQueriesPerWarp; ++i) {
        const int column = first + warp * kQueriesPerWarp + i;
        query_position[i] = column < tokens ? positions[column] : -1;
        running_max[i]    = -INFINITY;
        running_sum[i]    = 0.0F;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            query[i][e]  = 0.0F;
            output[i][e] = 0.0F;
        }
        if (column < tokens) {
            const uint4 packed = *reinterpret_cast<const uint4*>(
                q + (static_cast<std::int64_t>(column) * query_heads + head) * kHeadDim + feature);
            const auto* bits = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
            for (int e = 0; e < 8; ++e) { query[i][e] = __bfloat162float(bits[e]); }
            if (lane == 0) { atomicMax(&block_max_position, query_position[i]); }
        }
    }
    __syncthreads();
    const int last_key = block_max_position;

    for (int k0 = 0; k0 <= last_key; k0 += kTileKeys) {
        // Stage the tile's keys and values (16-byte chunks); keys past the block's last query position
        // are zero (never read uninitialized cache into a product).
        const int page = paged_kv_physical_page(table, k0);
        for (int chunk = static_cast<int>(threadIdx.x); chunk < kTileKeys * (kHeadDim / 8); chunk += kWarps * 32) {
            const int key = chunk / (kHeadDim / 8);
            const int d   = (chunk % (kHeadDim / 8)) * 8;
            uint4 k_bits  = make_uint4(0, 0, 0, 0);
            uint4 v_bits  = make_uint4(0, 0, 0, 0);
            if (k0 + key <= last_key) {
                const std::int64_t offset =
                    paged_kv_element_offset<kHeadDim, KVHeads>(page, kv_head, (k0 + key) & kPagedKVPageMask, d);
                k_bits = *reinterpret_cast<const uint4*>(keys + offset);
                v_bits = *reinterpret_cast<const uint4*>(values + offset);
            }
            *reinterpret_cast<uint4*>(&key_tile[key][d])   = k_bits;
            *reinterpret_cast<uint4*>(&value_tile[key][d]) = v_bits;
        }
        __syncthreads();

        float key_values[8];
        float value_values[8];
#pragma unroll
        for (int i = 0; i < kQueriesPerWarp; ++i) {
            // Scores: each lane's eight-feature partial for every key, then lane r holds key r's sum.
            float partial[kTileKeys];
#pragma unroll
            for (int r = 0; r < kTileKeys; ++r) {
                const uint4 packed = *reinterpret_cast<const uint4*>(&key_tile[r][feature]);
                const auto* bits   = reinterpret_cast<const __nv_bfloat16*>(&packed);
                float sum          = 0.0F;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    key_values[e] = __bfloat162float(bits[e]);
                    sum           = fmaf(query[i][e], key_values[e], sum);
                }
                partial[r] = sum;
            }
            const float dot   = reduce_scatter(partial, lane);
            const bool live   = k0 + lane <= query_position[i];
            const float score = live ? dot * scale : -INFINITY;
            const float tile_max = warp_max(score);
            const float new_max  = fmaxf(running_max[i], tile_max);
            if (new_max == -INFINITY) { continue; } // an inactive column (no keys)
            const float correction = expf(running_max[i] - new_max);
            const float p          = live ? expf(score - new_max) : 0.0F;
            running_sum[i]         = running_sum[i] * correction + warp_sum(p);
            running_max[i]         = new_max;
#pragma unroll
            for (int e = 0; e < 8; ++e) { output[i][e] *= correction; }
#pragma unroll
            for (int r = 0; r < kTileKeys; ++r) {
                const float pr     = __shfl_sync(kFullMask, p, r);
                const uint4 packed = *reinterpret_cast<const uint4*>(&value_tile[r][feature]);
                const auto* bits   = reinterpret_cast<const __half*>(&packed);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    value_values[e] = __half2float(bits[e]);
                    output[i][e]    = fmaf(pr, value_values[e], output[i][e]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kQueriesPerWarp; ++i) {
        const int column = first + warp * kQueriesPerWarp + i;
        if (column >= tokens) { continue; }
        alignas(16) __nv_bfloat16 result[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { result[e] = __float2bfloat16_rn(output[i][e] / running_sum[i]); }
        *reinterpret_cast<uint4*>(out + (static_cast<std::int64_t>(column) * query_heads + head) * kHeadDim + feature) =
            *reinterpret_cast<const uint4*>(result);
    }
}

template <int KVHeads>
void launch(const Tensor& q, const PagedKVBatchLayerView& cache, const Tensor& positions, const Tensor& table_rows,
            int tokens, int query_heads, float scale, Tensor& out, cudaStream_t stream) {
    const dim3 grid((tokens + kColumns - 1) / kColumns, query_heads);
    causal_attention_reference_kernel<KVHeads><<<grid, kWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(cache.k_pages.data),
        static_cast<const __half*>(cache.v_pages.data), static_cast<const std::int32_t*>(cache.block_tables.data),
        static_cast<const std::int32_t*>(table_rows.data), cache.block_tables.ne[0],
        static_cast<const std::int32_t*>(positions.data), tokens, query_heads, scale,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void causal_softmax_attention_reference(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& kv_table_rows, AttentionHeadGeometry geometry, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    require(geometry.head_dim == kHeadDim && valid_attention_head_geometry(geometry) &&
                (geometry.kv_heads == 2 || geometry.kv_heads == 4),
            "the geometry must be D=256 with 2 or 4 KV heads");
    require(cache.storage == KvCacheStorage::BFloat16 && cache.head_dim == kHeadDim &&
                cache.num_kv_heads == geometry.kv_heads,
            "the cache must be BF16 storage of the geometry's KV heads");
    const int tokens = q.ne[1] == geometry.query_heads ? q.ne[2] : 0;
    require(tokens > 0 && q.dtype == DType::BF16 && q.is_contiguous() && q.ne[0] == kHeadDim && q.ne[3] == 1,
            "q must be contiguous BF16 [256, Hq, T]");
    require(k.dtype == DType::BF16 && v.dtype == DType::BF16 && k.is_contiguous() && v.is_contiguous() &&
                k.ne[0] == kHeadDim && k.ne[1] == geometry.kv_heads && k.ne[2] == tokens && k.ne[3] == 1 &&
                v.ne[0] == kHeadDim && v.ne[1] == geometry.kv_heads && v.ne[2] == tokens && v.ne[3] == 1,
            "k and v must be contiguous BF16 [256, Hkv, T]");
    require(out.dtype == DType::BF16 && out.is_contiguous() && out.ne[0] == kHeadDim &&
                out.ne[1] == geometry.query_heads && out.ne[2] == tokens && out.ne[3] == 1,
            "out must be contiguous BF16 [256, Hq, T]");
    require(positions.dtype == DType::I32 && positions.is_contiguous() && positions.ne[0] == tokens &&
                kv_table_rows.dtype == DType::I32 && kv_table_rows.ne[0] == 1,
            "positions must be I32 [T] and the table row I32 [1]");
    require(std::isfinite(scale), "scale must be finite");
    // The call's keys and values first, as the production routes append them.
    const Tensor k_batch         = k.view({kHeadDim, geometry.kv_heads, tokens, 1});
    const Tensor v_batch         = v.view({kHeadDim, geometry.kv_heads, tokens, 1});
    const Tensor positions_batch = positions.view({tokens, 1});
    kv_cache_append_batch(k_batch, v_batch, positions_batch, kv_table_rows, cache, stream);
    if (geometry.kv_heads == 4) {
        launch<4>(q, cache, positions, kv_table_rows, tokens, geometry.query_heads, scale, out, stream);
    } else {
        launch<2>(q, cache, positions, kv_table_rows, tokens, geometry.query_heads, scale, out, stream);
    }
}

} // namespace infernix::ops
