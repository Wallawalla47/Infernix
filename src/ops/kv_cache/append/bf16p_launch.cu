#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/bf16p_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kWarps = 8;

// One warp a (token, KV head) row pair: its K row, then its V row (bf16p_codec.cuh). Positions are the call's own
// (positions[token]); with MultiBatch, blockIdx.z selects the sequence. A row with no packed form sets its sequence's
// flag: unpackable[table row] for batch metadata, *unpackable for one sequence's; none when unpackable is null.
template <typename Geometry, typename Metadata, bool MultiBatch>
__launch_bounds__(kWarps * 32) __global__ void kv_cache_append_bf16p_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ positions,
    Metadata metadata, std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ side_k, std::uint8_t* __restrict__ side_v, std::int32_t width,
    std::int32_t* unpackable) {
    if constexpr (MultiBatch) {
        const int batch = static_cast<int>(blockIdx.z);
        metadata.table_rows += batch;
        if (metadata.valid_columns) { metadata.valid_columns += batch; }
        const auto offset = static_cast<std::int64_t>(batch) * width * kBf16pRowValues * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += static_cast<std::int64_t>(batch) * width;
    }
    __shared__ unsigned scratch[kWarps][256];
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (unit >= metadata.valid_tokens(width) * Geometry::KVHeads) { return; }
    const int head = unit % Geometry::KVHeads, token = unit / Geometry::KVHeads;
    const int position          = positions[token];
    const std::int32_t page     = paged_kv_physical_page(metadata.block_table(), position);
    const std::int32_t offset   = position & kPagedKVPageMask;
    const std::int64_t row      = paged_kv_element_offset<kBf16pRowBytes, Geometry::KVHeads>(page, head, offset, 0);
    const std::int64_t side     = paged_kv_element_offset<kBf16pSideBytes, Geometry::KVHeads>(page, head, offset, 0);
    const std::int64_t source   = (static_cast<std::int64_t>(token) * Geometry::KVHeads + head) * kBf16pRowValues + 8 * lane;
    const uint4 key             = *reinterpret_cast<const uint4*>(k + source);
    const uint4 value           = *reinterpret_cast<const uint4*>(v + source);
    const bool packed_key       = bf16p_encode_row(key, cache_k + row, side_k + side, scratch[warp], lane);
    const bool packed_value     = bf16p_encode_row(value, cache_v + row, side_v + side, scratch[warp], lane);
    if (!(packed_key && packed_value) && lane == 0 && unpackable != nullptr) {
        if constexpr (requires { metadata.table_rows; }) { unpackable += metadata.table_rows[0]; }
        *unpackable = 1;
    }
}

template <typename Geometry, typename Metadata, bool MultiBatch, typename CacheView>
void launch_for(const Tensor& k, const Tensor& v, const Tensor& positions, const CacheView& cache, Metadata metadata,
                cudaStream_t stream) {
    const std::int32_t width = k.ne[2];
    const dim3 grid(static_cast<unsigned>(div_up(width * Geometry::KVHeads, kWarps)), 1,
                    static_cast<unsigned>(MultiBatch ? k.ne[3] : 1));
    kv_cache_append_bf16p_kernel<Geometry, Metadata, MultiBatch><<<grid, kWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data), metadata, static_cast<std::uint8_t*>(cache.k_pages.data),
        static_cast<std::uint8_t*>(cache.v_pages.data), static_cast<std::uint8_t*>(cache.k_scale_pages.data),
        static_cast<std::uint8_t*>(cache.v_scale_pages.data), width, cache.unpackable);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Metadata, bool MultiBatch, typename CacheView>
void dispatch(const Tensor& k, const Tensor& v, const Tensor& positions, const CacheView& cache, Metadata metadata,
              cudaStream_t stream) {
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_for<KVCacheAppendD256Kv4, Metadata, MultiBatch>(k, v, positions, cache, metadata, stream);
    } else {
        launch_for<KVCacheAppendD256Kv2, Metadata, MultiBatch>(k, v, positions, cache, metadata, stream);
    }
}

} // namespace

void kv_cache_append_bf16p_launch(const Tensor& k, const Tensor& v, const Tensor& positions, PagedKVLayerView cache,
                                  cudaStream_t stream) {
    dispatch<PagedKVDirectMetadata, false>(
        k, v, positions, cache, PagedKVDirectMetadata{static_cast<const std::int32_t*>(cache.block_table.data)},
        stream);
}

void kv_cache_append_bf16p_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables        = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns = Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows    = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride  = cache.block_tables.ne[0],
        };
        if (k.ne[3] > 1) {
            dispatch<PagedKVBatchMetadata<Masked>, true>(k, v, positions, cache, metadata, stream);
        } else {
            dispatch<PagedKVBatchMetadata<Masked>, false>(k, v, positions, cache, metadata, stream);
        }
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace infernix::ops::detail
