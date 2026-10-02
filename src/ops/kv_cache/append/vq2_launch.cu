#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/kv_cache/append/vq2_kernel.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;
// Rows from which an append encodes one row per warp (a 256-column call).
constexpr int kWarpAppendRows = 2048;

KVCacheVqWindowTarget window_slots(const PagedKVWindowView& window) {
    return {
        .k_codes  = static_cast<std::int8_t*>(window.k_codes.data),
        .v_codes  = static_cast<std::int8_t*>(window.v_codes.data),
        .k_scales = static_cast<__half*>(window.k_scales.data),
        .v_scales = static_cast<__half*>(window.v_scales.data),
        .tags     = static_cast<std::int32_t*>(window.tags.data),
        .slots    = static_cast<const std::int32_t*>(window.slots.data),
    };
}

// The selected device's SM count: the append grids are sized in whole waves of it.
int vq_append_multiprocessors() {
    static const int count = [] {
        int device = 0;
        int value  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return count;
}

template <typename Geometry, KVCacheVqKeyCodec Codec, KVCacheVqWindowMode Mode, bool MultiBatch,
          typename CacheView, typename Metadata>
void launch_for(const Tensor& k, const Tensor& v, const Tensor& positions, const CacheView& cache,
                Metadata metadata, KVCacheVqWindowTarget window, cudaStream_t stream) {
    const auto width = static_cast<std::int32_t>(k.ne[2]);
    const int rows   = 2 * width * Geometry::KVHeads;
    const auto batch = static_cast<unsigned>(MultiBatch ? k.ne[3] : 1);
    const auto args  = [&](auto kernel, dim3 grid, int threads) {
        kernel<<<grid, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata,
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<__half*>(cache.k_scale_pages.data),
            static_cast<__half*>(cache.v_scale_pages.data), window, width);
    };
    // Each block stages the codebook once and loops over rows. A few rows (a decode or
    // verification step) take a whole block each for the lowest latency; a wide call's rows take a
    // warp each, three blocks per SM, for the highest throughput.
    if (rows < kWarpAppendRows) {
        const int kMaxBlocks = 2 * vq_append_multiprocessors();
        args(kv_cache_append_vq_kernel<Geometry, Metadata, Codec, Mode, MultiBatch>,
             dim3(static_cast<unsigned>(std::min(rows, kMaxBlocks)), 1, batch),
             kKVCacheVqAppendThreads);
    } else {
        const int kMaxBlocks = 3 * vq_append_multiprocessors();
        args(kv_cache_append_vq_warp_kernel<Geometry, Metadata, Codec, Mode, MultiBatch>,
             dim3(static_cast<unsigned>(std::min((rows + kKVCacheVqAppendWarpRows - 1) /
                                                     kKVCacheVqAppendWarpRows,
                                                 kMaxBlocks)),
                  1, batch),
             32 * kKVCacheVqAppendWarpRows);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <KVCacheVqWindowMode Mode, bool MultiBatch, typename CacheView, typename Metadata>
void dispatch(const Tensor& k, const Tensor& v, const Tensor& positions, const CacheView& cache,
              Metadata metadata, KVCacheVqWindowTarget window, cudaStream_t stream) {
    const auto run = [&]<KVCacheVqKeyCodec Codec>() {
        if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads)
            launch_for<KVCacheAppendD256Kv4, Codec, Mode, MultiBatch>(k, v, positions, cache,
                                                                      metadata, window, stream);
        else
            launch_for<KVCacheAppendD256Kv2, Codec, Mode, MultiBatch>(k, v, positions, cache,
                                                                      metadata, window, stream);
    };
    if (cache.storage == KvCacheStorage::Vq2)
        run.template operator()<KVCacheVqKeyCodec::Vq2>();
    else if (cache.storage == KvCacheStorage::Q4KeyVq2Value)
        run.template operator()<KVCacheVqKeyCodec::Q4>();
    else
        throw std::logic_error("vector-quantized append on another KV storage");
}

template <bool MultiBatch, typename CacheView, typename Metadata>
void dispatch_mode(const Tensor& k, const Tensor& v, const Tensor& positions,
                   const CacheView& cache, Metadata metadata, const KVCacheVqStaging* staging,
                   cudaStream_t stream) {
    if (staging != nullptr)
        dispatch<KVCacheVqWindowMode::Staging, MultiBatch>(
            k, v, positions, cache, metadata,
            KVCacheVqWindowTarget{staging->k_codes, staging->v_codes, staging->k_scales,
                                  staging->v_scales, staging->tags},
            stream);
    else if (cache.window.present())
        dispatch<KVCacheVqWindowMode::Slots, MultiBatch>(k, v, positions, cache, metadata,
                                                         window_slots(cache.window), stream);
    else
        dispatch<KVCacheVqWindowMode::None, MultiBatch>(k, v, positions, cache, metadata, {},
                                                        stream);
}

// One warp per (column, kv head, role): copies a staged window row and its tag into the row's
// slot when the column's position is a sink or lies in the call's final kKVWindowRingTokens.
template <int KVHeads>
__launch_bounds__(256) __global__ void kv_cache_vq_window_commit_kernel(
    KVCacheVqStaging staging, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, KVCacheVqWindowTarget window,
    std::int32_t width) {
    const int unit  = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int valid = valid_columns == nullptr ? width : min(width, max(0, valid_columns[0]));
    if (unit >= 2 * valid * KVHeads) return;
    const int role     = unit & 1;
    const int column   = (unit >> 1) / KVHeads;
    const int kv_head  = (unit >> 1) % KVHeads;
    const int position = positions[0] + column;
    const int last     = positions[0] + valid - 1;
    if (position >= kKVWindowSinkTokens && position <= last - kKVWindowRingTokens) return;
    const int row      = window.slots == nullptr ? 0 : window.slots[0];
    const int slot     = kv_window_slot(position);
    const std::int64_t staged = static_cast<std::int64_t>(kv_head) * width + column;
    const std::int8_t* source = (role ? staging.v_codes : staging.k_codes) + staged * 256;
    std::int8_t* destination  = (role ? window.v_codes : window.k_codes) +
                               kv_window_code_index<KVHeads>(row, kv_head, slot, 0);
    reinterpret_cast<int2*>(destination)[lane] = reinterpret_cast<const int2*>(source)[lane];
    if (lane < kKVWindowGroups) {
        (role ? window.v_scales : window.k_scales)[kv_window_scale_index<KVHeads>(row, kv_head,
                                                                                   slot, lane)] =
            (role ? staging.v_scales : staging.k_scales)[staged * kKVWindowGroups + lane];
    }
    if (lane == 0) {
        window.tags[kv_window_tag_index<KVHeads>(row, kv_head, slot, role)] =
            staging.tags[staged * 2 + role];
    }
}

} // namespace

void kv_cache_append_vq_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                               PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    dispatch_mode<false>(k, v, positions, cache, metadata, nullptr, stream);
}

void kv_cache_append_vq_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& valid_columns, const Tensor& table_rows,
                                     PagedKVBatchLayerView cache, const KVCacheVqStaging* staging,
                                     cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (k.ne[3] > 1) {
            if (staging != nullptr)
                throw std::logic_error("staged window appends are single-sequence");
            dispatch_mode<true>(k, v, positions, cache, metadata, nullptr, stream);
        } else {
            dispatch_mode<false>(k, v, positions, cache, metadata, staging, stream);
        }
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

void kv_cache_vq_window_commit_launch(const KVCacheVqStaging& staging, const Tensor& positions,
                                      const Tensor& valid_columns,
                                      const PagedKVBatchLayerView& cache, std::int32_t width,
                                      cudaStream_t stream) {
    if (!cache.window.present()) return;
    const auto* valid = static_cast<const std::int32_t*>(valid_columns.data);
    const auto* pos   = static_cast<const std::int32_t*>(positions.data);
    const int units   = 2 * width * cache.num_kv_heads;
    const dim3 grid(static_cast<unsigned>((units + 7) / 8));
    if (cache.num_kv_heads == 4)
        kv_cache_vq_window_commit_kernel<4><<<grid, kBlock, 0, stream>>>(
            staging, pos, valid, window_slots(cache.window), width);
    else
        kv_cache_vq_window_commit_kernel<2><<<grid, kBlock, 0, stream>>>(
            staging, pos, valid, window_slots(cache.window), width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
