#pragma once

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/grouped_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/tiled_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/merge.cuh"
#include <stdexcept>

namespace infernix::ops::detail {

template <class Geometry, bool Writable>
void validate_bf16_kv_operands(const CausalAttentionOperands& p,
                               const Bf16KvCacheView<Writable>& cache) {
    const auto aligned = [](const void* pointer) {
        return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
    };
    if (p.head_dim != Geometry::kHeadDim || cache.head_dim != Geometry::kHeadDim ||
        p.query_heads != Geometry::QHeads || cache.kv_heads != Geometry::KVHeads || !aligned(p.q) ||
        !aligned(p.out) || !aligned(cache.keys) || !aligned(cache.values) || !p.positions ||
        !cache.tables || p.width <= 0 || p.batch < 1 || p.batch > 65535 ||
        cache.table_stride <= 0 || p.visible_capacity <= 0 ||
        static_cast<std::int64_t>(p.visible_capacity) >
            static_cast<std::int64_t>(cache.table_stride) * kPagedKVPageSize ||
        (cache.valid_columns && !cache.table_rows) || (cache.key_side == nullptr) != (cache.value_side == nullptr) ||
        (cache.key_side && (!aligned(cache.key_side) || !aligned(cache.value_side))))
        throw std::invalid_argument("BF16 KV template operands do not match its geometry/layout");
}

template <int Bytes, auto Kernel>
int bf16_kv_dynamic_shared() {
    static_assert(Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        static const auto status =
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        CUDA_CHECK(status);
    }
    return Bytes;
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input>
void launch_bf16_kv_grouped_mma(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache,
                                Input input, Bf16KvPartition partition, CausalPartialView partials,
                                cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_bf16_kv_operands<G>(p, cache);
    if ((S::kFixedWidth && p.width != S::kFixedWidth) || partition.capacity < 1 ||
        partition.capacity > 256 || partition.normal_target < 1 || partition.long_target < 1 ||
        partition.key_rows != S::kKeyRows ||
        partition.live(p.visible_capacity).splits > partition.capacity || !partials.acc ||
        !partials.maximum || !partials.sum || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr))
        throw std::invalid_argument("BF16 grouped attention: invalid partition or metadata");
    if constexpr (Input::writes_cache) {
        if (!input.k || !input.v) throw std::invalid_argument("BF16 grouped append requires K/V");
        if (cache.key_side) throw std::invalid_argument("BF16 grouped attention: the packed storage appends first");
    }
    // Only verification rows carry a tree: masked rows of the grouped append route, or of the packed
    // storage's cached route (its rows appended first).
    const bool packed      = cache.key_side != nullptr;
    const bool tree_capable = Masked && (Input::writes_cache || packed);
    if (!tree_capable && cache.tree_masks)
        throw std::invalid_argument("BF16 grouped attention: this route has no tree");
    const auto launch = [&]<class T, bool Tree, bool Packed>() {
        constexpr auto kernel = bf16_kv_grouped_mma_kernel<G, T, MultiBatch, Masked, Input, Tree, Packed>;
        constexpr int bytes   = sizeof(Bf16KvGroupedStorage<G, T, Packed ? kBf16pTileRing : 1>);
        // Packed instances add their producer warps.
        constexpr int threads = Packed ? T::kThreads + 32 * kBf16pProducerWarps<T> : T::kThreads;
        int dynamic           = 0;
        if constexpr (bytes > 48 * 1024) dynamic = bf16_kv_dynamic_shared<bytes, kernel>();
        const dim3 grid(G::KVHeads * div_up(p.width * G::GroupSize, T::kQueryRows),
                        partition.capacity, p.batch);
        // A programmatic dependent of the RoPE / append kernel before it (the kernel waits at entry): the Qwen3.5 decode
        // graphs, INT8 KV, leave-one-out: the 35B-A3B MTP +0.24 %, the groupwise 27B DFlash2 +0.09 %, the NVFP4 27B
        // level (candidates §58). Same bits.
        CUDA_CHECK(pdl::launch_consumer(
                                    {grid, dim3(threads), static_cast<std::size_t>(dynamic), stream}, kernel,
            p.q, input, p.positions, cache.keys, cache.values, cache.tables, cache.valid_columns,
            cache.table_rows, cache.table_stride, p.width, p.visible_capacity, p.scale, partition,
            partials, cache.tree_masks, cache.key_side, cache.value_side));
    };
    if constexpr (!Input::writes_cache) {
        if (packed) {
            // 65-96 rows (a 24/4 call of 11-16 columns, say): one 96-row tile, so one CTA decodes each KV tile, not
            // two. The partition stays the 64-row plan's, so the bits are the same.
            const auto launch_packed = [&]<class T>() {
                if constexpr (Masked) {
                    if (cache.tree_masks) {
                        launch.template operator()<T, true, true>();
                        return;
                    }
                }
                launch.template operator()<T, false, true>();
            };
            if constexpr (S::kQueryRows == 64 && S::kWarpsKV == 1 && !S::kFixedWidth) {
                const int rows = p.width * G::GroupSize;
                if (rows > 64 && rows <= 96) {
                    launch_packed.template operator()<Bf16KvGroupedMmaSchedule<96, S::kKeyRows, 1, 1, 0>>();
                    return;
                }
            }
            launch_packed.template operator()<S>();
            return;
        }
    }
    if constexpr (Masked && Input::writes_cache) {
        if (cache.tree_masks) {
            launch.template operator()<S, true, false>();
            return;
        }
    }
    launch.template operator()<S, false, false>();
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable>
void launch_bf16_kv_merge(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache,
                          Bf16KvPartition partition, CausalPartialView partials,
                          cudaStream_t stream) {
    if (partition.capacity > S::kThreads)
        throw std::invalid_argument("BF16 merge capacity exceeds the reduction block");
    const dim3 grid(G::QHeads, div_up(G::kHeadDim, S::kDChunk), p.width * p.batch);
    // A captured merge launches as a programmatic dependent of the grouped kernel, which triggers it
    // once its KV loop ends (as the other formats' merges, 6049964f); eager launches serialize.
    CUDA_CHECK(pdl::launch_consumer(pdl::LaunchConfig{grid, dim3(S::kThreads), 0, stream},
                                    bf16_kv_merge_kernel<G, S, MultiBatch, Masked>,
                                    static_cast<const float*>(partials.acc), static_cast<const float*>(partials.maximum),
                                    static_cast<const float*>(partials.sum), p.positions, cache.valid_columns,
                                    p.width, p.batch, p.visible_capacity, partition, p.out));
}

// splits > 1 (the bf16-prompt-split fast path): the key blocks of every CTA in `splits` equal
// shares, their partials in `partial` (causal_partial_index over p.width columns), then the merge
// (after the last segment). segment.hi > 0: only the blocks in [segment.lo, segment.hi), resuming
// and suspending state in segment.state (Bf16KvSegment).
template <class Geometry, class Schedule>
void launch_bf16_kv_tiled_mma(const CausalAttentionOperands& p, Bf16KvReadView cache,
                              cudaStream_t stream, int splits = 1, CausalPartialView partial = {},
                              Bf16KvSegment segment = {}, bool last_segment = true) {
    validate_bf16_kv_operands<Geometry>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("BF16 tiled attention requires a complete single-row query");
    if (splits < 1 || splits > 8 || (splits > 1 && (!partial.acc || !partial.maximum || !partial.sum)))
        throw std::invalid_argument("BF16 tiled attention: invalid prompt split");
    const auto launch_segment = [&]<class Metadata, bool Segmented>(Metadata metadata) {
        if (splits > 1) {
            constexpr auto kernel = bf16_kv_tiled_mma_kernel<Geometry, Schedule, Metadata, true, Segmented>;
            const int bytes =
                bf16_kv_dynamic_shared<bf16_kv_tiled_shared_bytes<Geometry, Schedule>, kernel>();
            const dim3 grid(div_up(p.width, Schedule::kQueryRows), Geometry::QHeads, splits);
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.q, cache.keys, cache.values, metadata,
                                                                p.positions, p.scale, p.out, p.width,
                                                                partial, segment);
            CUDA_CHECK(cudaGetLastError());
            if (!last_segment) return;
            bf16_kv_prompt_merge_kernel<Geometry, Metadata>
                <<<dim3(Geometry::QHeads, p.width), Geometry::kHeadDim, 0, stream>>>(partial, metadata, splits,
                                                                                     p.width, p.out);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        constexpr auto kernel = bf16_kv_tiled_mma_kernel<Geometry, Schedule, Metadata, false, Segmented>;
        const int bytes =
            bf16_kv_dynamic_shared<bf16_kv_tiled_shared_bytes<Geometry, Schedule>, kernel>();
        const dim3 grid(div_up(p.width, Schedule::kQueryRows), Geometry::QHeads);
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.q, cache.keys, cache.values, metadata,
                                                            p.positions, p.scale, p.out, p.width,
                                                            CausalPartialView{}, segment);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto launch = [&]<class Metadata>(Metadata metadata) {
        if (segment.hi > 0) {
            launch_segment.template operator()<Metadata, true>(metadata);
        } else {
            launch_segment.template operator()<Metadata, false>(metadata);
        }
    };
    if (!cache.table_rows) {
        launch(PagedKVDirectMetadata{cache.tables});
    } else if (cache.valid_columns) {
        launch(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride});
    } else {
        launch(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride});
    }
}

} // namespace infernix::ops::detail
