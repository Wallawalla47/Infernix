#pragma once

#include "infernix/ops/kv_cache_append.h"

#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops::detail {

void kv_cache_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                            PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_nvfp4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_nvfp4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        PagedKVBatchLayerView cache, cudaStream_t stream);

void kv_cache_append_k8v4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                 PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_k8v4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                       const Tensor& valid_columns, const Tensor& table_rows,
                                       PagedKVBatchLayerView cache, cudaStream_t stream);

// Per-call window staging of a wide vector-quantized append: every column's INT8-G64 window row
// ([256, width, Hkv] codes and [4, width, Hkv] scales per role) and its K/V tags [2, width, Hkv].
struct KVCacheVqStaging {
    std::int8_t* k_codes = nullptr;
    std::int8_t* v_codes = nullptr;
    __half* k_scales     = nullptr;
    __half* v_scales     = nullptr;
    std::int32_t* tags   = nullptr;
};

// Vector-quantized formats (Vq2, Q4KeyVq2Value). The single-sequence form and a batched call
// without staging write the window slots of the call's sinks and final kKVWindowRingTokens
// columns; with staging, every column's window row goes to staging instead and
// kv_cache_vq_window_commit_launch moves the slot-bound ones into the window afterwards.
void kv_cache_append_vq_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                               PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_vq_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& valid_columns, const Tensor& table_rows,
                                     PagedKVBatchLayerView cache, const KVCacheVqStaging* staging,
                                     cudaStream_t stream);

void kv_cache_vq_window_commit_launch(const KVCacheVqStaging& staging, const Tensor& positions,
                                      const Tensor& valid_columns,
                                      const PagedKVBatchLayerView& cache, std::int32_t width,
                                      cudaStream_t stream);

void kv_cache_append_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  PagedKVBatchLayerView cache, cudaStream_t stream);

struct KVCacheAppendPrefixPlan {
    std::int32_t tokens;
    std::int32_t min_count;
    std::int32_t max_count;
};

[[nodiscard]] KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope);

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& table_rows,
                                   PagedKVBatchLayerView cache, const KVCacheAppendPrefixPlan& plan,
                                   cudaStream_t stream);
void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& lanes,
                                   CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream);

} // namespace infernix::ops::detail
