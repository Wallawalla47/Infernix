// ninfer::ops - causal cached Softmax Attention validation and finite route dispatch.
#include "ninfer/ops/softmax_attention.h"

#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim             = 256;
constexpr float kExpectedScale              = 0.0625f;
// The widest verification a multi-request call admits (an ngram copy round of 63 drafts).
constexpr std::int32_t kMaximumVerifyTokens = 64;
constexpr std::int32_t kMaximumBatchSize    = 8;
// Query rows of one eight-warp fast INT8 prompt CTA (int8/fast_tiled_launch.cuh checks it).
constexpr std::int32_t kPromptWaveRows = 128;

void require_causal_geometry(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        !((geometry.query_heads == 24 && geometry.kv_heads == 4) ||
          (geometry.query_heads == 16 && geometry.kv_heads == 2))) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::uint32_t validate_cache(const PagedKVLayerView& cache, std::int32_t kv_heads, const char* op) {
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    if (cache.k_pages.dtype != layout.key.data_dtype ||
        cache.v_pages.dtype != layout.value.data_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache data dtype");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, op, "block table");
    require_contiguous_nonnull(cache.block_table, op, "block table");

    const auto validate_scale = [&](const Tensor& tensor, const PagedKVVectorLayout& vector,
                                    const char* name) {
        if (!vector.has_scale()) {
            if (tensor.data != nullptr) {
                throw std::invalid_argument(std::string(op) +
                                            ": unscaled KV cache must not have scales");
            }
            return;
        }
        if (tensor.dtype != vector.scale_dtype) {
            throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
        }
        require_shape(tensor, vector.scale_leading_extent, kPagedKVPageSize, kv_heads,
                      physical_pages, op, name);
        require_contiguous_nonnull(tensor, op, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache k scale pages");
    validate_scale(cache.v_scale_pages, layout.value, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t validate_batch_cache(const PagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                   const char* op) {
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_tables.ne[0];
    const std::int32_t table_rows     = cache.block_tables.ne[1];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 || table_rows <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    if (cache.k_pages.dtype != layout.key.data_dtype ||
        cache.v_pages.dtype != layout.value.data_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache data dtype");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_tables.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block tables must be I32");
    }
    require_shape(cache.block_tables, logical_pages, table_rows, 1, 1, op, "block tables");
    require_contiguous_nonnull(cache.block_tables, op, "block tables");

    const auto validate_scale = [&](const Tensor& tensor, const PagedKVVectorLayout& vector,
                                    const char* name) {
        if (!vector.has_scale()) {
            if (tensor.data != nullptr) {
                throw std::invalid_argument(std::string(op) +
                                            ": unscaled KV cache must not have scales");
            }
            return;
        }
        if (tensor.dtype != vector.scale_dtype) {
            throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
        }
        require_shape(tensor, vector.scale_leading_extent, kPagedKVPageSize, kv_heads,
                      physical_pages, op, name);
        require_contiguous_nonnull(tensor, op, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache k scale pages");
    validate_scale(cache.v_scale_pages, layout.value, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

void validate_envelope(CausalAttentionExecutionEnvelope envelope, const PagedKVLayerView& cache,
                       std::int32_t tokens, const char* op) {
    const std::uint32_t capacity = validate_cache(cache, cache.num_kv_heads, op);
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                AttentionHeadGeometry geometry, const PagedKVLayerView& cache,
                                CausalAttentionExecutionEnvelope envelope, float scale,
                                const char* op) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(q, kHeadDim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    validate_envelope(envelope, cache, tokens, op);
}

void validate_batched_attention_tensors(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& kv_table_rows,
                                        const Tensor& out, const PagedKVBatchLayerView& cache,
                                        AttentionHeadGeometry geometry,
                                        CausalAttentionExecutionEnvelope envelope, float scale,
                                        const char* op) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    const bool masked = valid_columns.data != nullptr;
    if (positions.dtype != DType::I32 || kv_table_rows.dtype != DType::I32 ||
        (masked && valid_columns.dtype != DType::I32)) {
        throw std::invalid_argument(std::string(op) + ": batch metadata must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatchSize ||
        (batch > 1 && width > kMaximumVerifyTokens)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
    }
    require_shape(q, kHeadDim, q_heads, width, batch, op, "q");
    require_shape(positions, width, batch, 1, 1, op, "positions");
    if (masked) { require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns"); }
    require_shape(kv_table_rows, batch, 1, 1, 1, op, "KV table rows");
    require_shape(out, kHeadDim, q_heads, width, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    if (masked) { require_contiguous_nonnull(valid_columns, op, "valid columns"); }
    require_contiguous_nonnull(kv_table_rows, op, "KV table rows");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    const std::uint32_t capacity = validate_batch_cache(cache, kv_heads, op);
    if (cache.block_tables.ne[1] < batch || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity ||
        (!masked && envelope.max_visible_keys < static_cast<std::uint32_t>(width))) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope or table");
    }
}

} // namespace

std::int32_t causal_softmax_attention_prompt_wave_tokens(AttentionHeadGeometry geometry) {
    require_causal_geometry(geometry, "causal_softmax_attention prompt wave");
    int device          = 0;
    int multiprocessors = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    // The fast prompt kernel runs one CTA per SM over at most kPromptWaveRows query rows of one
    // head, and kPromptWaveRows is a multiple of the tiled kernel's 64-row query tile.
    const std::int32_t row_blocks = std::max(1, multiprocessors / geometry.query_heads);
    return row_blocks * kPromptWaveRows;
}

std::size_t causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage,
    CausalAttentionExecutionEnvelope envelope, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width, DeviceExecutionView execution) {
    require_causal_geometry(geometry, "causal_softmax_attention workspace");
    const std::int32_t q_heads = geometry.query_heads;
    bool supported_dtype       = true;
    try {
        (void)paged_kv_storage_layout(cache_storage, kHeadDim);
    } catch (const std::invalid_argument&) { supported_dtype = false; }
    if (execution.multiprocessor_count <= 0 || !supported_dtype || batch_size <= 0 ||
        batch_size > kMaximumBatchSize || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumVerifyTokens) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument(
            "causal_softmax_attention workspace: invalid profile or interval");
    }

    if (cache_storage == KvCacheStorage::BFloat16)
        return detail::bf16_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                               execution.multiprocessor_count);

    if (cache_storage == KvCacheStorage::Fp8E4M3Row256)
        return detail::fp8_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                              execution.multiprocessor_count);

    if (cache_storage == KvCacheStorage::Int8Group64)
        return detail::int8_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                               execution.multiprocessor_count);

    if (cache_storage == KvCacheStorage::Nvfp4Group16)
        return detail::nvfp4_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                                execution.multiprocessor_count);

    return detail::k8v4_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                           execution.multiprocessor_count);
}

void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution) {
    causal_softmax_attention(q, k, v, positions, valid_columns, kv_table_rows, Tensor{}, geometry,
                             scale, cache, envelope, workspace, out, execution);
}

void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, const Tensor& tree_masks,
                              AttentionHeadGeometry geometry, float scale,
                              PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution) {
    constexpr const char* op = "causal_softmax_attention";
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument(std::string(op) + ": SM count must be positive");
    }
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       geometry, envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("causal_softmax_attention: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t kv_heads = geometry.kv_heads;
    require_shape(k, kHeadDim, kv_heads, width, batch, op, "k");
    require_shape(v, kHeadDim, kv_heads, width, batch, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    if (tree_masks.data != nullptr) {
        if (tree_masks.dtype != DType::I32 || width < 2 || width > 32 ||
            valid_columns.data == nullptr) {
            throw std::invalid_argument(std::string(op) +
                                        ": tree masks must be I32 with 2<=W<=32 and valid columns");
        }
        require_shape(tree_masks, width, batch, 1, 1, op, "tree masks");
        require_contiguous_nonnull(tree_masks, op, "tree masks");
    }

    if (cache.storage == KvCacheStorage::BFloat16) {
        detail::bf16_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows,
                                         tree_masks, scale, cache, envelope, workspace, out,
                                         execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        detail::fp8_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows,
                                        tree_masks, scale, cache, envelope, workspace, out,
                                        execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Int8Group64) {
        detail::int8_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows,
                                         tree_masks, scale, cache, envelope, workspace, out,
                                         execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        detail::nvfp4_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows,
                                          tree_masks, scale, cache, envelope, workspace, out,
                                          execution);
        return;
    }

    detail::k8v4_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, tree_masks,
                                     scale, cache, envelope, workspace, out, execution);
}

void causal_softmax_attention_cached(const Tensor& q, const Tensor& positions,
                                     AttentionHeadGeometry geometry, float scale,
                                     const PagedKVLayerView& cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     WorkspaceArena& workspace, Tensor& out,
                                     DeviceExecutionView execution) {
    constexpr const char* op = "causal_softmax_attention_cached";
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument(std::string(op) + ": SM count must be positive");
    }
    validate_attention_tensors(q, positions, out, geometry, cache, envelope, scale, op);

    if (cache.storage == KvCacheStorage::BFloat16) {
        detail::bf16_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                         execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        detail::fp8_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                        execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Int8Group64) {
        detail::int8_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                         execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        detail::nvfp4_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                          execution);
        return;
    }

    detail::k8v4_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                     execution);
}

} // namespace ninfer::ops
