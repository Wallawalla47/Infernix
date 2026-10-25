// infernix::ops - causal cached Softmax Attention validation and finite route dispatch.
#include "infernix/ops/softmax_attention.h"

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
#include "ops/softmax_attention/dense/causal_cache/vq2/launch.h"
#include "ops/softmax_attention/dense/causal_cache/column_invariant.h"
#include "core/arena.h"
#include "core/temporary_switches.h"
#include "ops/kv_cache/append/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr std::int32_t kHeadDim             = 256;
constexpr float kExpectedScale              = 0.0625f;
// The widest verification a multi-request call admits (an ngram copy round of 63 drafts).
constexpr std::int32_t kMaximumVerifyTokens = 64;
constexpr std::int32_t kMaximumBatchSize    = 8;
// Query rows of one eight-warp fast INT8 prompt CTA (int8/fast_tiled_launch.cuh checks it).
constexpr std::int32_t kPromptWaveRows = 128;

void require_causal_geometry(AttentionHeadGeometry geometry, const char* op) {
    // The registered geometries: 24/4 (27B), 16/2 (35B-A3B), 32/2 (122B-A10B), head dim 256.
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        !detail::causal_registered_heads(geometry.query_heads) ||
        geometry.kv_heads != detail::causal_kv_heads(geometry.query_heads)) {
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

// A column-invariant round staged column-major: q, k, v and out as [256, heads, 1, W * B] and
// positions as [1, W * B], column j's B rows contiguous from row j * B, and every column's
// validity as [W][B] (validity only for a masked round).
struct StagedColumns {
    Tensor q, k, v, out, positions, valid;
};

template <class Allocator>
StagedColumns stage_columns(Allocator& allocator, AttentionHeadGeometry geometry,
                            std::int32_t width, std::int32_t batch, bool masked) {
    const std::int32_t rows = width * batch;
    StagedColumns out;
    out.q         = allocator.alloc(DType::BF16, {kHeadDim, geometry.query_heads, 1, rows});
    out.k         = allocator.alloc(DType::BF16, {kHeadDim, geometry.kv_heads, 1, rows});
    out.v         = allocator.alloc(DType::BF16, {kHeadDim, geometry.kv_heads, 1, rows});
    out.out       = allocator.alloc(DType::BF16, {kHeadDim, geometry.query_heads, 1, rows});
    out.positions = allocator.alloc(DType::I32, {1, rows});
    if (masked) { out.valid = allocator.alloc(DType::I32, {batch, width}); }
    return out;
}

// INFERNIX_TMP_VERIFY_ROWS_ATTENTION (candidates §10.3 KW-01): a column-invariant round appends all
// of its columns' K/V in one launch, then attends every column as a row of one-column calls over
// the appended keys, up to kRowsPerCall rows per launch, instead of one append-and-attend call per
// column. Each row keeps the key partition and merge order its one-column call plans (for the
// round's batch), and a row reads only keys at or before its own position, which the append wrote
// exactly as the fused one-column append writes them; so every column's bits are unchanged. The
// formats whose appended rows are independent per token take it; the VQ2 and K4V2 exact window
// keeps the column order.
constexpr std::int32_t kRowsPerCall = kMaximumBatchSize;

bool verify_rows_attention(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
    case KvCacheStorage::Int8Group64:
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Nvfp4Group16:
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return temporary_switches::flag("INFERNIX_TMP_VERIFY_ROWS_ATTENTION");
    default:
        return false;
    }
}

std::size_t cached_rows_workspace_bytes(KvCacheStorage storage, std::int32_t heads,
                                        std::int32_t call_batch, std::int32_t rows,
                                        CausalAttentionExecutionEnvelope envelope, int sms) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return detail::bf16_kv_cached_rows_workspace_bytes(heads, call_batch, rows, envelope, sms);
    case KvCacheStorage::Int8Group64:
        return detail::int8_kv_cached_rows_workspace_bytes(heads, call_batch, rows, envelope, sms);
    case KvCacheStorage::Fp8E4M3Row256:
        return detail::fp8_kv_cached_rows_workspace_bytes(heads, call_batch, rows, envelope, sms);
    case KvCacheStorage::Nvfp4Group16:
        return detail::nvfp4_kv_cached_rows_workspace_bytes(heads, call_batch, rows, envelope, sms);
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return detail::k8v4_kv_cached_rows_workspace_bytes(heads, call_batch, rows, envelope, sms);
    default:
        throw std::logic_error("causal_softmax_attention: rows variant on an unsupported format");
    }
}

void cached_rows_attention(KvCacheStorage storage, const Tensor& q, const Tensor& positions,
                           const Tensor& valid, const Tensor& rows, float scale,
                           PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                           std::int32_t call_batch, WorkspaceArena& workspace, Tensor& out,
                           DeviceExecutionView execution) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return detail::bf16_kv_cached_rows_attention(q, positions, valid, rows, scale, cache,
                                                     envelope, call_batch, workspace, out, execution);
    case KvCacheStorage::Int8Group64:
        return detail::int8_kv_cached_rows_attention(q, positions, valid, rows, scale, cache,
                                                     envelope, call_batch, workspace, out, execution);
    case KvCacheStorage::Fp8E4M3Row256:
        return detail::fp8_kv_cached_rows_attention(q, positions, valid, rows, scale, cache,
                                                    envelope, call_batch, workspace, out, execution);
    case KvCacheStorage::Nvfp4Group16:
        return detail::nvfp4_kv_cached_rows_attention(q, positions, valid, rows, scale, cache,
                                                      envelope, call_batch, workspace, out, execution);
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return detail::k8v4_kv_cached_rows_attention(q, positions, valid, rows, scale, cache,
                                                     envelope, call_batch, workspace, out, execution);
    default:
        throw std::logic_error("causal_softmax_attention: rows variant on an unsupported format");
    }
}

// Column `column`'s B rows of a staged tensor whose rows are `row_bytes` long.
void* column_rows(const Tensor& staged, std::int32_t column, std::int32_t batch,
                  std::size_t row_bytes) {
    return static_cast<unsigned char*>(staged.data) +
           static_cast<std::size_t>(column) * batch * row_bytes;
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

    const auto routes = [&](std::int32_t first, std::int32_t last,
                            CausalAttentionExecutionEnvelope routed) -> std::size_t {
        const int sms = execution.multiprocessor_count;
        if (cache_storage == KvCacheStorage::BFloat16)
            return detail::bf16_kv_workspace_bytes(q_heads, batch_size, first, last, routed, sms);
        if (cache_storage == KvCacheStorage::Fp8E4M3Row256)
            return detail::fp8_kv_workspace_bytes(q_heads, batch_size, first, last, routed, sms);
        if (cache_storage == KvCacheStorage::Int8Group64)
            return detail::int8_kv_workspace_bytes(q_heads, batch_size, first, last, routed, sms);
        if (cache_storage == KvCacheStorage::Nvfp4Group16)
            return detail::nvfp4_kv_workspace_bytes(q_heads, batch_size, first, last, routed, sms);
        if (kv_storage_has_exact_window(cache_storage))
            return detail::vq_kv_workspace_bytes(q_heads, geometry.kv_heads, batch_size, first,
                                                 last, routed);
        return detail::k8v4_kv_workspace_bytes(q_heads, batch_size, first, last, routed, sms);
    };
    if (!envelope.column_invariant || max_width == 1) {
        return routes(min_width, max_width, envelope);
    }
    // Column by column: the staged column beside each one-column call's own workspace (a
    // one-column call also serves width one).
    CausalAttentionExecutionEnvelope column = envelope;
    column.column_invariant                 = false;
    WorkspaceLayoutBuilder layout;
    (void)stage_columns(layout, geometry, max_width, batch_size, true);
    std::size_t one = routes(1, 1, column);
    if (verify_rows_attention(cache_storage)) {
        // The rows variant: the round's table rows beside the widest rows call's partials.
        (void)layout.alloc(DType::I32, {max_width * batch_size});
        const std::int32_t rows = std::min(kRowsPerCall, max_width * batch_size);
        one = std::max(one, cached_rows_workspace_bytes(cache_storage, q_heads, batch_size, rows,
                                                        column, execution.multiprocessor_count));
    }
    if (one != 0) { (void)layout.alloc_bytes(one); }
    return layout.peak_bytes(1);
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

    if (envelope.column_invariant && width > 1 && tree_masks.data == nullptr) {
        // Column by column, in order: column j appends its K/V and attends as a one-column call,
        // after columns 0..j-1 have appended theirs, so it computes what that token decoded alone
        // computes. The round is staged column-major once, so each call reads its column in
        // place, and the outputs return in one scatter.
        auto scope            = workspace.scope();
        const bool masked     = valid_columns.data != nullptr;
        StagedColumns staged  = stage_columns(workspace, geometry, width, batch, masked);
        cudaStream_t stream   = execution.stream;
        const std::size_t qb  = static_cast<std::size_t>(kHeadDim) * geometry.query_heads * 2;
        const std::size_t kb  = static_cast<std::size_t>(kHeadDim) * kv_heads * 2;
        const std::size_t pb  = sizeof(std::int32_t);
        const detail::CausalColumnPermutation inputs[] = {
            {q.data, staged.q.data, static_cast<std::int32_t>(qb)},
            {k.data, staged.k.data, static_cast<std::int32_t>(kb)},
            {v.data, staged.v.data, static_cast<std::int32_t>(kb)},
            {positions.data, staged.positions.data, static_cast<std::int32_t>(pb)},
        };
        CausalAttentionExecutionEnvelope column_envelope = envelope;
        column_envelope.column_invariant                 = false;
        if (verify_rows_attention(cache.storage)) {
            // The rows variant: q and positions column-major (K/V are appended in place), every
            // column's table row, one append of the whole round, then the rows calls.
            const detail::CausalColumnPermutation gathered[] = {inputs[0], inputs[3]};
            detail::causal_columns_permute_launch(gathered, 2, width, batch, true, stream);
            if (masked) {
                detail::causal_column_valid_launch(valid_columns, staged.valid, width, batch, stream);
            }
            const std::int32_t rows = width * batch;
            Tensor table_rows       = workspace.alloc(DType::I32, {rows});
            detail::causal_column_table_rows_launch(kv_table_rows, table_rows, width, batch, stream);
            detail::kv_cache_append_batch_launch(k, v, positions, valid_columns, kv_table_rows,
                                                 cache, stream);
            for (std::int32_t first = 0; first < rows; first += kRowsPerCall) {
                const std::int32_t count = std::min(kRowsPerCall, rows - first);
                const Tensor call_q(static_cast<unsigned char*>(staged.q.data) + first * qb,
                                    DType::BF16, {kHeadDim, geometry.query_heads, 1, count});
                const Tensor call_positions(static_cast<std::int32_t*>(staged.positions.data) + first,
                                            DType::I32, {1, count});
                const Tensor call_rows(static_cast<std::int32_t*>(table_rows.data) + first,
                                       DType::I32, {count});
                Tensor call_valid;
                if (masked) {
                    call_valid = Tensor(static_cast<std::int32_t*>(staged.valid.data) + first,
                                        DType::I32, {count});
                }
                Tensor call_out(static_cast<unsigned char*>(staged.out.data) + first * qb,
                                DType::BF16, {kHeadDim, geometry.query_heads, 1, count});
                cached_rows_attention(cache.storage, call_q, call_positions, call_valid, call_rows,
                                      scale, cache, column_envelope, batch, workspace, call_out,
                                      execution);
            }
            const detail::CausalColumnPermutation output{staged.out.data, out.data,
                                                         static_cast<std::int32_t>(qb)};
            detail::causal_columns_permute_launch(&output, 1, width, batch, false, stream);
            return;
        }
        detail::causal_columns_permute_launch(inputs, 4, width, batch, true, stream);
        if (masked) {
            detail::causal_column_valid_launch(valid_columns, staged.valid, width, batch, stream);
        }
        for (std::int32_t column = 0; column < width; ++column) {
            const Tensor column_q(column_rows(staged.q, column, batch, qb), DType::BF16,
                                  {kHeadDim, geometry.query_heads, 1, batch});
            const Tensor column_k(column_rows(staged.k, column, batch, kb), DType::BF16,
                                  {kHeadDim, kv_heads, 1, batch});
            const Tensor column_v(column_rows(staged.v, column, batch, kb), DType::BF16,
                                  {kHeadDim, kv_heads, 1, batch});
            const Tensor column_positions(column_rows(staged.positions, column, batch, pb),
                                          DType::I32, {1, batch});
            Tensor column_out(column_rows(staged.out, column, batch, qb), DType::BF16,
                              {kHeadDim, geometry.query_heads, 1, batch});
            Tensor valid;
            if (masked) {
                valid = Tensor(static_cast<std::int32_t*>(staged.valid.data) +
                                   static_cast<std::size_t>(column) * batch,
                               DType::I32, {batch});
            }
            causal_softmax_attention(column_q, column_k, column_v, column_positions, valid,
                                     kv_table_rows, Tensor{}, geometry, scale, cache,
                                     column_envelope, workspace, column_out, execution);
        }
        const detail::CausalColumnPermutation output{staged.out.data, out.data,
                                                     static_cast<std::int32_t>(qb)};
        detail::causal_columns_permute_launch(&output, 1, width, batch, false, stream);
        return;
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

    if (kv_storage_has_exact_window(cache.storage)) {
        detail::vq_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows,
                                       tree_masks, scale, cache, envelope, workspace, out,
                                       execution.stream);
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

    if (kv_storage_has_exact_window(cache.storage)) {
        detail::vq_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                       execution.stream);
        return;
    }

    detail::k8v4_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                     execution);
}

} // namespace infernix::ops
