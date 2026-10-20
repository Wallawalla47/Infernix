#include "core/weight.h"
#include "infernix/ops/context_kv_materialize.h"

#include "core/layout.h"
#include "ops/common/dflash_rope_table.h"
#include "ops/context_kv_materialize/launch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

// The registered context widths: Qwen3.8-27B's DFlash2 (5120) and Qwen3.8-Flash-Next's (2560).
bool registered_hidden(std::int32_t hidden) { return hidden == 5120 || hidden == 2560; }
constexpr std::int32_t kKVSize     = 1024;
constexpr std::int32_t kHeadDim    = 128;
constexpr std::int32_t kKVHeads    = 8;
constexpr std::int32_t kCapacity   = 2048;
// The widest multi-request append: every row of an ngram copy round of 63 drafts.
constexpr std::int32_t kBatchedWidth = 64;
constexpr const char* kOp          = "context_kv_materialize";

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, std::int32_t n3, std::size_t alignment, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != n3 || !tensor.is_contiguous() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_weight(const Weight& weight, std::int32_t hidden, const char* name) {
    const std::uint64_t code_bytes =
        static_cast<std::uint64_t>(kKVSize) * static_cast<std::uint64_t>(hidden);
    const std::uint64_t scale_bytes =
        static_cast<std::uint64_t>(kKVSize) * static_cast<std::uint64_t>(hidden / 32) * 2U;
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.group_size != 32 ||
        weight.ndim != 2 || weight.n != kKVSize || weight.k != hidden ||
        weight.shape[0] != kKVSize || weight.shape[1] != hidden ||
        weight.padded_shape[0] != kKVSize || weight.padded_shape[1] != hidden ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < code_bytes + scale_bytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_profile(std::int32_t width, std::int32_t batch) {
    if (batch < 1 || batch > 8 || width < 1) {
        throw std::invalid_argument("context_kv_materialize: invalid W/B");
    }
    if (width <= kBatchedWidth) return;
    if (batch == 1 && width <= kCapacity) return;
    throw std::invalid_argument("context_kv_materialize: unsupported W/B profile");
}

void require_interval(std::int32_t batch, std::int32_t min_width, std::int32_t max_width) {
    if (batch < 1 || batch > 8 || min_width < 1 || max_width < min_width) {
        throw std::invalid_argument("context_kv_materialize workspace: invalid interval");
    }
    if (batch == 1) {
        if (max_width > kCapacity) {
            throw std::invalid_argument("context_kv_materialize workspace: W exceeds 2048");
        }
        return;
    }
    if (max_width > kBatchedWidth) {
        throw std::invalid_argument("context_kv_materialize workspace: batched W exceeds 64");
    }
}

template <class Allocator>
Tensor allocate_key_scratch(Allocator& allocator, std::int32_t columns) {
    return allocator.alloc(
        DType::FP32, {kKVSize, columns, static_cast<std::int32_t>(kContextKVMaterializeLayers)});
}

std::size_t key_scratch_capacity(std::int32_t columns) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_key_scratch(layout, columns);
    return layout.peak_bytes(1);
}

void validate_cache(const CyclicKVCacheLayerView& cache, std::int32_t padded,
                    std::int32_t lane_capacity) {
    if (cache.capacity != kCapacity ||
        cache.padded_capacity != static_cast<std::uint32_t>(padded) ||
        cache.num_kv_heads != kKVHeads || cache.head_dim != kHeadDim ||
        cache.lane_capacity != lane_capacity || padded < kCapacity || lane_capacity <= 0) {
        throw std::invalid_argument("context_kv_materialize: invalid cyclic cache geometry");
    }
    require_tensor(cache.k, DType::BF16, kHeadDim, padded, kKVHeads, lane_capacity, 16, "cache K");
    require_tensor(cache.v, DType::FP16, kHeadDim, padded, kKVHeads, lane_capacity, 16, "cache V");
}

} // namespace

std::size_t context_kv_materialize_workspace_capacity_bytes(std::int32_t batch_size,
                                                            std::int32_t min_width,
                                                            std::int32_t max_width) {
    require_interval(batch_size, min_width, max_width);
    std::size_t capacity = 0;
    // Every W admits max_count=0..W. Thus max_width includes the envelopes of all smaller W,
    // including scratch-requiring routes below the fully fused interval.
    for (int count = 1; count <= max_width; ++count) {
        const int columns = count * batch_size;
        if (detail::context_kv_materialize_uses_scratch(
                detail::context_kv_materialize_route(columns)))
            capacity = std::max(capacity, key_scratch_capacity(columns));
    }
    return capacity;
}

void context_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    float rope_theta, ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace,
    cudaStream_t stream) {
    const std::int32_t hidden = context.ne[0];
    const std::int32_t width  = context.ne[1];
    const std::int32_t batch  = context.ne[2];
    if (!registered_hidden(hidden)) {
        throw std::invalid_argument("context_kv_materialize: H must be 5120 or 2560");
    }
    if (!(rope_theta > 1.0F) || !std::isfinite(rope_theta)) {
        throw std::invalid_argument("context_kv_materialize: theta must be finite and above 1");
    }
    require_profile(width, batch);
    require_tensor(context, DType::BF16, hidden, width, batch, 1, 16, "context");
    require_tensor(positions, DType::I32, width, batch, 1, 1, alignof(std::int32_t), "positions");
    require_tensor(counts, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "counts");
    require_tensor(state_slots, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "state slots");
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(width) ||
        envelope.max_count > static_cast<std::uint32_t>(kCapacity)) {
        throw std::invalid_argument("context_kv_materialize: invalid execution envelope");
    }

    if (layers.front().cache.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("context_kv_materialize: padded capacity exceeds int32");
    }
    const std::int32_t padded = static_cast<std::int32_t>(layers.front().cache.padded_capacity);
    const std::int32_t lane_capacity = layers.front().cache.lane_capacity;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const ContextKVMaterializeLayerView& layer = layers[index];
        require_weight(layer.key_weight, hidden, "key weight");
        require_weight(layer.value_weight, hidden, "value weight");
        require_tensor(layer.key_norm_weight, DType::BF16, kHeadDim, 1, 1, 1, 4, "key norm weight");
        validate_cache(layer.cache, padded, lane_capacity);
    }

    if (envelope.max_count == 0) return;
    const int columns = envelope.max_count * batch;
    const auto route  = detail::context_kv_materialize_route(columns);
    auto scope        = workspace.scope();
    Tensor key_scratch;
    if (detail::context_kv_materialize_uses_scratch(route))
        key_scratch = allocate_key_scratch(workspace, columns);
    // theta^(-2j/128): the stored words of the 1e7 table, else prepare_rope's (so these keys rotate
    // exactly as a prepared rmsnorm_rope rotates the query block's).
    static constexpr std::array<double, 64> k1e7{INFERNIX_DFLASH_ROPE_1E7_INV_FREQUENCY};
    std::array<double, 64> inv_frequency = k1e7;
    if (rope_theta != 1.0e7F) {
        for (int pair = 0; pair < 64; ++pair) {
            inv_frequency[static_cast<std::size_t>(pair)] =
                std::pow(static_cast<double>(rope_theta), -2.0 * pair / kHeadDim);
        }
    }
    detail::context_kv_materialize_launch(context, positions, counts, state_slots, layers,
                                          inv_frequency, envelope, route, key_scratch, stream);
}

} // namespace infernix::ops
