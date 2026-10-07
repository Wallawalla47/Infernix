#include "models/qwen3_5/execution/vision_tower.h"

#include "core/layout.h"
#include "core/nvtx.h"
#include "models/qwen3_5/execution/vision_weight_stream.h"
#include "infernix/ops/add_bias.h"
#include "infernix/ops/gelu.h"
#include "infernix/ops/layer_norm.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/residual_add.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/softmax_attention.h"
#include "infernix/ops/vision_pos_embed.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace infernix::models::qwen3_5::execution {
namespace {

std::int32_t dim(std::uint64_t v) {
    if (v > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision dimension exceeds int32");
    }
    return static_cast<std::int32_t>(v);
}

constexpr std::size_t kAlignment = 256;

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

void project(const Tensor& x, const LinearParameters& p, Tensor& out, DeviceSpan scratch, cudaStream_t stream) {
    WorkspaceArena arena(scratch);
    ops::linear(x, p.weight, out, p.policy, arena, stream);
}

std::size_t capacity(const LinearParameters& p, std::int32_t first, std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k, p.policy, first, last);
}

// The first `columns` columns of a contiguous [rows, capacity] buffer, as [rows, columns].
Tensor columns_of(const Tensor& storage, std::int32_t columns) {
    return Tensor(storage.data, storage.dtype, {storage.ne[0], columns});
}

} // namespace

void vision_embed(cudaStream_t stream, const VisionConfig& config, const VisionParameters& parameters,
                  const VisionItemControl& control, std::span<const std::uint16_t> patches,
                  const VisionEmbedBuffers& buffers, Tensor& x, VisionWeightStream* weight_stream) {
    nvtx::ScopedRange patch_range(nvtx::Name::VisionPatchEmbedding, nvtx::Category::Vision,
                                  static_cast<std::uint64_t>(control.patch_count));
    Tensor position_ids = buffers.position_ids;
    Tensor patch_bf16   = buffers.patches;
    Tensor pos_indices  = buffers.pos_indices;
    Tensor pos_weights  = buffers.pos_weights;
    copy_host(control.position_ids.data(), position_ids, stream);
    copy_host(patches.data(), patch_bf16, stream);
    if (weight_stream != nullptr) { weight_stream->prelude_ready(stream); }
    project(patch_bf16, parameters.patch_embedding, x, buffers.patch_scratch, stream);
    ops::add_bias(parameters.patch_embedding_bias, x, stream);
    // The artifact records the source table shape [rows, hidden], while Tensor's contiguous matrix
    // convention is [inner, columns]. The payload is already row-major, so this is a zero-copy
    // [hidden, rows] view, not a transpose.
    copy_host(control.position_table_indices.data(), pos_indices, stream);
    copy_host(control.position_table_weights.data(), pos_weights, stream);
    Tensor position_table =
        parameters.position_embedding.reshape({dim(config.hidden_size), dim(config.num_position_embeddings)});
    ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
}

void vision_layer(DeviceExecutionView execution, const VisionConfig& config, const VisionBlockParameters& block,
                  const Tensor& position_ids, std::int32_t segment_length, const VisionLayerBuffers& buffers,
                  Tensor& x) {
    const cudaStream_t stream  = execution.stream;
    const std::int32_t patches = x.ne[1];
    const std::int32_t H = dim(config.hidden_size), heads = dim(config.num_heads), D = H / heads;
    {
        nvtx::ScopedRange attention_range(nvtx::Name::VisionAttention, nvtx::Category::Attention, 0);
        Tensor attended = buffers.attention_norm;
        {
            Tensor qkv = buffers.qkv;
            {
                Tensor h = buffers.attention_norm;
                ops::layer_norm(x, block.norm1.weight, block.norm1.bias, 1.0e-6F, h, stream);
                project(h, block.qkv, qkv, buffers.qkv_scratch, stream);
            }
            ops::add_bias(block.qkv_bias, qkv, stream);
            const std::size_t plane_bytes = static_cast<std::size_t>(H) * 2;
            Tensor q(qkv.data, DType::BF16, {D, heads, patches});
            Tensor k(static_cast<unsigned char*>(qkv.data) + plane_bytes, DType::BF16, {D, heads, patches});
            Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane_bytes, DType::BF16, {D, heads, patches});
            q.nb[2] = qkv.nb[1];
            k.nb[2] = qkv.nb[1];
            v.nb[2] = qkv.nb[1];
            ops::rope(position_ids, D, 10'000.0F, q, k, execution);
            Tensor attended_heads = attended.view({D, heads, patches});
            ops::packed_softmax_attention(q, k, v, {D, heads, heads},
                                          static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))),
                                          segment_length, attended_heads, stream);
        }
        Tensor projected(buffers.qkv.data, DType::BF16, {H, patches});
        project(attended, block.output, projected, buffers.projection_scratch, stream);
        ops::add_bias(block.output_bias, projected, stream);
        ops::residual_add(projected, x, stream);
    }
    {
        nvtx::ScopedRange mlp_range(nvtx::Name::VisionMlp, nvtx::Category::PostMixer, 0);
        Tensor down = buffers.mlp_norm;
        Tensor up   = buffers.mlp_up;
        {
            Tensor h = buffers.mlp_norm;
            ops::layer_norm(x, block.norm2.weight, block.norm2.bias, 1.0e-6F, h, stream);
            project(h, block.fc1, up, buffers.up_scratch, stream);
        }
        ops::add_bias(block.fc1_bias, up, stream);
        ops::gelu(up, ops::GeluMode::Tanh, stream);
        project(up, block.fc2, down, buffers.down_scratch, stream);
        ops::add_bias(block.fc2_bias, down, stream);
        ops::residual_add(down, x, stream);
    }
}

void vision_merge(cudaStream_t stream, const VisionConfig& config, const VisionParameters& parameters,
                  const Tensor& x, const VisionMergerBuffers& buffers, Tensor& output) {
    const std::int32_t tokens = output.ne[1];
    nvtx::ScopedRange merge_range(nvtx::Name::VisionMerge, nvtx::Category::Vision, static_cast<std::uint64_t>(tokens));
    Tensor normalized = buffers.normalized;
    ops::layer_norm(x, parameters.merger_norm.weight, parameters.merger_norm.bias, 1.0e-6F, normalized, stream);
    Tensor merged = normalized.view({dim(config.merger_width()), tokens});
    Tensor hidden = buffers.hidden;
    project(merged, parameters.merger_fc1, hidden, buffers.first_scratch, stream);
    ops::add_bias(parameters.merger_fc1_bias, hidden, stream);
    ops::gelu(hidden, ops::GeluMode::Exact, stream);
    project(hidden, parameters.merger_fc2, output, buffers.second_scratch, stream);
    ops::add_bias(parameters.merger_fc2_bias, output, stream);
}

// ---------------------------------------------------------------------------------- the pass

VisionPassLayout plan_vision_pass(const VisionConfig& config, const VisionParameters& parameters,
                                  std::span<const std::size_t> item_patches) {
    const std::size_t merge = std::size_t(config.spatial_merge_size) * config.spatial_merge_size;
    if (item_patches.empty()) { throw std::invalid_argument("Vision pass has no items"); }
    VisionPassLayout out;
    std::size_t max_patches = 0;
    for (const std::size_t p : item_patches) {
        if (p == 0 || p % merge != 0) { throw std::invalid_argument("Vision pass item patches must be a positive multiple of merge^2"); }
        max_patches = std::max(max_patches, p);
    }
    out.max_patches               = dim(max_patches);
    const std::int32_t P          = out.max_patches;
    const std::int32_t V          = dim(max_patches / merge);
    const std::int32_t H          = dim(config.hidden_size);
    const std::int32_t minimum    = dim(merge);
    std::size_t qkv_bytes = 0, projection_bytes = 0, up_bytes = 0, down_bytes = 0;
    for (const auto& layer : parameters.layers) {
        qkv_bytes        = std::max(qkv_bytes, capacity(layer.qkv, minimum, P));
        projection_bytes = std::max(projection_bytes, capacity(layer.output, minimum, P));
        up_bytes         = std::max(up_bytes, capacity(layer.fc1, minimum, P));
        down_bytes       = std::max(down_bytes, capacity(layer.fc2, minimum, P));
    }
    LayoutBuilder builder;
    const auto scratch = [&](std::size_t bytes, const char* label) {
        // A borrowed WorkspaceArena needs backing even when its Op uses no scratch.
        return builder.add(std::max(std::size_t{1}, bytes), kAlignment, label);
    };
    // Live for the whole pass: each item's residual and position ids.
    for (const std::size_t p : item_patches) {
        VisionPassLayout::Item item;
        item.x            = builder.add_tensor(DType::BF16, {H, dim(p)}, kAlignment, "vision pass residual");
        item.position_ids = builder.add_tensor(DType::I32, {dim(p), 2}, kAlignment, "vision pass position ids");
        out.items.push_back(item);
    }
    // One stage's buffers at a time, sized for the largest item.
    {
        auto scope        = builder.scope();
        out.patches       = builder.add_tensor(DType::BF16, {dim(config.patch_width()), P}, kAlignment, "vision pass patches");
        out.pos_indices   = builder.add_tensor(DType::I32, {4, P}, kAlignment, "vision pass position indices");
        out.pos_weights   = builder.add_tensor(DType::FP32, {4, P}, kAlignment, "vision pass position weights");
        out.patch_scratch = scratch(capacity(parameters.patch_embedding, minimum, P), "vision pass patch scratch");
    }
    {
        auto scope             = builder.scope();
        out.qkv                = builder.add_tensor(DType::BF16, {3 * H, P}, kAlignment, "vision pass QKV");
        out.attention_norm     = builder.add_tensor(DType::BF16, {H, P}, kAlignment, "vision pass attention norm");
        out.qkv_scratch        = scratch(qkv_bytes, "vision pass QKV scratch");
        out.projection_scratch = scratch(projection_bytes, "vision pass projection scratch");
    }
    {
        auto scope       = builder.scope();
        out.mlp_up       = builder.add_tensor(DType::BF16, {dim(config.intermediate_size), P}, kAlignment, "vision pass MLP up");
        out.mlp_norm     = builder.add_tensor(DType::BF16, {H, P}, kAlignment, "vision pass MLP norm");
        out.up_scratch   = scratch(up_bytes, "vision pass MLP up scratch");
        out.down_scratch = scratch(down_bytes, "vision pass MLP down scratch");
    }
    {
        auto scope                 = builder.scope();
        out.normalized             = builder.add_tensor(DType::BF16, {H, P}, kAlignment, "vision pass merger norm");
        out.merger_first_scratch   = scratch(capacity(parameters.merger_fc1, 1, V), "vision pass merger first scratch");
        out.merger_second_scratch  = scratch(capacity(parameters.merger_fc2, 1, V), "vision pass merger second scratch");
    }
    out.bytes = builder.finish(kAlignment, "vision pass");
    return out;
}

VisionTowerPass::VisionTowerPass(DeviceContext& device, const VisionConfig& config, const VisionParameters& parameters,
                                 std::vector<VisionTowerItem> items, DeviceSpan arena, const VisionPassLayout& layout,
                                 VisionWeightStream* weight_stream)
    : device_(device), config_(config), parameters_(parameters), items_(std::move(items)), arena_(arena),
      layout_(layout), stream_(weight_stream) {
    if (items_.size() != layout_.items.size() || arena_.data == nullptr || arena_.bytes < layout_.bytes) {
        throw std::invalid_argument("Vision pass items or arena disagree with the layout");
    }
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const VisionTowerItem& item = items_[i];
        const Tensor x              = layout_.items[i].x.bind(arena_);
        if (item.control == nullptr || static_cast<std::size_t>(x.ne[1]) != item.control->patch_count ||
            item.patches.size() != item.control->patch_count * static_cast<std::size_t>(config_.patch_width()) ||
            item.output.dtype != DType::BF16 || item.output.ne[0] != parameters_.merger_fc2.weight.n ||
            static_cast<std::size_t>(item.output.ne[1]) != item.control->merged_count || !item.output.is_contiguous()) {
            throw std::invalid_argument("Vision pass item does not match its layout or output");
        }
    }
}

void VisionTowerPass::advance(std::uint32_t end_stage) {
    const auto depth = static_cast<std::uint32_t>(parameters_.layers.size());
    end_stage        = std::min(end_stage, stages(depth));
    const cudaStream_t s = device_.stream;
    for (; next_ < end_stage; ++next_) {
        if (next_ == 0) {
            for (std::size_t i = 0; i < items_.size(); ++i) {
                const VisionTowerItem& item = items_[i];
                const auto P                = static_cast<std::int32_t>(item.control->patch_count);
                Tensor x                    = layout_.items[i].x.bind(arena_);
                const VisionEmbedBuffers buffers{.position_ids  = layout_.items[i].position_ids.bind(arena_),
                                                 .patches       = columns_of(layout_.patches.bind(arena_), P),
                                                 .pos_indices   = columns_of(layout_.pos_indices.bind(arena_), P),
                                                 .pos_weights   = columns_of(layout_.pos_weights.bind(arena_), P),
                                                 .patch_scratch = layout_.patch_scratch.bind(arena_)};
                vision_embed(s, config_, parameters_, *item.control, item.patches, buffers, x, stream_);
            }
        } else if (next_ <= depth) {
            const std::uint32_t layer = next_ - 1;
            nvtx::ScopedRange layer_range(nvtx::Name::VisionLayer, nvtx::Category::Vision, layer);
            if (stream_ != nullptr) { stream_->arrive(layer, s); }
            for (std::size_t i = 0; i < items_.size(); ++i) {
                const VisionTowerItem& item = items_[i];
                const auto P                = static_cast<std::int32_t>(item.control->patch_count);
                Tensor x                    = layout_.items[i].x.bind(arena_);
                const VisionLayerBuffers buffers{.qkv                = columns_of(layout_.qkv.bind(arena_), P),
                                                 .attention_norm     = columns_of(layout_.attention_norm.bind(arena_), P),
                                                 .mlp_up             = columns_of(layout_.mlp_up.bind(arena_), P),
                                                 .mlp_norm           = columns_of(layout_.mlp_norm.bind(arena_), P),
                                                 .qkv_scratch        = layout_.qkv_scratch.bind(arena_),
                                                 .projection_scratch = layout_.projection_scratch.bind(arena_),
                                                 .up_scratch         = layout_.up_scratch.bind(arena_),
                                                 .down_scratch       = layout_.down_scratch.bind(arena_)};
                vision_layer(device_.execution_view().on_stream(s), config_, parameters_.layers[layer],
                             layout_.items[i].position_ids.bind(arena_),
                             item.control->segment_length, buffers, x);
            }
        } else {
            if (stream_ != nullptr) { stream_->merger_ready(s); }
            for (std::size_t i = 0; i < items_.size(); ++i) {
                VisionTowerItem& item = items_[i];
                const auto P          = static_cast<std::int32_t>(item.control->patch_count);
                const auto V          = static_cast<std::int32_t>(item.control->merged_count);
                const Tensor x        = layout_.items[i].x.bind(arena_);
                const VisionMergerBuffers buffers{
                    .normalized     = columns_of(layout_.normalized.bind(arena_), P),
                    .hidden         = Tensor(x.data, DType::BF16, {dim(config_.merger_width()), V}),
                    .first_scratch  = layout_.merger_first_scratch.bind(arena_),
                    .second_scratch = layout_.merger_second_scratch.bind(arena_)};
                vision_merge(s, config_, parameters_, x, buffers, item.output);
            }
        }
    }
}

} // namespace infernix::models::qwen3_5::execution
