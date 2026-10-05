#pragma once

// The Vision tower's mathematics, shared by Qwen3.5 and Qwen4Exp
// (docs/maintainer/qwen3_8-flash-next-design.md §19.3.2, "Ownership and the shared tower"). The
// stages hold the op sequence of one item and nothing about where its buffers live: Qwen3.5's
// VisionContext binds them to its workspace plan, VisionTowerPass to a layer-major multi-item
// pass. Program-free: no Program types, no media acquisition.

#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/vision_control.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

class VisionWeightStream;

// Buffers of one item's patch and position embedding (P patches).
struct VisionEmbedBuffers {
    Tensor position_ids;      // I32 [P, 2]: written here, read by every layer
    Tensor patches;           // BF16 [patch_width, P]
    Tensor pos_indices;       // I32 [4, P]
    Tensor pos_weights;       // FP32 [4, P]
    DeviceSpan patch_scratch; // linear workspace of the patch projection
};

// Buffers of one encoder layer over P patches.
struct VisionLayerBuffers {
    Tensor qkv;            // BF16 [3H, P]; its first [H, P] then holds the attention projection
    Tensor attention_norm; // BF16 [H, P]; then the attended heads
    Tensor mlp_up;         // BF16 [I, P]
    Tensor mlp_norm;       // BF16 [H, P]; then the MLP down projection
    DeviceSpan qkv_scratch, projection_scratch, up_scratch, down_scratch;
};

// Buffers of the patch merger: V = P / merge^2 merged tokens.
struct VisionMergerBuffers {
    Tensor normalized; // BF16 [H, P]
    Tensor hidden;     // BF16 [merger_width, V]; may alias the residual x
    DeviceSpan first_scratch, second_scratch;
};

// Uploads the item's (pageable) patches and control arrays on `stream`, then x = patch projection
// + bias + bilinear position embedding. x: BF16 [H, P]. With a weight stream, the projection waits
// for the streamed prelude weights (the uploads overlap it).
void vision_embed(cudaStream_t stream, const VisionConfig& config, const VisionParameters& parameters,
                  const VisionItemControl& control, std::span<const std::uint16_t> patches,
                  const VisionEmbedBuffers& buffers, Tensor& x, VisionWeightStream* weight_stream = nullptr);

// One encoder layer in place on x: LN, fused QKV + bias, 2-D RoPE, non-causal attention per
// segment (equal-length form), output projection + bias + residual, LN, fc1 + bias, tanh-GELU,
// fc2 + bias + residual.
void vision_layer(DeviceExecutionView execution, const VisionConfig& config, const VisionBlockParameters& block,
                  const Tensor& position_ids, std::int32_t segment_length, const VisionLayerBuffers& buffers,
                  Tensor& x);

// The merger: LN, view [merger_width, V], fc1 + bias, exact GELU, fc2 + bias into `output`
// (BF16 [out, V]).
void vision_merge(cudaStream_t stream, const VisionConfig& config, const VisionParameters& parameters,
                  const Tensor& x, const VisionMergerBuffers& buffers, Tensor& output);

// ---------------------------------------------------------------------------------- the pass

// One item of a pass: its pageable patches, its control plan and its caller-owned output.
struct VisionTowerItem {
    std::span<const std::uint16_t> patches;
    const VisionItemControl* control = nullptr;
    Tensor output; // BF16 [out, V]
};

// Device layout of a pass: each item's residual x and position ids, then one shared scope for
// whichever stage runs (embedding, a layer, the merger), sized for the largest item. The merger's
// hidden aliases the item's x. bytes ≈ 2,304·ΣP + 10,912·P_max for the Qwen3.5 tower.
struct VisionPassLayout {
    struct Item {
        TensorRegion x, position_ids;
    };
    std::vector<Item> items;
    TensorRegion patches, pos_indices, pos_weights, qkv, attention_norm, mlp_up, mlp_norm, normalized;
    LayoutRegion patch_scratch, qkv_scratch, projection_scratch, up_scratch, down_scratch;
    LayoutRegion merger_first_scratch, merger_second_scratch;
    std::int32_t max_patches = 0;
    std::size_t bytes        = 0;
};

[[nodiscard]] VisionPassLayout plan_vision_pass(const VisionConfig& config, const VisionParameters& parameters,
                                                std::span<const std::size_t> item_patches);

// A resumable, layer-major encode of several items (design §19.3.2): stage 0 embeds every item,
// stage 1 + l runs layer l for every item, stage depth + 1 merges every item into its output.
// Every Op runs per item with its own T = P_i, so an item's embedding is bitwise the same alone
// or with other items, while each layer's weights are read once for all items. With a weight
// stream, each stage waits for its weights' upload.
class VisionTowerPass {
public:
    VisionTowerPass(DeviceContext& device, const VisionConfig& config, const VisionParameters& parameters,
                    std::vector<VisionTowerItem> items, DeviceSpan arena, const VisionPassLayout& layout,
                    VisionWeightStream* weight_stream);

    [[nodiscard]] static std::uint32_t stages(std::uint32_t depth) noexcept { return depth + 2; }
    [[nodiscard]] std::uint32_t next_stage() const noexcept { return next_; }
    [[nodiscard]] bool done() const noexcept { return next_ == stages(static_cast<std::uint32_t>(parameters_.layers.size())); }
    // Enqueues stages [next_stage(), end_stage) on the compute stream.
    void advance(std::uint32_t end_stage);

private:
    DeviceContext& device_;
    const VisionConfig& config_;
    const VisionParameters& parameters_;
    std::vector<VisionTowerItem> items_;
    DeviceSpan arena_;
    const VisionPassLayout& layout_;
    VisionWeightStream* stream_;
    std::uint32_t next_ = 0;
};

} // namespace ninfer::models::qwen3_5::execution
