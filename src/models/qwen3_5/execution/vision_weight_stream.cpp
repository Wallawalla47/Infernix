#include "models/qwen3_5/execution/vision_weight_stream.h"

#include <stdexcept>

namespace infernix::models::qwen3_5::execution {
namespace {

// Rebases a pointer inside a group's pinned bytes into the group's staged slot, through the
// segment that contains it. A group's objects are not necessarily adjacent in the pinned block,
// so each object is staged independently.
const void* stage_pointer(const VisionOverlayGroup& group, std::byte* staging, const std::byte* block,
                          const void* pointer) {
    if (pointer == nullptr) { return nullptr; }
    const std::size_t pinned = static_cast<std::size_t>(static_cast<const std::byte*>(pointer) - block);
    for (const VisionOverlaySegment& segment : group.segments) {
        if (pinned >= segment.pinned_offset && pinned < segment.pinned_offset + segment.bytes) {
            return staging + segment.staging_offset + (pinned - segment.pinned_offset);
        }
    }
    throw std::logic_error("overlay weight is outside its staged group");
}

Tensor stage_tensor(const VisionOverlayGroup& group, std::byte* staging, const std::byte* block, const Tensor& source) {
    Tensor out = source;
    out.data   = const_cast<std::byte*>(static_cast<const std::byte*>(stage_pointer(group, staging, block, source.data)));
    return out;
}

Weight stage_weight(const VisionOverlayGroup& group, std::byte* staging, const std::byte* block, const Weight& source) {
    Weight out  = source;
    out.payload = static_cast<const std::byte*>(stage_pointer(group, staging, block, source.payload));
    out.qdata   = stage_pointer(group, staging, block, source.qdata);
    out.qhigh   = stage_pointer(group, staging, block, source.qhigh);
    out.scales  = stage_pointer(group, staging, block, source.scales);
    return out;
}

// Copies a group's pinned segments into its device slot on `stream`.
void stage_group(cudaStream_t stream, std::byte* staging, const VisionOverlayGroup& group, const std::byte* block) {
    for (const VisionOverlaySegment& segment : group.segments) {
        CUDA_CHECK(cudaMemcpyAsync(staging + segment.staging_offset, block + segment.pinned_offset, segment.bytes,
                                   cudaMemcpyHostToDevice, stream));
    }
}

} // namespace

VisionWeightStream::VisionWeightStream(DeviceContext& device, const VisionOverlayAssets& assets, std::byte* staging,
                                       std::span<const cudaEvent_t> wait_before_first_upload)
    : device_(device), assets_(assets) {
    const VisionOverlayLayout& layout = assets.layout;
    prelude_ = staging;
    merger_  = prelude_ + vision_staging_align(layout.prelude.bytes);
    slot_[0] = merger_ + vision_staging_align(layout.merger.bytes);
    slot_[1] = slot_[0] + vision_staging_align(layout.slot_bytes);
    for (cudaEvent_t& event : uploaded_) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    for (cudaEvent_t* event : {&prelude_event_, &merger_event_, &compute_fence_, &final_event_}) {
        CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    }
    // Nothing reaches the staging memory before its previous users are done.
    CUDA_CHECK(cudaEventRecord(compute_fence_, device_.stream));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
    for (const cudaEvent_t event : wait_before_first_upload) { CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), event, 0)); }
    stage_group(copy_stream(), prelude_, layout.prelude, assets.pinned_block);
    CUDA_CHECK(cudaEventRecord(prelude_event_, copy_stream()));
    upload_bytes_ = layout.prelude.bytes;
    next_upload_  = 0;
    upload_next_layer();
    upload_next_layer();
}

VisionWeightStream::~VisionWeightStream() {
    if (!finished_) { (void)cudaStreamSynchronize(copy_stream()); }
    for (cudaEvent_t event : uploaded_) { (void)cudaEventDestroy(event); }
    for (cudaEvent_t event : {prelude_event_, merger_event_, compute_fence_, final_event_}) {
        (void)cudaEventDestroy(event);
    }
}

VisionParameters VisionWeightStream::window_weights(const VisionParameters& host) const {
    const VisionOverlayLayout& layout = assets_.layout;
    const std::byte* block            = assets_.pinned_block;
    const VisionOverlayGroup& prelude = layout.prelude;
    const VisionOverlayGroup& merger  = layout.merger;

    VisionParameters out       = host;
    out.patch_embedding.weight = stage_weight(prelude, prelude_, block, host.patch_embedding.weight);
    out.patch_embedding_bias   = stage_tensor(prelude, prelude_, block, host.patch_embedding_bias);
    out.position_embedding     = stage_tensor(prelude, prelude_, block, host.position_embedding);
    for (std::size_t layer = 0; layer < host.layers.size(); ++layer) {
        const VisionOverlayGroup& group  = layout.layers[layer];
        std::byte* staging               = slot_[layer % 2];
        const VisionBlockParameters& src = host.layers[layer];
        VisionBlockParameters dst        = src;
        dst.norm1.weight  = stage_tensor(group, staging, block, src.norm1.weight);
        dst.norm1.bias    = stage_tensor(group, staging, block, src.norm1.bias);
        dst.norm2.weight  = stage_tensor(group, staging, block, src.norm2.weight);
        dst.norm2.bias    = stage_tensor(group, staging, block, src.norm2.bias);
        dst.qkv.weight    = stage_weight(group, staging, block, src.qkv.weight);
        dst.qkv_bias      = stage_tensor(group, staging, block, src.qkv_bias);
        dst.output.weight = stage_weight(group, staging, block, src.output.weight);
        dst.output_bias   = stage_tensor(group, staging, block, src.output_bias);
        dst.fc1.weight    = stage_weight(group, staging, block, src.fc1.weight);
        dst.fc1_bias      = stage_tensor(group, staging, block, src.fc1_bias);
        dst.fc2.weight    = stage_weight(group, staging, block, src.fc2.weight);
        dst.fc2_bias      = stage_tensor(group, staging, block, src.fc2_bias);
        out.layers[layer] = dst;
    }
    out.merger_norm.weight = stage_tensor(merger, merger_, block, host.merger_norm.weight);
    out.merger_norm.bias   = stage_tensor(merger, merger_, block, host.merger_norm.bias);
    out.merger_fc1.weight  = stage_weight(merger, merger_, block, host.merger_fc1.weight);
    out.merger_fc1_bias    = stage_tensor(merger, merger_, block, host.merger_fc1_bias);
    out.merger_fc2.weight  = stage_weight(merger, merger_, block, host.merger_fc2.weight);
    out.merger_fc2_bias    = stage_tensor(merger, merger_, block, host.merger_fc2_bias);
    return out;
}

void VisionWeightStream::reset(cudaStream_t compute) {
    CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
    next_upload_ = 0;
    upload_next_layer();
    upload_next_layer();
}

void VisionWeightStream::prelude_ready(cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, prelude_event_, 0));
}

void VisionWeightStream::merger_ready(cudaStream_t compute) {
    if (!merger_uploaded_) { throw std::logic_error("Vision weight stream: the merger waits for the last layer's upload"); }
    CUDA_CHECK(cudaStreamWaitEvent(compute, merger_event_, 0));
}

void VisionWeightStream::arrive(std::uint32_t layer, cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[layer % 2], 0));
    const auto layers = static_cast<std::uint32_t>(assets_.layout.layers.size());
    if (next_upload_ == layer + 1 && next_upload_ < layers) {
        CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
        upload_next_layer();
    }
}

void VisionWeightStream::finish(cudaStream_t compute) {
    CUDA_CHECK(cudaEventRecord(final_event_, copy_stream()));
    CUDA_CHECK(cudaStreamWaitEvent(compute, final_event_, 0));
    finished_ = true;
}

void VisionWeightStream::upload_next_layer() {
    const auto layers = static_cast<std::uint32_t>(assets_.layout.layers.size());
    if (next_upload_ >= layers) { return; }
    const std::uint32_t layer       = next_upload_++;
    const VisionOverlayGroup& group = assets_.layout.layers[layer];
    stage_group(copy_stream(), slot_[layer % 2], group, assets_.pinned_block);
    CUDA_CHECK(cudaEventRecord(uploaded_[layer % 2], copy_stream()));
    upload_bytes_ += group.bytes;
    // The merger follows the last layer, so it never delays layer 0.
    if (next_upload_ == layers && !merger_uploaded_) {
        stage_group(copy_stream(), merger_, assets_.layout.merger, assets_.pinned_block);
        CUDA_CHECK(cudaEventRecord(merger_event_, copy_stream()));
        upload_bytes_ += assets_.layout.merger.bytes;
        merger_uploaded_ = true;
    }
}

} // namespace infernix::models::qwen3_5::execution
