// The Program side of Qwen4Exp's Vision encode window (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.2): admission plans the window, reserve keeps the media, progress runs the layer-major pass
// in steps, prefill calls scatter the handoff's embeddings, and the handoff returns to the expert
// cache once the prompt's last call is enqueued (or the lane is released).

#include "models/qwen4_exp/program/program_impl.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::detail {

namespace vx = qwen3_5::execution;

std::shared_ptr<const VisionAdmission> ProgramImpl::plan_vision(const qwen3_5::PreparedPromptData& prompt) const {
    if (!parameters_.vision || !parameters_.model.config().vision) {
        throw std::invalid_argument("Qwen4Exp: images and video need an engine started with --vision");
    }
    const qwen3_5::VisionConfig& config = *parameters_.model.config().vision;
    auto out     = std::make_shared<VisionAdmission>();
    out->control = qwen3_5::plan_vision_control(prompt, config);
    std::vector<VisionWindowItem> items;
    items.reserve(out->control.items.size());
    for (std::size_t i = 0; i < out->control.items.size(); ++i) {
        const auto& plan = out->control.items[i];
        items.push_back({.token_begin = plan.token_begin,
                         .token_end   = plan.token_end,
                         .patches     = static_cast<std::uint32_t>(prompt.vision_items[i].patch_count),
                         .merged      = static_cast<std::uint32_t>(plan.merged_count)});
    }
    std::vector<std::size_t> patches;
    for (const auto& item : items) { patches.push_back(item.patches); }
    const std::size_t pass_bytes = vx::plan_vision_pass(config, *parameters_.vision, patches).bytes;
    const auto& overlay          = parameters_.model.overlay_vision();
    const std::size_t staging    = overlay ? vx::vision_staging_align(overlay->layout.staging_bytes) : 0;
    out->window = plan_vision_window(items, 0, pass_bytes, staging, work_capacity_, residency_->frame_stride(),
                                     static_cast<std::uint32_t>(parameters_.vision->merger_fc2.weight.n),
                                     config.hidden_size, config.depth);
    return out;
}

void ProgramImpl::vision_reserve(Lane& lane, std::shared_ptr<const VisionAdmission> admission,
                                 const qwen3_5::PreparedPromptData& prompt) {
    auto vision        = std::make_unique<LaneVision>();
    const auto& window = admission->window;
    if (window.items.empty()) {
        vision->encoded = true; // every item lies in the reused prefix
    } else {
        // The encoded items are a suffix of the prompt's items (those ending past the reused prefix).
        const std::uint32_t first = window.items.front();
        vision->control           = qwen3_5::build_vision_control(prompt, admission->control, first);
        for (std::size_t k = 0; k < vision->control.items.size(); ++k) {
            const auto& payload = prompt.media_payloads.at(first + k);
            const auto& control = vision->control.items[k];
            if (!payload || payload->patch_elements !=
                                control.patch_count * static_cast<std::size_t>(parameters_.model.config().vision->patch_width())) {
                throw std::invalid_argument("Qwen4Exp: a Vision item's patches do not match its grid");
            }
            vision->payloads.push_back(payload);
            for (const std::int32_t token : control.scatter_indices) {
                vision->visual.push_back(static_cast<std::uint32_t>(token));
            }
        }
        if (!std::is_sorted(vision->visual.begin(), vision->visual.end()) ||
            vision->visual.size() != window.encoded_tokens) {
            throw std::logic_error("Qwen4Exp: the encoded visual tokens disagree with the window plan");
        }
    }
    vision->admission = std::move(admission);
    lane.vision       = std::move(vision);
}

void ProgramImpl::vision_step(Lane& lane) {
    LaneVision& v                        = *lane.vision;
    const VisionWindowPlan& window       = v.admission->window;
    const qwen3_5::VisionConfig& config  = *parameters_.model.config().vision;
    const cudaStream_t s                 = device_.stream;
    const auto out                       = parameters_.vision->merger_fc2.weight.n;
    if (!v.pass) {
        // Lend the handoff (W) or handoff and window (L); promotions still landing in the run are
        // waited for by compute and the weight stream's transfer stream.
        const cudaStream_t writers[] = {device_.transfer_stream};
        v.lease   = residency_->lend(window.frames, s, writers);
        v.handoff = Tensor(v.lease.memory.data, DType::BF16, {out, static_cast<std::int32_t>(window.encoded_tokens)});
        DeviceSpan memory;
        if (window.lease) {
            const std::size_t at = align(window.handoff_bytes);
            memory = {static_cast<std::byte*>(v.lease.memory.data) + at, v.lease.memory.bytes - at};
        } else {
            // Placement W: the whole pass runs in this call from the prefill workspace, idle between
            // units and stream-ordered on compute.
            work_->reset();
            memory = work_->alloc_bytes(window.window_bytes);
        }
        const vx::VisionParameters* weights = &*parameters_.vision;
        std::size_t staging                 = 0;
        if (const auto& overlay = parameters_.model.overlay_vision()) {
            staging   = vx::vision_staging_align(overlay->layout.staging_bytes);
            v.stream  = std::make_unique<vx::VisionWeightStream>(device_, *overlay, static_cast<std::byte*>(memory.data));
            v.weights = v.stream->window_weights(*parameters_.vision);
            weights   = &*v.weights;
        }
        const DeviceSpan arena{static_cast<std::byte*>(memory.data) + staging, memory.bytes - staging};
        std::vector<std::size_t> patches;
        std::vector<vx::VisionTowerItem> items;
        std::size_t column = 0;
        for (std::size_t k = 0; k < v.control.items.size(); ++k) {
            const auto& control = v.control.items[k];
            patches.push_back(control.patch_count);
            const Tensor output(static_cast<std::byte*>(v.handoff.data) + 2ULL * out * column, DType::BF16,
                                {out, static_cast<std::int32_t>(control.merged_count)});
            items.push_back({.patches = v.payloads[k]->span(), .control = &control, .output = output});
            column += control.merged_count;
        }
        v.layout = std::make_unique<vx::VisionPassLayout>(vx::plan_vision_pass(config, *parameters_.vision, patches));
        if (v.layout->bytes > arena.bytes) { throw std::logic_error("Qwen4Exp: the Vision pass exceeds its window"); }
        v.timer.emplace(device_);
        v.timer->start();
        v.pass = std::make_unique<vx::VisionTowerPass>(device_, config, *weights, std::move(items), arena, *v.layout,
                                                       v.stream.get());
    }
    // Placement W: every stage now. L: `layers_per_step` layers per call; the embedding rides with the
    // first step and the merger with the last.
    const std::uint32_t stages = vx::VisionTowerPass::stages(config.depth);
    std::uint32_t end          = stages;
    if (window.lease) {
        const std::uint32_t next = v.pass->next_stage();
        end                      = (next == 0 ? 1 : next) + window.layers_per_step;
        if (end > config.depth) { end = stages; }
    }
    v.pass->advance(end);
    if (!v.pass->done()) { return; }
    if (v.stream) { v.stream->finish(s); }
    v.timer->record_stop();
    if (window.lease) {
        // Only the handoff stays lent; the window's frames go back once compute reaches here.
        const auto handoff_frames = static_cast<std::uint32_t>(
            (align(window.handoff_bytes) + residency_->frame_stride() - 1) / residency_->frame_stride());
        if (handoff_frames < v.lease.count) {
            ExpertResidency::FrameLease tail{.first = v.lease.first + handoff_frames, .count = v.lease.count - handoff_frames};
            residency_->give_back(tail, s);
            v.lease.count = handoff_frames;
        }
    }
    v.pass.reset();
    v.layout.reset();
    v.weights.reset();
    v.stream.reset();
    v.payloads.clear(); // the media payloads are not needed again
    v.encoded = true;
}

void ProgramImpl::vision_release(Lane& lane) noexcept {
    try {
        LaneVision& v        = *lane.vision;
        const cudaStream_t s = device_.stream;
        // Compute is ordered after every upload into the window before its memory goes back.
        if (v.stream) { v.stream->finish(s); }
        if (v.encoded && v.timer) { lane.vision_seconds = v.timer->elapsed_ms() * 1e-3; }
        v.pass.reset();
        v.layout.reset();
        v.weights.reset();
        v.stream.reset();
        if (v.lease.valid()) { residency_->give_back(v.lease, s); }
    } catch (...) {}
    lane.vision.reset();
}

const execution::VisionInput* ProgramImpl::stage_vision(Lane& lane, std::uint32_t begin, std::int32_t width) {
    if (!lane.vision || lane.vision->visual.empty()) { return nullptr; }
    const LaneVision& v = *lane.vision;
    auto* words         = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.vision_columns);
    const VisualSlice slice =
        visual_slice(v.visual, begin, begin + static_cast<std::uint32_t>(width), begin,
                     std::span<std::int32_t>(words, static_cast<std::size_t>(columns_)));
    if (slice.count == 0) { return nullptr; }
    const std::int32_t out = v.handoff.ne[0];
    auto* base             = static_cast<std::byte*>(io_device_.p);
    vision_input_          = {Tensor(static_cast<std::byte*>(v.handoff.data) + 2ULL * out * slice.first, DType::BF16,
                                     {out, static_cast<std::int32_t>(slice.count)}),
                              Tensor(base + io_layout_.vision_columns, DType::I32, {static_cast<std::int32_t>(slice.count)})};
    return &vision_input_;
}

const execution::VisionInput* ProgramImpl::stage_mtp_vision(Lane& lane, std::uint32_t first, std::int32_t cells,
                                                            std::int32_t subchunks) {
    if (!lane.vision || lane.vision->visual.empty() || cells <= 0) { return nullptr; }
    const LaneVision& v    = *lane.vision;
    const std::int32_t out = v.handoff.ne[0];
    auto* words            = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.mtp_vision_columns);
    auto* base             = static_cast<std::byte*>(io_device_.p);
    mtp_vision_inputs_.assign(static_cast<std::size_t>(subchunks), execution::VisionInput{});
    for (std::int32_t k = 0; k < subchunks; ++k) {
        // Cell c of sub-chunk k sits at first + 512k + c and takes token first + 512k + c + 1.
        const std::int32_t n    = std::min(execution::kMtpChunkColumns, cells - k * execution::kMtpChunkColumns);
        const std::uint32_t lo  = first + static_cast<std::uint32_t>(k * execution::kMtpChunkColumns) + 1U;
        const std::size_t at    = static_cast<std::size_t>(k) * execution::kMtpChunkColumns;
        const VisualSlice slice = visual_slice(v.visual, lo, lo + static_cast<std::uint32_t>(n), lo,
                                               std::span<std::int32_t>(words + at, static_cast<std::size_t>(n)));
        if (slice.count == 0) { continue; }
        mtp_vision_inputs_[static_cast<std::size_t>(k)] = {
            Tensor(static_cast<std::byte*>(v.handoff.data) + 2ULL * out * slice.first, DType::BF16,
                   {out, static_cast<std::int32_t>(slice.count)}),
            Tensor(base + io_layout_.mtp_vision_columns + 4ULL * at, DType::I32, {static_cast<std::int32_t>(slice.count)})};
    }
    return mtp_vision_inputs_.data();
}

} // namespace ninfer::models::qwen4_exp::detail
