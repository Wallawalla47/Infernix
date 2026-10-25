// The DFlash2 drafter in Qwen4Exp's Program (docs/maintainer/dflash.md, Qwen3.8-Flash-Next): every
// lane keeps a context ring of the drafter's K/V for its last `window` positions, projected from the
// target's features (the stream mean of the tapped layers' completed residuals). A speculating round
// proposes once (execution::DFlash2::propose, one graph per batch size); its drafts and their laws
// are verified by sparse rejection sampling; the commit then writes the committed columns' context
// into the ring (one graph per batch size and width). Prefill calls, forced tokens and layer walks
// write their last window's context eagerly. The ring rides in prefix snapshots (the state image's
// drafter group); its contents only steer drafts, which verification accepts or rejects, so a stale
// ring costs acceptance, never output.

#include "models/qwen4_exp/program/program_impl.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::qwen4_exp::detail {

void ProgramImpl::allocate_dflash2(std::int32_t lanes, std::uint64_t& allocated) {
    dflash_io_ = plan_.dflash_io;
    allocate(dflash_features_, plan_.dflash_features, allocated);
    // A zero ring: every slot a proposal may read before its context lands holds finite values.
    allocate(dflash_ring_backing_, plan_.dflash_ring, allocated);
    dflash_ring_backing_.fill(0);
    dflash_ring_ = std::make_unique<CyclicKVCache>(DeviceSpan{dflash_ring_backing_.p, dflash_ring_backing_.bytes},
                                                   plan_.dflash_ring_layout);
    allocate(dflash_device_, 4ULL * dflash_io_.words, allocated);
    dflash_device_.fill(0);
    dflash_host_ = PinnedHostBuffer(4ULL * dflash_io_.words);
    for (auto& drafts : dflash_drafts_) { drafts.assign(static_cast<std::size_t>(dflash_k_), 0); }
    dflash_propose_graphs_.resize(static_cast<std::size_t>(lanes));
    dflash_context_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_);
}

void ProgramImpl::create_dflash2_drafter() {
    std::vector<CyclicKVCacheLayerView> ring;
    for (std::uint32_t l = 0; l < dflash_ring_->layer_count(); ++l) { ring.push_back(dflash_ring_->layer_view(l)); }
    // --lm-head-draft binds the proposal head; the drafts then read it instead of the text head.
    const ProposalHead head = parameters_.draft_head.rows ? ProposalHead::Optimized : ProposalHead::Full;
    drafter_ = std::make_unique<execution::DFlash2>(parameters_, *parameters_.model.config().dflash2, head, device_,
                                                    *work_, std::move(ring), options_.max_context);
}

void ProgramImpl::dflash2_propose(std::span<const std::uint32_t> lanes, const std::function<void()>& while_drafting) {
    const cudaStream_t s = device_.stream;
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    const std::int32_t K = dflash_k_, W = K + 1;
    constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
    // Each row's block starts at its unprocessed anchor; the ring holds every earlier position. The
    // selector draws a sampled row's drafts with its seed (verify uploads the configs again, with
    // grammar masks, after this round's synchronize).
    auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
    for (std::int32_t b = 0; b < batch; ++b) {
        const Lane& lane                     = lanes_[lanes[b]];
        dflash_host(dflash_io_.anchors)[b]   = lane.history.back();
        dflash_host(dflash_io_.frontiers)[b] = static_cast<std::int32_t>(lane.state_tokens);
        dflash_host(dflash_io_.valid)[b]     = W;
        dflash_host(dflash_io_.slots)[b]     = static_cast<std::int32_t>(lanes[b]);
        configs[b]                           = lane.sampling;
        configs[b].mask                      = {};
        configs[b].nonfinite                 = nullptr; // a draft's selection never fails the request
    }
    upload_pinned(dflash_device_.p, dflash_host_.data(), 4ULL * dflash_io_.ids, s);
    upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
    const execution::DFlash2ProposalCall call{
        .anchors       = Tensor(dflash_device(dflash_io_.anchors), DType::I32, {batch}),
        .frontiers     = Tensor(dflash_device(dflash_io_.frontiers), DType::I32, {batch}),
        .valid_columns = Tensor(dflash_device(dflash_io_.valid), DType::I32, {batch}),
        .slots         = Tensor(dflash_device(dflash_io_.slots), DType::I32, {batch}),
        .configs       = static_cast<const ops::SamplingConfig*>(configs_.p),
        .batch         = batch,
        .width         = W,
        .ids           = Tensor(dflash_device(dflash_io_.ids), DType::I32, {W, batch}),
        .positions     = Tensor(dflash_device(dflash_io_.positions), DType::I32, {W, batch}),
        .drafts        = Tensor(dflash_device(dflash_io_.drafts), DType::I32, {K, batch}),
        .candidates    = Tensor(dflash_device(dflash_io_.candidates), DType::I32, {C, K, batch}),
        .proposal_q    = Tensor(dflash_device(dflash_io_.q), DType::FP32, {C, K, batch}),
        .confidence    = Tensor(dflash_device(dflash_io_.confidence), DType::FP32, {K, batch})};
    replay(dflash_propose_graphs_[static_cast<std::size_t>(batch - 1)], [&] { drafter_->propose(call); });
    // The drafts and their probabilities (adjacent regions) in one copy.
    CUDA_CHECK(cudaMemcpyAsync(dflash_host(dflash_io_.drafts), dflash_device(dflash_io_.drafts),
                               4ULL * (dflash_io_.confidence - dflash_io_.drafts + static_cast<std::size_t>(K) * batch),
                               cudaMemcpyDeviceToHost, s));
    if (while_drafting) {
        device_.flush();
        while_drafting();
    }
    device_.synchronize();
    const auto* confidence = reinterpret_cast<const float*>(dflash_host(dflash_io_.confidence));
    for (std::int32_t b = 0; b < batch; ++b) {
        dflash_confident_[b] = K;
        for (std::int32_t j = 0; j < K; ++j) {
            dflash_drafts_[b][static_cast<std::size_t>(j)] = dflash_host(dflash_io_.drafts)[b * K + j];
            dflash_confidence_[b][static_cast<std::size_t>(j)] = confidence[b * K + j];
            if (dflash_confident_[b] == K && static_cast<double>(confidence[b * K + j]) < options_.draft_min_p) {
                dflash_confident_[b] = j;
            }
        }
    }
}

std::pair<std::size_t, std::size_t> ProgramImpl::dflash2_verify_planes(std::int32_t batch, std::int32_t K) {
    const cudaStream_t s     = device_.stream;
    constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
    // INFERNIX_TMP_NGRAM_CHAIN: row 0's copied tail after its drafter drafts is a copy too.
    bool copies              = chain_tail_ > 0;
    for (std::int32_t b = 0; b < batch; ++b) { copies = copies || (from_ngram_[b] && !drafts_[b].empty()); }
    // Row b's plane j stands at (b * K + j) * C in both layouts when K is the proposal's.
    if (!copies && K == dflash_k_) { return {dflash_io_.candidates, dflash_io_.q}; }
    std::int32_t* host_candidates = dflash_host(dflash_io_.verify_candidates);
    auto* host_q                  = reinterpret_cast<float*>(dflash_host(dflash_io_.verify_q));
    for (std::int32_t b = 0; b < batch; ++b) {
        const auto n = static_cast<std::int32_t>(drafts_[b].size());
        if (n == 0) { continue; }
        const std::size_t row = static_cast<std::size_t>(b) * K * C;
        if (from_ngram_[b]) {
            for (std::int32_t j = 0; j < n; ++j) {
                const std::int32_t token = drafts_[b][static_cast<std::size_t>(j)];
                for (std::int32_t c = 0; c < C; ++c) {
                    host_candidates[row + j * C + c] = (token + c) % token_domain_;
                    host_q[row + j * C + c]          = c == 0 ? 1.0F : 0.0F;
                }
            }
            upload_pinned(dflash_device(dflash_io_.verify_candidates + row), host_candidates + row, 4ULL * C * n, s);
            upload_pinned(dflash_device(dflash_io_.verify_q + row), host_q + row, 4ULL * C * n, s);
            continue;
        }
        const std::size_t from     = static_cast<std::size_t>(b) * dflash_k_ * C;
        const std::int32_t neural  = n - (b == 0 ? chain_tail_ : 0);
        CUDA_CHECK(cudaMemcpyAsync(dflash_device(dflash_io_.verify_candidates + row), dflash_device(dflash_io_.candidates + from),
                                   4ULL * C * neural, cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(dflash_device(dflash_io_.verify_q + row), dflash_device(dflash_io_.q + from),
                                   4ULL * C * neural, cudaMemcpyDeviceToDevice, s));
        if (neural < n) {
            // The chained tail one-hot at its copied tokens, after the drafter's planes.
            const std::size_t tail = row + static_cast<std::size_t>(neural) * C;
            for (std::int32_t j = neural; j < n; ++j) {
                const std::int32_t token = drafts_[b][static_cast<std::size_t>(j)];
                for (std::int32_t c = 0; c < C; ++c) {
                    host_candidates[row + j * C + c] = (token + c) % token_domain_;
                    host_q[row + j * C + c]          = c == 0 ? 1.0F : 0.0F;
                }
            }
            upload_pinned(dflash_device(dflash_io_.verify_candidates + tail), host_candidates + tail, 4ULL * C * (n - neural), s);
            upload_pinned(dflash_device(dflash_io_.verify_q + tail), host_q + tail, 4ULL * C * (n - neural), s);
        }
    }
    return {dflash_io_.verify_candidates, dflash_io_.verify_q};
}

void ProgramImpl::dflash2_catch_up(std::span<const SequenceHandle> rows, std::int32_t W, const std::int32_t* commit) {
    const auto batch = static_cast<std::int32_t>(rows.size());
    auto* io         = static_cast<std::byte*>(io_device_.p);
    // The round's features [F, W, batch] (column b * W + j), its positions [W, batch] and lanes, and
    // each row's committed column count; the launch covers every count 0..W.
    const execution::DFlash2ContextCall call{
        .features  = dflash_feature_columns(0, W * batch).view({dflash_feature_rows(), W, batch}),
        .positions = Tensor(io + io_layout_.positions, DType::I32, {W, batch}),
        .counts    = Tensor(spec_device(spec_layout_.commit), DType::I32, {batch}),
        .slots     = Tensor(io + io_layout_.slots, DType::I32, {batch}),
        .min_count = 0,
        .max_count = static_cast<std::uint32_t>(W)};
    replay(dflash_context_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)],
           [&] { drafter_->materialize(call); });
    for (std::int32_t b = 0; b < batch; ++b) {
        Lane& lane = lanes_[ContractAccess::lane(rows[static_cast<std::size_t>(b)])];
        if (commit[b] > 0) { lane.dflash_frontier = lane.state_tokens; }
    }
}

void ProgramImpl::dflash2_wait_copies(const Lane& lane) {
    // The drafter group's events are the last of a restore's and a copy-out's (layer order).
    for (const std::span<const cudaEvent_t> events : prefix_waits(lane)) {
        if (!events.empty()) { CUDA_CHECK(cudaStreamWaitEvent(device_.stream, events.back(), 0)); }
    }
}

void ProgramImpl::dflash2_context(std::int32_t column, std::int32_t n, std::int32_t first, std::int32_t slot) {
    if (n <= 0) { return; }
    const cudaStream_t s    = device_.stream;
    std::int32_t* positions = dflash_host(dflash_io_.context_positions);
    for (std::int32_t i = 0; i < n; ++i) { positions[i] = first + i; }
    *dflash_host(dflash_io_.context_count) = n;
    *dflash_host(dflash_io_.context_slot)  = slot;
    upload_pinned(dflash_device(dflash_io_.context_positions), positions,
                  4ULL * (dflash_io_.context_slot + 1 - dflash_io_.context_positions), s);
    const execution::DFlash2ContextCall call{
        .features  = dflash_feature_columns(column, n).view({dflash_feature_rows(), n, 1}),
        .positions = Tensor(dflash_device(dflash_io_.context_positions), DType::I32, {n, 1}),
        .counts    = Tensor(dflash_device(dflash_io_.context_count), DType::I32, {1}),
        .slots     = Tensor(dflash_device(dflash_io_.context_slot), DType::I32, {1}),
        .min_count = static_cast<std::uint32_t>(n),
        .max_count = static_cast<std::uint32_t>(n)};
    drafter_->materialize(call);
}

void ProgramImpl::dflash2_call_context(std::int32_t batch, std::int32_t width) {
    if (batch != 1) { throw std::logic_error("Qwen4Exp DFlash2: a prefill call or forced tokens of several rows"); }
    // The call's staged positions (sequential) and lane, as forward_call left its features.
    const auto* positions  = reinterpret_cast<const std::int32_t*>(host_io() + io_layout_.positions);
    const auto* slots      = reinterpret_cast<const std::int32_t*>(host_io() + io_layout_.slots);
    const auto [column, n] = dflash_call_region(width);
    // A restore into the lane's ring or a copy-out of it (a tap after its previous call) lands first.
    dflash2_wait_copies(lanes_.at(static_cast<std::size_t>(slots[0])));
    dflash2_context(column, n, positions[width - n], slots[0]);
}

void ProgramImpl::dflash2_walk_features(std::size_t first, std::size_t end, std::uint32_t layer, const Tensor& residual) {
    // The span's last window [from, T) lands in the call region; this group holds columns [lo, hi).
    const std::int32_t T    = walk_.offsets.back() + walk_.widths.back();
    const std::int32_t from = T - std::min(T, dim(plan_.dflash2_config->window));
    const std::int32_t lo = walk_.offsets[first], hi = walk_.offsets[end - 1] + walk_.widths[end - 1];
    const std::int32_t a = std::max(lo, from);
    if (a >= hi) { return; }
    forward_->capture_draft_features(layer, residual.slice(1, a - lo, hi - a),
                                     dflash_feature_columns(plan_.dflash_round_columns + (a - from), hi - a));
}

void ProgramImpl::dflash2_walk_context(Lane& lane, std::uint32_t index) {
    const std::int32_t T = walk_.offsets.back() + walk_.widths.back();
    const std::int32_t n = std::min(T, dim(plan_.dflash2_config->window));
    // Copies of the ring land first: a restore or copy-out still in flight, and the span's inner taps
    // (each copied the ring during the span).
    dflash2_wait_copies(lane);
    for (const LayerWalk::Tap& tap : walk_.taps) {
        if (tap.staged == 0) { continue; }
        if (const auto events = prefix_->staged_events(tap.staged); !events.empty()) {
            CUDA_CHECK(cudaStreamWaitEvent(device_.stream, events.back(), 0));
        }
    }
    dflash2_context(plan_.dflash_round_columns, n, walk_.begins.front() + T - n, static_cast<std::int32_t>(index));
    lane.dflash_frontier = lane.state_tokens;
}

} // namespace infernix::models::qwen4_exp::detail
