#pragma once

// The Program's offloaded routed experts (docs/maintainer/qwen3_5-122b-offload-design.md §9.2, §10,
// §11): one runtime::expert_offload::ExpertOffload over the artifact's offloaded MoE layers, and the
// forward's view of it (execution::OffloadedExperts: every layer's expert source with the shared
// expert as the call's last job, the route log, the landings, the side stream and the prefill
// stream). The Program plans fixed_bytes() among its fixed allocations, constructs this after them
// (the engine's VRAM source checks them) and start()s it once they are allocated; nothing here is
// shared with another Program.

#include "models/qwen3_5/execution/offloaded_moe.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/vision_overlay.h"
#include "runtime/expert_offload/expert_offload.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>

namespace infernix::models::qwen3_5::detail {

// The Program's round geometry as the engine sizes its tables, staging and frames.
struct OffloadGeometry {
    std::int32_t lanes     = 1;
    std::int32_t max_width = 1; // a verification round's columns per row (1: plain decode only)
    std::int32_t columns   = 1; // the most columns one round routes
    std::int32_t chunk     = 1024;
    std::uint32_t walk_max_tokens = 65536;
    bool speculative             = false;
    std::uint32_t graph_bound    = 0; // graph executables the frame sizing leaves room for
    // A full-width prefill pass's workspace (WorkspacePlan::text_prefill_wide, mtp_prefill_wide): what
    // the cache lends a pass wider than kOffloadedMoeStaticColumns; 0 without lent passes.
    std::uint64_t wide_pass_bytes = 0;
    // Plain decode's one-column divisor word (ExpertOffload::plain_divisor), when the Program keeps one.
    const std::int32_t* one_column_divisor_device = nullptr;
    int device = 0;
};

// What the Engine instance resolved for a Program with offloaded experts (runtime/engine/
// model_instance.cpp): the engine's options (EngineOptions and the RAM ledger's tier) and where its
// diagnostics go.
struct ProgramOffloadOptions {
    runtime::expert_offload::OffloadOptions engine;
    runtime::expert_offload::ExpertOffload::Diagnostic diagnostic;
};

// The engine's setup for these Parameters (an artifact with offloaded experts): every offloaded
// layer's routed bank and device scales, in text-layer order.
[[nodiscard]] runtime::expert_offload::OffloadSetup offload_setup(const execution::Parameters& parameters,
                                                                  const OffloadGeometry& geometry);

class ProgramOffload {
public:
    using Diagnostic = runtime::expert_offload::ExpertOffload::Diagnostic;

    // The device bytes the constructor allocates (the engine's staging and tables), for the plan.
    [[nodiscard]] static std::uint64_t fixed_bytes(const execution::Parameters& parameters,
                                                   const OffloadGeometry& geometry,
                                                   const runtime::expert_offload::OffloadOptions& options);

    ProgramOffload(const execution::Parameters& parameters, const OffloadGeometry& geometry,
                   runtime::expert_offload::OffloadOptions options, Diagnostic diagnostic);
    ~ProgramOffload();
    ProgramOffload(const ProgramOffload&)            = delete;
    ProgramOffload& operator=(const ProgramOffload&) = delete;

    // After the Program's fixed allocations: the engine's start (link, CPU service, frames, warm
    // start), then the forward's view. `lendable_frames` as ExpertOffload::start.
    void start(cudaStream_t compute,
               const std::function<void(std::uint64_t frames, std::uint64_t pitch)>& lendable_frames = {});

    [[nodiscard]] runtime::expert_offload::ExpertOffload& engine() noexcept { return engine_; }
    [[nodiscard]] const runtime::expert_offload::ExpertOffload& engine() const noexcept { return engine_; }
    // Valid after start(); every pointer in it is stable for the Program's life.
    [[nodiscard]] execution::OffloadedExperts& experts() noexcept { return experts_; }

    // ---- an eager pass of a lane's tokens through the text layers (Flash-Next design §19.3.8,
    // §19.3.12; Qwen3.5-122B design §10): a prompt's (prefill, replay) or forced tokens' ----
    enum class PassKind { Prompt, Forced };
    // Before a pass of at most `columns` tokens: wakes the CPU service and opens the round, then picks
    // the pass's route. A pass of at least kStreamMinColumns streams when the ring can be lent; a
    // streamed pass narrow enough for the measured link and CPU gates its layers on their routing and
    // gives the CPU their thinnest misses. Each layer's ring half is released once its experts are
    // enqueued (only a layer walk, which Qwen3.5 does not run yet, keeps them). A pass wider than
    // execution::kOffloadedMoeStaticColumns gets the MoE layers' buffers (OffloadedCall::workspace)
    // from the expert frames: lent with the stream's ring, else alone, else (the cache cannot lend
    // them now) allocated for the prompt (RecoverableExecutionError when that fails too). The forward
    // enqueues the route download.
    [[nodiscard]] execution::OffloadedCall begin_pass(cudaStream_t compute, std::int32_t columns);
    // After the pass's synchronization: fails the round on an expert error (RecoverableExecutionError
    // naming `model`; the caller drops the prefix entries it published) and applies the cache update.
    // A prompt pass credits `lane`'s prompt routes and promotes at the decode rate only at decode
    // width (a wider one never; its routes still count, Flash-Next §19.3.17), and at the prompt's end
    // (`prompt_end`) the lane's routes seed the cache; a forced pass promotes kDecodePromotionsPerLayer.
    // `position`: the pass's first position.
    void end_pass(cudaStream_t compute, const execution::OffloadedCall& call, PassKind kind, std::uint32_t lane,
                  std::int32_t columns, std::uint32_t position, bool prompt_end, const std::string& model);
    // A failed pass: the stream ends without the cache update.
    void abandon_pass(const execution::OffloadedCall& call) noexcept;
    // Once a prompt's passes end: the stream's leases (ExpertOffload::return_stream) and the wide
    // workspace lent or allocated outside them.
    void return_stream(cudaStream_t compute);
    void release_wide(cudaStream_t compute) noexcept;

    // ---- a decode or verification round (graphed or eager; Flash-Next design §19.3.5) ----
    // Before its forward: settles an update still pending, wakes the CPU service and opens the round
    // with landings.
    void begin_round(cudaStream_t compute);
    // After the forward is enqueued: the round's route download (`columns` routed columns).
    void enqueue_routes(cudaStream_t compute, std::int32_t columns) { engine_.enqueue_route_download(compute, columns); }
    // After the round's synchronization: the expert error check (RecoverableExecutionError naming
    // `model`), then the cache update, deferred past the commit's enqueue (settle): one row of `width`
    // columns per lane of `lanes`, the longest row advancing `tokens`; with `live` (a verification's
    // accepted columns, row-major) only its flagged columns count as uses.
    void end_round(std::span<const std::uint32_t> lanes, std::int32_t width, std::uint32_t tokens,
                   std::span<const std::uint8_t> live, const std::string& model);
    // Once the round's commit is enqueued (or before the next round): the deferred cache update.
    void settle(cudaStream_t compute) { engine_.settle(compute); }
    // A failed or discarded round: its update is dropped.
    void drop_round() noexcept { engine_.drop_deferred(); }

    // An overlay vision window's memory (design D8): expert frames lent for the window, after the
    // expert stream's leases are returned (the next streamed pass lends its ring again).
    [[nodiscard]] execution::VisionWindowSource& vision_window() noexcept { return vision_window_; }

private:
    class VisionFrames final : public execution::VisionWindowSource {
    public:
        explicit VisionFrames(ProgramOffload& owner) noexcept : owner_(owner), engine_(owner.engine_) {}
        [[nodiscard]] DeviceSpan lend(std::size_t bytes, cudaStream_t compute, cudaStream_t writer) override;
        void give_back(cudaStream_t compute) noexcept override;

    private:
        ProgramOffload& owner_;
        runtime::expert_offload::ExpertOffload& engine_;
        runtime::expert_offload::ExpertResidency::FrameLease lease_;
    };

    void build_experts();
    // The workspace a pass of `columns` borrows (0 at or below the static width).
    [[nodiscard]] std::uint64_t wide_bytes(std::int32_t columns) const;
    [[nodiscard]] DeviceSpan wide_workspace(std::uint64_t bytes, cudaStream_t compute);

    const execution::Parameters& parameters_;
    std::uint64_t wide_pass_bytes_ = 0; // OffloadGeometry::wide_pass_bytes
    runtime::expert_offload::ExpertOffload engine_;
    execution::OffloadedExperts experts_;
    VisionFrames vision_window_{*this};
    // A wide pass's buffers outside the stream's lease: a frame lease, or the allocation used when the
    // cache could not lend them.
    runtime::expert_offload::ExpertResidency::FrameLease wide_lease_;
    DeviceBuffer wide_fallback_;
};

} // namespace infernix::models::qwen3_5::detail
