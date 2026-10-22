#include "models/qwen3_5/program/offload.h"

#include "core/device.h"
#include "infernix/types.h"
#include "runtime/contract/execution.h"

#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace infernix::models::qwen3_5::detail {

namespace expert_offload = runtime::expert_offload;

runtime::expert_offload::OffloadSetup offload_setup(const execution::Parameters& parameters,
                                                    const OffloadGeometry& geometry) {
    const Model& model = parameters.model;
    if (!model.offloaded_experts()) { throw std::logic_error("Qwen3.5 offload: the artifact offloads no experts"); }
    const auto& moe = std::get<MoeConfig>(model.config().text.ffn);
    expert_offload::OffloadSetup setup;
    setup.layout = {static_cast<std::uint32_t>(parameters.text.layers.size()), moe.num_experts,
                    moe.num_experts_per_tok};
    for (const execution::BlockParameters& block : parameters.text.layers) {
        // Every text MoE layer offloads its bank, or none does (load.cpp).
        const auto* layer = std::get_if<execution::OffloadedMoeParameters>(&block.ffn);
        if (layer == nullptr || layer->device_scales == nullptr ||
            layer->layer != static_cast<std::uint32_t>(setup.banks.size())) {
            throw std::logic_error("Qwen3.5 offload: every text layer must be an offloaded MoE layer, in order");
        }
        setup.banks.push_back(layer->bank);
        setup.device_scales.push_back(layer->device_scales);
    }
    // With the SSD tier (the RAM ledger keeps only part of the experts in RAM) the banks are views into
    // the artifact and the store reads their records; null in full mode.
    setup.store                     = model.expert_store();
    setup.artifact_id               = model.info().artifact_id;
    setup.lanes                     = geometry.lanes;
    setup.max_width                 = geometry.max_width;
    setup.columns                   = geometry.columns;
    setup.chunk                     = geometry.chunk;
    setup.walk_max_tokens           = geometry.walk_max_tokens;
    setup.speculative               = geometry.speculative;
    setup.graph_bound               = geometry.graph_bound;
    // The Program captures every executable right after start() (program_impl.cpp prepare_graphs).
    setup.eager_graphs              = true;
    setup.one_column_divisor_device = geometry.one_column_divisor_device;
    setup.device                    = geometry.device;
    return setup;
}

std::uint64_t ProgramOffload::fixed_bytes(const execution::Parameters& parameters, const OffloadGeometry& geometry,
                                          const expert_offload::OffloadOptions& options) {
    return expert_offload::ExpertOffload::fixed_bytes(offload_setup(parameters, geometry), options);
}

ProgramOffload::ProgramOffload(const execution::Parameters& parameters, const OffloadGeometry& geometry,
                               expert_offload::OffloadOptions options, Diagnostic diagnostic)
    : parameters_(parameters), wide_pass_bytes_(geometry.wide_pass_bytes),
      engine_(offload_setup(parameters, geometry), std::move(options), std::move(diagnostic)) {}

ProgramOffload::~ProgramOffload() {
    if (experts_.split_ready != nullptr) { cudaEventDestroy(experts_.split_ready); }
}

void ProgramOffload::start(cudaStream_t compute,
                           const std::function<void(std::uint64_t, std::uint64_t)>& lendable_frames) {
    engine_.start(compute, lendable_frames);
    build_experts();
}

void ProgramOffload::build_experts() {
    const auto& moe          = std::get<MoeConfig>(parameters_.model.config().text.ffn);
    const auto shared_expert = static_cast<std::int32_t>(moe.num_experts);
    experts_.sources.clear();
    experts_.sources.reserve(parameters_.text.layers.size());
    for (const execution::BlockParameters& block : parameters_.text.layers) {
        const auto& layer = std::get<execution::OffloadedMoeParameters>(block.ffn);
        const auto& shared = layer.shared_bank->planes;
        // The shared expert is one more record in the routed bank's layout (design §8.6): always
        // resident, read in place.
        if (shared.experts != 1 || shared.record_stride != layer.bank->planes.record_stride ||
            shared.hidden != layer.bank->planes.hidden || shared.intermediate != layer.bank->planes.intermediate ||
            shared.device_records == nullptr) {
            throw std::logic_error("Qwen3.5 offload: the shared expert must be one resident record of the routed geometry");
        }
        ops::MoeExpertSource source = engine_.source(layer.layer);
        source.shared_expert        = shared_expert;
        source.shared_record        = reinterpret_cast<const std::uint8_t*>(shared.device_records);
        experts_.sources.push_back(source);
    }
    experts_.tail_skip_rank = engine_.tail_skip_rank();
    experts_.route_log     = engine_.route_log();
    experts_.route_stride  = engine_.route_stride();
    experts_.landing       = engine_.landing_table();
    experts_.landed        = engine_.landed_log();
    experts_.landing_slots = expert_offload::ExpertOffload::landing_slots();
    experts_.side_stream   = engine_.overlap_stream();
    const auto events      = engine_.overlap_events();
    for (std::size_t i = 0; i < events.size(); ++i) { experts_.side_events[i] = events[i]; }
    experts_.stream = engine_.stream();
    experts_.split  = engine_.split_policy();
    // A gated split reads the layer's dispatch offsets ([E + 2] with the shared job) on the host.
    experts_.split_offsets = PinnedHostBuffer((static_cast<std::size_t>(moe.num_experts) + 2) * sizeof(std::int32_t));
    if (experts_.split_ready == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&experts_.split_ready, cudaEventDisableTiming));
    }
    experts_.split_columns.assign(moe.num_experts, 0);
    experts_.split_cpu.assign(moe.num_experts, 0);
    experts_.engine = &engine_;
}

std::uint64_t ProgramOffload::wide_bytes(std::int32_t columns) const {
    // The planner's full-width pass (0 on the lossless route, whose layers run their own path): one
    // size for every wide pass, so a prompt's passes reuse one lease.
    return columns > execution::kOffloadedMoeStaticColumns ? wide_pass_bytes_ : 0;
}

DeviceSpan ProgramOffload::wide_workspace(std::uint64_t bytes, cudaStream_t compute) {
    const auto& stream = engine_.stream_leases().wide;
    if (stream.valid() && stream.memory.bytes >= bytes) { return {stream.memory.data, bytes}; }
    if (wide_lease_.valid() && wide_lease_.memory.bytes >= bytes) { return {wide_lease_.memory.data, bytes}; }
    if (wide_fallback_.p != nullptr && wide_fallback_.bytes >= bytes) { return {wide_fallback_.p, bytes}; }
    // A wider pass than the prompt's earlier ones: its larger lease replaces the held one.
    if (wide_lease_.valid()) { engine_.give_back(wide_lease_, compute); }
    const cudaStream_t writers[] = {compute};
    wide_lease_ = engine_.lend(bytes, compute, writers);
    if (wide_lease_.valid()) { return {wide_lease_.memory.data, bytes}; }
    // The cache cannot lend it now (it shrank under VRAM pressure, or a vision window holds frames):
    // the prompt allocates it; a failed allocation fails this prompt's requests, not the Engine.
    wide_fallback_ = DeviceBuffer();
    try {
        wide_fallback_ = DeviceBuffer(static_cast<std::size_t>(bytes));
    } catch (const std::exception& error) {
        throw runtime::RecoverableExecutionError("the prefill workspace (" + std::to_string(bytes >> 20) +
                                                 " MiB) could not be lent from the expert cache or allocated: " +
                                                 error.what());
    }
    return {wide_fallback_.p, bytes};
}

execution::OffloadedCall ProgramOffload::begin_pass(cudaStream_t compute, std::int32_t columns) {
    // A decode round's deferred update reads the host route log this pass's download overwrites.
    engine_.settle(compute);
    engine_.before_round(compute, /*landing=*/false);
    execution::OffloadedCall call;
    call.release_stream    = true;
    const std::uint64_t wide = wide_bytes(columns);
    // The ring is sized with the wide lease's evictions; without room for both the stream runs alone
    // and the wide workspace comes from wide_workspace.
    call.streamed = engine_.lend_stream(columns, wide, /*walk_bytes=*/0, compute) ||
                    (wide > 0 && engine_.lend_stream(columns, 0, /*walk_bytes=*/0, compute));
    if (call.streamed) {
        // A pass the CPU can share gates the stream on each layer's routing (Flash-Next §19.3.12).
        call.split = engine_.split_policy() != nullptr && columns <= engine_.split_columns();
        engine_.stream()->begin(engine_.ring_span(), engine_.residency().host_table(), compute, call.split);
    }
    if (wide > 0) {
        try {
            call.workspace = wide_workspace(wide, compute);
        } catch (...) {
            abandon_pass(call);
            throw;
        }
    }
    return call;
}

void ProgramOffload::release_wide(cudaStream_t compute) noexcept {
    try {
        if (wide_lease_.valid()) { engine_.give_back(wide_lease_, compute); }
    } catch (...) {}
    wide_lease_    = {};
    wide_fallback_ = DeviceBuffer();
}

void ProgramOffload::return_stream(cudaStream_t compute) {
    engine_.return_stream(compute);
    release_wide(compute);
}

void ProgramOffload::end_pass(cudaStream_t compute, const execution::OffloadedCall& call, PassKind kind,
                              std::uint32_t lane, std::int32_t columns, std::uint32_t position, bool prompt_end,
                              const std::string& model) {
    using Offload = runtime::expert_offload::ExpertOffload;
    if (call.streamed) { engine_.stream()->end(); }
    engine_.check_error(model);
    const bool prompt            = kind == PassKind::Prompt;
    const std::size_t promotions = !prompt ? Offload::kDecodePromotionsPerLayer
                                   : columns <= Offload::kServedCallColumns
                                       ? engine_.decode_budget(static_cast<std::uint32_t>(columns))
                                       : 0;
    const auto lane_id = static_cast<std::int32_t>(lane);
    if (prompt) { engine_.note_prompt_routes(lane, columns); }
    engine_.after_call(compute, columns, static_cast<std::uint32_t>(columns), promotions,
                       {.kind     = prompt ? runtime::expert_offload::RouteTraceKind::PrefillChunk
                                         : runtime::expert_offload::RouteTraceKind::ForcedTokens,
                        .rows     = 1,
                        .width    = columns,
                        .position = position,
                        .lanes    = std::span<const std::int32_t>(&lane_id, 1)});
    if (prompt && prompt_end) { engine_.seed_prompt(lane); }
}

void ProgramOffload::abandon_pass(const execution::OffloadedCall& call) noexcept {
    if (call.streamed && engine_.stream() != nullptr) { engine_.stream()->end(); }
}

void ProgramOffload::begin_round(cudaStream_t compute) {
    engine_.settle(compute);
    engine_.before_round(compute, /*landing=*/true);
}

void ProgramOffload::end_round(std::span<const std::uint32_t> lanes, std::int32_t width, std::uint32_t tokens,
                               std::span<const std::uint8_t> live, const std::string& model) {
    engine_.check_error(model);
    std::array<std::int32_t, kMaximumConcurrency> rows{};
    if (lanes.size() > rows.size()) { throw std::invalid_argument("Qwen3.5 offload: a round of too many rows"); }
    for (std::size_t r = 0; r < lanes.size(); ++r) { rows[r] = static_cast<std::int32_t>(lanes[r]); }
    const auto count = static_cast<std::int32_t>(lanes.size());
    engine_.defer(count * width, tokens,
                  {.kind     = width > 1 ? runtime::expert_offload::RouteTraceKind::Verify
                                         : runtime::expert_offload::RouteTraceKind::Decode,
                   .rows     = count,
                   .width    = width,
                   .position = runtime::expert_offload::RouteTrace::kUnknownPosition,
                   .lanes    = std::span<const std::int32_t>(rows.data(), lanes.size())},
                  live);
}

DeviceSpan ProgramOffload::VisionFrames::lend(std::size_t bytes, cudaStream_t compute, cudaStream_t writer) {
    if (lease_.valid()) { throw std::logic_error("Qwen3.5 offload: a vision window is already lent"); }
    owner_.return_stream(compute);
    const cudaStream_t writers[] = {writer};
    lease_ = engine_.lend(bytes, compute, writers);
    if (!lease_.valid()) {
        throw runtime::RecoverableExecutionError("the expert cache cannot lend " + std::to_string(bytes >> 20) +
                                                 " MiB for the vision window now; retry with fewer or smaller images");
    }
    return lease_.memory;
}

void ProgramOffload::VisionFrames::give_back(cudaStream_t compute) noexcept {
    try {
        if (lease_.valid()) { engine_.give_back(lease_, compute); }
    } catch (...) {}
    lease_ = {};
}

} // namespace infernix::models::qwen3_5::detail
