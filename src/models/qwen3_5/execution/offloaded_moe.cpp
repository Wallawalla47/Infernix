#include "models/qwen3_5/execution/offloaded_moe.h"

#include "core/layout.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/residual_add.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace infernix::models::qwen3_5::execution {
namespace {

namespace moe = ops::offloaded_moe;

// The layer's allocations in order (execution and sizing share them).
struct Buffers {
    Tensor logits, ids, weights, gate, outputs, y;
    DeviceSpan dispatch, experts;
};

template <class A>
Buffers allocate(A& a, const OffloadedMoeParameters& p, const moe::ExpertShape& shape, std::int32_t T) {
    const auto E = static_cast<std::int32_t>(p.experts), K = static_cast<std::int32_t>(p.top_k);
    const std::int32_t entries  = (K + 1) * T; // the routed entries and the shared job's
    const std::int32_t max_jobs = std::min(E, K * T) + 1;
    Buffers b;
    b.logits   = a.alloc(DType::FP32, {E + 1, T});
    b.ids      = a.alloc(DType::I32, {K, T});
    b.weights  = a.alloc(DType::FP32, {K, T});
    b.gate     = a.alloc(DType::FP32, {T});
    b.dispatch = a.alloc_bytes(ops::moe_dispatch_bytes(E + 1, entries));
    b.outputs  = a.alloc(DType::BF16, {shape.hidden, entries});
    b.experts  = a.alloc_bytes(
        ops::moe_experts_workspace_bytes(shape, max_jobs, entries, T, moe::ExpertActivation::kA16, moe::WideA16::kExact));
    b.y        = a.alloc(DType::BF16, {shape.hidden, T});
    return b;
}

// While the host computes the call's misses, the CPU wait warms the next layer's weights into L2 in the
// order its chain reads them (Flash-Next §19.3.20b: a one-column call warms up to 40 MiB of the
// chain, +1.2 % plain decode; a wider call 16 MiB of its input projection, more measured slower; the
// 122B's 64 and 80 MiB chain caps measured 0.8 and 2 % slower, design §0.2).
void warm_next(const BlockParameters& next, std::int32_t columns, ops::MoeExpertSource& source) {
    constexpr std::size_t kChainWarm = std::size_t{40} << 20, kInputWarm = std::size_t{16} << 20;
    const bool whole_chain = columns == 1;
    std::array<std::pair<const void*, std::size_t>, 12> chain{};
    std::size_t spans = 0;
    const auto add = [&](const void* data, std::size_t bytes) {
        if (spans < chain.size() && data != nullptr) { chain[spans++] = {data, bytes}; }
    };
    const auto add_weight = [&](const Weight& w, std::size_t cap) { add(w.payload, std::min(w.payload_bytes, cap)); };
    const auto add_projection = [&](const ops::ProjectionWeights& projection, std::size_t cap) {
        if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&projection)) {
            add_weight(pair->first, cap);
            add_weight(pair->second, cap);
        } else {
            add_weight(std::get<LinearParameters>(projection).weight, cap);
        }
    };
    const LinearParameters* output = nullptr;
    if (const auto* g = std::get_if<GdnParameters>(&next.mixer)) {
        add_projection(g->projection, whole_chain ? ~std::size_t{0} : kInputWarm);
        if (whole_chain) { add_projection(g->control, ~std::size_t{0}); }
        output = &g->output;
    } else {
        const auto& a = std::get<AttentionParameters>(next.mixer);
        add_projection(a.projection, whole_chain ? ~std::size_t{0} : kInputWarm);
        output = &a.output;
    }
    if (whole_chain) {
        add_weight(output->weight, ~std::size_t{0});
        if (const auto* m = std::get_if<OffloadedMoeParameters>(&next.ffn)) {
            add(m->router.data, m->router.bytes());
            add(m->shared_score.data, m->shared_score.bytes());
        }
    }
    std::size_t left = whole_chain ? kChainWarm : ~std::size_t{0};
    int i            = 0;
    for (std::size_t n = 0; n < spans && i < ops::MoeL2Warm::kSpans && left >= 16; ++n) {
        const auto [data, bytes] = chain[n];
        if (reinterpret_cast<std::uintptr_t>(data) % 16 != 0 || bytes < 16) { continue; }
        const std::size_t take  = std::min(bytes, left);
        source.l2_warm.ptr[i]   = data;
        source.l2_warm.bytes[i] = take;
        left -= take;
        ++i;
    }
}

} // namespace

std::size_t offloaded_moe_workspace_bytes(const OffloadedMoeParameters& p, std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) { throw std::invalid_argument("offloaded MoE: invalid column interval"); }
    // Every allocation grows with T: the interval's widest call bounds it.
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, p, p.bank->shape(), last);
    return layout.peak_bytes(1);
}

void offloaded_moe(const Tensor& hidden, const OffloadedMoeParameters& p, Tensor& residual,
                   const OffloadedMoeContext& context, WorkspaceArena& workspace, DeviceExecutionView execution) {
    if (context.experts == nullptr) { throw std::logic_error("offloaded MoE: the Program's experts are not bound"); }
    OffloadedExperts& experts  = *context.experts;
    const OffloadedCall& call  = context.call;
    const cudaStream_t s       = execution.stream;
    const std::int32_t T       = hidden.ne[1];
    const auto E = static_cast<std::int32_t>(p.experts), K = static_cast<std::int32_t>(p.top_k);
    ops::MoeExpertSource source = experts.sources.at(p.layer);
    // A wide pass's arena is already the span the expert cache lent the pass (prefill.cpp).
    auto scope = workspace.scope();
    Buffers b  = allocate(workspace, p, source.shape, T);

    // Router and shared-gate logits: one FP32 dot product each, read at BF16 by the route.
    const Tensor* rows[] = {&p.router, &p.shared_score};
    ops::projection_fp32(hidden, rows, b.logits, s);
    ops::MoeRouting routing{b.ids, b.weights, b.gate, call.mask};
    ops::MoeDispatch dispatch = ops::carve_moe_dispatch(b.dispatch.data, E + 1, (K + 1) * T);
    std::int32_t* route_log   = nullptr;
    if (experts.route_log != nullptr) {
        if (static_cast<std::size_t>(K) * T > experts.route_stride) {
            throw std::invalid_argument("offloaded MoE: the route log is too small for this call");
        }
        route_log = experts.route_log + static_cast<std::size_t>(p.layer) * experts.route_stride;
    }

    // --expert-tail-skip acts in generation rounds only; prefill and forced tokens stay exact.
    if (call.graphed) { source.tail_skip_rank = experts.tail_skip_rank; }

    // Graphed rounds take the fork route with landings; eager calls overlap staging with compute.
    if (experts.side_stream != nullptr) {
        if (call.graphed) {
            source.fork_stream    = experts.side_stream;
            source.fork_events[0] = experts.side_events[0];
            source.fork_events[1] = experts.side_events[1];
            if (experts.landing != nullptr) {
                source.landing       = experts.landing + static_cast<std::size_t>(p.layer) * experts.landing_slots;
                source.landed        = experts.landed + static_cast<std::size_t>(p.layer) * experts.landing_slots;
                source.landing_slots = experts.landing_slots;
            }
        } else {
            source.overlap_stream = experts.side_stream;
            for (std::size_t i = 0; i < experts.side_events.size(); ++i) { source.overlap_events[i] = experts.side_events[i]; }
        }
    }
    if (source.cpu.max_jobs > 0 && context.next != nullptr) { warm_next(*context.next, T, source); }

    // A streamed chunk routes and dispatches first: its host-side split reads the dispatch.
    runtime::expert_offload::ExpertStream* const streaming =
        call.streamed && experts.stream != nullptr && experts.stream->active() ? experts.stream : nullptr;
    if (streaming != nullptr) {
        ops::moe_route_dispatch(b.logits, K, routing, dispatch, route_log, s, ops::RouterLogits::kBf16, /*shared_job=*/true);
        if (streaming->gated()) {
            // The layer's routing decides which experts the link copies and which the CPU serves: every
            // narrow miss is offered, capped at the policy's count (Flash-Next §19.3.12).
            if (experts.split_offsets.size() < (static_cast<std::size_t>(E) + 2) * sizeof(std::int32_t)) {
                experts.split_offsets = PinnedHostBuffer((static_cast<std::size_t>(E) + 2) * sizeof(std::int32_t));
            }
            CUDA_CHECK(cudaMemcpyAsync(experts.split_offsets.data(), dispatch.offsets,
                                       (static_cast<std::size_t>(E) + 1) * sizeof(std::int32_t), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaEventRecord(experts.split_ready, s));
            CUDA_CHECK(cudaEventSynchronize(experts.split_ready));
            const auto* offsets = static_cast<const std::int32_t*>(experts.split_offsets.data());
            experts.split_columns.resize(static_cast<std::size_t>(E));
            experts.split_cpu.assign(static_cast<std::size_t>(E), 0);
            for (std::int32_t e = 0; e < E; ++e) { experts.split_columns[static_cast<std::size_t>(e)] = offsets[e + 1] - offsets[e]; }
            const std::uint32_t cpu = call.split && experts.split != nullptr && source.cpu.max_jobs > 0
                                          ? experts.split->choose(p.layer, experts.split_columns, experts.split_cpu)
                                          : 0U;
            experts.split_streamed_experts += streaming->plan_layer(p.layer, experts.split_columns, experts.split_cpu);
            experts.split_cpu_experts += cpu;
            if (cpu > 0) {
                source.cpu.max_columns               = moe::kMaxCpuCallColumns;
                source.cpu.pcie_divisor              = 0;
                source.cpu.one_column_divisor        = 0;
                source.cpu.one_column_divisor_device = nullptr;
                source.cpu.wide_from                 = 1;
                source.cpu.wide_jobs                 = static_cast<std::int32_t>(cpu);
            } else {
                source.cpu = ops::MoeCpuChannel{};
            }
        }
        streaming->before_experts(p.layer, s);
        source.prefetched      = streaming->slots(p.layer);
        source.prefetch_base   = streaming->base();
        source.prefetch_stride = streaming->pitch();
        // Every expert absent when the chunk began has a ring slot: the call stages nothing and runs as
        // one pass over every job (an expert evicted since is read in place from its host record).
        if (streaming->complete(p.layer) && source.host_table == nullptr) {
            source.staging_slots  = 0;
            source.overlap_stream = nullptr;
        }
    }

    const std::int32_t max_jobs = std::min(E, K * T) + 1;
    source.wide_a16             = moe::WideA16::kExact;
    const bool planned =
        streaming == nullptr && ops::moe_route_dispatch_plan(b.logits, K, routing, dispatch, route_log, hidden, source,
                                                             max_jobs, b.experts.data, b.outputs, s,
                                                             ops::RouterLogits::kBf16, /*shared_job=*/true);
    ops::moe_experts(hidden, dispatch, source, K, max_jobs, b.experts.data, b.outputs, s, /*wait_for_cpu=*/true, planned);
    if (streaming != nullptr && call.release_stream) { streaming->after_experts(p.layer, s); }
    // y = bf16(sum_i w_i y_i + s y_shared): the routed entries, then the shared job's T columns.
    const Tensor routed = b.outputs.slice(1, 0, K * T);
    const Tensor shared = b.outputs.slice(1, K * T, T);
    ops::moe_combine(routed, routing, shared, b.y, s);
    ops::residual_add(b.y, residual, s);
}

} // namespace infernix::models::qwen3_5::execution
