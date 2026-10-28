#include "models/qwen3_5/execution/offloaded_moe.h"

#include "core/layout.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/residual_add.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace infernix::models::qwen3_5::execution {
namespace {

namespace moe = ops::offloaded_moe;

// The layer's allocations in order (execution and sizing share them). A shared expert bank is the
// call's last job (its outputs the last T columns of `outputs`); dense shared projections write
// `product` and `shared`.
struct Buffers {
    Tensor logits, ids, weights, gate, outputs, product, shared, y;
    DeviceSpan dispatch, shared_scratch, experts;
    // A gated call's copy-ahead guess of the next layer's routing, sized for at most the CPU channel's width
    // (the widest gated call) so the layout grows with T; empty below a streamed call's width.
    Tensor guess_ids, guess_weights, guess_gate;
    DeviceSpan guess_dispatch;
};

template <class A>
Buffers allocate(A& a, const OffloadedMoeParameters& p, const moe::ExpertShape& shape, std::int32_t T,
                 moe::WideA16 wide_a16) {
    const auto E = static_cast<std::int32_t>(p.experts), K = static_cast<std::int32_t>(p.top_k);
    const std::int32_t shared_job = p.shared ? 0 : 1;
    const std::int32_t entries    = (K + shared_job) * T; // the routed entries and the shared job's
    const std::int32_t max_jobs   = std::min(E, K * T) + shared_job;
    Buffers b;
    b.logits   = a.alloc(DType::FP32, {E + 1, T});
    b.ids      = a.alloc(DType::I32, {K, T});
    b.weights  = a.alloc(DType::FP32, {K, T});
    b.gate     = a.alloc(DType::FP32, {T});
    b.dispatch = a.alloc_bytes(ops::moe_dispatch_bytes(E + shared_job, entries));
    b.outputs  = a.alloc(DType::BF16, {shape.hidden, entries});
    if (p.shared) {
        // Its own buffers, before the experts' workspace: the shared stream reads and writes nothing the
        // routed experts use before the join.
        const auto& gu = p.shared->gate_up;
        const auto& dn = p.shared->down;
        b.product        = a.alloc(DType::BF16, {gu.weight.n / 2, T});
        b.shared         = a.alloc(DType::BF16, {shape.hidden, T});
        // (An arena is never empty: the FP8-row routes take no scratch.)
        b.shared_scratch = a.alloc_bytes(std::max<std::size_t>(
            {256, ops::linear_swiglu_workspace_capacity_bytes(gu.weight.qtype, gu.weight.n, gu.weight.k, gu.policy, T, T),
             ops::linear_workspace_capacity_bytes(dn.weight.qtype, dn.weight.n, dn.weight.k, dn.policy, T, T)}));
    }
    b.experts  = a.alloc_bytes(
        ops::moe_experts_workspace_bytes(shape, max_jobs, entries, T, moe::ExpertActivation::kA16, wide_a16));
    b.y        = a.alloc(DType::BF16, {shape.hidden, T});
    if (T >= kRouterMmaMinColumns) {
        const std::int32_t g = std::min(T, moe::kMaxCpuCallColumns);
        b.guess_ids      = a.alloc(DType::I32, {K, g});
        b.guess_weights  = a.alloc(DType::FP32, {K, g});
        b.guess_gate     = a.alloc(DType::FP32, {g});
        b.guess_dispatch = a.alloc_bytes(ops::moe_dispatch_bytes(E, K * g));
    }
    return b;
}

// The dense shared expert: silu(gate) * up in one call, then its down projection.
void dense_shared_expert(const Tensor& hidden, const DenseParameters& shared, Buffers& b, cudaStream_t s) {
    WorkspaceArena scratch(b.shared_scratch);
    ops::linear_swiglu(hidden, shared.gate_up.weight, b.product, shared.gate_up.policy, scratch, s);
    ops::linear(b.product, shared.down.weight, b.shared, shared.down.policy, scratch, s);
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

std::size_t offloaded_moe_workspace_bytes(const OffloadedMoeParameters& p, std::int32_t first, std::int32_t last,
                                          moe::WideA16 wide_a16) {
    if (first <= 0 || last < first) { throw std::invalid_argument("offloaded MoE: invalid column interval"); }
    // Every allocation grows with T (the A4 residual planes too): the interval's widest call bounds it.
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, p, p.bank->shape(), last, wide_a16);
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
    Buffers b  = allocate(workspace, p, source.shape, T, experts.wide_a16);

    // Router and shared-gate logits: one FP32 dot product each, read at BF16 by the route.
    const Tensor* rows[] = {&p.router, &p.shared_score};
    if (experts.router_mma && T >= kRouterMmaMinColumns) {
        ops::projection_fp32_router_mma(hidden, rows, b.logits, s);
    } else {
        ops::projection_fp32(hidden, rows, b.logits, s);
    }
    ops::MoeRouting routing{b.ids, b.weights, b.gate, call.mask};
    // A shared expert bank is the call's last job; dense shared projections run beside the call.
    const bool shared_job     = !p.shared;
    const std::int32_t sj     = shared_job ? 1 : 0;
    ops::MoeDispatch dispatch = ops::carve_moe_dispatch(b.dispatch.data, E + sj, (K + sj) * T);
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
            source.fork_stream     = experts.side_stream;
            source.fork_events[0]  = experts.side_events[0];
            source.fork_events[1]  = experts.side_events[1];
            source.split_stream    = experts.split_stream;
            source.split_events[0] = experts.split_events[0];
            source.split_events[1] = experts.split_events[1];
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
    // The copy-ahead (ExpertStream::plan_ahead, candidates BX-09): a gated call also routes this layer's input
    // through the next layer's router and copies the experts that guess makes heavy while this layer ends, so the
    // link does not wait idle for the next layer's routing. Placement only: the same record bytes are read.
    const OffloadedMoeParameters* const next =
        context.next != nullptr ? std::get_if<OffloadedMoeParameters>(&context.next->ffn) : nullptr;
    const bool ahead = streaming != nullptr && streaming->gated() && next != nullptr && call.release_stream &&
                       T <= moe::kMaxCpuCallColumns && b.guess_dispatch.data != nullptr;
    if (streaming != nullptr) {
        ops::moe_route_dispatch(b.logits, K, routing, dispatch, route_log, s, ops::RouterLogits::kBf16, shared_job);
        if (streaming->gated()) {
            // The layer's routing decides which experts the link copies and which the CPU serves: every
            // narrow miss is offered, capped at the policy's count (Flash-Next §19.3.12).
            if (experts.split_offsets.size() < 2 * (static_cast<std::size_t>(E) + 2) * sizeof(std::int32_t)) {
                experts.split_offsets = PinnedHostBuffer(2 * (static_cast<std::size_t>(E) + 2) * sizeof(std::int32_t));
            }
            auto* offsets = static_cast<std::int32_t*>(experts.split_offsets.data());
            CUDA_CHECK(cudaMemcpyAsync(offsets, dispatch.offsets, (static_cast<std::size_t>(E) + 1) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, s));
            if (ahead) {
                // The route has read `logits`, so the guess reuses them. It decides placement only, so the
                // router's tensor-core form (a gated call has at least 256 columns).
                const Tensor* next_rows[] = {&next->router, &next->shared_score};
                ops::projection_fp32_router_mma(hidden, next_rows, b.logits, s);
                ops::MoeRouting guess{b.guess_ids, b.guess_weights, b.guess_gate};
                ops::MoeDispatch guessed = ops::carve_moe_dispatch(b.guess_dispatch.data, E, K * T);
                ops::moe_route_dispatch(b.logits, K, guess, guessed, nullptr, s, ops::RouterLogits::kBf16);
                CUDA_CHECK(cudaMemcpyAsync(offsets + E + 2, guessed.offsets, (static_cast<std::size_t>(E) + 1) * sizeof(std::int32_t),
                                           cudaMemcpyDeviceToHost, s));
            }
            CUDA_CHECK(cudaEventRecord(experts.split_ready, s));
            CUDA_CHECK(cudaEventSynchronize(experts.split_ready));
            experts.split_columns.resize(static_cast<std::size_t>(E));
            experts.split_priced.resize(static_cast<std::size_t>(E));
            experts.split_cpu.assign(static_cast<std::size_t>(E), 0);
            for (std::int32_t e = 0; e < E; ++e) { experts.split_columns[static_cast<std::size_t>(e)] = offsets[e + 1] - offsets[e]; }
            // Experts copied ahead for this layer are on their way: the CPU split neither takes nor prices them.
            std::copy(experts.split_columns.begin(), experts.split_columns.end(), experts.split_priced.begin());
            streaming->mask_ahead(p.layer, experts.split_priced);
            const std::uint32_t cpu = call.split && experts.split != nullptr && source.cpu.max_jobs > 0
                                          ? experts.split->choose(p.layer, experts.split_priced, experts.split_cpu)
                                          : 0U;
            experts.split_streamed_experts += streaming->plan_layer(p.layer, experts.split_columns, experts.split_cpu);
            experts.split_cpu_experts += cpu;
            if (cpu > 0) {
                source.cpu.max_columns               = moe::kMaxCpuCallColumns;
                source.cpu.pcie_divisor              = 0;
                source.cpu.pcie_share                = 0;
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

    // The next layer's guessed experts, behind this layer's copies on the link: enqueued once this layer's
    // experts are (the GPU then works while the host plans), or before them when that call waits for the CPU.
    const auto copy_ahead = [&] {
        if (!ahead) { return; }
        const auto* guessed = static_cast<const std::int32_t*>(experts.split_offsets.data()) + E + 2;
        experts.guess_columns.resize(static_cast<std::size_t>(E));
        for (std::int32_t e = 0; e < E; ++e) { experts.guess_columns[static_cast<std::size_t>(e)] = guessed[e + 1] - guessed[e]; }
        (void)streaming->plan_ahead(next->layer, experts.guess_columns, runtime::expert_offload::ExpertStream::ahead_rows(T), 0);
    };
    const std::int32_t max_jobs = std::min(E, K * T) + sj;
    source.wide_a16             = experts.wide_a16;
    source.narrow_a16           = experts.narrow_a16;
    const bool planned =
        streaming == nullptr && ops::moe_route_dispatch_plan(b.logits, K, routing, dispatch, route_log, hidden, source,
                                                             max_jobs, b.experts.data, b.outputs, s,
                                                             ops::RouterLogits::kBf16, shared_job);
    if (shared_job) {
        // This call waits for the CPU's experts before it returns: the guessed copies go first.
        copy_ahead();
        ops::moe_experts(hidden, dispatch, source, K, max_jobs, b.experts.data, b.outputs, s, /*wait_for_cpu=*/true,
                         planned);
        if (streaming != nullptr && call.release_stream) { streaming->after_experts(p.layer, s); }
        // y = bf16(sum_i w_i y_i + s y_shared): the routed entries, then the shared job's T columns.
        const Tensor routed = b.outputs.slice(1, 0, K * T);
        const Tensor shared = b.outputs.slice(1, K * T, T);
        ops::moe_combine(routed, routing, shared, b.y, s);
        ops::residual_add(b.y, residual, s);
        return;
    }
    // The dense shared expert runs on its own stream while the routed experts stage and compute and the
    // host computes the CPU-served misses; it reads only `hidden` and writes only its own buffers. (Its A/B
    // against the compute stream, 2026-10-26, one request: decode +0.4 % and prefill 0.0 % for the compute
    // stream, neither resolved from run-to-run noise.)
    const cudaStream_t shared_s = experts.shared_stream != nullptr ? experts.shared_stream : s;
    if (shared_s != s) {
        CUDA_CHECK(cudaEventRecord(experts.shared_events[0], s));
        CUDA_CHECK(cudaStreamWaitEvent(shared_s, experts.shared_events[0], 0));
    }
    ops::moe_experts(hidden, dispatch, source, K, max_jobs, b.experts.data, b.outputs, s, /*wait_for_cpu=*/false,
                     planned);
    if (streaming != nullptr && call.release_stream) { streaming->after_experts(p.layer, s); }
    copy_ahead();
    dense_shared_expert(hidden, *p.shared, b, shared_s);
    ops::moe_experts_cpu_wait(hidden, dispatch, source, max_jobs, b.experts.data, b.outputs, s);
    if (shared_s != s) {
        CUDA_CHECK(cudaEventRecord(experts.shared_events[1], shared_s));
        CUDA_CHECK(cudaStreamWaitEvent(s, experts.shared_events[1], 0));
    }
    ops::moe_combine(b.outputs, routing, b.shared, b.y, s);
    ops::residual_add(b.y, residual, s);
}

} // namespace infernix::models::qwen3_5::execution
