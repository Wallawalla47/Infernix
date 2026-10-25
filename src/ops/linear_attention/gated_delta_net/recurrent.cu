#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"
#include "ops/linear_attention/gated_delta_net/tree_closed_form.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace infernix::ops::detail::gated_delta_net {
namespace {

static_assert(sizeof(GdnReplayFoldKernelRow) == 16);
static_assert(alignof(GdnReplayFoldKernelRow) == 16);
static_assert(sizeof(GdnReplayFoldKernelRows) == 128);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

// Runs `launch` with the stored state's element type: FP32, or FP16 (fp16-gdn-state).
template <class Launch>
void with_state_type(const Tensor& state, Launch&& launch) {
    if (state.dtype == DType::FP32) {
        launch.template operator()<float>();
    } else if (state.dtype == DType::FP16) {
        launch.template operator()<__half>();
    } else {
        throw std::invalid_argument("gated_delta_net: the recurrent state must be FP32 or FP16");
    }
}

template <bool NormalizeQK, class State>
void launch_recurrent_direct_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& state_read, Tensor& state_write, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_bf16_direct_kernel<NormalizeQK, State><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<const State*>(state_read.data),
        static_cast<State*>(state_write.data), static_cast<__nv_bfloat16*>(out.data), q.ne[2],
        heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

template <bool NormalizeInputs, class State>
void launch_recurrent_batch_update_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         Tensor& ssm_states, const Tensor& source_state_slots,
                                         const Tensor& destination_state_slots, Tensor& out,
                                         cudaStream_t stream, const BatchUpdateGating* gating) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    BatchUpdateAccess<State> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const State*>(ssm_states.data),
        static_cast<State*>(ssm_states.data),
        static_cast<const std::int32_t*>(source_state_slots.data),
        static_cast<const std::int32_t*>(destination_state_slots.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        state_slot_stride,
        scale,
    };
    if (gating != nullptr) {
        // A single column has no stride of its own; its heads are contiguous either way.
        const auto column_stride = [&](const Tensor& t) {
            return t.ne[1] == 1 ? std::int64_t{heads.H_v} : t.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
        };
        access.gate_a    = static_cast<const __nv_bfloat16*>(gating->a.data);
        access.gate_b    = static_cast<const __nv_bfloat16*>(gating->b.data);
        access.gate_a_ld = column_stride(gating->a);
        access.gate_b_ld = column_stride(gating->b);
        access.a_log     = static_cast<const float*>(gating->a_log.data);
        access.dt_bias   = static_cast<const float*>(gating->dt_bias.data);
    }
    CUDA_CHECK(pdl::launch_consumer({grid, block, 0, stream}, recurrent_batch_update_kernel<NormalizeInputs, State>,
                                    access));
}

template <bool Masked, class State>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const RecordAccess<Masked, State> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const State*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    CUDA_CHECK(
        pdl::launch_consumer({grid, block, 0, stream}, recurrent_record_kernel<Masked, State>, access));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class State>
void launch_replay_fold_fixed(const GdnReplayRecords& records,
                              LinearAttentionStateAllLayersView states,
                              const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                              cudaStream_t stream) {
    const FoldAccess<Geometry, State> access{
        static_cast<const __nv_bfloat16*>(records.key.data),
        static_cast<const __nv_bfloat16*>(records.value.data),
        reinterpret_cast<const uint2*>(records.gate.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        static_cast<State*>(states.recurrent_layer0.data),
        static_cast<__nv_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(State)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
    };
    const dim3 grid(static_cast<unsigned>(Geometry::kValueHeads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(Geometry::kLayers * (kStateDim / kBlockDv)));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_fold_kernel<Geometry, State><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_replay_fold_geometry(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                                 const GdnReplayFoldKernelRows& rows, std::int32_t active_rows, cudaStream_t stream) {
    with_state_type(states.recurrent_layer0, [&]<class State>() {
        launch_replay_fold_fixed<Geometry, State>(records, states, rows, active_rows, stream);
    });
}

template <class Geometry>
bool matches_fold_geometry(const GdnReplayRecords& records) {
    return records.spec.layers == Geometry::kLayers && records.spec.qk_heads == Geometry::kQkHeads &&
           records.spec.value_heads == Geometry::kValueHeads &&
           records.spec.conv_channels == Geometry::kConvChannels;
}

} // namespace

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream) {
    launch_recurrent_inout(q, k, v, g, beta, scale, normalize_qk, ssm_state, ssm_state, out, stream);
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream) {
    with_state_type(ssm_state_in, [&]<class State>() {
        if (normalize_qk) {
            launch_recurrent_direct_fixed<true, State>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                                       out, stream);
        } else {
            launch_recurrent_direct_fixed<false, State>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                                        out, stream);
        }
    });
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, Tensor& ssm_states,
                                   const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, Tensor& out,
                                   cudaStream_t stream, const BatchUpdateGating* gating) {
    with_state_type(ssm_states, [&]<class State>() {
        if (normalize_qk) {
            launch_recurrent_batch_update_fixed<true, State>(q, k, v, g, beta, scale, ssm_states,
                                                             source_state_slots, destination_state_slots,
                                                             out, stream, gating);
        } else {
            launch_recurrent_batch_update_fixed<false, State>(q, k, v, g, beta, scale, ssm_states,
                                                              source_state_slots, destination_state_slots,
                                                              out, stream, gating);
        }
    });
}

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream) {
    with_state_type(ssm_states, [&]<class State>() {
        if (valid_columns.data == nullptr) {
            launch_recurrent_record_fixed<false, State>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                                        initial_state_slots, key_record, value_record,
                                                        gate_record, out, stream);
        } else {
            launch_recurrent_record_fixed<true, State>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                                       initial_state_slots, key_record, value_record,
                                                       gate_record, out, stream);
        }
    });
}

void launch_recurrent_tree_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  const Tensor& ssm_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots, const Tensor& tree_rows,
                                  Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                                  Tensor& out, cudaStream_t stream, bool closed_form) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    with_state_type(ssm_states, [&]<class State>() {
        const RecordAccess<true, State> access{
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const float*>(g.data),
            static_cast<const float*>(beta.data),
            static_cast<const State*>(ssm_states.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<__nv_bfloat16*>(key_record.data),
            static_cast<__nv_bfloat16*>(value_record.data),
            reinterpret_cast<uint2*>(gate_record.data),
            static_cast<__nv_bfloat16*>(out.data),
            heads,
            q.ne[2],
            state_slot_stride,
            scale,
        };
        const auto* trees = static_cast<const SpeculativeTreeRow*>(tree_rows.data);
        if (closed_form && q.ne[2] <= kTreeClosedFormColumns) {
            // FastPath::GdnTreeClosedForm (tree_closed_form.cuh); wider rows walk, which is faster there.
            CUDA_CHECK(pdl::launch_consumer({grid, block, 0, stream}, recurrent_tree_closed_form_kernel<State>,
                                            access, trees));
        } else {
            // Two slots hold every branch nesting of a tree with at most four side columns (K+5
            // columns at K=7); deeper nestings rebuild their branch state.
            CUDA_CHECK(pdl::launch_consumer({grid, block, 0, stream}, recurrent_tree_walk_kernel<2, State>, access,
                                            trees));
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream) {
    if (matches_fold_geometry<FoldGeometry48x48>(records)) {
        return launch_replay_fold_geometry<FoldGeometry48x48>(records, states, rows, active_rows, stream);
    }
    if (matches_fold_geometry<FoldGeometry30x32>(records)) {
        return launch_replay_fold_geometry<FoldGeometry30x32>(records, states, rows, active_rows, stream);
    }
    if (matches_fold_geometry<FoldGeometry36x48>(records)) {
        return launch_replay_fold_geometry<FoldGeometry36x48>(records, states, rows, active_rows, stream);
    }
    if (matches_fold_geometry<FoldGeometry36x64>(records)) {
        return launch_replay_fold_geometry<FoldGeometry36x64>(records, states, rows, active_rows, stream);
    }
    throw std::invalid_argument("GDN replay fold launcher received an unregistered geometry");
}

} // namespace infernix::ops::detail::gated_delta_net
