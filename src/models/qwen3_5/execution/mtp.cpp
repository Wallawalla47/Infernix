#include "models/qwen3_5/execution/mtp.h"

#include "core/layout.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_pair.h"
#include "infernix/ops/mtp_pack.h"

#include <algorithm>
#include <array>

namespace infernix::models::qwen3_5::execution {

namespace {

std::size_t row_bytes(const LinearParameters& p, std::int32_t first, std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                                first, last);
}

// The K/V rows: one linear_pair call, or under LinearPolicy::A16ColumnInvariant (which
// linear_pair has no route for) two linear calls.
std::size_t kv_rows_bytes(const std::array<LinearParameters, 4>& rows, std::int32_t first,
                          std::int32_t last) {
    if (ops::column_invariant(rows[1].policy)) {
        return std::max(row_bytes(rows[1], first, last), row_bytes(rows[3], first, last));
    }
    return ops::linear_pair_workspace_capacity_bytes(rows[1].weight, rows[3].weight, first, last);
}

} // namespace

std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                           std::int32_t first, std::int32_t last) {
    if (!parameters.packed) {
        // The fused packed path is unavailable; the row-split path projects K/V and Q/gate
        // independently. Size the peak of the two independent projections.
        const auto& rows = *parameters.rows;
        const auto qgate = std::max(row_bytes(rows[0], first, last), row_bytes(rows[2], first, last));
        return std::max(qgate, kv_rows_bytes(rows, first, last));
    }
    const auto& p = *parameters.packed;
    const auto& w = p.weight;
    if (!parameters.rows) {
        return ops::attn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first,
                                                             last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(
        ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                   const AttentionConfig& config, std::int32_t first,
                                   std::int32_t last) {
    if (parameters.rows) { return kv_rows_bytes(*parameters.rows, first, last); }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                           const AttentionConfig& config, std::int32_t first,
                                           std::int32_t last) {
    if (parameters.rows) {
        const auto bytes = [&](std::size_t index) {
            const auto& p = (*parameters.rows)[index];
            const auto& w = p.weight;
            return ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last);
        };
        return std::max(bytes(0), bytes(2));
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!parameters.packed) {
        // The four attention weights do not share one contiguous parent region, so the fused
        // packed / attn_input_proj path is unavailable. Project K/V and Q/gate independently.
        if (!parameters.rows) {
            throw std::invalid_argument("mtp_projection: no available projection");
        }
        mtp_query_gate_projection(hidden, parameters, config, query, gate, workspace, stream);
        mtp_kv_projection(hidden, parameters, config, key, value, workspace, stream);
        return;
    }
    const auto& p = *parameters.packed;
    if (!parameters.rows) {
        ops::attn_input_proj(hidden, p.weight, query, gate, key, value, p.policy, workspace,
                             stream);
        return;
    }
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    Tensor packed      = workspace.alloc(DType::BF16, {p.weight.n, columns});
    ops::linear(hidden, p.weight, packed, p.policy, workspace, stream);
    Tensor q =
        query.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor k =
        key.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    Tensor g =
        gate.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor v =
        value.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    ops::mtp_split_attn_in(packed, q, k, g, v, stream);
}

void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        const LinearParameters& k = (*parameters.rows)[1];
        const LinearParameters& v = (*parameters.rows)[3];
        if (ops::column_invariant(k.policy)) {
            {
                auto scope = workspace.scope();
                ops::linear(hidden, k.weight, key, k.policy, workspace, stream);
            }
            ops::linear(hidden, v.weight, value, v.policy, workspace, stream);
            return;
        }
        ops::linear_pair(hidden, k.weight, v.weight, key, value, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor query = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    Tensor gate  = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        const auto& q = (*parameters.rows)[0];
        const auto& g = (*parameters.rows)[2];
        {
            auto scope = workspace.scope();
            ops::linear(hidden, q.weight, query, q.policy, workspace, stream);
        }
        ops::linear(hidden, g.weight, gate, g.policy, workspace, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor key   = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    Tensor value = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

} // namespace infernix::models::qwen3_5::execution
