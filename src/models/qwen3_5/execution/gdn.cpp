#include "models/qwen3_5/execution/gdn.h"

#include "infernix/ops/gdn_gating_proj.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/lora.h"
#include "infernix/ops/rmsnorm.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::qwen3_5::execution {
namespace {

// Whether the projection takes a LoRA term (GdnParameters::lora_input's addend).
bool projection_adapted(const GdnParameters& parameters) {
    return parameters.lora_input && parameters.lora_input->addend.present();
}

} // namespace

std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters, std::int32_t first,
                                           std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("GDN projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& w = single->weight;
        return ops::gdn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, single->policy,
                                                            first, last);
    }
    return 0;
}

std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                         std::int32_t batch, std::int32_t first_width,
                                         std::int32_t last_width) {
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    const bool lora    = projection_adapted(parameters);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width, lora);
    } else {
        bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), projection_policy(parameters.projection),
            batch, first_width, last_width, lora);
    }
    // The native overlap contract takes a disjoint span even for a zero-scratch fused route.
    return std::max(std::size_t{1}, bytes);
}

std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                       std::int32_t batch, std::int32_t first_width,
                                       std::int32_t last_width, bool tree) {
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    const bool lora    = projection_adapted(parameters);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width, lora);
        // A verification tree projects through the materialized projection at every width.
        if (tree) {
            bytes = std::max(bytes, ops::gdn_input_proj_workspace_capacity_bytes(
                                        w.qtype, w.n, w.k, single->policy, first_width * batch,
                                        last_width * batch));
        }
    } else {
        bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), projection_policy(parameters.projection),
            batch, first_width, last_width, lora);
    }
    return std::max(std::size_t{1}, bytes);
}

void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream, const ops::LoraAddend* lora) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj(hidden, pair->first, pair->second, qkv, z, pair->policy, stream, lora);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj(hidden, single.weight, qkv, z, single.policy, workspace, stream,
                            ops::LinearRoutes::Exact, lora);
    }
}

std::optional<ops::LoraAddend> gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                                                const GdnParameters& parameters, Tensor& hidden, Tensor& g,
                                                Tensor& beta, Tensor* u, WorkspaceArena& workspace,
                                                DeviceExecutionView execution) {
    if (parameters.lora_input && u == nullptr) {
        throw std::logic_error("GDN control: an adapted input needs its U");
    }
    if (parameters.lora_control.present()) {
        // The control's term reads U = A h, so the norm runs alone (its D = 5120 schedule at every width: a column's h
        // does not depend on the call), then the shrink and the control with its term.
        ops::rmsnorm(residual, norm, epsilon, true, hidden, execution.stream);
        ops::lora_shrink(hidden, parameters.lora_input->a, *u, execution.stream);
        ops::LoraAddend control = parameters.lora_control;
        control.u               = u->data;
        if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.control)) {
            ops::gdn_gating_proj(hidden, pair->first, pair->second, parameters.a_log, parameters.dt_bias, control, g,
                                 beta, execution.stream);
        } else {
            ops::gdn_gating_proj(hidden, std::get<LinearParameters>(parameters.control).weight, parameters.a_log,
                                 parameters.dt_bias, control, g, beta, execution.stream);
        }
    } else {
        if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.control)) {
            ops::gdn_norm_gating_proj(residual, norm, epsilon, pair->first, pair->second, parameters.a_log,
                                      parameters.dt_bias, pair->policy, workspace, hidden, g, beta, execution);
        } else {
            const auto& single = std::get<LinearParameters>(parameters.control);
            ops::gdn_norm_gating_proj(residual, norm, epsilon, single.weight, parameters.a_log, parameters.dt_bias,
                                      single.policy, workspace, hidden, g, beta, execution);
        }
        if (parameters.lora_input) {
            ops::lora_shrink(hidden, parameters.lora_input->a, *u, execution.stream);
        }
    }
    if (!projection_adapted(parameters)) { return std::nullopt; }
    ops::LoraAddend projection = parameters.lora_input->addend;
    projection.u               = u->data;
    return projection;
}

void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream, const ops::LoraAddend* lora) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(gdn_snapshot_workspace_bytes(
        parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, pair->first, pair->second, parameters.convolution,
                                          conv_states, valid_columns, initial_slots,
                                          destination_slots, query, key, value, z, pair->policy,
                                          scratch, stream, lora);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_snapshot(
            hidden, single.weight, parameters.convolution, conv_states, valid_columns,
            initial_slots, destination_slots, query, key, value, z, single.policy, scratch, stream, lora);
    }
}

void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, WorkspaceArena& workspace, cudaStream_t stream,
                           const Tensor* tree_rows, const ops::LoraAddend* lora) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(gdn_record_workspace_bytes(
        parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1], tree_rows != nullptr)));
    if (tree_rows != nullptr) {
        const auto* single = std::get_if<LinearParameters>(&parameters.projection);
        if (single == nullptr) {
            throw std::invalid_argument("tree verification needs a single GDN parent");
        }
        ops::gdn_input_proj_conv_record(hidden, single->weight, parameters.convolution, conv_states,
                                        valid_columns, initial_slots, *tree_rows, conv_record,
                                        query, key, value, z, single->policy, scratch, stream, lora);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_record(hidden, pair->first, pair->second, parameters.convolution,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, z, pair->policy, scratch, stream, lora);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_record(hidden, single.weight, parameters.convolution, conv_states,
                                        valid_columns, initial_slots, conv_record, query, key,
                                        value, z, single.policy, scratch, stream, lora);
    }
}

std::optional<ops::LoraAddend> gdn_output_lora(const GdnParameters& parameters, const Tensor& input,
                                               WorkspaceArena& workspace, cudaStream_t stream) {
    if (!parameters.lora_output) { return std::nullopt; }
    Tensor u = workspace.alloc(DType::BF16, {parameters.lora_output->a.rows, input.ne[1]});
    ops::lora_shrink(input, parameters.lora_output->a, u, stream);
    ops::LoraAddend lora = parameters.lora_output->addend;
    lora.u               = u.data;
    return lora;
}

} // namespace infernix::models::qwen3_5::execution
