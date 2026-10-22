#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_geometry.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops::detail {
namespace {

// The registered parents by K: [16384,5120] (q/k/v 10240, z 6144; a fused one-row route and the A8
// residual profile) and Qwen3.5-122B-A10B's static-scale [20480,3072] (q/k/v 12288, z 8192).
struct Fp8GdnConvShape {
    std::int32_t hidden;
    std::int32_t channels;
    std::int32_t z_rows;
    bool fused;
    bool residual;
};

Fp8GdnConvShape conv_shape(std::int32_t input_rows) {
    if (input_rows == Fp8N16384K5120::kInputRows) { return {input_rows, 10240, 6144, true, true}; }
    if (input_rows == 3072) { return {input_rows, 12288, 8192, false, false}; }
    throw std::invalid_argument("fp8 GDN conv: unsupported K");
}

enum class Fp8GdnConvScheduleId : std::uint8_t {
    FusedA16,
    MaterializedA16,
    MaterializedA8,
    // A16ColumnInvariant: the column-invariant projection into the BF16 plane at every B and W,
    // then the per-column convolution, so a row's result does not depend on the other rows.
    MaterializedInvariantA16,
    // AllowA8Residual: the two-plane A8 projection, then the generic convolution.
    MaterializedA8Residual,
};

struct Fp8GdnConvPlan {
    Fp8GdnConvScheduleId schedule;
};

struct Fp8GdnProjectedWorkspace {
    Tensor projected;
};

template <class Allocator>
Fp8GdnProjectedWorkspace allocate_projected(Allocator& allocator, const Fp8GdnConvShape& shape,
                                            std::int32_t columns) {
    return {allocator.alloc(DType::BF16, {shape.channels, columns}, 256)};
}

void require_policy(LinearPolicy policy, const char* operation) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument(std::string(operation) + ": invalid compute policy");
    }
}

Fp8GdnConvPlan b1_a16_plan(const Fp8GdnConvShape& shape, std::int32_t width) {
    const bool fused = shape.fused && width <= 3;
    return {fused ? Fp8GdnConvScheduleId::FusedA16 : Fp8GdnConvScheduleId::MaterializedA16};
}

std::size_t snapshot_capacity(const Fp8GdnConvShape& shape, Fp8GdnConvPlan maximum_plan,
                              std::int32_t materialized_columns, std::int32_t maximum_columns) {
    if (materialized_columns == 0) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected(layout, shape, materialized_columns);
    if (maximum_plan.schedule == Fp8GdnConvScheduleId::MaterializedA8) {
        (void)allocate_fp8_a8_workspace(
            layout, maximum_columns, shape.hidden,
            fp8_gdn_input_partial_capacity_bytes(shape.hidden, maximum_columns));
    } else if (maximum_plan.schedule == Fp8GdnConvScheduleId::MaterializedA8Residual) {
        (void)allocate_fp8_a8_workspace(layout, maximum_columns, shape.hidden, 0, true);
    }
    return layout.peak_bytes(1);
}

std::size_t record_capacity(const Fp8GdnConvShape& shape, Fp8GdnConvPlan plan,
                            std::int32_t aggregate_columns) {
    if (plan.schedule == Fp8GdnConvScheduleId::MaterializedA8Residual) {
        return fp8_a8_workspace_capacity_bytes(aggregate_columns, shape.hidden, 0, true);
    }
    if (plan.schedule != Fp8GdnConvScheduleId::MaterializedA8) { return 0; }
    return fp8_a8_workspace_capacity_bytes(
        aggregate_columns, shape.hidden,
        fp8_gdn_input_partial_capacity_bytes(shape.hidden, aggregate_columns));
}

void launch_projection(const Tensor& x, const Weight& weight, Tensor& projected, Tensor& z,
                       Fp8GdnConvScheduleId schedule, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    if (schedule == Fp8GdnConvScheduleId::MaterializedA16) {
        fp8_gdn_input_a16_dispatch(x, weight, projected, z, stream);
        return;
    }
    if (schedule == Fp8GdnConvScheduleId::MaterializedInvariantA16) {
        fp8_gdn_input_invariant_launch(x, weight, projected, z, stream);
        return;
    }
    if (schedule == Fp8GdnConvScheduleId::MaterializedA8) {
        fp8_gdn_input_a8_dispatch(x, weight, projected, z, workspace, stream);
        return;
    }
    if (schedule == Fp8GdnConvScheduleId::MaterializedA8Residual) {
        fp8_gdn_input_a8_residual_dispatch(x, weight, projected, z, workspace, stream);
        return;
    }
    throw std::logic_error("fp8 GDN materialized projection received a fused plan");
}

// Record and snapshot choose the same arithmetic for the same physical block.
Fp8GdnConvPlan fp8_gdn_conv_resolve_plan(const Fp8GdnConvShape& shape, LinearPolicy policy,
                                         std::int32_t width, std::int32_t batch_size) {
    if (column_invariant(policy)) { return {Fp8GdnConvScheduleId::MaterializedInvariantA16}; }
    if (allows_a8_residual(policy) && width * batch_size >= 17) {
        if (!shape.residual) {
            throw std::invalid_argument(
                "fp8 GDN conv: a static input scale has no A8 residual route");
        }
        return {Fp8GdnConvScheduleId::MaterializedA8Residual};
    }
    if (batch_size == 1) {
        if (allows_a8(policy) && width >= 17) { return {Fp8GdnConvScheduleId::MaterializedA8}; }
        return b1_a16_plan(shape, width);
    }
    if (allows_a8(policy) && width * batch_size >= 17) {
        return {Fp8GdnConvScheduleId::MaterializedA8};
    }
    return {Fp8GdnConvScheduleId::MaterializedA16};
}

Fp8GdnConvPlan fp8_gdn_snapshot_resolve_plan(const Fp8GdnConvShape& shape, LinearPolicy policy,
                                             std::int32_t width, std::int32_t batch_size) {
    require_policy(policy, "fp8 GDN snapshot");
    if (width <= 0 || batch_size <= 0 || batch_size > 8 || (batch_size > 1 && width > 64)) {
        throw std::invalid_argument("fp8 GDN snapshot: invalid B/W domain");
    }
    return fp8_gdn_conv_resolve_plan(shape, policy, width, batch_size);
}

Fp8GdnConvPlan fp8_gdn_record_resolve_plan(const Fp8GdnConvShape& shape, LinearPolicy policy,
                                           std::int32_t width, std::int32_t batch_size) {
    require_policy(policy, "fp8 GDN record");
    if (width < 2 || width > 64 || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument("fp8 GDN record: invalid B/W domain");
    }
    return fp8_gdn_conv_resolve_plan(shape, policy, width, batch_size);
}

} // namespace

std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy, std::int32_t input_rows,
                                                      std::int32_t batch_size,
                                                      std::int32_t min_width,
                                                      std::int32_t max_width) {
    if (min_width <= 0 || max_width < min_width) {
        throw std::invalid_argument("fp8 GDN snapshot workspace: invalid width interval");
    }
    const Fp8GdnConvShape shape = conv_shape(input_rows);
    (void)fp8_gdn_snapshot_resolve_plan(shape, policy, min_width, batch_size);
    const Fp8GdnConvPlan maximum =
        fp8_gdn_snapshot_resolve_plan(shape, policy, max_width, batch_size);
    const bool fused_minimum = shape.fused && !column_invariant(policy) && batch_size == 1;
    const std::int32_t largest_materialized_width =
        !fused_minimum || max_width > 3 ? max_width : 0;
    return snapshot_capacity(shape, maximum, batch_size * largest_materialized_width,
                             batch_size * max_width);
}

std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy policy, std::int32_t input_rows,
                                                    std::int32_t batch_size,
                                                    std::int32_t min_width,
                                                    std::int32_t max_width) {
    if (min_width < 2 || max_width < min_width) {
        throw std::invalid_argument("fp8 GDN record workspace: invalid width interval");
    }
    const Fp8GdnConvShape shape = conv_shape(input_rows);
    (void)fp8_gdn_record_resolve_plan(shape, policy, min_width, batch_size);
    const Fp8GdnConvPlan maximum =
        fp8_gdn_record_resolve_plan(shape, policy, max_width, batch_size);
    return record_capacity(shape, maximum, batch_size * max_width);
}

namespace {

void launch_snapshot_plan(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                          Tensor& conv_states, const Tensor& valid_columns,
                          const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                          Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                          const Fp8GdnConvShape& shape, Fp8GdnConvPlan plan,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (plan.schedule == Fp8GdnConvScheduleId::FusedA16) {
        fp8_gdn_snapshot_fused_launch(x, weight, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value, z,
                                      stream);
        return;
    }

    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    const std::int32_t aggregate_columns = width * batch;
    auto scope                           = workspace.scope();
    Fp8GdnProjectedWorkspace scratch     = allocate_projected(workspace, shape, aggregate_columns);
    Tensor x_flat(x.data, DType::BF16, {shape.hidden, aggregate_columns});
    Tensor z_flat(z.data, DType::BF16, {shape.z_rows, aggregate_columns});
    launch_projection(x_flat, weight, scratch.projected, z_flat, plan.schedule, workspace, stream);

    Tensor projected(scratch.projected.data, DType::BF16, {shape.channels, width, batch});
    gdn_projected_conv_snapshot_launch(projected, conv_weight, conv_states, valid_columns,
                                       initial_slot, snapshot_base_slot, query, key, value, stream);
}

void launch_record_plan(const Tensor* tree_rows, const Tensor& x, const Weight& weight,
                        const Tensor& conv_weight, const Tensor& conv_states,
                        const Tensor& valid_columns, const Tensor& initial_slot,
                        Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                        const Fp8GdnConvShape& shape, Fp8GdnConvPlan plan,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    // A tree convolves over ancestors, which the fused epilogue cannot address.
    if (tree_rows != nullptr && plan.schedule == Fp8GdnConvScheduleId::FusedA16) {
        plan.schedule = Fp8GdnConvScheduleId::MaterializedA16;
    }
    if (plan.schedule == Fp8GdnConvScheduleId::FusedA16) {
        fp8_gdn_record_fused_launch(x, weight, conv_weight, conv_states, valid_columns,
                                    initial_slot, conv_record, query, key, value, z, stream);
        return;
    }

    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    const std::int32_t aggregate_columns = width * batch;
    auto scope                           = workspace.scope();
    Tensor x_flat(x.data, DType::BF16, {shape.hidden, aggregate_columns});
    Tensor record_flat(conv_record.data, DType::BF16, {shape.channels, aggregate_columns});
    Tensor z_flat(z.data, DType::BF16, {shape.z_rows, aggregate_columns});
    launch_projection(x_flat, weight, record_flat, z_flat, plan.schedule, workspace, stream);
    if (tree_rows != nullptr) {
        gdn_projected_conv_record_tree_launch(conv_record, conv_weight, conv_states, valid_columns,
                                              initial_slot, *tree_rows, query, key, value, stream);
        return;
    }
    gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                     initial_slot, query, key, value, stream);
}

} // namespace

void fp8_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                               Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                               Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    const Fp8GdnConvShape shape = conv_shape(weight.k);
    launch_snapshot_plan(x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                         snapshot_base_slot, query, key, value, z, shape,
                         fp8_gdn_snapshot_resolve_plan(shape, policy, x.ne[1], x.ne[2]),
                         workspace, stream);
}

void fp8_gdn_record_dispatch(const Tensor* tree_rows, const Tensor& x, const Weight& weight,
                             const Tensor& conv_weight, const Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slot,
                             Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                             Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    const Fp8GdnConvShape shape = conv_shape(weight.k);
    launch_record_plan(tree_rows, x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                       conv_record, query, key, value, z, shape,
                       fp8_gdn_record_resolve_plan(shape, policy, x.ne[1], x.ne[2]), workspace,
                       stream);
}

} // namespace infernix::ops::detail
