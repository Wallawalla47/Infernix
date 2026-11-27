#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include <algorithm>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
struct Plan {
    Q8DynamicConvAddSchedule schedule;
    std::size_t workspace_bytes;
};

// The registered drafters: H = 5120 with C in {4096, 17408} (Qwen3.8-27B), H = 2560 with C in
// {4096, 7680} (Qwen3.8-Flash-Next), H = 3072 with C in {4096, 9216} (Qwen3.5-122B-A10B) and H = 2304
// with C in {4096, 6912} (Inkling 2 Mini).
bool registered(int hidden, int input_rows) {
    return (hidden == 5120 && (input_rows == 4096 || input_rows == 17408)) ||
           (hidden == 2560 && (input_rows == 4096 || input_rows == 7680)) ||
           (hidden == 3072 && (input_rows == 4096 || input_rows == 9216)) ||
           (hidden == 2304 && (input_rows == 4096 || input_rows == 6912));
}

Plan resolve_plan(int hidden, int input_rows, int width, int batch) {
    if (!registered(hidden, input_rows))
        throw std::invalid_argument(
            "linear dynamic grouped conv add: (H, C) must be (5120, 4096|17408), (2560, 4096|7680), (3072, 4096|9216) "
            "or (2304, 4096|6912)");
    if (width < 2 || width > 16 || batch < 1 || batch > 8)
        throw std::invalid_argument("linear dynamic grouped conv add: invalid W/B profile");
    const int columns = width * batch;
    using S           = Q8DynamicConvAddSchedule;
    const S schedule  = columns <= 88 ? S::TiledMma : S::MmaK128;
    return {schedule, static_cast<std::size_t>(hidden) * columns * sizeof(std::uint16_t)};
}
} // namespace

bool nvfp4_parent_registered(int hidden, int input_rows) noexcept { return hidden == 2304 && input_rows == 6912; }

std::size_t q8_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    int hidden, int input_rows, int min_width, int max_width, int min_batch, int max_batch, QType qtype) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument(
            "linear dynamic grouped conv add workspace: invalid W/B interval");
    // Every route shares one projected BF16 matrix (a multiple of 256 bytes at every registered H); its capacity is
    // monotonic in W and B. An NVFP4 parent adds the NVFP4 linear's A16 workspace after it.
    const std::size_t projected = resolve_plan(hidden, input_rows, max_width, max_batch).workspace_bytes;
    if (qtype != QType::NVFP4) { return projected; }
    if (!nvfp4_parent_registered(hidden, input_rows)) {
        throw std::invalid_argument("linear dynamic grouped conv add workspace: no NVFP4 route at this (H, C)");
    }
    return projected + ops::linear_workspace_capacity_bytes(QType::NVFP4, hidden, input_rows, LinearPolicy::A16Only,
                                                            min_width * min_batch, max_width * max_batch);
}

const char* q8_linear_dynamic_grouped_conv_add_route_name(int hidden, int input_rows, int width,
                                                          int batch) {
    switch (resolve_plan(hidden, input_rows, width, batch).schedule) {
    case Q8DynamicConvAddSchedule::TiledMma:
        return "dynamic_grouped_conv_add.q8.tiled_mma.materialized_bf16";
    case Q8DynamicConvAddSchedule::MmaK128:
        return "dynamic_grouped_conv_add.q8.r64_c64_k128.materialized_bf16";
    }
    throw std::logic_error("linear dynamic grouped conv add: invalid production schedule");
}

void q8_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& weight,
                                                 const Tensor& base, const Tensor& delta,
                                                 Tensor& residual, WorkspaceArena& workspace,
                                                 cudaStream_t stream) {
    const int hidden         = residual.ne[0];
    const Plan plan          = resolve_plan(hidden, x.ne[0], x.ne[1], x.ne[2]);
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(plan.workspace_bytes);
    Tensor projected(storage.data, DType::BF16, {hidden, x.ne[1], x.ne[2]});
    if (weight.qtype == QType::NVFP4) {
        // The NVFP4 linear's A16 route projects every column, then the finish (the Q8 route's, on its matrix).
        const int tokens  = x.ne[1] * x.ne[2];
        const Tensor flat = x.view({x.ne[0], tokens});
        Tensor result     = projected.view({hidden, tokens});
        ops::linear(flat, weight, result, LinearPolicy::A16Only, workspace, stream);
        dynamic_grouped_conv_add_finish_launch(projected, base, delta, residual, x.ne[1], tokens, stream);
        return;
    }
    q8_dynamic_grouped_conv_add_materialized_launch(plan.schedule, x, weight, base, delta, residual,
                                                    projected, stream);
}
} // namespace infernix::ops::detail
