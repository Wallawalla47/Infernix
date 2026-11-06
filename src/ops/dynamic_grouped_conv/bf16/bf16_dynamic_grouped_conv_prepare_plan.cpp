#include "core/weight.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "infernix/ops/rmsnorm.h"
#include <algorithm>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next's drafter calls (H 2560, at most 16 columns: one row of up to 16, or a pair of up
// to 8) take the fused kernel; every other shape the RMSNorm, split-K projection and reduction.
DynamicConvPrepareRoute resolve_route(int hidden, int tokens) {
    if (bf16_dynamic_grouped_conv_prepare_fused_supported(hidden, tokens)) return {0, 0, 0, true};
    // Complete cold-cache Graph timings favor split-8 R16 through 48 columns.
    // R32 halves split scratch and avoids the repeated wide tile above that range.
    if (tokens <= 48) return {16, tokens <= 8 ? 8 : tokens <= 16 ? 16 : tokens <= 32 ? 32 : 48, 8};
    return {32, tokens <= 96 ? 32 : 64, 4};
}

// The split partials of every coefficient row (H / 4 of them); the fused kernel needs none.
std::size_t capacity(int hidden, DynamicConvPrepareRoute route, int tokens) {
    return route.fused ? 0 : static_cast<std::size_t>(hidden / 4) * tokens * route.split_k * sizeof(float);
}

void execute(DynamicConvPrepareRoute route, const Tensor& residual, const Tensor& norm, float eps,
             const Tensor& base, const Weight& weight, Tensor& prepared, Tensor& finish,
             WorkspaceArena& workspace, cudaStream_t stream) {
    if (route.fused) {
        bf16_dynamic_grouped_conv_prepare_fused_launch(residual, norm, eps, base, weight, prepared, finish, stream);
        return;
    }
    auto scope       = workspace.scope();
    const int tokens = residual.ne[1] * residual.ne[2];
    float* partial   = static_cast<float*>(
        workspace.alloc_bytes(capacity(residual.ne[0], route, tokens)).data);
    rmsnorm(residual, norm, eps, false, prepared, stream);
    bf16_dynamic_grouped_conv_prepare_partial_launch(route, prepared, weight, partial, stream);
    bf16_dynamic_grouped_conv_prepare_reduce_launch(route, base, partial, prepared, finish, stream);
}
} // namespace

std::size_t bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(int hidden, int min_width,
                                                                       int max_width,
                                                                       int min_batch,
                                                                       int max_batch) {
    if (hidden != 5120 && hidden != 2560 && hidden != 3072)
        throw std::invalid_argument("dynamic grouped conv prepare workspace: H must be 5120, 2560 or 3072");
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument("dynamic grouped conv prepare workspace: invalid W/B interval");
    std::size_t maximum = 0;
    for (int w = min_width; w <= max_width; ++w)
        for (int b = min_batch; b <= max_batch; ++b)
            maximum = std::max(maximum, capacity(hidden, resolve_route(hidden, w * b), w * b));
    return maximum;
}

void bf16_dynamic_grouped_conv_prepare_dispatch(const Tensor& residual, const Tensor& norm,
                                                float eps, const Tensor& base, const Weight& weight,
                                                Tensor& prepared, Tensor& finish,
                                                WorkspaceArena& workspace, cudaStream_t stream) {
    execute(resolve_route(residual.ne[0], residual.ne[1] * residual.ne[2]), residual, norm, eps, base, weight,
            prepared, finish, workspace, stream);
}
} // namespace infernix::ops::detail
