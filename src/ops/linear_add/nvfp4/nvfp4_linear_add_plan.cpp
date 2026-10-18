#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    ColumnInvariantA16,
    A16,
    A4,
    A4Residual,
};

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 || (input_rows != 6144 && input_rows != 17408)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (column_invariant(policy)) { return Nvfp4LinearAddRoute::ColumnInvariantA16; }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
        allows_a8_residual(policy)) {
        return Nvfp4LinearAddRoute::A16;
    }
    if (allows_a4_residual(policy)) {
        // Two-term K=17408: 37.5 us at 8-14 tokens, against A16's 37.7-38.0 at 8 and 44.4-51.8 at
        // 9-14 (RTX 5090, cold weights): the residual's error buys time from 9.
        return tokens >= 9 ? Nvfp4LinearAddRoute::A4Residual : Nvfp4LinearAddRoute::A16;
    }
    if (!allows_a4(policy)) { throw std::invalid_argument("nvfp4 linear_add: unsupported policy"); }
    // With the three-stage A4 tile, K=6144 is ahead of A16 from 8 tokens too: 21.1 us against
    // 23.1-31.3 us at 8-16 tokens on RTX 5090.
    const std::int32_t first_a4 = 8;
    return tokens >= first_a4 ? Nvfp4LinearAddRoute::A4 : Nvfp4LinearAddRoute::A16;
}


} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    switch (resolve_route(output_rows, input_rows, policy, max_tokens)) {
    case Nvfp4LinearAddRoute::A4:
        return nvfp4_a4_workspace_capacity_bytes(max_tokens, input_rows);
    case Nvfp4LinearAddRoute::A4Residual:
        return nvfp4_a4_workspace_capacity_bytes(max_tokens, input_rows, kNvfp4A4ResidualTerms);
    case Nvfp4LinearAddRoute::ColumnInvariantA16:
    case Nvfp4LinearAddRoute::A16:
        break;
    }
    return 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream, bool wide_down_tile) {
    const Nvfp4LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Nvfp4LinearAddRoute::ColumnInvariantA16) {
        nvfp4_linear_add_invariant_launch(x, weight, residual, stream);
        return;
    }
    if (route == Nvfp4LinearAddRoute::A16) {
        // The faster 8 K-warp tiles change the reduction order: their fast path's route flag only.
        nvfp4_linear_add_a16_launch(x, weight, residual, wide_down_tile, stream);
        return;
    }
    auto scope = workspace.scope();
    if (route == Nvfp4LinearAddRoute::A4Residual) {
        nvfp4_linear_add_a4_residual_launch(
            x, weight, residual,
            allocate_nvfp4_a4_workspace(workspace, x.ne[1], weight.k, kNvfp4A4ResidualTerms),
            stream);
        return;
    }
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_a4_launch(x, weight, residual, scratch, stream);
}

} // namespace infernix::ops::detail
