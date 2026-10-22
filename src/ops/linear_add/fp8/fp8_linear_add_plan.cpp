#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_geometry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

enum class Fp8LinearAddRoute : std::uint8_t {
    ColumnInvariantA16,
    A16,
    A8,
};

// Qwen3.5-122B-A10B's [3072,8192]; the other registered problems are [5120,6144] and [5120,17408].
bool n3072(std::int32_t output_rows, std::int32_t input_rows) {
    return output_rows == 3072 && input_rows == 8192;
}

Fp8LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                LinearPolicy policy, std::int32_t tokens) {
    const bool n5120 =
        output_rows == Fp8N5120K6144::kOutputRows &&
        (input_rows == Fp8N5120K6144::kInputRows || input_rows == Fp8N5120K17408::kInputRows);
    if (tokens <= 0 || !(n5120 || n3072(output_rows, input_rows))) {
        throw std::invalid_argument("fp8 linear_add: unsupported shape");
    }
    if (column_invariant(policy)) { return Fp8LinearAddRoute::ColumnInvariantA16; }
    // Two E4M3 terms (AllowA8Residual) never beat A16 on these 5,120 outputs: 34 against 29 us at
    // 17-32 columns of K=6144 and 75-116 against 68-121 us at 24-128 of K=17408 (RTX 5090), the
    // doubled MMA work on too few CTAs. The static-scale [3072,8192] has no residual profile.
    if (policy == LinearPolicy::A16Only || allows_residual(policy)) {
        return Fp8LinearAddRoute::A16;
    }
    const std::int32_t first_a8 = input_rows == Fp8N5120K17408::kInputRows ? 20 : 17;
    if (!allows_a8(policy)) { throw std::invalid_argument("fp8 linear_add: unsupported policy"); }
    return tokens >= first_a8 ? Fp8LinearAddRoute::A8 : Fp8LinearAddRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    if (n3072(weight.n, weight.k)) return fp8_linear_add_n3072_a16_launch(x, weight, residual, stream);
    if (x.ne[1] == 1) return fp8_linear_add_decode_launch(x, weight, residual, stream);
    fp8_linear_add_matrix_launch(x, weight, residual, stream);
}

} // namespace

std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    switch (resolve_route(output_rows, input_rows, policy, max_tokens)) {
    case Fp8LinearAddRoute::A8:
        return fp8_a8_workspace_capacity_bytes(
            max_tokens, input_rows,
            n3072(output_rows, input_rows)
                ? fp8_linear_add_n3072_partial_capacity_bytes(max_tokens)
                : fp8_linear_add_partial_capacity_bytes(input_rows, max_tokens));
    case Fp8LinearAddRoute::ColumnInvariantA16:
    case Fp8LinearAddRoute::A16:
        break;
    }
    return 0;
}

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    const Fp8LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    const bool static_scale       = n3072(weight.n, weight.k);
    if (route == Fp8LinearAddRoute::ColumnInvariantA16) {
        if (static_scale) {
            fp8_linear_add_n3072_invariant_launch(x, weight, residual, stream);
        } else {
            fp8_linear_add_invariant_launch(x, weight, residual, stream);
        }
        return;
    }
    if (route == Fp8LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
    if (static_scale) {
        fp8_linear_add_n3072_a8_launch(x, weight, residual, workspace, stream);
    } else {
        fp8_linear_add_a8_launch(x, weight, residual, workspace, stream);
    }
}

} // namespace infernix::ops::detail
