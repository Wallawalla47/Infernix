#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "ops/linear/fp8/fp8_geometry.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

enum class Fp8AttnInputRoute : std::uint8_t {
    ColumnInvariantA16,
    A16,
    A8,
    A8Residual,
};

Fp8AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 attn_input_proj: T must be positive"); }
    if (column_invariant(policy)) { return Fp8AttnInputRoute::ColumnInvariantA16; }
    if (policy == LinearPolicy::A16Only || allows_a4_residual(policy)) {
        return Fp8AttnInputRoute::A16;
    }
    if (allows_a8_residual(policy)) {
        return tokens >= 17 ? Fp8AttnInputRoute::A8Residual : Fp8AttnInputRoute::A16;
    }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 attn_input_proj: unsupported policy");
    }
    return tokens >= 17 ? Fp8AttnInputRoute::A8 : Fp8AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1)
        fp8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= kFp8AttnInputLastSmallMmaT)
        fp8_attn_input_a16_small_mma_launch(x, weight, q, gate, k, v, stream);
    else
        fp8_attn_input_a16_gemm_launch(x, weight, q, gate, k, v, stream);
}

} // namespace

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    switch (resolve_route(policy, max_tokens)) {
    case Fp8AttnInputRoute::A8:
        return fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N14336K5120::kInputRows,
                                               fp8_attn_input_partial_capacity_bytes(max_tokens));
    case Fp8AttnInputRoute::A8Residual:
        return fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N14336K5120::kInputRows, 0, true);
    case Fp8AttnInputRoute::ColumnInvariantA16:
    case Fp8AttnInputRoute::A16:
        break;
    }
    return 0;
}

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream) {
    const Fp8AttnInputRoute route = resolve_route(policy, x.ne[1]);
    if (route == Fp8AttnInputRoute::ColumnInvariantA16) {
        fp8_attn_input_invariant_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (route == Fp8AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 attn_input_proj requires caller workspace");
    }
    auto scope = workspace->scope();
    if (route == Fp8AttnInputRoute::A8Residual) {
        fp8_attn_input_a8_residual_launch(
            x, weight, q, gate, k, v,
            allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k, 0, true), stream);
        return;
    }
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k, fp8_attn_input_partial_capacity_bytes(x.ne[1]));
    fp8_attn_input_a8_launch(x, weight, q, gate, k, v, scratch, stream);
}

std::size_t fp8_attn_input_n17408_workspace_capacity_bytes(LinearPolicy policy,
                                                           std::int32_t min_tokens,
                                                           std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 attn_input_proj workspace: invalid token interval");
    }
    for (const std::int32_t tokens : {min_tokens, max_tokens}) {
        if (resolve_route(policy, tokens) == Fp8AttnInputRoute::A8Residual) {
            throw std::invalid_argument(
                "fp8 attn_input_proj: a static input scale has no A8 residual route");
        }
    }
    if (resolve_route(policy, max_tokens) != Fp8AttnInputRoute::A8) { return 0; }
    return fp8_a8_workspace_capacity_bytes(max_tokens, 3072,
                                           fp8_attn_input_n17408_partial_capacity_bytes(max_tokens));
}

void fp8_attn_input_n17408_dispatch(const Tensor& x, const Weight& weight, Tensor& q,
                                    Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                                    WorkspaceArena* workspace, cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Fp8AttnInputRoute::ColumnInvariantA16:
        fp8_attn_input_n17408_invariant_launch(x, weight, q, gate, k, v, stream);
        return;
    case Fp8AttnInputRoute::A16:
        fp8_attn_input_n17408_a16_launch(x, weight, q, gate, k, v, stream);
        return;
    case Fp8AttnInputRoute::A8Residual:
        throw std::invalid_argument(
            "fp8 attn_input_proj: a static input scale has no A8 residual route");
    case Fp8AttnInputRoute::A8:
        break;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 attn_input_proj requires caller workspace");
    }
    auto scope                   = workspace->scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k, fp8_attn_input_n17408_partial_capacity_bytes(x.ne[1]));
    fp8_attn_input_n17408_a8_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace infernix::ops::detail
