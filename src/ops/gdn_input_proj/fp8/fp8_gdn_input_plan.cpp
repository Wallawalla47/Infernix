#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "ops/linear/fp8/fp8_geometry.h"

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

enum class Fp8GdnInputRoute : std::uint8_t {
    ColumnInvariantA16,
    A16,
    A8,
    A8Residual,
};

Fp8GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 gdn_input_proj: T must be positive"); }
    if (column_invariant(policy)) { return Fp8GdnInputRoute::ColumnInvariantA16; }
    // `--a8-activations`: the one-term 8-bit route at every width.
    if (a8_only(policy)) { return Fp8GdnInputRoute::A8; }
    if (policy == LinearPolicy::A16Only || allows_a4_residual(policy)) {
        return Fp8GdnInputRoute::A16;
    }
    if (allows_a8_residual(policy)) {
        return tokens >= 17 ? Fp8GdnInputRoute::A8Residual : Fp8GdnInputRoute::A16;
    }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 gdn_input_proj: unsupported policy");
    }
    return tokens >= 17 ? Fp8GdnInputRoute::A8 : Fp8GdnInputRoute::A16;
}

// The Qwen3.5-122B-A10B parent [20480,3072]; otherwise the [16384,5120] parent.
bool n20480(std::int32_t input_rows) {
    if (input_rows == 3072) { return true; }
    if (input_rows == Fp8N16384K5120::kInputRows) { return false; }
    throw std::invalid_argument("fp8 gdn_input_proj: unsupported K");
}

void require_residual_profile(std::int32_t input_rows) {
    if (n20480(input_rows)) {
        throw std::invalid_argument(
            "fp8 gdn_input_proj: a static input scale has no A8 residual route");
    }
}

// The [20480,3072] parent has no adapter form.
void require_lora_profile(std::int32_t input_rows, const LoraAddend* lora) {
    if (lora != nullptr && n20480(input_rows)) {
        throw std::invalid_argument("fp8 gdn_input_proj: the [20480,3072] parent takes no LoRA term");
    }
}

} // namespace

std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t input_rows,
                                                 std::int32_t max_tokens) {
    return n20480(input_rows) ? fp8_gdn_input_n20480_partial_capacity_bytes(max_tokens)
                              : fp8_gdn_input_partial_capacity_bytes(max_tokens);
}

std::size_t fp8_gdn_input_workspace_capacity_bytes(QType qtype, LinearPolicy policy, std::int32_t input_rows,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 gdn_input_proj workspace: invalid token interval");
    }
    for (const std::int32_t tokens : {min_tokens, max_tokens}) {
        if (qtype == QType::FP8_E4M3FN_ROW_F32 && resolve_route(policy, tokens) == Fp8GdnInputRoute::A8Residual) {
            throw std::invalid_argument("fp8 gdn_input_proj: a static input scale has no A8 residual route");
        }
    }
    switch (resolve_route(policy, max_tokens)) {
    case Fp8GdnInputRoute::A8:
        return fp8_a8_workspace_capacity_bytes(
            max_tokens, input_rows, fp8_gdn_input_partial_capacity_bytes(input_rows, max_tokens));
    case Fp8GdnInputRoute::A8Residual:
        require_residual_profile(input_rows);
        return fp8_a8_workspace_capacity_bytes(max_tokens, input_rows, 0, true);
    case Fp8GdnInputRoute::ColumnInvariantA16:
    case Fp8GdnInputRoute::A16:
        break;
    }
    return 0;
}

void fp8_gdn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream, const LoraAddend* lora) {
    require_lora_profile(weight.k, lora);
    if (n20480(weight.k)) {
        fp8_gdn_input_n20480_invariant_launch(x, weight, qkv, z, stream);
    } else {
        fp8_gdn_input_n16384_invariant_launch(x, weight, qkv, z, stream, lora);
    }
}

void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream, const LoraAddend* lora) {
    require_lora_profile(weight.k, lora);
    if (n20480(weight.k)) {
        fp8_gdn_input_n20480_a16_launch(x, weight, qkv, z, stream);
    } else if (x.ne[1] == 1) {
        fp8_gdn_input_decode_launch(x, weight, qkv, z, stream, lora);
    } else {
        fp8_gdn_input_matrix_launch(x, weight, qkv, z, stream, lora);
    }
}

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream, const LoraAddend* lora) {
    require_lora_profile(weight.k, lora);
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_gdn_input_partial_capacity_bytes(weight.k, x.ne[1]));
    if (n20480(weight.k)) {
        fp8_gdn_input_n20480_a8_launch(x, weight, qkv, z, scratch, stream);
    } else {
        fp8_gdn_input_a8_launch(x, weight, qkv, z, scratch, stream, lora);
    }
}

void fp8_gdn_input_a8_residual_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                        Tensor& z, WorkspaceArena& workspace,
                                        cudaStream_t stream, const LoraAddend* lora) {
    require_residual_profile(weight.k);
    if (weight.qtype == QType::FP8_E4M3FN_ROW_F32) {
        throw std::invalid_argument("fp8 gdn_input_proj: a static input scale has no A8 residual route");
    }
    auto scope = workspace.scope();
    fp8_gdn_input_a8_residual_launch(
        x, weight, qkv, z, allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k, 0, true),
        stream, lora);
}

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream,
                            const LoraAddend* lora) {
    const Fp8GdnInputRoute route = resolve_route(policy, x.ne[1]);
    if (route == Fp8GdnInputRoute::ColumnInvariantA16) {
        fp8_gdn_input_invariant_launch(x, weight, qkv, z, stream, lora);
        return;
    }
    if (route == Fp8GdnInputRoute::A16) {
        fp8_gdn_input_a16_dispatch(x, weight, qkv, z, stream, lora);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 gdn_input_proj requires caller workspace");
    }
    if (route == Fp8GdnInputRoute::A8Residual) {
        fp8_gdn_input_a8_residual_dispatch(x, weight, qkv, z, *workspace, stream, lora);
        return;
    }
    fp8_gdn_input_a8_dispatch(x, weight, qkv, z, *workspace, stream, lora);
}

} // namespace infernix::ops::detail
