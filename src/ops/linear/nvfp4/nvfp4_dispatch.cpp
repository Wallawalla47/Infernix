#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_a8_plan.h"
#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
const std::array kShapes{&kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
                         &kNvfp4N5120K6144, &kNvfp4N5120K17408, &kNvfp4N248320K5120,
                         &kNvfp4N5120K4096, &kNvfp4N5120K25600};

const Nvfp4LinearShape* find_shape(std::int32_t n, std::int32_t k) noexcept {
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return shape;
    return nullptr;
}

const Nvfp4LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("nvfp4 linear: unsupported policy");
    const auto* shape = find_shape(n, k);
    if (shape == nullptr) throw std::invalid_argument("nvfp4 linear: unsupported shape");
    if (column_invariant(policy) && shape->invariant == nullptr)
        throw std::invalid_argument("nvfp4 linear: no column-invariant route at this shape");
    return *shape;
}
} // namespace

bool nvfp4_linear_column_invariant_route(std::int32_t n, std::int32_t k) noexcept {
    const auto* shape = find_shape(n, k);
    return shape != nullptr && shape->invariant != nullptr;
}

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    if (a8_only(policy)) return nvfp4_a8_workspace_capacity_bytes(k, max_tokens);
    return allows_a4(policy) && shape.a4 != nullptr && shape.uses_a4(min_tokens, max_tokens)
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (out.dtype == DType::FP32 && !shape.fp32_output)
        throw std::invalid_argument("nvfp4 linear: FP32 output is the vocabulary head's only");
    if (column_invariant(policy)) return shape.invariant(x, weight, out, stream);
    if (a8_only(policy)) {
        if (workspace == nullptr) throw std::invalid_argument("nvfp4 A8 linear requires caller workspace");
        return nvfp4_linear_a8_launch(x, weight, out, *workspace, stream);
    }
    if (!allows_a4(policy) || shape.a4 == nullptr || !shape.uses_a4(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    shape.a4(x, weight, out, scratch, stream);
}
} // namespace infernix::ops::detail
