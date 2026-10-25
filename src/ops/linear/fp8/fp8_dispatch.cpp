#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_format.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
const std::array kShapes{&kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120, &kFp8N5120K6144,
                         &kFp8N5120K17408,  &kFp8N248320K5120, &kFp8N16384K2560, &kFp8N13312K2560,
                         &kFp8N2560K6144,   &kFp8N1280K2560,   &kFp8N2560K640};

const Fp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("fp8 linear: unsupported shape");
}

// The single-term A8 route takes precedence where the policy permits it; the pair route only
// replaces an A16 route.
bool uses_pair(const Fp8LinearShape& shape, LinearPolicy policy, LinearRoutes routes,
               std::int32_t tokens) {
    return has_route(routes, LinearRoutes::Fp8A8Pair) && shape.pair != nullptr &&
           tokens >= shape.pair_min_tokens && !(allows_a8(policy) && shape.uses_a8(tokens, tokens));
}
} // namespace

bool fp8_linear_column_invariant_route(std::int32_t n, std::int32_t k) noexcept {
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return shape->invariant != nullptr;
    return false;
}

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    if (column_invariant(policy) && shape.invariant == nullptr)
        throw std::invalid_argument("fp8 linear: the shape registers no column-invariant route");
    if (column_invariant(policy)) return 0; // one A16 schedule, no route flag applies
    const std::size_t single = allows_a8(policy) && shape.uses_a8(min_tokens, max_tokens)
                                   ? fp8_a8_workspace_capacity_bytes(
                                         max_tokens, k,
                                         shape.partial_capacity_bytes
                                             ? shape.partial_capacity_bytes(max_tokens)
                                             : 0)
                                   : 0;
    // The query does not see the route flags, so it also covers the pair route wherever the
    // interval reaches it.
    const std::size_t pair = shape.pair != nullptr && max_tokens >= shape.pair_min_tokens
                                 ? fp8_a8_pair_workspace_capacity_bytes(max_tokens, k)
                                 : 0;
    return std::max(single, pair);
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  LinearRoutes routes, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    // FP32 output: the vocabulary head's A16 and column-invariant routes only.
    if (out.dtype == DType::FP32 && &shape != &kFp8N248320K5120) {
        throw std::invalid_argument("fp8 linear: FP32 output needs the [248320, 5120] head");
    }
    if (column_invariant(policy)) {
        if (shape.invariant == nullptr)
            throw std::invalid_argument("fp8 linear: the shape registers no column-invariant route");
        return shape.invariant(x, weight, out, stream);
    }
    if (uses_pair(shape, policy, routes, x.ne[1])) {
        if (workspace == nullptr)
            throw std::invalid_argument("fp8 A8 pair linear requires caller workspace");
        auto scope         = workspace->scope();
        const auto scratch = allocate_fp8_a8_pair_workspace(*workspace, x.ne[1], weight.k);
        return shape.pair(x, weight, out, scratch, stream);
    }
    if (!allows_a8(policy) || !shape.uses_a8(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k,
        shape.partial_capacity_bytes ? shape.partial_capacity_bytes(x.ne[1]) : 0);
    shape.a8(x, weight, out, scratch, stream);
}
} // namespace infernix::ops::detail
