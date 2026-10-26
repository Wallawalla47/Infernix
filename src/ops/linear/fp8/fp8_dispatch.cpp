#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_format.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// fp8_e4m3fn_row_bf16 problems, and fp8_e4m3fn_row_f32 ones (their kernels read FP32 row words).
const std::array kShapes{&kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120, &kFp8N5120K6144,
                         &kFp8N5120K17408,  &kFp8N248320K5120, &kFp8N16384K2560, &kFp8N13312K2560,
                         &kFp8N2560K6144,   &kFp8N1280K2560,   &kFp8N2560K640};
const std::array kF32Shapes{&kFp8F32N3072K1024, &kFp8F32N248320K3072};

const Fp8LinearShape* find_shape(QType qtype, std::int32_t n, std::int32_t k) noexcept {
    if (qtype == QType::FP8_E4M3FN_ROW_F32) {
        for (const auto* shape : kF32Shapes)
            if (shape->n == n && shape->k == k) return shape;
        return nullptr;
    }
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return shape;
    return nullptr;
}

const Fp8LinearShape& resolve_shape(QType qtype, std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    if (const auto* shape = find_shape(qtype, n, k)) return *shape;
    throw std::invalid_argument("fp8 linear: unsupported shape");
}

// The vocabulary heads, the problems that write FP32 logits (A16 and column-invariant routes).
bool vocabulary_head(const Fp8LinearShape& shape) noexcept {
    return &shape == &kFp8N248320K5120 || &shape == &kFp8F32N248320K3072;
}

// The single-term A8 route takes precedence where the policy permits it; the pair route only
// replaces an A16 route.
bool uses_pair(const Fp8LinearShape& shape, LinearPolicy policy, LinearRoutes routes,
               std::int32_t tokens) {
    return has_route(routes, LinearRoutes::Fp8A8Pair) && shape.pair != nullptr &&
           tokens >= shape.pair_min_tokens && !(allows_a8(policy) && shape.uses_a8(tokens, tokens));
}

// fp8-a8-pair-block32 takes the column-scaled pair's admission and precedence over it.
bool uses_pair_block32(const Fp8LinearShape& shape, LinearPolicy policy, LinearRoutes routes,
                       std::int32_t tokens) {
    return has_route(routes, LinearRoutes::Fp8A8PairBlock32) && shape.pair_block32 != nullptr &&
           tokens >= shape.pair_min_tokens && !(allows_a8(policy) && shape.uses_a8(tokens, tokens));
}

// The FP16-accumulate route likewise replaces only an A16 route, and takes precedence over both pairs.
bool uses_f16(const Fp8LinearShape& shape, LinearPolicy policy, LinearRoutes routes, std::int32_t tokens) {
    return has_route(routes, LinearRoutes::F16Accumulate) && shape.f16 != nullptr &&
           tokens >= shape.f16_min_tokens && !(allows_a8(policy) && shape.uses_a8(tokens, tokens));
}
} // namespace

bool fp8_linear_column_invariant_route(QType qtype, std::int32_t n, std::int32_t k) noexcept {
    const auto* shape = find_shape(qtype, n, k);
    return shape != nullptr && shape->invariant != nullptr;
}

std::size_t fp8_linear_workspace_capacity_bytes(QType qtype, std::int32_t n, std::int32_t k,
                                                LinearPolicy policy, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(qtype, n, k, policy);
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
    // The query does not see the route flags, so it also covers both pair routes and the FP16 route
    // wherever the interval reaches them.
    const bool pair_reached = max_tokens >= shape.pair_min_tokens;
    const std::size_t pair  = shape.pair != nullptr && pair_reached
                                  ? fp8_a8_pair_workspace_capacity_bytes(max_tokens, k)
                                  : 0;
    const std::size_t pair_block32 =
        shape.pair_block32 != nullptr && pair_reached
            ? fp8_a8_pair_block32_workspace_capacity_bytes(max_tokens, k)
            : 0;
    const std::size_t f16 = shape.f16 != nullptr && max_tokens >= shape.f16_min_tokens
                                ? f16_accumulate_workspace_capacity_bytes(max_tokens, k)
                                : 0;
    return std::max({single, pair, pair_block32, f16});
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  LinearRoutes routes, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const auto& shape = resolve_shape(weight.qtype, weight.n, weight.k, policy);
    // FP32 output: the vocabulary heads' A16 and column-invariant routes only.
    if (out.dtype == DType::FP32 && !vocabulary_head(shape)) {
        throw std::invalid_argument("fp8 linear: FP32 output needs a [248320, 5120] or [248320, 3072] head");
    }
    if (column_invariant(policy)) {
        if (shape.invariant == nullptr)
            throw std::invalid_argument("fp8 linear: the shape registers no column-invariant route");
        return shape.invariant(x, weight, out, stream);
    }
    if (uses_f16(shape, policy, routes, x.ne[1])) {
        if (workspace == nullptr)
            throw std::invalid_argument("fp8 f16-accumulate linear requires caller workspace");
        auto scope         = workspace->scope();
        const auto scratch = allocate_f16_accumulate_workspace(*workspace, x.ne[1], weight.k);
        return shape.f16(x, weight, out, scratch, stream);
    }
    if (uses_pair_block32(shape, policy, routes, x.ne[1])) {
        if (workspace == nullptr)
            throw std::invalid_argument("fp8 A8 pair block32 linear requires caller workspace");
        auto scope         = workspace->scope();
        const auto scratch = allocate_fp8_a8_pair_block32_workspace(*workspace, x.ne[1], weight.k);
        return shape.pair_block32(x, weight, out, scratch, stream);
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
