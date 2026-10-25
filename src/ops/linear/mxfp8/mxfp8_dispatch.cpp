#include "ops/linear/mxfp8/mxfp8_dispatch.h"

#include "ops/linear/mxfp8/mxfp8_format.h"
#include "ops/linear/mxfp8/mxfp8_shapes.h"

#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

const std::array kShapes{&kMxfp8N16384K2560, &kMxfp8N13952K2560, &kMxfp8N2560K6144, &kMxfp8N1280K2560,
                         &kMxfp8N2560K640};

// Without a route flag every admitted policy runs the exact A16 route, which gives a column the same
// bits at every T, so it serves A16ColumnInvariant as it is.
const Mxfp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) { throw std::invalid_argument("mxfp8 linear: unsupported policy"); }
    for (const auto* shape : kShapes) {
        if (shape->n == n && shape->k == k) { return *shape; }
    }
    throw std::invalid_argument("mxfp8 linear: unsupported shape");
}

// The FP16-accumulate route: flagged, registered, wide enough, and never under A16ColumnInvariant (whose
// route is the exact one at every width).
bool uses_f16(const Mxfp8LinearShape& shape, LinearPolicy policy, LinearRoutes routes, std::int32_t tokens) {
    return has_route(routes, LinearRoutes::F16Accumulate) && !column_invariant(policy) && shape.f16 != nullptr &&
           tokens >= shape.f16_min_tokens;
}

// The block32 pair route likewise (prefill widths, never column-invariant); f16-accumulate takes
// precedence, as on row-scaled FP8.
bool uses_pair_block32(const Mxfp8LinearShape& shape, LinearPolicy policy, LinearRoutes routes, std::int32_t tokens) {
    return has_route(routes, LinearRoutes::Fp8A8PairBlock32) && !column_invariant(policy) &&
           shape.pair_block32 != nullptr && tokens >= shape.pair_min_tokens;
}

} // namespace

std::size_t mxfp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                  std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("mxfp8 linear workspace: invalid token interval");
    }
    const Mxfp8LinearShape& shape = resolve_shape(n, k, policy);
    if (column_invariant(policy)) { return 0; }
    // The query does not see the route flags, so it covers both routes wherever the interval reaches them.
    const std::size_t f16 = shape.f16 != nullptr && max_tokens >= shape.f16_min_tokens
                                ? f16_accumulate_workspace_capacity_bytes(max_tokens, k)
                                : 0;
    const std::size_t pair = shape.pair_block32 != nullptr && max_tokens >= shape.pair_min_tokens
                                 ? fp8_a8_pair_block32_workspace_capacity_bytes(max_tokens, k)
                                 : 0;
    return f16 > pair ? f16 : pair;
}

void mxfp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy, LinearRoutes routes,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_mxfp8_weight(weight, "mxfp8 linear");
    if (x.ne[1] <= 0) { throw std::invalid_argument("mxfp8 linear: T must be positive"); }
    const Mxfp8LinearShape& shape = resolve_shape(weight.n, weight.k, policy);
    if (uses_f16(shape, policy, routes, x.ne[1])) {
        if (workspace == nullptr) {
            throw std::invalid_argument("mxfp8 f16-accumulate linear requires caller workspace");
        }
        auto scope         = workspace->scope();
        const auto scratch = allocate_f16_accumulate_workspace(*workspace, x.ne[1], weight.k);
        return shape.f16(x, weight, out, scratch, stream);
    }
    if (uses_pair_block32(shape, policy, routes, x.ne[1])) {
        if (workspace == nullptr) {
            throw std::invalid_argument("mxfp8 A8 pair block32 linear requires caller workspace");
        }
        auto scope         = workspace->scope();
        const auto scratch = allocate_fp8_a8_pair_block32_workspace(*workspace, x.ne[1], weight.k);
        return shape.pair_block32(x, weight, out, scratch, stream);
    }
    shape.a16(x, weight, out, stream);
}

std::size_t mxfp8_gdn_input_workspace_capacity_bytes(std::int32_t max_tokens) {
    return mxfp8_linear_workspace_capacity_bytes(16384, 2560, LinearPolicy::A16Only, 1, max_tokens);
}

} // namespace infernix::ops::detail
